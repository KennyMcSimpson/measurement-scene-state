"""Parameter-free composable measurements read from an explicit scene state."""

from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.nn.functional as functional
from torch import Tensor, nn

from mcss.geometry import generate_rays, intersect_aabb
from mcss.types import Cameras, SceneState

MEASUREMENTS = frozenset({"rgb", "depth", "normal", "point", "visibility", "uncertainty"})


class FixedMeasurementRenderer(nn.Module):
    """Read all primary scene measurements with fixed grid sampling and ray integration.

    The module owns no learned tensors. It is deliberately separate from the context encoder so
    a no-bypass audit can verify that predicted targets originate from typed scene fields.
    """

    def __init__(
        self,
        *,
        n_samples: int = 64,
        ray_chunk_size: int = 16_384,
        background_color: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> None:
        super().__init__()
        if n_samples < 2:
            raise ValueError("n_samples must be at least two")
        if ray_chunk_size < 1:
            raise ValueError("ray_chunk_size must be positive")
        if len(background_color) != 3:
            raise ValueError("background_color must contain RGB values")
        self.n_samples = n_samples
        self.ray_chunk_size = ray_chunk_size
        self.background_color = tuple(float(value) for value in background_color)

    def forward(
        self,
        state: SceneState,
        cameras: Cameras,
        measurements: Iterable[str] | None = None,
    ) -> dict[str, Tensor]:
        """Render requested measurements as tensors shaped ``[B, V, C, H, W]``."""

        requested = MEASUREMENTS if measurements is None else frozenset(measurements)
        unknown = requested - MEASUREMENTS
        if unknown:
            raise ValueError(f"unknown measurement names: {sorted(unknown)}")
        outputs, _ = self._render(state, cameras, include_features=False)
        return {name: outputs[name] for name in sorted(requested)}

    def render_features(self, state: SceneState, cameras: Cameras) -> Tensor:
        """Integrate optional state features for matched learned-decoder controls only."""

        if state.features is None:
            raise ValueError("render_features requires SceneState.features")
        _, features = self._render(state, cameras, include_features=True)
        if features is None:
            raise RuntimeError("feature integration unexpectedly produced no features")
        return features

    def _render(
        self, state: SceneState, cameras: Cameras, *, include_features: bool
    ) -> tuple[dict[str, Tensor], Tensor | None]:
        if len(cameras.leading_shape) != 2:
            raise ValueError("renderer expects cameras with shape [B, V, ...]")
        batch_size, _ = cameras.leading_shape
        if batch_size != state.density_logits.shape[0]:
            raise ValueError("camera and state batch sizes must match")
        if state.density_logits.device != cameras.device:
            raise ValueError("state and cameras must be on the same device")

        origins, directions = generate_rays(cameras)
        _, view_count, height, width, _ = origins.shape
        ray_count = view_count * height * width
        origins = origins.reshape(batch_size, ray_count, 3)
        directions = directions.reshape(batch_size, ray_count, 3)
        near, far, hit = intersect_aabb(origins, directions, state.bounds)
        normal_grid = _state_normal_grid(state)
        color_volume = state.color if state.appearance is None else state.appearance.color

        rendered: dict[str, list[Tensor]] = {name: [] for name in MEASUREMENTS}
        rendered_features: list[Tensor] = []
        for start in range(0, ray_count, self.ray_chunk_size):
            stop = min(start + self.ray_chunk_size, ray_count)
            chunk_outputs, chunk_features = self._render_chunk(
                state,
                normal_grid,
                color_volume,
                origins[:, start:stop],
                directions[:, start:stop],
                near[:, start:stop],
                far[:, start:stop],
                hit[:, start:stop],
                include_features=include_features,
            )
            for name, value in chunk_outputs.items():
                rendered[name].append(value)
            if chunk_features is not None:
                rendered_features.append(chunk_features)

        output = {
            name: _reshape_image(torch.cat(parts, dim=1), batch_size, view_count, height, width)
            for name, parts in rendered.items()
        }
        feature_output = None
        if include_features:
            feature_output = _reshape_image(
                torch.cat(rendered_features, dim=1), batch_size, view_count, height, width
            )
        return output, feature_output

    def _render_chunk(
        self,
        state: SceneState,
        normal_grid: Tensor,
        color_volume: Tensor,
        origins: Tensor,
        directions: Tensor,
        near: Tensor,
        far: Tensor,
        hit: Tensor,
        *,
        include_features: bool,
    ) -> tuple[dict[str, Tensor], Tensor | None]:
        batch_size, ray_count, _ = origins.shape
        fractions = (
            torch.arange(self.n_samples, device=origins.device, dtype=origins.dtype) + 0.5
        ) / self.n_samples
        distances = near.unsqueeze(-1) + (far - near).unsqueeze(-1) * fractions
        points = origins.unsqueeze(-2) + directions.unsqueeze(-2) * distances.unsqueeze(-1)
        sample_valid = hit.unsqueeze(-1)

        density_logits = _sample_volume(state.density_logits, points, state.bounds).squeeze(-1)
        colors = _sample_volume(color_volume, points, state.bounds)
        log_variance = _sample_volume(state.log_variance, points, state.bounds).squeeze(-1)
        with torch.autocast(device_type=points.device.type, enabled=False):
            normals = _sample_volume(normal_grid, points.float(), state.bounds.float())
        density = torch.nn.functional.softplus(density_logits) * sample_valid
        deltas = ((far - near) / self.n_samples).unsqueeze(-1)
        alpha = 1.0 - torch.exp(-density * deltas)
        transmittance = torch.cumprod(
            torch.cat((torch.ones_like(alpha[..., :1]), 1.0 - alpha + 1e-10), dim=-1), dim=-1
        )[..., :-1]
        weights = transmittance * alpha
        visibility = weights.sum(dim=-1, keepdim=True)
        depth = (weights * distances).sum(dim=-1, keepdim=True)
        rgb = (weights.unsqueeze(-1) * colors).sum(dim=-2)
        background = torch.as_tensor(
            self.background_color, device=rgb.device, dtype=rgb.dtype
        ).view(1, 1, 3)
        rgb = rgb + (1.0 - visibility) * background
        with torch.autocast(device_type=weights.device.type, enabled=False):
            normal = _normalize((weights.float().unsqueeze(-1) * normals).sum(dim=-2))
        point = origins + directions * depth
        uncertainty = (weights * log_variance.exp()).sum(dim=-1, keepdim=True)

        outputs = {
            "depth": depth,
            "normal": normal,
            "point": point,
            "rgb": rgb,
            "uncertainty": uncertainty,
            "visibility": visibility,
        }
        features = None
        if include_features:
            if state.features is None:
                raise ValueError("state features are required for learned controls")
            sampled_features = _sample_volume(state.features, points, state.bounds)
            features = (weights.unsqueeze(-1) * sampled_features).sum(dim=-2)
        return outputs, features


def _sample_volume(volume: Tensor, points: Tensor, bounds: Tensor) -> Tensor:
    """Trilinearly sample ``[B, C, D, H, W]`` at world points ``[B, R, S, 3]``."""

    minimum = bounds[:, 0].view(-1, 1, 1, 3)
    extent = (bounds[:, 1] - bounds[:, 0]).view(-1, 1, 1, 3)
    grid = (points - minimum) / extent * 2.0 - 1.0
    batch_size, rays, samples, _ = grid.shape
    grid = grid.reshape(batch_size, rays * samples, 1, 1, 3)
    sampled = functional.grid_sample(
        volume, grid, mode="bilinear", padding_mode="zeros", align_corners=False
    )
    return sampled.reshape(batch_size, volume.shape[1], rays, samples).permute(0, 2, 3, 1)


def _state_normal_grid(state: SceneState) -> Tensor:
    """Finite-difference density gradients converted to outward world normals."""

    with torch.autocast(device_type=state.density_logits.device.type, enabled=False):
        density = torch.nn.functional.softplus(state.density_logits.float())
        depth, height, width = state.spatial_shape
        extent = state.bounds[:, 1].float() - state.bounds[:, 0].float()
        dx = functional.pad(density[..., 2:] - density[..., :-2], (1, 1, 0, 0, 0, 0))
        dy = functional.pad(density[:, :, :, 2:, :] - density[:, :, :, :-2, :], (0, 0, 1, 1, 0, 0))
        dz = functional.pad(density[:, :, 2:, :, :] - density[:, :, :-2, :, :], (0, 0, 0, 0, 1, 1))
        x_spacing = (extent[:, 0] / width).view(-1, 1, 1, 1, 1)
        y_spacing = (extent[:, 1] / height).view(-1, 1, 1, 1, 1)
        z_spacing = (extent[:, 2] / depth).view(-1, 1, 1, 1, 1)
        gradient = torch.cat(
            (dx / (2.0 * x_spacing), dy / (2.0 * y_spacing), dz / (2.0 * z_spacing)),
            dim=1,
        )
        return -_normalize(gradient, channel_dim=1)


def _reshape_image(values: Tensor, batch_size: int, views: int, height: int, width: int) -> Tensor:
    return values.reshape(batch_size, views, height, width, values.shape[-1]).permute(0, 1, 4, 2, 3)


def _normalize(values: Tensor, *, channel_dim: int = -1) -> Tensor:
    norm = torch.linalg.vector_norm(values, dim=channel_dim, keepdim=True)
    normalized = values / norm.clamp_min(1e-6)
    return torch.where(norm > 1e-6, normalized, torch.zeros_like(normalized))
