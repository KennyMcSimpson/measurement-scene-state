"""Cross-view evidence construction and bounded residual completion in a typed 3D state."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn.functional as functional
from torch import Tensor, nn

from mcss.geometry import make_voxel_centers, project_world
from mcss.model.encoder2d import PyramidImageEncoder
from mcss.model.state_encoder import Residual3DBlock
from mcss.types import (
    Cameras,
    DualEvidence,
    SceneState,
    StateAppearance,
    StateEvidence,
    TransportEvidence,
)


class EvidenceResidualStateEncoder(nn.Module):
    """Build an inspectable evidence state and add only a bounded, gated completion residual."""

    def __init__(
        self,
        *,
        voxel_resolution: int | Sequence[int],
        image_feature_dim: int,
        state_feature_dim: int,
        refinement_blocks: int,
        evidence_temperature: float,
        observed_residual_floor: float,
        completion_residual_scale: float,
        appearance_resolution_scale: int = 1,
    ) -> None:
        super().__init__()
        if isinstance(voxel_resolution, int):
            resolution = (voxel_resolution,) * 3
        else:
            resolution = tuple(voxel_resolution)
        if len(resolution) != 3 or min(resolution) < 4:
            raise ValueError("voxel_resolution must contain D, H, W values of at least four")
        if image_feature_dim < 4 or state_feature_dim < 4:
            raise ValueError("feature dimensions must be at least four")
        if refinement_blocks < 0:
            raise ValueError("refinement_blocks cannot be negative")
        if evidence_temperature <= 0:
            raise ValueError("evidence_temperature must be positive")
        if not 0.0 <= observed_residual_floor <= 1.0:
            raise ValueError("observed_residual_floor must be within [0, 1]")
        if completion_residual_scale <= 0:
            raise ValueError("completion_residual_scale must be positive")
        if appearance_resolution_scale not in {1, 2}:
            raise ValueError("appearance_resolution_scale must be one or two")

        self.voxel_resolution = resolution
        self.evidence_temperature = float(evidence_temperature)
        self.observed_residual_floor = float(observed_residual_floor)
        self.completion_residual_scale = float(completion_residual_scale)
        self.appearance_resolution_scale = appearance_resolution_scale
        self.image_encoder = PyramidImageEncoder(image_feature_dim)

        observation_channels = image_feature_dim + 3 + 3 + 1
        fused_channels = observation_channels * 2 + 3
        hidden_channels = max(32, state_feature_dim * 2)
        self.input_projection = nn.Sequential(
            nn.Conv3d(fused_channels, hidden_channels, 3, padding=1, bias=False),
            nn.GroupNorm(_group_count(hidden_channels), hidden_channels),
            nn.SiLU(inplace=True),
        )
        self.refinement = nn.Sequential(
            *(Residual3DBlock(hidden_channels) for _ in range(refinement_blocks))
        )
        self.density_residual_head = nn.Conv3d(hidden_channels, 1, 1)
        self.color_residual_head = nn.Conv3d(hidden_channels, 3, 1)
        self.variance_head = nn.Conv3d(hidden_channels, 1, 1)
        self.feature_head = nn.Conv3d(hidden_channels, state_feature_dim, 1)
        nn.init.zeros_(self.density_residual_head.weight)
        nn.init.zeros_(self.density_residual_head.bias)
        nn.init.zeros_(self.color_residual_head.weight)
        nn.init.zeros_(self.color_residual_head.bias)
        nn.init.constant_(self.variance_head.bias, -3.0)
        self.appearance_input_projection: nn.Module | None = None
        self.appearance_refinement: nn.Module | None = None
        self.appearance_residual_head: nn.Conv3d | None = None
        if appearance_resolution_scale == 2:
            appearance_hidden_channels = max(16, state_feature_dim)
            self.appearance_input_projection = nn.Sequential(
                nn.Conv3d(
                    hidden_channels + 5,
                    appearance_hidden_channels,
                    3,
                    padding=1,
                    bias=False,
                ),
                nn.GroupNorm(
                    _group_count(appearance_hidden_channels), appearance_hidden_channels
                ),
                nn.SiLU(inplace=True),
            )
            self.appearance_refinement = nn.Sequential(
                *(
                    Residual3DBlock(appearance_hidden_channels)
                    for _ in range(max(1, refinement_blocks // 2))
                )
            )
            self.appearance_residual_head = nn.Conv3d(appearance_hidden_channels, 3, 1)
            nn.init.zeros_(self.appearance_residual_head.weight)
            nn.init.zeros_(self.appearance_residual_head.bias)

    def forward(self, context_rgb: Tensor, cameras: Cameras, bounds: Tensor) -> SceneState:
        if context_rgb.ndim != 5 or context_rgb.shape[2] != 3:
            raise ValueError("context_rgb must have shape [B, V, 3, H, W]")
        batch_size, view_count, _, image_height, image_width = context_rgb.shape
        if cameras.leading_shape != (batch_size, view_count):
            raise ValueError("context cameras must align with context_rgb")
        if cameras.image_size != (image_height, image_width):
            raise ValueError("context camera image size must match context_rgb")
        if bounds.shape != (batch_size, 2, 3):
            raise ValueError("bounds must have shape [B, 2, 3]")

        encoded = self.image_encoder(
            context_rgb.reshape(batch_size * view_count, 3, image_height, image_width)
        )
        _, channels, feature_height, feature_width = encoded.shape
        encoded = encoded.reshape(batch_size, view_count, channels, feature_height, feature_width)

        points_per_view, normalized_grid, camera_depth, valid = _project_voxel_grid(
            bounds,
            cameras,
            self.voxel_resolution,
            image_height,
            image_width,
        )

        sampled_features = _sample_images(encoded, normalized_grid)
        sampled_rgb = _sample_images(context_rgb, normalized_grid)
        evidence_features = _fixed_rgb_pyramid(context_rgb, normalized_grid)
        confidence, provenance = _cross_view_evidence(
            evidence_features,
            sampled_rgb,
            valid,
            temperature=self.evidence_temperature,
        )
        unknown_probability = 1.0 - confidence

        camera_centers = cameras.c2w[..., :3, 3].unsqueeze(-2)
        view_direction = _normalize(points_per_view - camera_centers)
        scene_diagonal = torch.linalg.vector_norm(bounds[:, 1] - bounds[:, 0], dim=-1)
        normalized_depth = camera_depth / scene_diagonal[:, None, None].clamp_min(1e-6)
        observations = torch.cat(
            (sampled_features, sampled_rgb, view_direction, normalized_depth.unsqueeze(-1)), dim=-1
        )
        mean = (observations * provenance).sum(dim=1)
        variance = ((observations - mean[:, None]).square() * provenance).sum(dim=1)
        coverage = valid.to(observations.dtype).mean(dim=1, keepdim=False).unsqueeze(-1)
        fused = torch.cat((mean, variance, coverage, confidence, unknown_probability), dim=-1)
        fused_grid = _points_to_volume(fused, self.voxel_resolution)

        hidden = self.refinement(self.input_projection(fused_grid))
        density_residual = (
            torch.tanh(self.density_residual_head(hidden)) * self.completion_residual_scale
        )
        color_logit_residual = (
            torch.tanh(self.color_residual_head(hidden)) * self.completion_residual_scale
        )

        confidence_grid = _points_to_volume(confidence, self.voxel_resolution)
        unknown_grid = 1.0 - confidence_grid
        provenance_grid = provenance.squeeze(-1).reshape(
            batch_size, view_count, *self.voxel_resolution
        )
        base_color_points = _base_color_from_provenance(sampled_rgb, provenance)
        base_color = _points_to_volume(base_color_points, self.voxel_resolution).clamp(0.0, 1.0)
        base_density_logits = 4.0 * confidence_grid - 2.0
        completion_gate = self.observed_residual_floor + (
            1.0 - self.observed_residual_floor
        ) * unknown_grid
        density_logits = base_density_logits + completion_gate * density_residual
        base_color_logits = torch.logit(base_color.clamp(1e-4, 1.0 - 1e-4))
        color = torch.sigmoid(base_color_logits + completion_gate * color_logit_residual)
        log_variance = (self.variance_head(hidden) + 2.0 * unknown_grid).clamp(-8.0, 4.0)
        features = self.feature_head(hidden)
        evidence = StateEvidence(
            confidence=confidence_grid,
            unknown_probability=unknown_grid,
            completion_gate=completion_gate,
            provenance=provenance_grid,
            base_density_logits=base_density_logits,
            base_color=base_color,
            density_residual=density_residual,
            color_logit_residual=color_logit_residual,
        )
        appearance = None
        if self.appearance_resolution_scale == 2:
            appearance = self._build_highres_appearance(
                context_rgb,
                cameras,
                bounds,
                hidden,
            )
        return SceneState(
            density_logits,
            color,
            log_variance,
            bounds,
            features,
            evidence,
            appearance,
        )

    def _build_highres_appearance(
        self,
        context_rgb: Tensor,
        cameras: Cameras,
        bounds: Tensor,
        hidden: Tensor,
    ) -> StateAppearance:
        if (
            self.appearance_input_projection is None
            or self.appearance_refinement is None
            or self.appearance_residual_head is None
        ):
            raise RuntimeError("high-resolution appearance modules are not initialized")
        image_height, image_width = context_rgb.shape[-2:]
        resolution = tuple(
            axis * self.appearance_resolution_scale for axis in self.voxel_resolution
        )
        _, normalized_grid, _, valid = _project_voxel_grid(
            bounds,
            cameras,
            resolution,
            image_height,
            image_width,
        )
        sampled_rgb = _sample_images(context_rgb, normalized_grid)
        fixed_features = _fixed_rgb_pyramid(context_rgb, normalized_grid)
        confidence, provenance = _cross_view_evidence(
            fixed_features,
            sampled_rgb,
            valid,
            temperature=self.evidence_temperature,
        )
        confidence_grid = _points_to_volume(confidence, resolution)
        unknown_grid = 1.0 - confidence_grid
        completion_gate = self.observed_residual_floor + (
            1.0 - self.observed_residual_floor
        ) * unknown_grid
        provenance_grid = provenance.squeeze(-1).reshape(
            context_rgb.shape[0], context_rgb.shape[1], *resolution
        )
        base_color_points = _base_color_from_provenance(sampled_rgb, provenance)
        base_color = _points_to_volume(base_color_points, resolution).clamp(0.0, 1.0)

        lifted_hidden = functional.interpolate(
            hidden,
            size=resolution,
            mode="trilinear",
            align_corners=False,
        )
        appearance_inputs = torch.cat(
            (lifted_hidden, base_color, confidence_grid, unknown_grid), dim=1
        )
        appearance_hidden = self.appearance_refinement(
            self.appearance_input_projection(appearance_inputs)
        )
        residual = (
            torch.tanh(self.appearance_residual_head(appearance_hidden))
            * self.completion_residual_scale
        )
        return StateAppearance(
            confidence=confidence_grid,
            unknown_probability=unknown_grid,
            completion_gate=completion_gate,
            provenance=provenance_grid,
            base_color=base_color,
            color_logit_residual=residual,
        )


class DualEvidenceStateEncoder(EvidenceResidualStateEncoder):
    """Add a detached local ray-competition gate without changing V5 trainable parameters."""

    def __init__(
        self,
        *,
        voxel_resolution: int | Sequence[int],
        image_feature_dim: int,
        state_feature_dim: int,
        refinement_blocks: int,
        evidence_temperature: float,
        surface_peak_temperature: float,
        observed_residual_floor: float,
        completion_residual_scale: float,
        appearance_resolution_scale: int = 1,
        surface_peak_chunk_size: int = 4096,
    ) -> None:
        super().__init__(
            voxel_resolution=voxel_resolution,
            image_feature_dim=image_feature_dim,
            state_feature_dim=state_feature_dim,
            refinement_blocks=refinement_blocks,
            evidence_temperature=evidence_temperature,
            observed_residual_floor=observed_residual_floor,
            completion_residual_scale=completion_residual_scale,
            appearance_resolution_scale=appearance_resolution_scale,
        )
        if surface_peak_temperature <= 0:
            raise ValueError("surface_peak_temperature must be positive")
        if surface_peak_chunk_size < 1:
            raise ValueError("surface_peak_chunk_size must be positive")
        self.surface_peak_temperature = float(surface_peak_temperature)
        self.surface_peak_chunk_size = int(surface_peak_chunk_size)

    def forward(self, context_rgb: Tensor, cameras: Cameras, bounds: Tensor) -> SceneState:
        v5_state = super().forward(context_rgb, cameras, bounds)
        if v5_state.evidence is None:
            raise RuntimeError("dual evidence requires the V5 evidence construction")
        appearance_evidence = v5_state.evidence
        peakness = _surface_peakness(
            context_rgb,
            cameras,
            bounds,
            self.voxel_resolution,
            evidence_temperature=self.evidence_temperature,
            surface_peak_temperature=self.surface_peak_temperature,
            chunk_size=self.surface_peak_chunk_size,
        ).to(dtype=appearance_evidence.confidence.dtype)
        appearance_confidence = appearance_evidence.confidence
        localization_support = (appearance_confidence * peakness).clamp(0.0, 1.0)
        localization_unknown = 1.0 - localization_support
        appearance_unknown = 1.0 - appearance_confidence
        localization_gate = self.observed_residual_floor + (
            1.0 - self.observed_residual_floor
        ) * localization_unknown
        appearance_gate = appearance_evidence.completion_gate
        base_density_logits = 4.0 * localization_support - 2.0
        density_logits = (
            base_density_logits + localization_gate * appearance_evidence.density_residual
        )
        dual_evidence = DualEvidence(
            surface_localization_support=localization_support,
            appearance_confidence=appearance_confidence,
            surface_peakness=peakness,
            localization_unknown_probability=localization_unknown,
            appearance_unknown_probability=appearance_unknown,
            localization_completion_gate=localization_gate,
            appearance_completion_gate=appearance_gate,
            provenance=appearance_evidence.provenance,
            base_density_logits=base_density_logits,
            base_color=appearance_evidence.base_color,
            density_residual=appearance_evidence.density_residual,
            color_logit_residual=appearance_evidence.color_logit_residual,
        )
        return SceneState(
            density_logits=density_logits,
            color=v5_state.color,
            log_variance=v5_state.log_variance,
            bounds=v5_state.bounds,
            features=v5_state.features,
            evidence=None,
            appearance=v5_state.appearance,
            dual_evidence=dual_evidence,
        )


class TransportEvidenceStateEncoder(EvidenceResidualStateEncoder):
    """Transport fixed local ray evidence to its metric 3D candidate positions."""

    def __init__(
        self,
        *,
        voxel_resolution: int | Sequence[int],
        image_feature_dim: int,
        state_feature_dim: int,
        refinement_blocks: int,
        evidence_temperature: float,
        surface_peak_temperature: float,
        observed_residual_floor: float,
        completion_residual_scale: float,
        appearance_resolution_scale: int = 1,
        surface_peak_chunk_size: int = 4096,
    ) -> None:
        super().__init__(
            voxel_resolution=voxel_resolution,
            image_feature_dim=image_feature_dim,
            state_feature_dim=state_feature_dim,
            refinement_blocks=refinement_blocks,
            evidence_temperature=evidence_temperature,
            observed_residual_floor=observed_residual_floor,
            completion_residual_scale=completion_residual_scale,
            appearance_resolution_scale=appearance_resolution_scale,
        )
        if surface_peak_temperature <= 0:
            raise ValueError("surface_peak_temperature must be positive")
        if surface_peak_chunk_size < 1:
            raise ValueError("surface_peak_chunk_size must be positive")
        self.surface_peak_temperature = float(surface_peak_temperature)
        self.surface_peak_chunk_size = int(surface_peak_chunk_size)

    def forward(self, context_rgb: Tensor, cameras: Cameras, bounds: Tensor) -> SceneState:
        v5_state = super().forward(context_rgb, cameras, bounds)
        if v5_state.evidence is None:
            raise RuntimeError("transport evidence requires the V5 evidence construction")
        appearance_evidence = v5_state.evidence
        coverage, transported_appearance, peakness, localization_support = (
            _transport_surface_evidence(
                context_rgb,
                cameras,
                bounds,
                self.voxel_resolution,
                appearance_evidence.confidence,
                evidence_temperature=self.evidence_temperature,
                surface_peak_temperature=self.surface_peak_temperature,
                chunk_size=self.surface_peak_chunk_size,
            )
        )
        evidence_dtype = appearance_evidence.confidence.dtype
        coverage = coverage.to(dtype=evidence_dtype)
        transported_appearance = transported_appearance.to(dtype=evidence_dtype)
        peakness = peakness.to(dtype=evidence_dtype)
        localization_support = localization_support.to(dtype=evidence_dtype)
        appearance_confidence = appearance_evidence.confidence
        localization_unknown = 1.0 - localization_support
        appearance_unknown = 1.0 - appearance_confidence
        localization_gate = self.observed_residual_floor + (
            1.0 - self.observed_residual_floor
        ) * localization_unknown
        appearance_gate = appearance_evidence.completion_gate
        base_density_logits = 4.0 * localization_support - 2.0
        density_logits = (
            base_density_logits + localization_gate * appearance_evidence.density_residual
        )
        transport_evidence = TransportEvidence(
            surface_localization_support=localization_support,
            transported_appearance_confidence=transported_appearance,
            surface_peakness=peakness,
            transport_coverage=coverage,
            appearance_confidence=appearance_confidence,
            localization_unknown_probability=localization_unknown,
            appearance_unknown_probability=appearance_unknown,
            localization_completion_gate=localization_gate,
            appearance_completion_gate=appearance_gate,
            provenance=appearance_evidence.provenance,
            base_density_logits=base_density_logits,
            base_color=appearance_evidence.base_color,
            density_residual=appearance_evidence.density_residual,
            color_logit_residual=appearance_evidence.color_logit_residual,
        )
        return SceneState(
            density_logits=density_logits,
            color=v5_state.color,
            log_variance=v5_state.log_variance,
            bounds=v5_state.bounds,
            features=v5_state.features,
            evidence=None,
            appearance=v5_state.appearance,
            dual_evidence=None,
            transport_evidence=transport_evidence,
        )


@torch.no_grad()
def _trilinear_splat_candidates(
    points: Tensor,
    values: Tensor,
    valid: Tensor,
    bounds: Tensor,
    resolution: tuple[int, int, int],
) -> Tensor:
    """Splat candidate values onto native voxel centers with boundary mass conservation."""

    if points.ndim != 4 or points.shape[-1] != 3:
        raise ValueError("points must have shape [B, C, K, 3]")
    candidate_shape = points.shape[:-1]
    scalar_values = values.ndim == 3 and values.shape == candidate_shape
    channel_values = (
        values.ndim == 4
        and values.shape[:-1] == candidate_shape
        and values.shape[-1] >= 1
    )
    if not scalar_values and not channel_values:
        raise ValueError(
            "values must have shape [B, C, K] or [B, C, K, M] aligned with points"
        )
    if valid.shape != candidate_shape:
        raise ValueError("valid must align with candidate points")
    if valid.dtype != torch.bool:
        raise ValueError("valid must be boolean")
    if bounds.shape != (points.shape[0], 2, 3):
        raise ValueError("bounds must have shape [B, 2, 3]")
    if len(resolution) != 3 or min(resolution) < 1:
        raise ValueError("resolution must contain positive D, H, W values")
    if not torch.isfinite(points).all() or not torch.isfinite(values).all():
        raise ValueError("splat inputs must be finite")

    points = points.detach().float()
    values = values.detach().float()
    bounds = bounds.detach().float()
    depth, height, width = resolution
    axis_sizes = torch.tensor(
        [width, height, depth],
        device=points.device,
        dtype=points.dtype,
    )
    cell_size = (bounds[:, 1] - bounds[:, 0]) / axis_sizes
    coordinates = (
        (points - bounds[:, None, None, 0]) / cell_size[:, None, None] - 0.5
    )
    base = torch.floor(coordinates).to(torch.int64)
    fraction = coordinates - base.to(coordinates.dtype)
    neighbor_offsets = torch.tensor(
        [
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
            [0, 1, 1],
            [1, 0, 0],
            [1, 0, 1],
            [1, 1, 0],
            [1, 1, 1],
        ],
        device=points.device,
        dtype=torch.int64,
    )
    neighbor = base[..., None, :] + neighbor_offsets
    within_grid = (
        (neighbor[..., 0] >= 0)
        & (neighbor[..., 0] < width)
        & (neighbor[..., 1] >= 0)
        & (neighbor[..., 1] < height)
        & (neighbor[..., 2] >= 0)
        & (neighbor[..., 2] < depth)
    )
    axis_weights = torch.where(
        neighbor_offsets.to(torch.bool)[None, None, None],
        fraction[..., None, :],
        1.0 - fraction[..., None, :],
    )
    weights = axis_weights.prod(dim=-1)
    weights *= (within_grid & valid[..., None]).to(weights.dtype)
    weight_sum = weights.sum(dim=-1, keepdim=True)
    weights = torch.where(weight_sum > 0, weights / weight_sum.clamp_min(1e-12), weights)

    x = neighbor[..., 0].clamp(0, width - 1)
    y = neighbor[..., 1].clamp(0, height - 1)
    z = neighbor[..., 2].clamp(0, depth - 1)
    linear_index = z * (height * width) + y * width + x
    values_by_channel = values.unsqueeze(-1) if scalar_values else values
    channel_count = values_by_channel.shape[-1]
    output = torch.zeros(
        points.shape[0],
        channel_count,
        depth * height * width,
        device=points.device,
        dtype=torch.float32,
    )
    indices = linear_index.reshape(points.shape[0], 1, -1).expand(
        -1, channel_count, -1
    )
    contributions = (weights[..., None] * values_by_channel[..., None, :]).permute(
        0, 4, 1, 2, 3
    )
    output.scatter_add_(
        2,
        indices,
        contributions.reshape(points.shape[0], channel_count, -1),
    )
    return output[:, 0] if scalar_values else output


@torch.no_grad()
def _transport_surface_evidence(
    context_rgb: Tensor,
    cameras: Cameras,
    bounds: Tensor,
    resolution: tuple[int, int, int],
    appearance_confidence: Tensor,
    *,
    evidence_temperature: float,
    surface_peak_temperature: float,
    chunk_size: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Return coverage, transported appearance, peakness, and localization support."""

    if context_rgb.ndim != 5 or context_rgb.shape[2] != 3:
        raise ValueError("context_rgb must have shape [B, V, 3, H, W]")
    batch_size, view_count, _, image_height, image_width = context_rgb.shape
    if cameras.leading_shape != (batch_size, view_count):
        raise ValueError("context cameras must align with context_rgb")
    if cameras.image_size != (image_height, image_width):
        raise ValueError("context camera image size must match context_rgb")
    if bounds.shape != (batch_size, 2, 3):
        raise ValueError("bounds must have shape [B, 2, 3]")
    if appearance_confidence.shape != (batch_size, 1, *resolution):
        raise ValueError("appearance_confidence must align with the native grid")
    if evidence_temperature <= 0 or surface_peak_temperature <= 0:
        raise ValueError("evidence temperatures must be positive")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")

    zero = torch.zeros(
        batch_size,
        1,
        *resolution,
        device=context_rgb.device,
        dtype=torch.float32,
    )
    if view_count < 3:
        return zero, zero.clone(), zero.clone(), zero.clone()

    with torch.autocast(device_type=context_rgb.device.type, enabled=False):
        rgb = context_rgb.detach().float()
        bounds_fp32 = bounds.detach().float()
        cameras_fp32 = cameras.to(dtype=torch.float32)
        appearance_points = appearance_confidence.detach().float().reshape(batch_size, -1)
        centers = make_voxel_centers(bounds_fp32, resolution).reshape(batch_size, -1, 3)
        point_count = centers.shape[1]
        center_points_per_view = centers[:, None].expand(
            batch_size, view_count, point_count, 3
        )
        center_pixels, _, _ = project_world(center_points_per_view, cameras_fp32)
        center_grid = _pixels_to_grid(center_pixels, image_height, image_width)
        reference_features = _fixed_rgb_pyramid(rgb, center_grid)
        reference_rgb = _sample_images(rgb, center_grid)

        depth, height, width = resolution
        axis_sizes = torch.tensor(
            [width, height, depth],
            device=bounds_fp32.device,
            dtype=bounds_fp32.dtype,
        )
        delta = ((bounds_fp32[:, 1] - bounds_fp32[:, 0]) / axis_sizes).amin(dim=-1)
        offsets = torch.tensor(
            [-2.0, -1.0, 0.0, 1.0, 2.0],
            device=bounds_fp32.device,
            dtype=bounds_fp32.dtype,
        )
        camera_centers = cameras_fp32.c2w[..., :3, 3]
        accumulators = torch.zeros(
            batch_size,
            3,
            point_count,
            device=bounds_fp32.device,
            dtype=torch.float32,
        )
        other_view_mask = ~torch.eye(
            view_count, device=bounds_fp32.device, dtype=torch.bool
        )

        for reference_index in range(view_count):
            for start in range(0, point_count, chunk_size):
                stop = min(start + chunk_size, point_count)
                center_chunk = centers[:, start:stop]
                direction = _normalize(
                    center_chunk - camera_centers[:, reference_index, None]
                )
                candidates = center_chunk[:, :, None] + (
                    direction[:, :, None]
                    * delta[:, None, None, None]
                    * offsets[None, None, :, None]
                )
                inside = (
                    (candidates >= bounds_fp32[:, None, None, 0])
                    & (candidates <= bounds_fp32[:, None, None, 1])
                ).all(dim=-1)
                chunk_points = candidates.reshape(batch_size, -1, 3)
                points_per_view = chunk_points[:, None].expand(
                    batch_size, view_count, chunk_points.shape[1], 3
                )
                pixels, _, projected_valid = project_world(points_per_view, cameras_fp32)
                normalized_grid = _pixels_to_grid(pixels, image_height, image_width)
                sampled_features = _fixed_rgb_pyramid(rgb, normalized_grid).reshape(
                    batch_size, view_count, stop - start, 5, -1
                )
                sampled_rgb = _sample_images(rgb, normalized_grid).reshape(
                    batch_size, view_count, stop - start, 5, 3
                )
                projected_valid = projected_valid.reshape(
                    batch_size, view_count, stop - start, 5
                )
                observer_valid = projected_valid & inside[:, None]
                observer_valid &= other_view_mask[reference_index][None, :, None, None]
                observer_count = observer_valid.sum(dim=1)
                reference_valid = projected_valid[:, reference_index] & inside
                candidate_valid = reference_valid & (observer_count >= 2)

                reference_feature = reference_features[
                    :, reference_index, start:stop
                ][:, None, :, None]
                reference_color = reference_rgb[:, reference_index, start:stop][
                    :, None, :, None
                ]
                feature_agreement = (
                    functional.cosine_similarity(
                        reference_feature,
                        sampled_features,
                        dim=-1,
                        eps=1e-6,
                    )
                    + 1.0
                ) * 0.5
                rgb_error = (sampled_rgb - reference_color).square().mean(dim=-1)
                rgb_agreement = torch.exp(-rgb_error / evidence_temperature)
                pair_score = feature_agreement * rgb_agreement
                score = (
                    pair_score * observer_valid.to(pair_score.dtype)
                ).sum(dim=1) / observer_count.to(pair_score.dtype).clamp_min(1.0)
                excess, contributes = _positive_candidate_excess(
                    score,
                    candidate_valid,
                    temperature=surface_peak_temperature,
                )
                transport_valid = candidate_valid & contributes[..., None]
                source_appearance = appearance_points[:, start:stop, None].expand_as(excess)
                values = torch.stack(
                    (
                        torch.ones_like(excess),
                        source_appearance,
                        source_appearance * excess,
                    ),
                    dim=-1,
                )
                accumulators += _trilinear_splat_candidates(
                    candidates,
                    values,
                    transport_valid,
                    bounds_fp32,
                    resolution,
                )

        coverage = accumulators[:, 0]
        appearance_mass = accumulators[:, 1]
        localized_mass = accumulators[:, 2]
        transported_appearance = torch.where(
            coverage > 0,
            appearance_mass / coverage.clamp_min(1e-12),
            torch.zeros_like(coverage),
        ).clamp(0.0, 1.0)
        peakness = torch.where(
            appearance_mass > 0,
            localized_mass / appearance_mass.clamp_min(1e-12),
            torch.zeros_like(appearance_mass),
        ).clamp(0.0, 1.0)
        localization_support = torch.where(
            coverage > 0,
            localized_mass / coverage.clamp_min(1e-12),
            torch.zeros_like(coverage),
        ).clamp(0.0, 1.0)

        def volume(values: Tensor) -> Tensor:
            return values.reshape(batch_size, 1, depth, height, width).detach()

        return (
            volume(coverage),
            volume(transported_appearance),
            volume(peakness),
            volume(localization_support),
        )


@torch.no_grad()
def _surface_peakness(
    context_rgb: Tensor,
    cameras: Cameras,
    bounds: Tensor,
    resolution: tuple[int, int, int],
    *,
    evidence_temperature: float,
    surface_peak_temperature: float,
    chunk_size: int,
    _telemetry_records: list[tuple[Tensor, Tensor, Tensor, Tensor | None]] | None = None,
    _telemetry_appearance_confidence: Tensor | None = None,
) -> Tensor:
    """Return native-grid local ray-competition support in detached FP32."""

    if context_rgb.ndim != 5 or context_rgb.shape[2] != 3:
        raise ValueError("context_rgb must have shape [B, V, 3, H, W]")
    batch_size, view_count, _, image_height, image_width = context_rgb.shape
    if cameras.leading_shape != (batch_size, view_count):
        raise ValueError("context cameras must align with context_rgb")
    if bounds.shape != (batch_size, 2, 3):
        raise ValueError("bounds must have shape [B, 2, 3]")
    if _telemetry_appearance_confidence is not None and (
        _telemetry_appearance_confidence.shape != (batch_size, 1, *resolution)
    ):
        raise ValueError("telemetry appearance confidence must align with the native grid")
    if evidence_temperature <= 0 or surface_peak_temperature <= 0:
        raise ValueError("evidence temperatures must be positive")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    if view_count < 2:
        return torch.zeros(
            batch_size,
            1,
            *resolution,
            device=context_rgb.device,
            dtype=torch.float32,
        )

    with torch.autocast(device_type=context_rgb.device.type, enabled=False):
        rgb = context_rgb.detach().float()
        bounds_fp32 = bounds.detach().float()
        cameras_fp32 = cameras.to(dtype=torch.float32)
        appearance_points = (
            None
            if _telemetry_appearance_confidence is None
            else _telemetry_appearance_confidence.detach().float().reshape(batch_size, -1)
        )
        centers = make_voxel_centers(bounds_fp32, resolution).reshape(batch_size, -1, 3)
        point_count = centers.shape[1]
        center_points_per_view = centers[:, None].expand(
            batch_size, view_count, point_count, 3
        )
        center_pixels, _, center_valid = project_world(
            center_points_per_view, cameras_fp32
        )
        center_grid = _pixels_to_grid(center_pixels, image_height, image_width)
        reference_features = _fixed_rgb_pyramid(rgb, center_grid)
        reference_rgb = _sample_images(rgb, center_grid)

        depth, height, width = resolution
        axis_sizes = torch.tensor(
            [width, height, depth],
            device=bounds_fp32.device,
            dtype=bounds_fp32.dtype,
        )
        delta = ((bounds_fp32[:, 1] - bounds_fp32[:, 0]) / axis_sizes).amin(dim=-1)
        offsets = torch.tensor(
            [-2.0, -1.0, 0.0, 1.0, 2.0],
            device=bounds_fp32.device,
            dtype=bounds_fp32.dtype,
        )
        camera_centers = cameras_fp32.c2w[..., :3, 3]
        peakness_sum = torch.zeros(
            batch_size, point_count, device=bounds_fp32.device, dtype=torch.float32
        )
        reference_count = torch.zeros_like(peakness_sum)
        other_view_mask = ~torch.eye(
            view_count, device=bounds_fp32.device, dtype=torch.bool
        )

        for reference_index in range(view_count):
            for start in range(0, point_count, chunk_size):
                stop = min(start + chunk_size, point_count)
                center_chunk = centers[:, start:stop]
                direction = _normalize(
                    center_chunk - camera_centers[:, reference_index, None]
                )
                candidates = center_chunk[:, :, None] + (
                    direction[:, :, None]
                    * delta[:, None, None, None]
                    * offsets[None, None, :, None]
                )
                inside = (
                    (candidates >= bounds_fp32[:, None, None, 0])
                    & (candidates <= bounds_fp32[:, None, None, 1])
                ).all(dim=-1)
                chunk_points = candidates.reshape(batch_size, -1, 3)
                points_per_view = chunk_points[:, None].expand(
                    batch_size, view_count, chunk_points.shape[1], 3
                )
                pixels, _, projected_valid = project_world(
                    points_per_view, cameras_fp32
                )
                normalized_grid = _pixels_to_grid(pixels, image_height, image_width)
                sampled_features = _fixed_rgb_pyramid(rgb, normalized_grid).reshape(
                    batch_size, view_count, stop - start, 5, -1
                )
                sampled_rgb = _sample_images(rgb, normalized_grid).reshape(
                    batch_size, view_count, stop - start, 5, 3
                )
                projected_valid = projected_valid.reshape(
                    batch_size, view_count, stop - start, 5
                )
                observer_valid = projected_valid & inside[:, None]
                observer_valid &= other_view_mask[reference_index][None, :, None, None]
                reference_valid = projected_valid[:, reference_index] & inside
                candidate_valid = reference_valid & observer_valid.any(dim=1)

                reference_feature = reference_features[
                    :, reference_index, start:stop
                ][:, None, :, None]
                reference_color = reference_rgb[:, reference_index, start:stop][
                    :, None, :, None
                ]
                feature_agreement = (
                    functional.cosine_similarity(
                        reference_feature,
                        sampled_features,
                        dim=-1,
                        eps=1e-6,
                    )
                    + 1.0
                ) * 0.5
                rgb_error = (sampled_rgb - reference_color).square().mean(dim=-1)
                rgb_agreement = torch.exp(-rgb_error / evidence_temperature)
                pair_score = feature_agreement * rgb_agreement
                score = (
                    pair_score * observer_valid.to(pair_score.dtype)
                ).sum(dim=1) / (view_count - 1)
                if _telemetry_records is not None:
                    _telemetry_records.append(
                        (
                            score.detach().reshape(-1, score.shape[-1]).cpu(),
                            candidate_valid.detach()
                            .reshape(-1, candidate_valid.shape[-1])
                            .cpu(),
                            observer_valid.detach()
                            .sum(dim=1)
                            .reshape(-1, candidate_valid.shape[-1])
                            .cpu(),
                            (
                                None
                                if appearance_points is None
                                else appearance_points[:, start:stop].reshape(-1).cpu()
                            ),
                        )
                    )
                peakness, contributes = _normalized_center_peakness(
                    score,
                    candidate_valid,
                    temperature=surface_peak_temperature,
                )
                peakness_sum[:, start:stop] += peakness
                reference_count[:, start:stop] += contributes.to(reference_count.dtype)

        peakness = peakness_sum / reference_count.clamp_min(1.0)
        peakness = torch.where(reference_count > 0, peakness, torch.zeros_like(peakness))
        return _points_to_volume(peakness.unsqueeze(-1), resolution).detach()


def _surface_peakness_with_telemetry(
    context_rgb: Tensor,
    cameras: Cameras,
    bounds: Tensor,
    resolution: tuple[int, int, int],
    *,
    evidence_temperature: float,
    surface_peak_temperature: float,
    chunk_size: int,
    appearance_confidence: Tensor | None = None,
) -> tuple[Tensor, dict[str, object]]:
    """Run the production cue while collecting context-only candidate statistics."""

    records: list[tuple[Tensor, Tensor, Tensor, Tensor | None]] = []
    peakness = _surface_peakness(
        context_rgb,
        cameras,
        bounds,
        resolution,
        evidence_temperature=evidence_temperature,
        surface_peak_temperature=surface_peak_temperature,
        chunk_size=chunk_size,
        _telemetry_records=records,
        _telemetry_appearance_confidence=appearance_confidence,
    )
    if records:
        scores = torch.cat([record[0] for record in records], dim=0)
        valid = torch.cat([record[1] for record in records], dim=0)
        observer_count = torch.cat([record[2] for record in records], dim=0)
        profile_appearance = (
            None
            if records[0][3] is None
            else torch.cat([record[3] for record in records if record[3] is not None], dim=0)
        )
    else:
        scores = torch.empty(0, 5)
        valid = torch.empty(0, 5, dtype=torch.bool)
        observer_count = torch.empty(0, 5, dtype=torch.int64)
        profile_appearance = None
    report = _candidate_profile_telemetry(
        scores,
        valid,
        observer_count,
        offsets=torch.tensor([-2, -1, 0, 1, 2]),
        temperature=surface_peak_temperature,
        appearance_confidence=profile_appearance,
        observer_capacity=max(context_rgb.shape[1] - 1, 1),
    )
    report["surface_peakness"] = _tensor_summary(peakness)
    return peakness, report


def _normalized_center_peakness(
    scores: Tensor,
    valid: Tensor,
    *,
    temperature: float,
    center_index: int = 2,
) -> tuple[Tensor, Tensor]:
    """Normalize center probability so a flat valid candidate profile maps to zero."""

    if scores.ndim < 1 or scores.shape != valid.shape:
        raise ValueError("scores and valid must have the same non-empty shape")
    if valid.dtype != torch.bool:
        raise ValueError("valid must be boolean")
    if not 0 <= center_index < scores.shape[-1]:
        raise ValueError("center_index must identify a candidate")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if not torch.isfinite(scores).all():
        raise ValueError("scores must be finite")

    candidate_count = valid.sum(dim=-1)
    contributes = valid[..., center_index] & (candidate_count >= 3)
    safe_valid = valid.clone()
    safe_valid[..., center_index] |= ~valid.any(dim=-1)
    logits = (scores.float() / temperature).masked_fill(~safe_valid, -torch.inf)
    center_probability = torch.softmax(logits, dim=-1)[..., center_index]
    count = candidate_count.to(center_probability.dtype)
    peakness = ((count * center_probability - 1.0) / (count - 1.0).clamp_min(1.0)).clamp(
        0.0, 1.0
    )
    return torch.where(contributes, peakness, torch.zeros_like(peakness)), contributes


def _positive_candidate_excess(
    scores: Tensor,
    valid: Tensor,
    *,
    temperature: float,
    min_candidates: int = 3,
) -> tuple[Tensor, Tensor]:
    """Return valid-only probability mass above the flat-profile baseline."""

    if scores.ndim < 1 or scores.shape != valid.shape:
        raise ValueError("scores and valid must have the same non-empty shape")
    if valid.dtype != torch.bool:
        raise ValueError("valid must be boolean")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if not 2 <= min_candidates <= scores.shape[-1]:
        raise ValueError("min_candidates must be between two and the candidate count")
    if not torch.isfinite(scores).all():
        raise ValueError("scores must be finite")

    candidate_count = valid.sum(dim=-1)
    contributes = candidate_count >= min_candidates
    safe_valid = valid.clone()
    safe_valid[..., 0] |= ~valid.any(dim=-1)
    logits = (scores.float() / temperature).masked_fill(~safe_valid, -torch.inf)
    probabilities = torch.softmax(logits, dim=-1)
    count = candidate_count.to(probabilities.dtype)
    excess = (
        (count[..., None] * probabilities - 1.0)
        / (count[..., None] - 1.0).clamp_min(1.0)
    ).clamp(0.0, 1.0)
    excess *= valid.to(excess.dtype)
    return torch.where(contributes[..., None], excess, torch.zeros_like(excess)), contributes


def _candidate_profile_telemetry(
    scores: Tensor,
    valid: Tensor,
    observer_count: Tensor,
    *,
    offsets: Tensor,
    temperature: float,
    center_index: int = 2,
    appearance_confidence: Tensor | None = None,
    observer_capacity: int | None = None,
) -> dict[str, object]:
    """Summarize fixed depth-candidate competition without target labels."""

    if scores.ndim != 2 or scores.shape != valid.shape:
        raise ValueError("scores and valid must have shape [N, K]")
    if observer_count.shape != scores.shape:
        raise ValueError("observer_count must align with scores")
    if valid.dtype != torch.bool:
        raise ValueError("valid must be boolean")
    if offsets.ndim != 1 or offsets.shape[0] != scores.shape[1]:
        raise ValueError("offsets must identify every candidate")
    if not 0 <= center_index < scores.shape[1]:
        raise ValueError("center_index must identify a candidate")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if not torch.isfinite(scores).all():
        raise ValueError("scores must be finite")
    if (observer_count < 0).any():
        raise ValueError("observer_count cannot be negative")
    if observer_capacity is not None and observer_capacity < 1:
        raise ValueError("observer_capacity must be positive")
    if observer_capacity is not None and (observer_count > observer_capacity).any():
        raise ValueError("observer_count cannot exceed observer_capacity")
    if appearance_confidence is not None and appearance_confidence.shape != scores.shape[:1]:
        raise ValueError("appearance_confidence must have shape [N]")

    scores = scores.detach().float().cpu()
    valid = valid.detach().cpu()
    observer_count = observer_count.detach().to(dtype=torch.int64, device="cpu")
    offsets = offsets.detach().float().cpu()
    if appearance_confidence is not None:
        appearance_confidence = appearance_confidence.detach().float().cpu()
        if not torch.isfinite(appearance_confidence).all():
            raise ValueError("appearance_confidence must be finite")
    valid_count = valid.sum(dim=-1)
    contributes = valid[:, center_index] & (valid_count >= 3)
    profile_total = int(scores.shape[0])
    contributing_total = int(contributes.sum().item())
    report: dict[str, object] = {
        "profiles": {
            "total": profile_total,
            "contributing": contributing_total,
            "contributing_fraction": (
                float(contributing_total / profile_total) if profile_total else 0.0
            ),
        },
        "valid_candidate_count_histogram": _integer_histogram(valid_count),
        "observer_count_histogram": _integer_histogram(observer_count[valid]),
        "competition": {},
        "strata": {},
    }
    if contributing_total == 0:
        return report

    report["competition"] = _competition_telemetry(
        scores[contributes],
        valid[contributes],
        offsets=offsets,
        temperature=temperature,
        center_index=center_index,
    )
    if appearance_confidence is not None:
        positive = contributes & (appearance_confidence > 1e-6)
        top_threshold = float(torch.quantile(appearance_confidence, 0.9).item())
        top_decile = contributes & (appearance_confidence >= top_threshold)
        report["strata"] = {
            "appearance_positive": _competition_stratum(
                scores,
                valid,
                positive,
                offsets=offsets,
                temperature=temperature,
                center_index=center_index,
                threshold=1e-6,
            ),
            "appearance_top_decile": _competition_stratum(
                scores,
                valid,
                top_decile,
                offsets=offsets,
                temperature=temperature,
                center_index=center_index,
                threshold=top_threshold,
            ),
        }
    if observer_capacity is not None:
        observer_mean_scores = (
            scores
            * float(observer_capacity)
            / observer_count.to(dtype=scores.dtype).clamp_min(1.0)
        )
        report["counterfactuals"] = {
            "valid_observer_mean": _candidate_profile_telemetry(
                observer_mean_scores,
                valid,
                observer_count,
                offsets=offsets,
                temperature=temperature,
                center_index=center_index,
                appearance_confidence=appearance_confidence,
            ),
            "min_two_observers": _candidate_profile_telemetry(
                observer_mean_scores,
                valid & (observer_count >= 2),
                observer_count,
                offsets=offsets,
                temperature=temperature,
                center_index=center_index,
                appearance_confidence=appearance_confidence,
            ),
        }
    return report


def _competition_stratum(
    scores: Tensor,
    valid: Tensor,
    selected: Tensor,
    *,
    offsets: Tensor,
    temperature: float,
    center_index: int,
    threshold: float,
) -> dict[str, object]:
    profile_count = int(selected.sum().item())
    return {
        "profiles": profile_count,
        "appearance_threshold": float(threshold),
        "competition": (
            _competition_telemetry(
                scores[selected],
                valid[selected],
                offsets=offsets,
                temperature=temperature,
                center_index=center_index,
            )
            if profile_count
            else {}
        ),
    }


def _competition_telemetry(
    profile_scores: Tensor,
    profile_valid: Tensor,
    *,
    offsets: Tensor,
    temperature: float,
    center_index: int,
) -> dict[str, object]:
    profile_count = profile_valid.sum(dim=-1).float()
    masked_scores = profile_scores.masked_fill(~profile_valid, -torch.inf)
    logits = (profile_scores / temperature).masked_fill(~profile_valid, -torch.inf)
    probabilities = torch.softmax(logits, dim=-1)
    safe_probabilities = torch.where(profile_valid, probabilities, torch.zeros_like(probabilities))

    maximum = masked_scores.max(dim=-1).values
    maximum_mask = profile_valid & torch.isclose(
        profile_scores,
        maximum[:, None],
        rtol=1e-6,
        atol=1e-7,
    )
    maximum_count = maximum_mask.sum(dim=-1)
    unique_winner = maximum_count == 1
    center_is_maximum = maximum_mask[:, center_index]
    unique_center = unique_winner & center_is_maximum
    center_fractional = maximum_mask[:, center_index].float() / maximum_count.float()
    chance_rate = 1.0 / profile_count

    other_valid = profile_valid.clone()
    other_valid[:, center_index] = False
    other_maximum = profile_scores.masked_fill(~other_valid, -torch.inf).max(dim=-1).values
    center_margin = profile_scores[:, center_index] - other_maximum
    minimum = profile_scores.masked_fill(~profile_valid, torch.inf).min(dim=-1).values
    score_range = maximum - minimum
    score_mean = (
        torch.where(profile_valid, profile_scores, torch.zeros_like(profile_scores)).sum(dim=-1)
        / profile_count
    )
    score_variance = (
        torch.where(
            profile_valid,
            (profile_scores - score_mean[:, None]).square(),
            torch.zeros_like(profile_scores),
        ).sum(dim=-1)
        / profile_count
    )
    entropy = -(
        safe_probabilities * safe_probabilities.clamp_min(torch.finfo(torch.float32).tiny).log()
    ).sum(dim=-1) / profile_count.log()
    sorted_probabilities = safe_probabilities.sort(dim=-1, descending=True).values
    positive_excess = (
        (profile_count[:, None] * probabilities - 1.0)
        / (profile_count[:, None] - 1.0).clamp_min(1.0)
    ).clamp(0.0, 1.0) * profile_valid.to(probabilities.dtype)
    excess_per_profile = positive_excess.sum(dim=-1)
    noncenter_excess = positive_excess.clone()
    noncenter_excess[:, center_index] = 0.0
    total_excess = float(excess_per_profile.sum().item())

    winner_offsets = offsets[masked_scores.argmax(dim=-1)[unique_winner]]
    winner_histogram: dict[str, int] = {}
    for offset in offsets:
        count = int(torch.isclose(winner_offsets, offset).sum().item())
        if count:
            winner_histogram[_number_key(float(offset.item()))] = count

    return {
        "center_argmax_inclusive_rate": float(center_is_maximum.float().mean().item()),
        "center_unique_argmax_rate": float(unique_center.float().mean().item()),
        "center_argmax_fractional_rate": float(center_fractional.mean().item()),
        "chance_argmax_rate": float(chance_rate.mean().item()),
        "center_argmax_excess_over_chance": float(
            (center_fractional - chance_rate).mean().item()
        ),
        "unique_winner_fraction": float(unique_winner.float().mean().item()),
        "winner_offset_histogram": winner_histogram,
        "noncenter_positive_excess_fraction": (
            float(noncenter_excess.sum().item()) / total_excess if total_excess > 0 else 0.0
        ),
        "zero_positive_excess_profile_count": int((excess_per_profile == 0).sum().item()),
        "positive_excess_total": total_excess,
        "center_margin": _tensor_summary(center_margin),
        "score_range": _tensor_summary(score_range),
        "score_std": _tensor_summary(score_variance.sqrt()),
        "softmax_entropy_normalized": _tensor_summary(entropy),
        "p_center": _tensor_summary(probabilities[:, center_index]),
        "p_max": _tensor_summary(sorted_probabilities[:, 0]),
        "p_second": _tensor_summary(sorted_probabilities[:, 1]),
    }


def _integer_histogram(values: Tensor) -> dict[str, int]:
    values = values.detach().to(dtype=torch.int64, device="cpu").reshape(-1)
    if values.numel() == 0:
        return {}
    unique, counts = torch.unique(values, sorted=True, return_counts=True)
    return {
        str(int(value.item())): int(count.item())
        for value, count in zip(unique, counts, strict=True)
    }


def _number_key(value: float) -> str:
    return str(int(value)) if value.is_integer() else f"{value:g}"


def _tensor_summary(values: Tensor) -> dict[str, float]:
    values = values.detach().float().cpu().reshape(-1)
    if values.numel() == 0:
        return {}
    quantiles = torch.quantile(values, torch.tensor([0.1, 0.5, 0.9]))
    return {
        "count": float(values.numel()),
        "mean": float(values.mean().item()),
        "std": float(values.std(unbiased=False).item()),
        "min": float(values.min().item()),
        "p10": float(quantiles[0].item()),
        "p50": float(quantiles[1].item()),
        "p90": float(quantiles[2].item()),
        "max": float(values.max().item()),
    }


def _cross_view_evidence(
    features: Tensor,
    rgb: Tensor,
    valid: Tensor,
    *,
    temperature: float,
) -> tuple[Tensor, Tensor]:
    """Return correspondence confidence `[B,N,1]` and view provenance `[B,V,N,1]`."""

    if features.ndim != 4 or rgb.ndim != 4 or valid.ndim != 3:
        raise ValueError("features, rgb, and valid must have [B,V,N,...] shapes")
    if features.shape[:3] != rgb.shape[:3] or features.shape[:3] != valid.shape:
        raise ValueError("features, rgb, and valid must align")
    if rgb.shape[-1] != 3:
        raise ValueError("rgb must have three channels")
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    # Evidence audits the context; it must not become a trainable escape route for completion.
    features = features.detach()
    rgb = rgb.detach()

    mask = valid.unsqueeze(-1).to(features.dtype)
    count = mask.sum(dim=1).clamp_min(1.0)
    normalized = functional.normalize(features, dim=-1, eps=1e-6) * mask
    feature_sum = normalized.sum(dim=1)
    pair_dot_sum = feature_sum.square().sum(dim=-1, keepdim=True) - mask.sum(dim=1)
    raw_count = mask.sum(dim=1)
    pair_count = (raw_count * (raw_count - 1.0)).clamp_min(1.0)
    pair_correlation = (pair_dot_sum / pair_count).clamp(-1.0, 1.0)
    feature_agreement = (pair_correlation + 1.0) * 0.5

    rgb_mean = (rgb * mask).sum(dim=1) / count
    rgb_variance = ((rgb - rgb_mean[:, None]).square() * mask).sum(dim=1) / count
    rgb_agreement = torch.exp(-rgb_variance.mean(dim=-1, keepdim=True) / temperature)
    view_count = features.shape[1]
    if view_count == 1:
        pair_support = torch.zeros_like(raw_count)
    else:
        pair_support = ((raw_count - 1.0) / (view_count - 1.0)).clamp(0.0, 1.0)
    confidence = (pair_support * feature_agreement * rgb_agreement).clamp(0.0, 1.0)

    consensus = functional.normalize(feature_sum, dim=-1, eps=1e-6)
    feature_score = (normalized * consensus[:, None]).sum(dim=-1, keepdim=True)
    rgb_score = -(rgb - rgb_mean[:, None]).square().mean(dim=-1, keepdim=True) / temperature
    score = feature_score / temperature + rgb_score
    score = score.masked_fill(~valid.unsqueeze(-1), -1e4)
    provenance = torch.softmax(score, dim=1) * valid.unsqueeze(-1).to(score.dtype)
    has_valid = valid.any(dim=1, keepdim=True).unsqueeze(-1)
    provenance = provenance * has_valid.to(provenance.dtype)
    provenance = provenance / provenance.sum(dim=1, keepdim=True).clamp_min(1e-8)
    return confidence, provenance


def _project_voxel_grid(
    bounds: Tensor,
    cameras: Cameras,
    resolution: tuple[int, int, int],
    image_height: int,
    image_width: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    centers = make_voxel_centers(bounds, resolution)
    point_count = math.prod(resolution)
    flat_centers = centers.reshape(bounds.shape[0], point_count, 3)
    points_per_view = flat_centers[:, None].expand(
        bounds.shape[0], cameras.leading_shape[1], point_count, 3
    )
    with torch.autocast(device_type=bounds.device.type, enabled=False):
        pixels, camera_depth, valid = project_world(
            points_per_view.float(), cameras.to(dtype=torch.float32)
        )
        normalized_grid = _pixels_to_grid(pixels, image_height, image_width)
    return points_per_view, normalized_grid, camera_depth, valid


def _base_color_from_provenance(sampled_rgb: Tensor, provenance: Tensor) -> Tensor:
    base_color = (sampled_rgb * provenance).sum(dim=1)
    has_provenance = provenance.sum(dim=1) > 0.0
    return torch.where(
        has_provenance.expand_as(base_color),
        base_color,
        torch.full_like(base_color, 0.5),
    )


def _sample_images(images: Tensor, normalized_grid: Tensor) -> Tensor:
    batch_size, view_count, channels, _, _ = images.shape
    point_count = normalized_grid.shape[-2]
    flattened_images = images.reshape(batch_size * view_count, channels, *images.shape[-2:])
    grid = normalized_grid.reshape(batch_size * view_count, point_count, 1, 2)
    sampled = functional.grid_sample(
        flattened_images, grid, mode="bilinear", padding_mode="zeros", align_corners=True
    )
    return sampled.reshape(batch_size, view_count, channels, point_count).permute(0, 1, 3, 2)


def _fixed_rgb_pyramid(images: Tensor, normalized_grid: Tensor) -> Tensor:
    batch_size, view_count, channels, height, width = images.shape
    flattened = images.reshape(batch_size * view_count, channels, height, width)
    levels = [flattened]
    for scale in (2, 4):
        if min(height, width) >= scale:
            levels.append(functional.avg_pool2d(flattened, kernel_size=scale, stride=scale))
    sampled = [
        _sample_images(
            level.reshape(batch_size, view_count, channels, *level.shape[-2:]),
            normalized_grid,
        )
        for level in levels
    ]
    return torch.cat(sampled, dim=-1)


def _pixels_to_grid(pixels: Tensor, height: int, width: int) -> Tensor:
    x = pixels[..., 0] / max(width - 1, 1) * 2.0 - 1.0
    y = pixels[..., 1] / max(height - 1, 1) * 2.0 - 1.0
    return torch.stack((x, y), dim=-1)


def _points_to_volume(values: Tensor, resolution: tuple[int, int, int]) -> Tensor:
    return values.reshape(values.shape[0], *resolution, values.shape[-1]).permute(0, 4, 1, 2, 3)


def _normalize(values: Tensor) -> Tensor:
    return values / torch.linalg.vector_norm(values, dim=-1, keepdim=True).clamp_min(1e-8)


def _group_count(channels: int) -> int:
    for groups in (8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1
