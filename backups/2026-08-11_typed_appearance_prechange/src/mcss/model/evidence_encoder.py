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
from mcss.types import Cameras, SceneState, StateEvidence


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

        self.voxel_resolution = resolution
        self.evidence_temperature = float(evidence_temperature)
        self.observed_residual_floor = float(observed_residual_floor)
        self.completion_residual_scale = float(completion_residual_scale)
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

        centers = make_voxel_centers(bounds, self.voxel_resolution)
        point_count = math.prod(self.voxel_resolution)
        flat_centers = centers.reshape(batch_size, point_count, 3)
        points_per_view = flat_centers[:, None].expand(batch_size, view_count, point_count, 3)
        with torch.autocast(device_type=context_rgb.device.type, enabled=False):
            pixels, camera_depth, valid = project_world(
                points_per_view.float(), cameras.to(dtype=torch.float32)
            )
            normalized_grid = _pixels_to_grid(pixels, image_height, image_width)

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
        base_color_points = (sampled_rgb * provenance).sum(dim=1)
        has_provenance = provenance.sum(dim=1) > 0.0
        base_color_points = torch.where(
            has_provenance.expand_as(base_color_points),
            base_color_points,
            torch.full_like(base_color_points, 0.5),
        )
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
        return SceneState(density_logits, color, log_variance, bounds, features, evidence)


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
