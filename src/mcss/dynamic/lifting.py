"""Differentiable camera lifting for the dynamic carrier's fixed voxel candidates."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as functional
from torch import Tensor

from mcss.dynamic.cache import ObservationCache
from mcss.geometry import project_world


@dataclass(frozen=True)
class LiftedStatistics:
    """View-aggregated candidate features with geometry-only support information."""

    observed_feature_statistics: Tensor
    support_counts: Tensor
    support_weights: Tensor


def lift_cached_observations(
    cache: ObservationCache,
    candidate_points: Tensor,
    normalized_xyz: Tensor,
    *,
    feature_dim: int,
) -> LiftedStatistics:
    """Sample fixed world candidates from an observed cache without selecting by content.

    A candidate is observed only when its projection is finite, in front of the camera, and
    inside that observation's input image.  The support count says nothing about a surface.
    """

    if not cache.entries:
        raise ValueError("carrier requires at least one cached observation")
    if candidate_points.ndim != 2 or candidate_points.shape[-1] != 3:
        raise ValueError("candidate_points must have shape [N, 3]")
    if normalized_xyz.shape != candidate_points.shape:
        raise ValueError("normalized_xyz must align with candidate_points")
    if feature_dim < 1:
        raise ValueError("feature_dim must be positive")

    sampled_features: list[Tensor] = []
    valid_views: list[Tensor] = []
    view_directions: list[Tensor] = []
    for entry in cache.entries:
        features = entry.features
        observation = entry.observation
        if features.shape[0] != feature_dim:
            raise ValueError("cached feature channels do not match carrier configuration")
        if features.device != candidate_points.device:
            raise ValueError("cached observations and candidate points must share a device")
        if observation.camera.device != candidate_points.device:
            raise ValueError("cached camera and candidate points must share a device")

        sampled, valid = _sample_valid_features(
            features,
            candidate_points,
            observation.camera,
        )
        origin = observation.camera.c2w[:3, 3].to(
            device=candidate_points.device,
            dtype=candidate_points.dtype,
        )
        direction = _normalize(candidate_points - origin)
        sampled_features.append(sampled)
        valid_views.append(valid)
        view_directions.append(direction)

    features_by_view = torch.stack(sampled_features, dim=0)
    valid_by_view = torch.stack(valid_views, dim=0)
    directions_by_view = torch.stack(view_directions, dim=0)
    mask = valid_by_view.unsqueeze(-1).to(features_by_view.dtype)
    count = mask.sum(dim=0)
    denominator = count.clamp_min(1.0)
    mean_features = (features_by_view * mask).sum(dim=0) / denominator
    variance_features = ((features_by_view - mean_features.unsqueeze(0)).square() * mask).sum(
        dim=0
    ) / denominator
    mean_view_direction = (directions_by_view * mask).sum(dim=0) / denominator
    support_counts = valid_by_view.sum(dim=0)
    support_weights = (support_counts >= 2).to(features_by_view.dtype)
    statistics = torch.cat(
        (
            mean_features,
            variance_features,
            support_counts.to(features_by_view.dtype).unsqueeze(-1),
            normalized_xyz.to(dtype=features_by_view.dtype),
            mean_view_direction,
        ),
        dim=-1,
    )
    return LiftedStatistics(statistics, support_counts, support_weights)


def _sample_valid_features(features: Tensor, points: Tensor, camera) -> tuple[Tensor, Tensor]:
    """Project and bilinearly sample one image feature map at fixed candidate locations."""

    if features.ndim != 3:
        raise ValueError("cached features must have shape [F, H, W]")
    if camera.leading_shape:
        raise ValueError("dynamic lifting requires one unbatched camera")
    if features.device != camera.device:
        raise ValueError("cached features and camera must share a device")

    input_height, input_width = camera.image_size
    sample_dtype = torch.float32
    points32 = points.to(dtype=sample_dtype)
    camera32 = camera.to(dtype=sample_dtype)
    pixels, depth, inside = project_world(points32, camera32)
    finite = torch.isfinite(pixels).all(dim=-1) & torch.isfinite(depth)
    valid = inside & finite
    grid = _pixels_to_grid(pixels, input_height, input_width).unsqueeze(0).unsqueeze(2)
    safe_grid = torch.nan_to_num(grid, nan=2.0, posinf=2.0, neginf=-2.0)
    sampled = functional.grid_sample(
        features.to(dtype=sample_dtype).unsqueeze(0),
        safe_grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )
    sampled = sampled.squeeze(0).squeeze(-1).transpose(0, 1)
    sampled = torch.where(valid.unsqueeze(-1), sampled, torch.zeros_like(sampled))
    return sampled.to(dtype=features.dtype), valid


def _pixels_to_grid(pixels: Tensor, height: int, width: int) -> Tensor:
    x = pixels[..., 0] / max(width - 1, 1) * 2.0 - 1.0
    y = pixels[..., 1] / max(height - 1, 1) * 2.0 - 1.0
    return torch.stack((x, y), dim=-1)


def _normalize(vectors: Tensor) -> Tensor:
    return vectors / torch.linalg.vector_norm(vectors, dim=-1, keepdim=True).clamp_min(1e-8)
