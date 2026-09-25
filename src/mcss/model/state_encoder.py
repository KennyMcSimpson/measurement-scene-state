"""Camera-aware unprojection into an explicit typed voxel state."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn.functional as functional
from torch import Tensor, nn

from mcss.geometry import make_voxel_centers, project_world
from mcss.model.encoder2d import ImageEncoder
from mcss.types import Cameras, SceneState


class Residual3DBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        groups = _group_count(channels)
        self.block = nn.Sequential(
            nn.Conv3d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, channels),
            nn.SiLU(inplace=True),
            nn.Conv3d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, channels),
        )
        self.activation = nn.SiLU(inplace=True)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.activation(inputs + self.block(inputs))


class UnprojectiveStateEncoder(nn.Module):
    """Fuse calibrated image observations at fixed world-space voxel centers."""

    def __init__(
        self,
        *,
        voxel_resolution: int | Sequence[int],
        image_feature_dim: int,
        state_feature_dim: int,
        refinement_blocks: int,
    ) -> None:
        super().__init__()
        if isinstance(voxel_resolution, int):
            resolution = (voxel_resolution,) * 3
        else:
            resolution = tuple(voxel_resolution)
        if len(resolution) != 3 or min(resolution) < 4:
            raise ValueError("voxel_resolution must contain D, H, W values of at least four")
        if state_feature_dim < 4:
            raise ValueError("state_feature_dim must be at least four")
        if refinement_blocks < 0:
            raise ValueError("refinement_blocks cannot be negative")
        self.voxel_resolution = resolution
        self.image_feature_dim = image_feature_dim
        self.state_feature_dim = state_feature_dim
        self.image_encoder = ImageEncoder(image_feature_dim)
        observation_channels = image_feature_dim + 3 + 3 + 1
        fused_channels = observation_channels * 2 + 1
        hidden_channels = max(16, state_feature_dim * 2)
        self.input_projection = nn.Sequential(
            nn.Conv3d(fused_channels, hidden_channels, 3, padding=1, bias=False),
            nn.GroupNorm(_group_count(hidden_channels), hidden_channels),
            nn.SiLU(inplace=True),
        )
        self.refinement = nn.Sequential(
            *(Residual3DBlock(hidden_channels) for _ in range(refinement_blocks))
        )
        self.density_head = nn.Conv3d(hidden_channels, 1, 1)
        self.color_head = nn.Conv3d(hidden_channels, 3, 1)
        self.variance_head = nn.Conv3d(hidden_channels, 1, 1)
        self.feature_head = nn.Conv3d(hidden_channels, state_feature_dim, 1)
        nn.init.constant_(self.density_head.bias, -2.0)
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
        camera_centers = cameras.c2w[..., :3, 3].unsqueeze(-2)
        view_direction = _normalize(points_per_view - camera_centers)
        scene_diagonal = torch.linalg.vector_norm(bounds[:, 1] - bounds[:, 0], dim=-1)
        normalized_depth = camera_depth / scene_diagonal[:, None, None].clamp_min(1e-6)
        observations = torch.cat(
            (sampled_features, sampled_rgb, view_direction, normalized_depth.unsqueeze(-1)), dim=-1
        )
        mask = valid.unsqueeze(-1).to(observations.dtype)
        count = mask.sum(dim=1).clamp_min(1.0)
        mean = (observations * mask).sum(dim=1) / count
        variance = (((observations - mean[:, None]) ** 2) * mask).sum(dim=1) / count
        coverage = mask.sum(dim=1) / view_count
        fused = torch.cat((mean, variance, coverage), dim=-1)
        fused = fused.reshape(
            batch_size,
            *self.voxel_resolution,
            fused.shape[-1],
        ).permute(0, 4, 1, 2, 3)

        hidden = self.refinement(self.input_projection(fused))
        density_logits = self.density_head(hidden)
        color = torch.sigmoid(self.color_head(hidden))
        log_variance = self.variance_head(hidden).clamp(-8.0, 4.0)
        features = self.feature_head(hidden)
        return SceneState(density_logits, color, log_variance, bounds, features)


def _sample_images(images: Tensor, grid: Tensor) -> Tensor:
    batch_size, views, channels, height, width = images.shape
    point_count = grid.shape[-2]
    sampled = functional.grid_sample(
        images.reshape(batch_size * views, channels, height, width),
        grid.reshape(batch_size * views, point_count, 1, 2),
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )
    return sampled.reshape(batch_size, views, channels, point_count).permute(0, 1, 3, 2)


def _pixels_to_grid(pixels: Tensor, height: int, width: int) -> Tensor:
    x = pixels[..., 0] / max(width - 1, 1) * 2.0 - 1.0
    y = pixels[..., 1] / max(height - 1, 1) * 2.0 - 1.0
    return torch.stack((x, y), dim=-1)


def _normalize(values: Tensor) -> Tensor:
    return values / torch.linalg.vector_norm(values, dim=-1, keepdim=True).clamp_min(1e-8)


def _group_count(channels: int) -> int:
    for groups in (8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1
