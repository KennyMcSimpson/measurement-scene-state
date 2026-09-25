"""Compact context-image feature encoder."""

from __future__ import annotations

import torch
import torch.nn.functional as functional
from torch import Tensor, nn


class ConvNormAct(nn.Sequential):
    def __init__(self, input_channels: int, output_channels: int, *, stride: int = 1) -> None:
        super().__init__(
            nn.Conv2d(input_channels, output_channels, 3, stride=stride, padding=1, bias=False),
            nn.GroupNorm(_group_count(output_channels), output_channels),
            nn.SiLU(inplace=True),
        )


class Residual2DBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            ConvNormAct(channels, channels),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(_group_count(channels), channels),
        )
        self.activation = nn.SiLU(inplace=True)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.activation(inputs + self.block(inputs))


class ImageEncoder(nn.Module):
    """Encode each context view independently while preserving normalized image coordinates."""

    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        if feature_dim < 4:
            raise ValueError("feature_dim must be at least four")
        self.network = nn.Sequential(
            ConvNormAct(3, feature_dim, stride=2),
            Residual2DBlock(feature_dim),
            Residual2DBlock(feature_dim),
        )

    def forward(self, images: Tensor) -> Tensor:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("images must have shape [N, 3, H, W]")
        return self.network(images)


class PyramidImageEncoder(nn.Module):
    """Fuse half- and quarter-resolution context features at a stable output stride of two."""

    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        if feature_dim < 4:
            raise ValueError("feature_dim must be at least four")
        coarse_dim = feature_dim * 2
        self.fine = nn.Sequential(
            ConvNormAct(3, feature_dim, stride=2),
            Residual2DBlock(feature_dim),
            Residual2DBlock(feature_dim),
        )
        self.coarse = nn.Sequential(
            ConvNormAct(feature_dim, coarse_dim, stride=2),
            Residual2DBlock(coarse_dim),
            Residual2DBlock(coarse_dim),
        )
        self.lateral = nn.Conv2d(coarse_dim, feature_dim, 1, bias=False)
        self.fusion = nn.Sequential(
            ConvNormAct(feature_dim * 2, feature_dim),
            Residual2DBlock(feature_dim),
        )

    def forward(self, images: Tensor) -> Tensor:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("images must have shape [N, 3, H, W]")
        fine = self.fine(images)
        coarse = self.lateral(self.coarse(fine))
        coarse = functional.interpolate(
            coarse, size=fine.shape[-2:], mode="bilinear", align_corners=False
        )
        return self.fusion(torch.cat((fine, coarse), dim=1))


def _group_count(channels: int) -> int:
    for groups in (8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1
