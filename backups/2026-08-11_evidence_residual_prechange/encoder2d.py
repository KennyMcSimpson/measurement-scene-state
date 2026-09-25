"""Compact context-image feature encoder."""

from __future__ import annotations

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


def _group_count(channels: int) -> int:
    for groups in (8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1
