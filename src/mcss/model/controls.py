"""Capacity-matched learned target predictors used only as experimental controls."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from mcss.geometry import generate_rays
from mcss.measurements import FixedMeasurementRenderer
from mcss.types import Cameras, SceneState


class LearnedHeadsControl(nn.Module):
    """Predict each measurement with a separate learned per-ray head."""

    def __init__(self, feature_dim: int, renderer: FixedMeasurementRenderer) -> None:
        super().__init__()
        self.renderer = renderer
        self.heads = nn.ModuleDict(
            {
                "rgb": nn.Conv2d(feature_dim, 3, 1),
                "depth": nn.Conv2d(feature_dim, 1, 1),
                "normal": nn.Conv2d(feature_dim, 3, 1),
                "point": nn.Conv2d(feature_dim, 3, 1),
                "visibility": nn.Conv2d(feature_dim, 1, 1),
                "uncertainty": nn.Conv2d(feature_dim, 1, 1),
            }
        )

    def forward(self, state: SceneState, cameras: Cameras) -> dict[str, Tensor]:
        features = self.renderer.render_features(state, cameras)
        batch_size, views, channels, height, width = features.shape
        flat = features.reshape(batch_size * views, channels, height, width)
        raw = {name: head(flat) for name, head in self.heads.items()}
        return _activate_and_reshape(raw, batch_size, views)


class FreeDecoderControl(nn.Module):
    """A query-conditioned learned decoder over integrated latent state features."""

    def __init__(self, feature_dim: int, renderer: FixedMeasurementRenderer) -> None:
        super().__init__()
        self.renderer = renderer
        hidden = max(16, feature_dim * 2)
        self.decoder = nn.Sequential(
            nn.Conv2d(feature_dim + 6, hidden, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, hidden, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, 12, 1),
        )

    def forward(self, state: SceneState, cameras: Cameras) -> dict[str, Tensor]:
        features = self.renderer.render_features(state, cameras)
        origins, directions = generate_rays(cameras)
        query = torch.cat((origins, directions), dim=-1).permute(0, 1, 4, 2, 3)
        batch_size, views, channels, height, width = features.shape
        decoded = self.decoder(
            torch.cat((features, query), dim=2).reshape(
                batch_size * views, channels + 6, height, width
            )
        )
        raw = {
            "rgb": decoded[:, 0:3],
            "depth": decoded[:, 3:4],
            "normal": decoded[:, 4:7],
            "point": decoded[:, 7:10],
            "visibility": decoded[:, 10:11],
            "uncertainty": decoded[:, 11:12],
        }
        return _activate_and_reshape(raw, batch_size, views)


def _activate_and_reshape(raw: dict[str, Tensor], batch_size: int, views: int) -> dict[str, Tensor]:
    outputs = {
        "rgb": torch.sigmoid(raw["rgb"]),
        "depth": torch.nn.functional.softplus(raw["depth"]),
        "normal": _normalize_channels(raw["normal"]),
        "point": raw["point"],
        "visibility": torch.sigmoid(raw["visibility"]),
        "uncertainty": torch.nn.functional.softplus(raw["uncertainty"]),
    }
    return {
        name: value.reshape(batch_size, views, value.shape[1], value.shape[2], value.shape[3])
        for name, value in outputs.items()
    }


def _normalize_channels(values: Tensor) -> Tensor:
    return values / torch.linalg.vector_norm(values, dim=1, keepdim=True).clamp_min(1e-8)
