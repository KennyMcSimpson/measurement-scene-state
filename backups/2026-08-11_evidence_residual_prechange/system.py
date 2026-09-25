"""Top-level scene-state systems and controlled model construction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from torch import Tensor, nn

from mcss.measurements import FixedMeasurementRenderer
from mcss.model.controls import FreeDecoderControl, LearnedHeadsControl
from mcss.model.state_encoder import UnprojectiveStateEncoder
from mcss.types import Cameras, SceneState

ModelMode = Literal["fixed", "learned_heads", "free_decoder"]


@dataclass(frozen=True)
class ModelConfig:
    mode: ModelMode = "fixed"
    voxel_resolution: int | tuple[int, int, int] = 32
    image_feature_dim: int = 32
    state_feature_dim: int = 32
    refinement_blocks: int = 3
    n_samples: int = 64
    ray_chunk_size: int = 16_384

    def __post_init__(self) -> None:
        if self.mode not in {"fixed", "learned_heads", "free_decoder"}:
            raise ValueError(f"unsupported model mode: {self.mode}")
        if isinstance(self.voxel_resolution, int):
            resolution = (self.voxel_resolution,) * 3
        else:
            resolution = tuple(self.voxel_resolution)
        if len(resolution) != 3 or min(resolution) < 4:
            raise ValueError("voxel_resolution must contain D, H, W values of at least four")
        object.__setattr__(self, "voxel_resolution", resolution)


@dataclass
class ModelOutput:
    predictions: dict[str, Tensor]
    state: SceneState


class MeasurementCompleteSystem(nn.Module):
    """Construct one state from context RGB and answer target camera queries."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.state_encoder = UnprojectiveStateEncoder(
            voxel_resolution=config.voxel_resolution,
            image_feature_dim=config.image_feature_dim,
            state_feature_dim=config.state_feature_dim,
            refinement_blocks=config.refinement_blocks,
        )
        self.renderer = FixedMeasurementRenderer(
            n_samples=config.n_samples, ray_chunk_size=config.ray_chunk_size
        )
        if config.mode == "fixed":
            self.control: nn.Module | None = None
        elif config.mode == "learned_heads":
            self.control = LearnedHeadsControl(config.state_feature_dim, self.renderer)
        else:
            self.control = FreeDecoderControl(config.state_feature_dim, self.renderer)

    def forward(
        self,
        context_rgb: Tensor,
        context_cameras: Cameras,
        target_cameras: Cameras,
        bounds: Tensor,
    ) -> ModelOutput:
        state = self.state_encoder(context_rgb, context_cameras, bounds)
        if self.control is None:
            predictions = self.renderer(state, target_cameras)
        else:
            predictions = self.control(state, target_cameras)
        return ModelOutput(predictions, state)


def build_model(config: ModelConfig) -> MeasurementCompleteSystem:
    return MeasurementCompleteSystem(config)
