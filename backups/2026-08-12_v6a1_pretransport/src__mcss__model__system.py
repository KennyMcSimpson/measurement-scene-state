"""Top-level scene-state systems and controlled model construction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from torch import Tensor, nn

from mcss.measurements import FixedMeasurementRenderer
from mcss.model.controls import FreeDecoderControl, LearnedHeadsControl
from mcss.model.evidence_encoder import DualEvidenceStateEncoder, EvidenceResidualStateEncoder
from mcss.model.state_encoder import UnprojectiveStateEncoder
from mcss.types import Cameras, SceneState

ModelMode = Literal["fixed", "learned_heads", "free_decoder"]
StateArchitecture = Literal["legacy", "evidence_residual", "dual_evidence"]
AppearanceResolutionScale = Literal[1, 2]


@dataclass(frozen=True)
class ModelConfig:
    mode: ModelMode = "fixed"
    state_architecture: StateArchitecture = "legacy"
    voxel_resolution: int | tuple[int, int, int] = 32
    image_feature_dim: int = 32
    state_feature_dim: int = 32
    refinement_blocks: int = 3
    evidence_temperature: float = 0.1
    observed_residual_floor: float = 0.1
    completion_residual_scale: float = 1.5
    surface_peak_temperature: float = 0.05
    appearance_resolution_scale: AppearanceResolutionScale = 1
    n_samples: int = 64
    ray_chunk_size: int = 16_384

    def __post_init__(self) -> None:
        if self.mode not in {"fixed", "learned_heads", "free_decoder"}:
            raise ValueError(f"unsupported model mode: {self.mode}")
        if self.state_architecture not in {"legacy", "evidence_residual", "dual_evidence"}:
            raise ValueError(f"unsupported state architecture: {self.state_architecture}")
        if self.evidence_temperature <= 0:
            raise ValueError("evidence_temperature must be positive")
        if not 0.0 <= self.observed_residual_floor <= 1.0:
            raise ValueError("observed_residual_floor must be within [0, 1]")
        if self.completion_residual_scale <= 0:
            raise ValueError("completion_residual_scale must be positive")
        if self.surface_peak_temperature <= 0:
            raise ValueError("surface_peak_temperature must be positive")
        if self.appearance_resolution_scale not in {1, 2}:
            raise ValueError("appearance_resolution_scale must be one or two")
        if self.state_architecture == "legacy" and self.appearance_resolution_scale != 1:
            raise ValueError(
                "appearance_resolution_scale=2 requires state_architecture: evidence_residual"
            )
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
        if config.state_architecture == "legacy":
            self.state_encoder = UnprojectiveStateEncoder(
                voxel_resolution=config.voxel_resolution,
                image_feature_dim=config.image_feature_dim,
                state_feature_dim=config.state_feature_dim,
                refinement_blocks=config.refinement_blocks,
            )
        elif config.state_architecture == "evidence_residual":
            self.state_encoder = EvidenceResidualStateEncoder(
                voxel_resolution=config.voxel_resolution,
                image_feature_dim=config.image_feature_dim,
                state_feature_dim=config.state_feature_dim,
                refinement_blocks=config.refinement_blocks,
                evidence_temperature=config.evidence_temperature,
                observed_residual_floor=config.observed_residual_floor,
                completion_residual_scale=config.completion_residual_scale,
                appearance_resolution_scale=config.appearance_resolution_scale,
            )
        else:
            self.state_encoder = DualEvidenceStateEncoder(
                voxel_resolution=config.voxel_resolution,
                image_feature_dim=config.image_feature_dim,
                state_feature_dim=config.state_feature_dim,
                refinement_blocks=config.refinement_blocks,
                evidence_temperature=config.evidence_temperature,
                surface_peak_temperature=config.surface_peak_temperature,
                observed_residual_floor=config.observed_residual_floor,
                completion_residual_scale=config.completion_residual_scale,
                appearance_resolution_scale=config.appearance_resolution_scale,
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
