"""Diagnostic-only color-volume oracles for locating appearance capacity limits."""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn.functional as functional
from torch import Tensor, nn

from mcss.measurements import FixedMeasurementRenderer
from mcss.types import Cameras, SceneState, StateAppearance

OracleField = Literal["native", "typed_appearance"]
OracleVariant = Literal[
    "native_shared",
    "highres_shared",
    "native_per_view",
    "typed_shared",
    "typed_per_view",
]

NATIVE_VARIANTS = frozenset({"native_shared", "highres_shared", "native_per_view"})
TYPED_VARIANTS = frozenset({"typed_shared", "typed_per_view"})


class AppearanceOracle(nn.Module):
    """Optimize target-label color volumes while keeping accepted geometry fixed.

    This module is an inadmissible diagnostic upper bound. It must never be used as a scene
    encoder or reported as a model result.
    """

    def __init__(
        self,
        reference_state: SceneState,
        *,
        target_views: int,
        variant: OracleVariant,
        field: OracleField = "native",
    ) -> None:
        super().__init__()
        if target_views < 1:
            raise ValueError("target_views must be positive")
        if field not in {"native", "typed_appearance"}:
            raise ValueError(f"unsupported appearance oracle field: {field}")
        allowed_variants = NATIVE_VARIANTS if field == "native" else TYPED_VARIANTS
        if variant not in allowed_variants:
            raise ValueError(f"unsupported appearance oracle variant: {variant}")
        if field == "typed_appearance" and reference_state.appearance is None:
            raise ValueError("typed_appearance oracle requires SceneState.StateAppearance")

        self.field = field
        self.variant = variant
        self.target_views = target_views
        source_color = reference_state.color
        if field == "typed_appearance":
            if reference_state.appearance is None:
                raise RuntimeError("typed appearance validation unexpectedly failed")
            source_color = reference_state.appearance.color
        native_logits = torch.logit(source_color.detach().clamp(1e-4, 1.0 - 1e-4))
        density = reference_state.density_logits.detach().clone()
        log_variance = reference_state.log_variance.detach().clone()
        if variant == "highres_shared":
            size = tuple(axis * 2 for axis in reference_state.spatial_shape)
            native_logits = functional.interpolate(
                native_logits, size=size, mode="trilinear", align_corners=False
            )
            density = functional.interpolate(
                density, size=size, mode="trilinear", align_corners=False
            )
            log_variance = functional.interpolate(
                log_variance, size=size, mode="trilinear", align_corners=False
            )
        if variant in {"native_per_view", "typed_per_view"}:
            native_logits = native_logits.unsqueeze(1).expand(-1, target_views, -1, -1, -1, -1)

        self.color_logits = nn.Parameter(native_logits.clone())
        self.register_buffer("density_logits", density)
        self.register_buffer("log_variance", log_variance)
        self.register_buffer("bounds", reference_state.bounds.detach().clone())
        self.register_buffer("native_color", reference_state.color.detach().clone())
        self.appearance_provenance_channels = (
            0
            if reference_state.appearance is None
            else reference_state.appearance.provenance.shape[1]
        )

    @property
    def spatial_shape(self) -> tuple[int, int, int]:
        return tuple(self.color_logits.shape[-3:])

    @property
    def parameter_count(self) -> int:
        return self.color_logits.numel()

    def render_rgb(self, renderer: FixedMeasurementRenderer, cameras: Cameras) -> Tensor:
        if len(cameras.leading_shape) != 2:
            raise ValueError("appearance oracle expects cameras with shape [B, V, ...]")
        batch_size, view_count = cameras.leading_shape
        if batch_size != self.density_logits.shape[0]:
            raise ValueError("camera and oracle batch sizes must match")
        if view_count != self.target_views:
            raise ValueError("camera view count must match target_views")

        if self.variant not in {"native_per_view", "typed_per_view"}:
            state = self._state(torch.sigmoid(self.color_logits))
            return renderer(state, cameras, {"rgb"})["rgb"]

        rendered = []
        for view_index in range(self.target_views):
            indices = torch.tensor([view_index], device=cameras.device)
            state = self._state(torch.sigmoid(self.color_logits[:, view_index]))
            rendered.append(renderer(state, cameras.select_views(indices), {"rgb"})["rgb"])
        return torch.cat(rendered, dim=1)

    def _state(self, color: Tensor) -> SceneState:
        if self.field == "typed_appearance":
            scalar_shape = (color.shape[0], 1, *color.shape[-3:])
            scalar = torch.zeros(scalar_shape, device=color.device, dtype=color.dtype)
            provenance = torch.zeros(
                color.shape[0],
                self.appearance_provenance_channels,
                *color.shape[-3:],
                device=color.device,
                dtype=color.dtype,
            )
            appearance = StateAppearance(
                confidence=scalar,
                unknown_probability=torch.ones_like(scalar),
                completion_gate=torch.ones_like(scalar),
                provenance=provenance,
                base_color=torch.full_like(color, 0.5),
                color_logit_residual=torch.logit(color.clamp(1e-4, 1.0 - 1e-4)),
            )
            return SceneState(
                self.density_logits,
                self.native_color,
                self.log_variance,
                self.bounds,
                appearance=appearance,
            )
        return SceneState(
            self.density_logits,
            color,
            self.log_variance,
            self.bounds,
        )
