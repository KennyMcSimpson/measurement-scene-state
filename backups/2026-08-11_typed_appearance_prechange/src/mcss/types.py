"""Typed tensor contracts shared by datasets, models, and measurements."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Self

import torch
from torch import Tensor


def _require_finite(name: str, value: Tensor) -> None:
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} must contain only finite values")


@dataclass
class Cameras:
    """Calibrated OpenCV cameras with arbitrary leading batch/view dimensions."""

    intrinsics: Tensor
    c2w: Tensor
    image_size: tuple[int, int]

    def __post_init__(self) -> None:
        height, width = self.image_size
        if height <= 0 or width <= 0:
            raise ValueError("image_size values must be positive")
        if self.intrinsics.ndim < 2 or self.intrinsics.shape[-2:] != (3, 3):
            raise ValueError("intrinsics must have shape [..., 3, 3]")
        if self.c2w.ndim < 2 or self.c2w.shape[-2:] != (4, 4):
            raise ValueError("c2w must have shape [..., 4, 4]")
        if self.intrinsics.shape[:-2] != self.c2w.shape[:-2]:
            raise ValueError("intrinsics and c2w must share leading dimensions")
        if self.intrinsics.device != self.c2w.device:
            raise ValueError("intrinsics and c2w must be on the same device")
        _require_finite("intrinsics", self.intrinsics)
        _require_finite("c2w", self.c2w)
        expected_row = torch.tensor(
            [0.0, 0.0, 0.0, 1.0], device=self.c2w.device, dtype=self.c2w.dtype
        )
        if not torch.allclose(self.c2w[..., 3, :], expected_row, atol=1e-5, rtol=1e-5):
            raise ValueError("c2w must have a homogeneous final row [0, 0, 0, 1]")
        if (self.intrinsics[..., 0, 0] <= 0).any() or (self.intrinsics[..., 1, 1] <= 0).any():
            raise ValueError("intrinsics must have positive focal lengths")

    @property
    def leading_shape(self) -> tuple[int, ...]:
        return tuple(self.intrinsics.shape[:-2])

    @property
    def device(self) -> torch.device:
        return self.intrinsics.device

    @property
    def dtype(self) -> torch.dtype:
        return self.intrinsics.dtype

    def to(self, *args: object, **kwargs: object) -> Self:
        return Cameras(
            self.intrinsics.to(*args, **kwargs),
            self.c2w.to(*args, **kwargs),
            self.image_size,
        )

    def select_views(self, indices: Tensor) -> Self:
        """Select the final leading (view) dimension without changing camera convention."""

        if not self.leading_shape:
            raise ValueError("cameras require a view dimension for select_views")
        if indices.ndim != 1:
            raise ValueError("indices must be one-dimensional")
        indices = indices.to(device=self.device, dtype=torch.long)
        return Cameras(
            self.intrinsics[..., indices, :, :],
            self.c2w[..., indices, :, :],
            self.image_size,
        )


@dataclass
class StateEvidence:
    """Inspectable evidence/completion fields aligned with one typed 3D state."""

    confidence: Tensor
    unknown_probability: Tensor
    completion_gate: Tensor
    provenance: Tensor
    base_density_logits: Tensor
    base_color: Tensor
    density_residual: Tensor
    color_logit_residual: Tensor

    def __post_init__(self) -> None:
        scalar_fields = {
            "confidence": self.confidence,
            "unknown_probability": self.unknown_probability,
            "completion_gate": self.completion_gate,
            "base_density_logits": self.base_density_logits,
            "density_residual": self.density_residual,
        }
        color_fields = {
            "base_color": self.base_color,
            "color_logit_residual": self.color_logit_residual,
        }
        for name, value in {**scalar_fields, **color_fields, "provenance": self.provenance}.items():
            if value.ndim != 5:
                raise ValueError(f"{name} must have shape [B, C, D, H, W]")
            _require_finite(name, value)
        if any(value.shape[1] != 1 for value in scalar_fields.values()):
            raise ValueError("scalar evidence fields require one channel")
        if any(value.shape[1] != 3 for value in color_fields.values()):
            raise ValueError("color evidence fields require three channels")
        if self.provenance.shape[1] < 1:
            raise ValueError("provenance requires at least one source-view channel")

        reference = self.confidence
        for name, value in scalar_fields.items():
            if value.shape != reference.shape:
                raise ValueError(f"{name} must match confidence shape")
        for name, value in color_fields.items():
            if value.shape[0] != reference.shape[0] or value.shape[2:] != reference.shape[2:]:
                raise ValueError(f"{name} must share confidence batch and spatial dimensions")
        if (
            self.provenance.shape[0] != reference.shape[0]
            or self.provenance.shape[2:] != reference.shape[2:]
        ):
            raise ValueError("provenance must share confidence batch and spatial dimensions")

        devices = {value.device for value in (*scalar_fields.values(), *color_fields.values())}
        devices.add(self.provenance.device)
        if len(devices) != 1:
            raise ValueError("evidence fields must be on one device")
        for name, value in {
            "confidence": self.confidence,
            "unknown_probability": self.unknown_probability,
            "completion_gate": self.completion_gate,
            "base_color": self.base_color,
            "provenance": self.provenance,
        }.items():
            if ((value < 0.0) | (value > 1.0)).any():
                raise ValueError(f"{name} must be within [0, 1]")
        tolerance = 5e-3 if self.confidence.dtype in {torch.float16, torch.bfloat16} else 1e-5
        if not torch.allclose(
            self.confidence + self.unknown_probability,
            torch.ones_like(self.confidence),
            atol=tolerance,
            rtol=tolerance,
        ):
            raise ValueError("unknown_probability must equal one minus confidence")

        provenance_sum = self.provenance.sum(dim=1, keepdim=True)
        normalized = torch.isclose(
            provenance_sum,
            torch.ones_like(provenance_sum),
            atol=tolerance,
            rtol=tolerance,
        )
        empty = torch.isclose(
            provenance_sum,
            torch.zeros_like(provenance_sum),
            atol=tolerance,
            rtol=tolerance,
        )
        if not (normalized | empty).all() or (empty & (self.confidence > tolerance)).any():
            raise ValueError("provenance must sum to one, or zero where confidence is zero")

    def to(self, *args: object, **kwargs: object) -> Self:
        return StateEvidence(
            self.confidence.to(*args, **kwargs),
            self.unknown_probability.to(*args, **kwargs),
            self.completion_gate.to(*args, **kwargs),
            self.provenance.to(*args, **kwargs),
            self.base_density_logits.to(*args, **kwargs),
            self.base_color.to(*args, **kwargs),
            self.density_residual.to(*args, **kwargs),
            self.color_logit_residual.to(*args, **kwargs),
        )


@dataclass
class SceneState:
    """Explicit fields read by the fixed measurement program."""

    density_logits: Tensor
    color: Tensor
    log_variance: Tensor
    bounds: Tensor
    features: Tensor | None = None
    evidence: StateEvidence | None = None

    def __post_init__(self) -> None:
        tensors = {
            "density_logits": self.density_logits,
            "color": self.color,
            "log_variance": self.log_variance,
        }
        for name, value in tensors.items():
            if value.ndim != 5:
                raise ValueError(f"{name} must have shape [B, C, D, H, W]")
            _require_finite(name, value)
        if self.density_logits.shape[1] != 1 or self.log_variance.shape[1] != 1:
            raise ValueError("density_logits and log_variance require one channel")
        if self.color.shape[1] != 3:
            raise ValueError("color requires three RGB channels")
        if self.color.shape[0] != self.density_logits.shape[0]:
            raise ValueError("typed fields must share batch size")
        if self.color.shape[2:] != self.density_logits.shape[2:]:
            raise ValueError("typed fields must share spatial dimensions")
        if self.log_variance.shape != self.density_logits.shape:
            raise ValueError("log_variance must match density_logits shape")
        if self.bounds.shape != (self.density_logits.shape[0], 2, 3):
            raise ValueError("bounds must have shape [B, 2, 3]")
        _require_finite("bounds", self.bounds)
        if not (self.bounds[:, 1] > self.bounds[:, 0]).all():
            raise ValueError("bounds maximum must exceed minimum on every axis")
        if self.features is not None:
            if self.features.ndim != 5 or self.features.shape[0] != self.density_logits.shape[0]:
                raise ValueError("features must have shape [B, C, D, H, W]")
            if self.features.shape[2:] != self.density_logits.shape[2:]:
                raise ValueError("features must share state spatial dimensions")
            _require_finite("features", self.features)
        if self.evidence is not None:
            if self.evidence.confidence.shape != self.density_logits.shape:
                raise ValueError("evidence confidence must match density_logits shape")
            if self.evidence.base_color.shape != self.color.shape:
                raise ValueError("evidence base_color must match color shape")
            if self.evidence.confidence.device != self.density_logits.device:
                raise ValueError("evidence and typed state fields must be on the same device")

    @property
    def spatial_shape(self) -> tuple[int, int, int]:
        return tuple(self.density_logits.shape[-3:])

    def to(self, *args: object, **kwargs: object) -> Self:
        return SceneState(
            self.density_logits.to(*args, **kwargs),
            self.color.to(*args, **kwargs),
            self.log_variance.to(*args, **kwargs),
            self.bounds.to(*args, **kwargs),
            None if self.features is None else self.features.to(*args, **kwargs),
            None if self.evidence is None else self.evidence.to(*args, **kwargs),
        )


@dataclass
class SceneExample:
    """One sparse-view scene item before DataLoader collation."""

    context_rgb: Tensor
    context_cameras: Cameras
    target_rgb: Tensor
    target_cameras: Cameras
    bounds: Tensor
    target_depth: Tensor | None = None
    target_normal: Tensor | None = None
    target_point: Tensor | None = None
    target_visibility: Tensor | None = None
    target_support: Tensor | None = None
    scene_id: str | None = None

    def __post_init__(self) -> None:
        _validate_example_or_batch(self, batched=False)

    def to(self, *args: object, **kwargs: object) -> Self:
        return SceneExample(
            self.context_rgb.to(*args, **kwargs),
            self.context_cameras.to(*args, **kwargs),
            self.target_rgb.to(*args, **kwargs),
            self.target_cameras.to(*args, **kwargs),
            self.bounds.to(*args, **kwargs),
            _optional_to(self.target_depth, *args, **kwargs),
            _optional_to(self.target_normal, *args, **kwargs),
            _optional_to(self.target_point, *args, **kwargs),
            _optional_to(self.target_visibility, *args, **kwargs),
            _optional_to(self.target_support, *args, **kwargs),
            self.scene_id,
        )


@dataclass
class SceneBatch:
    """A collated batch of context observations and hidden target supervision."""

    context_rgb: Tensor
    context_cameras: Cameras
    target_rgb: Tensor
    target_cameras: Cameras
    bounds: Tensor
    target_depth: Tensor | None = None
    target_normal: Tensor | None = None
    target_point: Tensor | None = None
    target_visibility: Tensor | None = None
    target_support: Tensor | None = None
    scene_ids: tuple[str | None, ...] = ()

    def __post_init__(self) -> None:
        _validate_example_or_batch(self, batched=True)

    def to(self, *args: object, **kwargs: object) -> Self:
        return SceneBatch(
            self.context_rgb.to(*args, **kwargs),
            self.context_cameras.to(*args, **kwargs),
            self.target_rgb.to(*args, **kwargs),
            self.target_cameras.to(*args, **kwargs),
            self.bounds.to(*args, **kwargs),
            _optional_to(self.target_depth, *args, **kwargs),
            _optional_to(self.target_normal, *args, **kwargs),
            _optional_to(self.target_point, *args, **kwargs),
            _optional_to(self.target_visibility, *args, **kwargs),
            _optional_to(self.target_support, *args, **kwargs),
            self.scene_ids,
        )

    def targets(self) -> dict[str, Tensor]:
        result = {"rgb": self.target_rgb}
        optional = {
            "depth": self.target_depth,
            "normal": self.target_normal,
            "point": self.target_point,
            "visibility": self.target_visibility,
        }
        result.update({name: value for name, value in optional.items() if value is not None})
        if self.target_support is not None:
            result["support"] = self.target_support
        return result


def _optional_to(value: Tensor | None, *args: object, **kwargs: object) -> Tensor | None:
    return None if value is None else value.to(*args, **kwargs)


def _validate_example_or_batch(value: SceneExample | SceneBatch, *, batched: bool) -> None:
    rgb_ndim = 5 if batched else 4
    camera_shape = 2 if batched else 1
    bounds_shape = (value.context_rgb.shape[0], 2, 3) if batched else (2, 3)
    if value.context_rgb.ndim != rgb_ndim or value.context_rgb.shape[-3] != 3:
        raise ValueError("context_rgb must have RGB channel-first image shape")
    if value.target_rgb.ndim != rgb_ndim or value.target_rgb.shape[-3] != 3:
        raise ValueError("target_rgb must have RGB channel-first image shape")
    if value.context_rgb.shape[-2:] != value.context_cameras.image_size:
        raise ValueError("context_rgb spatial size must match context cameras")
    if value.target_rgb.shape[-2:] != value.target_cameras.image_size:
        raise ValueError("target_rgb spatial size must match target cameras")
    if len(value.context_cameras.leading_shape) != camera_shape:
        raise ValueError("context camera leading dimensions do not match example/batch shape")
    if len(value.target_cameras.leading_shape) != camera_shape:
        raise ValueError("target camera leading dimensions do not match example/batch shape")
    if batched:
        if value.context_cameras.leading_shape != value.context_rgb.shape[:2]:
            raise ValueError("context camera dimensions must match context_rgb")
        if value.target_cameras.leading_shape != value.target_rgb.shape[:2]:
            raise ValueError("target camera dimensions must match target_rgb")
    else:
        if value.context_cameras.leading_shape != value.context_rgb.shape[:1]:
            raise ValueError("context camera dimensions must match context_rgb")
        if value.target_cameras.leading_shape != value.target_rgb.shape[:1]:
            raise ValueError("target camera dimensions must match target_rgb")
    if value.bounds.shape != bounds_shape:
        raise ValueError("bounds shape does not match example/batch")
    _require_finite("context_rgb", value.context_rgb)
    _require_finite("target_rgb", value.target_rgb)
    _require_finite("bounds", value.bounds)
    target_views = value.target_rgb.shape[:-3]
    expected = {
        "target_depth": (1,),
        "target_normal": (3,),
        "target_point": (3,),
        "target_visibility": (1,),
        "target_support": (1,),
    }
    for name, channels in expected.items():
        tensor = getattr(value, name)
        if tensor is None:
            continue
        if tensor.shape[: len(target_views)] != target_views or tensor.shape[-3] != channels[0]:
            raise ValueError(f"{name} must align with target views and channel count")
        if tensor.shape[-2:] != value.target_rgb.shape[-2:]:
            raise ValueError(f"{name} spatial size must match target_rgb")
        _require_finite(name, tensor)
    if isinstance(value, SceneExample):
        if value.scene_id is not None and not isinstance(value.scene_id, str):
            raise ValueError("scene_id must be a string or null")
    elif value.scene_ids and len(value.scene_ids) != value.context_rgb.shape[0]:
        raise ValueError("scene_ids must align with the batch dimension")
