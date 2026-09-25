"""Runtime-only values: no dataset, teacher, or query readers are imported here."""

from dataclasses import dataclass, field, fields, is_dataclass
from enum import StrEnum
from hashlib import sha256
from math import isfinite
from numbers import Integral, Real
from uuid import uuid4

import torch
from torch import Tensor

from mcss.types import Cameras, SceneState


class Action(StrEnum):
    OFF = "OFF"
    FUSE = "FUSE"
    COMPLETE = "COMPLETE"
    ALL = "ALL"


ACTION_ORDER: tuple[Action, ...] = (Action.OFF, Action.FUSE, Action.COMPLETE, Action.ALL)


@dataclass(frozen=True)
class OnlineObservation:
    scene_id: str
    frame_id: int
    rgb: Tensor
    camera: Cameras

    def __post_init__(self):
        if not self.scene_id or not isinstance(self.frame_id, int) or self.frame_id < 0:
            raise ValueError("Observation identity is invalid")
        if self.rgb.shape != (3, *self.camera.image_size) or self.camera.leading_shape:
            raise ValueError("One observation requires RGB [3,H,W] and one unbatched camera")
        if self.rgb.device != self.camera.device or self.rgb.dtype != self.camera.dtype:
            raise ValueError("Observation RGB and camera must share device and dtype")
        if not torch.isfinite(self.rgb).all() or ((self.rgb < 0) | (self.rgb > 1)).any():
            raise ValueError("Observation RGB must be finite and within [0,1]")


@dataclass(frozen=True)
class FastWeights:
    episode_id: str
    delta_fuse: Tensor
    delta_complete: Tensor
    step: int = 0
    branch_id: str = field(default_factory=lambda: uuid4().hex)
    state_id: str = field(default_factory=lambda: uuid4().hex)

    def __post_init__(self):
        if not self.episode_id or not self.branch_id or not self.state_id or self.step < 0:
            raise ValueError("Fast state identity/version is invalid")
        if self.delta_fuse.ndim != 2 or self.delta_complete.shape != self.delta_fuse.shape:
            raise ValueError("Fast matrices must have matching [d,m] shapes")
        if self.delta_fuse.device != self.delta_complete.device:
            raise ValueError("Fast matrices must share device")
        if not all(torch.isfinite(t).all() for t in (self.delta_fuse, self.delta_complete)):
            raise ValueError("Nonfinite fast weights")

    def fork(self) -> "FastWeights":
        """Create a private trajectory branch; clones retain offline autograd paths."""
        return FastWeights(
            self.episode_id, self.delta_fuse.clone(), self.delta_complete.clone(), self.step
        )


@dataclass(frozen=True)
class WriteTrace:
    episode_id: str
    cache_revision: int
    fast_step: int
    fuse_activation: Tensor
    complete_activation: Tensor
    observed_feature_statistics: Tensor
    support_weights: Tensor
    candidate_ids: Tensor
    branch_id: str
    source_state_id: str


@dataclass(frozen=True)
class WriteProposal:
    episode_id: str
    cache_revision: int
    fast_step: int
    delta_fuse: Tensor
    delta_complete: Tensor
    branch_id: str
    source_state_id: str


@dataclass(frozen=True)
class ControlInput:
    """Values available to the policy before the current observation is cached.

    The first eight fields are the original runtime contract. New fields are
    optional so old callers can keep constructing a control input positionally.
    Tensors are accepted at the boundary for local use, but the runner records
    the serializable summary returned by :meth:`to_serializable`.
    """

    rgb_mse: float
    opacity_mean: float
    current_image_mean: float
    density_mean: float
    fast_norm: float
    observed_count: int
    previous_action: Action
    remaining_budget: float
    image_feature: Tensor | None = None
    image_feature_stats: Tensor | tuple[float, ...] | None = None
    rgb_residual_4x4: Tensor | tuple[float, ...] | None = None
    coverage: float = 0.0
    valid_count: int = 0
    camera_change: Tensor | tuple[float, ...] | None = None
    fast_state_stats: Tensor | tuple[float, ...] | None = None
    remaining_steps: int = 0

    def __post_init__(self):
        object.__setattr__(self, "previous_action", Action(self.previous_action))
        for name in (
            "rgb_mse",
            "opacity_mean",
            "current_image_mean",
            "density_mean",
            "fast_norm",
            "remaining_budget",
            "coverage",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value):
                raise ValueError(f"ControlInput {name} must be finite")
        if self.remaining_budget < 0:
            raise ValueError("ControlInput remaining_budget must be nonnegative")
        for name in ("observed_count", "valid_count", "remaining_steps"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
                raise ValueError(f"ControlInput {name} must be a nonnegative integer")

    def to_serializable(self) -> dict[str, object]:
        """Return a JSON-compatible prefix record without the raw feature map."""

        def values(value):
            if value is None:
                return None
            if isinstance(value, Tensor):
                value = value.detach().cpu().reshape(-1).tolist()
            return [float(item) for item in value]

        return {
            "rgb_mse": float(self.rgb_mse),
            "opacity_mean": float(self.opacity_mean),
            "current_image_mean": float(self.current_image_mean),
            "density_mean": float(self.density_mean),
            "fast_norm": float(self.fast_norm),
            "observed_count": int(self.observed_count),
            "previous_action": str(self.previous_action),
            "remaining_budget": float(self.remaining_budget),
            "image_feature_stats": values(self.image_feature_stats),
            "rgb_residual_4x4": values(self.rgb_residual_4x4),
            "coverage": float(self.coverage),
            "valid_count": int(self.valid_count),
            "camera_change": values(self.camera_change),
            "fast_state_stats": values(self.fast_state_stats),
            "remaining_steps": int(self.remaining_steps),
        }


@dataclass(frozen=True)
class SealedScene:
    episode_id: str
    scene_id: str
    split_id: str
    query_vault_id: str
    scene_state: SceneState
    state_hash: str
    observed_ids: tuple[int, ...]
    checkpoint_hash: str
    config_hash: str
    fast_state_hash: str
    anchor_c2w: Tensor


def hash_value(value) -> str:
    """Hash content, shape and dtype; device location is not part of tensor identity."""
    digest = sha256()

    def visit(item):
        if isinstance(item, Tensor):
            data = item.detach().contiguous().cpu()
            digest.update(f"Tensor:{data.dtype}:{tuple(data.shape)}:".encode())
            digest.update(data.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif is_dataclass(item):
            digest.update(type(item).__name__.encode())
            for field in fields(item):
                digest.update(field.name.encode())
                visit(getattr(item, field.name))
        elif isinstance(item, dict):
            for key in sorted(item):
                visit(key)
                visit(item[key])
        elif isinstance(item, (list, tuple)):
            digest.update(f"sequence:{len(item)}:".encode())
            for child in item:
                visit(child)
        else:
            digest.update(f"{type(item).__name__}:{item!r};".encode())

    visit(value)
    return digest.hexdigest()


def hash_scene_state(state: SceneState) -> str:
    return hash_value(state)


def hash_fast_content(fast: FastWeights) -> str:
    """Reproducible numerical identity, excluding the runtime-only branch nonce."""
    return hash_value(
        {"delta_fuse": fast.delta_fuse, "delta_complete": fast.delta_complete, "step": fast.step}
    )


def clone_scene_state(state: SceneState) -> SceneState:
    def clone(item):
        if isinstance(item, Tensor):
            return item.detach().clone()
        if is_dataclass(item):
            return type(item)(
                **{field.name: clone(getattr(item, field.name)) for field in fields(item)}
            )
        return item

    return clone(state)
