"""Prefix-only action policies and the fixed deployment feature schema."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import isfinite, log1p

import torch
import torch.nn.functional as functional
from torch import Tensor, nn

from mcss.dynamic.types import ACTION_ORDER, Action, ControlInput

FEATURE_SCHEMA_VERSION = "mcss.dynamic.policy_features.v1"
POLICY_FEATURE_SCHEMA_VERSION = FEATURE_SCHEMA_VERSION
RGB_RESIDUAL_GRID = (4, 4)
IMAGE_FEATURE_STATS_DIM = 4
CAMERA_CHANGE_DIM = 6
FAST_STATE_STATS_DIM = 4

_SCALAR_NAMES = (
    "rgb_mse",
    "opacity_mean",
    "current_image_mean",
    "density_mean",
)
_IMAGE_NAMES = (
    "image_feature_mean",
    "image_feature_std",
    "image_feature_min",
    "image_feature_max",
)
_RESIDUAL_NAMES = tuple(
    f"rgb_residual_c{channel}_r{row}_c{column}"
    for channel in range(3)
    for row in range(RGB_RESIDUAL_GRID[0])
    for column in range(RGB_RESIDUAL_GRID[1])
)
_PREFIX_NAMES = ("coverage", "valid_count_log1p", "observed_count_log1p")
_CAMERA_NAMES = (
    "camera_translation_x",
    "camera_translation_y",
    "camera_translation_z",
    "camera_rotation_x",
    "camera_rotation_y",
    "camera_rotation_z",
)
_FAST_NAMES = (
    "fast_fuse_norm",
    "fast_complete_norm",
    "fast_total_norm",
    "fast_max_abs",
)
_ACTION_NAMES = tuple(f"previous_action_{action.value.lower()}" for action in ACTION_ORDER)
_BUDGET_NAMES = ("remaining_budget_log1p", "remaining_steps_log1p")

FEATURE_NAMES = (
    _SCALAR_NAMES
    + _IMAGE_NAMES
    + _RESIDUAL_NAMES
    + _PREFIX_NAMES
    + _CAMERA_NAMES
    + _FAST_NAMES
    + _ACTION_NAMES
    + _BUDGET_NAMES
)
POLICY_FEATURE_NAMES = FEATURE_NAMES
FEATURE_SCHEMA = FEATURE_NAMES
POLICY_FEATURE_DIM = len(FEATURE_NAMES)


def _finite_tensor(value: object, *, name: str) -> Tensor:
    if isinstance(value, Tensor):
        tensor = value.detach()
    else:
        try:
            tensor = torch.as_tensor(value)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{name} must be numeric") from error
    if tensor.numel() == 0 or not torch.isfinite(tensor).all():
        raise ValueError(f"{name} must contain finite values")
    return tensor.to(dtype=torch.float32).reshape(-1)


def _fixed_vector(value: object, size: int, *, name: str) -> Tensor:
    if value is None:
        return torch.zeros(size, dtype=torch.float32)
    vector = _finite_tensor(value, name=name)
    if vector.numel() != size:
        raise ValueError(f"{name} must contain exactly {size} values")
    return vector


def _image_stats(control: ControlInput) -> Tensor:
    if control.image_feature_stats is not None:
        stats = _finite_tensor(control.image_feature_stats, name="image_feature_stats")
        if stats.numel() == IMAGE_FEATURE_STATS_DIM:
            return stats.cpu()
        raise ValueError(
            f"image_feature_stats must contain exactly {IMAGE_FEATURE_STATS_DIM} values"
        )
    if control.image_feature is None:
        return torch.zeros(IMAGE_FEATURE_STATS_DIM, dtype=torch.float32)
    values = _finite_tensor(control.image_feature, name="image_feature")
    return torch.stack(
        (values.mean(), values.std(unbiased=False), values.min(), values.max())
    ).cpu()


def _residual_summary(value: object) -> Tensor:
    size = 3 * RGB_RESIDUAL_GRID[0] * RGB_RESIDUAL_GRID[1]
    if value is None:
        return torch.zeros(size, dtype=torch.float32)
    tensor = _finite_tensor(value, name="rgb_residual_4x4")
    if tensor.numel() == size:
        return tensor.cpu()
    if isinstance(value, Tensor):
        raw = value.detach().to(dtype=torch.float32)
    else:
        raw = torch.as_tensor(value, dtype=torch.float32)
    if raw.ndim == 4 and raw.shape[0] == 1:
        raw = raw[0]
    if raw.ndim != 3 or raw.shape[0] != 3:
        raise ValueError("rgb_residual_4x4 must be [3,4,4] or a 3-channel image residual")
    pooled = functional.adaptive_avg_pool2d(raw.unsqueeze(0), RGB_RESIDUAL_GRID)[0]
    return pooled.reshape(-1).cpu()


def _action_one_hot(action: Action) -> Tensor:
    result = torch.zeros(len(ACTION_ORDER), dtype=torch.float32)
    result[ACTION_ORDER.index(Action(action))] = 1.0
    return result


def control_to_features(control: ControlInput) -> Tensor:
    """Encode only the currently observed control prefix in fixed schema order."""

    if not isinstance(control, ControlInput):
        raise TypeError("control must be a ControlInput")
    scalars = torch.tensor(
        [control.rgb_mse, control.opacity_mean, control.current_image_mean, control.density_mean],
        dtype=torch.float32,
    )
    prefix = torch.tensor(
        [
            control.coverage,
            log1p(float(control.valid_count)),
            log1p(float(control.observed_count)),
        ],
        dtype=torch.float32,
    )
    fast_stats = _fixed_vector(
        control.fast_state_stats,
        FAST_STATE_STATS_DIM,
        name="fast_state_stats",
    ).cpu()
    if control.fast_state_stats is None:
        fast_stats = torch.tensor(
            [float(control.fast_norm), 0.0, float(control.fast_norm), 0.0], dtype=torch.float32
        )
    camera = _fixed_vector(
        control.camera_change,
        CAMERA_CHANGE_DIM,
        name="camera_change",
    ).cpu()
    budget = torch.tensor(
        [
            log1p(max(float(control.remaining_budget), 0.0)),
            log1p(float(control.remaining_steps)),
        ],
        dtype=torch.float32,
    )
    features = torch.cat(
        (
            scalars,
            _image_stats(control),
            _residual_summary(control.rgb_residual_4x4),
            prefix,
            camera,
            fast_stats,
            _action_one_hot(control.previous_action),
            budget,
        )
    )
    if features.shape != (POLICY_FEATURE_DIM,) or not torch.isfinite(features).all():
        raise ValueError("control feature vector is nonfinite or has the wrong schema")
    return features


def learned_policy_work_units(input_dim: int, hidden_dim: int, output_dim: int = 4) -> int:
    """Declared multiply/add proxy for one normalized two-layer MLP inference."""

    if any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in (input_dim, hidden_dim, output_dim)
    ):
        raise ValueError("policy dimensions must be positive integers")
    return int(
        input_dim * hidden_dim
        + hidden_dim * output_dim
        + input_dim
        + hidden_dim
        + output_dim
    )


class FixedPolicy:
    def __init__(self, action: Action):
        self.action = Action(action)

    def choose(self, control: ControlInput, feasible_actions: tuple[Action, ...]):
        return self.action if self.action in feasible_actions else Action.OFF


class ResidualThresholdPolicy:
    """Untrained debugging policy; not the proposed learned dynamic controller."""

    def __init__(self, threshold: float = 0.03):
        if threshold < 0:
            raise ValueError("Residual threshold must be nonnegative")
        self.threshold = threshold

    def choose(self, control: ControlInput, feasible_actions: tuple[Action, ...]):
        action = Action.ALL if control.rgb_mse > self.threshold else Action.OFF
        return action if action in feasible_actions else Action.OFF


class LearnedActionPolicy(nn.Module):
    """Frozen deployment MLP predicting one value per fixed action."""

    def __init__(
        self,
        hidden_dim: int = 32,
        *,
        input_dim: int = POLICY_FEATURE_DIM,
        feature_mean: Tensor | Sequence[float] | None = None,
        feature_scale: Tensor | Sequence[float] | None = None,
        normalization: Mapping[str, object] | None = None,
        seed: int | None = None,
        target_mean: float = 0.0,
        target_scale: float = 1.0,
    ) -> None:
        super().__init__()
        if isinstance(hidden_dim, bool) or not isinstance(hidden_dim, int) or hidden_dim <= 0:
            raise ValueError("hidden_dim must be a positive integer")
        if isinstance(input_dim, bool) or not isinstance(input_dim, int) or input_dim <= 0:
            raise ValueError("input_dim must be a positive integer")
        if normalization is not None:
            if feature_mean is not None or feature_scale is not None:
                raise ValueError("provide normalization or explicit feature statistics, not both")
            feature_mean = normalization.get("mean")
            feature_scale = normalization.get("scale")
        mean = _fixed_vector(feature_mean, input_dim, name="feature_mean")
        scale = _fixed_vector(feature_scale, input_dim, name="feature_scale")
        if feature_scale is None:
            scale.fill_(1.0)
        if (scale <= 0).any():
            raise ValueError("feature_scale must be strictly positive")
        if not isfinite(target_mean) or not isfinite(target_scale) or target_scale <= 0:
            raise ValueError("target normalization must be finite with positive scale")
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.action_order = ACTION_ORDER
        self.feature_schema_version = FEATURE_SCHEMA_VERSION
        self.register_buffer("feature_mean", mean)
        self.register_buffer("feature_scale", scale)
        if seed is None:
            self.model = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, len(ACTION_ORDER)),
            )
        else:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed)
                self.model = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.SiLU(),
                    nn.Linear(hidden_dim, len(ACTION_ORDER)),
                )
        self.seed = seed
        self.target_mean = float(target_mean)
        self.target_scale = float(target_scale)

    @property
    def declared_work_units(self) -> int:
        return learned_policy_work_units(self.input_dim, self.hidden_dim, len(ACTION_ORDER))

    @property
    def normalization(self) -> dict[str, list[float]]:
        return {
            "mean": self.feature_mean.detach().cpu().tolist(),
            "scale": self.feature_scale.detach().cpu().tolist(),
        }

    def forward(self, features: Tensor) -> Tensor:
        if not isinstance(features, Tensor) or features.shape[-1] != self.input_dim:
            raise ValueError(f"features must have trailing dimension {self.input_dim}")
        if not torch.isfinite(features).all():
            raise ValueError("policy features must be finite")
        parameter = next(self.model.parameters())
        values = features.to(device=parameter.device, dtype=parameter.dtype)
        mean = self.feature_mean.to(device=parameter.device, dtype=parameter.dtype)
        scale = self.feature_scale.to(device=parameter.device, dtype=parameter.dtype)
        normalized = (values - mean) / scale
        scores = self.model(normalized)
        if not torch.isfinite(scores).all():
            raise ValueError("policy scores must be finite")
        return scores

    def action_values(self, control: ControlInput) -> Tensor:
        return self(control_to_features(control))

    def action_values_original(self, control: ControlInput) -> Tensor:
        return self.action_values(control) * self.target_scale + self.target_mean

    def choose(self, control: ControlInput, feasible_actions: tuple[Action, ...]):
        feasible = tuple(Action(action) for action in feasible_actions)
        if not feasible:
            raise ValueError("feasible_actions must be nonempty")
        if len(set(feasible)) != len(feasible):
            raise ValueError("feasible_actions must not contain duplicates")
        scores = self.action_values(control).reshape(-1)
        if scores.numel() != len(ACTION_ORDER) or not torch.isfinite(scores).all():
            raise ValueError("policy scores must contain one finite value per action")
        best_action = None
        best_score = None
        for action in ACTION_ORDER:
            if action not in feasible:
                continue
            score = float(scores[ACTION_ORDER.index(action)].item())
            if best_score is None or score > best_score:
                best_action, best_score = action, score
        if best_action is None:
            raise ValueError("feasible_actions contains no known action")
        return best_action


__all__ = [
    "ACTION_ORDER",
    "CAMERA_CHANGE_DIM",
    "FEATURE_NAMES",
    "FEATURE_SCHEMA",
    "FEATURE_SCHEMA_VERSION",
    "FixedPolicy",
    "LearnedActionPolicy",
    "POLICY_FEATURE_DIM",
    "POLICY_FEATURE_NAMES",
    "POLICY_FEATURE_SCHEMA_VERSION",
    "ResidualThresholdPolicy",
    "control_to_features",
    "learned_policy_work_units",
]
