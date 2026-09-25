"""Offline action-value labels for the dynamic scene-state controller.

This module is deliberately training-only.  It receives complete training
episodes and ``TrainingSupervision`` labels, but the online runner never
imports it.  A row is collected at a prefix before the arriving observation is
cached.  Each feasible action is then evaluated from the same cloned old
runner state, with OFF used for the continuation after the first action.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, fields, is_dataclass
from pathlib import Path

import torch
from torch import Tensor

from mcss.data.episodes import OnlineEpisodeSource
from mcss.dynamic.checkpoint import load_dynamic_checkpoint
from mcss.dynamic.policy import (
    ACTION_ORDER,
    FEATURE_SCHEMA_VERSION,
    POLICY_FEATURE_DIM,
    control_to_features,
    learned_policy_work_units,
)
from mcss.dynamic.runner import StreamingRunner
from mcss.dynamic.types import Action, ControlInput, OnlineObservation
from mcss.geometry import transform_cameras
from mcss.measurements import FixedMeasurementRenderer
from mcss.training.supervision import TrainingSupervision
from mcss.types import Cameras

TEACHER_SCHEMA_VERSION = "mcss.dynamic.action_teacher.v1"
ROW_SCHEMA_VERSION = "mcss.dynamic.action_teacher_row.v1"
DEFAULT_HORIZON = 3
DEFAULT_RENDER_SAMPLES = 16
DEFAULT_RAY_CHUNK_SIZE = 16_384
DEFAULT_MAX_UNITS = 1e12
DEFAULT_POLICY_HIDDEN_DIM = 32
DEFAULT_POLICY_WORK_UNITS = learned_policy_work_units(
    POLICY_FEATURE_DIM, DEFAULT_POLICY_HIDDEN_DIM, len(ACTION_ORDER)
)
_RUNTIME_BINDING_KEYS = frozenset(
    {
        "model_content_hash",
        "feature_schema_version",
        "render_protocol",
        "budget_protocol",
        "utility_protocol",
    }
)
_PROJECT = Path(__file__).resolve().parents[3]
_ACTION_VALUES = tuple(Action(action).value for action in ACTION_ORDER)
_DATASET_BINDING_KEYS = frozenset(
    {
        "source_hash",
        "source_hashes",
        "dataset_hash",
        "dataset_hashes",
        "teacher_dataset_hash",
        "teacher_source_hashes",
    }
)


@dataclass(frozen=True)
class _ResumeEpisodeIndex:
    """Compact durable-row index used while resuming one teacher collection."""

    row_fingerprints: Mapping[str, str]
    row_count: int
    duplicate_row_count: int

    @property
    def row_ids(self) -> frozenset[str]:
        return frozenset(self.row_fingerprints)


def _declared_policy_work_units(
    policy: object, *, fallback: int | None = DEFAULT_POLICY_WORK_UNITS
) -> int:
    """Read a policy's declared cost, keeping fixed roll-ins canonical by default."""

    value = getattr(policy, "declared_work_units", fallback)
    if callable(value):
        value = value()
    if type(value) is not int or value <= 0:
        raise ValueError("policy declared_work_units must be a positive integer")
    return value


@dataclass(frozen=True)
class ActionTeacherConfig:
    """Fixed choices that affect every row in one teacher collection."""

    horizon: int = DEFAULT_HORIZON
    continuation_action: Action = Action.OFF
    max_units: float = DEFAULT_MAX_UNITS
    renderer_samples: int = DEFAULT_RENDER_SAMPLES
    ray_chunk_size: int = DEFAULT_RAY_CHUNK_SIZE
    prefix_stride: int = 1
    max_prefixes_per_episode: int | None = None

    def __post_init__(self) -> None:
        if type(self.horizon) is not int or self.horizon < 1:
            raise ValueError("horizon must be a positive integer")
        if Action(self.continuation_action) is not Action.OFF:
            raise ValueError("action teacher continuation must be OFF")
        if not math.isfinite(float(self.max_units)) or self.max_units <= 0:
            raise ValueError("max_units must be finite and positive")
        if type(self.renderer_samples) is not int or self.renderer_samples < 2:
            raise ValueError("renderer_samples must be at least two")
        if type(self.ray_chunk_size) is not int or self.ray_chunk_size < 1:
            raise ValueError("ray_chunk_size must be a positive integer")
        if type(self.prefix_stride) is not int or self.prefix_stride < 1:
            raise ValueError("prefix_stride must be a positive integer")
        if self.max_prefixes_per_episode is not None and (
            type(self.max_prefixes_per_episode) is not int
            or self.max_prefixes_per_episode < 1
        ):
            raise ValueError("max_prefixes_per_episode must be positive when provided")


@dataclass(frozen=True)
class QueryUtility:
    """Shared query mask and utility definition used by all action branches."""

    valid_mask: Tensor
    weights: Tensor
    query_frame_ids: tuple[int, ...]
    query_cameras: Cameras | None = None
    query_depth: Tensor | None = None

    def __post_init__(self) -> None:
        if self.valid_mask.shape != self.weights.shape:
            raise ValueError("query mask and weights must have identical shapes")
        if self.valid_mask.ndim != 5:
            raise ValueError("query mask and weights must be [B,V,1,H,W]")
        if not self.valid_mask.any():
            raise ValueError("training queries contain no valid depth pixels")
        if not torch.isfinite(self.weights).all() or (self.weights < 0).any():
            raise ValueError("query weights must be finite and nonnegative")
        if not (self.valid_mask & (self.weights > 0)).any():
            raise ValueError("training query weights have no valid support")
        if self.query_depth is not None and self.query_depth.shape != self.valid_mask.shape:
            raise ValueError("query depth and mask must have identical shapes")
        if self.query_depth is not None and not isinstance(self.query_cameras, Cameras):
            raise ValueError("query cameras are required when query depth is provided")


class _CapturePolicy:
    """Capture the exact pre-append policy input without changing the choice."""

    def __init__(self, delegate: object) -> None:
        self.delegate = delegate
        self.control: ControlInput | None = None
        self.feasible_actions: tuple[Action, ...] = ()

    @property
    def declared_work_units(self) -> int:
        # Fixed roll-ins retain the canonical charge; learned roll-ins expose
        # their own architecture cost so the budget matches deployment.
        return _declared_policy_work_units(self.delegate)

    def choose(self, control: ControlInput, feasible_actions: tuple[Action, ...]) -> Action:
        self.control = control
        self.feasible_actions = tuple(Action(action) for action in feasible_actions)
        return Action(self.delegate.choose(control, self.feasible_actions))


class _StrictActionPolicy:
    def __init__(self, action: Action) -> None:
        self.action = Action(action)

    @property
    def declared_work_units(self) -> int:
        return DEFAULT_POLICY_WORK_UNITS

    def choose(self, control: ControlInput, feasible_actions: tuple[Action, ...]) -> Action:
        feasible = tuple(Action(item) for item in feasible_actions)
        if self.action not in feasible:
            raise ValueError(f"requested action {self.action.value} is infeasible")
        return self.action


class _SeededRollinPolicy:
    """Label-independent, deterministic random roll-in used for initial collection."""

    def __init__(self, seed: int, *, fixed_action: Action | None = Action.OFF) -> None:
        if type(seed) is not int:
            raise ValueError("roll-in seed must be an integer")
        self._generator = torch.Generator(device="cpu")
        self._generator.manual_seed(seed)
        self.fixed_action = None if fixed_action is None else Action(fixed_action)

    @property
    def declared_work_units(self) -> int:
        return DEFAULT_POLICY_WORK_UNITS

    def choose(self, control: ControlInput, feasible_actions: tuple[Action, ...]) -> Action:
        feasible = tuple(Action(item) for item in feasible_actions)
        if not feasible:
            raise ValueError("roll-in received no feasible actions")
        if self.fixed_action is not None:
            return self.fixed_action if self.fixed_action in feasible else Action.OFF
        index = int(torch.randint(len(feasible), (), generator=self._generator).item())
        return feasible[index]


def collect_episode_targets(
    runner: StreamingRunner,
    warmup: Sequence[OnlineObservation],
    stream: Sequence[OnlineObservation],
    supervision: TrainingSupervision,
    *,
    episode_id: str,
    scene_id: str,
    split_id: str = "train",
    query_vault_id: str,
    horizon: int = DEFAULT_HORIZON,
    continuation_action: Action = Action.OFF,
    prefix_stride: int = 1,
    max_prefixes_per_episode: int | None = None,
    selected_prefix_steps: Sequence[int] | None = None,
    rollin_policy: object | None = None,
    rollin_seed: int = 0,
    provenance: Mapping[str, object] | None = None,
    collection_id: str | None = None,
) -> list[dict[str, object]]:
    """Collect all-action rows for one train episode.

    ``runner`` is used as a fresh slow-checkpoint instance.  It is reset and
    consumed by the label-independent roll-in.  At each selected prefix a
    deep copy is made before the current observation is appended.  The four
    action branches are processed one at a time and each gets a forked fast
    identity, cache and scene tensors.
    """

    if collection_id is not None and (not isinstance(collection_id, str) or not collection_id):
        raise ValueError("collection_id must be a nonempty string when provided")

    config = ActionTeacherConfig(
        horizon=horizon,
        continuation_action=continuation_action,
        max_units=runner.max_units,
        renderer_samples=runner.renderer.n_samples,
        ray_chunk_size=runner.renderer.ray_chunk_size,
        prefix_stride=prefix_stride,
        max_prefixes_per_episode=max_prefixes_per_episode,
    )
    _validate_episode_identity(
        warmup,
        stream,
        supervision,
        episode_id=episode_id,
        scene_id=scene_id,
        split_id=split_id,
        query_vault_id=query_vault_id,
    )
    utility = _build_query_utility(supervision)
    prefixes = _select_prefix_steps(
        len(stream),
        config.prefix_stride,
        config.max_prefixes_per_episode,
        selected=selected_prefix_steps,
    )
    selected = set(prefixes)
    rollin = rollin_policy or _SeededRollinPolicy(rollin_seed, fixed_action=Action.OFF)
    capture = _CapturePolicy(rollin)
    runner.policy = capture
    runner.reset(
        tuple(warmup),
        episode_id=episode_id,
        scene_id=scene_id,
        split_id=split_id,
        query_vault_id=query_vault_id,
        stream_steps=len(stream),
        query_count=len(utility.query_frame_ids),
    )

    rows: list[dict[str, object]] = []
    for prefix_step, observation in enumerate(stream):
        snapshot = copy.deepcopy(runner) if prefix_step in selected else None
        runner.step(observation)
        if snapshot is None:
            continue
        if capture.control is None:
            raise RuntimeError("roll-in policy did not expose a ControlInput")
        if capture.control.observed_count != snapshot.cache.revision:
            raise RuntimeError("captured control does not describe the recorded prefix")
        rows.append(
            _collect_row(
                snapshot,
                observation,
                stream,
                prefix_step,
                capture.control,
                capture.feasible_actions,
                utility,
                config,
                provenance or {},
                collection_id=collection_id,
            )
        )
        del snapshot
    if runner.remaining_steps != 0:
        raise RuntimeError("teacher roll-in did not consume the complete stream")
    return rows


def _collect_row(
    snapshot: StreamingRunner,
    current_observation: OnlineObservation,
    stream: Sequence[OnlineObservation],
    prefix_step: int,
    control: ControlInput,
    feasible_actions: tuple[Action, ...],
    utility: QueryUtility,
    config: ActionTeacherConfig,
    provenance: Mapping[str, object],
    collection_id: str | None = None,
) -> dict[str, object]:
    feasible = tuple(Action(action) for action in feasible_actions)
    if not feasible or len(set(feasible)) != len(feasible):
        raise ValueError("runner returned an invalid feasible action set")
    horizon_used = min(config.horizon, len(stream) - prefix_step)
    losses: list[float | None] = []
    costs: list[float | None] = []
    for action in ACTION_ORDER:
        if action not in feasible:
            losses.append(None)
            costs.append(None)
            continue
        branch = copy.deepcopy(snapshot)
        # deepcopy preserves the source branch nonce; fork it before any write.
        branch.fast = branch.fast.fork()
        branch.policy = _StrictActionPolicy(action)
        starting_units = branch.budget.used_units
        first_record = branch.step(current_observation)
        if Action(first_record["action"]) is not action:
            raise RuntimeError("teacher branch did not execute its requested first action")
        branch_losses = [
            _query_absrel(branch, utility, branch.renderer.n_samples)
        ]
        for offset in range(1, horizon_used):
            # Continuation is label-independent, but it must reserve the same
            # learned-policy inference charge as deployment.
            branch.policy = _StrictActionPolicy(Action.OFF)
            branch.step(stream[prefix_step + offset])
            branch_losses.append(_query_absrel(branch, utility, branch.renderer.n_samples))
        losses.append(float(sum(branch_losses) / len(branch_losses)))
        costs.append(float(branch.budget.used_units - starting_units))
        del branch

    feasible_losses = [loss for loss in losses if loss is not None]
    if not feasible_losses:
        raise RuntimeError("at least one action must be feasible")
    mean_loss = sum(feasible_losses) / len(feasible_losses)
    advantages = [None if loss is None else float(mean_loss - loss) for loss in losses]
    row_provenance = _json_value(dict(provenance))
    return {
        "schema_version": ROW_SCHEMA_VERSION,
        "row_id": (
            f"{collection_id}:{snapshot.episode_id}:prefix:{prefix_step}"
            if collection_id is not None
            else f"{snapshot.episode_id}:prefix:{prefix_step}"
        ),
        "episode_id": snapshot.episode_id,
        "scene_id": snapshot.scene_id,
        "split_id": snapshot.split_id,
        "prefix_step": prefix_step,
        "prefix_frame_id": int(current_observation.frame_id),
        "control_input": _serialize_control(control),
        "features": [float(value) for value in control_to_features(control).tolist()],
        "actions": list(_ACTION_VALUES),
        "feasible_mask": [action in feasible for action in ACTION_ORDER],
        "losses": losses,
        "advantages": advantages,
        "costs": costs,
        "horizon_used": horizon_used,
        "provenance": row_provenance,
    }


@torch.no_grad()
def _query_absrel(
    branch: StreamingRunner, utility: QueryUtility, renderer_samples: int
) -> float:
    query_cameras = utility.query_cameras
    target = utility.query_depth
    if query_cameras is None or target is None:
        # Compatibility path for callers that used the old explicit context
        # helpers.  The public collector always embeds these tensors in the
        # QueryUtility, so labels never depend on mutable module state.
        query_cameras = _SUPERVISION_CAMERAS.get(branch.episode_id)
        target = _SUPERVISION_DEPTH.get(branch.episode_id)
    if query_cameras is None or target is None:
        raise RuntimeError("query supervision is not installed for this teacher branch")
    local_cameras = transform_cameras(
        # Query cameras are stored in the raw scene frame; the branch state is local to its
        # warmup anchor.  The same transformation is used for every action branch.
        query_cameras,
        torch.linalg.inv(branch.anchor_c2w),
    )
    prediction = branch.renderer(
        branch.scene_state, local_cameras, measurements=("depth",)
    )["depth"]
    if prediction.shape != target.shape or target.shape != utility.valid_mask.shape:
        raise ValueError("query prediction and common supervision mask shape mismatch")
    valid = utility.valid_mask
    safe_target = torch.where(valid, target, torch.ones_like(target))
    safe_prediction = torch.where(valid, prediction, safe_target)
    if not torch.isfinite(safe_prediction[valid]).all():
        raise RuntimeError("teacher produced a nonfinite depth prediction")
    numerator = (
        (safe_prediction - safe_target).abs() / safe_target.abs().clamp_min(1e-8)
    ) * utility.weights
    denominator = (utility.valid_mask.to(numerator.dtype) * utility.weights).sum()
    if float(denominator) <= 0:
        raise ValueError("query utility has no valid weighted support")
    value = float((numerator * utility.valid_mask).sum().item() / denominator.item())
    if not math.isfinite(value):
        raise RuntimeError("teacher produced a nonfinite depth AbsRel")
    # Keep the argument in the public helper signature to make the charged renderer protocol
    # explicit at call sites; the renderer itself owns the sample count.
    _ = renderer_samples
    return value


# Query tensors are installed only for the duration of one episode.  They avoid passing labels
# through the runtime runner and keep the branch helper's signature small.  The collector is
# single-threaded and clears them immediately after each episode.
_SUPERVISION_CAMERAS: dict[str, object] = {}
_SUPERVISION_DEPTH: dict[str, Tensor] = {}


def _build_query_utility(supervision: TrainingSupervision) -> QueryUtility:
    target = supervision.query_depth
    valid = torch.isfinite(target) & (target > 0)
    if target.ndim != 5 or target.shape[0] != 1 or target.shape[2] != 1:
        raise ValueError("TrainingSupervision query_depth must be [1,V,1,H,W]")
    per_query_support = valid.reshape(target.shape[0], target.shape[1], -1).any(dim=-1)
    if not bool(per_query_support.all()):
        invalid = [index for index, value in enumerate(per_query_support[0].tolist()) if not value]
        raise ValueError(f"training query views have no valid depth support: {invalid}")
    weights = torch.ones_like(target)
    return QueryUtility(
        valid,
        weights,
        tuple(supervision.frame_ids),
        query_cameras=supervision.query_cameras,
        query_depth=target,
    )


def _validate_episode_identity(
    warmup: Sequence[OnlineObservation],
    stream: Sequence[OnlineObservation],
    supervision: TrainingSupervision,
    *,
    episode_id: str,
    scene_id: str,
    split_id: str,
    query_vault_id: str,
) -> None:
    if split_id != "train":
        raise ValueError("action teacher only accepts the train split")
    if not episode_id or not scene_id or not query_vault_id:
        raise ValueError("episode identity fields must be nonempty")
    if supervision.episode_id != episode_id or supervision.scene_id != scene_id:
        raise ValueError("training supervision identity does not match the episode")
    if supervision.query_vault_id != query_vault_id:
        raise ValueError("training supervision query vault does not match the episode")
    observations = tuple(warmup) + tuple(stream)
    if not observations:
        raise ValueError("episode must contain warmup and stream observations")
    if any(item.scene_id != scene_id for item in observations):
        raise ValueError("episode observations must belong to one scene")
    frame_ids = [item.frame_id for item in observations]
    if frame_ids != sorted(frame_ids) or len(frame_ids) != len(set(frame_ids)):
        raise ValueError("episode observation frame IDs must be unique and ordered")
    if set(frame_ids) & set(supervision.frame_ids):
        raise ValueError("training query frame IDs must be disjoint from observations")


def _select_prefix_steps(
    length: int,
    stride: int,
    maximum: int | None,
    *,
    selected: Sequence[int] | None = None,
) -> tuple[int, ...]:
    if length < 1:
        raise ValueError("stream must contain at least one observation")
    if type(stride) is not int or stride < 1:
        raise ValueError("prefix stride must be a positive integer")
    if selected is not None:
        if isinstance(selected, (str, bytes)):
            raise ValueError("selected prefix steps must be a sequence of integers")
        selected_steps = list(selected)
        if not selected_steps:
            raise ValueError("selected prefix steps must be nonempty")
        if any(type(step) is not int or step < 0 or step >= length for step in selected_steps):
            raise ValueError("selected prefix steps must be unique steps within the stream")
        if selected_steps != sorted(set(selected_steps)):
            raise ValueError("selected prefix steps must be sorted and unique")
        return tuple(selected_steps)
    selected_steps = list(range(0, length, stride))
    if maximum is not None:
        if type(maximum) is not int or maximum < 1:
            raise ValueError("maximum prefixes must be positive")
        selected_steps = selected_steps[:maximum]
    return tuple(selected_steps)


def _serialize_control(control: ControlInput) -> dict[str, object]:
    """Convert every ControlInput field, including optional tensors, to JSON primitives."""

    def convert(value: object) -> object:
        if isinstance(value, Action):
            return value.value
        if isinstance(value, Tensor):
            if not torch.isfinite(value).all():
                raise ValueError("control input contains a nonfinite tensor")
            return [convert(item) for item in value.detach().cpu().tolist()]
        if is_dataclass(value):
            return {field.name: convert(getattr(value, field.name)) for field in fields(value)}
        if isinstance(value, Mapping):
            return {str(key): convert(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [convert(item) for item in value]
        if isinstance(value, bool) or value is None or isinstance(value, str):
            return value
        if isinstance(value, (int, float)):
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError("control input contains a nonfinite scalar")
            return value
        raise TypeError(f"control input field is not JSON-compatible: {type(value).__name__}")

    return convert(control)


def _json_value(value: object) -> object:
    """Normalize a nested provenance object before strict JSON serialization."""

    if isinstance(value, Tensor):
        return value.detach().cpu().tolist()
    if is_dataclass(value):
        return _json_value(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("provenance contains a nonfinite value")
    return value


def prepare_episode_context(
    supervision: TrainingSupervision,
    *,
    runner: StreamingRunner,
) -> None:
    """Install one episode's labels for the private branch scorer.

    The function exists for callers that use ``collect_episode_targets`` with a custom
    branch scorer.  It is intentionally explicit and must be paired with
    :func:`clear_episode_context`.
    """

    _SUPERVISION_CAMERAS[runner.episode_id] = supervision.query_cameras
    _SUPERVISION_DEPTH[runner.episode_id] = supervision.query_depth


def clear_episode_context(episode_id: str) -> None:
    _SUPERVISION_CAMERAS.pop(episode_id, None)
    _SUPERVISION_DEPTH.pop(episode_id, None)


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def implementation_hash() -> str:
    """Hash the teacher implementation and its public CLI companion."""

    paths = (
        _PROJECT / "src/mcss/training/action_teacher.py",
        _PROJECT / "scripts/collect_action_targets.py",
    )
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def make_binding(
    *,
    model_content_hash: str,
    renderer_samples: int,
    max_units: float,
    horizon: int,
    continuation_action: Action = Action.OFF,
    ray_chunk_size: int = DEFAULT_RAY_CHUNK_SIZE,
) -> dict[str, object]:
    """Return the canonical binding shared by teacher rows and policy checkpoints."""

    if not isinstance(model_content_hash, str) or not model_content_hash:
        raise ValueError("model_content_hash must be a nonempty string")
    config = ActionTeacherConfig(
        horizon=horizon,
        continuation_action=continuation_action,
        max_units=max_units,
        renderer_samples=renderer_samples,
        ray_chunk_size=ray_chunk_size,
    )
    render_protocol = {
        "renderer": "FixedMeasurementRenderer",
        "renderer_samples": config.renderer_samples,
        "ray_chunk_size": config.ray_chunk_size,
    }
    budget_protocol = {
        "max_units": config.max_units,
        # The policy architecture is fixed by the runtime feature contract.
        # This charge is reserved during teacher roll-in as well as deployment.
        "policy_work_units": DEFAULT_POLICY_WORK_UNITS,
    }
    utility_protocol = {
        "horizon": config.horizon,
        "horizon_window": "current_postwrite_plus_off_continuation",
        "continuation_action": Action.OFF.value,
        "utility": "mean_masked_depth_abs_rel",
        "query_mask": "finite_positive_depth_common_all_views",
        "query_weights": "uniform_valid_pixels",
    }
    return {
        "model_content_hash": str(model_content_hash),
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "render_protocol": render_protocol,
        "budget_protocol": budget_protocol,
        "utility_protocol": utility_protocol,
        "teacher_dataset_hash": "",
        "teacher_source_hashes": {},
    }


def _runtime_binding(binding: Mapping[str, object]) -> dict[str, object]:
    """Select only checkpoint/runtime fields before teacher hashes are known."""

    return {key: _json_value(binding[key]) for key in _RUNTIME_BINDING_KEYS if key in binding}


def _source_provenance(
    index_path: str | Path,
    entries: Sequence[Mapping[str, object]],
    *,
    checkpoint_path: str | Path,
    binding: Mapping[str, object],
) -> dict[str, object]:
    manifests: list[dict[str, str]] = []
    episode_manifests: dict[str, list[dict[str, str]]] = {}
    for entry in entries:
        episode_records: list[dict[str, str]] = []
        for key in ("online_manifest", "query_manifest"):
            path = Path(str(entry[key])).resolve()
            record = {"path": str(path), "sha256": _sha256_file(path)}
            manifests.append(record)
            episode_records.append(record)
        episode_manifests[str(entry["episode_id"])] = episode_records
    return {
        "teacher_schema_version": TEACHER_SCHEMA_VERSION,
        "binding": _json_value(dict(binding)),
        **_runtime_binding(binding),
        "teacher_dataset_hash": binding["teacher_dataset_hash"],
        "teacher_source_hashes": binding["teacher_source_hashes"],
        "source_data": {
            "index_path": str(Path(index_path).resolve()),
            "index_sha256": _sha256_file(index_path),
            "manifest_sha256": manifests,
            "episode_manifest_sha256": episode_manifests,
            "split": "train",
        },
        "implementation_sha256": implementation_hash(),
        "carrier_write_checkpoint_sha256": _sha256_file(checkpoint_path),
    }


def _episode_provenance(
    provenance: Mapping[str, object], entry: Mapping[str, object]
) -> dict[str, object]:
    """Keep row provenance local to one episode while the sidecar owns the full manifest list."""

    source_data = provenance.get("source_data")
    if not isinstance(source_data, Mapping):
        raise ValueError("teacher provenance is missing source_data")
    manifest_records = source_data.get("manifest_sha256")
    if not isinstance(manifest_records, Sequence) or isinstance(
        manifest_records, (str, bytes)
    ):
        raise ValueError("teacher provenance manifest_sha256 must be a sequence")
    by_path: dict[str, str] = {}
    for record in manifest_records:
        if not isinstance(record, Mapping):
            raise ValueError("teacher provenance contains an invalid manifest record")
        path = record.get("path")
        digest = record.get("sha256")
        if not isinstance(path, str) or not isinstance(digest, str):
            raise ValueError("teacher provenance manifest records require path and sha256")
        by_path[path] = digest

    compact_manifests: list[dict[str, str]] = []
    for key in ("online_manifest", "query_manifest"):
        path = str(Path(str(entry[key])).resolve())
        digest = by_path.get(path)
        if digest is None:
            raise ValueError(f"teacher provenance has no hash for {key}: {path}")
        compact_manifests.append({"path": path, "sha256": digest})

    compact_source_data = {
        str(key): _json_value(value)
        for key, value in source_data.items()
        if key not in {"manifest_sha256", "episode_manifest_sha256"}
    }
    compact_source_data["manifest_sha256"] = compact_manifests
    compact_source_data["manifest_scope"] = "episode"
    result = {
        str(key): _json_value(value)
        for key, value in provenance.items()
        if key != "source_data"
    }
    result["source_data"] = compact_source_data
    result["episode"] = {
        "episode_id": str(entry["episode_id"]),
        "scene_id": str(entry["scene_id"]),
        "split_id": str(entry["split_id"]),
        "query_vault_id": str(entry["query_vault_id"]),
    }
    return result


def _derive_episode_rollin_seed(seed: int, episode_id: str) -> int:
    """Derive a stable independent torch seed for one random-roll-in episode."""

    digest = hashlib.sha256(
        f"mcss.random_rollin.v1:{seed}:{episode_id}".encode()
    ).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def _episode_rollin(
    rollin: object, *, episode_id: str, seed: int
) -> tuple[object, int | None]:
    if isinstance(rollin, _SeededRollinPolicy) and rollin.fixed_action is None:
        episode_seed = _derive_episode_rollin_seed(seed, episode_id)
        return _SeededRollinPolicy(episode_seed, fixed_action=None), episode_seed
    return copy.deepcopy(rollin), None


def _metadata_digest(value: Mapping[str, object]) -> str:
    encoded = json.dumps(_json_value(value), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def collect_action_targets(
    index: Sequence[Mapping[str, object]],
    checkpoint: str | Path,
    output: str | Path,
    *,
    index_path: str | Path | None = None,
    device: str | torch.device = "cpu",
    horizon: int = DEFAULT_HORIZON,
    max_episodes: int | None = None,
    prefix_stride: int = 1,
    max_prefixes_per_episode: int | None = None,
    rollin_policy: str | Path | None = None,
    rollin_seed: int = 0,
    max_units: float = DEFAULT_MAX_UNITS,
    renderer_samples: int = DEFAULT_RENDER_SAMPLES,
    ray_chunk_size: int = DEFAULT_RAY_CHUNK_SIZE,
    selected_prefixes: Mapping[str, Sequence[int]] | None = None,
    resume: bool = False,
) -> dict[str, object]:
    """Collect resumable train-only action targets from a frozen B checkpoint."""

    entries = [dict(entry) for entry in index]
    if max_episodes is not None:
        if type(max_episodes) is not int or max_episodes < 1:
            raise ValueError("max_episodes must be positive")
        entries = entries[:max_episodes]
    _validate_index_entries(entries)
    _validate_selected_prefixes(selected_prefixes, entries)
    checkpoint_path = Path(checkpoint).resolve()
    carrier, write_rule, checkpoint_meta = load_dynamic_checkpoint(checkpoint_path, device=device)
    _validate_frozen_b_checkpoint(checkpoint_meta)
    base_runner = StreamingRunner(
        carrier,
        write_rule,
        FixedMeasurementRenderer(n_samples=renderer_samples, ray_chunk_size=ray_chunk_size),
        _StrictActionPolicy(Action.OFF),
        max_units=max_units,
    )
    binding = make_binding(
        model_content_hash=base_runner.checkpoint_hash,
        renderer_samples=renderer_samples,
        max_units=max_units,
        horizon=horizon,
        ray_chunk_size=ray_chunk_size,
    )
    rollin, rollin_meta = _load_rollin(
        rollin_policy,
        device=device,
        expected_binding=binding,
        seed=rollin_seed,
    )
    source_index = index_path or _index_digest_path(entries)
    provenance = _source_provenance(
        source_index,
        entries,
        checkpoint_path=checkpoint_path,
        binding=binding,
    )
    provenance["rollin"] = rollin_meta
    paths = _prepare_output(output, resume=resume)
    metadata = {
        "schema_version": TEACHER_SCHEMA_VERSION,
        "row_schema_version": ROW_SCHEMA_VERSION,
        "binding": binding,
        "provenance": provenance,
        "actions": list(_ACTION_VALUES),
        "episodes": [str(entry["episode_id"]) for entry in entries],
        "horizon": horizon,
        "prefix_stride": prefix_stride,
        "max_prefixes_per_episode": max_prefixes_per_episode,
        "ray_chunk_size": ray_chunk_size,
        "selected_prefixes": _json_value(selected_prefixes),
        "rollin": rollin_meta,
        "row_provenance_mode": "episode_compact_v1",
    }
    metadata_digest = _metadata_digest(metadata)
    lock_path = paths["lock"]
    try:
        lock_path.open("x", encoding="ascii").close()
    except FileExistsError as error:
        raise FileExistsError(
            f"action-target collection is already running: {lock_path}"
        ) from error
    try:
        _initialize_metadata(paths, metadata, metadata_digest, resume=resume)
        progress = _load_progress(
            paths["progress"], metadata_digest, paths["jsonl"], resume=resume
        )
        completed = set(progress["completed_episode_ids"])
        existing_index = _read_rows(paths["jsonl"], metadata=metadata)
        expected_ids = {str(entry["episode_id"]) for entry in entries}
        if not set(existing_index).issubset(expected_ids):
            raise ValueError("output JSONL contains an episode outside the selected index")
        if not completed.issubset(expected_ids):
            raise ValueError("progress claims an episode outside the selected index")
        for episode_id in tuple(completed):
            if episode_id not in existing_index:
                raise ValueError("progress claims a completed episode with no durable rows")
        for entry in entries:
            episode_id = str(entry["episode_id"])
            if episode_id in completed:
                continue
            rows = _collect_index_episode(
                entry,
                base_runner,
                rollin,
                device=device,
                horizon=horizon,
                prefix_stride=prefix_stride,
                max_prefixes_per_episode=max_prefixes_per_episode,
                selected_prefixes=(
                    None
                    if selected_prefixes is None
                    else selected_prefixes.get(episode_id)
                ),
                rollin_seed=rollin_seed,
                provenance=_episode_provenance(provenance, entry),
                renderer_ray_chunk_size=ray_chunk_size,
                collection_id=metadata_digest,
            )
            current_index = _index_from_rows(rows)
            durable_index = existing_index.get(episode_id)
            if durable_index is not None:
                if (
                    durable_index.row_ids != current_index.row_ids
                    or durable_index.row_count != current_index.row_count
                    or durable_index.duplicate_row_count != current_index.duplicate_row_count
                ):
                    raise ValueError("durable rows partially overlap an unfinished episode")
                if dict(durable_index.row_fingerprints) != dict(current_index.row_fingerprints):
                    raise ValueError(f"conflicting duplicate teacher row_id: {episode_id}")
                completed.add(episode_id)
                progress = {
                    "schema_version": TEACHER_SCHEMA_VERSION,
                    "metadata_sha256": metadata_digest,
                    "completed_episode_ids": [
                        str(item["episode_id"])
                        for item in entries
                        if str(item["episode_id"]) in completed
                    ],
                    "row_count": sum(value.row_count for value in existing_index.values()),
                }
                _atomic_json(paths["progress"], progress)
                continue
            with paths["jsonl"].open("a", encoding="utf-8", newline="\n") as handle:
                for row in rows:
                    handle.write(
                        json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False)
                        + "\n"
                    )
                handle.flush()
                os.fsync(handle.fileno())
            existing_index[episode_id] = current_index
            completed.add(episode_id)
            progress = {
                "schema_version": TEACHER_SCHEMA_VERSION,
                "metadata_sha256": metadata_digest,
                "completed_episode_ids": [
                    str(item["episode_id"])
                    for item in entries
                    if str(item["episode_id"]) in completed
                ],
                "row_count": sum(value.row_count for value in existing_index.values()),
            }
            _atomic_json(paths["progress"], progress)
        progress["status"] = "complete"
        _atomic_json(paths["progress"], progress)
    finally:
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass
    return {
        "status": "complete",
        "rows": progress["row_count"],
        "output": str(paths["jsonl"]),
        "metadata": str(paths["metadata"]),
    }


def _collect_index_episode(
    entry: Mapping[str, object],
    base_runner: StreamingRunner,
    rollin: object,
    *,
    device: str | torch.device,
    horizon: int,
    prefix_stride: int,
    max_prefixes_per_episode: int | None,
    selected_prefixes: Sequence[int] | None,
    rollin_seed: int,
    provenance: Mapping[str, object],
    renderer_ray_chunk_size: int,
    collection_id: str | None = None,
) -> list[dict[str, object]]:
    source = OnlineEpisodeSource.from_file(str(entry["online_manifest"]), device=device)
    if source.split_id != "train":
        raise ValueError("action teacher rejects a non-train online manifest")
    warmup = source.warmup()
    stream = tuple(source.next_observation() for _ in range(source.remaining_steps))
    context_ids = tuple(item.frame_id for item in (*warmup, *stream))
    supervision = TrainingSupervision.from_query_file(
        str(entry["query_manifest"]),
        episode_id=source.episode_id,
        scene_id=source.scene_id,
        query_vault_id=source.query_vault_id,
        context_frame_ids=context_ids,
        device=device,
    )
    _SUPERVISION_CAMERAS[source.episode_id] = supervision.query_cameras
    _SUPERVISION_DEPTH[source.episode_id] = supervision.query_depth
    try:
        episode_rollin, episode_seed = _episode_rollin(
            rollin, episode_id=source.episode_id, seed=rollin_seed
        )
        episode_provenance = dict(provenance)
        if episode_seed is not None:
            rollin_provenance = episode_provenance.get("rollin", {})
            if not isinstance(rollin_provenance, Mapping):
                raise ValueError("teacher provenance rollin metadata must be an object")
            episode_provenance["rollin"] = {
                **dict(rollin_provenance),
                "episode_seed": episode_seed,
            }
        episode_runner = StreamingRunner(
            copy.deepcopy(base_runner.carrier),
            copy.deepcopy(base_runner.write_rule),
            FixedMeasurementRenderer(
                n_samples=base_runner.renderer.n_samples,
                ray_chunk_size=renderer_ray_chunk_size,
            ),
            copy.deepcopy(episode_rollin),
            max_units=base_runner.max_units,
        )
        return collect_episode_targets(
            episode_runner,
            warmup,
            stream,
            supervision,
            episode_id=source.episode_id,
            scene_id=source.scene_id,
            split_id=source.split_id,
            query_vault_id=source.query_vault_id,
            horizon=horizon,
            prefix_stride=prefix_stride,
            max_prefixes_per_episode=max_prefixes_per_episode,
            selected_prefix_steps=selected_prefixes,
            rollin_policy=copy.deepcopy(episode_rollin),
            rollin_seed=rollin_seed,
            provenance=episode_provenance,
            collection_id=collection_id,
        )
    finally:
        clear_episode_context(source.episode_id)


def _validate_index_entries(entries: Sequence[Mapping[str, object]]) -> None:
    if not entries:
        raise ValueError("action teacher requires at least one episode")
    seen: set[str] = set()
    for entry in entries:
        required = {
            "episode_id",
            "scene_id",
            "split_id",
            "query_vault_id",
            "online_manifest",
            "query_manifest",
        }
        if set(entry) < required:
            raise ValueError(
                f"episode index entry is missing fields: {sorted(required - set(entry))}"
            )
        if entry["split_id"] != "train":
            raise ValueError("action teacher accepts train episodes only")
        episode_id = entry["episode_id"]
        if not isinstance(episode_id, str) or not episode_id or episode_id in seen:
            raise ValueError("episode index has invalid or duplicate episode_id")
        seen.add(episode_id)


def _validate_selected_prefixes(
    selected: Mapping[str, Sequence[int]] | None,
    entries: Sequence[Mapping[str, object]],
) -> None:
    if selected is None:
        return
    if not isinstance(selected, Mapping):
        raise ValueError("selected_prefixes must be a mapping from episode IDs to step lists")
    expected = {str(entry["episode_id"]) for entry in entries}
    unknown = set(selected) - expected
    if unknown:
        raise ValueError(f"selected_prefixes contains unknown episode IDs: {sorted(unknown)}")
    for episode_id, steps in selected.items():
        if isinstance(steps, (str, bytes)):
            raise ValueError(f"selected_prefixes[{episode_id!r}] must be a sequence of integers")
        if not isinstance(steps, Sequence):
            raise ValueError(f"selected_prefixes[{episode_id!r}] must be a sequence of integers")


def _validate_frozen_b_checkpoint(metadata: Mapping[str, object]) -> None:
    phase = str(metadata.get("phase", "")).lower()
    if phase not in {"b", "phase_b", "stage_b"} and not re.search(
        r"(?:^|[_-])(phase|stage)[_-]b(?:$|[_-])", phase
    ):
        raise ValueError("action teacher requires a frozen stage-B carrier/write checkpoint")


def _load_rollin(
    value: str | Path | None,
    *,
    device: str | torch.device,
    expected_binding: Mapping[str, object],
    seed: int,
) -> tuple[object, dict[str, object]]:
    if value is None or str(value).strip().upper() == "OFF":
        return _SeededRollinPolicy(seed, fixed_action=Action.OFF), {
            "kind": "fixed",
            "action": "OFF",
        }
    if str(value).strip().upper() == "RANDOM":
        return _SeededRollinPolicy(seed, fixed_action=None), {
            "kind": "seeded_random",
            "seed": seed,
            "seed_derivation": (
                "sha256(mcss.random_rollin.v1:seed:episode_id)[0:8] masked_to_63_bits"
            ),
        }
    path = Path(value).resolve()
    from mcss.dynamic.policy_checkpoint import load_policy_checkpoint

    policy, metadata = load_policy_checkpoint(
        path,
        expected_binding=_runtime_binding(expected_binding),
        device=device,
    )
    _validate_rollin_policy_work_units(policy, expected_binding)
    return policy, {
        "kind": "learned",
        "path": str(path),
        "sha256": _sha256_file(path),
        "metadata": _json_value(metadata),
    }


def _validate_rollin_policy_work_units(
    policy: object, expected_binding: Mapping[str, object]
) -> None:
    budget = expected_binding.get("budget_protocol")
    if not isinstance(budget, Mapping):
        raise ValueError("roll-in binding is missing budget_protocol")
    expected_units = budget.get("policy_work_units")
    if type(expected_units) is not int or expected_units <= 0:
        raise ValueError("roll-in binding has invalid policy_work_units")
    actual_units = _declared_policy_work_units(policy, fallback=None)
    if actual_units != expected_units:
        raise ValueError(
            "roll-in policy declared_work_units do not match the canonical binding"
        )


def _index_digest_path(entries: Sequence[Mapping[str, object]]) -> Path:
    """Create no file: use a deterministic pseudo path only for API-only provenance."""

    raise ValueError("index_path is required when collecting action targets")


def _prepare_output(output: str | Path, *, resume: bool) -> dict[str, Path]:
    target = Path(output)
    if target.suffix.lower() == ".jsonl":
        directory = target.parent
        jsonl = target
        metadata = target.with_name(f"{target.stem}.metadata.json")
        progress = target.with_name(f"{target.stem}.progress.json")
        lock = target.with_name(f"{target.stem}.lock")
    else:
        directory = target
        jsonl = target / "action_targets.jsonl"
        metadata = target / "metadata.json"
        progress = target / "progress.json"
        lock = target / ".collect.lock"
    directory.mkdir(parents=True, exist_ok=True)
    if not resume and any(path.exists() for path in (jsonl, metadata, progress)):
        raise FileExistsError("action-target output already exists; pass --resume")
    if not jsonl.exists():
        jsonl.open("x", encoding="utf-8").close()
    return {
        "directory": directory,
        "jsonl": jsonl,
        "metadata": metadata,
        "progress": progress,
        "lock": lock,
    }


def _initialize_metadata(
    paths: Mapping[str, Path], metadata: Mapping[str, object], digest: str, *, resume: bool
) -> None:
    if paths["metadata"].exists():
        existing = json.loads(paths["metadata"].read_text(encoding="utf-8"))
        if _metadata_digest(existing) != digest:
            raise ValueError("existing teacher metadata does not match this collection")
        return
    _atomic_json(paths["metadata"], metadata)


def _load_progress(
    path: Path, metadata_digest: str, jsonl_path: Path, *, resume: bool
) -> dict[str, object]:
    if not path.exists():
        if resume and jsonl_path.stat().st_size:
            raise ValueError("resume requires an atomic progress artifact")
        return {
            "schema_version": TEACHER_SCHEMA_VERSION,
            "metadata_sha256": metadata_digest,
            "completed_episode_ids": [],
            "row_count": 0,
        }
    value = json.loads(path.read_text(encoding="utf-8"))
    if (
        value.get("schema_version") != TEACHER_SCHEMA_VERSION
        or value.get("metadata_sha256") != metadata_digest
    ):
        raise ValueError("teacher progress does not match metadata")
    completed = value.get("completed_episode_ids")
    if not isinstance(completed, list) or not all(isinstance(item, str) for item in completed):
        raise ValueError("teacher progress completed_episode_ids is invalid")
    return dict(value)


def _row_fingerprint(row: Mapping[str, object]) -> str:
    fingerprint_value = dict(row)
    provenance = fingerprint_value.get("provenance")
    if isinstance(provenance, Mapping):
        fingerprint_provenance = dict(provenance)
        for key in _DATASET_BINDING_KEYS:
            fingerprint_provenance.pop(key, None)
        fingerprint_value["provenance"] = fingerprint_provenance
    encoded = json.dumps(
        _json_value(fingerprint_value), sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _index_from_rows(rows: Sequence[Mapping[str, object]]) -> _ResumeEpisodeIndex:
    fingerprints: dict[str, str] = {}
    row_count = 0
    duplicate_count = 0
    episode_id: str | None = None
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("teacher row must be an object")
        row_id = row.get("row_id")
        current_episode = row.get("episode_id")
        if not isinstance(row_id, str) or not row_id:
            raise ValueError("teacher row has no row_id")
        if not isinstance(current_episode, str) or not current_episode:
            raise ValueError("teacher row has no episode_id")
        if episode_id is None:
            episode_id = current_episode
        elif current_episode != episode_id:
            raise ValueError("rows for one episode index have mismatched episode_id")
        fingerprint = _row_fingerprint(row)
        previous = fingerprints.get(row_id)
        if previous is not None:
            if previous != fingerprint:
                raise ValueError(f"conflicting duplicate teacher row_id: {row_id}")
            duplicate_count += 1
        else:
            fingerprints[row_id] = fingerprint
        row_count += 1
    return _ResumeEpisodeIndex(fingerprints, row_count, duplicate_count)


def _validate_row_provenance(
    row: Mapping[str, object], episode_id: str, metadata: Mapping[str, object]
) -> None:
    """Validate compact row provenance against the immutable collection sidecar."""

    actual = row.get("provenance")
    expected_root = metadata.get("provenance")
    if actual is None or not isinstance(expected_root, Mapping):
        return
    if not isinstance(actual, Mapping):
        raise ValueError(f"teacher row provenance is invalid for episode {episode_id}")
    expected_source = expected_root.get("source_data")
    actual_source = actual.get("source_data")
    if not isinstance(expected_source, Mapping) or not isinstance(actual_source, Mapping):
        raise ValueError(f"teacher row provenance source_data is invalid for episode {episode_id}")
    expected_episode_manifests = expected_source.get("episode_manifest_sha256")
    expected_records = (
        expected_episode_manifests.get(episode_id)
        if isinstance(expected_episode_manifests, Mapping)
        else None
    )
    actual_records = actual_source.get("manifest_sha256")
    if expected_records is not None and actual_records != expected_records:
        raise ValueError(f"teacher row provenance manifest mismatch for episode {episode_id}")
    if actual_source.get("manifest_scope") not in {None, "episode"}:
        raise ValueError(f"teacher row provenance scope is invalid for episode {episode_id}")
    for key in ("index_path", "index_sha256", "split"):
        if actual_source.get(key) != expected_source.get(key):
            raise ValueError(f"teacher row provenance {key} mismatch for episode {episode_id}")
    actual_episode = actual.get("episode")
    if isinstance(actual_episode, Mapping) and actual_episode.get("episode_id") != episode_id:
        raise ValueError(f"teacher row provenance episode mismatch for episode {episode_id}")
    for key in (
        "teacher_schema_version",
        "binding",
        "model_content_hash",
        "feature_schema_version",
        "render_protocol",
        "budget_protocol",
        "utility_protocol",
        "teacher_dataset_hash",
        "teacher_source_hashes",
        "implementation_sha256",
        "carrier_write_checkpoint_sha256",
    ):
        if key in expected_root and actual.get(key) != expected_root.get(key):
            raise ValueError(f"teacher row provenance {key} mismatch for episode {episode_id}")
    expected_rollin = expected_root.get("rollin")
    actual_rollin = actual.get("rollin")
    if isinstance(expected_rollin, Mapping):
        if not isinstance(actual_rollin, Mapping):
            raise ValueError(f"teacher row provenance rollin is invalid for episode {episode_id}")
        for key, value in expected_rollin.items():
            if actual_rollin.get(key) != value:
                raise ValueError(f"teacher row provenance rollin mismatch for episode {episode_id}")
def _read_rows(
    path: Path, *, metadata: Mapping[str, object] | None = None
) -> dict[str, _ResumeEpisodeIndex]:
    grouped: dict[str, dict[str, str]] = {}
    row_counts: dict[str, int] = {}
    duplicate_counts: dict[str, int] = {}
    row_owners: dict[str, tuple[str, str]] = {}
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        lines = enumerate(handle, start=1)
        for line_number, raw_line in lines:
            line = raw_line.rstrip("\r\n")
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid teacher JSONL line {line_number}") from error
            if not isinstance(row, dict) or row.get("schema_version") != ROW_SCHEMA_VERSION:
                raise ValueError(f"invalid teacher row at line {line_number}")
            row_id = row.get("row_id")
            if not isinstance(row_id, str) or not row_id:
                raise ValueError(f"teacher row {line_number} has no row_id")
            episode_id = row.get("episode_id")
            if not isinstance(episode_id, str) or not episode_id:
                raise ValueError(f"teacher row {line_number} has no episode_id")
            if metadata is not None:
                _validate_row_provenance(row, episode_id, metadata)
            fingerprint = _row_fingerprint(row)
            previous_owner = row_owners.get(row_id)
            if previous_owner is not None:
                previous_episode, previous_fingerprint = previous_owner
                if previous_episode != episode_id or previous_fingerprint != fingerprint:
                    raise ValueError(f"conflicting duplicate teacher row_id: {row_id}")
                duplicate_counts[episode_id] = duplicate_counts.get(episode_id, 0) + 1
            else:
                row_owners[row_id] = (episode_id, fingerprint)
                grouped.setdefault(episode_id, {})[row_id] = fingerprint
            row_counts[episode_id] = row_counts.get(episode_id, 0) + 1
    return {
        episode_id: _ResumeEpisodeIndex(
            fingerprints,
            row_counts[episode_id],
            duplicate_counts.get(episode_id, 0),
        )
        for episode_id, fingerprints in grouped.items()
    }


def _atomic_json(path: Path, value: Mapping[str, object]) -> None:
    temporary = path.with_name(f"{path.name}.{os.getpid()}.{uuid.uuid4().hex}.part")
    try:
        payload = json.dumps(
            _json_value(dict(value)), sort_keys=True, indent=2, allow_nan=False
        ) + "\n"
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        last_error: PermissionError | None = None
        for attempt in range(5):
            try:
                os.replace(temporary, path)
                last_error = None
                break
            except PermissionError as error:
                last_error = error
                if attempt == 4:
                    raise
                time.sleep(0.05 * (attempt + 1))
        if last_error is not None:
            raise last_error
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


__all__ = [
    "ACTION_ORDER",
    "ActionTeacherConfig",
    "DEFAULT_HORIZON",
    "DEFAULT_POLICY_WORK_UNITS",
    "DEFAULT_RAY_CHUNK_SIZE",
    "FEATURE_SCHEMA_VERSION",
    "ROW_SCHEMA_VERSION",
    "TEACHER_SCHEMA_VERSION",
    "clear_episode_context",
    "collect_action_targets",
    "collect_episode_targets",
    "implementation_hash",
    "make_binding",
    "prepare_episode_context",
]
