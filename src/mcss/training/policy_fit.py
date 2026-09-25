"""Offline fitting for the prefix-only learned action policy.

The fitter deliberately consumes the compact all-actions-per-prefix teacher
rows.  It never opens an episode, query vault, image, or future observation;
the only model input is the already materialized ``features`` vector in each
row.  Scene splitting and normalization happen before the optimizer sees a
validation row so that recollected prefixes cannot leak across the split.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from mcss.dynamic.policy import (
    ACTION_ORDER,
    FEATURE_NAMES,
    FEATURE_SCHEMA_VERSION,
    POLICY_FEATURE_DIM,
    LearnedActionPolicy,
)
from mcss.dynamic.types import Action

ACTION_TEACHER_SCHEMA = "mcss.dynamic.action_teacher_row.v1"
TEACHER_ROW_SCHEMA = ACTION_TEACHER_SCHEMA
TEACHER_COLLECTION_SCHEMA = "mcss.dynamic.action_teacher.v1"
POLICY_TRAINING_SCHEMA = "mcss.dynamic.policy_training.v1"
_REQUIRED_RUNTIME_BINDING_KEYS = frozenset(
    {
        "model_content_hash",
        "feature_schema_version",
        "render_protocol",
        "budget_protocol",
        "utility_protocol",
    }
)

_ACTION_FEATURE_INDICES = tuple(
    FEATURE_NAMES.index(f"previous_action_{action.value.lower()}")
    for action in ACTION_ORDER
)

ROW_FIELDS = frozenset(
    {
        "schema_version",
        "row_id",
        "episode_id",
        "scene_id",
        "split_id",
        "prefix_step",
        "prefix_frame_id",
        "control_input",
        "features",
        "actions",
        "feasible_mask",
        "losses",
        "advantages",
        "costs",
        "horizon_used",
        "provenance",
    }
)

CONTROL_FIELDS = frozenset(
    {
        "rgb_mse",
        "opacity_mean",
        "current_image_mean",
        "density_mean",
        "fast_norm",
        "observed_count",
        "previous_action",
        "remaining_budget",
        "image_feature",
        "image_feature_stats",
        "rgb_residual_4x4",
        "coverage",
        "valid_count",
        "camera_change",
        "fast_state_stats",
        "remaining_steps",
    }
)


@dataclass(frozen=True)
class TeacherRow:
    """Validated, tensorized representation of one teacher prefix."""

    schema_version: str
    row_id: str
    episode_id: str
    scene_id: str
    split_id: str
    prefix_step: int
    prefix_frame_id: int
    control_input: Mapping[str, Any]
    features: Tensor
    actions: tuple[Action, ...]
    feasible_mask: Tensor
    losses: Tensor
    advantages: Tensor
    costs: Tensor
    horizon_used: int
    provenance: Mapping[str, Any]
    fingerprint: str

    @property
    def binding(self) -> dict[str, Any]:
        return _binding_from_provenance(self.provenance)


@dataclass(frozen=True)
class TeacherDataset:
    rows: tuple[TeacherRow, ...]
    binding: Mapping[str, Any]
    source_hashes: Mapping[str, str]
    dataset_hash: str


@dataclass(frozen=True)
class PolicyFitResult:
    """The trained policy and evidence needed to save/review its checkpoint."""

    policy: LearnedActionPolicy
    metrics: Mapping[str, Any]
    train_scene_ids: tuple[str, ...]
    validation_scene_ids: tuple[str, ...]
    feature_mean: Tensor
    feature_scale: Tensor
    target_mean: float
    target_scale: float
    binding: Mapping[str, Any]
    row_count: int

    def __getitem__(self, key: str) -> Any:
        if key == "policy":
            return self.policy
        if key == "metrics":
            return self.metrics
        if key == "binding":
            return self.binding
        return self.metrics[key]

    def checkpoint_provenance(self) -> dict[str, Any]:
        return {
            "schema_version": POLICY_TRAINING_SCHEMA,
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "train_scene_ids": list(self.train_scene_ids),
            "validation_scene_ids": list(self.validation_scene_ids),
            "row_count": self.row_count,
            "target_normalization": {
                "mean": float(self.target_mean),
                "scale": float(self.target_scale),
            },
            "metrics": _jsonable(dict(self.metrics)),
        }


def _require_exact_fields(value: Mapping[str, Any], expected: frozenset[str], name: str) -> None:
    actual = frozenset(value)
    if actual != expected:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        raise ValueError(f"{name} fields mismatch; missing={missing}, unexpected={unexpected}")


def _nonempty_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _finite_float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def _numeric_vector(value: Any, size: int, name: str) -> Tensor:
    if not isinstance(value, list) or len(value) != size:
        raise ValueError(f"{name} must be a list of exactly {size} values")
    result = [_finite_float(item, f"{name}[{index}]") for index, item in enumerate(value)]
    return torch.tensor(result, dtype=torch.float32)


def _nullable_vector(value: Any, mask: list[bool], name: str) -> Tensor:
    if not isinstance(value, list) or len(value) != len(mask):
        raise ValueError(f"{name} must be a list of exactly {len(mask)} values")
    output: list[float] = []
    for index, item in enumerate(value):
        if mask[index]:
            if item is None:
                raise ValueError(f"{name}[{index}] must be finite for a feasible action")
            output.append(_finite_float(item, f"{name}[{index}]"))
        else:
            if item is not None:
                raise ValueError(f"{name}[{index}] must be null for an infeasible action")
            output.append(0.0)
    return torch.tensor(output, dtype=torch.float32)


def _binding_from_provenance(provenance: Mapping[str, Any]) -> dict[str, Any]:
    """Extract shared identity fields while ignoring row-local provenance.

    Shared identity fields are deliberately read only from the flat
    provenance object.  A nested object is not a binding alias: accepting one
    here would allow a teacher with a future schema to silently bypass the
    checkpoint contract.
    """

    selected: dict[str, Any] = {}
    keys = {
        "model_content_hash",
        "feature_schema_version",
        "render_protocol",
        "budget_protocol",
        "utility_protocol",
        "source_hash",
        "source_hashes",
        "dataset_hash",
        "dataset_hashes",
        "teacher_dataset_hash",
        "teacher_source_hashes",
    }
    for key in keys:
        if key in provenance:
            selected[key] = _jsonable(provenance[key])
    return selected


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


def _runtime_binding(binding: Mapping[str, Any]) -> dict[str, Any]:
    """Return binding fields that must match across recollected datasets."""

    return {key: value for key, value in binding.items() if key not in _DATASET_BINDING_KEYS}


def _require_runtime_binding(binding: Mapping[str, Any]) -> None:
    missing = _REQUIRED_RUNTIME_BINDING_KEYS - set(binding)
    if missing:
        raise ValueError(
            "teacher metadata binding is missing checkpoint runtime fields: "
            f"{sorted(missing)}"
        )
    if binding["feature_schema_version"] != FEATURE_SCHEMA_VERSION:
        raise ValueError("teacher binding feature schema does not match runtime schema")


def parse_teacher_row(value: Mapping[str, Any], *, source: str = "row") -> TeacherRow:
    """Validate one all-actions teacher row using an exact top-level schema."""

    if not isinstance(value, Mapping):
        raise ValueError(f"{source} must be a JSON object")
    _require_exact_fields(value, ROW_FIELDS, source)
    if value["schema_version"] != ACTION_TEACHER_SCHEMA:
        raise ValueError(f"{source} has unsupported schema_version")

    row_id = _nonempty_string(value["row_id"], f"{source}.row_id")
    episode_id = _nonempty_string(value["episode_id"], f"{source}.episode_id")
    scene_id = _nonempty_string(value["scene_id"], f"{source}.scene_id")
    split_id = _nonempty_string(value["split_id"], f"{source}.split_id")
    if split_id != "train":
        raise ValueError(f"{source}.split_id must be 'train'")
    prefix_step = _nonnegative_int(value["prefix_step"], f"{source}.prefix_step")
    prefix_frame_id = _nonnegative_int(value["prefix_frame_id"], f"{source}.prefix_frame_id")
    horizon_used = _nonnegative_int(value["horizon_used"], f"{source}.horizon_used")
    if horizon_used == 0:
        raise ValueError(f"{source}.horizon_used must be positive")

    control = value["control_input"]
    if not isinstance(control, Mapping):
        raise ValueError(f"{source}.control_input must be an object")
    unknown_control = sorted(set(control) - CONTROL_FIELDS)
    if unknown_control:
        raise ValueError(f"{source}.control_input has unexpected fields: {unknown_control}")
    for key, item in control.items():
        if isinstance(item, bool):
            raise ValueError(f"{source}.control_input.{key} must not be boolean")
        if isinstance(item, (int, float)) and not math.isfinite(float(item)):
            raise ValueError(f"{source}.control_input.{key} must be finite")
    control_copy = dict(control)

    features = _numeric_vector(value["features"], POLICY_FEATURE_DIM, f"{source}.features")
    actions_value = value["actions"]
    expected_actions = [action.value for action in ACTION_ORDER]
    if not isinstance(actions_value, list) or actions_value != expected_actions:
        raise ValueError(f"{source}.actions must equal the fixed ACTION_ORDER")
    actions = ACTION_ORDER

    feasible_value = value["feasible_mask"]
    if not isinstance(feasible_value, list) or len(feasible_value) != len(ACTION_ORDER):
        raise ValueError(f"{source}.feasible_mask must contain four booleans")
    if any(type(item) is not bool for item in feasible_value):
        raise ValueError(f"{source}.feasible_mask must contain booleans")
    feasible = torch.tensor(feasible_value, dtype=torch.bool)
    if not bool(feasible.any()):
        raise ValueError(f"{source}.feasible_mask must allow at least one action")
    losses = _nullable_vector(value["losses"], feasible_value, f"{source}.losses")
    advantages = _nullable_vector(value["advantages"], feasible_value, f"{source}.advantages")
    costs = _nullable_vector(value["costs"], feasible_value, f"{source}.costs")
    provenance = value["provenance"]
    if not isinstance(provenance, Mapping):
        raise ValueError(f"{source}.provenance must be an object")
    provenance_copy = _jsonable(dict(provenance))
    if not isinstance(provenance_copy, dict):
        raise ValueError(f"{source}.provenance must be JSON-compatible")
    declared_feature_schema = provenance_copy.get("feature_schema_version")
    if declared_feature_schema is not None and declared_feature_schema != FEATURE_SCHEMA_VERSION:
        raise ValueError(f"{source}.provenance feature schema does not match runtime schema")

    # Canonical content is used for duplicate row IDs.  JSON's sorted form
    # makes ordering differences harmless while still rejecting label edits.
    fingerprint_value = dict(value)
    fingerprint_provenance = dict(provenance_copy)
    for key in _DATASET_BINDING_KEYS:
        fingerprint_provenance.pop(key, None)
    fingerprint_value["provenance"] = fingerprint_provenance
    fingerprint = hashlib.sha256(
        json.dumps(_jsonable(fingerprint_value), sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()
    return TeacherRow(
        schema_version=value["schema_version"],
        row_id=row_id,
        episode_id=episode_id,
        scene_id=scene_id,
        split_id=split_id,
        prefix_step=prefix_step,
        prefix_frame_id=prefix_frame_id,
        control_input=control_copy,
        features=features,
        actions=actions,
        feasible_mask=feasible,
        losses=losses,
        advantages=advantages,
        costs=costs,
        horizon_used=horizon_used,
        provenance=provenance_copy,
        fingerprint=fingerprint,
    )


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Tensor):
        return _jsonable(value.detach().cpu().tolist())
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("metadata must contain only finite numbers")
        return value
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    raise ValueError(f"value of type {type(value).__name__} is not JSON-compatible")


def _teacher_files(paths: Iterable[str | Path]) -> tuple[Path, ...]:
    files: list[Path] = []
    for raw_path in paths:
        path = Path(raw_path)
        if not path.exists():
            raise FileNotFoundError(path)
        if path.is_file():
            if path.suffix.lower() != ".jsonl":
                raise ValueError(f"teacher input must be JSONL: {path}")
            files.append(path)
        elif path.is_dir():
            files.extend(sorted(item for item in path.rglob("*.jsonl") if item.is_file()))
        else:
            raise ValueError(f"teacher input is neither file nor directory: {path}")
    unique = tuple(dict.fromkeys(item.resolve() for item in sorted(files)))
    if not unique:
        raise ValueError("teacher input contains no JSONL files")
    return unique


def _metadata_binding(path: Path) -> dict[str, Any]:
    """Read the action-teacher sidecar binding when one is present."""

    candidates = (
        path.with_name(f"{path.stem}.metadata.json"),
        path.parent / "metadata.json",
    )
    metadata_path = next((candidate for candidate in candidates if candidate.is_file()), None)
    if metadata_path is None:
        return {}
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid teacher metadata JSON: {metadata_path}") from error
    if not isinstance(metadata, Mapping):
        raise ValueError(f"teacher metadata must be an object: {metadata_path}")
    if metadata.get("schema_version") not in {None, TEACHER_COLLECTION_SCHEMA}:
        raise ValueError(f"unsupported teacher metadata schema: {metadata_path}")
    row_schema = metadata.get("row_schema_version")
    if row_schema not in {None, ACTION_TEACHER_SCHEMA}:
        raise ValueError(f"teacher metadata row schema mismatch: {metadata_path}")
    binding = metadata.get("binding")
    if binding is None:
        provenance = metadata.get("provenance")
        if isinstance(provenance, Mapping):
            binding = provenance.get("binding")
    if binding is None:
        return {}
    if not isinstance(binding, Mapping):
        raise ValueError(f"teacher metadata binding must be an object: {metadata_path}")
    normalized = _jsonable(dict(binding))
    if not isinstance(normalized, dict):
        raise ValueError(f"teacher metadata binding must be JSON-compatible: {metadata_path}")
    return normalized


def _read_rows(files: Sequence[Path]) -> tuple[list[TeacherRow], dict[str, Any]]:
    rows_by_id: dict[str, TeacherRow] = {}
    binding: dict[str, Any] | None = None
    for path in files:
        file_binding = _metadata_binding(path)
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"invalid teacher JSON at {path}:{line_number}") from error
                row = parse_teacher_row(raw, source=f"{path}:{line_number}")
                row_binding = {**file_binding, **row.binding}
                runtime_binding = _runtime_binding(row_binding)
                if binding is None:
                    binding = runtime_binding
                elif runtime_binding != binding:
                    raise ValueError(
                        f"teacher binding mismatch at {path}:{line_number}; "
                        "recollection may merge only exact bindings"
                    )
                # The fingerprint and binding checks above consume the raw
                # provenance.  Keep only the flat binding fields on the row so
                # expanded legacy manifests are not retained once validation
                # is complete.  This also handles episode_compact_v1 rows from
                # the current writer without depending on its row-local keys.
                compact_provenance = _binding_from_provenance(
                    {**file_binding, **row.provenance}
                )
                row = replace(row, provenance=compact_provenance)
                previous = rows_by_id.get(row.row_id)
                if previous is not None:
                    if previous.fingerprint != row.fingerprint:
                        raise ValueError(f"conflicting duplicate teacher row_id: {row.row_id}")
                    continue
                rows_by_id[row.row_id] = row
    if not rows_by_id:
        raise ValueError("teacher input contains no rows")
    return list(rows_by_id.values()), binding or {}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_teacher_dataset(paths: Iterable[str | Path]) -> TeacherDataset:
    """Load and merge teacher JSONL files with strict duplicate/binding checks."""

    files = _teacher_files(paths)
    rows, binding = _read_rows(files)
    source_hashes = {str(path): _sha256_file(path) for path in files}
    digest = hashlib.sha256()
    for path in files:
        digest.update(str(path).encode("utf-8"))
        digest.update(source_hashes[str(path)].encode("ascii"))
    for row in sorted(rows, key=lambda item: item.row_id):
        digest.update(row.row_id.encode("utf-8"))
        digest.update(row.fingerprint.encode("ascii"))
    return TeacherDataset(tuple(rows), binding, source_hashes, digest.hexdigest())


def load_teacher_rows(paths: Iterable[str | Path]) -> list[TeacherRow]:
    """Compatibility helper returning only validated, deduplicated rows."""

    return list(load_teacher_dataset(paths).rows)


def split_scene_ids(
    scene_ids: Iterable[str],
    *,
    holdout_fraction: float = 0.2,
    seed: int = 0,
    validation_scene_ids: Iterable[str] | None = None,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return deterministic disjoint train/validation scene IDs."""

    scenes = sorted(set(_nonempty_string(scene, "scene_id") for scene in scene_ids))
    if not scenes:
        raise ValueError("at least one scene is required")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    if validation_scene_ids is not None:
        validation = sorted(set(validation_scene_ids))
        unknown = set(validation) - set(scenes)
        if unknown:
            raise ValueError(f"validation_scene_ids are absent from rows: {sorted(unknown)}")
        if len(validation) == len(scenes):
            raise ValueError("at least one scene must remain in the training split")
        return tuple(scene for scene in scenes if scene not in validation), tuple(validation)
    if isinstance(holdout_fraction, bool) or not isinstance(holdout_fraction, (int, float)):
        raise ValueError("holdout_fraction must be a number in [0,1)")
    if not math.isfinite(float(holdout_fraction)) or not 0 <= holdout_fraction < 1:
        raise ValueError("holdout_fraction must be a number in [0,1)")
    if len(scenes) < 2 or holdout_fraction == 0:
        return tuple(scenes), ()
    count = max(1, math.ceil(len(scenes) * float(holdout_fraction)))
    count = min(count, len(scenes) - 1)
    ranked = sorted(
        scenes,
        key=lambda scene: hashlib.sha256(f"{seed}:{scene}".encode()).hexdigest(),
    )
    validation = set(ranked[:count])
    return tuple(scene for scene in scenes if scene not in validation), tuple(
        scene for scene in scenes if scene in validation
    )


def _coerce_rows(rows: Iterable[TeacherRow | Mapping[str, Any]]) -> list[TeacherRow]:
    result: list[TeacherRow] = []
    for index, row in enumerate(rows):
        if isinstance(row, TeacherRow):
            result.append(row)
        elif isinstance(row, Mapping):
            result.append(parse_teacher_row(row, source=f"rows[{index}]"))
        else:
            raise TypeError("rows must contain TeacherRow objects or teacher row mappings")
    if not result:
        raise ValueError("at least one teacher row is required")
    dedup: dict[str, TeacherRow] = {}
    binding: dict[str, Any] | None = None
    for row in result:
        row_runtime_binding = _runtime_binding(row.binding)
        if binding is None:
            binding = row_runtime_binding
        elif row_runtime_binding != binding:
            raise ValueError("teacher binding mismatch across rows")
        previous = dedup.get(row.row_id)
        if previous is not None and previous.fingerprint != row.fingerprint:
            raise ValueError(f"conflicting duplicate teacher row_id: {row.row_id}")
        dedup.setdefault(row.row_id, row)
    return list(dedup.values())


def _row_tensors(rows: Sequence[TeacherRow], scene_ids: set[str]) -> tuple[Tensor, Tensor, Tensor]:
    selected = [row for row in rows if row.scene_id in scene_ids]
    if not selected:
        raise ValueError("training split contains no rows")
    features = torch.stack([row.features for row in selected]).float()
    targets = torch.stack([row.advantages for row in selected]).float()
    masks = torch.stack([row.feasible_mask for row in selected]).bool()
    if not torch.isfinite(features).all() or not torch.isfinite(targets).all():
        raise ValueError("teacher tensors must be finite")
    return features, targets, masks


def _argmax_action(values: Tensor, mask: Tensor) -> int:
    best: int | None = None
    best_value: float | None = None
    for index, _action in enumerate(ACTION_ORDER):
        if not bool(mask[index]):
            continue
        value = float(values[index])
        if best is None or value > best_value:  # fixed ACTION_ORDER tie break
            best, best_value = index, value
    if best is None:
        raise ValueError("a row has no feasible actions")
    return best


def _validation_metrics(
    policy: LearnedActionPolicy,
    rows: Sequence[TeacherRow],
    validation_scene_ids: set[str],
    *,
    target_mean: float,
    target_scale: float,
    tie_tolerance: float,
) -> dict[str, Any]:
    selected = [row for row in rows if row.scene_id in validation_scene_ids]
    if not selected:
        return {
            "heldout_row_count": 0,
            "heldout_scene_count": 0,
            "heldout_loss": None,
            "heldout_regret": None,
            "heldout_top_accuracy": None,
            "heldout_near_tie_coverage": None,
        }
    features = torch.stack([row.features for row in selected]).to(next(policy.parameters()).device)
    with torch.no_grad():
        predictions = policy(features).detach().cpu() * target_scale + target_mean
    losses: list[float] = []
    regrets: list[float] = []
    accuracy: list[float] = []
    near_ties: list[float] = []
    for row, predicted in zip(selected, predictions, strict=True):
        feasible = row.feasible_mask
        target = row.advantages
        mask = feasible
        pred_norm = (predicted - target_mean) / target_scale
        normalized_target = (target - target_mean) / target_scale
        losses.append(float(((pred_norm - normalized_target).square()[mask]).mean()))
        selected_index = _argmax_action(predicted, mask)
        feasible_target = target[mask]
        best = float(feasible_target.max())
        selected_value = float(target[selected_index])
        regrets.append(max(0.0, best - selected_value))
        top = torch.nonzero(mask & (target >= best - tie_tolerance), as_tuple=False).reshape(-1)
        accuracy.append(float(selected_index in {int(item) for item in top}))
        near_ties.append(float((feasible_target.max() - feasible_target.min()) <= tie_tolerance))
    return {
        "heldout_row_count": len(selected),
        "heldout_scene_count": len(validation_scene_ids),
        "heldout_loss": float(sum(losses) / len(losses)),
        "heldout_regret": float(sum(regrets) / len(regrets)),
        "heldout_top_accuracy": float(sum(accuracy) / len(accuracy)),
        "heldout_near_tie_coverage": float(sum(near_ties) / len(near_ties)),
    }


def fit_action_policy(
    rows: Iterable[TeacherRow | Mapping[str, Any]],
    *,
    hidden_dim: int = 32,
    epochs: int = 100,
    learning_rate: float = 1e-3,
    weight_decay: float = 0.0,
    seed: int = 0,
    device: str | torch.device = "cpu",
    holdout_fraction: float = 0.2,
    validation_scene_ids: Iterable[str] | None = None,
    tie_tolerance: float = 1e-4,
    init_policy: LearnedActionPolicy | None = None,
) -> PolicyFitResult:
    """Fit normalized action-advantage regression with a scene holdout.

    ``advantages`` remain continuous labels.  Every feasible action contributes
    to the masked loss, including actions close to the row winner, so the
    result does not collapse teacher supervision into hard one-hot targets.
    """

    normalized_rows = _coerce_rows(rows)
    if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs <= 0:
        raise ValueError("epochs must be a positive integer")
    if not math.isfinite(float(learning_rate)) or learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive")
    if not math.isfinite(float(weight_decay)) or weight_decay < 0:
        raise ValueError("weight_decay must be finite and nonnegative")
    if not math.isfinite(float(tie_tolerance)) or tie_tolerance < 0:
        raise ValueError("tie_tolerance must be finite and nonnegative")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    train_ids, validation_ids = split_scene_ids(
        (row.scene_id for row in normalized_rows),
        holdout_fraction=holdout_fraction,
        seed=seed,
        validation_scene_ids=validation_scene_ids,
    )
    train_features, train_targets, train_mask = _row_tensors(
        normalized_rows, set(train_ids)
    )
    _require_runtime_binding(_runtime_binding(normalized_rows[0].binding))
    feature_mean = train_features.mean(dim=0)
    feature_scale = train_features.std(dim=0, unbiased=False)
    feature_scale = torch.where(
        feature_scale <= 1e-6,
        torch.ones_like(feature_scale),
        feature_scale,
    )
    # Previous-action one-hot values are categorical indicators, so centering
    # or scaling them would change their identity semantics at inference time.
    feature_mean = feature_mean.clone()
    feature_scale = feature_scale.clone()
    feature_mean[list(_ACTION_FEATURE_INDICES)] = 0.0
    feature_scale[list(_ACTION_FEATURE_INDICES)] = 1.0
    train_target_values = train_targets[train_mask]
    if train_target_values.numel() == 0:
        raise ValueError("training rows contain no feasible action labels")
    target_mean = float(train_target_values.mean())
    target_scale = max(float(train_target_values.std(unbiased=False)), 1e-6)
    train_targets = (train_targets - target_mean) / target_scale

    target_device = torch.device(device)
    if target_device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("requested CUDA device is unavailable")
    torch.manual_seed(seed)
    random.seed(seed)
    if init_policy is None:
        policy = LearnedActionPolicy(
            hidden_dim=hidden_dim,
            input_dim=POLICY_FEATURE_DIM,
            feature_mean=feature_mean,
            feature_scale=feature_scale,
            target_mean=target_mean,
            target_scale=target_scale,
            seed=seed,
        )
    else:
        if not isinstance(init_policy, LearnedActionPolicy):
            raise TypeError("init_policy must be a LearnedActionPolicy")
        if init_policy.input_dim != POLICY_FEATURE_DIM or init_policy.hidden_dim != hidden_dim:
            raise ValueError("init_policy architecture does not match requested policy")
        policy = LearnedActionPolicy(
            hidden_dim=hidden_dim,
            input_dim=POLICY_FEATURE_DIM,
            feature_mean=feature_mean,
            feature_scale=feature_scale,
            target_mean=target_mean,
            target_scale=target_scale,
            seed=seed,
        )
        policy.model.load_state_dict(init_policy.model.state_dict(), strict=True)
    policy.to(target_device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate, weight_decay=weight_decay)
    # The policy owns the single feature normalization step.  Keep these rows
    # raw and pass the train-scene statistics through its registered buffers.
    train_features = train_features.to(target_device)
    train_targets = train_targets.to(target_device)
    train_mask = train_mask.to(target_device)
    training_loss = float("nan")
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        predictions = policy(train_features)
        squared = (predictions - train_targets).square()
        loss = squared[train_mask].mean()
        if not torch.isfinite(loss):
            raise ValueError("policy training produced a nonfinite loss")
        loss.backward()
        optimizer.step()
        training_loss = float(loss.detach().cpu())
    policy.eval()
    policy.feature_mean.copy_(feature_mean.to(policy.feature_mean.device))
    policy.feature_scale.copy_(feature_scale.to(policy.feature_scale.device))
    policy.target_mean = target_mean
    policy.target_scale = target_scale
    train_scene_set = set(train_ids)
    train_rows = [row for row in normalized_rows if row.scene_id in train_scene_set]
    train_near_ties: list[float] = []
    for row in train_rows:
        target = row.advantages
        feasible_target = target[row.feasible_mask]
        train_near_ties.append(
            float((feasible_target.max() - feasible_target.min()) <= tie_tolerance)
        )
    metrics = {
        "training_loss": training_loss,
        "training_row_count": len(train_rows),
        "training_scene_count": len(train_ids),
        "training_near_tie_coverage": float(sum(train_near_ties) / len(train_near_ties)),
        "target_mean": target_mean,
        "target_scale": target_scale,
        **_validation_metrics(
            policy,
            normalized_rows,
            set(validation_ids),
            target_mean=target_mean,
            target_scale=target_scale,
            tie_tolerance=tie_tolerance,
        ),
    }
    return PolicyFitResult(
        policy=policy,
        metrics=metrics,
        train_scene_ids=tuple(train_ids),
        validation_scene_ids=tuple(validation_ids),
        feature_mean=feature_mean.detach().cpu(),
        feature_scale=feature_scale.detach().cpu(),
        target_mean=target_mean,
        target_scale=target_scale,
        binding=_runtime_binding(normalized_rows[0].binding),
        row_count=len(normalized_rows),
    )


def load_init_policy(
    path: str | Path, *, device: str | torch.device = "cpu"
) -> LearnedActionPolicy:
    """Load an existing policy through the policy worker's strict checkpoint API."""

    from mcss.dynamic.policy_checkpoint import load_policy_checkpoint

    policy, _ = load_policy_checkpoint(path, device=device)
    return policy


def _json_default(value: Any) -> Any:
    if isinstance(value, Tensor):
        return value.detach().cpu().tolist()
    raise TypeError(f"cannot encode {type(value).__name__}")


def main(argv: Sequence[str] | None = None) -> int:
    """Small importable entry point used by ``scripts/train_action_policy.py``."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher", nargs="+", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--init-policy", type=Path, default=None)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--holdout-fraction", type=float, default=0.2)
    args = parser.parse_args(argv)

    if args.output.exists() or args.output.with_name(f"{args.output.name}.part").exists():
        raise FileExistsError(f"refusing to overwrite existing policy output: {args.output}")
    dataset = load_teacher_dataset(args.teacher)
    init_policy = (
        None
        if args.init_policy is None
        else load_init_policy(args.init_policy, device=args.device)
    )
    result = fit_action_policy(
        dataset.rows,
        hidden_dim=args.hidden_dim,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        seed=args.seed,
        device=args.device,
        holdout_fraction=args.holdout_fraction,
        init_policy=init_policy,
    )
    binding = dict(dataset.binding)
    missing_runtime = {
        "model_content_hash",
        "feature_schema_version",
        "render_protocol",
        "budget_protocol",
        "utility_protocol",
    } - set(binding)
    if missing_runtime:
        raise ValueError(
            "teacher metadata binding is missing checkpoint runtime fields: "
            f"{sorted(missing_runtime)}"
        )
    if binding["feature_schema_version"] != FEATURE_SCHEMA_VERSION:
        raise ValueError("teacher binding feature schema does not match runtime schema")
    binding["teacher_dataset_hash"] = dataset.dataset_hash
    binding["teacher_source_hashes"] = dict(dataset.source_hashes)
    provenance = result.checkpoint_provenance()
    provenance.update(
        {
            "seed": args.seed,
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "teacher_dataset_hash": dataset.dataset_hash,
            "teacher_source_hashes": dict(dataset.source_hashes),
        }
    )
    from mcss.dynamic.policy_checkpoint import save_policy_checkpoint

    args.output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_hash = save_policy_checkpoint(
        args.output, result.policy, binding=binding, provenance=provenance
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "checkpoint_sha256": checkpoint_hash,
                "binding": binding,
                **_jsonable(dict(result.metrics)),
            },
            sort_keys=True,
            default=_json_default,
        )
    )
    return 0


__all__ = [
    "ACTION_TEACHER_SCHEMA",
    "CONTROL_FIELDS",
    "PolicyFitResult",
    "ROW_FIELDS",
    "TEACHER_COLLECTION_SCHEMA",
    "TEACHER_ROW_SCHEMA",
    "TeacherDataset",
    "TeacherRow",
    "fit_action_policy",
    "load_init_policy",
    "load_teacher_dataset",
    "load_teacher_rows",
    "main",
    "parse_teacher_row",
    "split_scene_ids",
]
