"""Strict experiment configuration loading and validation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml

from mcss.model.system import ModelConfig

DatasetName = Literal["synthetic", "manifest", "replica", "hypersim"]
SpatialProtocol = Literal["legacy_manifest_bounds", "context_local_metric"]
Bounds3D = tuple[tuple[float, float, float], tuple[float, float, float]]


@dataclass(frozen=True)
class DatasetConfig:
    name: DatasetName
    root: str | None
    image_size: tuple[int, int]
    context_views: int
    target_views: int
    length: int = 64
    sample_stride: int = 1
    spatial_protocol: SpatialProtocol = "legacy_manifest_bounds"
    local_bounds_m: Bounds3D | None = None
    scene_ids: tuple[str, ...] | None = None


@dataclass(frozen=True)
class TrainingConfig:
    output_dir: str
    batch_size: int
    learning_rate: float
    max_steps: int
    num_workers: int
    amp: bool
    amp_dtype: Literal["float16", "bfloat16"]
    log_every: int
    checkpoint_every: int
    measurements: tuple[str, ...]
    gradient_accumulation: int = 1
    weight_decay: float = 0.0
    resume: str | None = None
    lr_schedule: Literal["constant", "warmup_cosine"] = "constant"
    warmup_steps: int = 0
    min_lr_ratio: float = 0.05
    ema_decay: float = 0.98
    evidence_residual_weight: float = 0.0
    context_subset_geometry_weight: float = 0.0


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int
    device: str
    dataset: DatasetConfig
    model: ModelConfig
    training: TrainingConfig

    def __post_init__(self) -> None:
        if self.training.context_subset_geometry_weight == 0.0:
            return
        if self.dataset.context_views != 4:
            raise ValueError(
                "context_subset_geometry_weight requires exactly four context views"
            )
        if self.model.state_architecture != "evidence_residual":
            raise ValueError(
                "context_subset_geometry_weight requires state_architecture: evidence_residual"
            )
        if self.model.mode != "fixed":
            raise ValueError("context_subset_geometry_weight requires model.mode: fixed")


def load_config(path: str | Path) -> ExperimentConfig:
    config_path = Path(path)
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise ValueError(f"invalid YAML in {config_path}") from error
    if not isinstance(raw, dict):
        raise ValueError("config root must be a mapping")
    _reject_unknown(raw, {"seed", "device", "dataset", "model", "training"}, "config")
    _require_keys(raw, {"seed", "device", "dataset", "model", "training"}, "config")
    if not isinstance(raw["seed"], int):
        raise ValueError("seed must be an integer")
    if not isinstance(raw["device"], str):
        raise ValueError("device must be a string")
    return ExperimentConfig(
        seed=raw["seed"],
        device=raw["device"],
        dataset=_parse_dataset(raw["dataset"]),
        model=_parse_model(raw["model"]),
        training=_parse_training(raw["training"]),
    )


def _parse_dataset(value: Any) -> DatasetConfig:
    if not isinstance(value, dict):
        raise ValueError("dataset must be a mapping")
    allowed = {
        "name",
        "root",
        "image_size",
        "context_views",
        "target_views",
        "length",
        "sample_stride",
        "spatial_protocol",
        "local_bounds_m",
        "scene_ids",
    }
    _reject_unknown(value, allowed, "dataset")
    _require_keys(
        value, {"name", "root", "image_size", "context_views", "target_views", "length"}, "dataset"
    )
    name = value["name"]
    if name not in {"synthetic", "manifest", "replica", "hypersim"}:
        raise ValueError("dataset.name must be synthetic, manifest, replica, or hypersim")
    root = value["root"]
    if root is not None and not isinstance(root, str):
        raise ValueError("dataset.root must be a string or null")
    if name != "synthetic" and not root:
        raise ValueError("prepared real datasets require dataset.root")
    image_size = _parse_image_size(value["image_size"])
    context_views = _positive_int(value["context_views"], "dataset.context_views")
    target_views = _positive_int(value["target_views"], "dataset.target_views")
    length = _positive_int(value["length"], "dataset.length")
    sample_stride = _positive_int(value.get("sample_stride", 1), "dataset.sample_stride")
    spatial_protocol = value.get("spatial_protocol", "legacy_manifest_bounds")
    if spatial_protocol not in {"legacy_manifest_bounds", "context_local_metric"}:
        raise ValueError(
            "dataset.spatial_protocol must be legacy_manifest_bounds or context_local_metric"
        )
    raw_local_bounds = value.get("local_bounds_m")
    local_bounds_m = None if raw_local_bounds is None else _parse_bounds(raw_local_bounds)
    if spatial_protocol == "context_local_metric" and local_bounds_m is None:
        raise ValueError("dataset.local_bounds_m is required for context_local_metric")
    raw_scene_ids = value.get("scene_ids")
    scene_ids = None
    if raw_scene_ids is not None:
        if not isinstance(raw_scene_ids, list) or not raw_scene_ids:
            raise ValueError("dataset.scene_ids must be a non-empty list")
        if not all(isinstance(scene_id, str) and scene_id for scene_id in raw_scene_ids):
            raise ValueError("dataset.scene_ids must contain non-empty strings")
        if len(set(raw_scene_ids)) != len(raw_scene_ids):
            raise ValueError("dataset.scene_ids must contain unique values")
        scene_ids = tuple(raw_scene_ids)
    return DatasetConfig(
        name=name,
        root=root,
        image_size=image_size,
        context_views=context_views,
        target_views=target_views,
        length=length,
        sample_stride=sample_stride,
        spatial_protocol=spatial_protocol,
        local_bounds_m=local_bounds_m,
        scene_ids=scene_ids,
    )


def _parse_model(value: Any) -> ModelConfig:
    if not isinstance(value, dict):
        raise ValueError("model must be a mapping")
    allowed = {
        "mode",
        "state_architecture",
        "voxel_resolution",
        "image_feature_dim",
        "state_feature_dim",
        "refinement_blocks",
        "evidence_temperature",
        "observed_residual_floor",
        "completion_residual_scale",
        "surface_peak_temperature",
        "appearance_resolution_scale",
        "n_samples",
        "ray_chunk_size",
    }
    required = {
        "mode",
        "voxel_resolution",
        "image_feature_dim",
        "state_feature_dim",
        "refinement_blocks",
        "n_samples",
        "ray_chunk_size",
    }
    _reject_unknown(value, allowed, "model")
    _require_keys(value, required, "model")
    parsed = dict(value)
    parsed["voxel_resolution"] = _parse_voxel_resolution(value["voxel_resolution"])
    try:
        return ModelConfig(**parsed)
    except (TypeError, ValueError) as error:
        raise ValueError(f"invalid model config: {error}") from error


def _parse_training(value: Any) -> TrainingConfig:
    if not isinstance(value, dict):
        raise ValueError("training must be a mapping")
    allowed = {
        "output_dir",
        "batch_size",
        "learning_rate",
        "max_steps",
        "num_workers",
        "amp",
        "amp_dtype",
        "log_every",
        "checkpoint_every",
        "measurements",
        "gradient_accumulation",
        "weight_decay",
        "resume",
        "lr_schedule",
        "warmup_steps",
        "min_lr_ratio",
        "ema_decay",
        "evidence_residual_weight",
        "context_subset_geometry_weight",
    }
    required = {
        "output_dir",
        "batch_size",
        "learning_rate",
        "max_steps",
        "num_workers",
        "amp",
        "log_every",
        "checkpoint_every",
        "measurements",
    }
    _reject_unknown(value, allowed, "training")
    _require_keys(value, required, "training")
    if not isinstance(value["output_dir"], str) or not value["output_dir"]:
        raise ValueError("training.output_dir must be a non-empty string")
    if not isinstance(value["learning_rate"], (int, float)) or value["learning_rate"] <= 0:
        raise ValueError("training.learning_rate must be positive")
    if not isinstance(value["amp"], bool):
        raise ValueError("training.amp must be boolean")
    amp_dtype = value.get("amp_dtype", "float16")
    if amp_dtype not in {"float16", "bfloat16"}:
        raise ValueError("training.amp_dtype must be float16 or bfloat16")
    if not isinstance(value["measurements"], list) or not all(
        isinstance(name, str) for name in value["measurements"]
    ):
        raise ValueError("training.measurements must be a list of strings")
    resume = value.get("resume")
    if resume is not None and not isinstance(resume, str):
        raise ValueError("training.resume must be a string or null")
    max_steps = _positive_int(value["max_steps"], "training.max_steps")
    gradient_accumulation = _positive_int(
        value.get("gradient_accumulation", 1), "training.gradient_accumulation"
    )
    if max_steps % gradient_accumulation != 0:
        raise ValueError("training.max_steps must be divisible by gradient_accumulation")
    lr_schedule = value.get("lr_schedule", "constant")
    if lr_schedule not in {"constant", "warmup_cosine"}:
        raise ValueError("training.lr_schedule must be constant or warmup_cosine")
    warmup_steps = _nonnegative_int(value.get("warmup_steps", 0), "training.warmup_steps")
    total_updates = max_steps // gradient_accumulation
    if warmup_steps > total_updates:
        raise ValueError("training.warmup_steps cannot exceed optimizer updates")
    if lr_schedule == "constant" and warmup_steps:
        raise ValueError("training.warmup_steps requires lr_schedule: warmup_cosine")
    min_lr_ratio = value.get("min_lr_ratio", 0.05)
    if not isinstance(min_lr_ratio, (int, float)) or not 0.0 <= min_lr_ratio <= 1.0:
        raise ValueError("training.min_lr_ratio must be between zero and one")
    ema_decay = value.get("ema_decay", 0.98)
    if not isinstance(ema_decay, (int, float)) or not 0.0 <= ema_decay < 1.0:
        raise ValueError("training.ema_decay must be in [0, 1)")
    evidence_residual_weight = value.get("evidence_residual_weight", 0.0)
    if (
        not isinstance(evidence_residual_weight, (int, float))
        or evidence_residual_weight < 0.0
    ):
        raise ValueError("training.evidence_residual_weight must be non-negative")
    context_subset_geometry_weight = value.get("context_subset_geometry_weight", 0.0)
    if (
        isinstance(context_subset_geometry_weight, bool)
        or not isinstance(context_subset_geometry_weight, (int, float))
        or not math.isfinite(context_subset_geometry_weight)
    ):
        raise ValueError(
            "training.context_subset_geometry_weight must be a finite non-negative number"
        )
    if context_subset_geometry_weight < 0.0:
        raise ValueError("training.context_subset_geometry_weight must be non-negative")
    checkpoint_every = _positive_int(
        value["checkpoint_every"], "training.checkpoint_every"
    )
    if (
        context_subset_geometry_weight > 0.0
        and checkpoint_every % gradient_accumulation != 0
    ):
        raise ValueError(
            "training.checkpoint_every must be divisible by gradient_accumulation "
            "when context_subset_geometry_weight is enabled"
        )
    return TrainingConfig(
        output_dir=value["output_dir"],
        batch_size=_positive_int(value["batch_size"], "training.batch_size"),
        learning_rate=float(value["learning_rate"]),
        max_steps=max_steps,
        num_workers=_nonnegative_int(value["num_workers"], "training.num_workers"),
        amp=value["amp"],
        amp_dtype=amp_dtype,
        log_every=_positive_int(value["log_every"], "training.log_every"),
        checkpoint_every=checkpoint_every,
        measurements=tuple(value["measurements"]),
        gradient_accumulation=gradient_accumulation,
        weight_decay=float(value.get("weight_decay", 0.0)),
        resume=resume,
        lr_schedule=lr_schedule,
        warmup_steps=warmup_steps,
        min_lr_ratio=float(min_lr_ratio),
        ema_decay=float(ema_decay),
        evidence_residual_weight=float(evidence_residual_weight),
        context_subset_geometry_weight=float(context_subset_geometry_weight),
    )


def _parse_image_size(value: Any) -> tuple[int, int]:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError("dataset.image_size must be [height, width]")
    return (
        _positive_int(value[0], "dataset.image_size[0]"),
        _positive_int(value[1], "dataset.image_size[1]"),
    )


def _parse_voxel_resolution(value: Any) -> tuple[int, int, int]:
    if isinstance(value, int):
        resolution = (value, value, value)
    elif isinstance(value, list) and len(value) == 3:
        resolution = tuple(
            _positive_int(axis, f"model.voxel_resolution[{index}]")
            for index, axis in enumerate(value)
        )
    else:
        raise ValueError("model.voxel_resolution must be an integer or [D, H, W]")
    if min(resolution) < 4:
        raise ValueError("model.voxel_resolution values must be at least four")
    return resolution


def _parse_bounds(value: Any) -> Bounds3D:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError("dataset.local_bounds_m must be [[minx, miny, minz], [maxx, maxy, maxz]]")
    rows: list[tuple[float, float, float]] = []
    for row_index, row in enumerate(value):
        if not isinstance(row, list) or len(row) != 3:
            raise ValueError(f"dataset.local_bounds_m[{row_index}] must contain three values")
        if not all(isinstance(axis, (int, float)) and math.isfinite(axis) for axis in row):
            raise ValueError("dataset.local_bounds_m values must be finite numbers")
        rows.append(tuple(float(axis) for axis in row))  # type: ignore[arg-type]
    if any(upper <= lower for lower, upper in zip(rows[0], rows[1], strict=True)):
        raise ValueError("dataset.local_bounds_m maximum must exceed minimum on every axis")
    return rows[0], rows[1]


def _positive_int(value: Any, name: str) -> int:
    if not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _nonnegative_int(value: Any, name: str) -> int:
    if not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _reject_unknown(value: dict[str, Any], allowed: set[str], name: str) -> None:
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f"unknown {name} keys: {sorted(unknown)}")


def _require_keys(value: dict[str, Any], required: set[str], name: str) -> None:
    missing = required - set(value)
    if missing:
        raise ValueError(f"missing {name} keys: {sorted(missing)}")
