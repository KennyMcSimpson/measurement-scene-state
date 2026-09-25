"""Immutable orchestration for V5 diagnostics required before V6 development."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import re
import time
from collections.abc import Sequence
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import torch
import yaml

from mcss.config import ExperimentConfig, load_config
from mcss.data.collate import collate_scene_examples
from mcss.data.manifest_dataset import ManifestSceneDataset, _find_manifests
from mcss.engine import Trainer
from mcss.evidence_audit import audit_evidence


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run immutable V5 diagnostics required before V6 development."
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--checkpoint-sweep", nargs="*", type=Path, default=[])
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--sample-counts", nargs="+", type=int)
    parser.add_argument("--ray-chunk-size", type=int)
    parser.add_argument("--surface-audit", action="store_true")
    parser.add_argument("--audit-samples", type=int, default=256)
    parser.add_argument("--audit-windows-per-scene", type=int)
    args = parser.parse_args(argv)
    if args.checkpoint is None and not args.checkpoint_sweep:
        parser.error("provide --checkpoint or --checkpoint-sweep")
    if args.sample_counts is not None and args.checkpoint is None:
        parser.error("--sample-counts requires --checkpoint")
    if args.surface_audit and args.checkpoint is None:
        parser.error("--surface-audit requires --checkpoint")
    if args.audit_samples < 2:
        parser.error("--audit-samples must be at least two")
    if args.audit_windows_per_scene is not None and args.audit_windows_per_scene < 1:
        parser.error("--audit-windows-per-scene must be positive")

    output_root = args.output_root.resolve()
    phase_report = output_root / "phase0_report.json"
    _refuse_existing(phase_report)
    actions: list[str] = []
    reports: dict[str, str] = {}
    if args.checkpoint_sweep:
        report = run_checkpoint_sweep(
            args.config,
            args.checkpoint_sweep,
            output_root / "checkpoints",
            n_samples=None,
            ray_chunk_size=args.ray_chunk_size,
        )
        del report
        actions.append("checkpoint_sweep")
        reports["checkpoint_sweep"] = str(output_root / "checkpoints" / "checkpoint_sweep.json")
    if args.checkpoint is not None and args.sample_counts is not None:
        report = run_renderer_sweep(
            args.config,
            args.checkpoint,
            output_root / "renderer",
            sample_counts=tuple(args.sample_counts),
            ray_chunk_size=args.ray_chunk_size,
        )
        del report
        actions.append("renderer_sweep")
        reports["renderer_sweep"] = str(output_root / "renderer" / "renderer_sweep.json")
    if args.surface_audit and args.checkpoint is not None:
        report = run_surface_audit(
            args.config,
            args.checkpoint,
            output_root / "surface",
            n_samples=args.audit_samples,
            windows_per_scene=args.audit_windows_per_scene,
        )
        del report
        actions.append("surface_audit")
        reports["surface_audit"] = str(output_root / "surface" / "evidence_audit.json")
    if not actions:
        parser.error("no diagnostic action requested")
    _atomic_json(
        phase_report,
        {
            "schema_version": "mcss.phase0.v1",
            "source": _source_metadata(
                args.config.resolve(),
                (args.checkpoint or args.checkpoint_sweep[0]).resolve(),
            ),
            "actions": actions,
            "reports": reports,
        },
    )
    print(json.dumps({"phase0_report": str(phase_report), "actions": actions}, sort_keys=True))
    return 0


def pilot_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Write immutable label-blind V5 pilot selections and configs."
    )
    parser.add_argument("--train-config", required=True, type=Path)
    parser.add_argument("--validation-config", required=True, type=Path)
    parser.add_argument("--selection-path", required=True, type=Path)
    parser.add_argument("--train-config-output", required=True, type=Path)
    parser.add_argument("--validation-config-output", required=True, type=Path)
    parser.add_argument("--train-output-dir", required=True, type=Path)
    parser.add_argument("--validation-output-dir", required=True, type=Path)
    parser.add_argument("--train-count", type=int, default=16)
    parser.add_argument("--validation-count", type=int, default=8)
    parser.add_argument("--train-max-steps", type=int, default=10_000)
    parser.add_argument("--partition-registry", type=Path)
    args = parser.parse_args(argv)
    report = write_pilot_artifacts(
        args.train_config,
        args.validation_config,
        selection_path=args.selection_path,
        train_config_output=args.train_config_output,
        validation_config_output=args.validation_config_output,
        train_output_dir=args.train_output_dir,
        validation_output_dir=args.validation_output_dir,
        train_count=args.train_count,
        validation_count=args.validation_count,
        train_max_steps=args.train_max_steps,
        partition_registry=args.partition_registry,
    )
    train_split = report["splits"]["train"]
    validation_split = report["splits"]["validation"]
    print(
        json.dumps(
            {
                "selection_path": str(args.selection_path.resolve()),
                "selected_train_scenes": len(train_split["selected_scene_ids"]),
                "selected_validation_scenes": len(
                    validation_split["selected_scene_ids"]
                ),
            },
            sort_keys=True,
        )
    )
    return 0


def run_checkpoint_sweep(
    config_path: str | Path,
    checkpoints: Sequence[str | Path],
    output_root: str | Path,
    *,
    n_samples: int | None = None,
    ray_chunk_size: int | None = None,
) -> dict[str, object]:
    """Evaluate each checkpoint once without sharing or overwriting output directories."""

    config_path = Path(config_path).resolve()
    output_root = Path(output_root).resolve()
    report_path = output_root / "checkpoint_sweep.json"
    _refuse_existing(report_path)
    config = load_config(config_path)
    runs: list[dict[str, object]] = []
    for raw_checkpoint in checkpoints:
        checkpoint = Path(raw_checkpoint).resolve()
        step = _checkpoint_step(checkpoint)
        count = config.model.n_samples if n_samples is None else n_samples
        child = output_root / f"step_{step:06d}_samples_{count:04d}"
        run_config = _evaluation_config(
            config,
            child,
            n_samples=count,
            ray_chunk_size=ray_chunk_size,
        )
        trainer = Trainer(run_config)
        metrics = trainer.evaluate(checkpoint)
        runs.append(
            {
                "checkpoint": str(checkpoint),
                "step": step,
                "n_samples": count,
                "ray_chunk_size": run_config.model.ray_chunk_size,
                "output_dir": str(child),
                "metrics": metrics,
                "report_sha256": _sha256(child / "evaluation_report.json"),
            }
        )
    report = {
        "schema_version": "mcss.checkpoint_sweep.v1",
        "source_config": {
            "path": str(config_path),
            "sha256": _sha256(config_path),
        },
        "runs": runs,
    }
    _atomic_json(report_path, report)
    return report


def select_pilot_scenes(
    root: str | Path,
    *,
    split: str,
    count: int,
) -> tuple[str, ...]:
    """Select a label-blind deterministic scene subset from manifest IDs only."""

    if not isinstance(split, str) or not split or split != split.strip():
        raise ValueError("split must be a non-empty trimmed string")
    if not isinstance(count, int) or count < 1:
        raise ValueError("count must be a positive integer")
    records = _pilot_manifest_records(Path(root), split=split)
    scene_ids = [str(record["scene_id"]) for record in records]
    if count > len(scene_ids):
        raise ValueError(f"requested {count} pilot scenes but only {len(scene_ids)} are available")
    return tuple(
        sorted(
            scene_ids,
            key=lambda scene_id: hashlib.sha256(
                f"v6-pilot:{split}:{scene_id}".encode()
            ).hexdigest(),
        )[:count]
    )


def write_pilot_artifacts(
    train_config_path: str | Path,
    validation_config_path: str | Path,
    *,
    selection_path: str | Path,
    train_config_output: str | Path,
    validation_config_output: str | Path,
    train_output_dir: str | Path,
    validation_output_dir: str | Path,
    train_count: int = 16,
    validation_count: int = 8,
    train_max_steps: int = 10_000,
    partition_registry: str | Path | None = None,
) -> dict[str, Any]:
    """Write immutable label-blind pilot selections and derived V5 configs."""

    train_config_path = Path(train_config_path).resolve()
    validation_config_path = Path(validation_config_path).resolve()
    selection_path = Path(selection_path).resolve()
    train_config_output = Path(train_config_output).resolve()
    validation_config_output = Path(validation_config_output).resolve()
    destinations = (selection_path, train_config_output, validation_config_output)
    for destination in destinations:
        _refuse_existing(destination)

    train_config = load_config(train_config_path)
    validation_config = load_config(validation_config_path)
    if train_config.dataset.root is None or validation_config.dataset.root is None:
        raise ValueError("pilot configs require prepared manifest dataset roots")
    train_root = Path(train_config.dataset.root).resolve()
    validation_root = Path(validation_config.dataset.root).resolve()
    train_records = _pilot_manifest_records(train_root, split="train")
    validation_records = _pilot_manifest_records(validation_root, split="val")
    train_ids = select_pilot_scenes(train_root, split="train", count=train_count)
    validation_ids = select_pilot_scenes(
        validation_root,
        split="val",
        count=validation_count,
    )
    partition_audit = _audit_pilot_partitions(
        partition_registry,
        train_ids=train_ids,
        validation_ids=validation_ids,
    )

    train_payload = _pilot_config_payload(
        train_config_path,
        root=train_config.dataset.root,
        scene_ids=train_ids,
        length=train_count,
        max_steps=train_max_steps,
        output_dir=train_output_dir,
    )
    validation_payload = _pilot_config_payload(
        validation_config_path,
        root=validation_config.dataset.root,
        scene_ids=validation_ids,
        length=validation_count,
        max_steps=1,
        output_dir=validation_output_dir,
    )
    _write_validated_yaml_configs(
        (
            (train_config_output, train_payload),
            (validation_config_output, validation_payload),
        )
    )

    report: dict[str, Any] = {
        "schema_version": "mcss.pilot_selection.v1",
        "selection_rule": "ascending sha256('v6-pilot:<split>:<scene_id>')",
        "label_blind": True,
        "partition_audit": partition_audit,
        "source_configs": {
            "train": {
                "path": str(train_config_path),
                "sha256": _sha256(train_config_path),
            },
            "validation": {
                "path": str(validation_config_path),
                "sha256": _sha256(validation_config_path),
            },
        },
        "splits": {
            "train": _pilot_split_report(
                train_root,
                train_records,
                train_ids,
                split_token="train",
            ),
            "validation": _pilot_split_report(
                validation_root,
                validation_records,
                validation_ids,
                split_token="val",
            ),
        },
        "generated_configs": {
            "train": {
                "path": str(train_config_output),
                "sha256": _sha256(train_config_output),
                "max_steps": train_max_steps,
                "output_dir": str(train_output_dir),
            },
            "validation": {
                "path": str(validation_config_output),
                "sha256": _sha256(validation_config_output),
                "max_steps": 1,
                "output_dir": str(validation_output_dir),
            },
        },
    }
    _atomic_json(selection_path, report)
    return report


def _audit_pilot_partitions(
    registry_path: str | Path | None,
    *,
    train_ids: Sequence[str],
    validation_ids: Sequence[str],
) -> dict[str, Any]:
    if registry_path is None:
        return {"performed": False, "passed": None}
    registry = Path(registry_path).resolve()
    if not registry.is_file():
        raise FileNotFoundError(registry)
    with registry.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"scene_name", "protocol_partition"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError("partition registry requires scene_name and protocol_partition")
        partitions: dict[str, str] = {}
        for row in reader:
            scene_id = row.get("scene_name")
            partition = row.get("protocol_partition")
            if not scene_id or not partition:
                raise ValueError("partition registry contains an incomplete row")
            if scene_id in partitions:
                raise ValueError(f"partition registry contains duplicate scene: {scene_id}")
            partitions[scene_id] = partition
    failures = [
        {"scene_id": scene_id, "expected": expected, "actual": partitions.get(scene_id)}
        for expected, scene_ids in (("train", train_ids), ("val", validation_ids))
        for scene_id in scene_ids
        if partitions.get(scene_id) != expected
    ]
    if failures:
        raise ValueError(f"pilot partition mismatch: {failures}")
    return {
        "performed": True,
        "passed": True,
        "registry_path": str(registry),
        "registry_sha256": _sha256(registry),
        "checked_scene_count": len(train_ids) + len(validation_ids),
        "allowed_partitions": {"train": "train", "validation": "val"},
    }


def _pilot_manifest_records(root: Path, *, split: str) -> list[dict[str, str]]:
    manifest_paths = _find_manifests(root)
    if not manifest_paths:
        raise FileNotFoundError(f"no manifest.json found below {root}")
    records: list[dict[str, str]] = []
    seen: set[str] = set()
    for manifest_path in manifest_paths:
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid manifest JSON: {manifest_path}") from error
        scene_id = payload.get("scene_id") if isinstance(payload, dict) else None
        if not isinstance(scene_id, str) or not scene_id:
            raise ValueError(f"manifest has no valid scene_id: {manifest_path}")
        if scene_id in seen:
            raise ValueError(f"duplicate scene_id discovered for pilot selection: {scene_id}")
        seen.add(scene_id)
        records.append(
            {
                "scene_id": scene_id,
                "manifest_path": str(manifest_path.resolve()),
                "manifest_sha256": _sha256(manifest_path),
                "selection_sha256": hashlib.sha256(
                    f"v6-pilot:{split}:{scene_id}".encode()
                ).hexdigest(),
            }
        )
    return records


def _pilot_config_payload(
    source_path: Path,
    *,
    root: str,
    scene_ids: tuple[str, ...],
    length: int,
    max_steps: int,
    output_dir: str | Path,
) -> dict[str, Any]:
    try:
        source = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise ValueError(f"invalid YAML in {source_path}") from error
    if not isinstance(source, dict):
        raise ValueError(f"config root must be a mapping: {source_path}")
    payload = copy.deepcopy(source)
    dataset = payload.get("dataset")
    training = payload.get("training")
    if not isinstance(dataset, dict) or not isinstance(training, dict):
        raise ValueError("pilot source config requires dataset and training mappings")
    dataset["root"] = root
    dataset["scene_ids"] = list(scene_ids)
    dataset["length"] = length
    training["max_steps"] = max_steps
    training["output_dir"] = str(output_dir)
    return payload


def _write_validated_yaml_configs(
    outputs: Sequence[tuple[Path, dict[str, Any]]],
) -> None:
    temporaries: list[tuple[Path, Path]] = []
    try:
        for output, payload in outputs:
            output.parent.mkdir(parents=True, exist_ok=True)
            temporary = output.with_name(f"{output.name}.{os.getpid()}.part")
            _refuse_existing(temporary)
            temporary.write_text(
                yaml.safe_dump(payload, sort_keys=False),
                encoding="utf-8",
            )
            load_config(temporary)
            temporaries.append((temporary, output))
        for temporary, output in temporaries:
            os.replace(temporary, output)
    finally:
        for temporary, _ in temporaries:
            temporary.unlink(missing_ok=True)


def _pilot_split_report(
    root: Path,
    records: Sequence[dict[str, str]],
    selected_ids: Sequence[str],
    *,
    split_token: str,
) -> dict[str, Any]:
    by_scene_id = {record["scene_id"]: record for record in records}
    return {
        "root": str(root),
        "split_token": split_token,
        "candidate_count": len(records),
        "selected_scene_ids": list(selected_ids),
        "selected_manifests": [by_scene_id[scene_id] for scene_id in selected_ids],
    }


def run_renderer_sweep(
    config_path: str | Path,
    checkpoint: str | Path,
    output_root: str | Path,
    *,
    sample_counts: Sequence[int] = (64, 128, 256),
    ray_chunk_size: int | None = None,
) -> dict[str, object]:
    """Evaluate one frozen state checkpoint with multiple fixed renderer sample counts."""

    config_path = Path(config_path).resolve()
    checkpoint = Path(checkpoint).resolve()
    output_root = Path(output_root).resolve()
    report_path = output_root / "renderer_sweep.json"
    _refuse_existing(report_path)
    counts = tuple(int(value) for value in sample_counts)
    if not counts or len(set(counts)) != len(counts) or any(value < 2 for value in counts):
        raise ValueError("sample_counts must contain unique values of at least two")
    if ray_chunk_size is not None and ray_chunk_size < 1:
        raise ValueError("ray_chunk_size must be positive")
    config = load_config(config_path)

    runs: list[dict[str, object]] = []
    for count in counts:
        child = output_root / f"samples_{count:04d}"
        run_config = _evaluation_config(
            config,
            child,
            n_samples=count,
            ray_chunk_size=ray_chunk_size,
        )
        started = time.perf_counter()
        trainer = Trainer(run_config)
        metrics = trainer.evaluate(checkpoint)
        evaluation_report = json.loads(
            (child / "evaluation_report.json").read_text(encoding="utf-8")
        )
        runs.append(
            {
                "n_samples": count,
                "ray_chunk_size": run_config.model.ray_chunk_size,
                "output_dir": str(child),
                "runtime_seconds": time.perf_counter() - started,
                "metrics": metrics,
                "evaluation": evaluation_report["evaluation"],
                "report_sha256": _sha256(child / "evaluation_report.json"),
            }
        )
    report = {
        "schema_version": "mcss.renderer_sweep.v1",
        "source": _source_metadata(config_path, checkpoint),
        "runs": runs,
    }
    _atomic_json(report_path, report)
    return report


@torch.no_grad()
def run_surface_audit(
    config_path: str | Path,
    checkpoint: str | Path,
    output_dir: str | Path,
    *,
    n_samples: int = 256,
    windows_per_scene: int | None = None,
) -> dict[str, object]:
    """Compare frozen V5 evidence with target-visible depth after every model forward."""

    config_path = Path(config_path).resolve()
    checkpoint = Path(checkpoint).resolve()
    output_dir = Path(output_dir).resolve()
    report_path = output_dir / "evidence_audit.json"
    _refuse_existing(report_path)
    if n_samples < 2:
        raise ValueError("n_samples must be at least two")
    config = load_config(config_path)
    trainer = Trainer(
        replace(config, training=replace(config.training, output_dir=str(output_dir)))
    )
    trainer.load_checkpoint(checkpoint, load_optimizer=False)
    trainer.model.eval()

    if windows_per_scene is None:
        audit_indices = tuple(range(len(trainer.dataset)))
        window_selection: dict[str, object] = {
            "rule": "all_dataset_windows",
            "windows_per_scene": None,
            "candidate_count": len(trainer.dataset),
            "selected_count": len(audit_indices),
        }
    else:
        if not isinstance(trainer.dataset, ManifestSceneDataset):
            raise ValueError("bounded per-scene audit selection requires a manifest dataset")
        candidates = [
            (trainer.dataset.manifests[manifest_index].scene_id, start)
            for manifest_index, start in trainer.dataset.entries
        ]
        audit_indices = select_audit_windows(
            candidates,
            windows_per_scene=windows_per_scene,
        )
        window_selection = {
            "rule": "lowest_sha256(v6-surface-audit:<scene_id>:<window_start>)",
            "windows_per_scene": windows_per_scene,
            "candidate_count": len(candidates),
            "selected_count": len(audit_indices),
            "selected": [
                {
                    "dataset_index": index,
                    "scene_id": candidates[index][0],
                    "window_start": candidates[index][1],
                    "token_sha256": hashlib.sha256(
                        f"v6-surface-audit:{candidates[index][0]}:{candidates[index][1]}".encode()
                    ).hexdigest(),
                }
                for index in audit_indices
            ],
        }

    scene_records: dict[str, list[dict[str, object]]] = {}
    sample_count = 0
    started = time.perf_counter()
    for index in audit_indices:
        batch = collate_scene_examples([trainer.dataset[index]]).to(trainer.device)
        with torch.autocast(
            device_type=trainer.device.type,
            dtype=trainer.amp_dtype,
            enabled=trainer.amp_enabled,
        ):
            output = trainer.model(
                batch.context_rgb,
                batch.context_cameras,
                batch.target_cameras,
                batch.bounds,
            )
        if batch.target_depth is None:
            raise ValueError("surface audit requires target depth after model forward")
        record = audit_evidence(
            output.state,
            batch.target_cameras,
            batch.target_depth,
            batch.target_visibility,
            predicted_depth=output.predictions["depth"],
            n_samples=n_samples,
        )
        scene_id = batch.scene_ids[0] or f"__sample_{sample_count:06d}"
        scene_records.setdefault(scene_id, []).append(record)
        sample_count += 1

    per_scene = {
        scene_id: _summarize_scene(records)
        for scene_id, records in sorted(scene_records.items())
    }
    report = {
        "schema_version": "mcss.phase0_surface_audit.v1",
        "diagnostic_only": True,
        "target_labels_used_post_forward": True,
        "checkpoint_selection_allowed": False,
        "source": _source_metadata(config_path, checkpoint),
        "resolved_config": asdict(config),
        "evaluation": {
            "sample_count": sample_count,
            "scene_count": len(per_scene),
            "n_samples_per_ray": n_samples,
            "runtime_seconds": time.perf_counter() - started,
        },
        "window_selection": window_selection,
        "scene_macro": _scene_macro(per_scene),
        "per_scene": per_scene,
    }
    _atomic_json(report_path, report)
    return report


def select_audit_windows(
    candidates: Sequence[tuple[str, int]],
    *,
    windows_per_scene: int,
) -> tuple[int, ...]:
    """Select a fixed label-blind number of window starts per scene."""

    if windows_per_scene < 1:
        raise ValueError("windows_per_scene must be positive")
    grouped: dict[str, list[tuple[str, int]]] = {}
    for index, (scene_id, window_start) in enumerate(candidates):
        if not scene_id or window_start < 0:
            raise ValueError("audit candidates require a scene id and non-negative start")
        token = f"v6-surface-audit:{scene_id}:{window_start}"
        grouped.setdefault(scene_id, []).append(
            (hashlib.sha256(token.encode()).hexdigest(), index)
        )
    selected: list[int] = []
    for scene_id in sorted(grouped):
        ranked = grouped[scene_id]
        selected.extend(index for _, index in sorted(ranked)[:windows_per_scene])
    return tuple(sorted(selected))


def _evaluation_config(
    config: ExperimentConfig,
    output_dir: Path,
    *,
    n_samples: int,
    ray_chunk_size: int | None,
) -> ExperimentConfig:
    for name in ("evaluation.json", "evaluation_report.json"):
        _refuse_existing(output_dir / name)
    model = replace(
        config.model,
        n_samples=n_samples,
        ray_chunk_size=(
            config.model.ray_chunk_size if ray_chunk_size is None else ray_chunk_size
        ),
    )
    return replace(
        config,
        model=model,
        training=replace(config.training, output_dir=str(output_dir)),
    )


def _summarize_scene(records: Sequence[dict[str, object]]) -> dict[str, object]:
    if not records:
        raise ValueError("scene summary requires at least one audit record")
    comparisons: dict[str, dict[str, object]] = {}
    for comparison in ("surface_vs_free", "surface_vs_behind"):
        values: dict[str, list[float]] = {
            "auroc": [],
            "auprc": [],
            "brier": [],
            "ece": [],
        }
        comparison_records: list[dict[str, object]] = []
        for record in records:
            raw_comparisons = record["comparisons"]
            if not isinstance(raw_comparisons, dict):
                raise TypeError("audit comparisons must be a mapping")
            metrics = raw_comparisons[comparison]
            if not isinstance(metrics, dict):
                raise TypeError("audit comparison metrics must be a mapping")
            comparison_records.append(metrics)
            for name in values:
                value = metrics.get(name)
                if isinstance(value, (int, float)) and math.isfinite(float(value)):
                    values[name].append(float(value))
        comparisons[comparison] = {
            **{
                name: (sum(items) / len(items) if items else None)
                for name, items in values.items()
            },
            "count": sum(int(item["count"]) for item in comparison_records),
            "positive_count": sum(
                int(item["positive_count"]) for item in comparison_records
            ),
            "negative_count": sum(
                int(item["negative_count"]) for item in comparison_records
            ),
            "reliability": _aggregate_reliability(comparison_records),
        }
    return {
        "sample_count": len(records),
        "evidence_field": _common_value(records, "evidence_field"),
        "label_semantics": _common_value(records, "label_semantics"),
        "n_samples": _common_value(records, "n_samples"),
        "surface_band_m": _common_or_values(records, "surface_band_m"),
        "bins": {
            name: _aggregate_bin(records, name) for name in ("free", "surface", "behind")
        },
        "comparisons": comparisons,
        "error_by_risk": _aggregate_error_by_risk(records),
        "ray_counts": {
            name: sum(int(record["rays"][name]) for record in records)  # type: ignore[index]
            for name in (
                "total",
                "valid",
                "invalid_target_depth",
                "target_invisible",
                "missed_state_bounds",
            )
        },
        "windows": list(records),
    }


def _aggregate_error_by_risk(
    records: Sequence[dict[str, object]],
) -> dict[str, object]:
    reports = [record.get("error_by_risk") for record in records]
    if not all(isinstance(report, dict) for report in reports):
        raise TypeError("surface audit records require error_by_risk mappings")
    typed_reports: list[dict[str, object]] = reports  # type: ignore[assignment]
    decile_sets = [report["deciles"] for report in typed_reports]
    if not all(isinstance(deciles, list) and len(deciles) == 10 for deciles in decile_sets):
        raise TypeError("error_by_risk requires ten deciles")
    aggregate_deciles: list[dict[str, object]] = []
    for index in range(10):
        items = [deciles[index] for deciles in decile_sets]
        if not all(isinstance(item, dict) for item in items):
            raise TypeError("error_by_risk deciles must be mappings")
        count = sum(int(item["count"]) for item in items)

        aggregate_deciles.append(
            {
                "index": index,
                "count": count,
                "mean_risk": _weighted_decile_value(items, count, "mean_risk"),
                "mean_absolute_error_m": _weighted_decile_value(
                    items, count, "mean_absolute_error_m"
                ),
                "mean_absolute_relative_error": _weighted_decile_value(
                    items,
                    count,
                    "mean_absolute_relative_error"
                ),
            }
        )
    absolute = [
        float(item["mean_absolute_error_m"])
        for item in aggregate_deciles
        if item["mean_absolute_error_m"] is not None
    ]
    relative = [
        float(item["mean_absolute_relative_error"])
        for item in aggregate_deciles
        if item["mean_absolute_relative_error"] is not None
    ]
    return {
        "count": sum(int(report["count"]) for report in typed_reports),
        "risk_semantics": _common_value(typed_reports, "risk_semantics"),
        "deciles": aggregate_deciles,
        "absolute_error_monotonic_violations": _monotonic_violations(absolute),
        "absolute_relative_error_monotonic_violations": _monotonic_violations(relative),
        "absolute_error_spearman": _ordered_spearman(absolute),
        "absolute_relative_error_spearman": _ordered_spearman(relative),
        "per_window": typed_reports,
    }


def _weighted_decile_value(
    items: Sequence[dict[str, object]],
    count: int,
    name: str,
) -> float | None:
    if count == 0:
        return None
    return sum(
        int(item["count"]) * float(item[name])
        for item in items
        if item[name] is not None
    ) / count


def _aggregate_bin(
    records: Sequence[dict[str, object]],
    name: str,
) -> dict[str, object]:
    entries: list[dict[str, object]] = []
    for record in records:
        bins = record["bins"]
        if not isinstance(bins, dict) or not isinstance(bins.get(name), dict):
            raise TypeError("audit bins must contain mappings for every depth region")
        entries.append(bins[name])
    count = sum(int(entry["count"]) for entry in entries)
    if count == 0:
        mean = None
        std = None
    else:
        weighted_sum = sum(
            int(entry["count"]) * float(entry["mean"])
            for entry in entries
            if entry["mean"] is not None
        )
        mean = weighted_sum / count
        second_moment = sum(
            int(entry["count"])
            * (float(entry["std"]) ** 2 + float(entry["mean"]) ** 2)
            for entry in entries
            if entry["mean"] is not None and entry["std"] is not None
        ) / count
        std = math.sqrt(max(0.0, second_moment - mean**2))
    return {
        "count": count,
        "mean": mean,
        "std": std,
        "per_window_deciles": [entry["deciles"] for entry in entries],
    }


def _aggregate_reliability(
    comparisons: Sequence[dict[str, object]],
) -> list[dict[str, object]]:
    reliability_sets = [item["reliability"] for item in comparisons]
    if not all(isinstance(items, list) and len(items) == 10 for items in reliability_sets):
        raise TypeError("audit reliability must contain ten bins")
    aggregate: list[dict[str, object]] = []
    for index in range(10):
        items = [reliability[index] for reliability in reliability_sets]
        if not all(isinstance(item, dict) for item in items):
            raise TypeError("audit reliability entries must be mappings")
        count = sum(int(item["count"]) for item in items)
        mean_confidence = None
        positive_rate = None
        if count:
            mean_confidence = sum(
                int(item["count"]) * float(item["mean_confidence"])
                for item in items
                if item["mean_confidence"] is not None
            ) / count
            positive_rate = sum(
                int(item["count"]) * float(item["positive_rate"])
                for item in items
                if item["positive_rate"] is not None
            ) / count
        aggregate.append(
            {
                "lower": items[0]["lower"],
                "upper": items[0]["upper"],
                "count": count,
                "mean_confidence": mean_confidence,
                "positive_rate": positive_rate,
            }
        )
    return aggregate


def _common_value(records: Sequence[dict[str, object]], name: str) -> object:
    first = records[0][name]
    if any(record[name] != first for record in records[1:]):
        raise ValueError(f"audit records disagree on {name}")
    return first


def _common_or_values(records: Sequence[dict[str, object]], name: str) -> object:
    values = [record[name] for record in records]
    return values[0] if all(value == values[0] for value in values[1:]) else values


def _scene_macro(per_scene: dict[str, dict[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for comparison in ("surface_vs_free", "surface_vs_behind"):
        metric_summary: dict[str, object] = {}
        for metric in ("auroc", "auprc", "brier", "ece"):
            values = []
            for scene in per_scene.values():
                comparisons = scene["comparisons"]
                if not isinstance(comparisons, dict):
                    raise TypeError("scene comparisons must be a mapping")
                item = comparisons[comparison]
                if not isinstance(item, dict):
                    raise TypeError("scene comparison must be a mapping")
                value = item.get(metric)
                if isinstance(value, (int, float)) and math.isfinite(float(value)):
                    values.append(float(value))
            metric_summary[metric] = _mean_and_bootstrap(values)
        result[comparison] = metric_summary
    risk_summary: dict[str, object] = {}
    for metric in (
        "absolute_error_monotonic_violations",
        "absolute_relative_error_monotonic_violations",
        "absolute_error_spearman",
        "absolute_relative_error_spearman",
    ):
        values = []
        for scene in per_scene.values():
            report = scene["error_by_risk"]
            if not isinstance(report, dict):
                raise TypeError("scene error_by_risk must be a mapping")
            value = report.get(metric)
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                values.append(float(value))
        risk_summary[metric] = _mean_and_bootstrap(values)
    result["error_by_risk"] = risk_summary
    return result


def _monotonic_violations(values: Sequence[float]) -> int:
    return sum(right + 1e-8 < left for left, right in zip(values, values[1:], strict=False))


def _ordered_spearman(values: Sequence[float]) -> float | None:
    if len(values) < 2:
        return None
    tensor = torch.tensor(values, dtype=torch.float64)
    unique, inverse, counts = torch.unique(
        tensor, sorted=True, return_inverse=True, return_counts=True
    )
    del unique
    cumulative = counts.cumsum(dim=0)
    rank_start = cumulative - counts
    ranks = ((rank_start + cumulative - 1).to(torch.float64) * 0.5)[inverse]
    ordered = torch.arange(tensor.numel(), dtype=torch.float64)
    ordered -= ordered.mean()
    ranks -= ranks.mean()
    denominator = torch.linalg.vector_norm(ordered) * torch.linalg.vector_norm(ranks)
    if denominator <= 0:
        return None
    return float((ordered * ranks).sum() / denominator)


def _mean_and_bootstrap(values: Sequence[float]) -> dict[str, object]:
    if not values:
        return {"scene_count": 0, "mean": None, "ci95": None}
    tensor = torch.tensor(values, dtype=torch.float64)
    if tensor.numel() == 1:
        value = float(tensor[0])
        return {"scene_count": 1, "mean": value, "ci95": [value, value]}
    generator = torch.Generator().manual_seed(17)
    indices = torch.randint(
        tensor.numel(),
        (2000, tensor.numel()),
        generator=generator,
    )
    bootstrap = tensor[indices].mean(dim=1)
    interval = torch.quantile(bootstrap, torch.tensor([0.025, 0.975], dtype=torch.float64))
    return {
        "scene_count": tensor.numel(),
        "mean": float(tensor.mean()),
        "ci95": [float(interval[0]), float(interval[1])],
        "bootstrap_seed": 17,
        "bootstrap_resamples": 2000,
    }


def _source_metadata(config_path: Path, checkpoint: Path) -> dict[str, object]:
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    return {
        "config_path": str(config_path),
        "config_sha256": _sha256(config_path),
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
    }


def _checkpoint_step(path: Path) -> int:
    match = re.search(r"step_(\d+)", path.stem)
    if match is None:
        raise ValueError(f"checkpoint filename must contain step_N: {path}")
    return int(match.group(1))


def _refuse_existing(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing Phase 0 artifact: {path}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


if __name__ == "__main__":
    raise SystemExit(main())
