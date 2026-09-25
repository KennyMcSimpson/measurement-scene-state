"""Evaluate one strict dynamic checkpoint over sealed development episodes."""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from time import perf_counter
from typing import Any

import torch

from mcss.data.episodes import OnlineEpisodeSource
from mcss.data.pilot_provenance import (
    DEFAULT_PREPARED_ROOT,
    PilotEntryProvenance,
    validate_pilot_entries,
)
from mcss.dynamic.checkpoint import SCHEMA_VERSION, load_dynamic_checkpoint
from mcss.dynamic.policy import (
    FEATURE_SCHEMA_VERSION,
    FixedPolicy,
    LearnedActionPolicy,
    ResidualThresholdPolicy,
)
from mcss.dynamic.runner import StreamingRunner
from mcss.dynamic.types import Action, hash_value
from mcss.evaluation.sealed_queries import evaluate_sealed
from mcss.measurements import FixedMeasurementRenderer

EPISODE_INDEX_SCHEMA = "mcss.dynamic.episode_index.v1"
DEFAULT_POLICIES = ("OFF", "FUSE", "COMPLETE", "ALL")
POLICY_NAMES = frozenset((*DEFAULT_POLICIES, "RESIDUAL_THRESHOLD", "LEARNED"))
DEFAULT_RAY_CHUNK_SIZE = 2048


def load_episode_index(path: str | Path) -> list[dict[str, Any]]:
    """Load the compiler-owned dynamic episode index without opening observations."""

    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid dynamic episode index JSON: {source}") from error
    if not isinstance(payload, dict) or payload.get("schema_version") != EPISODE_INDEX_SCHEMA:
        raise ValueError(f"expected dynamic episode index schema {EPISODE_INDEX_SCHEMA!r}")
    if set(payload) != {"schema_version", "episodes"}:
        raise ValueError("dynamic episode index fields mismatch")
    episodes = payload["episodes"]
    if not isinstance(episodes, list) or not episodes or not all(
        isinstance(entry, dict) for entry in episodes
    ):
        raise ValueError("dynamic episode index requires a non-empty list of objects")
    return episodes


def evaluate_dynamic_checkpoint(
    entries: Sequence[Mapping[str, Any]],
    checkpoint_path: str | Path,
    output_dir: str | Path,
    *,
    device: str | torch.device = "cpu",
    policies: Sequence[str] = DEFAULT_POLICIES,
    render_samples: int = 16,
    ray_chunk_size: int = DEFAULT_RAY_CHUNK_SIZE,
    residual_threshold: float = 0.03,
    max_episodes: int | None = None,
    prepared_root: str | Path = DEFAULT_PREPARED_ROOT,
    max_units: float = 1e12,
    policy_checkpoint: str | Path | None = None,
    warmup_count: int = 4,
    stream_count: int = 8,
    query_count: int = 4,
) -> dict[str, Any]:
    """Run a checkpoint on selected scenes and persist only reproducible report artifacts."""

    selected_entries = _select_entries(entries, max_episodes)
    selected_policies = _validate_policies(policies, residual_threshold)
    if type(render_samples) is not int or render_samples < 2:
        raise ValueError("render_samples must be an integer of at least two")
    if type(ray_chunk_size) is not int or ray_chunk_size < 1:
        raise ValueError("ray_chunk_size must be a positive integer")
    _validate_protocol_counts(warmup_count, stream_count, query_count)
    if "LEARNED" in selected_policies and policy_checkpoint is None:
        raise ValueError("LEARNED policy requires an explicit policy_checkpoint")
    if "LEARNED" not in selected_policies and policy_checkpoint is not None:
        raise ValueError("policy_checkpoint is only valid when LEARNED is selected")
    output = Path(output_dir)
    if (output / "report.json").exists():
        raise FileExistsError("refusing to overwrite a completed checkpoint evaluation report")

    # This completes all JSON/hash/path checks before any online RGB or query data is decoded.
    _require_dev_entries(selected_entries)
    provenances = validate_pilot_entries(selected_entries, prepared_root=prepared_root)
    _validate_dev_protocol(
        selected_entries,
        provenances,
        warmup_count=warmup_count,
        stream_count=stream_count,
        query_count=query_count,
    )
    target_device = _resolve_device(device)
    base_carrier, base_write_rule, checkpoint = load_dynamic_checkpoint(
        checkpoint_path, target_device
    )
    if checkpoint["schema_version"] != SCHEMA_VERSION:
        raise RuntimeError("dynamic checkpoint loader returned an unexpected schema")
    expected_state_hash = hash_value(
        {"carrier": base_carrier.state_dict(), "write_rule": base_write_rule.state_dict()}
    )
    learned_policy, policy_metadata, policy_binding = _load_learned_policy(
        policy_checkpoint,
        selected_policies=selected_policies,
        expected_state_hash=expected_state_hash,
        render_samples=render_samples,
        ray_chunk_size=ray_chunk_size,
        max_units=max_units,
        device=target_device,
    )
    policy_file_sha256 = _policy_file_sha256(policy_checkpoint)
    policy_content_hash = _policy_content_hash(policy_metadata)

    output.mkdir(parents=True, exist_ok=True)
    actual_subset = {
        "requested_max_episodes": max_episodes,
        "selected_episodes": len(selected_entries),
        "selected_episode_ids": [str(entry["episode_id"]) for entry in selected_entries],
    }
    resolved = {
        "checkpoint": checkpoint,
        "device": str(target_device),
        "policies": list(selected_policies),
        "render_samples": render_samples,
        "ray_chunk_size": ray_chunk_size,
        "residual_threshold": residual_threshold,
        "max_units": max_units,
        "protocol": {
            "warmup_count": warmup_count,
            "stream_count": stream_count,
            "query_count": query_count,
        },
        "policy_checkpoint": None if policy_checkpoint is None else str(Path(policy_checkpoint)),
        "policy_file_sha256": policy_file_sha256,
        "policy_content_hash": policy_content_hash,
        "policy_binding": policy_binding,
        "actual_subset": actual_subset,
        "manifest_provenance": [item.to_dict() for item in provenances],
        "work_accounting": "declared operator-work proxy; not measured FLOPs",
    }
    _write_json(output / "resolved_config.json", resolved)

    records: list[dict[str, Any]] = []
    for entry, provenance in zip(selected_entries, provenances, strict=True):
        for policy_name in selected_policies:
            record = _evaluate_entry(
                entry,
                provenance,
                base_carrier=base_carrier,
                base_write_rule=base_write_rule,
                checkpoint=checkpoint,
                expected_state_hash=expected_state_hash,
                device=target_device,
                policy_name=policy_name,
                render_samples=render_samples,
                ray_chunk_size=ray_chunk_size,
                residual_threshold=residual_threshold,
                max_units=max_units,
                learned_policy=learned_policy,
                policy_metadata=policy_metadata,
                policy_file_sha256=policy_file_sha256,
                policy_content_hash=policy_content_hash,
                policy_binding=policy_binding,
            )
            path = output / f"{record['episode_id']}_{policy_name}.json"
            _write_json_exclusive(path, record)
            records.append(record)
            print(
                f"Completed {record['scene_id']} {policy_name}: "
                f"{record['total_seconds']:.2f}s total",
                flush=True,
            )

    summaries, scene_results, metric_names = aggregate_scene_records(records)
    _write_summary_csv(output / "summary.csv", summaries, metric_names)
    report = {
        "schema": "mcss.dynamic.checkpoint_evaluation.v1",
        "status": "complete",
        "checkpoint": checkpoint,
        "model_state_hash": expected_state_hash,
        "actual_subset": actual_subset,
        "policies": list(selected_policies),
        "render_samples": render_samples,
        "ray_chunk_size": ray_chunk_size,
        "protocol": {
            "warmup_count": warmup_count,
            "stream_count": stream_count,
            "query_count": query_count,
        },
        "policy_checkpoint": None if policy_checkpoint is None else str(Path(policy_checkpoint)),
        "policy_file_sha256": policy_file_sha256,
        "policy_content_hash": policy_content_hash,
        "policy_binding": policy_binding,
        "work_accounting": "declared operator-work proxy; not measured FLOPs",
        "metric_keys": list(metric_names),
        "manifest_provenance": [item.to_dict() for item in provenances],
        "records": records,
        "scene_results": scene_results,
        "summaries": summaries,
    }
    report = _json_safe(report)
    _write_json_exclusive(output / "report.json", report)
    return report


def aggregate_scene_records(
    records: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], tuple[str, ...]]:
    """Aggregate sealed-query averages at scene level and pair each policy with OFF."""

    if not records:
        raise ValueError("checkpoint evaluation requires at least one record")
    grouped: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    metric_names: set[str] = set()
    for record in records:
        split_id = _record_identifier(record, "split_id")
        scene_id = _record_identifier(record, "scene_id")
        policy = _record_identifier(record, "policy")
        averages = _record_averages(record)
        metric_names.update(averages)
        grouped[(split_id, scene_id, policy)].append(record)
    ordered_metrics = tuple(sorted(metric_names))

    scene_results: list[dict[str, Any]] = []
    for (split_id, scene_id, policy), members in sorted(grouped.items()):
        scene_results.append(
            {
                "split_id": split_id,
                "scene_id": scene_id,
                "policy": policy,
                "episode_ids": sorted(
                    _record_identifier(record, "episode_id") for record in members
                ),
                "record_count": len(members),
                "metrics_mean": {
                    name: _mean_finite(_record_averages(record).get(name) for record in members)
                    for name in ordered_metrics
                },
                "metric_valid_record_counts": {
                    name: _finite_count(_record_averages(record).get(name) for record in members)
                    for name in ordered_metrics
                },
            }
        )

    off_by_scene = {
        (item["split_id"], item["scene_id"]): item
        for item in scene_results
        if item["policy"] == "OFF"
    }
    if len(off_by_scene) != len({(item["split_id"], item["scene_id"]) for item in scene_results}):
        raise ValueError("scene aggregation requires an OFF record for every selected scene")
    for item in scene_results:
        off_metrics = off_by_scene[(item["split_id"], item["scene_id"])]["metrics_mean"]
        item["paired_delta_vs_off"] = {
            name: _paired_delta(item["metrics_mean"][name], off_metrics[name])
            for name in ordered_metrics
        }

    by_split_policy: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for item in scene_results:
        by_split_policy[(item["split_id"], item["policy"])].append(item)
    summaries: list[dict[str, Any]] = []
    for (split_id, policy), members in sorted(by_split_policy.items()):
        summaries.append(
            {
                "split_id": split_id,
                "policy": policy,
                "scene_count": len(members),
                "scene_ids": sorted(str(item["scene_id"]) for item in members),
                "metrics_mean": {
                    name: _mean_finite(item["metrics_mean"][name] for item in members)
                    for name in ordered_metrics
                },
                "metric_valid_scene_counts": {
                    name: _finite_count(item["metrics_mean"][name] for item in members)
                    for name in ordered_metrics
                },
                "paired_delta_vs_off_mean": {
                    name: _mean_finite(item["paired_delta_vs_off"][name] for item in members)
                    for name in ordered_metrics
                },
            }
        )
    return summaries, scene_results, ordered_metrics


def _evaluate_entry(
    entry: Mapping[str, Any],
    provenance: PilotEntryProvenance,
    *,
    base_carrier: torch.nn.Module,
    base_write_rule: torch.nn.Module,
    checkpoint: Mapping[str, Any],
    expected_state_hash: str,
    device: torch.device,
    policy_name: str,
    render_samples: int,
    ray_chunk_size: int,
    residual_threshold: float,
    max_units: float,
    learned_policy: LearnedActionPolicy | None,
    policy_metadata: Mapping[str, Any] | None,
    policy_file_sha256: str | None,
    policy_content_hash: str | None,
    policy_binding: Mapping[str, Any] | None,
) -> dict[str, Any]:
    source = OnlineEpisodeSource.from_file(str(entry["online_manifest"]), device=device)
    _validate_source(source, entry, provenance)
    policy = _make_policy(policy_name, residual_threshold, learned_policy)
    renderer = FixedMeasurementRenderer(
        n_samples=render_samples,
        ray_chunk_size=ray_chunk_size,
    )
    runner = StreamingRunner(
        copy.deepcopy(base_carrier),
        copy.deepcopy(base_write_rule),
        renderer,
        policy,
        max_units=max_units,
    )
    if runner.checkpoint_hash != expected_state_hash:
        raise RuntimeError("policy trajectory did not start from the loaded checkpoint state")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    warmup = source.warmup()
    started = perf_counter()
    runner.reset(
        warmup,
        episode_id=source.episode_id,
        scene_id=source.scene_id,
        split_id=source.split_id,
        query_vault_id=source.query_vault_id,
        stream_steps=source.remaining_steps,
        query_count=int(entry["query_count"]),
    )
    while source.remaining_steps:
        runner.step(source.next_observation())
    actual_actions = [str(item["action"]) for item in runner.history]
    _require_fixed_policy_actions(
        policy_name,
        actual_actions,
        expected_count=int(entry["stream_steps"]),
    )
    sealed = runner.seal()
    if sealed.checkpoint_hash != expected_state_hash:
        raise RuntimeError("sealed scene does not identify the loaded checkpoint state")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    adaptation_seconds = perf_counter() - started

    evaluation_started = perf_counter()
    metrics = evaluate_sealed(sealed, str(entry["query_manifest"]), renderer, device=device)
    if len(metrics["per_query"]) != int(entry["query_count"]):
        raise ValueError("sealed evaluation query count differs from the declared episode protocol")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    evaluation_seconds = perf_counter() - evaluation_started
    for _ in range(int(entry["query_count"])):
        runner.budget.record("query_render", runner.budget.render_units)
    total_seconds = adaptation_seconds + evaluation_seconds

    return {
        "episode_id": source.episode_id,
        "scene_id": source.scene_id,
        "split_id": source.split_id,
        "policy": policy_name,
        "requested_policy": policy_name,
        "trained_policy": policy_name == "LEARNED",
        "checkpoint_hash": sealed.checkpoint_hash,
        "checkpoint_file_sha256": checkpoint["sha256"],
        "policy_checkpoint_file_sha256": policy_file_sha256,
        "policy_file_sha256": policy_file_sha256,
        "policy_content_hash": policy_content_hash,
        "policy_binding": None if policy_binding is None else copy.deepcopy(policy_binding),
        "policy_metadata": None if policy_metadata is None else copy.deepcopy(policy_metadata),
        "render_protocol": {
            "n_samples": render_samples,
            "ray_chunk_size": ray_chunk_size,
        },
        "budget_protocol": {
            "max_units": float(max_units),
            "units_kind": "declared_operator_work_proxy_not_measured_flops",
        },
        "sealed_state_hash": sealed.state_hash,
        "fast_state_hash": sealed.fast_state_hash,
        "observed_ids": list(sealed.observed_ids),
        "manifest_provenance": provenance.to_dict(),
        "actions": actual_actions,
        "actual_actions": actual_actions,
        "adaptation_seconds": adaptation_seconds,
        "evaluation_seconds": evaluation_seconds,
        "total_seconds": total_seconds,
        "peak_cuda_allocated_bytes": (
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
        ),
        "metrics": metrics,
        "budget": runner.budget.report(),
        "history": copy.deepcopy(runner.history),
    }


def _select_entries(
    entries: Sequence[Mapping[str, Any]], max_episodes: int | None
) -> tuple[Mapping[str, Any], ...]:
    if not entries:
        raise ValueError("checkpoint evaluation requires at least one episode entry")
    if max_episodes is not None and (type(max_episodes) is not int or max_episodes < 1):
        raise ValueError("max_episodes must be a positive integer when specified")
    selected = tuple(entries if max_episodes is None else entries[:max_episodes])
    if not selected:
        raise ValueError("max_episodes selected no episode entries")
    return selected


def _require_dev_entries(entries: Sequence[Mapping[str, Any]]) -> None:
    non_dev = [
        entry.get("episode_id", "<unknown>")
        for entry in entries
        if entry.get("split_id") != "dev"
    ]
    if non_dev:
        raise ValueError(f"checkpoint evaluation accepts only dev entries; found {non_dev}")


def _validate_dev_protocol(
    entries: Sequence[Mapping[str, Any]],
    provenances: Sequence[PilotEntryProvenance],
    *,
    warmup_count: int,
    stream_count: int,
    query_count: int,
) -> None:
    scene_ids = [item.scene_id for item in provenances]
    if len(scene_ids) != len(set(scene_ids)):
        raise ValueError("checkpoint evaluation requires unique dev scene_id values")
    for entry in entries:
        online = _read_validated_manifest(entry["online_manifest"], "online")
        if (
            len(online["warmup"]) != warmup_count
            or len(online["stream"]) != stream_count
            or entry["stream_steps"] != stream_count
            or entry["query_count"] != query_count
        ):
            raise ValueError(
                "checkpoint evaluation requires exactly "
                f"warmup={warmup_count}, stream={stream_count}, and query={query_count}"
            )


def _read_validated_manifest(path: Any, kind: str) -> Mapping[str, Any]:
    if not isinstance(path, str):
        raise ValueError(f"validated {kind} manifest path must be a string")
    source = Path(path)
    try:
        manifest = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid validated {kind} manifest JSON: {source}") from error
    if not isinstance(manifest, Mapping):
        raise ValueError(f"validated {kind} manifest must be an object")
    return manifest


def _validate_policies(policies: Sequence[str], residual_threshold: float) -> tuple[str, ...]:
    selected = tuple(policies)
    if not selected or not all(
        isinstance(name, str) and name in POLICY_NAMES for name in selected
    ):
        raise ValueError(f"policies must be selected from {sorted(POLICY_NAMES)}")
    if len(selected) != len(set(selected)):
        raise ValueError("policies must be unique")
    if "OFF" not in selected:
        raise ValueError("policies must include OFF for matched paired deltas")
    if (
        not isinstance(residual_threshold, (int, float))
        or not math.isfinite(residual_threshold)
        or residual_threshold < 0
    ):
        raise ValueError("residual_threshold must be non-negative")
    return selected


def _resolve_device(value: str | torch.device) -> torch.device:
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def _validate_protocol_counts(warmup_count: int, stream_count: int, query_count: int) -> None:
    for name, value in (
        ("warmup_count", warmup_count),
        ("stream_count", stream_count),
        ("query_count", query_count),
    ):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")


def _validate_source(
    source: OnlineEpisodeSource, entry: Mapping[str, Any], provenance: PilotEntryProvenance
) -> None:
    for name in ("episode_id", "scene_id", "split_id", "query_vault_id"):
        source_value = getattr(source, name)
        if source_value != entry[name] or source_value != getattr(provenance, name):
            raise ValueError(
                f"online source identity disagrees with validated provenance for {name}"
            )
    if source.remaining_steps != entry["stream_steps"]:
        raise ValueError("online source stream length disagrees with the selected index entry")


def _load_learned_policy(
    path: str | Path | None,
    *,
    selected_policies: Sequence[str],
    expected_state_hash: str,
    render_samples: int,
    ray_chunk_size: int,
    max_units: float,
    device: torch.device,
) -> tuple[
    LearnedActionPolicy | None,
    Mapping[str, Any] | None,
    Mapping[str, Any] | None,
]:
    if "LEARNED" not in selected_policies:
        return None, None, None
    if path is None:
        raise ValueError("LEARNED policy requires an explicit policy_checkpoint")
    from mcss.dynamic.policy_checkpoint import load_policy_checkpoint

    expected_binding = {
        "model_content_hash": expected_state_hash,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "render_protocol": {
            "renderer": "FixedMeasurementRenderer",
            "renderer_samples": render_samples,
            "ray_chunk_size": ray_chunk_size,
        },
    }
    policy, metadata = load_policy_checkpoint(
        path,
        expected_binding=expected_binding,
        device=device,
    )
    if not isinstance(policy, LearnedActionPolicy):
        raise TypeError("policy checkpoint loader returned a non-learned policy")
    if not isinstance(metadata, Mapping):
        raise ValueError("policy checkpoint loader must return metadata")
    binding = metadata.get("binding")
    if not isinstance(binding, Mapping):
        raise ValueError("policy checkpoint metadata must include a binding mapping")
    expected_binding["budget_protocol"] = {
        "max_units": float(max_units),
        "policy_work_units": policy.declared_work_units,
    }
    # The worker compares nested binding values exactly.  Reload after learning
    # the checkpoint's architecture so max_units and policy work are checked by
    # the worker together, including for non-default hidden dimensions.
    policy, metadata = load_policy_checkpoint(
        path,
        expected_binding=expected_binding,
        device=device,
    )
    binding = metadata.get("binding")
    if not isinstance(binding, Mapping):
        raise ValueError("policy checkpoint metadata must include a binding mapping")
    _validate_policy_runtime_binding(binding, policy, max_units=max_units)
    normalized_metadata = dict(metadata)
    normalized_metadata.setdefault("content_hash", hash_value(policy.state_dict()))
    return policy, normalized_metadata, binding


def _validate_policy_runtime_binding(
    binding: Mapping[str, Any], policy: LearnedActionPolicy, *, max_units: float
) -> None:
    budget = binding.get("budget_protocol")
    if not isinstance(budget, Mapping):
        raise ValueError("policy checkpoint binding is missing budget_protocol")
    policy_units = budget.get("policy_work_units")
    if type(policy_units) is not int or policy_units != policy.declared_work_units:
        raise ValueError("policy checkpoint budget protocol has incompatible policy_work_units")
    declared_max = budget.get("max_units")
    if (
        isinstance(declared_max, bool)
        or not isinstance(declared_max, (int, float))
        or not math.isfinite(float(declared_max))
        or declared_max != float(max_units)
    ):
        raise ValueError("policy checkpoint budget protocol has incompatible max_units")


def _policy_file_sha256(path: str | Path | None) -> str | None:
    if path is None:
        return None
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _policy_content_hash(metadata: Mapping[str, Any] | None) -> str | None:
    if metadata is None:
        return None
    for key in ("content_hash", "policy_content_hash"):
        value = metadata.get(key)
        if isinstance(value, str) and value:
            return value
    binding = metadata.get("binding")
    if isinstance(binding, Mapping):
        for key in ("content_hash", "policy_content_hash"):
            value = binding.get(key)
            if isinstance(value, str) and value:
                return value
    return None


def _make_policy(
    name: str,
    residual_threshold: float,
    learned_policy: LearnedActionPolicy | None = None,
):
    if name == "RESIDUAL_THRESHOLD":
        return ResidualThresholdPolicy(threshold=float(residual_threshold))
    if name == "LEARNED":
        if learned_policy is None:
            raise ValueError("LEARNED policy requires an explicit policy checkpoint")
        return copy.deepcopy(learned_policy)
    return FixedPolicy(Action(name))


def _require_fixed_policy_actions(
    policy_name: str, actual_actions: Sequence[str], *, expected_count: int
) -> None:
    if policy_name in {"RESIDUAL_THRESHOLD", "LEARNED"}:
        return
    if len(actual_actions) != expected_count or any(
        action != policy_name for action in actual_actions
    ):
        raise RuntimeError(
            f"fixed policy {policy_name} fell back or diverged; "
            f"actual actions={list(actual_actions)}"
        )


def _record_identifier(record: Mapping[str, Any], name: str) -> str:
    value = record.get(name)
    if not isinstance(value, str) or not value:
        raise ValueError(f"checkpoint evaluation record requires a non-empty {name}")
    return value


def _record_averages(record: Mapping[str, Any]) -> Mapping[str, Any]:
    metrics = record.get("metrics")
    if not isinstance(metrics, Mapping) or not isinstance(metrics.get("averages"), Mapping):
        raise ValueError("checkpoint evaluation record requires metrics.averages")
    return metrics["averages"]


def _mean_finite(values: Sequence[Any] | Any) -> float | None:
    finite = [float(value) for value in values if _is_finite_number(value)]
    return sum(finite) / len(finite) if finite else None


def _finite_count(values: Sequence[Any] | Any) -> int:
    return sum(_is_finite_number(value) for value in values)


def _paired_delta(value: Any, off_value: Any) -> float | None:
    if not _is_finite_number(value) or not _is_finite_number(off_value):
        return None
    return float(value) - float(off_value)


def _is_finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _write_summary_csv(
    path: Path, summaries: Sequence[Mapping[str, Any]], metric_names: Sequence[str]
) -> None:
    names = [
        "split_id",
        "policy",
        "scene_count",
        "scene_ids",
        *(f"metric_{name}_mean" for name in metric_names),
        *(f"metric_{name}_valid_scene_count" for name in metric_names),
        *(f"paired_delta_{name}_vs_off_mean" for name in metric_names),
    ]
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=names)
        writer.writeheader()
        for summary in summaries:
            row = {
                "split_id": summary["split_id"],
                "policy": summary["policy"],
                "scene_count": summary["scene_count"],
                "scene_ids": ";".join(summary["scene_ids"]),
            }
            row.update(
                {
                    f"metric_{name}_mean": summary["metrics_mean"][name]
                    for name in metric_names
                }
            )
            row.update(
                {
                    f"metric_{name}_valid_scene_count": summary["metric_valid_scene_counts"][
                        name
                    ]
                    for name in metric_names
                }
            )
            row.update(
                {
                    f"paired_delta_{name}_vs_off_mean": summary["paired_delta_vs_off_mean"][name]
                    for name in metric_names
                }
            )
            writer.writerow(row)


def _write_json(path: Path, value: Any) -> None:
    content = json.dumps(_json_safe(value), indent=2, sort_keys=True) + "\n"
    path.write_text(content, encoding="utf-8")


def _write_json_exclusive(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(_json_safe(value), handle, indent=2, sort_keys=True)
        handle.write("\n")


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value
