"""Resumable, protocol-bound evaluation for the causal DL3DV benchmark.

The report layer owns the boundary between the online DL3DV adapter and the
post-seal RGB evaluator.  It prepares and freezes all provenance before an
observation source is opened, consumes the context in source order, seals the
scene, and only then requests lazy query callbacks.
"""

from __future__ import annotations

import copy
import hashlib
import inspect
import json
import math
import os
import re
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from time import perf_counter
from typing import Any

import torch

from mcss.data import dl3dv_benchmark as dl3dv_module
from mcss.data.dl3dv_benchmark import (
    DL3DV_EVAL_IMAGE_SIZE,
    DL3DV_SCENE_COUNT,
    Dl3dvBenchmarkMetadata,
    Dl3dvOnlineObservationSource,
    Dl3dvProtocol,
    load_benchmark_metadata,
    validate_complete_coverage,
)
from mcss.dynamic.checkpoint import SCHEMA_VERSION as DYNAMIC_CHECKPOINT_SCHEMA
from mcss.dynamic.checkpoint import load_dynamic_checkpoint
from mcss.dynamic.policy import (
    FEATURE_SCHEMA_VERSION,
    FixedPolicy,
    LearnedActionPolicy,
    ResidualThresholdPolicy,
)
from mcss.dynamic.policy_checkpoint import load_policy_checkpoint
from mcss.dynamic.runner import StreamingRunner
from mcss.dynamic.types import Action, hash_value
from mcss.evaluation.sealed_rgb import RGBMetric, evaluate_sealed_rgb, official_tttlrm_metrics
from mcss.measurements import FixedMeasurementRenderer

DL3DV_REPORT_SCHEMA = "mcss.dl3dv.evaluation.v1"
DL3DV_CANDIDATE_SCHEMA = "mcss.dl3dv.evaluation_candidate.v1"
DEFAULT_RENDER_SAMPLES = 48
DEFAULT_RAY_CHUNK_SIZE = 2048
DEFAULT_MAX_UNITS = 1e12
DEFAULT_POLICY = "OFF"
_POLICY_NAMES = frozenset({"OFF", "FUSE", "COMPLETE", "ALL", "RESIDUAL_THRESHOLD", "LEARNED"})
_SCENE_ID_RE = re.compile(r"[^A-Za-z0-9_.-]+")


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 digest of one file."""

    source = Path(path)
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_data_root(root: str | Path | None) -> str:
    """Hash a data root without opening benchmark RGB files.

    A directory fingerprint covers the relative file names, sizes, and mtimes.
    The benchmark images remain unread during candidate preparation; callers
    that have a stronger immutable data manifest can pass its digest directly
    through ``data_hash``.
    """

    if root is None:
        return _canonical_hash({"kind": "missing", "root": None})
    path = Path(root)
    if path.is_file():
        return sha256_file(path)
    if not path.exists():
        return _canonical_hash({"kind": "missing", "root": str(path)})
    entries: list[dict[str, Any]] = []
    for item in sorted(path.rglob("*")):
        if item.is_file():
            stat = item.stat()
            entries.append(
                {
                    "path": item.relative_to(path).as_posix(),
                    "size": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                }
            )
    return _canonical_hash({"kind": "directory_manifest", "root": str(path), "files": entries})


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(_json_safe(value), sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(payload).hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if hasattr(value, "item"):
        try:
            return _json_safe(value.item())
        except (AttributeError, ValueError, TypeError):
            pass
    return repr(value)


def _write_json_exclusive(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(_json_safe(value), handle, indent=2, sort_keys=True)
        handle.write("\n")


def _write_json_replace(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.part")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(_json_safe(value), handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON artifact: {path}") from error


def _metadata_payload(metadata: Any) -> dict[str, Any]:
    to_dict = getattr(metadata, "to_dict", None)
    if callable(to_dict):
        value = to_dict()
        if isinstance(value, Mapping):
            return dict(value)
    scenes = tuple(getattr(metadata, "scenes", ()))
    protocol = getattr(metadata, "protocol", None)
    return {
        "schema": "injected_metadata",
        "scenes": [str(getattr(scene, "scene_id", "")) for scene in scenes],
        "protocol": {
            "mode": getattr(protocol, "mode", None),
            "input_count": getattr(protocol, "input_count", None),
            "fold_size": getattr(protocol, "fold_size", None),
            "warmup_length": getattr(protocol, "warmup_length", None),
        },
    }


def _resolve_metadata(
    metadata: Dl3dvBenchmarkMetadata | str | Path,
) -> tuple[Any, str | None]:
    if isinstance(metadata, (str, Path)):
        path = Path(metadata)
        return load_benchmark_metadata(path), sha256_file(path)
    return metadata, None


def _validate_coverage(metadata: Any) -> dict[str, Any]:
    scenes = tuple(getattr(metadata, "scenes", ()))
    coverage_method = getattr(metadata, "coverage", None)
    if callable(coverage_method):
        coverage = dict(coverage_method())
    else:
        coverage = validate_complete_coverage(scenes)
    if int(coverage.get("scene_count", len(scenes))) != DL3DV_SCENE_COUNT:
        raise ValueError(
            f"DL3DV full benchmark requires {DL3DV_SCENE_COUNT} scenes, "
            f"got {coverage.get('scene_count', len(scenes))}"
        )
    if len(scenes) != DL3DV_SCENE_COUNT:
        raise ValueError(
            f"DL3DV metadata requires {DL3DV_SCENE_COUNT} scene objects, got {len(scenes)}"
        )
    return coverage


def _validate_protocol(
    metadata: Any,
    *,
    mode: str,
    input_count: int,
) -> Dl3dvProtocol:
    scenes = tuple(getattr(metadata, "scenes", ()))
    if not scenes:
        raise ValueError("DL3DV metadata contains no scenes")
    protocol = getattr(metadata, "protocol", scenes[0].protocol)
    if not isinstance(protocol, Dl3dvProtocol):
        raise TypeError("DL3DV metadata protocol must be a Dl3dvProtocol")
    requested = Dl3dvProtocol(mode, input_count)
    if protocol != requested:
        raise ValueError(
            "requested DL3DV protocol differs from metadata: "
            f"requested={requested.name}, metadata={protocol.name}"
        )
    for scene in scenes:
        if scene.protocol != protocol:
            raise ValueError("DL3DV metadata scenes do not share one protocol")
    return protocol


def _call_source_factory(factory: Callable[..., Any], scene: Any, device: torch.device) -> Any:
    """Call injected factories without swallowing TypeErrors raised by the factory body."""

    attempts = (((scene,), {"device": device}), ((scene, device), {}), ((scene,), {}))
    try:
        signature = inspect.signature(factory)
    except (TypeError, ValueError):
        return factory(scene, device=device)
    for args, kwargs in attempts:
        try:
            signature.bind(*args, **kwargs)
        except TypeError:
            continue
        return factory(*args, **kwargs)
    raise TypeError("source_factory must accept (scene), (scene, device), or device keyword")


def _resolve_device(value: str | torch.device) -> torch.device:
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def _sync_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _load_learned_policy(
    path: str | Path | None,
    *,
    policy_name: str,
    expected_state_hash: str,
    max_units: float,
    device: torch.device,
) -> tuple[LearnedActionPolicy | None, dict[str, Any] | None, dict[str, Any] | None]:
    if policy_name != "LEARNED":
        if path is not None:
            raise ValueError("policy_checkpoint requires policy='LEARNED'")
        return None, None, None
    if path is None:
        raise ValueError("policy='LEARNED' requires an explicit policy_checkpoint")
    expected_binding: dict[str, Any] = {
        "model_content_hash": expected_state_hash,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "render_protocol": {
            "renderer": "FixedMeasurementRenderer",
            "renderer_samples": DEFAULT_RENDER_SAMPLES,
            "ray_chunk_size": DEFAULT_RAY_CHUNK_SIZE,
        },
    }
    policy, metadata = load_policy_checkpoint(
        path,
        expected_binding=expected_binding,
        device=device,
    )
    if not isinstance(policy, LearnedActionPolicy):
        raise TypeError("policy checkpoint loader returned a non-learned policy")
    expected_binding["budget_protocol"] = {
        "max_units": float(max_units),
        "policy_work_units": policy.declared_work_units,
    }
    policy, metadata = load_policy_checkpoint(
        path,
        expected_binding=expected_binding,
        device=device,
    )
    binding = metadata.get("binding")
    if not isinstance(binding, Mapping):
        raise ValueError("policy checkpoint metadata must include a binding mapping")
    budget = binding.get("budget_protocol")
    if not isinstance(budget, Mapping):
        raise ValueError("policy checkpoint binding is missing budget_protocol")
    if budget.get("policy_work_units") != policy.declared_work_units:
        raise ValueError("policy checkpoint policy_work_units do not match policy architecture")
    if budget.get("max_units") != float(max_units):
        raise ValueError("policy checkpoint max_units do not match benchmark budget")
    normalized = dict(metadata)
    normalized["content_hash"] = hash_value(policy.state_dict())
    return policy, normalized, dict(binding)


def _make_policy(
    name: str,
    *,
    learned_policy: LearnedActionPolicy | None,
    residual_threshold: float,
) -> Any:
    if name == "LEARNED":
        if learned_policy is None:
            raise ValueError("learned policy is unavailable")
        return copy.deepcopy(learned_policy)
    if name == "RESIDUAL_THRESHOLD":
        return ResidualThresholdPolicy(threshold=residual_threshold)
    return FixedPolicy(Action(name))


def _scene_record_path(output: Path, index: int, scene_id: str) -> Path:
    safe = _SCENE_ID_RE.sub("_", scene_id).strip("._") or "scene"
    return output / "records" / f"{index:03d}_{safe}.json"


def _metric_spec(metrics: Sequence[RGBMetric]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for metric in metrics:
        if metric.name in result:
            raise ValueError(f"duplicate RGB metric name: {metric.name}")
        result[metric.name] = {
            "preprocessing": metric.preprocessing,
            "full_resolution": metric.full_resolution,
        }
    return result


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _scene_means(
    per_image: Sequence[Mapping[str, Any]], metric_names: Sequence[str]
) -> dict[str, Any]:
    means: dict[str, float | None] = {}
    counts: dict[str, int] = {}
    for name in metric_names:
        values = [
            float(record["metrics"][name])
            for record in per_image
            if record.get("status") == "ok"
            and isinstance(record.get("metrics"), Mapping)
            and _finite(record["metrics"].get(name))
        ]
        counts[name] = len(values)
        means[name] = sum(values) / len(values) if values else None
    return {"metrics": means, "valid_image_counts": counts}


def aggregate_dl3dv_records(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate per-scene DL3DV records with explicit macro/micro counts."""

    scene_records = [dict(record) for record in records]
    metric_names: set[str] = set()
    for record in scene_records:
        scene_mean = record.get("scene_mean")
        if isinstance(scene_mean, Mapping) and isinstance(scene_mean.get("metrics"), Mapping):
            metric_names.update(str(name) for name in scene_mean["metrics"])
        for image in record.get("per_image", ()):
            if isinstance(image, Mapping) and isinstance(image.get("metrics"), Mapping):
                metric_names.update(str(name) for name in image["metrics"])
    ordered_metrics = tuple(sorted(metric_names))

    per_scene: list[dict[str, Any]] = []
    micro_values: dict[str, list[float]] = {name: [] for name in ordered_metrics}
    macro_values: dict[str, list[float]] = {name: [] for name in ordered_metrics}
    failed_scene_count = 0
    failed_image_count = 0
    for record in scene_records:
        images = [item for item in record.get("per_image", ()) if isinstance(item, Mapping)]
        scene_mean = _scene_means(images, ordered_metrics)
        if not images and isinstance(record.get("scene_mean"), Mapping):
            source_mean = record["scene_mean"].get("metrics", {})
            source_counts = record["scene_mean"].get("valid_image_counts", {})
            scene_mean = {
                "metrics": {
                    name: source_mean.get(name) for name in ordered_metrics
                },
                "valid_image_counts": {
                    name: int(source_counts.get(name, 0)) for name in ordered_metrics
                },
            }
        record["scene_mean"] = scene_mean
        record["metrics_mean"] = dict(scene_mean["metrics"])
        record["valid_counts"] = dict(scene_mean["valid_image_counts"])
        if record.get("status") == "failed":
            failed_scene_count += 1
        failed_image_count += sum(1 for image in images if image.get("status") == "failed")
        for name in ordered_metrics:
            value = scene_mean["metrics"].get(name)
            if _finite(value):
                macro_values[name].append(float(value))
            for image in images:
                metric_values = image.get("metrics")
                if isinstance(metric_values, Mapping) and _finite(metric_values.get(name)):
                    micro_values[name].append(float(metric_values[name]))
        per_scene.append(record)

    macro = {
        "metrics": {
            name: sum(values) / len(values) if values else None
            for name, values in macro_values.items()
        },
        "valid_scene_counts": {name: len(values) for name, values in macro_values.items()},
    }
    micro = {
        "metrics": {
            name: sum(values) / len(values) if values else None
            for name, values in micro_values.items()
        },
        "valid_image_counts": {name: len(values) for name, values in micro_values.items()},
    }
    return {
        "metric_names": list(ordered_metrics),
        "scene_results": per_scene,
        "macro_scene_mean": macro,
        "micro_image_mean": micro,
        "macro": macro,
        "micro": micro,
        "failure_counts": {
            "scenes": failed_scene_count,
            "images": failed_image_count,
        },
    }


def _failed_scene_record(
    scene: Any,
    *,
    error: BaseException,
    metric_names: Sequence[str],
    actions: Sequence[str] = (),
    observed_ids: Sequence[int] = (),
    adaptation_seconds: float | None = None,
    evaluation_seconds: float | None = None,
    peak_cuda_allocated_bytes: int | None = None,
) -> dict[str, Any]:
    return {
        "schema": f"{DL3DV_REPORT_SCHEMA}.scene",
        "status": "failed",
        "episode_id": str(getattr(scene, "episode_id", f"dl3dv:{scene.scene_id}")),
        "scene_id": str(scene.scene_id),
        "split_id": str(getattr(scene, "split_id", "test")),
        "query_vault_id": str(getattr(scene, "query_vault_id", f"dl3dv:{scene.scene_id}:queries")),
        "target_count": len(getattr(scene, "query_indices", ())),
        "per_image": [],
        "scene_mean": {
            "metrics": {name: None for name in metric_names},
            "valid_image_counts": {name: 0 for name in metric_names},
        },
        "valid_counts": {name: 0 for name in metric_names},
        "failure": {"type": type(error).__name__, "message": str(error)},
        "failure_count": 1,
        "actions": list(actions),
        "observed_ids": list(observed_ids),
        "adaptation_seconds": adaptation_seconds,
        "evaluation_seconds": evaluation_seconds,
        "total_seconds": None
        if adaptation_seconds is None or evaluation_seconds is None
        else adaptation_seconds + evaluation_seconds,
        "peak_cuda_allocated_bytes": peak_cuda_allocated_bytes,
    }


def _evaluate_scene(
    scene: Any,
    *,
    base_carrier: torch.nn.Module,
    base_write_rule: torch.nn.Module,
    expected_state_hash: str,
    learned_policy: LearnedActionPolicy | None,
    policy_name: str,
    residual_threshold: float,
    device: torch.device,
    max_units: float,
    metrics: Sequence[RGBMetric],
    source_factory: Callable[..., Any],
) -> dict[str, Any]:
    metric_names = tuple(metric.name for metric in metrics)
    renderer = FixedMeasurementRenderer(
        n_samples=DEFAULT_RENDER_SAMPLES,
        ray_chunk_size=DEFAULT_RAY_CHUNK_SIZE,
    )
    runner: StreamingRunner | None = None
    actions: list[str] = []
    observed_ids: list[int] = []
    adaptation_started = perf_counter()
    adaptation_seconds: float | None = None
    evaluation_seconds: float | None = None
    peak_cuda_allocated_bytes: int | None = None
    try:
        source = _call_source_factory(source_factory, scene, device)
        warmup = tuple(source.warmup())
        runner = StreamingRunner(
            copy.deepcopy(base_carrier),
            copy.deepcopy(base_write_rule),
            renderer,
            _make_policy(
                policy_name,
                learned_policy=learned_policy,
                residual_threshold=residual_threshold,
            ),
            max_units=max_units,
        )
        if runner.checkpoint_hash != expected_state_hash:
            raise RuntimeError("scene runner state does not match the loaded dynamic checkpoint")
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        runner.reset(
            warmup,
            episode_id=source.episode_id,
            scene_id=source.scene_id,
            split_id=source.split_id,
            query_vault_id=source.query_vault_id,
            stream_steps=source.remaining_steps,
            query_count=source.query_count,
        )
        while source.remaining_steps:
            runner.step(source.next_observation())
        if source.remaining_steps != 0:
            raise RuntimeError("DL3DV source did not consume its complete online stream")
        sealed = runner.seal()
        observed_ids = list(sealed.observed_ids)
        expected_ids = tuple(scene.arrival_indices)
        if tuple(sealed.observed_ids) != expected_ids:
            raise RuntimeError(
                "DL3DV context arrival order differs from metadata: "
                f"{sealed.observed_ids} != {expected_ids}"
            )
        if sealed.checkpoint_hash != expected_state_hash:
            raise RuntimeError("sealed scene does not identify the loaded checkpoint state")
        actions = [str(item["action"]) for item in runner.history]
        adaptation_seconds = perf_counter() - adaptation_started
        _sync_cuda(device)
        evaluation_started = perf_counter()
        # This is the first access to the query vault.  It is deliberately after seal().
        queries = source.queries_for_sealed(sealed)
        if len(queries) != len(scene.query_indices):
            raise RuntimeError("DL3DV query callback count differs from metadata")
        metric_report = evaluate_sealed_rgb(
            sealed,
            queries,
            renderer,
            metrics=metrics,
            device=device,
        )
        _sync_cuda(device)
        evaluation_seconds = perf_counter() - evaluation_started
        if device.type == "cuda":
            peak_cuda_allocated_bytes = int(torch.cuda.max_memory_allocated(device))
        pixel_count = int(scene.target_size[0] * scene.target_size[1])
        per_image = []
        for item in metric_report["per_image"]:
            values = {name: float(item[name]) for name in metric_names}
            per_image.append(
                {
                    "frame_id": int(item["frame_id"]),
                    "status": "ok",
                    "metrics": values,
                    "valid_counts": {name: pixel_count for name in metric_names},
                    "valid_count": pixel_count,
                    "failure": None,
                }
            )
        scene_mean = _scene_means(per_image, metric_names)
        return {
            "schema": f"{DL3DV_REPORT_SCHEMA}.scene",
            "status": "ok",
            "episode_id": sealed.episode_id,
            "scene_id": sealed.scene_id,
            "split_id": sealed.split_id,
            "query_vault_id": sealed.query_vault_id,
            "state_hash": sealed.state_hash,
            "checkpoint_hash": sealed.checkpoint_hash,
            "fast_state_hash": sealed.fast_state_hash,
            "target_frame_ids": [int(query.frame_id) for query in queries],
            "target_count": len(per_image),
            "per_image": per_image,
            "scene_mean": scene_mean,
            "metrics_mean": dict(scene_mean["metrics"]),
            "valid_counts": dict(scene_mean["valid_image_counts"]),
            "failure": None,
            "failure_count": 0,
            "actions": actions,
            "observed_ids": observed_ids,
            "adaptation_seconds": adaptation_seconds,
            "evaluation_seconds": evaluation_seconds,
            "total_seconds": adaptation_seconds + evaluation_seconds,
            "peak_cuda_allocated_bytes": peak_cuda_allocated_bytes,
            "budget": runner.budget.report(),
            "history": copy.deepcopy(runner.history),
            "metric_spec": metric_report["metric_spec"],
        }
    except Exception as error:
        if runner is not None:
            actions = [str(item["action"]) for item in runner.history]
        if adaptation_seconds is None:
            adaptation_seconds = perf_counter() - adaptation_started
        return _failed_scene_record(
            scene,
            error=error,
            metric_names=metric_names,
            actions=actions,
            observed_ids=observed_ids,
            adaptation_seconds=adaptation_seconds,
            evaluation_seconds=evaluation_seconds,
            peak_cuda_allocated_bytes=peak_cuda_allocated_bytes,
        )


def evaluate_dl3dv_benchmark(
    metadata: Dl3dvBenchmarkMetadata | str | Path,
    checkpoint_path: str | Path,
    output_dir: str | Path,
    *,
    mode: str = "full",
    input_count: int = 32,
    device: str | torch.device = "cpu",
    policy: str = DEFAULT_POLICY,
    policy_checkpoint: str | Path | None = None,
    residual_threshold: float = 0.03,
    max_units: float = DEFAULT_MAX_UNITS,
    max_scenes: int | None = None,
    resume: bool = False,
    metrics: Sequence[RGBMetric] | None = None,
    include_lpips: bool = True,
    source_factory: Callable[..., Any] | None = None,
    manifest_hash: str | None = None,
    data_hash: str | None = None,
    source_hashes: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Evaluate one dynamic checkpoint over the pinned DL3DV-140 split.

    ``max_scenes`` is intentionally a diagnostic subset.  Its report status is
    never ``complete`` and it cannot be mistaken for the 140-scene benchmark.
    """

    if max_scenes is not None and (type(max_scenes) is not int or max_scenes < 1):
        raise ValueError("max_scenes must be a positive integer when specified")
    if type(max_units) not in (int, float) or not math.isfinite(float(max_units)) or max_units <= 0:
        raise ValueError("max_units must be a finite positive number")
    if policy_checkpoint is not None and policy == DEFAULT_POLICY:
        policy = "LEARNED"
    policy = str(policy).upper()
    if policy not in _POLICY_NAMES:
        raise ValueError(f"policy must be selected from {sorted(_POLICY_NAMES)}")
    if policy == "RESIDUAL_THRESHOLD" and (
        not math.isfinite(float(residual_threshold)) or residual_threshold < 0
    ):
        raise ValueError("residual_threshold must be finite and nonnegative")

    resolved_metadata, path_manifest_hash = _resolve_metadata(metadata)
    coverage = _validate_coverage(resolved_metadata)
    protocol = _validate_protocol(resolved_metadata, mode=mode, input_count=input_count)
    scenes = tuple(resolved_metadata.scenes)
    selected_scenes = scenes if max_scenes is None else scenes[:max_scenes]
    target_device = _resolve_device(device)

    base_carrier, base_write_rule, checkpoint_metadata = load_dynamic_checkpoint(
        checkpoint_path,
        target_device,
    )
    if checkpoint_metadata.get("schema_version") != DYNAMIC_CHECKPOINT_SCHEMA:
        raise RuntimeError("dynamic checkpoint loader returned an unexpected schema")
    expected_state_hash = hash_value(
        {"carrier": base_carrier.state_dict(), "write_rule": base_write_rule.state_dict()}
    )
    learned_policy, policy_metadata, policy_binding = _load_learned_policy(
        policy_checkpoint,
        policy_name=policy,
        expected_state_hash=expected_state_hash,
        max_units=float(max_units),
        device=target_device,
    )
    metric_objects = tuple(
        metrics if metrics is not None else official_tttlrm_metrics(include_lpips=include_lpips)
    )
    metric_spec = _metric_spec(metric_objects)
    source_factory = source_factory or Dl3dvOnlineObservationSource

    module_paths = {
        "dl3dv_adapter": Path(dl3dv_module.__file__).resolve(),
        "sealed_rgb_evaluator": Path(inspect.getfile(evaluate_sealed_rgb)).resolve(),
        "streaming_runner": Path(inspect.getfile(StreamingRunner)).resolve(),
    }
    protocol_notes = (
        Path(__file__).resolve().parents[3]
        / "outputs"
        / "benchmark_protocol_20260920"
        / "tttLRM_protocol_notes.md"
    )
    if protocol_notes.is_file():
        module_paths["protocol_notes"] = protocol_notes
    computed_source_hashes = {name: sha256_file(path) for name, path in module_paths.items()}
    if source_hashes is not None:
        computed_source_hashes.update(
            {str(key): str(value) for key, value in source_hashes.items()}
        )
    resolved_manifest_hash = (
        manifest_hash or path_manifest_hash or _canonical_hash(_metadata_payload(resolved_metadata))
    )
    raw_root = getattr(resolved_metadata, "raw_root", None)
    resolved_data_hash = data_hash or hash_data_root(raw_root)
    policy_file_hash = None if policy_checkpoint is None else sha256_file(policy_checkpoint)
    hashes = {
        "source_sha256": _canonical_hash(computed_source_hashes),
        "source_files": computed_source_hashes,
        "manifest_sha256": resolved_manifest_hash,
        "data_sha256": resolved_data_hash,
        "dynamic_checkpoint_sha256": str(checkpoint_metadata["sha256"]),
        "dynamic_model_content_sha256": expected_state_hash,
        "policy_sha256": policy_file_hash,
        "policy_content_sha256": (
            None if policy_metadata is None else policy_metadata.get("content_hash")
        ),
    }
    protocol_payload = {
        "mode": protocol.mode,
        "input_count": protocol.input_count,
        "fold_size": protocol.fold_size,
        "warmup_length": protocol.warmup_length,
        "target_stride": protocol.fold_size,
        "target_size": list(DL3DV_EVAL_IMAGE_SIZE),
    }
    manifest_candidate = {
        "schema": f"{DL3DV_CANDIDATE_SCHEMA}.manifest",
        "manifest_sha256": resolved_manifest_hash,
        "data_sha256": resolved_data_hash,
        "coverage": {
            "expected_scene_count": DL3DV_SCENE_COUNT,
            "metadata_scene_count": len(scenes),
            "selected_scene_count": len(selected_scenes),
            "max_scenes": max_scenes,
        },
        "scene_ids": [str(scene.scene_id) for scene in selected_scenes],
        "target_frame_ids": {
            str(scene.scene_id): [int(frame_id) for frame_id in scene.query_indices]
            for scene in selected_scenes
        },
        "protocol": protocol_payload,
    }
    config_candidate = {
        "schema": f"{DL3DV_CANDIDATE_SCHEMA}.config",
        "protocol": protocol_payload,
        "renderer": {
            "name": "FixedMeasurementRenderer",
            "n_samples": DEFAULT_RENDER_SAMPLES,
            "ray_chunk_size": DEFAULT_RAY_CHUNK_SIZE,
        },
        "budget": {"max_units": float(max_units)},
        "policy": {
            "name": policy,
            "residual_threshold": float(residual_threshold),
            "checkpoint_sha256": policy_file_hash,
            "binding": policy_binding,
        },
        "metrics": metric_spec,
        "device": str(target_device),
        "canonicalization": "none; warmup-only adapter gauge",
        "hashes": hashes,
    }
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifest_candidate_path = output / "manifest_candidate.json"
    config_candidate_path = output / "config_candidate.json"
    if manifest_candidate_path.exists() or config_candidate_path.exists():
        if not resume:
            raise FileExistsError(
                "evaluation output already contains frozen candidates; use resume=True"
            )
        if not manifest_candidate_path.is_file() or not config_candidate_path.is_file():
            raise ValueError("resume candidate mismatch: one frozen candidate is missing")
        if _read_json(manifest_candidate_path) != _json_safe(manifest_candidate):
            raise ValueError("resume candidate mismatch: manifest candidate differs")
        if _read_json(config_candidate_path) != _json_safe(config_candidate):
            raise ValueError("resume candidate mismatch: config candidate differs")
    else:
        if resume and (output / "report.json").exists():
            raise ValueError("resume candidate mismatch: frozen candidates are missing")
        _write_json_exclusive(manifest_candidate_path, manifest_candidate)
        _write_json_exclusive(config_candidate_path, config_candidate)

    record_paths = {
        str(scene.scene_id): _scene_record_path(output, index, str(scene.scene_id))
        for index, scene in enumerate(selected_scenes)
    }
    if len(record_paths) != len(selected_scenes):
        raise ValueError("DL3DV selected scene IDs must be unique")
    records: list[dict[str, Any]] = []
    for scene in selected_scenes:
        path = record_paths[str(scene.scene_id)]
        if resume and path.is_file():
            record = _read_json(path)
            if not isinstance(record, Mapping) or record.get("scene_id") != scene.scene_id:
                raise ValueError(f"resume record does not match scene: {path}")
            records.append(dict(record))
            continue
        record = _evaluate_scene(
            scene,
            base_carrier=base_carrier,
            base_write_rule=base_write_rule,
            expected_state_hash=expected_state_hash,
            learned_policy=learned_policy,
            policy_name=policy,
            residual_threshold=float(residual_threshold),
            device=target_device,
            max_units=float(max_units),
            metrics=metric_objects,
            source_factory=source_factory,
        )
        _write_json_exclusive(path, record)
        records.append(record)
        partial_report = _build_report(
            records,
            coverage=coverage,
            max_scenes=max_scenes,
            protocol=protocol_payload,
            policy=policy,
            max_units=float(max_units),
            hashes=hashes,
            manifest_candidate=manifest_candidate,
            config_candidate=config_candidate,
            status="in_progress",
        )
        _write_json_replace(output / "report.json", partial_report)

    has_failures = any(record.get("status") == "failed" for record in records)
    status = (
        "complete"
        if max_scenes is None and not has_failures
        else "diagnostic_partial"
        if max_scenes is not None
        else "complete_with_failures"
    )
    report = _build_report(
        records,
        coverage=coverage,
        max_scenes=max_scenes,
        protocol=protocol_payload,
        policy=policy,
        max_units=float(max_units),
        hashes=hashes,
        manifest_candidate=manifest_candidate,
        config_candidate=config_candidate,
        status=status,
    )
    _write_json_replace(output / "report.json", report)
    return report


def _build_report(
    records: Sequence[Mapping[str, Any]],
    *,
    coverage: Mapping[str, Any],
    max_scenes: int | None,
    protocol: Mapping[str, Any],
    policy: str,
    max_units: float,
    hashes: Mapping[str, Any],
    manifest_candidate: Mapping[str, Any],
    config_candidate: Mapping[str, Any],
    status: str,
) -> dict[str, Any]:
    aggregation = aggregate_dl3dv_records(records)
    selected_count = len(records)
    return {
        "schema": DL3DV_REPORT_SCHEMA,
        "status": status,
        "benchmark_claim": "full_dl3dv_140" if max_scenes is None else "diagnostic_partial",
        "coverage": {
            "status": "complete" if max_scenes is None else "partial",
            "expected_scene_count": DL3DV_SCENE_COUNT,
            "metadata_scene_count": int(coverage.get("scene_count", DL3DV_SCENE_COUNT)),
            "selected_scene_count": selected_count,
            "max_scenes": max_scenes,
            "is_full_benchmark": max_scenes is None,
        },
        "protocol": dict(protocol),
        "renderer": {
            "name": "FixedMeasurementRenderer",
            "n_samples": DEFAULT_RENDER_SAMPLES,
            "ray_chunk_size": DEFAULT_RAY_CHUNK_SIZE,
        },
        "budget": {"max_units": float(max_units)},
        "policy": policy,
        "hashes": dict(hashes),
        "manifest_candidate": dict(manifest_candidate),
        "config_candidate": dict(config_candidate),
        "records": list(records),
        "scene_results": aggregation["scene_results"],
        "aggregation": aggregation,
        "macro_scene_mean": aggregation["macro_scene_mean"],
        "micro_image_mean": aggregation["micro_image_mean"],
        "failure_counts": aggregation["failure_counts"],
    }


__all__ = [
    "DEFAULT_MAX_UNITS",
    "DEFAULT_POLICY",
    "DEFAULT_RAY_CHUNK_SIZE",
    "DEFAULT_RENDER_SAMPLES",
    "DL3DV_CANDIDATE_SCHEMA",
    "DL3DV_REPORT_SCHEMA",
    "aggregate_dl3dv_records",
    "evaluate_dl3dv_benchmark",
    "hash_data_root",
    "sha256_file",
]
