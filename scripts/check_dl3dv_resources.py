"""Preflight dynamic DL3DV resources with deterministic synthetic RGB only.

This diagnostic exercises the deployed dynamic carrier, write rule, fixed
renderer, and sealed RGB evaluator at the official 536x960 resolution.  It
does not open a dataset manifest or any external image.  The target query is
analytic and is intentionally evaluated only after the scene has been sealed.
"""

from __future__ import annotations

import argparse
import dataclasses
import gc
import hashlib
import inspect
import json
import math
import os
import time
import traceback
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from mcss.dynamic.checkpoint import load_dynamic_checkpoint
from mcss.dynamic.policy import FixedPolicy
from mcss.dynamic.runner import StreamingRunner
from mcss.dynamic.types import Action, OnlineObservation
from mcss.evaluation.sealed_rgb import RGBQuery, evaluate_sealed_rgb, official_tttlrm_metrics
from mcss.geometry import look_at, make_intrinsics
from mcss.measurements import FixedMeasurementRenderer
from mcss.types import Cameras

IMAGE_SIZE = (536, 960)
WARMUP_COUNT = 4
ALLOWED_INPUT_COUNTS = (16, 32)
RENDER_SAMPLES = 48
RAY_CHUNK_SIZE = 2048
MAX_UNITS = 1e12
SEED = 20260920
SCHEMA_VERSION = "mcss.dynamic.resource_preflight.v2"
EPISODE_ID = "synthetic_resource_preflight"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="new output directory; an existing directory is always rejected",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="torch device, including an explicit CUDA index (default: cuda:0)",
    )
    parser.add_argument(
        "--input-count",
        required=True,
        type=int,
        choices=ALLOWED_INPUT_COUNTS,
        help="total synthetic input frames, including the four warmup frames",
    )
    return parser


def sha256_file(path: str | Path) -> str:
    """Return a file SHA-256 without loading the complete file into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(_json_safe(value), sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(payload).hexdigest()


def _json_safe(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if dataclasses.is_dataclass(value):
        return {
            field.name: _json_safe(getattr(value, field.name))
            for field in dataclasses.fields(value)
        }
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    return str(value)


def finite_value(value: Any) -> bool:
    """Return whether a nested runtime value contains only finite numbers."""

    if torch.is_tensor(value):
        return bool(torch.isfinite(value).all().item())
    if dataclasses.is_dataclass(value):
        return all(
            finite_value(getattr(value, field.name)) for field in dataclasses.fields(value)
        )
    if isinstance(value, Mapping):
        return all(finite_value(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite_value(item) for item in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def _source_paths() -> tuple[Path, ...]:
    """Return source files whose identity bounds this diagnostic."""

    module_objects = (
        StreamingRunner,
        OnlineObservation,
        Action,
        FixedPolicy,
        FixedMeasurementRenderer,
        evaluate_sealed_rgb,
        official_tttlrm_metrics,
        load_dynamic_checkpoint,
        look_at,
        make_intrinsics,
        Cameras,
    )
    paths = [Path(__file__).resolve()]
    paths.extend(Path(inspect.getfile(item)).resolve() for item in module_objects)
    return tuple(dict.fromkeys(paths))


def source_manifest() -> dict[str, Any]:
    """Hash this CLI and the shared runtime modules it calls."""

    files: dict[str, str] = {}
    for path in _source_paths():
        if not path.is_file():
            raise FileNotFoundError(f"source file for preflight hash is missing: {path}")
        files[str(path)] = sha256_file(path)
    return {"files": files, "sha256": _canonical_hash(files)}


def _is_rigid_pose(pose: torch.Tensor) -> bool:
    if pose.shape != (4, 4) or not torch.isfinite(pose).all():
        return False
    expected_row = torch.tensor([0.0, 0.0, 0.0, 1.0], device=pose.device, dtype=pose.dtype)
    rotation = pose[:3, :3]
    rotation_is_orthonormal = torch.allclose(
        rotation.transpose(0, 1) @ rotation,
        torch.eye(3, device=pose.device),
        atol=1e-5,
        rtol=1e-5,
    )
    return bool(
        torch.allclose(pose[3], expected_row, atol=1e-5, rtol=1e-5)
        and rotation_is_orthonormal
        # The project uses OpenCV x-right/y-down/z-forward camera axes, so the
        # orthogonal camera basis has determinant -1 while remaining rigid.
        and torch.isclose(torch.linalg.det(rotation).abs(), torch.ones((), device=pose.device))
    )


def synthetic_camera(frame_id: int, intrinsics: torch.Tensor, device: torch.device) -> Cameras:
    """Build a deterministic legal rigid camera for one synthetic frame."""

    phase = 0.13 * float(frame_id)
    eye = torch.tensor(
        [0.010 * frame_id, 0.002 * math.sin(phase), -3.0 + 0.005 * frame_id],
        device=device,
        dtype=torch.float32,
    )
    target = torch.zeros(3, device=device, dtype=torch.float32)
    pose = look_at(eye, target)
    if not _is_rigid_pose(pose):
        raise RuntimeError(f"synthetic camera pose is not rigid at frame {frame_id}")
    return Cameras(intrinsics.detach().clone(), pose, IMAGE_SIZE)


def synthetic_rgb(
    frame_id: int,
    x_grid: torch.Tensor,
    y_grid: torch.Tensor,
) -> torch.Tensor:
    """Return deterministic analytic RGB in channel-first [0, 1] form."""

    phase = 0.13 * float(frame_id)
    red = 0.22 + 0.20 * x_grid + 0.025 * torch.sin(6.0 * y_grid + phase)
    green = 0.28 + 0.18 * y_grid + 0.025 * torch.cos(5.0 * x_grid - phase)
    blue = 0.24 + 0.08 * x_grid + 0.10 * y_grid + 0.020 * torch.sin(
        4.0 * x_grid + 3.0 * y_grid + phase
    )
    return torch.stack((red, green, blue), dim=0).clamp(0.0, 1.0).contiguous()


def synthetic_observation(
    frame_id: int,
    x_grid: torch.Tensor,
    y_grid: torch.Tensor,
    intrinsics: torch.Tensor,
    device: torch.device,
) -> OnlineObservation:
    camera = synthetic_camera(frame_id, intrinsics, device)
    rgb = synthetic_rgb(frame_id, x_grid, y_grid)
    return OnlineObservation(EPISODE_ID, frame_id, rgb, camera)


def _new_output_dir(path: str | Path) -> Path:
    output = Path(path).expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite existing preflight output: {output}")
    output.mkdir(parents=True, exist_ok=False)
    return output


def _write_report(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f"{path.name}.part")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(_json_safe(payload), handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, path)


def _append_history(path: Path, history: list[dict[str, Any]], event: Mapping[str, Any]) -> None:
    safe_event = _json_safe(dict(event))
    if not isinstance(safe_event, dict):
        raise TypeError("preflight history event must serialize to a JSON object")
    history.append(safe_event)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        json.dump(safe_event, handle, sort_keys=True, allow_nan=False)
        handle.write("\n")


def _sync_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _base_payload(
    checkpoint: Path,
    output: Path,
    device: str,
    input_count: int,
    source: Mapping[str, Any],
) -> dict[str, Any]:
    checkpoint_hash = sha256_file(checkpoint) if checkpoint.is_file() else None
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "not_started",
        "scope": "synthetic_only_resource_preflight",
        "source_sha256": source["sha256"],
        "source_files": source["files"],
        "checkpoint_sha256": checkpoint_hash,
        "source": {
            "kind": "analytic_synthetic_rgb",
            "external_images_opened": False,
            "external_manifests_opened": False,
            "description": (
                "RGB is generated in process; no external/train/dev/query pixels are opened"
            ),
            "seed": SEED,
            "source_manifest": dict(source),
        },
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": checkpoint_hash,
        },
        "output": str(output),
        "protocol": {
            "image_size": list(IMAGE_SIZE),
            "input_count": input_count,
            "warmup_count": WARMUP_COUNT,
            "stream_count": input_count - WARMUP_COUNT,
            "target_frame_id": input_count,
            "render_samples": RENDER_SAMPLES,
            "ray_chunk_size": RAY_CHUNK_SIZE,
            "policy": "ALL",
            "max_units": MAX_UNITS,
            "device": device,
            "camera_contract": (
                "look_at legal rigid c2w poses with fixed intrinsics and modest translations"
            ),
            "generation": "one analytic RGB frame generated and released at a time",
            "target_render": "one full-resolution target query after seal",
            "metrics": ["psnr", "ssim", "lpips"],
        },
        "runtime": {
            "history": [],
            "finite_checks": [],
            "actions": [],
            "completed_steps": 0,
        },
        "target_render": {"status": "not_started"},
    }


def run_preflight(
    checkpoint: str | Path,
    output: str | Path,
    *,
    device: str = "cuda:0",
    input_count: int = 32,
) -> dict[str, Any]:
    """Run the synthetic resource check and persist a report in a new directory."""

    if input_count not in ALLOWED_INPUT_COUNTS:
        raise ValueError(f"input_count must be one of {ALLOWED_INPUT_COUNTS}")
    target_device = torch.device(device)
    if target_device.type == "cuda" and target_device.index is None:
        raise ValueError("CUDA device must include an explicit index, for example cuda:0")

    checkpoint_path = Path(checkpoint).expanduser().resolve()
    output_path = _new_output_dir(output)
    history_path = output_path / "history.jsonl"
    history_path.touch()
    source_before = source_manifest()
    payload = _base_payload(
        checkpoint_path,
        output_path,
        str(target_device),
        input_count,
        source_before,
    )
    history: list[dict[str, Any]] = []
    _write_report(output_path / "report.json", payload)

    carrier = write_rule = renderer = runner = None
    warmup: list[OnlineObservation] = []
    step_index = -1
    stage = "startup"
    started = time.perf_counter()
    adaptation_started = None
    target_started = None
    gpu_info: dict[str, Any] = {}

    try:
        if target_device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
            torch.cuda.set_device(target_device)
        torch.manual_seed(SEED)
        if target_device.type == "cuda":
            torch.cuda.manual_seed_all(SEED)
        stage = "load_checkpoint"
        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)

        carrier, write_rule, checkpoint_metadata = load_dynamic_checkpoint(
            checkpoint_path, target_device
        )
        renderer = FixedMeasurementRenderer(
            n_samples=RENDER_SAMPLES,
            ray_chunk_size=RAY_CHUNK_SIZE,
        ).to(target_device)
        runner = StreamingRunner(
            carrier,
            write_rule,
            renderer,
            FixedPolicy(Action.ALL),
            max_units=MAX_UNITS,
        )
        intrinsics = make_intrinsics(IMAGE_SIZE, 60.0, device=target_device)
        x_grid = torch.linspace(
            0.0, 1.0, IMAGE_SIZE[1], device=target_device, dtype=torch.float32
        ).view(1, -1).expand(IMAGE_SIZE[0], -1)
        y_grid = torch.linspace(
            0.0, 1.0, IMAGE_SIZE[0], device=target_device, dtype=torch.float32
        ).view(-1, 1).expand(-1, IMAGE_SIZE[1])

        if target_device.type == "cuda":
            _sync_cuda(target_device)
            free_start, _ = torch.cuda.mem_get_info(target_device)
            torch.cuda.reset_peak_memory_stats(target_device)
            start_allocated = torch.cuda.memory_allocated(target_device)
            start_reserved = torch.cuda.memory_reserved(target_device)
            gpu = torch.cuda.get_device_properties(target_device)
            gpu_info.update(
                {
                    "name": gpu.name,
                    "total_memory_bytes": int(gpu.total_memory),
                    "torch_cuda_version": torch.version.cuda,
                    "pytorch_version": torch.__version__,
                    "free_start_bytes": int(free_start),
                    "start_allocated_bytes": int(start_allocated),
                    "start_reserved_bytes": int(start_reserved),
                }
            )

        adaptation_started = time.perf_counter()
        stage = "warmup"
        for frame_id in range(WARMUP_COUNT):
            observation = synthetic_observation(
                frame_id, x_grid, y_grid, intrinsics, target_device
            )
            warmup.append(observation)
            _append_history(
                history_path,
                history,
                {
                    "stage": "warmup",
                    "frame_id": frame_id,
                    "finite": finite_value(observation),
                    "image_size": list(IMAGE_SIZE),
                    "camera_rigid": _is_rigid_pose(observation.camera.c2w),
                },
            )
        runner.reset(
            warmup,
            episode_id=EPISODE_ID,
            scene_id=EPISODE_ID,
            split_id="synthetic",
            query_vault_id="synthetic-no-query",
            stream_steps=input_count - WARMUP_COUNT,
            query_count=1,
        )
        warmup.clear()
        finite_checks = payload["runtime"]["finite_checks"]
        reset_finite = finite_value(runner.scene_state) and finite_value(runner.fast)
        finite_checks.append({"stage": "after_reset", "finite": reset_finite})
        if not reset_finite:
            raise RuntimeError("nonfinite runtime state after warmup reset")

        for frame_id in range(WARMUP_COUNT, input_count):
            stage = "stream"
            step_index = frame_id
            observation = synthetic_observation(
                frame_id, x_grid, y_grid, intrinsics, target_device
            )
            record = runner.step(observation)
            finite_now = (
                finite_value(observation)
                and finite_value(runner.scene_state)
                and finite_value(runner.fast)
                and finite_value(record)
            )
            finite_checks.append({"stage": "stream", "frame_id": frame_id, "finite": finite_now})
            event = {
                "stage": "stream",
                "frame_id": int(record["frame_id"]),
                "action": str(record["action"]),
                "finite": finite_now,
                "record": record,
            }
            _append_history(history_path, history, event)
            payload["runtime"]["actions"].append(str(record["action"]))
            payload["runtime"]["completed_steps"] = len(payload["runtime"]["actions"])
            if not finite_now:
                raise RuntimeError(f"nonfinite runtime value at frame {frame_id}")
            del observation

        stage = "seal"
        sealed = runner.seal()
        sealed_finite = finite_value(sealed.scene_state) and finite_value(sealed.anchor_c2w)
        finite_checks.append({"stage": "after_seal", "finite": sealed_finite})
        _append_history(
            history_path,
            history,
            {
                "stage": "seal",
                "observed_ids": list(sealed.observed_ids),
                "state_hash": sealed.state_hash,
                "fast_state_hash": sealed.fast_state_hash,
                "checkpoint_model_hash": sealed.checkpoint_hash,
                "finite": sealed_finite,
            },
        )
        if not sealed_finite:
            raise RuntimeError("nonfinite sealed scene state")

        stage = "target_render"
        target_frame_id = input_count
        target_started = time.perf_counter()
        target_query = RGBQuery.for_sealed(
            sealed,
            target_frame_id,
            lambda: synthetic_camera(target_frame_id, intrinsics, target_device),
            lambda: synthetic_rgb(target_frame_id, x_grid, y_grid),
            image_size=IMAGE_SIZE,
        )
        metric_report = evaluate_sealed_rgb(
            sealed,
            [target_query],
            renderer,
            metrics=official_tttlrm_metrics(include_lpips=True),
            device=target_device,
        )
        _sync_cuda(target_device)
        target_elapsed = time.perf_counter() - target_started
        target_record = metric_report["per_image"][0]
        target_finite = finite_value(target_record)
        payload["target_render"] = {
            "status": "complete",
            "frame_id": target_frame_id,
            "image_size": list(IMAGE_SIZE),
            "elapsed_seconds": target_elapsed,
            "metrics": target_record,
            "metric_spec": metric_report["metric_spec"],
            "finite": target_finite,
        }
        finite_checks.append(
            {"stage": "target_render", "frame_id": target_frame_id, "finite": target_finite}
        )
        _append_history(
            history_path,
            history,
            {
                "stage": "target_render",
                "frame_id": target_frame_id,
                "image_size": list(IMAGE_SIZE),
                "elapsed_seconds": target_elapsed,
                "metrics": target_record,
                "finite": target_finite,
            },
        )
        if not target_finite:
            raise RuntimeError("target render metrics contain a nonfinite value")

        _sync_cuda(target_device)
        elapsed = time.perf_counter() - started
        if target_device.type == "cuda":
            free_end, _ = torch.cuda.mem_get_info(target_device)
            peak_allocated = torch.cuda.max_memory_allocated(target_device)
            peak_reserved = torch.cuda.max_memory_reserved(target_device)
            gpu_info.update(
                {
                    "free_end_bytes": int(free_end),
                    "peak_allocated_bytes": int(peak_allocated),
                    "peak_reserved_bytes": int(peak_reserved),
                    "peak_allocated_gib": float(peak_allocated / 2**30),
                    "peak_reserved_gib": float(peak_reserved / 2**30),
                    "peak_delta_allocated_bytes": int(peak_allocated - start_allocated),
                    "fits_observed_gpu_memory": bool(peak_reserved < gpu.total_memory),
                }
            )
        payload.update(
            {
                "status": "complete",
                "checkpoint_metadata": checkpoint_metadata,
                "runtime": {
                    **payload["runtime"],
                    "history": history,
                    "finite_checks": finite_checks,
                    "finite_state": all(item["finite"] for item in finite_checks),
                    "all_actions_are_ALL": all(
                        action == str(Action.ALL) for action in payload["runtime"]["actions"]
                    ),
                    "sealed_state_hash": sealed.state_hash,
                    "sealed_fast_state_hash": sealed.fast_state_hash,
                    "checkpoint_model_hash": runner.checkpoint_hash,
                    "elapsed_seconds": elapsed,
                    "adaptation_seconds": (
                        None
                        if adaptation_started is None
                        else float((target_started or time.perf_counter()) - adaptation_started)
                    ),
                    "target_render_seconds": target_elapsed,
                },
                "gpu": gpu_info,
                "resource_gate": {
                    "full_resolution": True,
                    "target_render_complete": True,
                    "finite_state": all(item["finite"] for item in finite_checks),
                    "fits_observed_gpu_memory": gpu_info.get("fits_observed_gpu_memory"),
                    "claim": (
                        "synthetic resource fit and renderer/metric availability only; "
                        "no benchmark performance claim"
                    ),
                },
            }
        )
    except torch.cuda.OutOfMemoryError as error:
        if stage == "target_render":
            payload["target_render"] = {
                "status": "oom",
                "frame_id": input_count,
            }
        payload.update(
            {
                "status": "oom",
                "failure": {
                    "type": type(error).__name__,
                    "message": str(error),
                    "stage": stage,
                    "step_index": step_index,
                    "traceback": traceback.format_exc(),
                },
            }
        )
    except Exception as error:
        if stage == "target_render":
            payload["target_render"] = {
                "status": "failed",
                "frame_id": input_count,
            }
        payload.update(
            {
                "status": "failed",
                "failure": {
                    "type": type(error).__name__,
                    "message": str(error),
                    "stage": stage,
                    "step_index": step_index,
                    "traceback": traceback.format_exc(),
                },
            }
        )
    finally:
        payload["runtime"]["history"] = history
        payload["runtime"]["finite_checks"] = payload["runtime"].get("finite_checks", [])
        payload["runtime"]["elapsed_seconds"] = time.perf_counter() - started
        try:
            source_after = source_manifest()
            payload["source"]["source_manifest_after"] = source_after
            payload["source"]["source_manifest_stable"] = (
                source_after["sha256"] == payload["source"]["source_manifest"]["sha256"]
            )
        except Exception as source_error:
            payload["source"]["source_manifest_after_error"] = str(source_error)
            payload["source"]["source_manifest_stable"] = False
        if checkpoint_path.is_file():
            try:
                checkpoint_after = sha256_file(checkpoint_path)
                payload["checkpoint"]["sha256_after"] = checkpoint_after
                payload["checkpoint"]["sha256_stable"] = (
                    checkpoint_after == payload["checkpoint"]["sha256"]
                )
            except Exception as checkpoint_error:
                payload["checkpoint"]["sha256_after_error"] = str(checkpoint_error)
                payload["checkpoint"]["sha256_stable"] = False
        carrier = write_rule = renderer = runner = None
        warmup.clear()
        gc.collect()
        if target_device.type == "cuda" and torch.cuda.is_available():
            try:
                _sync_cuda(target_device)
                peak_allocated = torch.cuda.max_memory_allocated(target_device)
                peak_reserved = torch.cuda.max_memory_reserved(target_device)
                gpu_info.update(
                    {
                        "peak_allocated_bytes": int(peak_allocated),
                        "peak_reserved_bytes": int(peak_reserved),
                        "peak_allocated_gib": float(peak_allocated / 2**30),
                        "peak_reserved_gib": float(peak_reserved / 2**30),
                    }
                )
                if "total_memory_bytes" in gpu_info:
                    gpu_info["fits_observed_gpu_memory"] = bool(
                        peak_reserved < gpu_info["total_memory_bytes"]
                    )
                torch.cuda.empty_cache()
                payload["runtime"]["allocated_after_cleanup_bytes"] = int(
                    torch.cuda.memory_allocated(target_device)
                )
            except Exception as cleanup_error:
                payload["runtime"]["cleanup_error"] = (
                    f"{type(cleanup_error).__name__}: {cleanup_error}"
                )
        if gpu_info:
            payload["gpu"] = gpu_info
        _write_report(output_path / "report.json", payload)

    return payload


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report = run_preflight(
            args.checkpoint,
            args.output,
            device=args.device,
            input_count=args.input_count,
        )
    except FileExistsError as error:
        print(f"ERROR: {error}", flush=True)
        return 2
    except (ValueError, RuntimeError) as error:
        print(f"ERROR: {error}", flush=True)
        return 2
    print(
        json.dumps(
            {
                "status": report["status"],
                "output": report["output"],
                "target_render": report["target_render"],
            },
            sort_keys=True,
            allow_nan=False,
        ),
        flush=True,
    )
    return 0 if report["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
