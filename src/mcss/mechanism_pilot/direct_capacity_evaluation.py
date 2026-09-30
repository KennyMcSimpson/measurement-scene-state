"""Post-seal direct-state evaluation and fixed diagnostic controls; no optimization."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from mcss.dynamic.types import clone_scene_state, hash_scene_state, hash_value
from mcss.geometry import generate_rays, intersect_aabb, transform_cameras
from mcss.measurements import FixedMeasurementRenderer
from mcss.mechanism_pilot.direct_capacity_contracts import (
    CapacityEvaluator,
    ContextOnlyLoader,
    StateSealBarrier,
)
from mcss.mechanism_pilot.small_training import sha, write_json
from mcss.mechanism_pilot.statistics import measurement_metrics


def _resolve(root, path):
    path = Path(path)
    return path if path.is_absolute() else root / path


def _summary_path(root, spec):
    path = _resolve(root, spec["optimization_path"])
    return path / "summary.json" if path.is_dir() else path


def _metrics(pred, rgb, depth):
    return measurement_metrics(
        pred["rgb"][0, 0],
        rgb[0, 0],
        pred["depth"][0, 0, 0],
        depth[0, 0, 0],
        pred["visibility"][0, 0, 0],
    )


def _coverage(state, camera, depth):
    origins, directions = generate_rays(camera)
    gt = depth[:, :, 0]
    valid = torch.isfinite(gt) & (gt > 0)
    points = origins + directions * torch.where(valid, gt, 0)[..., None]
    bounds = state.bounds[0]
    inside = ((points >= bounds[0]) & (points <= bounds[1])).all(-1)
    _, far, hit = intersect_aabb(origins, directions, bounds)
    return {
        "insideGTfraction": float(inside[valid].float().mean()),
        "ray_hitfraction": float(hit.float().mean()),
        "GTdistance_above_far_fraction": float((gt[valid] > far[valid]).float().mean()),
        "GTdistance_above_far_definition": "all GT-valid rays; a ray missing AABB has far=0",
        "valid_GT_pixels": int(valid.sum()),
    }


def _control(state, method):
    result = clone_scene_state(state)
    if method == "spatial_shuffle":
        generator = torch.Generator(device="cpu").manual_seed(20260927)
        count = state.density_logits[0, 0].numel()
        order = torch.randperm(count, generator=generator).to(state.density_logits.device)
        result.density_logits = state.density_logits.flatten(2)[:, :, order].reshape_as(
            state.density_logits
        )
        result.color = state.color.flatten(2)[:, :, order].reshape_as(state.color)
    elif method == "density_only":
        result.color = torch.full_like(state.color, 0.5)
    elif method == "color_only":
        result.density_logits = torch.full_like(state.density_logits, -2.0)
    elif method == "zero_density_color":
        result.density_logits = torch.full_like(state.density_logits, -100.0)
        result.color = torch.zeros_like(state.color)
    else:
        raise ValueError("Unknown state diagnostic")
    result.features = None
    return result


@torch.no_grad()
def evaluate_capacity(root, phase="context", device="cuda"):
    if phase not in ("context", "oracle"):
        raise ValueError("Unknown evaluation phase")
    root = Path(root)
    manifest_path, plan_path = root / "scene_manifest.json", root / "state_plan.json"
    manifest, plan = json.loads(manifest_path.read_text()), json.loads(plan_path.read_text())
    if len({r["scene_id"] for r in manifest["scenes"]}) != len(manifest["scenes"]):
        raise PermissionError("Duplicate capacity scene identity")
    allowed = tuple(r["scene_id"] for r in manifest["scenes"])
    holdout = tuple(manifest["FINAL_HOLDOUT_PROHIBITED"])
    if set(allowed) & set(holdout) or any(
        r.get("cohort_role") != "CAPACITY_DEV_EXPOSED" for r in manifest["scenes"]
    ):
        raise PermissionError("Only exposed capacity-dev scenes permitted; holdout stays closed")
    specs = [s for s in plan["states"] if s["phase"] == phase]
    if not specs or len({s["key"] for s in specs}) != len(specs):
        raise ValueError("Nonempty unique phase state plan required")
    for spec in specs:
        if spec["scene_id"] not in allowed or spec["scene_id"] in holdout:
            raise PermissionError("Planned state attempts holdout or unknown scene evaluation")
        if (phase == "context" and spec["track"] not in ("RGBD", "RGB_ONLY")) or (
            phase == "oracle" and spec["track"] != "QUERY_ORACLE"
        ):
            raise PermissionError("Oracle and context-only result phases must stay separate")
        if (
            not _resolve(root, spec["state_path"]).is_file()
            or not _summary_path(root, spec).is_file()
        ):
            raise PermissionError("Every planned optimization must finish before any query read")
    phase_completion_path = root / f"{phase}_optimization_complete.json"
    if not phase_completion_path.is_file():
        raise PermissionError("Phase optimization completion marker required before evaluator")
    phase_completion = json.loads(phase_completion_path.read_text())
    config_path = root / "config.json"
    config = json.loads(config_path.read_text())
    source_hashes = config["source_sha256"]
    if (
        phase_completion.get("status") != "PASS"
        or phase_completion.get("phase") != phase
        or phase_completion.get("states") != len(specs)
    ):
        raise PermissionError("Phase optimization completion does not match state plan")
    for path, digest in phase_completion["input_hashes"].items():
        if sha(Path(path)) != digest:
            raise PermissionError("Frozen optimization inputs changed")
    for path, digest in source_hashes.items():
        if sha(Path(path)) != digest:
            raise PermissionError("Frozen optimizer/loader source changed")
    completion_paths = []
    for spec in specs:
        completion_path = _summary_path(root, spec).parent / "completion.json"
        if not completion_path.is_file():
            raise PermissionError("State optimization completion record missing")
        completion = json.loads(completion_path.read_text())
        if (
            completion.get("plan_sha256") != sha(plan_path)
            or completion.get("state_file_sha256") != sha(_resolve(root, spec["state_path"]))
            or completion.get("spec") != spec
            or completion.get("phase") != phase
            or completion.get("source_hashes") != source_hashes
        ):
            raise PermissionError("State completion provenance differs from frozen plan")
        completion_paths.append(completion_path)
    raw = root / "raw"
    result_path = raw / f"{phase}_results.json"
    prediction_dir = raw / f"{phase}_predictions"
    if result_path.exists() or prediction_dir.exists():
        raise FileExistsError("Preserve prior phase evaluation; refusing overwrite")
    input_paths = [manifest_path, plan_path, config_path, phase_completion_path] + completion_paths
    input_paths += [_resolve(root, s["state_path"]) for s in specs]
    input_paths += [_summary_path(root, s) for s in specs]
    input_hashes = {str(p.resolve()): sha(p) for p in input_paths}
    barrier = StateSealBarrier(s["key"] for s in specs)
    records = {r["scene_id"]: r for r in manifest["scenes"]}
    for spec in specs:
        state = torch.load(
            _resolve(root, spec["state_path"]), map_location="cpu", weights_only=False
        )
        summary = json.loads(_summary_path(root, spec).read_text())
        if hash_scene_state(state) != summary["state_hash"]:
            raise PermissionError("Optimized state differs from checkpoint selection summary")
        if state.density_logits.shape[-3:] != (spec["grid"],) * 3 or state.features is not None:
            raise ValueError("Unexpected direct state grid/features")
        barrier.seal(spec["key"], spec["scene_id"], state)
    barrier.assert_ready()
    access = []
    for record in manifest["scenes"]:
        for frame in record["frames"]:
            for field in ("rgb", "depth"):
                path = _resolve(manifest_path.parent, frame[field])
                access.append(
                    {
                        "scene_id": record["scene_id"],
                        "frame_id": frame["frame_id"],
                        "purpose": "HASH_INPUT_AFTER_ALL_STATES_SEALED",
                        "channels": [field],
                        "path": str(path),
                    }
                )
                input_hashes[str(path.resolve())] = sha(path)
    evaluator_source_hashes = {str(Path(__file__).resolve()): sha(Path(__file__))}
    raw.mkdir(parents=True, exist_ok=True)
    prediction_dir.mkdir()
    write_json(
        raw / f"{phase}_evaluation_lock.json",
        {
            "phase": phase,
            "input_sha256": input_hashes,
            "optimizer_source_sha256": source_hashes,
            "evaluator_source_sha256": evaluator_source_hashes,
            "planned_state_keys": [s["key"] for s in specs],
            "all_states_sealed_before_evaluation": True,
            "final_holdout_touched": False,
            "new_carrier_trained": False,
            "dynamic_ttt_run": False,
        },
    )
    rows, context_rows = [], []
    evaluators = {
        sid: CapacityEvaluator(
            record,
            manifest["image_size"],
            manifest_path.parent,
            device,
            barrier=barrier,
            capacity_scene_ids=allowed,
            holdout_scene_ids=holdout,
            access_log=access,
        )
        for sid, record in records.items()
    }
    configs = {}
    for spec in specs:
        key = spec["track"], spec["grid"], spec["samples"], spec["bounds_mode"], spec["role"]
        configs.setdefault(key, {})[spec["scene_id"]] = spec
    started = time.perf_counter()
    for spec_index, spec in enumerate(specs):
        sid, record = spec["scene_id"], records[spec["scene_id"]]
        frozen = barrier.state(spec["key"])
        state = frozen.to(device=device)
        state_hash = hash_scene_state(state)
        renderer = FixedMeasurementRenderer(n_samples=spec["samples"], ray_chunk_size=2048).to(
            device
        )
        anchor_frame = next(
            f for f in record["frames"] if f["frame_id"] == record["roles"]["context_a"][0]
        )
        anchor = torch.tensor(anchor_frame["c2w"], dtype=torch.float32, device=device)
        inverse_anchor = torch.linalg.inv(anchor)
        if phase == "context":
            # Post-seal evaluator may measure depth even for RGB-only fitted states.
            barrier.assert_ready()
            first_event = len(access)
            batch = ContextOnlyLoader(
                record,
                manifest["image_size"],
                manifest_path.parent,
                device,
                track="RGBD",
                capacity_scene_ids=allowed,
                holdout_scene_ids=holdout,
                access_log=access,
            ).context(spec["role"])
            for event in access[first_event:]:
                event["purpose"] = "POST_SEAL_CONTEXT_EVALUATION_NOT_OPTIMIZER"
                event["trained_track"] = spec["track"]
            camera = transform_cameras(batch.cameras, inverse_anchor)
            for j, fid in enumerate(batch.frame_ids):
                one_camera = camera.select_views(torch.tensor([j], device=device))
                pred = renderer(state, one_camera, ("rgb", "depth", "visibility"))
                context_rows.append(
                    {
                        **spec,
                        "frame_id": fid,
                        "method": "direct",
                        **_metrics(pred, batch.rgb[:, j : j + 1], batch.depth[:, j : j + 1]),
                    }
                )
            barrier.assert_ready()
        controls = (
            phase == "context"
            and spec["bounds_mode"] == "CURRENT_BOUNDS"
            and spec["grid"] == 8
            and spec["samples"] == 64
        )
        for fid in record["roles"]["primary_query"]:
            batch = evaluators[sid].query(fid)
            camera = transform_cameras(batch.cameras, inverse_anchor)
            camera_hash = hash_value(camera)
            reference = None
            methods = ["direct"] + (
                [
                    "spatial_shuffle",
                    "density_only",
                    "color_only",
                    "zero_density_color",
                    "wrong_scene",
                ]
                if controls
                else []
            )
            for method in methods:
                donor_id = None
                if method == "direct":
                    used_state = state
                elif method == "wrong_scene":
                    config_key = (
                        spec["track"],
                        spec["grid"],
                        spec["samples"],
                        spec["bounds_mode"],
                        spec["role"],
                    )
                    group = configs[config_key]
                    ids = sorted(group)
                    if len(ids) < 2 or set(ids) != set(allowed):
                        raise PermissionError(
                            "Wrong-scene control requires matched states for every scene"
                        )
                    donor_id = ids[(ids.index(sid) + 1) % len(ids)]
                    used_state = barrier.state(group[donor_id]["key"]).to(device=device)
                else:
                    used_state = _control(state, method)
                before = hash_scene_state(used_state)
                tick = time.perf_counter()
                pred = renderer(used_state, camera, ("rgb", "depth", "visibility"))
                if str(device).startswith("cuda"):
                    torch.cuda.synchronize()
                seconds = time.perf_counter() - tick
                if method == "direct":
                    reference = {k: pred[k].clone() for k in ("rgb", "depth")}
                row = {
                    **spec,
                    "query_id": fid,
                    "method": method,
                    "donor_scene_id": donor_id,
                    "query_camera_hash": camera_hash,
                    "used_state_hash": before,
                    "seconds": seconds,
                    "prediction_hash": hash_value(pred),
                    **_metrics(pred, batch.rgb, batch.depth),
                    **_coverage(used_state, camera, batch.depth),
                    "prediction_change_from_direct": {
                        k: {
                            "rms": float((pred[k] - reference[k]).square().mean().sqrt()),
                            "max_abs": float((pred[k] - reference[k]).abs().max()),
                        }
                        for k in ("rgb", "depth")
                    },
                }
                prediction_path = prediction_dir / f"{spec_index:04d}_{fid}_{method}.npz"
                np.savez_compressed(
                    prediction_path,
                    rgb=pred["rgb"][0, 0].cpu().numpy(),
                    depth=pred["depth"][0, 0, 0].cpu().numpy(),
                    opacity=pred["visibility"][0, 0, 0].cpu().numpy(),
                )
                row["prediction_path"] = str(prediction_path.relative_to(root))
                rows.append(row)
                assert hash_scene_state(used_state) == before and hash_value(camera) == camera_hash
            assert hash_scene_state(state) == state_hash
            barrier.assert_ready()
    barrier.assert_ready()
    assert all(sha(Path(path)) == digest for path, digest in input_hashes.items())
    assert all(sha(Path(p)) == h for p, h in source_hashes.items())
    assert all(sha(Path(p)) == h for p, h in evaluator_source_hashes.items())
    write_json(result_path, rows)
    write_json(raw / f"{phase}_context_results.json", context_rows)
    write_json(raw / f"{phase}_GT_access.json", access)
    write_json(raw / f"{phase}_seal_events.json", barrier.events)
    write_json(
        raw / f"{phase}_evaluation_integrity.json",
        {
            "status": "PASS",
            "phase": phase,
            "n_states": len(specs),
            "n_query_rows": len(rows),
            "n_context_rows": len(context_rows),
            "all_states_immutable": True,
            "all_frozen_inputs_unchanged": True,
            "query_access_after_all_phase_states_sealed": True,
            "RGB_ONLY_depth_read_only_postseal_by_evaluator": True,
            "final_holdout_touched": False,
            "dynamic_ttt_run": False,
            "new_carrier_trained": False,
            "seconds": time.perf_counter() - started,
        },
    )
    return rows
