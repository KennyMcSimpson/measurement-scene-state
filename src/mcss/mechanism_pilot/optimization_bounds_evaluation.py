"""Post-all-budget-seal evaluation; immutable chains, no query-adaptive selection."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from mcss.dynamic.types import hash_scene_state, hash_value
from mcss.geometry import transform_cameras
from mcss.measurements import FixedMeasurementRenderer
from mcss.mechanism_pilot.direct_capacity_contracts import (
    CapacityEvaluator,
    ContextOnlyLoader,
    StateSealBarrier,
)
from mcss.mechanism_pilot.direct_capacity_evaluation import (
    _coverage,
    _metrics,
    _resolve,
    _summary_path,
)
from mcss.mechanism_pilot.small_training import sha, write_json


@torch.no_grad()
def evaluate_optimization_bounds(root, phase="context", device="cuda"):
    if phase not in ("context", "oracle", "secondary"):
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
        if (
            spec["track"]
            != {"context": "RGBD", "oracle": "QUERY_ORACLE", "secondary": "RGB_ONLY"}[phase]
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
    chains = {c["key"]: c for c in plan["chains"]}
    if len(chains) != len(plan["chains"]):
        raise PermissionError("Duplicate optimization chain")
    completion_paths, completions = [], {}
    for spec in specs:
        if spec["grid"] != 16 or spec["samples"] != 64:
            raise PermissionError("This experiment fixes grid16 and renderer64")
        if spec["selection"] not in ("FIXED_BUDGET", "CONTEXT_SELECTED"):
            raise PermissionError("Unknown frozen checkpoint-selection rule")
        if spec["chain_key"] not in chains:
            raise PermissionError("State has no frozen optimization chain")
        chain = chains[spec["chain_key"]]
        if chain["phase"] != phase or chain["scene_id"] != spec["scene_id"]:
            raise PermissionError("State/chain phase or scene mismatch")
        completion_path = _resolve(root, chain["optimization_path"]) / "completion.json"
        if not completion_path.is_file():
            raise PermissionError("Complete optimization chain required before media")
        if spec["chain_key"] not in completions:
            completion = json.loads(completion_path.read_text())
            expected_keys = {s["key"] for s in specs if s["chain_key"] == spec["chain_key"]}
            if (
                completion.get("phase") != phase
                or completion.get("chain_key") != spec["chain_key"]
                or completion.get("plan_sha256") != sha(plan_path)
                or completion.get("config_sha256") != sha(config_path)
                or completion.get("source_hashes") != source_hashes
                or set(completion.get("states", {})) != expected_keys
            ):
                raise PermissionError("Chain completion does not match frozen protocol")
            completions[spec["chain_key"]] = completion
            completion_paths.append(completion_path)
        item = completions[spec["chain_key"]]["states"][spec["key"]]
        summary = json.loads(_summary_path(root, spec).read_text())
        if (
            item.get("state_file_sha256") != sha(_resolve(root, spec["state_path"]))
            or item.get("summary_sha256") != sha(_summary_path(root, spec))
            or item.get("state_hash") != summary.get("state_hash")
        ):
            raise PermissionError("Selected budget state changed after chain completion")
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
        if (
            state.density_logits.shape[-3:] != (16,) * 3
            or state.features is not None
            or not torch.equal(
                state.bounds.cpu(),
                torch.tensor(spec["bounds"], dtype=state.bounds.dtype).reshape(1, 2, 3),
            )
        ):
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
        key = (
            spec["track"],
            spec["grid"],
            spec["samples"],
            spec["bounds_mode"],
            spec["role"],
            spec["budget"],
            spec["selection"],
        )
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
        if phase in ("context", "secondary"):
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
        controls = phase in ("context", "secondary")
        for fid in record["roles"]["primary_query"]:
            batch = evaluators[sid].query(fid)
            camera = transform_cameras(batch.cameras, inverse_anchor)
            camera_hash = hash_value(camera)
            reference = None
            methods = ["direct", "wrong_scene"] if controls else ["direct"]
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
                        spec["budget"],
                        spec["selection"],
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
                    raise ValueError("Only direct and matched wrong-scene controls allowed")
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
