"""Post-seal context-geometry supervision evaluation, no query-adaptive choices."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from mcss.dynamic.types import clone_scene_state, hash_scene_state, hash_value
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


def read(path):
    return json.loads(Path(path).read_text())


def ensure_phase(root, phase):
    root = Path(root)
    if phase not in ("baseline", "formal"):
        raise ValueError("Unknown supervision phase")
    manifest, plan, config = [
        read(root / f"{name}.json") for name in ("scene_manifest", "state_plan", "config")
    ]
    allowed = {r["scene_id"] for r in manifest["scenes"]}
    holdout = set(manifest["FINAL_HOLDOUT_PROHIBITED"])
    if len(allowed) != len(manifest["scenes"]) or allowed & holdout:
        raise PermissionError("Duplicate or prohibited scene")
    if any(r.get("cohort_role") != "CAPACITY_DEV_EXPOSED" for r in manifest["scenes"]):
        raise PermissionError("Only exposed attribution scenes allowed")
    all_specs = plan["states"]
    expected = {
        (sid, role, variant)
        for sid in allowed
        for role in ("A", "B")
        for variant in ("S0", "S1", "S2", "S3")
    }
    actual = {(s["scene_id"], s["role"], s["variant"]) for s in all_specs}
    if actual != expected or len(all_specs) != len(expected):
        raise PermissionError("Complete matched four-variant plan required")
    if len({s["key"] for s in all_specs}) != len(all_specs):
        raise PermissionError("Duplicate state key")
    specs = [s for s in all_specs if phase == "formal" or s["variant"] == "S0"]
    marker_path = root / f"{phase}_optimization_complete.json"
    if not marker_path.is_file():
        raise PermissionError("Complete phase marker required before any media")
    marker = read(marker_path)
    if (
        marker.get("status") != "PASS"
        or marker.get("phase") != phase
        or marker.get("states") != len(specs)
    ):
        raise PermissionError("Phase marker does not match frozen plan")
    if phase == "formal" and read(root / "baseline_reproduction.json").get("status") != "PASS":
        raise PermissionError("Baseline reproduction must PASS before formal evaluation")
    for path, digest in marker["input_sha256"].items():
        if sha(Path(path)) != digest:
            raise PermissionError("Frozen optimization input changed")
    for name, digest in read(root / "preregistration.json")["locked_file_sha256"].items():
        if sha(root / name) != digest:
            raise PermissionError(f"Preregistered input changed: {name}")
    for path, digest in config["source_sha256"].items():
        if sha(Path(path)) != digest:
            raise PermissionError(f"Frozen source changed: {path}")
    paths = [
        root / f"{n}.json" for n in ("scene_manifest", "state_plan", "config", "preregistration")
    ]
    paths.append(marker_path)
    for spec in specs:
        if spec["grid"] != 16 or spec["samples"] != 64:
            raise PermissionError("Only frozen 16-grid/64-sample states allowed")
        folder = _resolve(root, spec["optimization_path"])
        files = [folder / n for n in ("completion.json", "summary.json", "access.json")]
        state_path = _resolve(root, spec["state_path"])
        if not all(p.is_file() for p in files + [state_path]):
            raise PermissionError("All states/completions required before media reads")
        completion, summary = read(files[0]), read(files[1])
        checks = {
            "status": "PASS",
            "key": spec["key"],
            "plan_sha256": sha(root / "state_plan.json"),
            "config_sha256": sha(root / "config.json"),
            "source_sha256": config["source_sha256"],
            "state_file_sha256": sha(state_path),
            "summary_sha256": sha(files[1]),
            "access_sha256": sha(files[2]),
            "state_hash": summary["state_hash"],
            "query_selection": False,
        }
        if any(completion.get(k) != v for k, v in checks.items()):
            raise PermissionError("State completion differs from locked protocol")
        if (
            summary.get("selection") != "FIXED_BUDGET"
            or summary.get("selected_step") != 10000
            or summary.get("variant") != spec["variant"]
            or summary.get("supervision") != "CONTEXT_ONLY_RGBD"
        ):
            raise PermissionError("Only fixed-budget context RGBD states permitted")
        paths += files + [state_path]
    return manifest, plan, config, specs, paths


def diagnostic_state(state, method, seed=20260927):
    """Deterministic frozen controls; no renderer/state semantics changes."""
    out = clone_scene_state(state)
    if method == "zero":
        out.density_logits.fill_(-1e6)
        out.color.zero_()
    elif method == "spatial_shuffle":
        generator = torch.Generator(device="cpu").manual_seed(seed)
        permutation = torch.randperm(16**3, generator=generator).to(out.color.device)
        for name in ("density_logits", "color", "log_variance"):
            value = getattr(out, name)
            setattr(out, name, value.flatten(2)[:, :, permutation].reshape_as(value))
    else:
        raise ValueError("Unknown diagnostic state")
    return out


def baseline_comparison(root, rows, context_rows, config):
    comparisons = {}
    passed = True
    for name, current, id_key in (
        ("query", rows, "query_id"),
        ("context", context_rows, "frame_id"),
    ):
        historical = read(root / "raw" / f"historical_baseline_{name}.json")

        def keyed(values, id_key=id_key):
            out = {(r["scene_id"], r["role"], r[id_key]): r for r in values}
            if len(out) != len(values):
                raise PermissionError("Duplicate baseline reference identity")
            return out

        a, b = keyed(current), keyed(historical)
        if set(a) != set(b):
            raise PermissionError("Baseline/reference frame identity mismatch")
        differences = [abs(a[k]["depth_absrel"] - b[k]["depth_absrel"]) for k in a]

        # Equal scene, then role, then view weighting; every matched scene has both roles.
        def macro(values):
            scenes = sorted({k[0] for k in values})
            return float(
                np.mean(
                    [
                        np.mean(
                            [
                                np.mean(
                                    [
                                        v["depth_absrel"]
                                        for k, v in values.items()
                                        if k[0] == sid and k[1] == role
                                    ]
                                )
                                for role in ("A", "B")
                            ]
                        )
                        for sid in scenes
                    ]
                )
            )

        new_mean, old_mean = macro(a), macro(b)
        thresholds = config["baseline_reproduction"]
        ok = (
            abs(new_mean - old_mean) <= thresholds["mean_atol"]
            and max(differences) <= thresholds["per_row_atol"]
        )
        passed &= ok
        metrics = (
            "rgb_mse",
            "rgb_ssim",
            "depth_absrel",
            "depth_rmse",
            "depth_delta1",
            "opacity",
            "coverage",
        )
        comparisons[name] = {
            "status": "PASS" if ok else "FAIL",
            "rows": len(a),
            "current_scene_mean_absrel": new_mean,
            "historical_scene_mean_absrel": old_mean,
            "mean_absolute_difference": abs(new_mean - old_mean),
            "max_row_absrel_difference": max(differences),
            "max_metric_absolute_differences": {
                m: max(abs(a[k][m] - b[k][m]) for k in a) for m in metrics
            },
            "historical_raw_sha256": sha(root / "raw" / f"historical_baseline_{name}.json"),
        }
    report = {
        "status": "PASS" if passed else "FAIL",
        "comparisons": comparisons,
        "thresholds": config["baseline_reproduction"],
        "query_used_for_tuning": False,
        "failure_blocks_new_supervision": True,
    }
    write_json(root / "baseline_reproduction.json", report)
    return report


@torch.no_grad()
def evaluate_supervision(root, phase="baseline", device="cuda"):
    root = Path(root)
    manifest, plan, config, specs, input_paths = ensure_phase(root, phase)
    manifest_path = root / "scene_manifest.json"
    allowed = tuple(r["scene_id"] for r in manifest["scenes"])
    holdout = tuple(manifest["FINAL_HOLDOUT_PROHIBITED"])
    source_hashes = config["source_sha256"]
    if phase == "baseline":
        input_paths += [
            root / "raw" / f"historical_baseline_{kind}.json" for kind in ("query", "context")
        ]
    raw = root / "raw"
    result_path = raw / ("baseline_results.json" if phase == "baseline" else "query_results.json")
    prediction_dir = raw / f"{phase}_predictions"
    if result_path.exists() or prediction_dir.exists():
        raise FileExistsError("Preserve prior phase evaluation; refusing overwrite")
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
        key = (spec["role"], spec["variant"])
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
        if True:
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
                event["trained_variant"] = spec["variant"]
            camera = transform_cameras(batch.cameras, inverse_anchor)
            for j, fid in enumerate(batch.frame_ids):
                one_camera = camera.select_views(torch.tensor([j], device=device))
                if str(device).startswith("cuda"):
                    torch.cuda.synchronize()
                tick = time.perf_counter()
                pred = renderer(state, one_camera, ("rgb", "depth", "visibility"))
                if str(device).startswith("cuda"):
                    torch.cuda.synchronize()
                seconds = time.perf_counter() - tick
                context_rows.append(
                    {
                        **spec,
                        "frame_id": fid,
                        "method": "direct",
                        "seconds": seconds,
                        "camera_hash": hash_value(one_camera),
                        "used_state_hash": state_hash,
                        **_metrics(pred, batch.rgb[:, j : j + 1], batch.depth[:, j : j + 1]),
                    }
                )
            barrier.assert_ready()
        controls = phase == "formal"
        for fid in record["roles"]["primary_query"]:
            batch = evaluators[sid].query(fid)
            camera = transform_cameras(batch.cameras, inverse_anchor)
            camera_hash = hash_value(camera)
            reference = None
            methods = ["direct", "wrong_scene"] if controls else ["direct"]
            if phase == "formal" and spec["variant"] == "S3":
                methods += ["spatial_shuffle", "zero"]
            for method in methods:
                donor_id = None
                if method == "direct":
                    used_state = state
                elif method == "wrong_scene":
                    config_key = (spec["role"], spec["variant"])
                    group = configs[config_key]
                    ids = sorted(group)
                    if len(ids) < 2 or set(ids) != set(allowed):
                        raise PermissionError(
                            "Wrong-scene control requires matched states for every scene"
                        )
                    donor_id = ids[(ids.index(sid) + 1) % len(ids)]
                    used_state = barrier.state(group[donor_id]["key"]).to(device=device)
                elif method in ("spatial_shuffle", "zero"):
                    used_state = diagnostic_state(state, method)
                else:
                    raise ValueError("Unknown predeclared control")
                before = hash_scene_state(used_state)
                if str(device).startswith("cuda"):
                    torch.cuda.synchronize()
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
                row["prediction_file_sha256"] = sha(prediction_path)
                rows.append(row)
                assert hash_scene_state(used_state) == before and hash_value(camera) == camera_hash
            assert hash_scene_state(state) == state_hash
            barrier.assert_ready()
    barrier.assert_ready()
    assert all(sha(Path(path)) == digest for path, digest in input_hashes.items())
    assert all(sha(Path(p)) == h for p, h in source_hashes.items())
    assert all(sha(Path(p)) == h for p, h in evaluator_source_hashes.items())
    write_json(result_path, rows)
    write_json(
        raw / ("baseline_context_results.json" if phase == "baseline" else "context_results.json"),
        context_rows,
    )
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
            "context_RGBD_only_optimization": True,
            "final_holdout_touched": False,
            "dynamic_ttt_run": False,
            "new_carrier_trained": False,
            "seconds": time.perf_counter() - started,
        },
    )
    if phase == "baseline":
        baseline_comparison(root, rows, context_rows, config)
    return rows
