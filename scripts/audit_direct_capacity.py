"""Post-run direct-capacity audit from states, optimization logs and saved predictions.

No optimizer or renderer is invoked. Both phases must be complete before any audit
GT bytes are read, and protected holdout identities are permanently excluded.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from mcss.dynamic.types import hash_scene_state
from mcss.mechanism_pilot.small_training import sha, write_json
from mcss.mechanism_pilot.statistics import measurement_metrics

METRICS = (
    "rgb_mse",
    "rgb_ssim",
    "depth_absrel",
    "depth_rmse",
    "depth_delta1",
    "opacity",
    "coverage",
)
CHECKPOINT_HASH = "be7b8b6d2ef366cad245732b9ff227802db596da65082ac0e160f24541579e42"


def read(path):
    return json.loads(Path(path).read_text())


def location(root, path):
    p = Path(path)
    return p if p.is_absolute() else root / p


def check_hashes(mapping):
    for path, digest in mapping.items():
        assert sha(Path(path)) == digest, f"Frozen file changed: {path}"


def run(root, checkpoint):
    root = Path(root)
    destination = root / "audit/postrun_validation.json"
    if destination.exists():
        raise FileExistsError(destination)
    specs = read(root / "state_plan.json")["states"]
    assert len(specs) == 425 and len({s["key"] for s in specs}) == 425
    manifest = read(root / "scene_manifest.json")
    records = {r["scene_id"]: r for r in manifest["scenes"]}
    holdout = set(manifest["FINAL_HOLDOUT_PROHIBITED"])
    assert len(records) == 17 and len(holdout) == 12 and not set(records) & holdout
    assert all(r["cohort_role"] == "CAPACITY_DEV_EXPOSED" for r in records.values())
    preregistration = read(root / "preregistration.json")
    assert preregistration["config_sha256"] == sha(root / "config.json")
    assert preregistration["state_plan_sha256"] == sha(root / "state_plan.json")
    config = read(root / "config.json")
    assert config["state_plan_sha256"] == sha(root / "state_plan.json")
    assert config["scene_manifest_sha256"] == sha(root / "scene_manifest.json")
    check_hashes(config["source_sha256"])
    # All context and privileged-oracle optimization/evaluation phases must already exist.
    for phase in ("context", "oracle"):
        complete = read(root / f"{phase}_optimization_complete.json")
        assert complete["status"] == "PASS" and complete["phase"] == phase
        assert complete["states"] == sum(s["phase"] == phase for s in specs)
        check_hashes(complete["input_hashes"])
        integrity = read(root / "raw" / f"{phase}_evaluation_integrity.json")
        assert integrity["status"] == "PASS" and not integrity["final_holdout_touched"]
        assert not integrity["dynamic_ttt_run"] and not integrity["new_carrier_trained"]
        assert integrity["all_states_immutable"] and integrity["all_frozen_inputs_unchanged"]

    def optimizer_directory(spec):
        path = location(root, spec["optimization_path"])
        return path if path.is_dir() else path.parent

    context_completion_times = {
        spec["key"]: (optimizer_directory(spec) / "completion.json").stat().st_mtime_ns
        for spec in specs
        if spec["phase"] == "context"
    }
    context_eval_lock_time = (root / "raw/context_evaluation_lock.json").stat().st_mtime_ns
    context_eval_integrity_time = (
        (root / "raw/context_evaluation_integrity.json").stat().st_mtime_ns
    )
    oracle_lock_times = {
        spec["key"]: (optimizer_directory(spec) / "lock.json").stat().st_mtime_ns
        for spec in specs
        if spec["phase"] == "oracle"
    }
    oracle_complete_time = (root / "oracle_optimization_complete.json").stat().st_mtime_ns
    oracle_eval_lock_time = (root / "raw/oracle_evaluation_lock.json").stat().st_mtime_ns
    assert all(t <= context_eval_lock_time for t in context_completion_times.values())
    assert context_eval_integrity_time <= min(oracle_lock_times.values())
    assert oracle_complete_time <= oracle_eval_lock_time
    chronology = {
        "unit": "filesystem st_mtime_ns",
        "limitation": (
            "Auxiliary chronology evidence only; "
            "OS file timestamps are mutable and not tamper-proof"
        ),
        "latest_context_state_completion": max(context_completion_times.values()),
        "context_evaluation_lock": context_eval_lock_time,
        "context_evaluation_integrity": context_eval_integrity_time,
        "first_oracle_optimization_lock": min(oracle_lock_times.values()),
        "oracle_optimization_complete": oracle_complete_time,
        "oracle_evaluation_lock": oracle_eval_lock_time,
        "context_completion_before_context_evaluation": True,
        "context_evaluation_before_oracle_optimization": True,
        "oracle_completion_before_oracle_evaluation": True,
    }
    baseline = read(root / "baseline_integrity.json")
    assert baseline["status"] == "PASS" and baseline["checkpoint_sha256"] == CHECKPOINT_HASH
    assert sha(Path(checkpoint)) == CHECKPOINT_HASH
    state_hashes, optimization_locks, optimization_checks = {}, {}, []
    for spec in specs:
        sid, phase, track = spec["scene_id"], spec["phase"], spec["track"]
        assert sid in records and sid not in holdout
        directory = location(root, spec["optimization_path"])
        if not directory.is_dir():
            directory = directory.parent
        completed, summary, locked = (
            read(directory / name) for name in ("completion.json", "summary.json", "lock.json")
        )
        assert completed["spec"] == spec and completed["phase"] == phase
        assert completed["plan_sha256"] == sha(root / "state_plan.json")
        assert completed["state_file_sha256"] == sha(location(root, spec["state_path"]))
        assert completed["source_hashes"] == config["source_sha256"]
        assert locked["config"] == config["optimization"]
        assert locked["grid"] == spec["grid"] and locked["renderer_samples"] == spec["samples"]
        assert np.allclose(locked["bounds"], spec["bounds"], atol=1e-6, rtol=1e-6)
        assert locked["renderer_kernel_unchanged"] and locked["carrier_loaded"] is False
        record = records[sid]
        query_ids = record["roles"]["primary_query"]
        if phase == "context":
            expected_ids = record["roles"]["context_a" if spec["role"] == "A" else "context_b"]
            assert track in ("RGBD", "RGB_ONLY") and set(expected_ids).isdisjoint(query_ids)
        else:
            expected_ids = query_ids
            assert track == "QUERY_ORACLE"
        assert locked["supervision_frame_ids"] == expected_ids
        assert summary["shared_supervision_frame_ids"] == expected_ids
        assert locked["has_depth"] == summary["has_depth"] == (track != "RGB_ONLY")
        accesses = read(directory / "access.json")
        for event in accesses:
            assert event["scene_id"] == sid and sid not in holdout
            if phase == "context":
                assert event["frame_id"] in expected_ids
                assert event["purpose"] == f"CONTEXT_ONLY_{track}"
                if track == "RGB_ONLY":
                    assert "depth" not in event["channels"]
            else:
                if event["frame_id"] in query_ids:
                    assert event["purpose"] == "DIAGNOSTIC_QUERY_SUPERVISED_ORACLE_NOT_CONTEXT_ONLY"
                else:
                    assert event["frame_id"] == record["roles"]["context_a"][0]
                    assert (
                        event["purpose"] == "CONTEXT_ONLY_RGB_ONLY"
                        and "depth" not in event["channels"]
                    )
        checks = summary["full_objective_checkpoints"]
        expected_steps = [0] + list(
            range(
                config["optimization"]["checkpoint_interval"],
                config["optimization"]["steps"] + 1,
                config["optimization"]["checkpoint_interval"],
            )
        )
        if expected_steps[-1] != config["optimization"]["steps"]:
            expected_steps.append(config["optimization"]["steps"])
        assert [c["step"] for c in checks] == expected_steps
        assert all(np.isfinite(c["context_objective"]) for c in checks)
        best = min(checks, key=lambda c: (c["context_objective"], c["step"]))
        assert best["step"] == summary["selected_step"]
        assert np.isclose(
            best["context_objective"],
            summary["selected"]["context_objective"],
            atol=1e-6,
            rtol=1e-5,
        )
        assert summary["selection_used_only_supervision_objective"] and summary["all_steps_finite"]
        state = torch.load(
            location(root, spec["state_path"]), map_location="cpu", weights_only=False
        )
        assert hash_scene_state(state) == summary["state_hash"]
        assert state.features is None and state.density_logits.shape[-3:] == (spec["grid"],) * 3
        state_hashes[spec["key"]] = summary["state_hash"]
        optimization_locks[spec["key"]] = locked
        optimization_checks.append(
            {
                "key": spec["key"],
                "selected_step": best["step"],
                "earliest_minimum_objective": best["context_objective"],
            }
        )
    # Inspect the actual frozen plan/optimization locks, not only synthetic config contracts.
    for sid in records:
        for role in ("A", "B"):
            for track in ("RGBD", "RGB_ONLY"):
                group = [
                    s
                    for s in specs
                    if s["phase"] == "context"
                    and s["scene_id"] == sid
                    and s["role"] == role
                    and s["track"] == track
                    and s["bounds_mode"] == "CURRENT_BOUNDS"
                    and s["samples"] == 64
                ]
                assert sorted(s["grid"] for s in group) == [8, 16, 32]
                reference = optimization_locks[group[0]["key"]]
                for spec in group[1:]:
                    locked = optimization_locks[spec["key"]]
                    for field in (
                        "config",
                        "seed",
                        "bounds",
                        "renderer_samples",
                        "supervision_frame_ids",
                        "has_depth",
                        "sampling",
                    ):
                        assert locked[field] == reference[field]
                    assert (
                        records[spec["scene_id"]]["roles"]["primary_query"]
                        == records[sid]["roles"]["primary_query"]
                    )
        diagnostic = [s for s in specs if s["scene_id"] == sid and s["samples"] == 128]
        assert len(diagnostic) == 4
        for spec in diagnostic:
            assert spec["track"] == "RGBD" and spec["grid"] in (8, 16)
            assert spec["bounds_mode"] == "CURRENT_BOUNDS" and spec["phase"] == "context"
            reference_spec = next(
                s
                for s in specs
                if s["scene_id"] == sid
                and s["role"] == spec["role"]
                and s["track"] == "RGBD"
                and s["grid"] == spec["grid"]
                and s["samples"] == 64
                and s["bounds_mode"] == "CURRENT_BOUNDS"
                and s["phase"] == "context"
            )
            for field in (
                "config",
                "seed",
                "bounds",
                "supervision_frame_ids",
                "has_depth",
                "sampling",
            ):
                assert (
                    optimization_locks[spec["key"]][field]
                    == optimization_locks[reference_spec["key"]][field]
                )
    audit_access, truth_cache, phases = [], {}, {}
    for phase in ("context", "oracle"):
        phase_specs = [s for s in specs if s["phase"] == phase]
        seal_events = read(root / "raw" / f"{phase}_seal_events.json")
        n = len(phase_specs)
        assert all(e["event"] == "seal" for e in seal_events[:n])
        assert {e["state_id"] for e in seal_events[:n]} == {s["key"] for s in phase_specs}
        assert all(
            e["event"] == "query_camera_GT" and e["scene_id"] not in holdout
            for e in seal_events[n:]
        )
        for event in seal_events[:n]:
            assert event["state_hash"] == state_hashes[event["state_id"]]
        lock = read(root / "raw" / f"{phase}_evaluation_lock.json")
        assert lock["all_states_sealed_before_evaluation"] and not lock["final_holdout_touched"]
        check_hashes(lock["input_sha256"])
        check_hashes(lock["optimizer_source_sha256"])
        check_hashes(lock["evaluator_source_sha256"])
        access = read(root / "raw" / f"{phase}_GT_access.json")
        assert all(e["scene_id"] in records and e["scene_id"] not in holdout for e in access)
        assert all(
            e["purpose"]
            in (
                "HASH_INPUT_AFTER_ALL_STATES_SEALED",
                "POST_SEAL_CONTEXT_EVALUATION_NOT_OPTIMIZER",
                "SEALED_QUERY_EVALUATION",
            )
            for e in access
        )
        rows = read(root / "raw" / f"{phase}_results.json")
        direct_rows = defaultdict(list)
        groups = defaultdict(list)
        errors, exact = dict.fromkeys(METRICS, 0.0), dict.fromkeys(METRICS, 0)
        for row in rows:
            sid, fid = row["scene_id"], row["query_id"]
            assert sid in records and sid not in holdout
            assert fid in records[sid]["roles"]["primary_query"]
            groups[row["key"], fid].append(row)
            if row["method"] == "direct":
                direct_rows[row["key"]].append(row)
                assert row["used_state_hash"] == state_hashes[row["key"]]
            if row["method"] == "wrong_scene":
                assert row["donor_scene_id"] != sid and row["donor_scene_id"] in records
                donor = next(
                    s
                    for s in phase_specs
                    if s["scene_id"] == row["donor_scene_id"]
                    and all(
                        s[k] == row[k] for k in ("role", "track", "grid", "samples", "bounds_mode")
                    )
                )
                assert row["used_state_hash"] == state_hashes[donor["key"]]
            if (sid, fid) not in truth_cache:
                frame = next(f for f in records[sid]["frames"] if f["frame_id"] == fid)
                rgb_path, depth_path = (location(root, frame[k]) for k in ("rgb", "depth"))
                assert not any(
                    identity in str(rgb_path) or identity in str(depth_path) for identity in holdout
                )
                with Image.open(rgb_path) as im:
                    rgb = np.asarray(im.convert("RGB"), dtype=np.float32) / 255
                depth = np.load(depth_path).squeeze()
                truth_cache[sid, fid] = rgb, depth
                audit_access.append(
                    {
                        "scene_id": sid,
                        "query_id": fid,
                        "purpose": "POSTRUN_NPZ_RECOMPUTATION",
                        "rgb_path": str(rgb_path),
                        "depth_path": str(depth_path),
                        "final_holdout": False,
                    }
                )
            rgb, depth = truth_cache[sid, fid]
            with np.load(location(root, row["prediction_path"])) as pred:
                metrics = measurement_metrics(
                    pred["rgb"], rgb, pred["depth"], depth, pred["opacity"]
                )
            for metric in METRICS:
                difference = abs(metrics[metric] - row[metric])
                errors[metric] = max(errors[metric], difference)
                exact[metric] += metrics[metric] == row[metric]
                assert np.isclose(metrics[metric], row[metric], atol=1e-6, rtol=1e-5)
        assert len({(r["key"], r["query_id"], r["method"]) for r in rows}) == len(rows)
        expected_row_count = 0
        for spec in phase_specs:
            controls = (
                phase == "context"
                and spec["grid"] == 8
                and spec["samples"] == 64
                and spec["bounds_mode"] == "CURRENT_BOUNDS"
            )
            expected_methods = {"direct"} | (
                {
                    "spatial_shuffle",
                    "density_only",
                    "color_only",
                    "zero_density_color",
                    "wrong_scene",
                }
                if controls
                else set()
            )
            query_ids = records[spec["scene_id"]]["roles"]["primary_query"]
            expected_row_count += len(query_ids) * len(expected_methods)
            for fid in query_ids:
                assert {r["method"] for r in groups[spec["key"], fid]} == expected_methods
        assert len(rows) == expected_row_count
        assert set(direct_rows) == {s["key"] for s in phase_specs}
        for spec in phase_specs:
            selected = direct_rows[spec["key"]]
            assert sorted(r["query_id"] for r in selected) == sorted(
                records[spec["scene_id"]]["roles"]["primary_query"]
            )
            assert len({r["used_state_hash"] for r in selected}) == 1
        assert all(len({r["query_camera_hash"] for r in rr}) == 1 for rr in groups.values())
        phases[phase] = {
            "states": n,
            "rows_recomputed": len(rows),
            "metrics_max_absolute_error": errors,
            "metric_exact_row_counts": exact,
            "all_evaluator_queries_after_complete_seal": True,
            "shared_state_across_queries": True,
            "wrong_scene_preserves_camera": True,
        }
    check_hashes(config["source_sha256"])
    assert sha(Path(checkpoint)) == CHECKPOINT_HASH
    write_json(root / "audit/postrun_GT_access.json", audit_access)
    report = {
        "status": "PASS",
        "optimized_state_count": len(specs),
        "preregistration_config_and_state_plan_hashes_match": True,
        "filesystem_chronology_auxiliary": chronology,
        "completion_plan_state_source_hashes_match": True,
        "all_selected_checkpoints_minimum_full_objective_earliest_tie": True,
        "context_role_ids_only": True,
        "RGB_ONLY_optimizer_never_reads_depth": True,
        "query_oracle_explicitly_separate": True,
        "actual_resolution_sweep_only_grid_changes": True,
        "actual_samples_diagnostic_only_8_16_RGBD": True,
        "phases": phases,
        "optimization_checks": optimization_checks,
        "baseline_checkpoint_unchanged": True,
        "baseline_checkpoint_sha256": CHECKPOINT_HASH,
        "final_holdout_touched": False,
        "dynamic_ttt_run": False,
        "new_carrier_trained": False,
        "metric_recomputation_tolerance": {"atol": 1e-6, "rtol": 1e-5},
        "statistics_reproduction": "SEPARATE_INDEPENDENT_ANALYSIS_AUDIT",
        "audit_script_sha256": sha(Path(__file__)),
    }
    write_json(destination, report)
    return {"status": "PASS", "optimized_state_count": len(specs), "phases": phases}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument(
        "--checkpoint",
        default="outputs/EXP-3D-20260927-centered-training-1000a-600b-v1/training/phase_b_final.pt",
    )
    args = parser.parse_args()
    print(json.dumps(run(args.root, args.checkpoint), indent=2))
