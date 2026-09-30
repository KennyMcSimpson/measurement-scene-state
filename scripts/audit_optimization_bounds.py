"""Independent post-run budget/bounds audit; saved artifacts only, no model execution."""

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from mcss.dynamic.types import hash_scene_state, hash_value
from mcss.geometry import transform_cameras
from mcss.mechanism_pilot.small_training import sha, write_json
from mcss.mechanism_pilot.statistics import measurement_metrics
from mcss.types import Cameras

METRICS = (
    "rgb_mse",
    "rgb_ssim",
    "depth_absrel",
    "depth_rmse",
    "depth_delta1",
    "opacity",
    "coverage",
)
PHASES = ("context", "oracle", "secondary")


def read(path):
    return json.loads(Path(path).read_text())


def assert_hashes(mapping):
    for path, value in mapping.items():
        assert sha(Path(path)) == value, f"Frozen hash changed: {path}"


def independent_plateau(checks, trace, budget):
    window = [r for r in checks if 0.9 * budget <= r["step"] <= budget]
    gradients = [
        float(r["gradient_norm_before_clip"])
        for r in trace
        if 0.9 * budget <= int(r["step"]) <= budget
    ]
    assert len(window) >= 2 and len(gradients) >= 4
    objective_change = (window[0]["context_objective"] - window[-1]["context_objective"]) / max(
        abs(window[0]["context_objective"]), 1e-8
    )
    half = len(gradients) // 2
    before, after = float(np.median(gradients[:half])), float(np.median(gradients[half:]))
    gradient_change = abs(after - before) / max(abs(before), 1e-8)
    return {
        "relative_objective_improvement": objective_change,
        "relative_gradient_median_abs_change": gradient_change,
        "plateau": abs(objective_change) <= 0.001 and gradient_change <= 0.10,
        "n_trace_gradients": len(gradients),
        "checkpoint_steps": [r["step"] for r in window],
    }


def run(root):
    root = Path(root)
    destination = root / "audit/postrun_validation.json"
    if destination.exists():
        raise FileExistsError(destination)
    config, plan, manifest, prereg = (
        read(root / f"{name}.json")
        for name in ("config", "state_plan", "scene_manifest", "preregistration")
    )
    for name, digest in prereg["locked_file_sha256"].items():
        assert sha(root / name) == digest
    assert_hashes(config["source_sha256"])
    assert config["state_plan_sha256"] == sha(root / "state_plan.json")
    assert config["scene_manifest_sha256"] == sha(root / "scene_manifest.json")
    records = {r["scene_id"]: r for r in manifest["scenes"]}
    holdout = set(manifest["FINAL_HOLDOUT_PROHIBITED"])
    assert len(records) == 17 and len(holdout) == 12 and not set(records) & holdout
    assert all(r["cohort_role"] == "CAPACITY_DEV_EXPOSED" for r in records.values())
    chains, states = plan["chains"], plan["states"]
    aggregate_path = root / "raw/trajectory_summaries.json"
    if aggregate_path.exists():
        aggregate = read(aggregate_path)
        assert set(aggregate) == {c["key"] for c in chains}
        assert all(aggregate[c["key"]]["spec"] == c for c in chains)

    assert len(chains) == 153 and len(states) == 782
    assert len({c["key"] for c in chains}) == 153 and len({s["key"] for s in states}) == 782
    assert config["primary"] == {
        "track": "RGBD",
        "bounds_mode": "FROZEN_GT_FREE_BOUNDS",
        "budget": 10000,
        "selection": "FIXED_BUDGET",
    }
    budget_lock = read(root / "optimization_budget_lock.json")
    assert budget_lock["budgets"] == [1000, 3000, 10000]
    assert budget_lock["optimizer_loss_unchanged"] == config["optimization"]
    for phase in PHASES:
        complete = read(root / f"{phase}_optimization_complete.json")
        assert complete["status"] == "PASS" and complete["phase"] == phase
        assert complete["chains"] == sum(c["phase"] == phase for c in chains)
        assert complete["states"] == sum(s["phase"] == phase for s in states)
        assert complete["all_budgets_complete_before_query_evaluation"]
        assert not complete["final_holdout_touched"] and not complete["dynamic_ttt_run"]
        assert_hashes(complete["input_hashes"])
        result = read(root / "raw" / f"{phase}_evaluation_integrity.json")
        assert result["status"] == "PASS" and result["all_states_immutable"]
        assert result["all_frozen_inputs_unchanged"] and not result["final_holdout_touched"]
        assert not result["new_carrier_trained"] and not result["dynamic_ttt_run"]
    prior_seal = read(root / "audit/previous_experiment_seal.json")
    for path, item in prior_seal["files"].items():
        assert Path(path).stat().st_size == item["bytes"] and sha(Path(path)) == item["sha256"]
    old_checkpoint = Path(
        "outputs/EXP-3D-20260927-centered-training-1000a-600b-v1/training/phase_b_final.pt"
    )
    assert sha(old_checkpoint) == prior_seal["checkpoint_sha256"]
    bounds = read(root / "bounds_lock.json")
    old_bounds = read("outputs/EXP-3D-DIRECT-STATE-CAPACITY-V1/predeclared_geometry_bounds.json")
    assert len(bounds["bounds"]) == 34 and not bounds["current_scene_depth_used"]
    assert not bounds["query_camera_or_GT_used"] and bounds["all34_previous_alternatives_bit_exact"]
    assert bounds["prior_sha256"] == sha(root / "train_depth_prior.json")
    assert bounds["contract_sha256"] == sha(root / "bounds_contract.json")
    assert bounds["scene_manifest_sha256"] == sha(root / "scene_manifest.json")
    for key, value in bounds["bounds"].items():
        assert value["FROZEN_GT_FREE_BOUNDS"] == old_bounds["bounds"][key]["bounds"]
        assert value["CURRENT_BOUNDS"] == [[-6.0, -4.0, -6.0], [6.0, 4.0, 6.0]]
    selected_checks, plateau_checks, state_hashes, chain_locks = [], [], {}, {}
    for chain in chains:
        key, sid, phase = chain["key"], chain["scene_id"], chain["phase"]
        assert sid in records and sid not in holdout
        assert chain["grid"] == 16 and chain["samples"] == 64
        assert (
            chain["track"]
            == {"context": "RGBD", "oracle": "QUERY_ORACLE", "secondary": "RGB_ONLY"}[phase]
        )
        assert chain["budgets"] == ([10000] if phase == "secondary" else [1000, 3000, 10000])
        if phase == "secondary":
            assert chain["bounds_mode"] == config["primary"]["bounds_mode"]
        geometry_key = f"{sid}/{'A' if chain['role'] == 'joint_query' else chain['role']}"
        assert chain["bounds"] == bounds["bounds"][geometry_key][chain["bounds_mode"]]
        directory = root / chain["optimization_path"]
        completion, locked, summary, checkpoints = (
            read(directory / name)
            for name in (
                "completion.json",
                "lock.json",
                "trajectory_summary.json",
                "full_objective_checkpoints.json",
            )
        )
        assert completion["chain_key"] == key and completion["phase"] == phase
        assert completion["plan_sha256"] == sha(root / "state_plan.json")
        assert completion["config_sha256"] == sha(root / "config.json")
        assert completion["source_hashes"] == config["source_sha256"]
        assert not completion["query_selection"] and not completion["final_holdout_touched"]
        assert locked["config"] == config["optimization"] and locked["budgets"] == chain["budgets"]
        assert locked["grid"] == 16 and locked["samples"] == 64
        assert np.allclose(locked["bounds"], chain["bounds"], atol=1e-6, rtol=1e-6)
        seed = 20260927 + int(
            hashlib.sha256(f"{sid}|{chain['role']}|{chain['track']}".encode()).hexdigest()[:8], 16
        )
        assert locked["seed"] == seed
        assert locked["secondary_sanity"] == (phase == "secondary")
        assert locked["objective_threshold"] == 0.001 and locked["gradient_threshold"] == 0.10
        assert not locked["engineering_fixture"] and locked["single_shared_trajectory"]
        query_ids = records[sid]["roles"]["primary_query"]
        context_ids = records[sid]["roles"][
            "context_a" if chain["role"] in ("A", "joint_query") else "context_b"
        ]
        supervised_ids = query_ids if phase == "oracle" else context_ids
        assert locked["supervision_frame_ids"] == supervised_ids
        assert locked["has_depth"] == (phase != "secondary")
        for event in read(directory / "access.json"):
            assert event["scene_id"] == sid and sid not in holdout
            if phase == "oracle":
                if event["frame_id"] in query_ids:
                    assert event["purpose"] == "DIAGNOSTIC_QUERY_SUPERVISED_ORACLE_NOT_CONTEXT_ONLY"
                else:
                    assert event["frame_id"] == records[sid]["roles"]["context_a"][0]
                    assert (
                        event["purpose"] == "CONTEXT_ONLY_RGB_ONLY"
                        and "depth" not in event["channels"]
                    )
            else:
                assert event["frame_id"] in context_ids and event["frame_id"] not in query_ids
                assert event["purpose"] == f"CONTEXT_ONLY_{chain['track']}"
                if phase == "secondary":
                    assert "depth" not in event["channels"]
        assert summary["status"] == "PASS" and summary["steps_completed"] == 10000
        assert summary["full_objective_checkpoints"] == checkpoints
        assert [c["step"] for c in checkpoints] == list(range(0, 10001, 100))
        with (directory / "optimization_trace.csv").open() as handle:
            trace = list(csv.DictReader(handle))
        assert [int(r["step"]) for r in trace] == [1] + list(range(10, 10001, 10))
        assert len(summary["trace"]) == len(trace)
        for raw, saved in zip(trace, summary["trace"], strict=True):
            for field, value in saved.items():
                assert raw[field] == ("" if value is None else str(value))
        chain_specs = [s for s in states if s["chain_key"] == key]
        assert set(completion["states"]) == {s["key"] for s in chain_specs}
        assert {(s["budget"], s["selection"]) for s in chain_specs} == {
            (b, selection)
            for b in chain["budgets"]
            for selection in ("FIXED_BUDGET", "CONTEXT_SELECTED")
        }
        for budget in chain["budgets"]:
            diagnostic = independent_plateau(checkpoints, trace, budget)
            plateau_checks.append({"chain_key": key, "budget": budget, **diagnostic})
            for selection in ("FIXED_BUDGET", "CONTEXT_SELECTED"):
                spec = next(
                    s for s in chain_specs if s["budget"] == budget and s["selection"] == selection
                )
                for field in (
                    "scene_id",
                    "phase",
                    "role",
                    "track",
                    "grid",
                    "samples",
                    "bounds_mode",
                    "bounds",
                ):
                    assert spec[field] == chain[field]
                state_path = root / spec["state_path"]
                record = read(root / spec["optimization_path"] / "summary.json")
                hashes = completion["states"][spec["key"]]
                assert sha(state_path) == hashes["state_file_sha256"] == record["state_file_sha256"]
                assert (
                    sha(root / spec["optimization_path"] / "summary.json")
                    == hashes["summary_sha256"]
                )
                eligible = [c for c in checkpoints if c["step"] <= budget]
                selected = (
                    next(c for c in eligible if c["step"] == budget)
                    if selection == "FIXED_BUDGET"
                    else min(eligible, key=lambda c: (c["context_objective"], c["step"]))
                )
                assert record["selected_step"] == selected["step"]
                assert record["selected_full_objective"] == selected["context_objective"]
                assert record["selected_context_metrics"] == selected
                assert record["shared_supervision_frame_ids"] == supervised_ids
                assert record["budget"] == budget and record["rule"] == selection
                for field, value in diagnostic.items():
                    assert record["plateau_diagnostic"][field] == value
                state = torch.load(state_path, map_location="cpu", weights_only=False)
                assert state.density_logits.shape == (1, 1, 16, 16, 16) and state.features is None
                assert torch.equal(
                    state.bounds, torch.tensor(chain["bounds"], dtype=torch.float32)[None]
                )
                assert hash_scene_state(state) == record["state_hash"] == hashes["state_hash"]
                if selection == "FIXED_BUDGET":
                    snapshot = torch.load(
                        directory / "snapshots" / f"step_{budget}.pt",
                        map_location="cpu",
                        weights_only=False,
                    )
                    assert hash_scene_state(snapshot) == record["state_hash"]
                state_hashes[spec["key"]] = record["state_hash"]
                selected_checks.append(
                    {"key": spec["key"], "budget": budget, "selected_step": selected["step"]}
                )
        for step in (0, 100, 300):
            assert (directory / "snapshots" / f"step_{step}.pt").is_file()
        chain_locks[key] = locked
    for sid in records:
        for phase in ("context", "oracle"):
            # A/B roles may differ for joint CURRENT oracle; optimizer/loss still identical.
            selected = [c for c in chains if c["scene_id"] == sid and c["phase"] == phase]
            assert all(chain_locks[c["key"]]["config"] == config["optimization"] for c in selected)
    primary = read(root / "primary_analysis_complete.json")
    assert primary["status"] == "PASS" and primary["secondary_not_used"]
    assert primary["meaning"] == "PREDECLARED_PRIMARY_NOT_QUERY_MINIMUM"
    assert not any("secondary" in p for p in primary["inputs_sha256"])
    assert_hashes(primary["inputs_sha256"])
    secondary = read(root / "secondary_analysis/secondary_decision_audit.json")
    decision = read(root / "optimization_sufficiency.json")["primary"]
    secondary_decision = read(root / "secondary_analysis/optimization_sufficiency.json")["primary"]
    decision_hash = hashlib.sha256(
        json.dumps(decision, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert decision == secondary_decision == primary["primary_decision"]
    assert (
        decision_hash == primary["primary_decision_sha256"] == secondary["primary_decision_sha256"]
    )
    assert secondary["status"] == "PASS" and secondary["primary_bits_exactly_unchanged"]
    analysis_lock_path = root / "audit/analysis_source_lock.json"
    analysis_lock = read(analysis_lock_path)
    assert analysis_lock["status"] == "PASS"
    assert_hashes(analysis_lock["source_sha256"])
    assert analysis_lock["decision_rules_sha256"] == sha(root / "decision_rules.json")
    assert analysis_lock_path.stat().st_mtime_ns <= min(
        (root / "raw" / f"{phase}_evaluation_lock.json").stat().st_mtime_ns for phase in PHASES
    )

    # Filesystem mtimes are auxiliary chronology evidence, not tamper-proof authorization.
    assert (root / "raw/context_evaluation_integrity.json").stat().st_mtime_ns <= min(
        (root / c["optimization_path"] / "lock.json").stat().st_mtime_ns
        for c in chains
        if c["phase"] == "oracle"
    )
    assert (root / "primary_analysis_complete.json").stat().st_mtime_ns <= min(
        (root / c["optimization_path"] / "lock.json").stat().st_mtime_ns
        for c in chains
        if c["phase"] == "secondary"
    )
    truth_cache, audit_access, phase_reports = {}, [], {}
    for phase in PHASES:
        specs = [s for s in states if s["phase"] == phase]
        events = read(root / "raw" / f"{phase}_seal_events.json")
        assert all(e["event"] == "seal" for e in events[: len(specs)])
        assert {e["state_id"] for e in events[: len(specs)]} == {s["key"] for s in specs}
        assert all(
            e["event"] == "query_camera_GT" and e["scene_id"] not in holdout
            for e in events[len(specs) :]
        )
        for event in events[: len(specs)]:
            assert event["state_hash"] == state_hashes[event["state_id"]]
        eval_lock = read(root / "raw" / f"{phase}_evaluation_lock.json")
        for field in ("input_sha256", "optimizer_source_sha256", "evaluator_source_sha256"):
            assert_hashes(eval_lock[field])
        assert (
            max(
                (root / c["optimization_path"] / "completion.json").stat().st_mtime_ns
                for c in chains
                if c["phase"] == phase
            )
            <= (root / "raw" / f"{phase}_evaluation_lock.json").stat().st_mtime_ns
        )
        for event in read(root / "raw" / f"{phase}_GT_access.json"):
            assert event["scene_id"] in records and event["scene_id"] not in holdout
            assert event["purpose"] in (
                "HASH_INPUT_AFTER_ALL_STATES_SEALED",
                "POST_SEAL_CONTEXT_EVALUATION_NOT_OPTIMIZER",
                "SEALED_QUERY_EVALUATION",
            )
        rows = read(root / "raw" / f"{phase}_results.json")
        assert len({(r["key"], r["query_id"], r["method"]) for r in rows}) == len(rows)
        grouped = defaultdict(list)
        errors, exact = dict.fromkeys(METRICS, 0.0), dict.fromkeys(METRICS, 0)
        spec_by_key = {spec["key"]: spec for spec in specs}
        for row in rows:
            spec = spec_by_key[row["key"]]
            assert all(row.get(k) == v for k, v in spec.items()), "Raw row differs from state plan"
            sid, fid = row["scene_id"], row["query_id"]
            assert sid in records and sid not in holdout
            assert fid in records[sid]["roles"]["primary_query"]
            grouped[row["key"], fid].append(row)
            frame = next(f for f in records[sid]["frames"] if f["frame_id"] == fid)
            anchor = next(
                f
                for f in records[sid]["frames"]
                if f["frame_id"] == records[sid]["roles"]["context_a"][0]
            )
            # Reproduce camera arithmetic on the evaluator device; no renderer/model call.
            device = "cuda" if torch.cuda.is_available() else "cpu"
            camera = Cameras(
                torch.tensor(frame["intrinsics"], dtype=torch.float32, device=device)[None, None],
                torch.tensor(frame["c2w"], dtype=torch.float32, device=device)[None, None],
                tuple(manifest["image_size"]),
            )
            inverse = torch.linalg.inv(
                torch.tensor(anchor["c2w"], dtype=torch.float32, device=device)
            )
            assert hash_value(transform_cameras(camera, inverse)) == row["query_camera_hash"]

            if row["method"] == "direct":
                assert row["used_state_hash"] == state_hashes[row["key"]]
            else:
                assert row["method"] == "wrong_scene" and row["donor_scene_id"] != sid
                donor = next(
                    s
                    for s in specs
                    if s["scene_id"] == row["donor_scene_id"]
                    and all(
                        s[k] == row[k]
                        for k in ("role", "track", "bounds_mode", "budget", "selection")
                    )
                )
                assert row["used_state_hash"] == state_hashes[donor["key"]]
            if (sid, fid) not in truth_cache:
                frame = next(f for f in records[sid]["frames"] if f["frame_id"] == fid)
                paths = [
                    Path(frame[k]) if Path(frame[k]).is_absolute() else root / frame[k]
                    for k in ("rgb", "depth")
                ]
                assert not any(s in str(p) for s in holdout for p in paths)
                with Image.open(paths[0]) as im:
                    rgb = np.asarray(im.convert("RGB"), dtype=np.float32) / 255
                truth_cache[sid, fid] = rgb, np.load(paths[1]).squeeze()
                audit_access.append(
                    {
                        "scene_id": sid,
                        "query_id": fid,
                        "purpose": "POSTRUN_SAVED_NPZ_RECOMPUTATION",
                        "rgb_path": str(paths[0]),
                        "depth_path": str(paths[1]),
                        "final_holdout": False,
                    }
                )
            rgb, depth = truth_cache[sid, fid]
            with np.load(root / row["prediction_path"]) as pred:
                reconstructed = {
                    "rgb": torch.from_numpy(pred["rgb"])[None, None],
                    "depth": torch.from_numpy(pred["depth"])[None, None, None],
                    "visibility": torch.from_numpy(pred["opacity"])[None, None, None],
                }
                assert hash_value(reconstructed) == row["prediction_hash"]
                calculated = measurement_metrics(
                    pred["rgb"], rgb, pred["depth"], depth, pred["opacity"]
                )
            for metric in METRICS:
                difference = abs(calculated[metric] - row[metric])
                errors[metric] = max(errors[metric], difference)
                exact[metric] += calculated[metric] == row[metric]
                assert np.isclose(calculated[metric], row[metric], atol=1e-6, rtol=1e-5)
        for spec in specs:
            seen = []
            for fid in records[spec["scene_id"]]["roles"]["primary_query"]:
                group = grouped[spec["key"], fid]
                assert {r["method"] for r in group} == (
                    {"direct"} if phase == "oracle" else {"direct", "wrong_scene"}
                )
                assert len({r["query_camera_hash"] for r in group}) == 1
                seen.extend(r["used_state_hash"] for r in group if r["method"] == "direct")
            assert len(set(seen)) == 1
        context = read(root / "raw" / f"{phase}_context_results.json")
        for row in context:
            assert all(row.get(k) == v for k, v in spec_by_key[row["key"]].items())
        if phase == "oracle":
            assert not context
        else:
            assert len(context) == len(specs) * 3
            for row in context:
                role = "context_a" if row["role"] == "A" else "context_b"
                assert row["frame_id"] in records[row["scene_id"]]["roles"][role]
        phase_reports[phase] = {
            "states": len(specs),
            "query_rows": len(rows),
            "context_rows": len(context),
            "metric_max_absolute_error": errors,
            "metric_exact_row_counts": exact,
            "all_states_shared_across_queries": True,
            "all_query_GT_after_complete_seal": True,
            "wrong_scene_camera_preserved": True,
        }
    assert_hashes(config["source_sha256"])
    report = {
        "status": "PASS",
        "chain_count": len(chains),
        "state_count": len(states),
        "old_artifact_count_unchanged": len(prior_seal["files"]),
        "all34_old_gtfree_bounds_exact": True,
        "all_sources_inputs_preregistration_unchanged": True,
        "all_optimizers_loss_seeds_budget_ids_match": True,
        "fixed_and_context_selected_checkpoints_correct": True,
        "plateau_independently_recomputed": True,
        "secondary_cannot_change_primary": True,
        "primary_configuration": config["primary"],
        "phase_chronology_auxiliary": (
            "mtime checks PASS; OS timestamps are mutable, not tamper-proof"
        ),
        "selected_checkpoint_checks": selected_checks,
        "plateau_checks": plateau_checks,
        "phases": phase_reports,
        "raw_statistics_audit": "SEPARATE_ANALYSIS_AGENT",
        "final_holdout_touched": False,
        "new_carrier_trained": False,
        "dynamic_ttt_run": False,
        "metric_tolerance": {"atol": 1e-6, "rtol": 1e-5},
        "audit_script_sha256": sha(Path(__file__)),
    }
    write_json(root / "audit/postrun_GT_access.json", audit_access)
    write_json(destination, report)
    return {"status": "PASS", "chains": len(chains), "states": len(states), "phases": phase_reports}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.root), indent=2))
