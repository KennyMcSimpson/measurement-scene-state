"""Independent raw-only audit of support attribution; never loads a carrier for inference."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from mcss.dynamic.feedback import batched_camera
from mcss.dynamic.types import hash_scene_state, hash_value
from mcss.geometry import transform_cameras
from mcss.mechanism_pilot.small_training import TrainScene, sha, write_json
from mcss.mechanism_pilot.statistics import measurement_metrics
from mcss.mechanism_pilot.support_redesign_contracts import validate_dev_scene_ids
from mcss.mechanism_pilot.support_redesign_statistics import analyze

METRICS = (
    "rgb_mse",
    "rgb_ssim",
    "depth_absrel",
    "depth_rmse",
    "depth_delta1",
    "opacity",
    "coverage",
)


def run(root):
    root = Path(root)
    output = root / "audit/final_independent_audit.json"
    if output.exists():
        raise FileExistsError(output)
    run_dir = root / "raw/oracle_run"
    rows = json.loads((run_dir / "results.json").read_text())
    lock = json.loads((run_dir / "lock.json").read_text())
    roles = json.loads((root / "scene_role_lock.json").read_text())
    validate_dev_scene_ids(roles, sorted({r["scene_id"] for r in rows}))
    holdout = set(roles["FINAL_HOLDOUT_SCENES"])
    assert not set(lock["scene_ids"]) & holdout
    assert not lock["final_holdout_access"] and not lock["dynamic_ttt_run"]
    assert analyze(rows) == json.loads((root / "bootstrap_results.json").read_text())
    for group in ["input_sha256", "source_sha256"]:
        assert all(sha(Path(path)) == value for path, value in lock[group].items())
    previous = json.loads((root / "audit/previous_artifact_hashes.json").read_text())
    for path, info in previous["files"].items():
        assert Path(path).stat().st_size == info["bytes"] and sha(Path(path)) == info["sha256"]
    assert sha(Path(previous["old_carrier"])) == previous["old_carrier_sha256"]
    assert sha(Path(previous["old_residual"])) == previous["old_residual_sha256"]
    access = json.loads((run_dir / "access_log.json").read_text())
    assert all(r.get("scene_id") not in holdout for r in access)
    states = torch.load(run_dir / "sealed_states.pt", map_location="cpu", weights_only=False)
    plans = torch.load(run_dir / "geometry_plans.pt", map_location="cpu", weights_only=False)
    assert set(states) == set(plans)
    assert all(s.scene_id not in holdout for s in states.values())
    expected_states = len(lock["scene_ids"]) * len(lock["designs"]) * 3
    assert len(states) == expected_states
    first_oracle = next(
        i for i, r in enumerate(access) if r["kind"] == "ORACLE_GEOMETRY_GT_DEPTH_CAMERA"
    )
    r0 = [r for r in access[:first_oracle] if r["kind"] == "SEALED_STATE"]
    assert len(r0) == len(lock["scene_ids"]) * 3
    assert all(r["state_key"].startswith("R0/") and not r["oracle_geometry_used"] for r in r0)
    first_eval = next(i for i, r in enumerate(access) if r["kind"] == "QUERY_RGB_DEPTH_GT")
    assert sum(r["kind"] == "SEALED_STATE" for r in access[:first_eval]) == expected_states
    assert all(
        r["stage"] == "PRESEAL_DIAGNOSTIC_EXCEPTION" and r["deployable"] is False
        for r in access
        if r["kind"] == "ORACLE_GEOMETRY_GT_DEPTH_CAMERA"
    )
    events = json.loads((run_dir / "sealed_query_log.json").read_text())
    assert all(e["event"] == "seal" for e in events[:expected_states])
    assert all(e["event"] != "seal" for e in events[expected_states:])
    for key, state in states.items():
        assert hash_scene_state(state.scene_state) == state.state_hash
        plan = plans[key]
        assert plan.points.shape == (128, 3) and plan.candidate_ids.shape == (128,)
        assert len(torch.unique(plan.candidate_ids)) == 128
        assert len(torch.unique(plan.points, dim=0)) == 128
        assert torch.equal(state.scene_state.bounds[0], plan.bounds)
        ids = plan.candidate_ids
        voxel_fraction = torch.stack(
            ((ids % 8 + 0.5) / 8, ((ids // 8) % 8 + 0.5) / 8, (ids // 64 + 0.5) / 8), dim=-1
        )
        expected_points = plan.bounds[0] + voxel_fraction * (plan.bounds[1] - plan.bounds[0])
        assert torch.allclose(plan.points, expected_points, atol=1e-6, rtol=1e-6)
        expected_normalized = (plan.points - plan.bounds[0]) / (
            plan.bounds[1] - plan.bounds[0]
        ) * 2 - 1
        assert torch.allclose(plan.normalized_xyz, expected_normalized, atol=1e-6, rtol=1e-6)
    scenes = {}
    for filename in lock["input_sha256"]:
        if filename.endswith("/manifest.json") or filename.endswith("/dev_manifest.json"):
            manifest_path = Path(filename)
            manifest = json.loads(manifest_path.read_text())
            validate_dev_scene_ids(roles, [r["scene_id"] for r in manifest["scenes"]])
            for record in manifest["scenes"]:
                scenes[record["scene_id"]] = TrainScene(
                    record, manifest["image_size"], manifest_path.parent, "cpu", []
                )
    truth_cache, audit_access, max_errors, predictions = {}, [], dict.fromkeys(METRICS, 0.0), {}
    exact_counts = dict.fromkeys(METRICS, 0)
    for row in rows:
        sid, fid, design, method = (row[k] for k in ("scene_id", "query_id", "design", "method"))
        assert sid not in holdout
        scene = scenes[sid]
        if (sid, fid) not in truth_cache:
            frame = scene.frames[fid]
            rgb_path, depth_path = scene.path(frame, "rgb"), scene.path(frame, "depth")
            with Image.open(rgb_path) as im:
                rgb = np.asarray(im.convert("RGB"), dtype=np.float32) / 255
            depth = np.load(depth_path).squeeze()
            truth_cache[sid, fid] = (rgb, depth)
            audit_access.append(
                {
                    "scene_id": sid,
                    "query_id": fid,
                    "cohort": row["cohort"],
                    "kind": "POSTHOC_RAW_RECOMPUTATION_GT",
                    "rgb_path": str(rgb_path),
                    "rgb_sha256": sha(rgb_path),
                    "depth_path": str(depth_path),
                    "depth_sha256": sha(depth_path),
                    "final_holdout": False,
                }
            )
        truth_rgb, truth_depth = truth_cache[sid, fid]
        pred_path = run_dir / "predictions" / f"{sid}_{fid}_{design}_{method}.npz"
        with np.load(pred_path) as pred:
            metrics = measurement_metrics(
                pred["rgb"], truth_rgb, pred["depth"], truth_depth, pred["opacity"]
            )
        for metric in METRICS:
            error = abs(metrics[metric] - row[metric])
            max_errors[metric] = max(max_errors[metric], error)
            exact_counts[metric] += metrics[metric] == row[metric]
            assert np.isclose(metrics[metric], row[metric], rtol=1e-12, atol=1e-12)
        predictions[str(pred_path)] = sha(pred_path)
        camera = batched_camera(
            transform_cameras(
                scene.camera(scene.frames[fid]).to("cuda"),
                torch.linalg.inv(states[f"R0/{sid}/A"].anchor_c2w.to("cuda")),
            )
        )
        assert hash_value(camera) == row["query_camera_hash"]
    for sid, fid in truth_cache:
        assert (
            len(
                {
                    r["query_camera_hash"]
                    for r in rows
                    if r["scene_id"] == sid and r["query_id"] == fid
                }
            )
            == 1
        )
    old_path = Path(
        "outputs/EXP-3D-STATIC-CARRIER-ATTRIBUTION-V1/unseen_static/static_results.json"
    )
    old = json.loads(old_path.read_text())
    old_index = {(r["scene_id"], r["query_id"], r["method"]): r for r in old}
    reproduced = [r for r in rows if r["cohort"] == "exposed" and r["design"] == "R0"]
    assert len(reproduced) == 70
    for row in reproduced:
        prior = old_index[row["scene_id"], row["query_id"], row["method"]]
        assert all(row[k] == prior[k] for k in METRICS + ("query_camera_hash",))
    integrity = json.loads((run_dir / "integrity.json").read_text())
    assert integrity["writes"] == 0 and integrity["parameters_unchanged"]
    assert integrity["checkpoint_unchanged"] and not integrity["dynamic_ttt_run"]
    for group in ["input_sha256", "source_sha256"]:
        assert all(sha(Path(path)) == value for path, value in lock[group].items())
    summary = analyze(rows)
    findings = {}
    for cohort, group in list(summary["cohorts"].items()) + [("pooled", summary["pooled"])]:
        findings[cohort] = {
            design: {
                "full_gain": data["full_context_gain"]["mean"],
                "ci95": data["full_context_gain"]["ci95"],
                "adequate": data["oracle_adequacy"]["adequate"],
            }
            for design, data in group["designs"].items()
        }
    write_json(root / "audit/raw_recomputation_GT_access.json", audit_access)
    write_json(
        output,
        {
            "status": "PASS",
            "n_rows": len(rows),
            "n_scenes": len(scenes),
            "n_states": len(states),
            "prediction_npz_count": len(predictions),
            "raw_metrics_recomputed": True,
            "metric_max_absolute_error": max_errors,
            "metric_exact_row_counts": exact_counts,
            "metric_tolerance": {"atol": 1e-12, "rtol": 1e-12},
            "bootstrap_summary_exact": True,
            "all_frozen_inputs_sources_unchanged": True,
            "all_old_artifacts_unchanged": True,
            "old_artifact_count": len(previous["files"]),
            "checkpoint_and_residual_unchanged": True,
            "old_exposed_R0_exact_rows": len(reproduced),
            "holdout_model_or_score_access_count": 0,
            "n_protected_holdout_scenes": len(holdout),
            "r0_sealed_before_oracle_exception": True,
            "oracle_preseal_exception_explicit": True,
            "all_eval_GT_after_all_states_sealed": True,
            "unique128_voxel_centers_and_bounds_validated": True,
            "query_cameras_recomputed": True,
            "states_immutable": True,
            "dynamic_ttt_run": False,
            "diagnostic_findings": findings,
            "interpretation_limit": (
                "Volume-only can improve while combined oracle CI crosses zero; "
                "neither proves support sufficiency or representation impossibility. "
                "Geometry adequacy thresholds limit attribution."
            ),
            "audit_script_sha256": sha(Path(__file__)),
            "prediction_hashes": predictions,
        },
    )
    return {
        "status": "PASS",
        "rows": len(rows),
        "old_artifact_count": len(previous["files"]),
        "metric_max_absolute_error": max_errors,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.root), indent=2))
