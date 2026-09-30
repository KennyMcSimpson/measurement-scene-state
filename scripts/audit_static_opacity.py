"""Post-hoc diagnostic only: opacity-conditioned depth, never qualification metrics."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from mcss.dynamic.feedback import batched_camera
from mcss.dynamic.types import hash_scene_state, hash_value
from mcss.geometry import transform_cameras
from mcss.measurements import FixedMeasurementRenderer
from mcss.mechanism_pilot.small_training import TrainScene, sha, write_json
from mcss.mechanism_pilot.static_evaluation import summarize
from mcss.mechanism_pilot.statistics import paired_scene_bootstrap


def depth_metrics(prediction, truth, mask):
    if not mask.any():
        return None
    p = prediction[mask].double()
    t = truth[mask].double()
    return {
        "depth_absrel": float(((p - t).abs() / t).mean()),
        "depth_rmse": float((p - t).square().mean().sqrt()),
        "n_pixels": int(mask.sum()),
    }


def macro(rows, metric_key):
    result = {}
    for cohort in sorted({r["cohort"] for r in rows}):
        group = [r for r in rows if r["cohort"] == cohort]
        for method in ["A", "B", "anchor", "wrong_scene"]:
            values = {}
            for sid in sorted({r["scene_id"] for r in group}):
                rr = [
                    r[metric_key] for r in group if r["scene_id"] == sid and r["method"] == method
                ]
                values[sid] = (
                    None
                    if any(r is None for r in rr)
                    else float(np.mean([r["depth_absrel"] for r in rr]))
                )
            result[f"{cohort}/{method}"] = {
                "per_scene": values,
                "mean": None
                if any(v is None for v in values.values())
                else float(np.mean(list(values.values()))),
            }
        for left, right in [("anchor", "A"), ("anchor", "B"), ("wrong_scene", "A")]:
            left_values = result[f"{cohort}/{left}"]["per_scene"]
            right_values = result[f"{cohort}/{right}"]["per_scene"]
            result[f"{cohort}/{left}_minus_{right}"] = (
                None
                if any(left_values[s] is None or right_values[s] is None for s in left_values)
                else paired_scene_bootstrap(
                    {s: left_values[s] - right_values[s] for s in left_values}
                )
            )
    return result


@torch.no_grad()
def run(root, device="cuda"):
    root = Path(root)
    destination = root / "opacity_diagnostic.json"
    if destination.exists():
        raise FileExistsError(destination)
    all_diagnostics, audits, access = {}, {}, []
    renderer = FixedMeasurementRenderer(n_samples=64, ray_chunk_size=2048).to(device)
    for name in ["old_static", "unseen_static"]:
        run_dir = root / name
        lock = json.loads((run_dir / "lock.json").read_text())
        assert all(sha(Path(p)) == h for p, h in lock["input_sha256"].items())
        assert all(sha(Path(p)) == h for p, h in lock["source_sha256"].items())
        manifest_path = next(Path(p) for p in lock["input_sha256"] if p.endswith("/manifest.json"))
        manifest = json.loads(manifest_path.read_text())
        states = torch.load(run_dir / "sealed_states.pt", map_location=device, weights_only=False)
        seal_hash = sha(run_dir / "sealed_states.pt")
        original = json.loads((run_dir / "static_results.json").read_text())
        original_index = {
            (r["scene_id"], r["cohort"], r["query_id"], r["method"]): r for r in original
        }
        assert summarize(original) == json.loads((run_dir / "bootstrap_results.json").read_text())
        events = json.loads((run_dir / "sealed_query_access.json").read_text())
        n_states = len(states)
        assert all(e["event"] == "seal" for e in events[:n_states])
        assert all(e["event"] != "seal" for e in events[n_states:])
        assert len({e["candidate_id"] for e in events[:n_states]}) == n_states
        assert all(hash_scene_state(s.scene_state) == s.state_hash for s in states.values())
        integrity = json.loads((run_dir / "integrity.json").read_text())
        assert integrity["writes"] == 0 and integrity["dynamic_ttt_run"] is False
        records = manifest["scenes"]
        rows = []
        for scene_index, record in enumerate(records):
            sid = record["scene_id"]
            scene = TrainScene(record, manifest["image_size"], manifest_path.parent, device, [])
            anchor = states[f"{sid}/A"].anchor_c2w
            for cohort, query_ids in lock["scopes"][sid].items():
                for fid in query_ids:
                    frame = scene.frames[fid]
                    camera = batched_camera(
                        transform_cameras(scene.camera(frame), torch.linalg.inv(anchor))
                    )
                    camera_hash = hash_value(camera)
                    depth_path = scene.path(frame, "depth")
                    depth = torch.from_numpy(np.load(depth_path)).float().to(device).squeeze()
                    valid = torch.isfinite(depth) & (depth > 0)
                    access.append(
                        {
                            "run": name,
                            "scene_id": sid,
                            "query_id": fid,
                            "cohort": cohort,
                            "kind": "posthoc_diagnostic_GT_depth",
                            "path": str(depth_path),
                            "sha256": sha(depth_path),
                            "sealed_before_access": True,
                        }
                    )
                    predictions = {}
                    for method in ["A", "B", "anchor", "wrong_scene"]:
                        source_sid = (
                            records[(scene_index + 1) % len(records)]["scene_id"]
                            if method == "wrong_scene"
                            else sid
                        )
                        key = f"{source_sid}/{'A' if method == 'wrong_scene' else method}"
                        state = states[key]
                        prediction = renderer(
                            state.scene_state, camera, ("rgb", "depth", "visibility")
                        )
                        ref = original_index[sid, cohort, fid, method]
                        assert camera_hash == hash_value(camera) == ref["query_camera_hash"]
                        assert (
                            hash_value({"rgb": prediction["rgb"], "depth": prediction["depth"]})
                            == ref["prediction_hash"]
                        )
                        assert hash_scene_state(state.scene_state) == state.state_hash
                        predictions[method] = prediction
                    joint = valid.clone()
                    for pred in predictions.values():
                        joint &= pred["visibility"].squeeze() >= 0.95
                    for method, pred in predictions.items():
                        d, alpha = pred["depth"].squeeze(), pred["visibility"].squeeze()
                        norm = d / alpha.clamp_min(1e-6)
                        rows.append(
                            {
                                "scene_id": sid,
                                "cohort": cohort,
                                "query_id": fid,
                                "method": method,
                                "query_camera_hash": camera_hash,
                                "raw_all_GTvalid": depth_metrics(d, depth, valid),
                                "normalized_all_GTvalid": depth_metrics(norm, depth, valid),
                                "raw_joint_alpha95": depth_metrics(d, depth, joint),
                                "normalized_joint_alpha95": depth_metrics(norm, depth, joint),
                                "joint_alpha95_count": int(joint.sum()),
                                "valid_count": int(valid.sum()),
                                "mean_alpha": float(alpha[valid].mean()),
                                "fraction_alpha_below_095": float(
                                    (alpha[valid] < 0.95).float().mean()
                                ),
                                "fraction_alpha_below_1e6": float(
                                    (alpha[valid] < 1e-6).float().mean()
                                ),
                            }
                        )
        assert sha(run_dir / "sealed_states.pt") == seal_hash
        assert all(hash_scene_state(s.scene_state) == s.state_hash for s in states.values())
        assert all(sha(Path(p)) == h for p, h in lock["input_sha256"].items())
        assert all(sha(Path(p)) == h for p, h in lock["source_sha256"].items())
        all_diagnostics[name] = {
            "rows": rows,
            "summary": {
                key: macro(rows, key)
                for key in [
                    "raw_all_GTvalid",
                    "normalized_all_GTvalid",
                    "raw_joint_alpha95",
                    "normalized_joint_alpha95",
                ]
            },
        }
        audits[name] = {
            "status": "PASS",
            "input_hashes": "PASS",
            "source_hashes": "PASS",
            "sealed_state_hashes": "PASS",
            "all_sealed_before_queries": True,
            "bootstrap_recomputed_exact": True,
            "diagnostic_predictions_exact_original": True,
            "writes": 0,
            "dynamic_ttt_run": False,
            "raw_rows": len(original),
            "diagnostic_rows": len(rows),
            "state_archive_sha256": seal_hash,
        }
    previous = json.loads((root / "old_static_reproduction.json").read_text())
    old = json.loads(Path(previous["old_result_path"]).read_text())
    new = json.loads((root / "old_static/static_results.json").read_text())
    new_idx = {(r["scene_id"], r["query_id"], r["method"]): r for r in new}
    for r in old:
        n = new_idx[r["scene_id"], r["query_id"], r["method"]]
        assert all(r[k] == n[k] for k in previous["metrics"])
        assert r["query_camera_hash"] == n["query_camera_hash"]
    assert len(old) == 90
    write_json(
        destination,
        {
            "status": "DIAGNOSTIC_ONLY_NOT_PRIMARY_METRICS",
            "normalization": "sum(w*t)/clamp(sum(w), min=1e-6); no GT-valid pixel removal",
            "joint_subset_warning": (
                "ALL FOUR methods alpha>=.95 is prediction-conditioned; "
                "cannot serve as primary qualification or fairness claim"
            ),
            "unchanged_primary_qualification": True,
            "runs": all_diagnostics,
        },
    )
    write_json(root / "opacity_GT_access.json", access)
    write_json(
        root / "final_independent_audit.json",
        {
            "status": "PASS",
            "runs": audits,
            "old_reproduction_rows": 90,
            "old_reproduction_exact": True,
            "diagnostic_GT_access_count": len(access),
            "dynamic_ttt_run": False,
            "script_sha256": sha(Path(__file__)),
            "note": (
                "Normalization is a post-hoc renderer diagnostic "
                "and cannot replace preregistered results"
            ),
        },
    )
    return {name: value["summary"] for name, value in all_diagnostics.items()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    args = parser.parse_args()
    run(args.root, args.device)
    print("Opacity diagnostic and independent audit: PASS")
