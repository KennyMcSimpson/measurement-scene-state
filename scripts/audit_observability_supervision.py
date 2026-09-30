"""Independent completed-run audit: saved states/predictions only; no model inference."""

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from mcss.dynamic.types import clone_scene_state, hash_scene_state, hash_value
from mcss.geometry import transform_cameras
from mcss.mechanism_pilot.small_training import sha, write_json
from mcss.mechanism_pilot.statistics import measurement_metrics
from mcss.types import Cameras, SceneState

METRICS = (
    "rgb_mse",
    "rgb_ssim",
    "depth_absrel",
    "depth_rmse",
    "depth_delta1",
    "opacity",
    "coverage",
)


def read(path):
    return json.loads(Path(path).read_text())


def hashes(mapping):
    for path, digest in mapping.items():
        assert sha(Path(path)) == digest, f"Hash changed: {path}"


def resolve(root, path):
    path = Path(path)
    return path if path.is_absolute() else root / path


def sealed_events(events, specs, states):
    assert len(events) >= len(specs)
    expected = {s["key"]: s for s in specs}
    sealed = events[: len(specs)]
    assert {e["state_id"] for e in sealed} == set(expected)
    for event in sealed:
        assert event["event"] == "seal"
        key = event["state_id"]
        assert event["scene_id"] == expected[key]["scene_id"]
        assert event["state_hash"] == hash_scene_state(states[key])
    assert all(e["event"] == "query_camera_GT" for e in events[len(specs) :])


def control_state(state, method):
    state = clone_scene_state(state)
    if method == "zero":
        state.density_logits.fill_(-1e6)
        state.color.zero_()
    elif method == "spatial_shuffle":
        p = torch.randperm(16**3, generator=torch.Generator().manual_seed(20260927))
        for field in ("density_logits", "color", "log_variance"):
            value = getattr(state, field)
            setattr(state, field, value.flatten(2)[:, :, p].reshape_as(value))
    else:
        raise ValueError(method)
    return state


def audit(root):
    root = Path(root)
    manifest, config, plan = [
        read(root / f"{name}.json") for name in ("scene_manifest", "config", "state_plan")
    ]
    records = {r["scene_id"]: r for r in manifest["scenes"]}
    holdout = set(manifest["FINAL_HOLDOUT_PROHIBITED"])
    assert len(records) == 17 and not set(records) & holdout
    assert all(r["cohort_role"] == "CAPACITY_DEV_EXPOSED" for r in records.values())
    specs = plan["states"]
    assert len(specs) == 136
    assert {(s["scene_id"], s["role"], s["variant"]) for s in specs} == {
        (sid, role, variant)
        for sid in records
        for role in ("A", "B")
        for variant in ("S0", "S1", "S2", "S3")
    }
    bykey = {s["key"]: s for s in specs}
    assert len(bykey) == 136
    for name, digest in read(root / "preregistration.json")["locked_file_sha256"].items():
        assert sha(root / name) == digest
    hashes(config["source_sha256"])
    assert config["state_plan_sha256"] == sha(root / "state_plan.json")
    assert config["scene_manifest_sha256"] == sha(root / "scene_manifest.json")
    assert config["primary_variant"] == "S3" and config["shuffle_seed"] == 20260927
    assert config["optimization"] == {
        "steps": 10000,
        "learning_rate": 0.03,
        "batch_rays": 1024,
        "checkpoint_interval": 100,
        "rgb_weight": 1.0,
        "depth_weight": 1.0,
        "density_l2_weight": 1e-5,
        "tv_weight": 1e-4,
        "gradient_clip": 1.0,
        "selection": "BEST_FULL_SUPERVISION_OBJECTIVE_EARLIEST_TIE",
    }
    reproduction = read(root / "baseline_reproduction.json")
    assert reproduction["status"] == "PASS"
    for kind, id_field in (("query", "query_id"), ("context", "frame_id")):
        filename = "baseline_results.json" if kind == "query" else "baseline_context_results.json"
        current_rows = read(root / "raw" / filename)
        historical_rows = read(root / "raw" / f"historical_baseline_{kind}.json")

        def keyed(values, id_field=id_field):
            result = {(r["scene_id"], r["role"], r[id_field]): r for r in values}
            assert len(result) == len(values)
            return result

        current, historical = keyed(current_rows), keyed(historical_rows)
        assert set(current) == set(historical)

        def macro(values):
            return float(
                np.mean(
                    [
                        np.mean(
                            [
                                np.mean(
                                    [
                                        row["depth_absrel"]
                                        for key, row in values.items()
                                        if key[0] == sid and key[1] == role
                                    ]
                                )
                                for role in ("A", "B")
                            ]
                        )
                        for sid in records
                    ]
                )
            )

        difference = abs(macro(current) - macro(historical))
        maximum = max(
            abs(current[k]["depth_absrel"] - historical[k]["depth_absrel"]) for k in current
        )
        assert difference <= config["baseline_reproduction"]["mean_atol"]
        assert maximum <= config["baseline_reproduction"]["per_row_atol"]
        report = reproduction["comparisons"][kind]
        assert np.isclose(report["mean_absolute_difference"], difference, atol=1e-14)
        assert np.isclose(report["max_row_absrel_difference"], maximum, atol=1e-14)
    seal = read(root / "audit/previous_experiment_seal.json")
    assert len(seal["files"]) == 11285
    for path, value in seal["files"].items():
        p = Path(path)
        assert p.stat().st_size == value["bytes"] and sha(p) == value["sha256"], path
    bounds_lock = read(root / "bounds_lock.json")["bounds"]
    states, summaries, matched = {}, {}, defaultdict(list)
    for spec in specs:
        key, sid, role = spec["key"], spec["scene_id"], spec["role"]
        assert spec["grid"] == 16 and spec["samples"] == 64
        assert spec["bounds"] == bounds_lock[f"{sid}/{role}"]["FROZEN_GT_FREE_BOUNDS"]
        folder = root / spec["optimization_path"]
        summary, lock, complete = [
            read(folder / f"{name}.json") for name in ("summary", "lock", "completion")
        ]
        assert complete["status"] == "PASS" and complete["key"] == key
        assert complete["source_sha256"] == config["source_sha256"]
        assert complete["plan_sha256"] == sha(root / "state_plan.json")
        assert complete["config_sha256"] == sha(root / "config.json")
        assert complete["summary_sha256"] == sha(folder / "summary.json")
        assert complete["access_sha256"] == sha(folder / "access.json")
        assert complete["query_selection"] is False and summary["query_selection"] is False
        assert (
            summary["state_file_sha256"]
            == complete["state_file_sha256"]
            == sha(root / spec["state_path"])
        )
        expectedseed = 20260927 + int(
            hashlib.sha256(f"{sid}|{role}|RGBD".encode()).hexdigest()[:8], 16
        )
        assert summary["seed"] == lock["seed"] == expectedseed
        assert summary["config"] == lock["config"] == config["optimization"]
        assert summary["variant"] == lock["variant"] == spec["variant"]
        assert summary["selection"] == lock["selection"] == "FIXED_BUDGET"
        assert summary["steps_completed"] == summary["selected_step"] == 10000
        assert summary["all_steps_finite"] and not summary["engineering_fixture"]
        assert summary["supervision"] == "CONTEXT_ONLY_RGBD"
        contextids = records[sid]["roles"]["context_a" if role == "A" else "context_b"]
        assert summary["frame_ids"] == contextids
        assert summary["sampled_ray_count"] == 10240000
        assert len(summary["ray_stream_sha256"]) == 64
        b = torch.tensor(spec["bounds"], dtype=torch.float32)
        tau = float(0.5 * torch.linalg.vector_norm((b[1] - b[0]) / 16))
        assert np.isclose(summary["tau_surface"], tau, atol=1e-7, rtol=1e-7)
        assert summary["lambda_free"] == summary["lambda_surface"] == 0.1
        assert summary["epsilon"] == 1e-8
        for event in read(folder / "access.json"):
            assert event["scene_id"] == sid and event["frame_id"] in contextids
            assert event["purpose"] == "CONTEXT_ONLY_RGBD"
            assert set(event["channels"]) == {"RGB", "camera", "depth"}
        state = torch.load(root / spec["state_path"], map_location="cpu", weights_only=False)
        assert hash_scene_state(state) == summary["state_hash"] == complete["state_hash"]
        assert state.features is None and state.density_logits.shape == (1, 1, 16, 16, 16)
        assert torch.equal(state.bounds, b.reshape(1, 2, 3))
        initial = SceneState(
            torch.full_like(state.density_logits, -2),
            torch.full_like(state.color, 0.5),
            torch.full_like(state.log_variance, -3),
            state.bounds.clone(),
        )
        assert hash_scene_state(initial) == summary["initial_state_hash"]
        checkpoints = read(folder / "full_objective_checkpoints.json")
        assert [c["step"] for c in checkpoints] == list(range(0, 10001, 100))
        assert summary["final_context_metrics"] == checkpoints[-1]
        trace = read(folder / "trace.json")
        assert [t["step"] for t in trace] == [1] + list(range(10, 10001, 10))
        for row in checkpoints + trace:
            assert all(np.isfinite(v) for v in row.values() if isinstance(v, (int, float)))
            combined = row["render_objective"]
            if spec["variant"] in ("S1", "S3"):
                combined += 0.1 * row["loss_free"]
            if spec["variant"] in ("S2", "S3"):
                combined += 0.1 * row["loss_surface"]
            assert np.isclose(combined, row["context_objective"], atol=1e-6, rtol=1e-6)
            assert (
                row["surface_eligible_rays"] + row["empty_band_hit_rays"] == row["valid_hit_rays"]
            )
            assert row["valid_hit_rays"] + row["valid_miss_rays"] == row["valid_depth_rays"]
        states[key], summaries[key] = state, summary
        matched[sid, role].append(summary)
    for values in matched.values():
        assert len(values) == 4
        assert len({v["initial_state_hash"] for v in values}) == 1
        assert len({v["ray_stream_sha256"] for v in values}) == 1
    for phase in ("baseline", "formal"):
        subset = [s for s in specs if phase == "formal" or s["variant"] == "S0"]
        marker = read(root / f"{phase}_optimization_complete.json")
        assert marker["status"] == "PASS" and marker["states"] == len(subset)
        hashes(marker["input_sha256"])
        integrity = read(root / f"raw/{phase}_evaluation_integrity.json")
        assert integrity["status"] == "PASS" and integrity["all_states_immutable"]
        lock = read(root / f"raw/{phase}_evaluation_lock.json")
        hashes(lock["input_sha256"])
        hashes(lock["optimizer_source_sha256"])
        hashes(lock["evaluator_source_sha256"])
        sealed_events(read(root / f"raw/{phase}_seal_events.json"), subset, states)
        for event in read(root / f"raw/{phase}_GT_access.json"):
            assert event["scene_id"] in records and event["scene_id"] not in holdout
            assert event["purpose"] in (
                "HASH_INPUT_AFTER_ALL_STATES_SEALED",
                "POST_SEAL_CONTEXT_EVALUATION_NOT_OPTIMIZER",
                "SEALED_QUERY_EVALUATION",
            )
        assert (
            max(
                (root / s["optimization_path"] / "completion.json").stat().st_mtime_ns
                for s in subset
            )
            <= (root / f"raw/{phase}_evaluation_lock.json").stat().st_mtime_ns
        )
    gate_mtime = (root / "baseline_reproduction.json").stat().st_mtime_ns
    assert gate_mtime <= min(
        (root / s["optimization_path"] / "lock.json").stat().st_mtime_ns
        for s in specs
        if s["variant"] != "S0"
    )
    formal = read(root / "raw/query_results.json")
    references = read(root / "raw/reference_results.json")
    assert len(formal) == 680 and len(references) == 272
    assert {r["variant"] for r in references} == {"OLD_BASELINE", "ORACLE", "CARRIER", "ANCHOR"}
    contexts = read(root / "raw/context_results.json")
    assert len(contexts) == 408
    assert len({(r["key"], r["frame_id"]) for r in contexts}) == 408
    for row in contexts:
        assert all(row[k] == v for k, v in bykey[row["key"]].items())
        assert row["used_state_hash"] == hash_scene_state(states[row["key"]])
        assert (
            row["frame_id"]
            in records[row["scene_id"]]["roles"]["context_a" if row["role"] == "A" else "context_b"]
        )
    truth, cameras, access, errors = {}, {}, [], dict.fromkeys(METRICS, 0.0)
    groups = defaultdict(list)
    for row in formal + references:
        sid, fid = row["scene_id"], row["query_id"]
        assert (
            sid in records and sid not in holdout and fid in records[sid]["roles"]["primary_query"]
        )
        predpath = root / row["prediction_path"]
        assert sha(predpath) == row["prediction_file_sha256"]
        if (sid, fid) not in truth:
            frame = next(f for f in records[sid]["frames"] if f["frame_id"] == fid)
            rgbpath, depthpath = [resolve(root, frame[k]) for k in ("rgb", "depth")]
            with Image.open(rgbpath) as im:
                rgb = np.asarray(im.convert("RGB"), dtype=np.float32) / 255
            depth = np.load(depthpath).squeeze()
            truth[sid, fid] = (rgb, depth)
            access.append(
                {
                    "scene_id": sid,
                    "query_id": fid,
                    "purpose": "POSTRUN_SAVED_PREDICTION_RECOMPUTATION",
                    "rgb": str(rgbpath),
                    "depth": str(depthpath),
                    "final_holdout": False,
                }
            )
            anchor = next(
                f
                for f in records[sid]["frames"]
                if f["frame_id"] == records[sid]["roles"]["context_a"][0]
            )
            device = "cuda" if torch.cuda.is_available() else "cpu"
            camera = Cameras(
                torch.tensor(frame["intrinsics"], dtype=torch.float32, device=device)[None, None],
                torch.tensor(frame["c2w"], dtype=torch.float32, device=device)[None, None],
                tuple(manifest["image_size"]),
            )
            inverse = torch.linalg.inv(
                torch.tensor(anchor["c2w"], dtype=torch.float32, device=device)
            )
            cameras[sid, fid] = hash_value(transform_cameras(camera, inverse))
        rgb, depth = truth[sid, fid]
        with np.load(predpath) as p:
            metrics = measurement_metrics(p["rgb"], rgb, p["depth"], depth, p["opacity"])
            if row in formal:
                assert (
                    hash_value(
                        {
                            "rgb": torch.from_numpy(p["rgb"])[None, None],
                            "depth": torch.from_numpy(p["depth"])[None, None, None],
                            "visibility": torch.from_numpy(p["opacity"])[None, None, None],
                        }
                    )
                    == row["prediction_hash"]
                )
        for metric in METRICS:
            difference = abs(metrics[metric] - row[metric])
            errors[metric] = max(errors[metric], difference)
            assert np.isclose(metrics[metric], row[metric], atol=1e-6, rtol=1e-5), (
                sid,
                fid,
                metric,
                difference,
            )
        if row["variant"] not in ("S0", "S1", "S2", "S3"):
            assert sha(Path(row["reference_source"])) == row["prediction_file_sha256"]
            continue
        spec = bykey[row["key"]]
        assert all(row[k] == v for k, v in spec.items())
        assert row["query_camera_hash"] == cameras[sid, fid]
        used = states[row["key"]]
        if row["method"] == "wrong_scene":
            ids = sorted(records)
            donor = ids[(ids.index(sid) + 1) % len(ids)]
            assert row["donor_scene_id"] == donor
            ds = next(
                s
                for s in specs
                if s["scene_id"] == donor
                and s["role"] == row["role"]
                and s["variant"] == row["variant"]
            )
            used = states[ds["key"]]
        elif row["method"] in ("zero", "spatial_shuffle"):
            assert row["variant"] == "S3"
            used = control_state(used, row["method"])
        else:
            assert row["method"] == "direct"
        assert hash_scene_state(used) == row["used_state_hash"]
        groups[row["key"], fid].append(row)
    for spec in specs:
        for fid in records[spec["scene_id"]]["roles"]["primary_query"]:
            expected = {"direct", "wrong_scene"} | (
                {"zero", "spatial_shuffle"} if spec["variant"] == "S3" else set()
            )
            assert {r["method"] for r in groups[spec["key"], fid]} == expected
    # Independent numerical region accounting; retain every empty category as an explicit row.
    regionrows = read(root / "raw/region_results.json")
    regions = {
        (r["scene_id"], r["role"], r["query_id"], r["variant"], r["region"]): r for r in regionrows
    }
    assert len(regions) == len(regionrows)
    assert len(read(root / "raw/observability_summary.json")) == 68
    obs = read(root / "audit/observability_integrity.json")
    assert obs["status"] == "PASS"
    hashes(obs["input_sha256"])
    sealed_events(read(root / "raw/observability_seal_events.json"), specs, states)
    for event in read(root / "raw/observability_GT_access.json"):
        assert event["scene_id"] in records and event["scene_id"] not in holdout
    nr = 0
    for row in [r for r in formal if r["method"] == "direct"] + references:
        sid, role, fid = row["scene_id"], row["role"], row["query_id"]
        with np.load(root / f"raw/observability/{sid}_{role}_{fid}.npz") as a:
            valid = a["query_valid"]
            category = a["obs_class"]
            angle = a["triangulation_angle_bin"]
            assert np.array_equal(
                valid, np.isfinite(a["query_gt_depth"]) & (a["query_gt_depth"] > 0)
            )
            masks = {
                "OVERALL": valid,
                "OBS0": valid & (category == 0),
                "OBS1": valid & (category == 1),
                "OBS2PLUS": valid & (category == 2),
                "OBS01": valid & (category < 2),
                **{
                    name: valid & (angle == i)
                    for i, name in enumerate(
                        ("ANGLE_0_5", "ANGLE_5_15", "ANGLE_15_30", "ANGLE_GT30")
                    )
                },
                "OCCLUDED_ANY": valid & a["occluded_by_context_surface"].any(0),
                "CONFLICT_ANY": valid & a["depth_conflict_front"].any(0),
            }
            assert np.array_equal(masks["OBS0"] | masks["OBS1"] | masks["OBS2PLUS"], valid)
            assert np.array_equal((a["visible_view_count"].clip(max=2))[valid], category[valid])
            assert np.array_equal(
                sum(
                    masks[n].astype(int)
                    for n in ("ANGLE_0_5", "ANGLE_5_15", "ANGLE_15_30", "ANGLE_GT30")
                ),
                masks["OBS2PLUS"].astype(int),
            )
            with np.load(root / row["prediction_path"]) as p:
                predicted = p["depth"].squeeze().astype(np.float64)
            gt = a["query_gt_depth"]
            error = np.zeros(gt.shape, np.float64)
            error[valid] = np.abs(predicted[valid] - gt[valid]) / gt[valid]
            for name, mask in masks.items():
                r = regions[sid, role, fid, row["variant"], name]
                nr += 1
                count = int(mask.sum())
                total = float(error[mask].sum())
                assert r["pixel_count"] == count and r["total_valid"] == int(valid.sum())
                assert np.isclose(r["absrel_sum"], total, atol=1e-9, rtol=1e-12)
                assert (
                    (r["absrel_mean"] is None)
                    if not count
                    else np.isclose(r["absrel_mean"], total / count, atol=1e-12)
                )
    assert nr == len(regionrows)
    hashes(config["source_sha256"])
    for key, state in states.items():
        assert hash_scene_state(state) == summaries[key]["state_hash"]
    report = {
        "status": "PASS",
        "n_states": 136,
        "matched_initialization_and_streams": 34,
        "old_artifacts_unchanged": 11285,
        "n_formal_query_rows": 680,
        "n_context_rows": 408,
        "n_reference_rows": 272,
        "n_region_rows": nr,
        "full_region_partition_and_empty_rows_preserved": True,
        "all_geometry_loss_components_checked": True,
        "max_metric_absolute_differences": errors,
        "metric_tolerance": {"atol": 1e-6, "rtol": 1e-5},
        "GT_access_after_all_seals": True,
        "timestamps": "Auxiliary mtime chronology, not tamper-proof OS evidence",
        "statistics_reproduction": "SEPARATE_RAW_STATISTICS_AUDIT",
        "final_holdout_touched": False,
        "new_carrier_trained": False,
        "dynamic_ttt_run": False,
        "script_sha256": sha(Path(__file__)),
    }
    write_json(root / "audit/postrun_GT_access.json", access)
    write_json(root / "audit/postrun_validation.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.root), indent=2))
