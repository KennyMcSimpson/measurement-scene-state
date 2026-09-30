#!/usr/bin/env python3
"""Seal old artifacts, lock exposed capacity data, and replay existing R0 states.

No carrier construction, writer, optimization, or protected holdout media reads.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from mcss.dynamic.feedback import batched_camera
from mcss.dynamic.types import hash_scene_state, hash_value
from mcss.geometry import generate_rays, transform_cameras
from mcss.measurements import FixedMeasurementRenderer
from mcss.mechanism_pilot.small_training import TrainScene, sha, write_json
from mcss.mechanism_pilot.statistics import measurement_metrics
from mcss.types import Cameras

OLD_STATIC = Path("outputs/EXP-3D-STATIC-CARRIER-ATTRIBUTION-V1")
OLD_SUPPORT = Path("outputs/EXP-3D-SUPPORT-BOTTLENECK-REDESIGN-V1")
TRAIN = Path("outputs/EXP-3D-20260927-small-training-v1/data/manifest.json")
CHECKPOINT = Path(
    "outputs/EXP-3D-20260927-centered-training-1000a-600b-v1/training/phase_b_final.pt"
)
METRICS = (
    "rgb_mse",
    "rgb_ssim",
    "depth_absrel",
    "depth_rmse",
    "depth_delta1",
    "opacity",
    "coverage",
)


def prepare(root):
    root.mkdir(parents=True, exist_ok=True)
    (root / "audit").mkdir(exist_ok=True)
    initial = json.loads((OLD_SUPPORT / "scene_role_lock.json").read_text())
    protected = set(initial["FINAL_HOLDOUT_SCENES"])
    recorded = json.loads((OLD_SUPPORT / "data/hashes.json").read_text())
    archive = {}
    roots = [
        OLD_STATIC,
        OLD_SUPPORT,
        Path("docs/experiments") / OLD_STATIC.name,
        Path("docs/experiments") / OLD_SUPPORT.name,
    ]
    for old in roots:
        for path in sorted(old.rglob("*")):
            if not path.is_file():
                continue
            holdout_media = bool(protected & set(path.parts)) and (
                "raw" in path.parts or "prepared" in path.parts
            )
            if holdout_media:
                key = str(path.relative_to(OLD_SUPPORT / "data"))
                digest = recorded[key]
                mode = "previously_verified_SHA256_referenced_without_opening_protected_media"
            else:
                digest, mode = sha(path), "fresh_hash"
            archive[str(path)] = {
                "sha256": digest,
                "size": path.stat().st_size,
                "mtime_ns": path.stat().st_mtime_ns,
                "mode": mode,
            }
    for old in (OLD_STATIC, OLD_SUPPORT):
        source_path = old / "source_manifest.json"
        if source_path.exists():
            src = json.loads(source_path.read_text())
            for path, expected in src.items():
                p = Path(path)
                if p.exists():
                    archive[str(p)] = {
                        "sha256": sha(p),
                        "expected_previous_sha256": expected,
                        "size": p.stat().st_size,
                        "mtime_ns": p.stat().st_mtime_ns,
                        "mode": "fresh_source_hash",
                    }
    write_json(
        root / "audit/old_artifact_seal.json",
        {"created_unix": time.time(), "files": archive, "protected_media_read": False},
    )
    paths = [OLD_STATIC / "unseen_data/manifest.json", OLD_SUPPORT / "data/dev_manifest.json"]
    scenes = []
    for path in paths:
        for scene in json.loads(path.read_text())["scenes"]:
            scene["split"] = "capacity_dev_exposed"
            scene["cohort_role"] = "CAPACITY_DEV_EXPOSED"
            scene["source_manifest"] = str(path.resolve())
            for frame in scene["frames"]:
                for key in ("rgb", "depth"):
                    p = Path(frame[key])
                    frame[key] = str(
                        p.resolve() if p.is_absolute() else (path.parent / p).resolve()
                    )
            scenes.append(scene)
    ids = [s["scene_id"] for s in scenes]
    assert len(ids) == 17 and len(set(ids)) == 17 and not set(ids) & protected
    manifest = {
        "schema": "mcss.direct_capacity.exposed.v1",
        "image_size": [128, 160],
        "depth_semantics": "ray_distance_meters",
        "scenes": scenes,
        "scientific_role": "CAPACITY_DEV_EXPOSED_ATTRIBUTION_NOT_INDEPENDENT_CONFIRMATION",
        "FINAL_HOLDOUT_PROHIBITED": sorted(protected),
        "source_sha256": {str(p): sha(p) for p in paths},
        "query_access_requires_state_seal": True,
    }
    write_json(root / "scene_manifest.json", manifest)
    train = json.loads(TRAIN.read_text())
    values = []
    access = []
    for scene in train["scenes"]:
        for frame in scene["frames"]:
            if frame["frame_id"] not in [12, 13, 14, 15]:
                continue
            p = Path(frame["depth"])
            depth = np.load(p)
            values.append(depth[np.isfinite(depth) & (depth > 0)].astype(np.float64))
            access.append(
                {
                    "scene_id": scene["scene_id"],
                    "frame_id": frame["frame_id"],
                    "purpose": "frozen_global_training_depth_prior",
                    "path": str(p),
                    "sha256": sha(p),
                }
            )
    near, far = np.quantile(np.concatenate(values), [0.01, 0.99]).tolist()
    write_json(
        root / "train_depth_prior.json",
        {
            "near_m": near,
            "far_m": far,
            "quantiles": [0.01, 0.99],
            "population": "pooled finite positive TRAIN3 frame12..15 metric ray distance",
            "train_manifest_sha256": sha(TRAIN),
            "access": access,
        },
    )
    bounds = {}
    for scene in scenes:
        frames = {f["frame_id"]: f for f in scene["frames"]}
        anchor = torch.tensor(frames[scene["roles"]["context_a"][0]]["c2w"], dtype=torch.float64)
        for role, ids in [
            ("A", scene["roles"]["context_a"]),
            ("B", scene["roles"]["context_b"]),
            ("anchor", scene["roles"]["context_a"][:1]),
        ]:
            points = []
            for i in ids:
                f = frames[i]
                camera = Cameras(
                    torch.tensor(f["intrinsics"], dtype=torch.float64),
                    torch.tensor(f["c2w"], dtype=torch.float64),
                    (128, 160),
                )
                camera = transform_cameras(camera, torch.linalg.inv(anchor))
                origins, rays = generate_rays(camera)
                points.extend([(origins + rays * t).reshape(-1, 3) for t in (near, far)])
            p = torch.cat(points)
            lo, hi = p.amin(0), p.amax(0)
            center = (lo + hi) / 2
            extent = ((hi - lo) * 1.1).clamp_min(1.0)
            bounds[f"{scene['scene_id']}/{role}"] = {
                "bounds": torch.stack([center - extent / 2, center + extent / 2]).tolist(),
                "context_frame_ids": ids,
            }
    write_json(
        root / "predeclared_geometry_bounds.json",
        {
            "rule": (
                "context cameras only; unit-rays at TRAIN-only global q01/q99; "
                "AABB; 5 percent padding each side; min extent1m; same rule each context"
            ),
            "train_depth_prior_sha256": sha(root / "train_depth_prior.json"),
            "query_camera_or_GT_used": False,
            "bounds": bounds,
        },
    )
    write_json(
        root / "renderer_contract.json",
        {
            "renderer": "FixedMeasurementRenderer unchanged",
            "source_sha256": sha(Path("src/mcss/measurements.py")),
            "primary_samples": 64,
            "secondary_samples": 128,
            "ray_chunk_size": 2048,
            "depth": "unconditional sum(weights*metric_normalized_ray_t), no opacity normalization",
            "background_rgb": 0,
            "coordinate": "anchor OpenCV xyz",
            "image_size": [128, 160],
            "draft": True,
        },
    )
    write_json(
        root / "resolution_contract.json",
        {
            "primary_grid": [8, 8, 8],
            "planned_grid_sweep": [8, 16, 32],
            "renderer_samples_fixed_for_primary_sweep": 64,
            "secondary_matrix": {"grids": [8, 16], "samples": [64, 128]},
            "geometry_bounds_sensitivity_grids": [8],
            "query_scene_ids_unchanged": True,
            "draft": True,
        },
    )
    write_json(
        root / "capacity_contract.json",
        {
            "role": "CAPACITY_DEV_EXPOSED attribution only",
            "n_scenes": 17,
            "CURRENT_BOUNDS": [[-6, -4, -6], [6, 4, 6]],
            "alternative_bounds_file": "predeclared_geometry_bounds.json",
            "no_carrier_training": True,
            "no_support_redesign": True,
            "final_holdout_access": False,
            "direct_fields": ["density logits", "RGB color logits"],
            "draft": True,
        },
    )
    return manifest


@torch.no_grad()
def replay(root, manifest, device):
    saved = OLD_SUPPORT / "raw/oracle_run/sealed_states.pt"
    states = torch.load(saved, map_location=device, weights_only=False)
    old = json.loads((OLD_SUPPORT / "raw/oracle_run/results.json").read_text())
    reference = {
        (r["scene_id"], r["query_id"], r["method"]): r
        for r in old
        if r["design"] == "R0" and r["method"] in ("A", "B", "anchor", "prior")
    }
    renderer = FixedMeasurementRenderer(n_samples=64, ray_chunk_size=2048).to(device)
    logs = []
    rows = []
    comparisons = []
    output = root / "raw/baseline_predictions"
    output.mkdir(parents=True, exist_ok=True)
    for scene in manifest["scenes"]:
        sid = scene["scene_id"]
        loader = TrainScene(scene, manifest["image_size"], root, device, logs)
        sealed_anchor = states[f"R0/{sid}/A"].anchor_c2w
        for fid in scene["roles"]["primary_query"]:
            frame = loader.frames[fid]
            camera = batched_camera(
                transform_cameras(loader.camera(frame), torch.linalg.inv(sealed_anchor))
            )
            truth = loader.rgb(frame)
            depth = (
                torch.from_numpy(np.load(loader.path(frame, "depth"))).float().to(device).squeeze()
            )
            logs.append(
                {
                    "scene_id": sid,
                    "frame_id": fid,
                    "purpose": "replay_preexisting_sealed_R0_state_evaluator",
                    "kind": "query_camera_RGB_depth",
                }
            )
            for method in ("A", "B", "anchor", "prior"):
                if method == "prior":
                    pred = {
                        "rgb": torch.full_like(truth[None, None], 0.5),
                        "depth": torch.full_like(depth[None, None, None], 5.0),
                        "visibility": torch.ones_like(depth[None, None, None]),
                    }
                else:
                    sealed = states[f"R0/{sid}/{method}"]
                    assert hash_scene_state(sealed.scene_state) == sealed.state_hash
                    pred = renderer(sealed.scene_state, camera, ("rgb", "depth", "visibility"))
                    assert hash_scene_state(sealed.scene_state) == sealed.state_hash
                metrics = measurement_metrics(
                    pred["rgb"][0, 0],
                    truth,
                    pred["depth"][0, 0, 0],
                    depth,
                    pred["visibility"][0, 0, 0],
                )
                row = {
                    "scene_id": sid,
                    "query_id": fid,
                    "method": method,
                    "query_camera_hash": hash_value(camera),
                    "prediction_hash": hash_value(pred),
                    **metrics,
                }
                rows.append(row)
                ref = reference[(sid, fid, method)]
                deltas = {m: abs(row[m] - ref[m]) for m in METRICS}
                comparisons.append(
                    {
                        "scene_id": sid,
                        "query_id": fid,
                        "method": method,
                        "metric_abs_deltas": deltas,
                        "camera_hash_exact": row["query_camera_hash"] == ref["query_camera_hash"],
                        "prediction_hash_exact": row["prediction_hash"] == ref["prediction_hash"],
                    }
                )
                np.savez_compressed(
                    output / f"{sid}_{fid}_{method}.npz",
                    rgb=pred["rgb"][0, 0].cpu().numpy(),
                    depth=pred["depth"][0, 0, 0].cpu().numpy(),
                    opacity=pred["visibility"][0, 0, 0].cpu().numpy(),
                )
    write_json(root / "baseline_results.json", rows)
    write_json(root / "audit/baseline_access_log.json", logs)
    maxdelta = max(v for row in comparisons for v in row["metric_abs_deltas"].values())
    integrity = {
        "status": "PASS"
        if maxdelta <= 1e-6 and all(r["camera_hash_exact"] for r in comparisons)
        else "FAIL",
        "rows": len(rows),
        "max_metric_absolute_delta": maxdelta,
        "tolerance": 1e-6,
        "camera_hashes_all_exact": all(r["camera_hash_exact"] for r in comparisons),
        "prediction_hashes_all_exact": all(r["prediction_hash_exact"] for r in comparisons),
        "comparisons": comparisons,
        "checkpoint_sha256": sha(CHECKPOINT),
        "source_sealed_states_sha256": sha(saved),
        "source_results_sha256": sha(OLD_SUPPORT / "raw/oracle_run/results.json"),
        "carrier_called": False,
        "writer_called": False,
        "final_holdout_media_loaded": False,
    }
    write_json(root / "baseline_integrity.json", integrity)
    if integrity["status"] != "PASS":
        raise RuntimeError("Baseline not reproducible; stop capacity experiments")
    return integrity


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cuda")
    a = p.parse_args()
    root = Path(a.output)
    manifest = prepare(root)
    result = replay(root, manifest, a.device)
    print(json.dumps({k: v for k, v in result.items() if k != "comparisons"}, indent=2))
