#!/usr/bin/env python3
"""Write the SEG-V1 manifest after the 4 GiB budget was exceeded inside a scene (no network).

The frozen `prepare_rgbd_segment_data.py --stage prepare` checks the budget only before each scene;
on 2026-10-03 the shared budget ran out while scene ai_029_002 (lock position 64 of 72) was being
downloaded, and the range reader raised before the manifest was written. The frozen rule reads the
locked scenes in order until the budget, so the outcome is the 240 segments completed before that
scene; the scene being read and the scenes after it are not used.

This script rebuilds those records from the files the run left (prepared frames, the raw camera
metadata it downloaded, frame_quality.json), with the frozen functions of the locked preparation
script and no network access. Every rebuilt frame-role support must equal the run's own
role_support.json, so the records are exactly what the run would have written. Then the manifest,
the hashes, the cost and the integrity record are written with the budget stop disclosed.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import h5py
import numpy as np

from mcss.data.hypersim import convert_hypersim_pose
from mcss.mechanism_pilot.calibration import calibrated_intrinsics, published_scene_metadata
from mcss.mechanism_pilot.optimization_bounds_contracts import FrozenTrainingPrior

PREPARE = Path("scripts/prepare_rgbd_segment_data.py")
ROOT = Path("outputs/EXP-3D-RGBD-SEGMENT-DATA-V1")
BUDGET_STOP_SCENE = "ai_029_002"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def prep_module():
    spec = importlib.util.spec_from_file_location("rgbd_segment_data", PREPARE)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def main():
    repo, root = Path(".").resolve(), ROOT.resolve()
    prep = prep_module()
    lock_path = root / "candidate_lock.json"
    lock = read(lock_path)
    if lock != prep.build_lock(repo):
        raise PermissionError("Candidate lock no longer reproduces from its frozen sources")
    for name in ("manifest_seg.json", "preparation_integrity.json"):
        if (root / name).exists():
            raise FileExistsError(f"{name} exists; nothing to recover")
    order = lock["SEG_scenes"]
    stop = order.index(BUDGET_STOP_SCENE)
    completed_scenes, unread = order[:stop], order[stop + 1 :]
    done = read(root / "progress.json")["TRAIN"]
    if {s.rsplit("_seg", 1)[0] for s in done} - set(completed_scenes):
        raise PermissionError("A prepared segment belongs to the budget-stop scene or later")
    trx = prep.train_ext_module(repo)
    paths = prep.source_paths(repo)
    calibration = published_scene_metadata(paths["calibration"])
    bounds = read(paths["bounds"])
    prior = FrozenTrainingPrior(bounds["near_m"], bounds["far_m"], bounds["source_sha256"])
    camera_only_roles = trx.eval_v3_module(repo).v1_module(repo).role_rule(repo)
    qualities = read(root / "frame_quality.json")
    supports = read(root / "role_support.json")
    failures = read(root / "scene_failures.json")
    records, cameras = [], {}
    for seg in done:
        sid, k = seg.rsplit("_seg", 1)[0], int(seg.rsplit("_seg", 1)[1])
        if sid not in cameras:
            raw = root / "raw" / sid
            units = {
                r["parameter_name"]: r["parameter_value"]
                for r in trx.csv_rows(raw / "_detail/metadata_scene.csv")
            }

            def hdf(name, raw=raw):
                with h5py.File(raw / "_detail/cam_00" / name) as handle:
                    return np.array(handle["dataset"])

            ids = hdf("camera_keyframe_frame_indices.hdf5").reshape(-1)
            cameras[sid] = (
                float(units["meters_per_asset_unit"]),
                {int(fid): j for j, fid in enumerate(ids)},
                hdf("camera_keyframe_positions.hdf5"),
                hdf("camera_keyframe_orientations.hdf5"),
            )
        scale, mapping, positions, rotations = cameras[sid]
        prepared = root / "prepared" / seg
        frame_ids = sorted(int(p.stem) for p in prepared.glob("*.png"))
        if len(frame_ids) != prep.SEGMENT_LENGTH or seg in failures:
            raise PermissionError(f"Incomplete or failed segment listed as prepared: {seg}")
        frames = []
        for fid in frame_ids:
            pose = convert_hypersim_pose(
                rotations[mapping[fid]], positions[mapping[fid]], meters_per_asset_unit=scale
            )
            frames.append(
                {
                    "frame_id": fid,
                    "rgb": str(prepared / f"{fid:04d}.png"),
                    "depth": str(prepared / f"{fid:04d}.npy"),
                    "intrinsics": calibrated_intrinsics(calibration[sid], (128, 160)).tolist(),
                    "c2w": pose.tolist(),
                }
            )
        quality = {int(fid): value for fid, value in qualities[seg].items()}
        roles, support = camera_only_roles(seg, frames, quality, prior)
        if json.loads(json.dumps(support)) != supports[seg]:
            raise AssertionError(f"Rebuilt frame-role support differs from the run: {seg}")
        physical = lock["physical_scene_ids"][sid]
        records.append(
            {
                "scene_id": seg,
                "source_scene_id": sid,
                "segment": k,
                "split": prep.SPLIT,
                "historically_exposed": True,
                "historical_exposure_basis": lock["exposure_basis"],
                "physical_scene_id": f"{physical}#seg{k}",
                "source_physical_scene_id": physical,
                "official_split": "train",
                "roles": roles,
                "frames": frames,
            }
        )
    hashes = {}
    for sid in completed_scenes:
        for path in sorted((root / "raw" / sid).rglob("*")):
            if path.is_file():
                hashes[str(path)] = sha(path)
    for folder in sorted((root / "prepared").iterdir()):
        if folder.name.rsplit("_seg", 1)[0] in completed_scenes:
            for path in sorted(folder.iterdir()):
                hashes[str(path)] = sha(path)
    partial = sorted(str(p) for p in (root / "raw" / BUDGET_STOP_SCENE).rglob("*") if p.is_file())
    prep.write(root / "hashes.json", hashes)
    prep.write(
        root / "download_cost.json",
        {
            "network_bytes": f">= {prep.BUDGET} (the budget; the exact count was lost when the "
            f"budget was exceeded inside {BUDGET_STOP_SCENE})",
            "full_archives_downloaded": False,
            "budget_bytes": prep.BUDGET,
        },
    )
    status = "PASS" if len(records) >= prep.MIN_VALID_SEGMENTS else "BLOCKED_INSUFFICIENT_SEGMENTS"
    prep.write(
        root / "manifest_seg.json",
        {
            "schema": "mcss.rgbd_segment.data.v1",
            "image_size": [128, 160],
            "depth_semantics": "ray_distance_meters",
            "scenes": records,
            "candidate_lock_sha256": sha(lock_path),
            "protected_scene_ids": lock["protected_scene_ids"],
        },
    )
    prep.write(
        root / "preparation_integrity.json",
        {
            "status": status,
            "SEG": len(records),
            "scenes_with_segments": len({r["source_scene_id"] for r in records}),
            "budget_stop": True,
            "budget_stop_scene": BUDGET_STOP_SCENE,
            "budget_stop_detail": "the shared 4 GiB budget ran out inside the scene's download; "
            "the frozen per-scene check had passed before it, so the reader raised; the scene and "
            "the scenes after it in lock order are not used",
            "budget_stop_partial_files": partial,
            "unread_scenes": unread,
            "failures": failures,
            "candidate_replacement": "none; every locked scene is read in order until the budget",
            "recovered_by": "scripts/recover_rgbd_segment_data.py (no network; every rebuilt "
            "frame-role support equals the run's role_support.json)",
            "recovery_source_sha256": sha(Path(__file__)),
            "protected_media_read": False,
            "model_forward_run": False,
            "candidate_lock_sha256": sha(lock_path),
        },
    )
    print(
        status, len(records), "segments from", len(completed_scenes), "scenes; unread", len(unread)
    )


if __name__ == "__main__":
    main()
