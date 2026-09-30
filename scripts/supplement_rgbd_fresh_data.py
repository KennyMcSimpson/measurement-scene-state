#!/usr/bin/env python3
"""FRESH-V1 Amendment 1: one *valid* scene per volume (metadata-only rule, model-free trigger).

The frozen lock kept the lexicographically first metadata-eligible val scene of each volume.
When that scene failed a frozen, model-free validity check, its volume had no FRESH scene.
Amendment 1 lets such a volume use its next metadata-eligible val scene (same rules, still one
scene per volume). No model has been run on any FRESH scene; no score is involved.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import time
import zipfile
import zlib
from pathlib import Path

import h5py
import numpy as np
import requests
from PIL import Image

from mcss.data.hypersim import convert_hypersim_pose
from mcss.mechanism_pilot.calibration import calibrated_intrinsics, published_scene_metadata
from mcss.mechanism_pilot.data_prep import BoundedRangeReader, check_geometry, resize_rgb_depth
from mcss.mechanism_pilot.optimization_bounds_contracts import FrozenTrainingPrior
from mcss.mechanism_pilot.support_redesign_data import safe_source_assets
from mcss.mechanism_pilot.unseen_data import asset_identity, csv_rows, frame_quality, select_members


def base_module(repo):
    spec = importlib.util.spec_from_file_location(
        "fresh_scale_data", repo / "scripts/prepare_rgbd_fresh_scale_data.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def supplement_candidates(base, repo, lock, failures):
    """Next metadata-eligible val scene, per volume whose locked FRESH candidate failed."""
    paths = base.source_paths(repo)
    assets = safe_source_assets(paths["assets"])
    partitions = csv_rows(paths["partitions"])
    trajectories = {r["Animation"]: r for r in csv_rows(paths["trajectories"])}
    allowed = base.allowed_frames(paths["official_split"], "val")
    current = set(lock["current_scene_ids"])
    protected, excluded = set(lock["protected_scene_ids"]), set(lock["excluded_scene_ids"])
    failed_volumes = {base.volume(s) for s in lock["FRESH_V1_candidates"] if s in failures}
    valid_volumes = {base.volume(s) for s in lock["FRESH_V1_candidates"] if s not in failures}
    used_volumes = {base.volume(s) for s in current}
    blocked = {asset_identity(assets[s]) for s in current | protected if s in assets}
    blocked |= {asset_identity(assets[s]) for s in lock["FRESH_V1_candidates"]}
    chosen, filled = [], set()
    for row in sorted(partitions, key=lambda r: r["scene_name"]):
        sid, vol = row["scene_name"], base.volume(row["scene_name"])
        if (
            vol not in failed_volumes
            or vol in filled | valid_volumes | used_volumes
            or sid in lock["FRESH_V1_candidates"]
            or row["official_split"] != "val"
            or row["protocol_partition"] != "val"
            or row["previously_observed"] != "False"
            or sid in protected
            or sid in excluded
            or sid in current
            or sid not in assets
            or asset_identity(assets[sid]) in blocked
            or len(allowed.get(sid, [])) < 16
            or base.bad_trajectory(trajectories, sid)
        ):
            continue
        chosen.append(sid)
        filled.add(vol)
        blocked.add(asset_identity(assets[sid]))
    return chosen, assets


def run(repo, root):
    repo, root = Path(repo).resolve(), Path(root).resolve()
    base = base_module(repo)
    lock = json.loads((root / "candidate_lock.json").read_text())
    integrity = json.loads((root / "preparation_integrity.json").read_text())
    failures = json.loads((root / "scene_failures.json").read_text())
    fresh = json.loads((root / "manifest_fresh.json").read_text())
    if (root / "amendment1_lock.json").exists():
        raise FileExistsError("Amendment 1 already locked")
    candidates, assets = supplement_candidates(base, repo, lock, failures)
    amendment = {
        "schema": "mcss.rgbd_fresh_scale.fresh_amendment1.v1",
        "rule": "one VALID scene per volume: a volume whose locked FRESH candidate failed a "
        "frozen model-free validity check may use its next metadata-eligible val scene "
        "(lexicographic, same rules, still one scene per volume)",
        "trigger": {s: failures[s]["status"] for s in lock["FRESH_V1_candidates"] if s in failures},
        "candidate_lock_sha256": base.sha(root / "candidate_lock.json"),
        "preparation_integrity_before": integrity,
        "supplement_candidates": candidates,
        "model_forward_run_on_fresh": False,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    base.write(root / "amendment1_lock.json", amendment)
    paths = base.source_paths(repo)
    calibration = published_scene_metadata(paths["calibration"])
    bounds = json.loads(paths["bounds"].read_text())
    prior = FrozenTrainingPrior(bounds["near_m"], bounds["far_m"], bounds["source_sha256"])
    camera_only_roles = base.role_rule(repo)
    allowed = base.allowed_frames(paths["official_split"], "val")
    records, supplement_failures, network = [], {}, [0]
    with (root / "access.jsonl").open("a") as log:

        def access(sid, path, operation):
            if sid in set(lock["protected_scene_ids"]):
                raise PermissionError("Protected scene access prohibited")
            log.write(
                json.dumps(
                    {
                        "time": time.time(),
                        "scene_id": sid,
                        "path": str(path),
                        "operation": operation,
                        "purpose": "FRESH_AMENDMENT1_DATA_PREPARATION_NO_FORWARD",
                    }
                )
                + "\n"
            )
            log.flush()

        for sid in candidates:
            session = requests.Session()
            try:
                reader = BoundedRangeReader(base.URL.format(sid), session)
                reader.shared_budget = network
                with zipfile.ZipFile(reader) as archive:
                    plan = select_members(sid, archive.infolist(), allowed[sid])
                    for member in plan["member_metadata"]:
                        dest = root / "raw" / member["name"]
                        access(sid, member["name"], "download_locked_member")
                        if dest.exists():
                            content = dest.read_bytes()
                            if (
                                len(content) != member["bytes"]
                                or zlib.crc32(content) != member["crc32"]
                            ):
                                raise ValueError("Existing download failed CRC; stop")
                        else:
                            dest.parent.mkdir(parents=True, exist_ok=True)
                            dest.write_bytes(archive.read(member["name"]))
                reader.close()
            finally:
                session.close()
            raw = root / "raw" / sid
            units = {
                r["parameter_name"]: r["parameter_value"]
                for r in csv_rows(raw / "_detail/metadata_scene.csv")
            }
            scale = float(units["meters_per_asset_unit"])

            def hdf(path, sid=sid):
                access(sid, path, "decode_hdf5")
                with h5py.File(path) as handle:
                    return np.array(handle["dataset"])

            ids = hdf(raw / "_detail/cam_00/camera_keyframe_frame_indices.hdf5").reshape(-1)
            positions = hdf(raw / "_detail/cam_00/camera_keyframe_positions.hdf5")
            rotations = hdf(raw / "_detail/cam_00/camera_keyframe_orientations.hdf5")
            mapping = {int(fid): j for j, fid in enumerate(ids)}
            frames, quality, failed = [], {}, None
            if not np.isclose(
                scale, float(calibration[sid]["settings_units_info_meters_scale"]), rtol=1e-6
            ):
                failed = {"status": "INELIGIBLE_CALIBRATION_SCALE"}
            prepared = root / "prepared" / sid
            prepared.mkdir(parents=True, exist_ok=True)
            for fid in plan["frame_ids"]:
                if failed:
                    break
                rgb_path = raw / f"images/scene_cam_00_final_preview/frame.{fid:04d}.color.jpg"
                access(sid, rgb_path, "decode_RGB")
                with Image.open(rgb_path) as im:
                    rgb = np.asarray(im.convert("RGB"))
                depth = hdf(
                    raw / f"images/scene_cam_00_geometry_hdf5/frame.{fid:04d}.depth_meters.hdf5"
                )
                quality[fid] = frame_quality(rgb, depth)
                rotation = rotations[mapping[fid]]
                orth = float(np.max(np.abs(rotation.T @ rotation - np.eye(3))))
                det = float(np.linalg.det(rotation))
                quality[fid]["eligible"] = quality[fid]["eligible"] and orth <= 0.001 and det > 0
                pose = convert_hypersim_pose(
                    rotation, positions[mapping[fid]], meters_per_asset_unit=scale
                )
                if fid == plan["frame_ids"][0]:
                    position = hdf(
                        raw / f"images/scene_cam_00_geometry_hdf5/frame.{fid:04d}.position.hdf5"
                    )
                    try:
                        check_geometry(
                            sid,
                            depth,
                            position,
                            pose,
                            calibrated_intrinsics(calibration[sid], depth.shape),
                            scale,
                        )
                    except ValueError as exc:
                        failed = {"status": "INELIGIBLE_SOURCE_GEOMETRY", "error": str(exc)}
                        break
                rgb_small, depth_small = resize_rgb_depth(rgb, depth)
                rgb_out, depth_out = prepared / f"{fid:04d}.png", prepared / f"{fid:04d}.npy"
                Image.fromarray(rgb_small).save(rgb_out)
                np.save(depth_out, depth_small)
                frames.append(
                    {
                        "frame_id": fid,
                        "rgb": str(rgb_out),
                        "depth": str(depth_out),
                        "intrinsics": calibrated_intrinsics(calibration[sid], (128, 160)).tolist(),
                        "c2w": pose.tolist(),
                    }
                )
            if failed is None:
                try:
                    roles, _ = camera_only_roles(sid, frames, quality, prior)
                except ValueError as exc:
                    failed = {"status": "INELIGIBLE_FRAME_ROLES", "error": str(exc)}
            if failed is not None:
                supplement_failures[sid] = failed
                print("INELIGIBLE", sid, failed["status"], flush=True)
                continue
            records.append(
                {
                    "scene_id": sid,
                    "split": "FRESH_QUALIFICATION",
                    "historically_exposed": False,
                    "historical_exposure_basis": lock["independence_basis"],
                    "physical_scene_id": "|".join(asset_identity(assets[sid])),
                    "roles": roles,
                    "frames": frames,
                }
            )
            print("PREPARED", sid, flush=True)
    amended = dict(fresh)
    amended["scenes"] = fresh["scenes"] + records
    amended["amendment1_lock_sha256"] = base.sha(root / "amendment1_lock.json")
    base.write(root / "manifest_fresh_amended.json", amended)
    base.write(
        root / "amendment1_result.json",
        {
            "supplement_prepared": [r["scene_id"] for r in records],
            "supplement_failures": supplement_failures,
            "fresh_scene_count": len(amended["scenes"]),
            "network_bytes": network[0],
        },
    )
    print("FRESH_AMENDED", len(amended["scenes"]), [r["scene_id"] for r in amended["scenes"]])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--root", type=Path, default=Path("outputs/EXP-3D-RGBD-FRESH-SCALE-DATA-V1")
    )
    args = parser.parse_args()
    run(args.repo, args.root)
