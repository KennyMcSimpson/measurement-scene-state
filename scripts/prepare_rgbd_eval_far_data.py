#!/usr/bin/env python3
"""Lock and materialize EVAL-FAR: never-read far query frames for the EVAL-V3 and EVAL-V4 scenes.

Every EVAL-V3/V4 episode used the scene's first 16 official frames (contexts 0-4, queries 14/15),
where 94% of query surfaces lie inside the context-depth volume. EVAL-FAR keeps each scene's
prepared frames and context roles unchanged and adds two far query frames later on the same
cam_00 trajectory: the first valid official frame at or after position 32 of the scene's allowed
frame list, and the first valid one at or after position 48. No model is run and no score is read.

`--stage lock` is metadata only (the official split, the frozen EVAL-V3/V4 manifests and the
camera trajectories already on disk). `--stage prepare` fetches only the locked frame members
through exact HTTP ranges, applies the frozen frame-quality rule, and writes a manifest in the
format of the Hypersim cohorts with `far_query` roles.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import importlib.util
import io
import json
import zipfile
import zlib
from pathlib import Path

import h5py
import numpy as np
import requests
from PIL import Image

from mcss.data.hypersim import convert_hypersim_pose
from mcss.mechanism_pilot.calibration import calibrated_intrinsics, published_scene_metadata
from mcss.mechanism_pilot.data_prep import BoundedRangeReader, resize_rgb_depth
from mcss.mechanism_pilot.unseen_data import csv_rows, frame_quality

EXPERIMENT = "EXP-3D-RGBD-EVAL-FAR-DATA"
SPLIT = "EVAL_FAR"
URL = "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/{}.zip"
SOURCES = {
    "EVAL_V3": Path("outputs/EXP-3D-RGBD-EVAL-V3-DATA"),
    "EVAL_V4": Path("outputs/EXP-3D-RGBD-EVAL-V4-DATA"),
}
MANIFESTS = {"EVAL_V3": "manifest_eval_v3.json", "EVAL_V4": "manifest_eval_v4.json"}
POSITIONS = (32, 48)
CANDIDATES_PER_SLOT = 4
BUDGET = 1024**3
USE = (
    "EVAL-FAR is a mechanism cohort for queries far from the context frames. Its scenes were "
    "used by V11, INPAINT-V1, the EVAL-V4 replication and V12 with near queries (frames 14/15); "
    "the far query frames themselves were never read. It is never a qualification cohort; the "
    "protected final holdout is not opened."
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def read(path):
    return json.loads(Path(path).read_text())


def v1_module():
    spec = importlib.util.spec_from_file_location(
        "v1_prep", "scripts/prepare_rgbd_fresh_scale_data.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def far_slots(allowed, prepared):
    """Ordered candidates per slot: allowed frames from each position, never a prepared one."""
    frames = sorted(allowed)
    slots = []
    for position in POSITIONS:
        candidates = [f for f in frames[position:] if f not in prepared][:CANDIDATES_PER_SLOT]
        slots.append(candidates)
    return slots


def build_lock():
    v1 = v1_module()
    paths = v1.source_paths(Path(".").resolve())
    allowed = v1.allowed_frames(paths["official_split"], "train")
    scenes, skipped = {}, {}
    for origin, root in SOURCES.items():
        manifest = root / MANIFESTS[origin]
        for record in read(manifest)["scenes"]:
            sid = record["scene_id"]
            prepared = {f["frame_id"] for f in record["frames"]}
            slots = far_slots(allowed.get(sid, []), prepared)
            if any(not s for s in slots):
                skipped[sid] = "fewer than 49 official frames on cam_00"
                continue
            poses = root / "raw" / sid / "_detail" / "cam_00"
            scenes[sid] = {
                "origin": origin,
                "slots": slots,
                "trajectory_sha256": {
                    p.name: sha(p) for p in sorted(poses.glob("camera_keyframe_*.hdf5"))
                },
                "scene_metadata_sha256": sha(root / "raw" / sid / "_detail" / "metadata_scene.csv"),
            }
    return {
        "schema": "mcss.rgbd_eval_far.cohort.v1",
        "experiment": EXPERIMENT,
        "rule": "per scene, slot k takes the first frame of the allowed cam_00 list at or after "
        f"position {POSITIONS} (never a prepared frame) that passes the frozen frame-quality and "
        f"camera checks, at most {CANDIDATES_PER_SLOT} candidates per slot in list order; a scene "
        "without two valid far frames is excluded; contexts and all other roles are unchanged",
        "scenes": scenes,
        "skipped": skipped,
        "source_manifests_sha256": {o: sha(r / MANIFESTS[o]) for o, r in SOURCES.items()},
        "official_split_sha256": sha(paths["official_split"]),
        "calibration_sha256": sha(paths["calibration"]),
        "source_sha256": {str(Path(__file__).resolve()): sha(Path(__file__).resolve())},
        "use": USE,
        "data_budget_bytes": BUDGET,
        "image_or_depth_member_read": False,
        "model_forward_run": False,
        "created_utc": datetime.datetime.now(datetime.UTC).isoformat(),
    }


def lock(root):
    root = Path(root).resolve()
    if root.name != EXPERIMENT:
        raise PermissionError("Exact experiment directory required")
    if (root / "candidate_lock.json").exists():
        raise FileExistsError("Candidate lock exists; never re-lock")
    value = build_lock()
    write(root / "candidate_lock.json", value)
    print("LOCKED", len(value["scenes"]), "scenes; skipped", len(value["skipped"]), flush=True)


def trajectory_pose(raw, fid, scale):
    def hdf(path):
        with h5py.File(path) as handle:
            return np.array(handle["dataset"])

    ids = hdf(raw / "_detail/cam_00/camera_keyframe_frame_indices.hdf5").reshape(-1)
    positions = hdf(raw / "_detail/cam_00/camera_keyframe_positions.hdf5")
    rotations = hdf(raw / "_detail/cam_00/camera_keyframe_orientations.hdf5")
    j = {int(f): i for i, f in enumerate(ids)}[fid]
    rotation = rotations[j]
    orth = float(np.max(np.abs(rotation.T @ rotation - np.eye(3))))
    det = float(np.linalg.det(rotation))
    return convert_hypersim_pose(rotation, positions[j], meters_per_asset_unit=scale), orth, det


def prepare(root, opener=None):
    root = Path(root).resolve()
    lock_ = read(root / "candidate_lock.json")
    if (root / "manifest_eval_far.json").exists():
        raise FileExistsError("Completed manifest_eval_far.json exists; no overwrite")
    for origin, digest in lock_["source_manifests_sha256"].items():
        if sha(SOURCES[origin] / MANIFESTS[origin]) != digest:
            raise PermissionError(f"{origin} manifest changed since the lock")
    v1 = v1_module()
    paths = v1.source_paths(Path(".").resolve())
    calibration = published_scene_metadata(paths["calibration"])
    manifests = {
        o: {r["scene_id"]: r for r in read(SOURCES[o] / MANIFESTS[o])["scenes"]} for o in SOURCES
    }
    network, records, failures, qualities, hashes = [0], [], {}, {}, {}
    with (root / "access.jsonl").open("a") as log:
        for sid, entry in lock_["scenes"].items():
            source_root = SOURCES[entry["origin"]]
            raw = source_root / "raw" / sid
            for name, digest in entry["trajectory_sha256"].items():
                if sha(raw / "_detail/cam_00" / name) != digest:
                    raise PermissionError(f"{sid}: trajectory changed since the lock")
            units = {
                r["parameter_name"]: r["parameter_value"]
                for r in csv_rows(raw / "_detail/metadata_scene.csv")
            }
            scale = float(units["meters_per_asset_unit"])
            if opener is None:
                session = requests.Session()
                reader = BoundedRangeReader(URL.format(sid), session, byte_budget=BUDGET)
                reader.shared_budget = network
                archive = zipfile.ZipFile(reader)
            else:
                reader, archive = opener(sid)
            chosen, quality = [], {}
            with archive:
                names = {i.filename: i for i in archive.infolist()}
                for slot in entry["slots"]:
                    for fid in slot:
                        images = f"{sid}/images/scene_cam_00"
                        members = {
                            "rgb": f"{images}_final_preview/frame.{fid:04d}.color.jpg",
                            "depth": f"{images}_geometry_hdf5/frame.{fid:04d}.depth_meters.hdf5",
                        }
                        if any(m not in names for m in members.values()):
                            continue
                        data = {}
                        for kind, member in members.items():
                            blob = archive.read(member)
                            if zlib.crc32(blob) != names[member].CRC:
                                raise ValueError(f"{member}: CRC mismatch")
                            log.write(
                                json.dumps(
                                    {
                                        "scene": sid,
                                        "member": member,
                                        "purpose": "EVAL_FAR_DATA_PREPARATION_NO_FORWARD",
                                    }
                                )
                                + "\n"
                            )
                            data[kind] = blob
                        rgb = np.asarray(Image.open(io.BytesIO(data["rgb"])).convert("RGB"))
                        with h5py.File(io.BytesIO(data["depth"])) as handle:
                            depth = np.array(handle["dataset"]).astype(np.float64)
                        pose, orth, det = trajectory_pose(raw, fid, scale)
                        q = frame_quality(rgb, depth)
                        q.update(camera_orthogonality_error=orth, camera_determinant=det)
                        q["eligible"] = q["eligible"] and orth <= 0.001 and det > 0
                        quality[fid] = q
                        if not q["eligible"]:
                            continue
                        rgb_small, depth_small = resize_rgb_depth(rgb, depth)
                        prepared = root / "prepared" / sid
                        prepared.mkdir(parents=True, exist_ok=True)
                        rgb_out, depth_out = (
                            prepared / f"{fid:04d}.png",
                            prepared / f"{fid:04d}.npy",
                        )
                        Image.fromarray(rgb_small).save(rgb_out)
                        np.save(depth_out, depth_small)
                        hashes[str(rgb_out)], hashes[str(depth_out)] = sha(rgb_out), sha(depth_out)
                        chosen.append(
                            {
                                "frame_id": fid,
                                "rgb": str(rgb_out),
                                "depth": str(depth_out),
                                "intrinsics": calibrated_intrinsics(
                                    calibration[sid], (128, 160)
                                ).tolist(),
                                "c2w": pose.tolist(),
                            }
                        )
                        break
            qualities[sid] = {str(k): v for k, v in quality.items()}
            if len(chosen) != len(POSITIONS) or chosen[0]["frame_id"] == chosen[1]["frame_id"]:
                failures[sid] = {
                    "status": "NO_TWO_VALID_FAR_FRAMES",
                    "chosen": [c["frame_id"] for c in chosen],
                }
                print("INELIGIBLE", sid, flush=True)
                continue
            base = manifests[entry["origin"]][sid]
            record = json.loads(json.dumps(base))
            record["split"] = SPLIT
            record["origin_split"] = base["split"]
            record["frames"] = base["frames"] + chosen
            record["roles"] = {**base["roles"], "far_query": [c["frame_id"] for c in chosen]}
            records.append(record)
            print("PREPARED", sid, [c["frame_id"] for c in chosen], flush=True)
    write(root / "frame_quality.json", qualities)
    write(root / "scene_failures.json", failures)
    write(root / "hashes.json", hashes)
    status = "PASS" if len(records) >= 60 else "BLOCKED_INSUFFICIENT_VALID_SCENES"
    write(
        root / "manifest_eval_far.json",
        {
            "schema": "mcss.rgbd_eval_far.data.v1",
            "image_size": [128, 160],
            "depth_semantics": "ray_distance_meters",
            "scenes": records,
            "candidate_lock_sha256": sha(root / "candidate_lock.json"),
            "protected_scene_ids": [],
        },
    )
    write(
        root / "preparation_integrity.json",
        {
            "status": status,
            SPLIT: len(records),
            "failures": failures,
            "network_bytes": network[0],
            "budget_bytes": BUDGET,
            "model_forward_run": False,
            "score_computed": False,
            "candidate_lock_sha256": sha(root / "candidate_lock.json"),
        },
    )
    print(status, len(records), "failures", len(failures), "bytes", network[0], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("lock", "prepare"), required=True)
    parser.add_argument("--root", type=Path, default=Path("outputs") / EXPERIMENT)
    args = parser.parse_args()
    (lock if args.stage == "lock" else prepare)(args.root)


if __name__ == "__main__":
    main()
