#!/usr/bin/env python3
"""Lock and materialize SEG-V1, later cam_00 trajectory segments of the TRAIN72 scenes (no model).

`--stage lock` is metadata-only and freezes the ordered scene list and the segment rule before any
media byte is read; `--stage prepare` downloads the locked segments in order and applies to every
segment the frozen model-free validity rules and the frozen camera-only frame-role rule that made
the original 16-frame TRAIN72 records. No model is run and no score is read. The rule is fixed by
outputs/EXP-3D-RGBD-SEGMENT-DIVERSITY-V21/PLAN_BEFORE_V21_DATA.md, written before this script.

SEG-V1: for each TRAIN72 scene (official train split), the cam_00 frames that have both RGB and
depth and are public in the official split, sorted by frame id. Segment k (k = 1..4) takes the
positions 16k..16k+15; the original record is positions 0..15. A segment with fewer than 16 frames
or failing a rule is dropped, never replaced. A segment record gets the id `<scene>_seg<k>` and the
physical id `<scene physical id>#seg<k>` (validate_split requires unique physical ids), and keeps
the source scene and physical id. Status PASS needs at least 72 valid segments (the training set at
least doubles), else BLOCKED_INSUFFICIENT_SEGMENTS.
"""

from __future__ import annotations

import argparse
import hashlib
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
from mcss.mechanism_pilot.unseen_data import frame_quality

EXPERIMENT = "EXP-3D-RGBD-SEGMENT-DATA-V1"
SPLIT = "TRAIN"
URL = "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/{}.zip"
PLAN = Path("outputs/EXP-3D-RGBD-SEGMENT-DIVERSITY-V21/PLAN_BEFORE_V21_DATA.md")
PLAN_SHA256 = "f04e581e398ff1a47e14f3a0d46df2ebb0275a5d017b03ba3ce2fb26f74f2d24"
TRAIN_EXT_SCRIPT = Path("scripts/prepare_rgbd_train_ext_data.py")
SEGMENTS = (1, 2, 3, 4)
SEGMENT_LENGTH = 16
MIN_VALID_SEGMENTS = 72
BUDGET = 4 * 1024**3
SELECTION_SEED = "SEG-V1-20261003"
EXPOSURE_BASIS = (
    "SEG-V1 is training data only: later cam_00 segments of the TRAIN72 scenes, which are training "
    "scenes of every experiment since V6; no frame of an evaluation, protected or TRAIN-EXT scene "
    "is read. The protected official-test final holdout is not opened."
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def read(path):
    return json.loads(Path(path).read_text())


def train_ext_module(repo):
    """The frozen TRAIN-EXT preparation, reused as a library (never modified)."""
    spec = importlib.util.spec_from_file_location("rgbd_train_ext_data", repo / TRAIN_EXT_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def source_paths(repo):
    paths = train_ext_module(repo).source_paths(repo)
    paths["train_ext_preparation"] = paths.pop("preparation_source")
    paths.update(plan=repo / PLAN, preparation_source=Path(__file__).resolve())
    return paths


def selection_key(scene):
    return hashlib.sha256(f"{SELECTION_SEED}:{scene}".encode()).hexdigest()


def segment_id(scene, k):
    return f"{scene}_seg{k}"


def build_lock(repo):
    if sha(repo / PLAN) != PLAN_SHA256:
        raise PermissionError("The V21 plan changed or is missing")
    trx = train_ext_module(repo)
    paths = source_paths(repo)
    partitions = trx.csv_rows(paths["partitions"])
    sets, _, _ = trx.train_ext_sets(repo, paths, partitions)
    current = read(paths["current_manifest"])
    train72 = {r["scene_id"]: r for r in current["scenes"] if r["split"] == "TRAIN"}
    if len(train72) != 72:
        raise ValueError("TRAIN72 manifest expected")
    if set(train72) & sets["protected"]:
        raise PermissionError("A protected scene in TRAIN72")
    official = {r["scene_name"]: r["official_split"] for r in partitions}
    if any(official.get(s) != "train" for s in train72):
        raise PermissionError("SEG-V1 reads official train scenes only")
    order = sorted(train72, key=selection_key)
    return {
        "schema": "mcss.rgbd_segment.cohort.v1",
        "experiment": EXPERIMENT,
        "SEG_scenes": order,
        "segments": list(SEGMENTS),
        "segment_length": SEGMENT_LENGTH,
        "physical_scene_ids": {s: train72[s]["physical_scene_id"] for s in order},
        "rule": (
            "every TRAIN72 scene in sha256('" + SELECTION_SEED + ":'+scene) order; cam_00 frames "
            "with RGB and depth members that are public in the official train split, sorted by "
            "frame id; segment k in 1..4 = positions 16k..16k+15; the frozen model-free validity "
            "rules (calibration scale, geometry check on the segment's first frame, frame quality, "
            "camera matrix checks) and the frozen camera-only frame-role rule of the original "
            "records; a failing segment is dropped, never replaced"
        ),
        "minimum_valid_segments": MIN_VALID_SEGMENTS,
        "exposure_basis": EXPOSURE_BASIS,
        "protected_scene_ids": sorted(sets["protected"]),
        "excluded_from_reading": {
            k: sorted(v) for k, v in sets.items() if k not in ("train72", "previously_invalid")
        },
        "plan_sha256": PLAN_SHA256,
        "source_sha256": {str(p): sha(p) for p in paths.values()},
        "data_budget_bytes": BUDGET,
        "model_forward_run": False,
        "protected_media_read": False,
    }


def members_of(scene, infos, allowed_ids):
    """(metadata members, {k: (frame ids, members)}) of the locked segment rule."""
    names = {x.filename: x for x in infos}
    rgb, depth = {}, {}
    prefix = f"{scene}/images/"
    for name in names:
        if name.startswith(prefix + "scene_cam_00_final_preview/frame.") and name.endswith(
            ".color.jpg"
        ):
            rgb[int(name.split("frame.")[1][:4])] = name
        if name.startswith(prefix + "scene_cam_00_geometry_hdf5/frame.") and name.endswith(
            ".depth_meters.hdf5"
        ):
            depth[int(name.split("frame.")[1][:4])] = name
    ids = sorted(rgb.keys() & depth.keys() & set(allowed_ids))
    meta = [f"{scene}/_detail/metadata_scene.csv"] + [
        f"{scene}/_detail/cam_00/camera_keyframe_{k}.hdf5"
        for k in ("frame_indices", "positions", "orientations")
    ]
    if any(n not in names for n in meta):
        raise ValueError("Missing camera metadata")
    segments = {}
    for k in SEGMENTS:
        seg = ids[SEGMENT_LENGTH * k : SEGMENT_LENGTH * (k + 1)]
        if len(seg) < SEGMENT_LENGTH:
            continue
        members = [n for i in seg for n in (rgb[i], depth[i])]
        members.append(f"{prefix}scene_cam_00_geometry_hdf5/frame.{seg[0]:04d}.position.hdf5")
        if all(n in names for n in members):
            segments[k] = (seg, members)
    return names, meta, segments


def prepare(repo, root):
    repo, root = Path(repo).resolve(), Path(root).resolve()
    lock_path = root / "candidate_lock.json"
    lock = read(lock_path)
    if lock != build_lock(repo):
        raise PermissionError("Candidate lock no longer reproduces from its frozen sources")
    if (root / "manifest_seg.json").exists():
        raise FileExistsError("Completed manifest_seg.json exists; no overwrite")
    trx = train_ext_module(repo)
    v1 = trx.eval_v3_module(repo).v1_module(repo)
    paths = source_paths(repo)
    protected = set(lock["protected_scene_ids"])
    forbidden = set().union(*(set(v) for v in lock["excluded_from_reading"].values()))
    calibration = published_scene_metadata(paths["calibration"])
    bounds = read(paths["bounds"])
    prior = FrozenTrainingPrior(bounds["near_m"], bounds["far_m"], bounds["source_sha256"])
    camera_only_roles = v1.role_rule(repo)
    allowed = trx.allowed_frames(v1, paths["official_split"])
    network = [0]
    hashes, qualities, geometry, failures, supports = {}, {}, [], {}, {}
    records, budget_stop = [], False
    with (root / "access.jsonl").open("a") as log:

        def access(sid, path, operation):
            if sid in protected or sid in forbidden:
                raise PermissionError("Only TRAIN72 scenes may be read")
            entry = {
                "time": time.time(),
                "scene_id": sid,
                "path": str(path),
                "operation": operation,
                "purpose": "SEG_V1_DATA_PREPARATION_NO_FORWARD",
            }
            log.write(json.dumps(entry) + "\n")
            log.flush()

        for sid in lock["SEG_scenes"]:
            if network[0] >= BUDGET:
                budget_stop = True
                break
            session = requests.Session()
            try:
                reader = BoundedRangeReader(URL.format(sid), session)
                reader.shared_budget = network
                with zipfile.ZipFile(reader) as archive:
                    names, meta, segments = members_of(sid, archive.infolist(), allowed[sid])
                    for k in SEGMENTS:
                        if k not in segments:
                            failures[segment_id(sid, k)] = {"status": "INELIGIBLE_TOO_FEW_FRAMES"}
                    wanted = meta + [n for _, members in segments.values() for n in members]
                    for name in wanted:
                        dest = root / "raw" / name
                        access(sid, name, "download_locked_member")
                        info = names[name]
                        if dest.exists():
                            content = dest.read_bytes()
                            if len(content) != info.file_size or zlib.crc32(content) != info.CRC:
                                raise ValueError("Existing download failed CRC; stop")
                        else:
                            dest.parent.mkdir(parents=True, exist_ok=True)
                            dest.write_bytes(archive.read(name))
                        hashes[str(dest)] = sha(dest)
                reader.close()
            finally:
                session.close()
            write(root / "scene_failures.json", failures)
            raw = root / "raw" / sid
            units = {
                r["parameter_name"]: r["parameter_value"]
                for r in trx.csv_rows(raw / "_detail/metadata_scene.csv")
            }
            scale = float(units["meters_per_asset_unit"])
            if not np.isclose(
                scale, float(calibration[sid]["settings_units_info_meters_scale"]), rtol=1e-6
            ):
                for k in segments:
                    failures[segment_id(sid, k)] = {"status": "INELIGIBLE_CALIBRATION_SCALE"}
                write(root / "scene_failures.json", failures)
                continue

            def hdf(path, sid=sid):
                access(sid, path, "decode_hdf5")
                with h5py.File(path) as handle:
                    return np.array(handle["dataset"])

            ids = hdf(raw / "_detail/cam_00/camera_keyframe_frame_indices.hdf5").reshape(-1)
            positions = hdf(raw / "_detail/cam_00/camera_keyframe_positions.hdf5")
            rotations = hdf(raw / "_detail/cam_00/camera_keyframe_orientations.hdf5")
            mapping = {int(fid): j for j, fid in enumerate(ids)}
            for k, (frame_ids, _) in segments.items():
                seg = segment_id(sid, k)
                frames, quality = [], {}
                prepared = root / "prepared" / seg
                prepared.mkdir(parents=True, exist_ok=True)
                for fid in frame_ids:
                    rgb_path = raw / f"images/scene_cam_00_final_preview/frame.{fid:04d}.color.jpg"
                    access(sid, rgb_path, "decode_RGB")
                    with Image.open(rgb_path) as im:
                        rgb = np.asarray(im.convert("RGB"))
                    depth = hdf(
                        raw / f"images/scene_cam_00_geometry_hdf5/frame.{fid:04d}.depth_meters.hdf5"
                    )
                    if rgb.shape[:2] != depth.shape:
                        raise ValueError("RGB/depth dimensions differ")
                    quality[fid] = frame_quality(rgb, depth)
                    rotation = rotations[mapping[fid]]
                    orth = float(np.max(np.abs(rotation.T @ rotation - np.eye(3))))
                    det = float(np.linalg.det(rotation))
                    quality[fid].update(camera_orthogonality_error=orth, camera_determinant=det)
                    quality[fid]["eligible"] = (
                        quality[fid]["eligible"] and orth <= 0.001 and det > 0
                    )
                    pose = convert_hypersim_pose(
                        rotation, positions[mapping[fid]], meters_per_asset_unit=scale
                    )
                    if fid == frame_ids[0]:
                        position = hdf(
                            raw / f"images/scene_cam_00_geometry_hdf5/frame.{fid:04d}.position.hdf5"
                        )
                        try:
                            geometry.append(
                                check_geometry(
                                    sid,
                                    depth,
                                    position,
                                    pose,
                                    calibrated_intrinsics(calibration[sid], depth.shape),
                                    scale,
                                )
                                | {"segment": seg}
                            )
                        except ValueError as exc:
                            failures[seg] = {
                                "status": "INELIGIBLE_SOURCE_GEOMETRY",
                                "error": str(exc),
                            }
                            geometry.append({"scene_id": sid, "segment": seg, **failures[seg]})
                        write(root / "geometry_checks.json", geometry)
                        write(root / "scene_failures.json", failures)
                    rgb_small, depth_small = resize_rgb_depth(rgb, depth)
                    rgb_out, depth_out = prepared / f"{fid:04d}.png", prepared / f"{fid:04d}.npy"
                    Image.fromarray(rgb_small).save(rgb_out)
                    np.save(depth_out, depth_small)
                    hashes[str(rgb_out)], hashes[str(depth_out)] = sha(rgb_out), sha(depth_out)
                    frames.append(
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
                qualities[seg] = quality
                write(root / "frame_quality.json", qualities)
                if seg in failures:
                    print("INELIGIBLE", seg, failures[seg]["status"], flush=True)
                    continue
                try:
                    roles, support = camera_only_roles(seg, frames, quality, prior)
                except ValueError as exc:
                    failures[seg] = {"status": "INELIGIBLE_FRAME_ROLES", "error": str(exc)}
                    write(root / "scene_failures.json", failures)
                    print("INELIGIBLE", seg, "INELIGIBLE_FRAME_ROLES", flush=True)
                    continue
                physical = lock["physical_scene_ids"][sid]
                records.append(
                    {
                        "scene_id": seg,
                        "source_scene_id": sid,
                        "segment": k,
                        "split": SPLIT,
                        "historically_exposed": True,
                        "historical_exposure_basis": lock["exposure_basis"],
                        "physical_scene_id": f"{physical}#seg{k}",
                        "source_physical_scene_id": physical,
                        "official_split": "train",
                        "roles": roles,
                        "frames": frames,
                    }
                )
                supports[seg] = support
                write(root / "role_support.json", supports)
                write(root / "progress.json", {SPLIT: [r["scene_id"] for r in records]})
                print("PREPARED", seg, flush=True)
    write(root / "hashes.json", hashes)
    write(
        root / "download_cost.json",
        {"network_bytes": network[0], "full_archives_downloaded": False, "budget_bytes": BUDGET},
    )
    status = "PASS" if len(records) >= MIN_VALID_SEGMENTS else "BLOCKED_INSUFFICIENT_SEGMENTS"
    write(
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
    write(
        root / "preparation_integrity.json",
        {
            "status": status,
            "SEG": len(records),
            "scenes_with_segments": len({r["source_scene_id"] for r in records}),
            "budget_stop": budget_stop,
            "failures": failures,
            "candidate_replacement": "none; every locked scene is read in order until the budget",
            "protected_media_read": False,
            "model_forward_run": False,
            "network_bytes": network[0],
            "candidate_lock_sha256": sha(lock_path),
        },
    )
    print(status, len(records), "failures", len(failures), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("lock", "prepare"), required=True)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--root", type=Path, default=Path("outputs") / EXPERIMENT)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.stage == "lock":
        if (root / "candidate_lock.json").exists():
            raise FileExistsError("Candidate lock exists; never re-lock")
        lock = build_lock(args.repo.resolve())
        root.mkdir(parents=True, exist_ok=True)
        write(root / "candidate_lock.json", lock)
        print(json.dumps({"SEG_scenes": len(lock["SEG_scenes"]), "segments": lock["segments"]}))
    else:
        prepare(args.repo, root)


if __name__ == "__main__":
    main()
