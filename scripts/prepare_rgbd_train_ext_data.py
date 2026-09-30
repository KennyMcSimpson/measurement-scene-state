#!/usr/bin/env python3
"""Lock and materialize TRAIN-EXT, extra training scenes for the V14 data-scale test (no model).

`--stage lock` is metadata-only and freezes the ordered candidate list before any media byte is
read; `--stage prepare` downloads the locked candidates in order and applies the frozen validity
rules and the frozen camera-only frame-role rule of TRAIN-X and EVAL-V3. No model is run and no
score is read. The rule is fixed by
outputs/EXP-3D-RGBD-DATA-SCALE-V14/PLAN_BEFORE_TRAIN_EXT_DATA.md, written before this script.

TRAIN-EXT: official Hypersim train or val split (same protocol partition), not protected or
excluded, in no evaluation cohort (DEV, FRESH-V1/V2 candidates, EVAL-V3, EVAL-V4) and not in
TRAIN72, never found invalid by an earlier preparation, volume disjoint from DEV, source asset
unique against every evaluation, protected and TRAIN72 scene and within the set, cam_00 >= 16
public frames, non-BAD trajectory, in sha256(seed:scene) order; every valid scene is kept.
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
from mcss.mechanism_pilot.support_redesign_data import safe_source_assets
from mcss.mechanism_pilot.unseen_data import asset_identity, csv_rows, frame_quality, select_members

EXPERIMENT = "EXP-3D-RGBD-TRAIN-EXT-DATA"
SPLIT = "TRAIN"
URL = "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/{}.zip"
PLAN = Path("outputs/EXP-3D-RGBD-DATA-SCALE-V14/PLAN_BEFORE_TRAIN_EXT_DATA.md")
PLAN_SHA256 = "526c3623752317da05922d5368e341938ba9fc7b660b7e504746e910ae6eb1b6"
EVAL_V3_SCRIPT = Path("scripts/prepare_rgbd_eval_v3_data.py")
EVAL_MANIFESTS = {
    "EVAL_V3": Path("outputs/EXP-3D-RGBD-EVAL-V3-DATA/manifest_eval_v3.json"),
    "EVAL_V4": Path("outputs/EXP-3D-RGBD-EVAL-V4-DATA/manifest_eval_v4.json"),
}
FAILURES = {
    "FRESH_SCALE_V1": Path("outputs/EXP-3D-RGBD-FRESH-SCALE-DATA-V1/scene_failures.json"),
    "FRESH_V2": Path("outputs/EXP-3D-RGBD-FRESH-V2-DATA/scene_failures.json"),
    "EVAL_V3": Path("outputs/EXP-3D-RGBD-EVAL-V3-DATA/scene_failures.json"),
    "EVAL_V4": Path("outputs/EXP-3D-RGBD-EVAL-V4-DATA/scene_failures.json"),
}
SPLITS = ("train", "val")
TRAIN_EXT_MIN = 40
BUDGET = 2 * 1024**3
SELECTION_SEED = "TRAIN-EXT-20260930"
EXPOSURE_BASIS = (
    "TRAIN-EXT is training data only: official train or val scenes outside every evaluation "
    "cohort, DEV-volume disjoint and source-asset unique against every evaluation, protected and "
    "TRAIN72 scene; some were observed or read before (never as an evaluation scene). The "
    "protected official-test final holdout is not opened."
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def read(path):
    return json.loads(Path(path).read_text())


def eval_v3_module(repo):
    """The frozen EVAL-V3 preparation, reused as a library (never modified)."""
    spec = importlib.util.spec_from_file_location("rgbd_eval_v3_data", repo / EVAL_V3_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def source_paths(repo):
    paths = eval_v3_module(repo).source_paths(repo)
    paths["eval_v3_preparation"] = paths.pop("preparation_source")
    paths.update({f"{k.lower()}_manifest": repo / p for k, p in EVAL_MANIFESTS.items()})
    paths.update({f"{k.lower()}_failures": repo / p for k, p in FAILURES.items()})
    paths.update(plan=repo / PLAN, preparation_source=Path(__file__).resolve())
    return paths


def selection_key(scene):
    return hashlib.sha256(f"{SELECTION_SEED}:{scene}".encode()).hexdigest()


def select_candidates(
    partitions, assets, allowed, trajectories, excluded_sets, blocked_assets, fence
):
    """Metadata-only candidate order; `fence` holds the forbidden (DEV) volumes."""
    blocked = set(blocked_assets)
    selected = []
    for row in sorted(partitions, key=lambda r: selection_key(r["scene_name"])):
        sid = row["scene_name"]
        if (
            row["official_split"] not in SPLITS
            or row["protocol_partition"] != row["official_split"]
            or any(sid in s for s in excluded_sets)
            or sid not in assets
            or sid.split("_")[1] in fence
            or asset_identity(assets[sid]) in blocked
            or len(allowed.get(sid, [])) < 16
            or "BAD" in trajectories.get(sid + "_cam_00", {}).get("Scene type", "BAD").upper()
        ):
            continue
        selected.append(sid)
        blocked.add(asset_identity(assets[sid]))
    return selected


def train_ext_sets(repo, paths, partitions):
    """(excluded sets by reason, assets that block, DEV volume fence)."""
    current = read(paths["current_manifest"])
    v1_lock, fresh_v2_lock = read(paths["fresh_v1_lock"]), read(paths["fresh_v2_lock"])
    dev = {r["scene_id"] for r in current["scenes"] if r["split"] == "DEV"}
    train72 = {r["scene_id"] for r in current["scenes"] if r["split"] == "TRAIN"}
    if len(dev) != 8 or len(train72) != 72:
        raise ValueError("TRAIN72 + DEV manifest expected")
    fresh = set(v1_lock["FRESH_V1_candidates"]) | set(fresh_v2_lock["FRESH_V2_candidates"])
    evaluation = {
        k: {r["scene_id"] for r in read(repo / p)["scenes"]} for k, p in EVAL_MANIFESTS.items()
    }
    failed = set().union(*(set(read(repo / p)) for p in FAILURES.values()))
    protected = (
        set(current["protected_scene_ids"])
        | set(v1_lock["protected_scene_ids"])
        | set(fresh_v2_lock["protected_scene_ids"])
        | {
            r["scene_name"]
            for r in partitions
            if r["protocol_partition"] in ("final_holdout", "diagnostic_test")
        }
    )
    excluded = {next(iter(r.values())) for r in csv_rows(paths["exclusions"])}
    sets = {
        "protected": protected,
        "excluded": excluded,
        "dev": dev,
        "train72": train72,
        "fresh_candidates": fresh,
        "eval_v3": evaluation["EVAL_V3"],
        "eval_v4": evaluation["EVAL_V4"],
        "previously_invalid": failed,
    }
    blocking = dev | train72 | fresh | evaluation["EVAL_V3"] | evaluation["EVAL_V4"] | protected
    return sets, blocking, {s.split("_")[1] for s in dev}


def allowed_frames(v1, path):
    allowed = {}
    for split in SPLITS:
        for sid, ids in v1.allowed_frames(path, split).items():
            if sid in allowed:
                raise ValueError("A scene appears in two official splits")
            allowed[sid] = ids
    return allowed


def build_lock(repo):
    if sha(repo / PLAN) != PLAN_SHA256:
        raise PermissionError("The V14 plan changed or is missing")
    paths = source_paths(repo)
    ev3 = eval_v3_module(repo)
    v1 = ev3.v1_module(repo)
    assets = safe_source_assets(paths["assets"])
    partitions = csv_rows(paths["partitions"])
    trajectories = {r["Animation"]: r for r in csv_rows(paths["trajectories"])}
    sets, blocking, fence = train_ext_sets(repo, paths, partitions)
    blocked_assets = {asset_identity(assets[s]) for s in blocking if s in assets}
    allowed = allowed_frames(v1, paths["official_split"])
    candidates = select_candidates(
        partitions, assets, allowed, trajectories, sets.values(), blocked_assets, fence
    )
    observed = {r["scene_name"] for r in partitions if r["previously_observed"] == "True"}
    return {
        "schema": "mcss.rgbd_train_ext.cohort.v1",
        "experiment": EXPERIMENT,
        "TRAIN_EXT_candidates": candidates,
        "candidate_volumes": sorted({s.split("_")[1] for s in candidates}),
        "candidate_official_split": {
            r["scene_name"]: r["official_split"]
            for r in partitions
            if r["scene_name"] in candidates
        },
        "previously_observed": sorted(observed & set(candidates)),
        "minimum_valid": TRAIN_EXT_MIN,
        "fence_volumes": sorted(fence),
        "excluded_scene_ids": {k: sorted(v) for k, v in sets.items()},
        "protected_scene_ids": sorted(sets["protected"]),
        "assets": {s: assets[s] for s in candidates},
        "rule": (
            "official train or val split with the same protocol partition; not protected or "
            "excluded; not DEV, TRAIN72, a FRESH-V1/V2 candidate, EVAL-V3 or EVAL-V4; not "
            "invalid in an earlier preparation; volume disjoint from DEV; source asset unique vs "
            "every evaluation, protected and TRAIN72 scene and within the set; cam00 >=16 public "
            f"frames of its split; nonBAD trajectory; sha256('{SELECTION_SEED}:'+scene) order; "
            "every candidate passing the frozen model-free validity and camera-only role rules "
            "is kept (no volume cap)"
        ),
        "exposure_basis": EXPOSURE_BASIS,
        "plan_sha256": PLAN_SHA256,
        "source_sha256": {str(p): sha(p) for p in paths.values()},
        "data_budget_bytes": BUDGET,
        "model_forward_run": False,
        "protected_media_read": False,
    }


def prepare(repo, root):
    repo, root = Path(repo).resolve(), Path(root).resolve()
    lock_path = root / "candidate_lock.json"
    lock = read(lock_path)
    if lock != build_lock(repo):
        raise PermissionError("Candidate lock no longer reproduces from its frozen sources")
    if (root / "manifest_train_ext.json").exists():
        raise FileExistsError("Completed manifest_train_ext.json exists; no overwrite")
    ev3 = eval_v3_module(repo)
    v1 = ev3.v1_module(repo)
    paths = source_paths(repo)
    protected = set(lock["protected_scene_ids"])
    observed = set(lock["previously_observed"])
    calibration = published_scene_metadata(paths["calibration"])
    bounds = read(paths["bounds"])
    prior = FrozenTrainingPrior(bounds["near_m"], bounds["far_m"], bounds["source_sha256"])
    camera_only_roles = v1.role_rule(repo)
    allowed = allowed_frames(v1, paths["official_split"])
    network = [0]
    hashes, qualities, geometry, failures, supports = {}, {}, [], {}, {}
    records = []
    budget_stop = False
    with (root / "access.jsonl").open("a") as log:

        def access(sid, path, operation):
            if sid in protected:
                raise PermissionError("Protected scene access prohibited")
            entry = {
                "time": time.time(),
                "scene_id": sid,
                "path": str(path),
                "operation": operation,
                "purpose": "TRAIN_EXT_DATA_PREPARATION_NO_FORWARD",
            }
            log.write(json.dumps(entry) + "\n")
            log.flush()

        for sid in lock["TRAIN_EXT_candidates"]:
            if network[0] >= BUDGET:
                budget_stop = True
                break
            session = requests.Session()
            try:
                reader = BoundedRangeReader(URL.format(sid), session)
                reader.shared_budget = network
                with zipfile.ZipFile(reader) as archive:
                    try:
                        plan = select_members(sid, archive.infolist(), allowed[sid])
                    except ValueError as exc:
                        failures[sid] = {"status": "INELIGIBLE_MEMBERS", "error": str(exc)}
                        write(root / "scene_failures.json", failures)
                        print("INELIGIBLE", sid, "INELIGIBLE_MEMBERS", flush=True)
                        reader.close()
                        continue
                    plan.update(url=URL.format(sid), etag=reader.etag, split=SPLIT)
                    for member in plan["member_metadata"]:
                        name = member["name"]
                        dest = root / "raw" / name
                        access(sid, name, "download_locked_member")
                        if dest.exists():
                            content = dest.read_bytes()
                            if (
                                len(content) != member["bytes"]
                                or zlib.crc32(content) != member["crc32"]
                            ):
                                raise ValueError("Existing download failed CRC; stop")
                        else:
                            dest.parent.mkdir(parents=True, exist_ok=True)
                            dest.write_bytes(archive.read(name))
                        hashes[str(dest)] = sha(dest)
                reader.close()
            finally:
                session.close()
            raw = root / "raw" / sid
            units = {
                r["parameter_name"]: r["parameter_value"]
                for r in csv_rows(raw / "_detail/metadata_scene.csv")
            }
            scale = float(units["meters_per_asset_unit"])
            if not np.isclose(
                scale, float(calibration[sid]["settings_units_info_meters_scale"]), rtol=1e-6
            ):
                failures[sid] = {"status": "INELIGIBLE_CALIBRATION_SCALE"}
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
            frames, quality = [], {}
            prepared = root / "prepared" / sid
            prepared.mkdir(parents=True, exist_ok=True)
            for fid in plan["frame_ids"]:
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
                quality[fid]["eligible"] = quality[fid]["eligible"] and orth <= 0.001 and det > 0
                pose = convert_hypersim_pose(
                    rotation, positions[mapping[fid]], meters_per_asset_unit=scale
                )
                if fid == plan["frame_ids"][0]:
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
                        )
                    except ValueError as exc:
                        failures[sid] = {"status": "INELIGIBLE_SOURCE_GEOMETRY", "error": str(exc)}
                        geometry.append({"scene_id": sid, **failures[sid]})
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
                        "intrinsics": calibrated_intrinsics(calibration[sid], (128, 160)).tolist(),
                        "c2w": pose.tolist(),
                    }
                )
            qualities[sid] = quality
            write(root / "frame_quality.json", qualities)
            if sid in failures:
                print("INELIGIBLE", sid, failures[sid]["status"], flush=True)
                continue
            try:
                roles, support = camera_only_roles(sid, frames, quality, prior)
            except ValueError as exc:
                failures[sid] = {"status": "INELIGIBLE_FRAME_ROLES", "error": str(exc)}
                write(root / "scene_failures.json", failures)
                print("INELIGIBLE", sid, "INELIGIBLE_FRAME_ROLES", flush=True)
                continue
            records.append(
                {
                    "scene_id": sid,
                    "split": SPLIT,
                    "historically_exposed": sid in observed,
                    "historical_exposure_basis": lock["exposure_basis"],
                    "physical_scene_id": "|".join(asset_identity(lock["assets"][sid])),
                    "official_split": lock["candidate_official_split"][sid],
                    "roles": roles,
                    "frames": frames,
                }
            )
            supports[sid] = support
            write(root / "role_support.json", supports)
            write(root / "progress.json", {SPLIT: [r["scene_id"] for r in records]})
            print("PREPARED", sid, flush=True)
    write(root / "hashes.json", hashes)
    write(
        root / "download_cost.json",
        {"network_bytes": network[0], "full_archives_downloaded": False, "budget_bytes": BUDGET},
    )
    status = "PASS" if len(records) >= TRAIN_EXT_MIN else "BLOCKED_INSUFFICIENT_VALID_LOCKED_SCENES"
    write(
        root / "manifest_train_ext.json",
        {
            "schema": "mcss.rgbd_train_ext.data.v1",
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
            "TRAIN_EXT": len(records),
            "valid_volumes": sorted({r["scene_id"].split("_")[1] for r in records}),
            "budget_stop": budget_stop,
            "failures": failures,
            "candidate_replacement": (
                "none; every locked candidate is read in order until the budget"
            ),
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
        print(
            json.dumps(
                {
                    "TRAIN_EXT_candidates": len(lock["TRAIN_EXT_candidates"]),
                    "candidate_volumes": len(lock["candidate_volumes"]),
                    "previously_observed": len(lock["previously_observed"]),
                }
            )
        )
    else:
        prepare(args.repo, root)


if __name__ == "__main__":
    main()
