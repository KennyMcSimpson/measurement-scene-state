#!/usr/bin/env python3
"""Lock and materialize FRESH-V1 (independent qualification) and TRAIN-X (scale) scenes.

`--stage lock` is metadata-only and freezes the ordered candidate lists before any media byte
is read; `--stage prepare` downloads only locked members, applies the frozen V2 validity rules
and the frozen camera-only V2 frame-role rule, and never runs a model or looks at a score.

FRESH-V1: official Hypersim val split (outside the historical365 train pool), never observed
(configs/hypersim_er_partitions.csv), not protected or excluded, source asset unique, one scene
per volume, volume disjoint from every V2-V5 TRAIN/DEV scene.  TRAIN-X: official train split,
not protected or excluded, not already used, source asset unique, volume disjoint from DEV and
FRESH-V1, at most two new scenes per volume, deterministic hash order.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import time
import zipfile
import zlib
from collections import Counter
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

EXPERIMENT = "EXP-3D-RGBD-FRESH-SCALE-DATA-V1"
URL = "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/{}.zip"
CURRENT_MANIFEST = Path("outputs/EXP-3D-RGBD-EVIDENCE-CARRIER-V5/scene_split.json")
TRAIN_X_LOCKED, TRAIN_X_TARGET, TRAIN_X_PER_VOLUME, FRESH_MIN = 80, 48, 2, 6
BUDGET = 2 * 1024**3
SELECTION_SEED = "20260929"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def volume(scene):
    return scene.split("_")[1]


def bad_trajectory(trajectories, scene):
    return "BAD" in trajectories.get(scene + "_cam_00", {}).get("Scene type", "BAD").upper()


def select_fresh(partitions, assets, allowed, trajectories, protected, excluded, current):
    """Metadata-only FRESH-V1 rule in lexicographic order; no replacement search."""
    used_volumes = {volume(s) for s in current}
    blocked = {asset_identity(assets[s]) for s in current | protected if s in assets}
    selected, volumes = [], set()
    for row in sorted(partitions, key=lambda r: r["scene_name"]):
        sid = row["scene_name"]
        if (
            row["official_split"] != "val"
            or row["protocol_partition"] != "val"
            or row["previously_observed"] != "False"
            or sid in protected
            or sid in excluded
            or sid in current
            or sid not in assets
            or volume(sid) in used_volumes | volumes
            or asset_identity(assets[sid]) in blocked
            or len(allowed.get(sid, [])) < 16
            or bad_trajectory(trajectories, sid)
        ):
            continue
        selected.append(sid)
        volumes.add(volume(sid))
        blocked.add(asset_identity(assets[sid]))
    return selected


def select_train_x(
    partitions, assets, allowed, trajectories, protected, excluded, current, dev, fresh
):
    """Metadata-only TRAIN-X rule in sha256(seed:scene) order, capped per volume."""
    forbidden_volumes = {volume(s) for s in dev} | {volume(s) for s in fresh}
    blocked = {asset_identity(assets[s]) for s in current | protected | set(fresh) if s in assets}
    per_volume, selected = Counter(), []
    rows = sorted(
        partitions,
        key=lambda r: hashlib.sha256(f"{SELECTION_SEED}:{r['scene_name']}".encode()).hexdigest(),
    )
    for row in rows:
        sid = row["scene_name"]
        if (
            row["official_split"] != "train"
            or row["protocol_partition"] != "train"
            or sid in protected
            or sid in excluded
            or sid in current
            or sid not in assets
            or volume(sid) in forbidden_volumes
            or per_volume[volume(sid)] >= TRAIN_X_PER_VOLUME
            or asset_identity(assets[sid]) in blocked
            or len(allowed.get(sid, [])) < 16
            or bad_trajectory(trajectories, sid)
        ):
            continue
        selected.append(sid)
        per_volume[volume(sid)] += 1
        blocked.add(asset_identity(assets[sid]))
        if len(selected) == TRAIN_X_LOCKED:
            break
    return selected


def source_paths(repo):
    static = repo / "outputs/EXP-3D-STATIC-CARRIER-ATTRIBUTION-V1"
    return {
        "assets": static / "metadata/upstream_dataset_config.py",
        "partitions": repo / "configs/hypersim_er_partitions.csv",
        "exclusions": repo / "configs/hypersim_er_final_exclusions.csv",
        "official_split": static / "rgb_audit/official_metadata/metadata_images_split_scene_v1.csv",
        "trajectories": static / "rgb_audit/official_metadata/metadata_camera_trajectories.csv",
        "calibration": repo
        / "outputs/EXP-3D-20260927-training-preparation-v1/metadata/metadata_camera_parameters.csv",
        "bounds": repo / "outputs/EXP-3D-16G-OPTIMIZATION-BOUNDS-V1/bounds_contract.json",
        "current_manifest": repo / CURRENT_MANIFEST,
        "v2_role_rule": repo / "scripts/prepare_geometry_carrier_data.py",
        "preparation_source": Path(__file__).resolve(),
    }


def allowed_frames(path, split):
    allowed = {}
    for r in csv_rows(path):
        if (
            r["camera_name"] == "cam_00"
            and r["included_in_public_release"] == "True"
            and r["split_partition_name"] == split
        ):
            allowed.setdefault(r["scene_name"], []).append(int(r["frame_id"]))
    return allowed


def build_lock(repo):
    paths = source_paths(repo)
    assets = safe_source_assets(paths["assets"])
    partitions = csv_rows(paths["partitions"])
    trajectories = {r["Animation"]: r for r in csv_rows(paths["trajectories"])}
    current_manifest = json.loads(paths["current_manifest"].read_text())
    current = {r["scene_id"] for r in current_manifest["scenes"]}
    dev = {r["scene_id"] for r in current_manifest["scenes"] if r["split"] == "DEV"}
    excluded = {next(iter(r.values())) for r in csv_rows(paths["exclusions"])}
    protected = set(current_manifest["protected_scene_ids"]) | {
        r["scene_name"]
        for r in partitions
        if r["protocol_partition"] in ("final_holdout", "diagnostic_test")
    }
    allowed_val = allowed_frames(paths["official_split"], "val")
    allowed_train = allowed_frames(paths["official_split"], "train")
    fresh = select_fresh(
        partitions, assets, allowed_val, trajectories, protected, excluded, current
    )
    train_x = select_train_x(
        partitions, assets, allowed_train, trajectories, protected, excluded, current, dev, fresh
    )
    if len(fresh) < FRESH_MIN or len(train_x) < TRAIN_X_TARGET:
        raise ValueError(f"Insufficient metadata candidates: fresh {len(fresh)}, x {len(train_x)}")
    return {
        "schema": "mcss.rgbd_fresh_scale.cohort.v1",
        "experiment": EXPERIMENT,
        "FRESH_V1_candidates": fresh,
        "TRAIN_X_candidates": train_x,
        "TRAIN_X_target": TRAIN_X_TARGET,
        "current_scene_ids": sorted(current),
        "current_dev_ids": sorted(dev),
        "protected_scene_ids": sorted(protected),
        "excluded_scene_ids": sorted(excluded),
        "assets": {s: assets[s] for s in fresh + train_x},
        "fresh_rule": (
            "official val split & protocol val & previously_observed False; not protected/"
            "excluded/current; source asset unique vs current+protected; one scene per volume; "
            "volume disjoint from all 32 V2-V5 TRAIN/DEV scenes; cam00 >=16 public val frames; "
            "nonBAD trajectory; lexicographic; every valid candidate kept, no replacement"
        ),
        "train_x_rule": (
            "official train split & protocol train; not protected/excluded/current; source asset "
            "unique vs current+protected+fresh; volume disjoint from DEV and FRESH-V1; <=2 new "
            f"scenes per volume; sha256('{SELECTION_SEED}:'+scene) order; first "
            f"{TRAIN_X_LOCKED} locked; the first {TRAIN_X_TARGET} valid are kept, no replacement"
        ),
        "validity_rule": (
            "first16 officially included frames of the scene's split; finite RGB, nearblack "
            "<=.999, positive finite native depth >=.95; camera max|R^T R - I|<=.001 and det>0; "
            "native geometry p95 <= .01 m (position.hdf5, first frame)"
        ),
        "role_rule": "frozen V2 camera_only_roles (prepare_geometry_carrier_data.py)",
        "independence_basis": (
            "FRESH-V1 is outside the historical365 train pool (official val split), marked "
            "never observed, and volume-disjoint from TRAIN/DEV; vendor-level cross-asset "
            "reuse inside Hypersim cannot be excluded. The protected official-test final "
            "holdout is not opened."
        ),
        "source_sha256": {str(p): sha(p) for p in paths.values()},
        "data_budget_bytes": BUDGET,
        "model_forward_run": False,
        "protected_media_read": False,
    }


def role_rule(repo):
    spec = importlib.util.spec_from_file_location(
        "v2_data_rules", repo / "scripts/prepare_geometry_carrier_data.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.camera_only_roles


def prepare(repo, root):
    repo, root = Path(repo).resolve(), Path(root).resolve()
    lock_path = root / "candidate_lock.json"
    lock = json.loads(lock_path.read_text())
    if lock != build_lock(repo):
        raise PermissionError("Candidate lock no longer reproduces from its frozen sources")
    for name in ("manifest_fresh.json", "manifest_train_x.json"):
        if (root / name).exists():
            raise FileExistsError(f"Completed {name} exists; no overwrite")
    paths = source_paths(repo)
    protected = set(lock["protected_scene_ids"])
    calibration = published_scene_metadata(paths["calibration"])
    bounds = json.loads(paths["bounds"].read_text())
    prior = FrozenTrainingPrior(bounds["near_m"], bounds["far_m"], bounds["source_sha256"])
    camera_only_roles = role_rule(repo)
    allowed = {
        "FRESH_QUALIFICATION": allowed_frames(paths["official_split"], "val"),
        "TRAIN": allowed_frames(paths["official_split"], "train"),
    }
    network = [0]
    hashes, qualities, geometry, failures, supports = {}, {}, [], {}, {}
    records = {"FRESH_QUALIFICATION": [], "TRAIN": []}
    plans = []
    queue = [("FRESH_QUALIFICATION", s) for s in lock["FRESH_V1_candidates"]]
    queue += [("TRAIN", s) for s in lock["TRAIN_X_candidates"]]
    with (root / "access.jsonl").open("a") as log:

        def access(sid, split, path, operation):
            if sid in protected:
                raise PermissionError("Protected scene access prohibited")
            purpose = (
                "FRESH_QUALIFICATION_DATA_PREPARATION_NO_FORWARD"
                if split == "FRESH_QUALIFICATION"
                else "TRAIN_SCALE_DATA_PREPARATION_NO_FORWARD"
            )
            log.write(
                json.dumps(
                    {
                        "time": time.time(),
                        "scene_id": sid,
                        "path": str(path),
                        "operation": operation,
                        "purpose": purpose,
                    }
                )
                + "\n"
            )
            log.flush()

        for split, sid in queue:
            if split == "TRAIN" and len(records["TRAIN"]) == lock["TRAIN_X_target"]:
                break
            session = requests.Session()
            try:
                reader = BoundedRangeReader(URL.format(sid), session)
                reader.shared_budget = network
                with zipfile.ZipFile(reader) as archive:
                    try:
                        plan = select_members(sid, archive.infolist(), allowed[split][sid])
                    except ValueError as exc:
                        failures[sid] = {"status": "INELIGIBLE_MEMBERS", "error": str(exc)}
                        write(root / "scene_failures.json", failures)
                        print("INELIGIBLE", split, sid, "INELIGIBLE_MEMBERS", flush=True)
                        reader.close()
                        continue
                    plan.update(url=URL.format(sid), etag=reader.etag, split=split)
                    plans.append(plan)
                    write(root / "data_plan.json", plans)
                    for member in plan["member_metadata"]:
                        name = member["name"]
                        dest = root / "raw" / name
                        access(sid, split, name, "download_locked_member")
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

            def hdf(path, sid=sid, split=split):
                access(sid, split, path, "decode_hdf5")
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
                access(sid, split, rgb_path, "decode_RGB")
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
                print("INELIGIBLE", split, sid, failures[sid]["status"], flush=True)
                continue
            try:
                roles, support = camera_only_roles(sid, frames, quality, prior)
            except ValueError as exc:
                failures[sid] = {"status": "INELIGIBLE_FRAME_ROLES", "error": str(exc)}
                write(root / "scene_failures.json", failures)
                print("INELIGIBLE", split, sid, "INELIGIBLE_FRAME_ROLES", flush=True)
                continue
            records[split].append(
                {
                    "scene_id": sid,
                    "split": split,
                    "historically_exposed": split != "FRESH_QUALIFICATION",
                    "historical_exposure_basis": lock["independence_basis"]
                    if split == "FRESH_QUALIFICATION"
                    else "historical365 train pool; executed exposure unresolved",
                    "physical_scene_id": "|".join(asset_identity(lock["assets"][sid])),
                    "roles": roles,
                    "frames": frames,
                }
            )
            supports[sid] = support
            write(root / "role_support.json", supports)
            write(
                root / "progress.json",
                {k: [r["scene_id"] for r in v] for k, v in records.items()},
            )
            print("PREPARED", split, sid, flush=True)
    write(root / "hashes.json", hashes)
    write(
        root / "download_cost.json",
        {"network_bytes": network[0], "full_archives_downloaded": False, "budget_bytes": BUDGET},
    )
    status = (
        "PASS"
        if len(records["FRESH_QUALIFICATION"]) >= FRESH_MIN
        and len(records["TRAIN"]) == lock["TRAIN_X_target"]
        else "BLOCKED_INSUFFICIENT_VALID_LOCKED_SCENES"
    )
    for split, name in (
        ("FRESH_QUALIFICATION", "manifest_fresh.json"),
        ("TRAIN", "manifest_train_x.json"),
    ):
        write(
            root / name,
            {
                "schema": "mcss.rgbd_fresh_scale.data.v1",
                "image_size": [128, 160],
                "depth_semantics": "ray_distance_meters",
                "scenes": records[split],
                "candidate_lock_sha256": sha(lock_path),
                "protected_scene_ids": lock["protected_scene_ids"],
            },
        )
    write(
        root / "preparation_integrity.json",
        {
            "status": status,
            "FRESH_QUALIFICATION": len(records["FRESH_QUALIFICATION"]),
            "TRAIN_X": len(records["TRAIN"]),
            "failures": failures,
            "candidate_replacement": False,
            "protected_media_read": False,
            "model_forward_run": False,
            "network_bytes": network[0],
            "candidate_lock_sha256": sha(lock_path),
        },
    )
    print(status, {k: len(v) for k, v in records.items()}, "failures", len(failures), flush=True)


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
                    "FRESH_V1_candidates": lock["FRESH_V1_candidates"],
                    "TRAIN_X_candidates": len(lock["TRAIN_X_candidates"]),
                    "TRAIN_X_volumes": len({volume(s) for s in lock["TRAIN_X_candidates"]}),
                }
            )
        )
    else:
        prepare(args.repo, root)


if __name__ == "__main__":
    main()
