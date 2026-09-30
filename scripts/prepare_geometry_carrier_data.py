#!/usr/bin/env python3
"""Lock and materialize 24 TRAIN + 8 exposed DEV scenes, without model forward."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import time
import zipfile
import zlib
from pathlib import Path

import h5py
import numpy as np
import requests
import torch
from PIL import Image

from mcss.data.hypersim import convert_hypersim_pose
from mcss.mechanism_pilot.calibration import calibrated_intrinsics, published_scene_metadata
from mcss.mechanism_pilot.data_prep import BoundedRangeReader, check_geometry, resize_rgb_depth
from mcss.mechanism_pilot.optimization_bounds_contracts import (
    ContextCameraBundle,
    FrozenTrainingPrior,
    frozen_gt_free_bounds,
)
from mcss.mechanism_pilot.spatial import observed_camera_support
from mcss.mechanism_pilot.support_redesign_data import safe_source_assets
from mcss.mechanism_pilot.unseen_data import asset_identity, csv_rows, frame_quality, select_members
from mcss.types import Cameras

OLD_THREE = ("ai_001_001", "ai_002_001", "ai_003_001")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def select_cohort(old_ids, partition_rows, assets, allowed, trajectories, protected, count=12):
    """Metadata-only selection, prioritizing documented old100 membership."""
    if len(old_ids) != 20 or not set(OLD_THREE) <= set(old_ids):
        raise ValueError("Exactly twenty old exposed scenes including bounds-prior TRAIN3 required")
    if set(old_ids) & set(protected):
        raise PermissionError("Protected physical scene cannot become TRAIN or DEV")
    identities = {asset_identity(assets[s]) for s in old_ids}
    if len(identities) != len(old_ids):
        raise ValueError("Duplicated old physical asset")
    blocked_assets = identities | {asset_identity(assets[s]) for s in protected}
    volumes = {s.split("_")[1] for s in old_ids}
    selected = []
    for row in sorted(
        partition_rows, key=lambda r: (r["previously_observed"] != "True", r["scene_name"])
    ):
        sid = row["scene_name"]
        if (
            sid in protected
            or sid not in assets
            or row["protocol_partition"] != "train"
            or sid.split("_")[1] in volumes
            or asset_identity(assets[sid]) in blocked_assets
            or len(allowed.get(sid, [])) < 16
            or "BAD" in trajectories.get(sid + "_cam_00", {}).get("Scene type", "BAD").upper()
        ):
            continue
        selected.append(sid)
        volumes.add(sid.split("_")[1])
        blocked_assets.add(asset_identity(assets[sid]))
        if len(selected) == count:
            break
    if len(selected) != count:
        raise ValueError("Insufficient metadata-eligible candidates; no replacement search")
    dev = sorted(
        (s for s in old_ids if s not in OLD_THREE),
        key=lambda s: hashlib.sha256(f"20260928:{s}".encode()).hexdigest(),
    )[:8]
    train = sorted(set(old_ids) - set(dev)) + selected
    if {s.split("_")[1] for s in train} & {s.split("_")[1] for s in dev}:
        raise ValueError("TRAIN/DEV volume overlap")
    return train, dev, selected


def camera_only_roles(scene_id, frames, quality, prior):
    """First lexicographically feasible disjoint pairs; no score/depth input to support."""
    valid = sorted(f["frame_id"] for f in frames if quality[f["frame_id"]]["eligible"])
    if len(valid) < 7:
        raise ValueError("Fewer than seven valid frames in frozen first16; no substitutions")
    anchor, queries = valid[0], valid[-2:]
    rows = {f["frame_id"]: f for f in frames}
    ids = torch.div(torch.arange(128) * (16**3 - 1), 127, rounding_mode="floor")
    fractions = torch.stack(
        ((ids % 16 + 0.5) / 16, (ids // 16 % 16 + 0.5) / 16, (ids // 256 + 0.5) / 16), -1
    ).double()
    options = []
    for pair in itertools.combinations([i for i in valid if i != anchor and i not in queries], 2):
        frame_ids = (anchor, *pair)
        cameras = tuple(
            Cameras(
                torch.tensor(rows[i]["intrinsics"], dtype=torch.float64),
                torch.tensor(rows[i]["c2w"], dtype=torch.float64),
                (128, 160),
            )
            for i in frame_ids
        )
        bundle = ContextCameraBundle(scene_id, "A", frame_ids, cameras, cameras[0].c2w.clone())
        bounds = frozen_gt_free_bounds(bundle, prior)
        points = bounds[0] + fractions * (bounds[1] - bounds[0])
        support = observed_camera_support(points, cameras)["supported_candidates"]
        if support >= 2:
            options.append((pair, support))
    for a, b in itertools.combinations(options, 2):
        if set(a[0]) & set(b[0]):
            continue
        return {
            "context_a": [anchor, *a[0]],
            "context_b": [anchor, *b[0]],
            "primary_query": queries,
            "query": queries,
            "secondary_query": [],
            "stream": [],
            "anchor": anchor,
        }, {
            "status": "ELIGIBLE",
            "A_supported_candidates": a[1],
            "B_supported_candidates": b[1],
            "meaning": "Two-frustum overlap only; not occlusion/true surface visibility",
            "grid": 16,
            "tokens": 128,
        }
    raise ValueError("No two disjoint context pairs with >=2 camera-supported candidates")


def source_paths(repo):
    static = repo / "outputs/EXP-3D-STATIC-CARRIER-ATTRIBUTION-V1"
    return {
        "assets": static / "metadata/upstream_dataset_config.py",
        "partitions": repo / "configs/hypersim_er_partitions.csv",
        "old100": repo / "configs/hypersim_subset_v1.csv",
        "official_split": static / "rgb_audit/official_metadata/metadata_images_split_scene_v1.csv",
        "trajectories": static / "rgb_audit/official_metadata/metadata_camera_trajectories.csv",
        "old17": repo / "outputs/EXP-3D-DIRECT-STATE-CAPACITY-V1/scene_manifest.json",
        "old3": repo / "outputs/EXP-3D-20260927-small-training-v1/data/manifest.json",
        "b_final_lock": repo
        / "outputs/EXP-3D-20260927-centered-training-1000a-600b-v1/training/lock.json",
        "historical_audit": repo
        / "outputs/EXP-3D-SUPPORT-BOTTLENECK-REDESIGN-V1/data/historical_exposure_audit.json",
        "calibration": repo
        / "outputs/EXP-3D-20260927-training-preparation-v1/metadata/metadata_camera_parameters.csv",
        "bounds": repo / "outputs/EXP-3D-16G-OPTIMIZATION-BOUNDS-V1/bounds_contract.json",
        "preparation_source": Path(__file__).resolve(),
    }


def prepare(repo, output, lock_only=False, source_validity_revision=False):
    repo, output = Path(repo).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    paths = source_paths(repo)
    old17 = json.loads(paths["old17"].read_text())
    old3 = json.loads(paths["old3"].read_text())
    old_records = old3["scenes"] + old17["scenes"]
    assets, partitions = safe_source_assets(paths["assets"]), csv_rows(paths["partitions"])
    protected = sorted(
        set(old17["FINAL_HOLDOUT_PROHIBITED"])
        | {
            r["scene_name"]
            for r in partitions
            if r["protocol_partition"] in ("final_holdout", "diagnostic_test")
        }
    )
    allowed = {}
    for r in csv_rows(paths["official_split"]):
        if (
            r["camera_name"] == "cam_00"
            and r["included_in_public_release"] == "True"
            and r["split_partition_name"] == "train"
        ):
            allowed.setdefault(r["scene_name"], []).append(int(r["frame_id"]))
    trajectories = {r["Animation"]: r for r in csv_rows(paths["trajectories"])}
    train, dev, new = select_cohort(
        [s["scene_id"] for s in old_records],
        partitions,
        assets,
        allowed,
        trajectories,
        protected,
        count=26 if source_validity_revision else 12,
    )
    prior_data = json.loads(paths["bounds"].read_text())
    prior = FrozenTrainingPrior(
        prior_data["near_m"], prior_data["far_m"], prior_data["source_sha256"]
    )
    lock = {
        "schema": "mcss.geometry_carrier.cohort.v1",
        "TRAIN": train,
        "DEV": dev,
        "new_candidates": new,
        "FRESH_QUALIFICATION": [],
        "FRESH_QUALIFICATION_STATUS": "BLOCKED_INDEPENDENCE_UNRESOLVED",
        "protected_scene_ids": protected,
        "assets": {s: assets[s] for s in train + dev},
        "previously_observed": {
            r["scene_name"]: r["previously_observed"]
            for r in partitions
            if r["scene_name"] in train + dev
        },
        "source_sha256": {str(p): sha(p) for p in paths.values()},
        "selection_seed": 20260928,
        "dev_rule": "sha256('20260928:'+scene_id), lowest8 among exposed17; old3 TRAIN",
        "selection_rule": (
            "historical100 first, then lexicographic; TRAIN cam00 >=16 publicly included"
            " frames, nonBAD; new volume unique vs old20 and new peers; protected "
            "scene/asset blocked; no replacement"
        ),
        "new_role_rule": (
            "first valid anchor, last2 valid query; earliest feasible disjoint context "
            "pairs with >=2/128 frustum support; grid16 GTfree prior bounds; no "
            "score/depth geometry selection"
        ),
        "old_role_erratum": (
            "Old20 context roles unchanged. Old3 primary_query=[12,13], legacy "
            "query=[12,13,14,15] retained;14/15 unused. Old17 primary_query unchanged. "
            "No old file overwritten."
        ),
        "quality_rule": (
            "first16 official included train frames; finite RGB, nearblack<=.999, "
            "positivefinite native depth>=.95; old20 retained unchanged"
        ),
        "historical_independence_limit": (
            "Old365 scene-level execution logs unrecovered; absence from old100 is not "
            "virginity. All selected scenes exposed or historical-training-pool, not "
            "fresh. Official source archive/asset defines physical identity proxy; "
            "cross-asset vendor reuse cannot be excluded."
        ),
        "data_budget_bytes": 2 * 1024**3,
        "model_forward_run": False,
        "protected_media_read": False,
    }
    if source_validity_revision:
        previous = output / "candidate_lock_v2.json"
        failed_path = output / "scene_failures_v2.json"
        failed = json.loads(failed_path.read_text())
        lock["schema"] = "mcss.geometry_carrier.cohort.v3"
        lock["source_validity_amendment"] = {
            "previous_lock_sha256": sha(previous),
            "original_candidates": new[:12],
            "excluded_original": sorted(failed),
            "excluded_evidence_sha256": sha(failed_path),
            "supplement_candidates_locked": new[12:],
            "supplement_budget": 14,
            "selection": (
                "Retain original5 valid; first7 source/camera-valid supplementary scenes "
                "in locked order. No model scores. Stop at 7, no more than14 candidates."
            ),
            "camera_validity": (
                "For all used context/query frames: max(abs(R.T@R-I))<=0.001 and "
                "determinant>0; never orthonormalize or alter renderer."
            ),
            "original_geometry_p95_m": 0.01,
            "TRAIN_field_is_candidate_pool_not_final_split": True,
        }
    lockpath = output / (
        "candidate_lock_v3.json" if source_validity_revision else "candidate_lock.json"
    )
    if lockpath.exists():
        if json.loads(lockpath.read_text()) != lock:
            raise ValueError(
                "Existing candidate lock differs; preserve prior attempt, refuse overwrite"
            )
    else:
        write(lockpath, lock)
    if lock_only:
        return lock
    if (output / "manifest.json").exists():
        raise FileExistsError("Completed manifest exists; no overwrite")
    network = [0]
    plans, sessions, readers, archives = [], [], [], []
    try:
        for sid in new:
            session = requests.Session()
            sessions.append(session)
            url = f"https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/{sid}.zip"
            reader = BoundedRangeReader(url, session)
            reader.shared_budget = network
            readers.append(reader)
            archive = zipfile.ZipFile(reader)
            archives.append(archive)
            plan = select_members(sid, archive.infolist(), allowed[sid])
            plan.update(url=url, etag=reader.etag, archive_bytes=reader.size)
            plans.append(plan)
            print("MEMBER_PLAN", sid, flush=True)
        if (
            sum(p["compressed_bytes"] + p["uncompressed_bytes"] for p in plans)
            > lock["data_budget_bytes"]
        ):
            raise ValueError("Selected compressed+uncompressed data exceeds frozen budget")
        dataplan = {"candidate_lock_sha256": sha(lockpath), "scenes": plans}
        dp = output / ("data_lock_v3.json" if source_validity_revision else "data_lock.json")
        if dp.exists() and json.loads(dp.read_text()) != dataplan:
            raise ValueError("Remote metadata changed; refuse resumptive data modification")
        write(dp, dataplan)
        calibration = published_scene_metadata(paths["calibration"])
        records, hashes, qualities, geometry, feasibility = [], {}, {}, [], {}
        scene_failures = {}
        with (output / "access.jsonl").open("a") as log:

            def access(sid, path, operation):
                if sid in protected:
                    raise PermissionError("Protected scene access prohibited")
                log.write(
                    json.dumps(
                        {
                            "time": time.time(),
                            "scene_id": sid,
                            "path": str(path),
                            "operation": operation,
                            "purpose": "TRAIN_DEV_DATA_PREPARATION_NO_FORWARD",
                        }
                    )
                    + "\n"
                )
                log.flush()

            for record in old_records:
                sid = record["scene_id"]
                item = json.loads(json.dumps(record))
                item.update(
                    split="DEV" if sid in dev else "TRAIN",
                    historically_exposed=True,
                    physical_scene_id="|".join(asset_identity(assets[sid])),
                )
                item["roles"]["primary_query"] = (
                    [12, 13] if sid in OLD_THREE else item["roles"]["primary_query"]
                )
                item["roles"]["anchor"] = item["roles"]["context_a"][0]
                for frame in item["frames"]:
                    for kind in ("rgb", "depth"):
                        path = Path(frame[kind]).resolve()
                        access(sid, path, "hash_existing_prepared_" + kind)
                        hashes[str(path)] = sha(path)
                        frame[kind] = str(path)
                records.append(item)
            for plan, archive in zip(plans, archives, strict=True):
                sid = plan["scene_id"]
                if source_validity_revision and sid in failed:
                    scene_failures[sid] = failed[sid]
                    continue
                if source_validity_revision and len(records) == 32:
                    break
                for member in plan["member_metadata"]:
                    name = member["name"]
                    dest = output / "raw" / name
                    access(sid, name, "read_or_download_locked_member")
                    if dest.exists():
                        content = dest.read_bytes()
                        if (
                            len(content) != member["bytes"]
                            or zlib.crc32(content) != member["crc32"]
                        ):
                            raise ValueError("Existing download failed CRC; preserve and stop")
                    else:
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        content = archive.read(name)
                        dest.write_bytes(content)
                    hashes[str(dest)] = sha(dest)
                raw = output / "raw" / sid
                units = {
                    r["parameter_name"]: r["parameter_value"]
                    for r in csv_rows(raw / "_detail/metadata_scene.csv")
                }
                scale = float(units["meters_per_asset_unit"])
                if not np.isclose(
                    scale, float(calibration[sid]["settings_units_info_meters_scale"]), rtol=1e-6
                ):
                    raise ValueError("Calibration scale mismatch")

                def hdf(path, sid=sid):
                    access(sid, path, "decode_hdf5")
                    with h5py.File(path) as handle:
                        return np.array(handle["dataset"])

                ids = hdf(raw / "_detail/cam_00/camera_keyframe_frame_indices.hdf5").reshape(-1)
                poses = hdf(raw / "_detail/cam_00/camera_keyframe_positions.hdf5")
                rots = hdf(raw / "_detail/cam_00/camera_keyframe_orientations.hdf5")
                mapping = {int(fid): j for j, fid in enumerate(ids)}
                frames, quality = [], {}
                prepared = output / "prepared" / sid
                prepared.mkdir(parents=True, exist_ok=True)
                for fid in plan["frame_ids"]:
                    rgbpath = raw / f"images/scene_cam_00_final_preview/frame.{fid:04d}.color.jpg"
                    access(sid, rgbpath, "decode_RGB")
                    with Image.open(rgbpath) as im:
                        rgb = np.asarray(im.convert("RGB"))
                    depth = hdf(
                        raw / f"images/scene_cam_00_geometry_hdf5/frame.{fid:04d}.depth_meters.hdf5"
                    )
                    if rgb.shape[:2] != depth.shape:
                        raise ValueError("RGB/depth dimensions differ")
                    quality[fid] = frame_quality(rgb, depth)
                    if source_validity_revision:
                        rotation = rots[mapping[fid]]
                        orth_error = float(np.max(np.abs(rotation.T @ rotation - np.eye(3))))
                        determinant = float(np.linalg.det(rotation))
                        quality[fid]["camera_orthogonality_error"] = orth_error
                        quality[fid]["camera_determinant"] = determinant
                        quality[fid]["eligible"] = (
                            quality[fid]["eligible"] and orth_error <= 0.001 and determinant > 0
                        )
                    pose = convert_hypersim_pose(
                        rots[mapping[fid]], poses[mapping[fid]], meters_per_asset_unit=scale
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
                            scene_failures[sid] = {
                                "status": "INELIGIBLE_SOURCE_GEOMETRY",
                                "error": str(exc),
                            }
                            geometry.append({"scene_id": sid, **scene_failures[sid]})
                        write(output / "geometry_checks.json", geometry)
                        write(output / "scene_failures.json", scene_failures)
                    rgb_small, depth_small = resize_rgb_depth(rgb, depth)
                    rgbout, depthout = prepared / f"{fid:04d}.png", prepared / f"{fid:04d}.npy"
                    Image.fromarray(rgb_small).save(rgbout)
                    np.save(depthout, depth_small)
                    hashes[str(rgbout)], hashes[str(depthout)] = sha(rgbout), sha(depthout)
                    frames.append(
                        {
                            "frame_id": fid,
                            "rgb": str(rgbout),
                            "depth": str(depthout),
                            "intrinsics": calibrated_intrinsics(
                                calibration[sid], (128, 160)
                            ).tolist(),
                            "c2w": pose.tolist(),
                        }
                    )
                qualities[sid] = quality
                write(output / "frame_quality.json", qualities)
                if sid in scene_failures:
                    print("INELIGIBLE", sid, scene_failures[sid], flush=True)
                    continue
                try:
                    roles, diagnostic = camera_only_roles(sid, frames, quality, prior)
                except ValueError as exc:
                    scene_failures[sid] = {
                        "status": "INELIGIBLE_FRAME_ROLES",
                        "error": str(exc),
                    }
                    write(output / "scene_failures.json", scene_failures)
                    print("INELIGIBLE", sid, scene_failures[sid], flush=True)
                    continue
                feasibility[sid] = diagnostic
                records.append(
                    {
                        "scene_id": sid,
                        "split": "TRAIN",
                        "historically_exposed": True,
                        "historical_exposure_basis": (
                            "Old100 pool membership or historical365 train pool; precise "
                            "executed model "
                            "exposure unresolved"
                        ),
                        "physical_scene_id": "|".join(asset_identity(assets[sid])),
                        "roles": roles,
                        "frames": frames,
                    }
                )
                write(
                    output / "progress.json",
                    {
                        "completed_scene_ids": [s["scene_id"] for s in records],
                        "support": feasibility,
                    },
                )
                print("PREPARED", sid, flush=True)
        manifest = {
            "schema": "mcss.geometry_carrier.data.v1",
            "image_size": [128, 160],
            "depth_semantics": "ray_distance_meters",
            "scenes": records,
            "candidate_lock_sha256": sha(lockpath),
            "candidate_lock_path": str(lockpath),
            "FRESH_QUALIFICATION_STATUS": "BLOCKED_INDEPENDENCE_UNRESOLVED",
            "fresh_status": "BLOCKED_INDEPENDENCE_UNRESOLVED",
            "protected_scene_ids": protected,
        }
        if len(records) != 32 or sum(s["split"] == "TRAIN" for s in records) != 24:
            manifest["status"] = "BLOCKED_INSUFFICIENT_VALID_LOCKED_SCENES"
            manifest["scene_failures"] = scene_failures
            write(output / "diagnostic_manifest.json", manifest)
            write(output / "hashes.json", hashes)
            write(output / "support_feasibility.json", feasibility)
            write(
                output / "preparation_integrity.json",
                {
                    "status": "BLOCKED_INSUFFICIENT_VALID_LOCKED_SCENES",
                    "TRAIN": sum(s["split"] == "TRAIN" for s in records),
                    "DEV": 8,
                    "fresh": 0,
                    "protected_media_read": False,
                    "model_forward_run": False,
                    "failures": scene_failures,
                    "candidate_replacement": False,
                },
            )
            return manifest
        write(output / "manifest.json", manifest)
        write(output / "hashes.json", hashes)
        write(output / "geometry_checks.json", geometry)
        write(output / "support_feasibility.json", feasibility)
        write(output / "frame_role_lock.json", {s["scene_id"]: s["roles"] for s in records})
        write(
            output / "preparation_integrity.json",
            {
                "status": "PASS",
                "TRAIN": 24,
                "DEV": 8,
                "fresh": 0,
                "protected_media_read": False,
                "model_forward_run": False,
                "manifest_sha256": sha(output / "manifest.json"),
                "candidate_lock_sha256": sha(lockpath),
            },
        )
        return manifest
    except Exception as exc:
        write(
            output / "failure.json",
            {
                "status": "BLOCKED_DATA_PREPARATION",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "candidate_replacement": False,
                "completed_data_preserved": True,
            },
        )
        raise
    finally:
        write(
            output / "download_cost.json",
            {
                "network_bytes_this_attempt": network[0],
                "full_archives_downloaded": False,
                "budget_bytes": 2 * 1024**3,
            },
        )
        for archive in archives:
            archive.close()
        for reader in readers:
            reader.close()
        for session in sessions:
            session.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lock-only", action="store_true")
    parser.add_argument("--source-validity-revision", action="store_true")
    args = parser.parse_args()
    prepare(args.repo, args.output, args.lock_only, args.source_validity_revision)
