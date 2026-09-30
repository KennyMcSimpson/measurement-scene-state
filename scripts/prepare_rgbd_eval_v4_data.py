#!/usr/bin/env python3
"""Lock and materialize EVAL-V4, a second scene-disjoint evaluation cohort (no model, no score).

`--stage lock` is metadata-only and freezes the ordered candidate list before any media byte is
read; `--stage prepare` downloads locked candidates in order, applies the frozen V2 validity
rules and the frozen camera-only V2 frame-role rule, and never runs a model or looks at a score.

EVAL-V4: the EVAL-V3 eligibility rule with every EVAL-V3 candidate that was read (its 57 valid
and 21 invalid scenes) added to the used set, so only never-read scenes remain: official Hypersim
train split, never observed, not protected or excluded, never used or considered by an earlier
experiment (TRAIN72, DEV, FRESH-V1/V2, TRAIN-X, read EVAL-V3 candidates), source asset unique
against all of them and within the cohort, volume disjoint from the DEV and FRESH scenes, in a
new sha256(seed:scene) order. Like EVAL-V3 it may share volumes, never scenes or source assets,
with TRAIN72 and with EVAL-V3. Per volume the first two candidates that pass the model-free
validity rules are kept; later candidates of a full volume are never read.
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

EXPERIMENT = "EXP-3D-RGBD-EVAL-V4-DATA"
SPLIT = "EVAL_V4"
URL = "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/{}.zip"
CURRENT_MANIFEST = Path("outputs/EXP-3D-RGBD-TRAIN-SCALE-V6/scene_split.json")
V1_LOCK = Path("outputs/EXP-3D-RGBD-FRESH-SCALE-DATA-V1/candidate_lock.json")
V1_SCRIPT = Path("scripts/prepare_rgbd_fresh_scale_data.py")
FRESH_V2_LOCK = Path("outputs/EXP-3D-RGBD-FRESH-V2-DATA/candidate_lock.json")
EVAL_V3_LOCK = Path("outputs/EXP-3D-RGBD-EVAL-V3-DATA/candidate_lock.json")
EVAL_V3_INTEGRITY = Path("outputs/EXP-3D-RGBD-EVAL-V3-DATA/preparation_integrity.json")
PER_VOLUME = 2
EVAL_V4_MIN = 20
BUDGET = 2 * 1024**3
SELECTION_SEED = "EVAL-V4-20260929"
INDEPENDENCE_BASIS = (
    "EVAL-V4 scenes are marked never observed and were never used, considered or read by any "
    "earlier experiment (including every read EVAL-V3 candidate); their source assets differ "
    "from every used or protected scene; their volumes hold no DEV or FRESH scene. They may "
    "share volumes with TRAIN72 and EVAL-V3 scenes, so EVAL-V4 is a scene- and asset-disjoint "
    "mechanism cohort, weaker than the volume-disjoint FRESH cohorts, and never a qualification "
    "cohort. The protected official-test final holdout is not opened."
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def v1_module(repo):
    """The frozen FRESH-V1/TRAIN-X preparation, reused as a library (never modified)."""
    spec = importlib.util.spec_from_file_location("rgbd_fresh_scale_v1", repo / V1_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def source_paths(repo):
    paths = v1_module(repo).source_paths(repo)
    paths["v1_preparation"] = paths.pop("preparation_source")
    paths.update(
        current_manifest=repo / CURRENT_MANIFEST,
        fresh_v1_lock=repo / V1_LOCK,
        fresh_v2_lock=repo / FRESH_V2_LOCK,
        eval_v3_lock=repo / EVAL_V3_LOCK,
        eval_v3_integrity=repo / EVAL_V3_INTEGRITY,
        preparation_source=Path(__file__).resolve(),
    )
    return paths


def selection_key(scene):
    return hashlib.sha256(f"{SELECTION_SEED}:{scene}".encode()).hexdigest()


def select_candidates(partitions, assets, allowed, trajectories, protected, excluded, used, fence):
    """Metadata-only candidate order; `fence` scenes define the forbidden volumes."""
    v1 = v1_module(Path(__file__).resolve().parents[1])
    forbidden = {v1.volume(s) for s in fence}
    blocked = {asset_identity(assets[s]) for s in used | protected if s in assets}
    selected = []
    for row in sorted(partitions, key=lambda r: selection_key(r["scene_name"])):
        sid = row["scene_name"]
        if (
            row["official_split"] != "train"
            or row["protocol_partition"] != "train"
            or row["previously_observed"] != "False"
            or sid in protected
            or sid in excluded
            or sid in used
            or sid not in assets
            or v1.volume(sid) in forbidden
            or asset_identity(assets[sid]) in blocked
            or len(allowed.get(sid, [])) < 16
            or v1.bad_trajectory(trajectories, sid)
        ):
            continue
        selected.append(sid)
        blocked.add(asset_identity(assets[sid]))
    return selected


def volume_full(counts, scene):
    """True when the scene's volume already holds PER_VOLUME valid EVAL-V4 scenes."""
    return counts[scene.split("_")[1]] >= PER_VOLUME


def eval_v3_read(eval_v3_lock, eval_v3_integrity):
    """Every EVAL-V3 candidate whose media was read (valid or invalid); the rest were never read."""
    if eval_v3_integrity["status"] != "PASS" or eval_v3_integrity["budget_stop"]:
        raise PermissionError("EVAL-V3 preparation must have completed without a budget stop")
    unread = set(eval_v3_integrity["never_read_full_volume"])
    candidates = set(eval_v3_lock["EVAL_V3_candidates"])
    if not unread <= candidates:
        raise ValueError("Never-read scenes must be EVAL-V3 candidates")
    read = candidates - unread
    if len(read) != eval_v3_integrity["EVAL_V3"] + len(eval_v3_integrity["failures"]):
        raise ValueError("Read EVAL-V3 candidates must be exactly its valid plus invalid scenes")
    return read


def eval_v4_sets(current, v1_lock, fresh_v2_lock, eval_v3_lock, eval_v3_integrity):
    """(used, fence): EVAL-V3's sets plus every read EVAL-V3 candidate; same volume fence."""
    used, fence = eval_v3_sets(current, v1_lock, fresh_v2_lock)
    return used | set(eval_v3_lock["used_scene_ids"]) | eval_v3_read(
        eval_v3_lock, eval_v3_integrity
    ), fence


def eval_v3_sets(current, v1_lock, fresh_v2_lock):
    """(used, fence): every scene used or considered so far; DEV and FRESH scenes fence volumes."""
    dev = {r["scene_id"] for r in current["scenes"] if r["split"] == "DEV"}
    fresh = set(v1_lock["FRESH_V1_candidates"]) | set(fresh_v2_lock["FRESH_V2_candidates"])
    used = (
        {r["scene_id"] for r in current["scenes"]}
        | set(v1_lock["current_scene_ids"])
        | set(v1_lock["TRAIN_X_candidates"])
        | fresh
    )
    return used, dev | fresh


def build_lock(repo):
    paths = source_paths(repo)
    v1 = v1_module(repo)
    assets = safe_source_assets(paths["assets"])
    partitions = csv_rows(paths["partitions"])
    trajectories = {r["Animation"]: r for r in csv_rows(paths["trajectories"])}
    current = json.loads(paths["current_manifest"].read_text())
    v1_lock = json.loads(paths["fresh_v1_lock"].read_text())
    fresh_v2_lock = json.loads(paths["fresh_v2_lock"].read_text())
    eval_v3_lock = json.loads(paths["eval_v3_lock"].read_text())
    eval_v3_integrity = json.loads(paths["eval_v3_integrity"].read_text())
    used, fence = eval_v4_sets(current, v1_lock, fresh_v2_lock, eval_v3_lock, eval_v3_integrity)
    excluded = {next(iter(r.values())) for r in csv_rows(paths["exclusions"])}
    protected = (
        set(current["protected_scene_ids"])
        | set(v1_lock["protected_scene_ids"])
        | set(fresh_v2_lock["protected_scene_ids"])
        | set(eval_v3_lock["protected_scene_ids"])
        | {
            r["scene_name"]
            for r in partitions
            if r["protocol_partition"] in ("final_holdout", "diagnostic_test")
        }
    )
    allowed = v1.allowed_frames(paths["official_split"], "train")
    candidates = select_candidates(
        partitions, assets, allowed, trajectories, protected, excluded, used, fence
    )
    volumes = sorted({v1.volume(s) for s in candidates})
    if len(volumes) * PER_VOLUME < EVAL_V4_MIN:
        raise ValueError(f"Insufficient metadata candidates: {len(candidates)} in {volumes}")
    return {
        "schema": "mcss.rgbd_eval_v4.cohort.v1",
        "experiment": EXPERIMENT,
        "EVAL_V4_candidates": candidates,
        "candidate_volumes": volumes,
        "per_volume": PER_VOLUME,
        "minimum_valid": EVAL_V4_MIN,
        "fence_scene_ids": sorted(fence),
        "used_scene_ids": sorted(used),
        "protected_scene_ids": sorted(protected),
        "excluded_scene_ids": sorted(excluded),
        "assets": {s: assets[s] for s in candidates},
        "eval_rule": (
            "official train split & protocol train & previously_observed False; not protected/"
            "excluded; not used, considered or read by TRAIN72/DEV/FRESH-V1/FRESH-V2/TRAIN-X/"
            "EVAL-V3; source asset unique vs used+protected and within the cohort; volume disjoint "
            "from every DEV, FRESH-V1 and FRESH-V2 scene (volumes may be shared with TRAIN72 and "
            "EVAL-V3); cam00 >=16 public "
            f"train frames; nonBAD trajectory; sha256('{SELECTION_SEED}:'+scene) order; per volume "
            f"the first {PER_VOLUME} candidates passing the model-free validity rule are kept; "
            "later candidates of a full volume are never read"
        ),
        "validity_rule": v1_lock["validity_rule"],
        "role_rule": v1_lock["role_rule"],
        "independence_basis": INDEPENDENCE_BASIS,
        "source_sha256": {str(p): sha(p) for p in paths.values()},
        "data_budget_bytes": BUDGET,
        "model_forward_run": False,
        "protected_media_read": False,
    }


def prepare(repo, root):
    repo, root = Path(repo).resolve(), Path(root).resolve()
    lock_path = root / "candidate_lock.json"
    lock = json.loads(lock_path.read_text())
    if lock != build_lock(repo):
        raise PermissionError("Candidate lock no longer reproduces from its frozen sources")
    if (root / "manifest_eval_v4.json").exists():
        raise FileExistsError("Completed manifest_eval_v4.json exists; no overwrite")
    v1 = v1_module(repo)
    paths = source_paths(repo)
    protected = set(lock["protected_scene_ids"])
    calibration = published_scene_metadata(paths["calibration"])
    bounds = json.loads(paths["bounds"].read_text())
    prior = FrozenTrainingPrior(bounds["near_m"], bounds["far_m"], bounds["source_sha256"])
    camera_only_roles = v1.role_rule(repo)
    allowed = v1.allowed_frames(paths["official_split"], "train")
    network = [0]
    hashes, qualities, geometry, failures, supports = {}, {}, [], {}, {}
    records, skipped, counts = [], [], Counter()
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
                "purpose": "EVAL_V4_DATA_PREPARATION_NO_FORWARD",
            }
            log.write(json.dumps(entry) + "\n")
            log.flush()

        for sid in lock["EVAL_V4_candidates"]:
            if volume_full(counts, sid):
                skipped.append(sid)
                continue
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
                    "historically_exposed": False,
                    "historical_exposure_basis": lock["independence_basis"],
                    "physical_scene_id": "|".join(asset_identity(lock["assets"][sid])),
                    "roles": roles,
                    "frames": frames,
                }
            )
            counts[sid.split("_")[1]] += 1
            supports[sid] = support
            write(root / "role_support.json", supports)
            write(root / "progress.json", {SPLIT: [r["scene_id"] for r in records]})
            print("PREPARED", sid, flush=True)
    write(root / "hashes.json", hashes)
    write(
        root / "download_cost.json",
        {"network_bytes": network[0], "full_archives_downloaded": False, "budget_bytes": BUDGET},
    )
    status = "PASS" if len(records) >= EVAL_V4_MIN else "BLOCKED_INSUFFICIENT_VALID_LOCKED_SCENES"
    write(
        root / "manifest_eval_v4.json",
        {
            "schema": "mcss.rgbd_eval_v4.data.v1",
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
            SPLIT: len(records),
            "valid_per_volume": dict(sorted(counts.items())),
            "never_read_full_volume": skipped,
            "budget_stop": budget_stop,
            "failures": failures,
            "candidate_replacement": "model-free validity only, within volume, in locked order",
            "protected_media_read": False,
            "model_forward_run": False,
            "network_bytes": network[0],
            "candidate_lock_sha256": sha(lock_path),
        },
    )
    print(status, len(records), "failures", len(failures), "skipped", len(skipped), flush=True)


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
                    "EVAL_V4_candidates": len(lock["EVAL_V4_candidates"]),
                    "candidate_volumes": lock["candidate_volumes"],
                }
            )
        )
    else:
        prepare(args.repo, root)


if __name__ == "__main__":
    main()
