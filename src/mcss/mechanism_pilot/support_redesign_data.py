"""New support-redesign data cohorts, with sealed final holdout metadata.

Only source metadata, camera geometry, RGB/depth validity are read. No model output.
Old preparation modules and old experiment hashes are never modified.
"""

from __future__ import annotations

import ast
import json
import time
import zipfile
from pathlib import Path

import h5py
import numpy as np
import requests
from PIL import Image

from mcss.data.hypersim import convert_hypersim_pose
from mcss.dynamic.checkpoint import load_dynamic_checkpoint
from mcss.mechanism_pilot.calibration import calibrated_intrinsics, published_scene_metadata
from mcss.mechanism_pilot.data_prep import BoundedRangeReader, check_geometry, resize_rgb_depth
from mcss.mechanism_pilot.small_training import sha, write_json
from mcss.mechanism_pilot.unseen_data import (
    asset_identity,
    csv_rows,
    frame_quality,
    select_members,
    select_roles,
    validate_unseen,
)


def safe_source_assets(path):
    """Parse literal dictionaries and os.path.join of literal strings, never execute."""
    rows = {}
    for node in ast.walk(ast.parse(Path(path).read_text())):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "scenes"
            and node.func.attr == "append"
            and len(node.args) == 1
            and isinstance(node.args[0], ast.Dict)
        ):
            continue
        row = {}
        for key, value in zip(node.args[0].keys, node.args[0].values, strict=True):
            key = ast.literal_eval(key)
            if isinstance(value, ast.Call):
                if ast.unparse(value.func) != "os.path.join" or value.keywords:
                    raise ValueError("Unsupported source metadata expression")
                parts = [ast.literal_eval(a) for a in value.args]
                if not all(isinstance(a, str) and not a.startswith(("/", "\\")) for a in parts):
                    raise ValueError("Invalid literal source asset path")
                row[key] = "/".join(parts)
            else:
                row[key] = ast.literal_eval(value)
        rows[row["name"]] = row
    return rows


def validate_roles_disjoint(train, exposed, dev, holdout, assets):
    groups = [set(x) for x in (train, exposed, dev, holdout)]
    if any(groups[i] & groups[j] for i in range(4) for j in range(i + 1, 4)):
        raise ValueError("Train/exposed/dev/final holdout overlap")
    all_ids = train + exposed + dev + holdout
    identities = [asset_identity(assets[s]) for s in all_ids]
    if len(identities) != len(set(identities)):
        raise ValueError("Source asset overlap across cohorts")
    volumes = [s.split("_")[1] for s in all_ids]
    if len(volumes) != len(set(volumes)):
        raise ValueError("Source volume overlap across cohorts")


def prepare_support_redesign(
    output, experiment, checkpoint_path, training_dir, training_manifest, partitions, calibration
):
    output, experiment, training_dir = Path(output), Path(experiment), Path(training_dir)
    if output.exists():
        raise FileExistsError("Refusing overwrite of locked unseen cohort")
    metadata = experiment / "rgb_audit/official_metadata"
    upstream = experiment / "metadata/upstream_dataset_config.py"
    assets = safe_source_assets(upstream)
    train_manifest = json.loads(Path(training_manifest).read_text())
    train_ids = sorted(s["scene_id"] for s in train_manifest["scenes"])
    log_ids = sorted(
        {
            json.loads(s)["scene_id"]
            for s in (training_dir / "training.jsonl").read_text().splitlines()
        }
    )
    training_summary = json.loads((training_dir / "summary.json").read_text())
    if (
        train_ids != ["ai_001_001", "ai_002_001", "ai_003_001"]
        or train_ids != log_ids
        or training_summary["n_train_scenes"] != 3
    ):
        raise ValueError("Training identities not consistent")
    carrier, _, checkpoint = load_dynamic_checkpoint(checkpoint_path)
    if checkpoint["sha256"] != training_summary["checkpoint_sha256"]["phase_b_final"]:
        raise ValueError("Checkpoint hash mismatch")
    partition_rows = csv_rows(partitions)
    trajectories = {
        r["Animation"]: r for r in csv_rows(metadata / "metadata_camera_trajectories.csv")
    }
    split_rows = csv_rows(metadata / "metadata_images_split_scene_v1.csv")
    allowed = {}
    for row in split_rows:
        if (
            row["camera_name"] == "cam_00"
            and row["included_in_public_release"] == "True"
            and row["split_partition_name"] == "train"
        ):
            allowed.setdefault(row["scene_name"], []).append(int(row["frame_id"]))
    old_manifest = json.loads((experiment / "unseen_data/manifest.json").read_text())
    exposed_ids = sorted(s["scene_id"] for s in old_manifest["scenes"])
    # Entire old candidate cohort, including geometry-ineligible ai005003, is excluded.
    old_candidates = json.loads((experiment / "unseen_data/candidate_lock.json").read_text())[
        "candidate_scene_ids"
    ]
    selected, volumes = [], {s.split("_")[1] for s in train_ids + old_candidates}
    for row in sorted(partition_rows, key=lambda r: r["scene_name"]):
        scene = row["scene_name"]
        volume = scene.split("_")[1]
        if (
            volume in volumes
            or row["protocol_partition"] != "train"
            or scene not in assets
            or len(allowed.get(scene, [])) < 16
        ):
            continue
        trajectory = trajectories.get(scene + "_cam_00")
        if trajectory is None or "BAD" in trajectory["Scene type"].upper():
            continue
        selected.append(scene)
        volumes.add(volume)
        if len(selected) == 24:
            break
    dev_ids, holdout_ids = selected[::2], selected[1::2]
    validate_roles_disjoint(train_ids, exposed_ids, dev_ids, holdout_ids, assets)
    prereg = output.parent / "preregistration.json"
    if not prereg.exists():
        raise ValueError("Unified preregistration must exist before any cohort lock/download")
    validate_unseen(train_ids, selected, assets)
    output.mkdir(parents=True)
    source_paths = [
        upstream,
        Path(partitions),
        metadata / "metadata_camera_trajectories.csv",
        metadata / "metadata_images_split_scene_v1.csv",
        Path(training_manifest),
        training_dir / "summary.json",
        training_dir / "training.jsonl",
        Path(calibration),
    ]
    lock = {
        "schema": "mcss.support_redesign.candidates.v1",
        "preregistration_sha256": sha(prereg),
        "REDESIGN_DEV_SCENES": dev_ids,
        "FINAL_HOLDOUT_SCENES": holdout_ids,
        "REDESIGN_DEV_EXPOSED": exposed_ids,
        "native_geometry_p95_limit_m": 0.01,
        "holdout_model_access": "PROHIBITED_UNTIL_METHOD_LOCK_AND_DEV_GATE",
        "created_unix": time.time(),
        "checkpoint_sha256": checkpoint["sha256"],
        "train_scene_ids": train_ids,
        "candidate_scene_ids": selected,
        "assets": {s: assets[s] for s in train_ids + selected},
        "selection": (
            "lexicographic first eligible protocol-train cam00 non-BAD scene "
            "from each distinct untrained asset volume; 8 fixed candidates, no "
            "replacement"
        ),
        "physical_identity_limit": (
            "Distinct official volume/archive/asset identities; vendor cross-"
            "asset content reuse cannot be independently ruled out."
        ),
        "previously_observed": {
            r["scene_name"]: r["previously_observed"]
            for r in partition_rows
            if r["scene_name"] in selected
        },
        "source_sha256": {str(p): sha(p) for p in source_paths},
        "upstream_url": "https://github.com/apple/ml-hypersim/blob/main/evermotion_dataset/_dataset_config.py",
        "data_budget_bytes": 2 * 1024**3,
        "quality_rule": (
            "finite RGB, >99.9 percent pixels all channels <=2/255 rejected; "
            "finite positive native depth fraction >=0.95; no scores"
        ),
        "role_rule": (
            "first quality frame anchor, last two quality frames queries; "
            "enumerate disjoint pairs from remaining frames. Maximize "
            "min(A_support,B_support), minimize support gap, then mean baseline "
            "gap, then lex frame IDs. At least 7 quality frames; each context "
            ">=2 supported candidates. This minimum is not a quality proof."
        ),
        "model_predictions_read": False,
    }
    write_json(output / "candidate_lock.json", lock)
    write_json(
        output.parent / "scene_role_lock.json",
        {
            "created_unix": time.time(),
            "stage": "INITIAL_IMMUTABLE_COHORT_SELECTION",
            "candidate_lock_sha256": sha(output / "candidate_lock.json"),
            "TRAIN_SCENES": train_ids,
            "REDESIGN_DEV_EXPOSED": exposed_ids,
            "REDESIGN_DEV_SCENES": dev_ids,
            "FINAL_HOLDOUT_SCENES": holdout_ids,
            "FINAL_HOLDOUT_PROTECTED": True,
            "ineligible_holdout_transfer_to_dev": "PROHIBITED",
            "qualified_frame_roles_will_be_separate": "qualified_frame_role_lock.json",
            "source_assets": {s: assets[s] for s in train_ids + exposed_ids + selected},
            "model_predictions_read": False,
        },
    )
    print("CANDIDATE_LOCKED " + json.dumps(selected), flush=True)
    sessions, readers, archives, plans = [], [], [], []
    shared_budget = [0]
    calibration_rows = published_scene_metadata(calibration)
    try:
        for scene in selected:
            session = requests.Session()
            url = f"https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/{scene}.zip"
            reader = BoundedRangeReader(url, session)
            reader.shared_budget = shared_budget
            archive = zipfile.ZipFile(reader)
            plan = select_members(scene, archive.infolist(), allowed[scene])
            plan.update(url=url, etag=reader.etag, archive_bytes=reader.size)
            sessions.append(session)
            readers.append(reader)
            archives.append(archive)
            plans.append(plan)
            print("MEMBER_PLAN " + scene, flush=True)
        if sum(p["compressed_bytes"] + p["uncompressed_bytes"] for p in plans) > 2 * 1024**3:
            raise ValueError("Selected data exceeds 2GiB bound")
        write_json(
            output / "data_lock.json",
            {
                "created_unix": time.time(),
                "candidate_lock_sha256": sha(output / "candidate_lock.json"),
                "scenes": plans,
            },
        )
        print("ALL_MEMBERS_LOCKED", flush=True)
        hashes, quality_rows, feasibility, scenes, geometry = {}, {}, {}, [], []
        with (output / "access.jsonl").open("x") as log:

            def access(sid, path, operation):
                log.write(
                    json.dumps(
                        {
                            "time": time.time(),
                            "scene_id": sid,
                            "path": str(path),
                            "operation": operation,
                            "purpose": "unseen_data_geometry_only",
                            "official_split": "train",
                        }
                    )
                    + "\n"
                )
                log.flush()

            for plan, archive, _reader in zip(plans, archives, readers, strict=True):
                sid = plan["scene_id"]
                for member in plan["members"]:
                    dest = output / "raw" / member
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    access(sid, member, "download")
                    dest.write_bytes(archive.read(member))
                    hashes[str(dest.relative_to(output))] = sha(dest)
                raw = output / "raw" / sid
                units = {
                    r["parameter_name"]: r["parameter_value"]
                    for r in csv_rows(raw / "_detail/metadata_scene.csv")
                }
                scale = float(units["meters_per_asset_unit"])
                if not np.isclose(
                    scale,
                    float(calibration_rows[sid]["settings_units_info_meters_scale"]),
                    rtol=1e-6,
                ):
                    raise ValueError("Native calibration scale mismatch")

                def hdf(path, sid=sid):
                    access(sid, path, "decode_hdf5")
                    with h5py.File(path) as f:
                        return np.array(f["dataset"])

                ids = hdf(raw / "_detail/cam_00/camera_keyframe_frame_indices.hdf5").reshape(-1)
                poses = hdf(raw / "_detail/cam_00/camera_keyframe_positions.hdf5")
                rotations = hdf(raw / "_detail/cam_00/camera_keyframe_orientations.hdf5")
                mapping = {int(i): j for j, i in enumerate(ids)}
                frames, quality = [], {}
                prepared = output / "prepared" / sid
                prepared.mkdir(parents=True)
                for fid in plan["frame_ids"]:
                    rgbpath = raw / f"images/scene_cam_00_final_preview/frame.{fid:04d}.color.jpg"
                    access(sid, rgbpath, "decode_RGB")
                    with Image.open(rgbpath) as im:
                        rgb = np.asarray(im.convert("RGB"))
                    depth = hdf(
                        raw / f"images/scene_cam_00_geometry_hdf5/frame.{fid:04d}.depth_meters.hdf5"
                    )
                    quality[fid] = frame_quality(rgb, depth)
                    if rgb.shape[:2] != depth.shape:
                        raise ValueError("Native RGB/depth dimensions disagree")
                    pose = convert_hypersim_pose(
                        rotations[mapping[fid]], poses[mapping[fid]], meters_per_asset_unit=scale
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
                                    calibrated_intrinsics(calibration_rows[sid], depth.shape),
                                    scale,
                                )
                            )
                        except ValueError as exc:
                            geometry.append({"scene_id": sid, "status": "FAIL", "error": str(exc)})
                    rgb_small, depth_small = resize_rgb_depth(rgb, depth)
                    rgb_out, depth_out = prepared / f"{fid:04d}.png", prepared / f"{fid:04d}.npy"
                    Image.fromarray(rgb_small).save(rgb_out)
                    np.save(depth_out, depth_small)
                    for path in (rgb_out, depth_out):
                        hashes[str(path.relative_to(output))] = sha(path)
                    frames.append(
                        {
                            "frame_id": fid,
                            "rgb": str(rgb_out.resolve()),
                            "depth": str(depth_out.resolve()),
                            "intrinsics": calibrated_intrinsics(
                                calibration_rows[sid], (128, 160)
                            ).tolist(),
                            "c2w": pose.tolist(),
                        }
                    )
                roles, diagnostic = select_roles(
                    frames, quality, carrier._candidate_points.detach()
                )
                if any(r.get("status") == "FAIL" and r["scene_id"] == sid for r in geometry):
                    diagnostic.update(status="INELIGIBLE", reason="native_geometry_check_failed")
                quality_rows[sid], feasibility[sid] = quality, diagnostic
                if diagnostic["status"] == "ELIGIBLE":
                    scenes.append(
                        {
                            "scene_id": sid,
                            "split": "dev" if sid in dev_ids else "final_holdout",
                            "cohort_role": "REDESIGN_DEVELOPMENT"
                            if sid in dev_ids
                            else "FINAL_QUALIFICATION_HOLDOUT",
                            "physical_scene_id": "|".join(asset_identity(assets[sid])),
                            "frames": frames,
                            "roles": roles,
                        }
                    )
                write_json(output / "progress_quality.json", quality_rows)
                write_json(output / "progress_feasibility.json", feasibility)
                print("PREPARED " + sid + " " + json.dumps(diagnostic), flush=True)
        dev = [s for s in scenes if s["scene_id"] in dev_ids]
        holdout = [s for s in scenes if s["scene_id"] in holdout_ids]
        status = "READY" if len(dev) >= 8 and len(holdout) >= 8 else "BLOCKED_INSUFFICIENT_DATA"
        common = {
            "schema": "mcss.support_redesign.data.v1",
            "image_size": [128, 160],
            "depth_semantics": "ray_distance_meters",
            "DATA_COHORT_STATUS": status,
            "candidate_lock_sha256": sha(output / "candidate_lock.json"),
            "data_lock_sha256": sha(output / "data_lock.json"),
            "model_predictions_read": False,
        }
        manifest = dict(common, scenes=scenes)
        write_json(output / "manifest.json", manifest)
        write_json(output / "dev_manifest.json", dict(common, scenes=dev))
        write_json(
            output / "holdout_manifest.json",
            dict(
                common, scenes=holdout, inference_access="PROHIBITED_UNTIL_METHOD_LOCK_AND_DEV_GATE"
            ),
        )
        write_json(output / "frame_quality.json", quality_rows)
        write_json(output / "support_feasibility.json", feasibility)
        write_json(output / "geometry_checks.json", geometry)
        write_json(output / "hashes.json", hashes)
        role_lock = {
            "created_unix": time.time(),
            "manifest_sha256": sha(output / "manifest.json"),
            "candidate_lock_sha256": sha(output / "candidate_lock.json"),
            "TRAIN_SCENES": train_ids,
            "REDESIGN_DEV_EXPOSED": exposed_ids,
            "REDESIGN_DEV_SCENES": dev_ids,
            "FINAL_HOLDOUT_SCENES": holdout_ids,
            "eligible_redesign_dev": [s["scene_id"] for s in dev],
            "eligible_final_holdout": [s["scene_id"] for s in holdout],
            "roles": {s["scene_id"]: s["roles"] for s in scenes},
            "DATA_COHORT_STATUS": status,
            "model_predictions_read": False,
            "FINAL_HOLDOUT_PROTECTED": True,
            "holdout_inference_requires": "qualified dev gate AND frozen method lock",
            "source_assets": {s: assets[s] for s in train_ids + exposed_ids + selected},
            "dataset_hashes_sha256": sha(output / "hashes.json"),
            "initial_scene_role_lock_sha256": sha(output.parent / "scene_role_lock.json"),
            "manifest_hashes": {
                n: sha(output / n) for n in ["dev_manifest.json", "holdout_manifest.json"]
            },
        }
        write_json(output.parent / "qualified_frame_role_lock.json", role_lock)
        write_json(
            output / "download_cost.json",
            {
                "network_bytes": sum(r.transferred for r in readers),
                "budget_accounted_bytes": shared_budget[0],
                "full_archives_downloaded": False,
            },
        )
        return manifest
    finally:
        for obj in archives + readers + sessions:
            obj.close()
