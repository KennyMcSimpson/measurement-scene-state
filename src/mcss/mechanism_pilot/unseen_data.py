"""Bounded unseen-development preparation; selection uses metadata and geometry only."""

from __future__ import annotations

import ast
import csv
import itertools
import json
import re
import time
import zipfile
from pathlib import Path

import h5py
import numpy as np
import requests
import torch
from PIL import Image

from mcss.data.hypersim import convert_hypersim_pose
from mcss.dynamic.checkpoint import load_dynamic_checkpoint
from mcss.mechanism_pilot.calibration import calibrated_intrinsics, published_scene_metadata
from mcss.mechanism_pilot.data_prep import BoundedRangeReader, check_geometry, resize_rgb_depth
from mcss.mechanism_pilot.small_training import sha, write_json
from mcss.mechanism_pilot.spatial import observed_camera_support
from mcss.types import Cameras


def source_assets(path):
    """Extract literal upstream scenes.append dictionaries WITHOUT executing source."""
    result = {}
    for node in ast.walk(ast.parse(Path(path).read_text())):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if (
                isinstance(node.func.value, ast.Name)
                and node.func.value.id == "scenes"
                and node.func.attr == "append"
                and len(node.args) == 1
            ):
                try:
                    row = ast.literal_eval(node.args[0])
                except (ValueError, TypeError):
                    continue  # Never execute nonliteral upstream path expressions.
                result[row["name"]] = row
    return result


def asset_identity(row):
    return (
        row["archive_file"].replace("\\", "/").casefold(),
        row["asset_file"].replace("\\", "/").casefold(),
    )


def validate_unseen(train_ids, selected_ids, assets):
    if set(train_ids) & set(selected_ids):
        raise ValueError("Unseen scene appears in checkpoint training manifest")
    identities = [asset_identity(assets[s]) for s in list(train_ids) + list(selected_ids)]
    if len(identities) != len(set(identities)):
        raise ValueError("Source asset identity overlap or duplicate")
    train_volumes = {s.split("_")[1] for s in train_ids}
    if any(s.split("_")[1] in train_volumes for s in selected_ids):
        raise ValueError("Unseen volume overlaps training asset volume")


def frame_quality(rgb, depth):
    finite_rgb = bool(np.isfinite(rgb).all())
    black_fraction = float((rgb <= 2).all(-1).mean())
    valid_depth_fraction = float((np.isfinite(depth) & (depth > 0)).mean())
    return {
        "finite_rgb": finite_rgb,
        "near_black_fraction": black_fraction,
        "valid_depth_fraction": valid_depth_fraction,
        "eligible": finite_rgb and black_fraction <= 0.999 and valid_depth_fraction >= 0.95,
    }


def select_roles(frames, quality, candidates):
    valid = sorted(f["frame_id"] for f in frames if quality[f["frame_id"]]["eligible"])
    if len(valid) < 7:
        return None, {"status": "INELIGIBLE", "reason": "fewer_than_7_quality_frames"}
    anchor, query = valid[0], valid[-2:]
    rows = {f["frame_id"]: f for f in frames}

    def camera(i):
        f = rows[i]
        return Cameras(torch.tensor(f["intrinsics"]), torch.tensor(f["c2w"]), (128, 160))

    anchor_camera = camera(anchor)
    options = []
    for pair in itertools.combinations([i for i in valid if i != anchor and i not in query], 2):
        support = observed_camera_support(candidates, [anchor_camera] + [camera(i) for i in pair])[
            "supported_candidates"
        ]
        baseline = (
            sum(float((camera(i).c2w[:3, 3] - anchor_camera.c2w[:3, 3]).norm()) for i in pair) / 2
        )
        options.append((pair, support, baseline))
    eligible = []
    for a, b in itertools.combinations(options, 2):
        if set(a[0]) & set(b[0]):
            continue
        key = (-min(a[1], b[1]), abs(a[1] - b[1]), abs(a[2] - b[2]), a[0], b[0])
        eligible.append((key, a, b))
    _, a, b = min(eligible)
    diagnostics = {
        "status": "ELIGIBLE" if min(a[1], b[1]) >= 2 else "INELIGIBLE",
        "reason": None if min(a[1], b[1]) >= 2 else "less_than_2_supported_candidates",
        "context_a_supported": a[1],
        "context_b_supported": b[1],
        "context_a_baseline_m": a[2],
        "context_b_baseline_m": b[2],
        "quality_frame_ids": valid,
        "geometry_pairings_considered": len(eligible),
    }
    roles = {
        "context_a": [anchor, *a[0]],
        "context_b": [anchor, *b[0]],
        "primary_query": query,
        "query": query,
        "secondary_query": [],
        "stream": [],
    }
    return roles, diagnostics


def csv_rows(path):
    with Path(path).open(newline="") as f:
        return list(csv.DictReader(f))


def select_members(scene, infos, allowed_ids):
    names = {x.filename: x for x in infos}
    rgb, depth = {}, {}
    for name in names:
        m = re.fullmatch(
            re.escape(scene) + r"/images/scene_cam_00_final_preview/frame\.(\d{4})\.color\.jpg",
            name,
        )
        if m:
            rgb[int(m[1])] = name
        m = re.fullmatch(
            re.escape(scene)
            + r"/images/scene_cam_00_geometry_hdf5/frame\.(\d{4})\.depth_meters\.hdf5",
            name,
        )
        if m:
            depth[int(m[1])] = name
    ids = sorted(rgb.keys() & depth.keys() & set(allowed_ids))[:16]
    if len(ids) < 16:
        raise ValueError("Fewer than 16 officially included train RGB/depth frames")
    members = [f"{scene}/_detail/metadata_scene.csv"] + [
        f"{scene}/_detail/cam_00/camera_keyframe_{k}.hdf5"
        for k in ("frame_indices", "positions", "orientations")
    ]
    members += [n for i in ids for n in (rgb[i], depth[i])]
    members += [f"{scene}/images/scene_cam_00_geometry_hdf5/frame.{ids[0]:04d}.position.hdf5"]
    if any(n not in names for n in members):
        raise ValueError("Missing camera/GT position metadata")
    return {
        "scene_id": scene,
        "frame_ids": ids,
        "members": members,
        "compressed_bytes": sum(names[n].compress_size for n in members),
        "uncompressed_bytes": sum(names[n].file_size for n in members),
        "member_metadata": [
            {"name": n, "crc32": names[n].CRC, "bytes": names[n].file_size} for n in members
        ],
    }


def prepare_unseen(
    output, experiment, checkpoint_path, training_dir, training_manifest, partitions, calibration
):
    output, experiment, training_dir = Path(output), Path(experiment), Path(training_dir)
    if output.exists():
        raise FileExistsError("Refusing overwrite of locked unseen cohort")
    metadata = experiment / "rgb_audit/official_metadata"
    upstream = experiment / "metadata/upstream_dataset_config.py"
    assets = source_assets(upstream)
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
    selected, volumes = [], {s.split("_")[1] for s in train_ids}
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
        if len(selected) == 8:
            break
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
        "schema": "mcss.unseen_static.candidates.v1",
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
                            "split": "dev",
                            "physical_scene_id": "|".join(asset_identity(assets[sid])),
                            "frames": frames,
                            "roles": roles,
                        }
                    )
                write_json(output / "progress_quality.json", quality_rows)
                write_json(output / "progress_feasibility.json", feasibility)
                print("PREPARED " + sid + " " + json.dumps(diagnostic), flush=True)
        status = "READY" if len(scenes) >= 6 else "BLOCKED_INSUFFICIENT_DATA"
        manifest = {
            "schema": "mcss.static_unseen.data.v1",
            "image_size": [128, 160],
            "depth_semantics": "ray_distance_meters",
            "scenes": scenes,
            "UNSEEN_DEV_STATUS": status,
            "candidate_lock_sha256": sha(output / "candidate_lock.json"),
            "data_lock_sha256": sha(output / "data_lock.json"),
            "model_predictions_read": False,
        }
        write_json(output / "manifest.json", manifest)
        write_json(output / "frame_quality.json", quality_rows)
        write_json(output / "support_feasibility.json", feasibility)
        write_json(output / "geometry_checks.json", geometry)
        write_json(output / "hashes.json", hashes)
        write_json(
            output / "role_lock.json",
            {
                "created_unix": time.time(),
                "manifest_sha256": sha(output / "manifest.json"),
                "roles": {s["scene_id"]: s["roles"] for s in scenes},
                "N_UNSEEN_DEV_SCENES": len(scenes),
                "UNSEEN_DEV_STATUS": status,
            },
        )
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
