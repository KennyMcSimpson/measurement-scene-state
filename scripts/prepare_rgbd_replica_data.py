#!/usr/bin/env python3
"""Lock and materialize the Replica cross-dataset cohort (no model, no score).

Source: the Replica dataset (Straub et al. 2019) as rendered RGB-D sequences distributed by
NICE-SLAM (one ZIP of 8 scenes x 2000 frames, 1200x680, pinhole fx=fy=600). Its use is subject to
the Replica research license (research only, no redistribution); the user approved the download
on 2026-09-29. Only the needed members are fetched through exact HTTP ranges.

`--stage lock` is metadata only (the ZIP central directory, cam_params.json and the 8 camera
trajectories): it freezes the scenes, the frame rule, the crop/resize/depth conversion, the
validity rules and the frozen camera-only role rule before any image or depth byte is read.
`--stage prepare` fetches the locked members, applies the rules and writes a manifest in the
format of the Hypersim cohorts; it never runs a model or looks at a score.
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

import numpy as np
import requests
from PIL import Image

from mcss.mechanism_pilot.data_prep import BoundedRangeReader, resize_rgb_depth
from mcss.mechanism_pilot.optimization_bounds_contracts import FrozenTrainingPrior
from mcss.mechanism_pilot.unseen_data import frame_quality

EXPERIMENT = "EXP-3D-RGBD-REPLICA-DATA-V1"
SPLIT = "REPLICA"
URL = "https://cvg-data.inf.ethz.ch/nice-slam/data/Replica.zip"
SCENES = ("office0", "office1", "office2", "office3", "office4", "room0", "room1", "room2")
SOURCE_SIZE = (680, 1200)
TARGET_SIZE = (128, 160)
# Central crop to the median TRAIN72 Hypersim field of view (fx 138.56, fy 147.80 at 128x160):
# 520x692 source pixels centred on the principal point (599.5, 339.5) of the 1200x680 frames.
CROP = {"y0": 80, "x0": 254, "height": 520, "width": 692}
FRAMES_PER_SCENE = 16
SPACING_CANDIDATES = (50, 75, 100, 125)
HYPERSIM_CONSECUTIVE_ROTATION_DEG = 15.4  # median over TRAIN72 first16, camera metadata only
GEOMETRY_MEDIAN_ABSREL_MAX = 0.01
GEOMETRY_MIN_LANDING = 0.05
MIN_VALID_SCENES = 6
BUDGET = 1024**3
V6 = Path("outputs/EXP-3D-RGBD-TRAIN-SCALE-V6")
BOUNDS = Path("outputs/EXP-3D-16G-OPTIMIZATION-BOUNDS-V1/bounds_contract.json")
ROLE_SCRIPT = Path("scripts/prepare_geometry_carrier_data.py")
USE = (
    "Replica is a cross-dataset cohort: no scene, asset, volume or camera rig is shared with "
    "Hypersim. It is never a training set. Its first user must be named, with its analysis, in "
    "a plan written before any Replica result; the protected Hypersim final holdout is not "
    "opened."
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def read(path):
    return json.loads(Path(path).read_text())


def open_archive(budget=64 * 1024**2):
    """The remote ZIP through exact, ETag-pinned HTTP ranges (never a full download)."""
    reader = BoundedRangeReader(URL, requests.Session(), byte_budget=budget)
    return reader, zipfile.ZipFile(reader)


# ---------------------------------------------------------------- camera model (no image)
def source_intrinsics(params):
    camera = params["camera"]
    if (camera["h"], camera["w"]) != SOURCE_SIZE:
        raise ValueError("Unexpected Replica frame size")
    return np.array(
        [[camera["fx"], 0.0, camera["cx"]], [0.0, camera["fy"], camera["cy"]], [0.0, 0.0, 1.0]]
    )


def target_intrinsics(source):
    """Crop, then resize with the pixel-centre convention of resize_rgb_depth."""
    sy, sx = TARGET_SIZE[0] / CROP["height"], TARGET_SIZE[1] / CROP["width"]
    cx, cy = source[0, 2] - CROP["x0"], source[1, 2] - CROP["y0"]
    return np.array(
        [
            [source[0, 0] * sx, 0.0, (cx + 0.5) * sx - 0.5],
            [0.0, source[1, 1] * sy, (cy + 0.5) * sy - 0.5],
            [0.0, 0.0, 1.0],
        ]
    )


def trajectory(text):
    poses = np.array([list(map(float, line.split())) for line in text.splitlines() if line.strip()])
    return poses.reshape(-1, 4, 4)


def consecutive_motion(poses):
    steps = np.linalg.norm(np.diff(poses[:, :3, 3], axis=0), axis=-1)
    relative = np.einsum("nij,nik->njk", poses[:-1, :3, :3], poses[1:, :3, :3])
    cosine = np.clip((np.trace(relative, axis1=1, axis2=2) - 1) / 2, -1, 1)
    return steps, np.degrees(np.arccos(cosine))


def choose_spacing(trajectories):
    """Candidate whose across-scene median consecutive rotation is closest to Hypersim's."""
    table = {}
    for spacing in SPACING_CANDIDATES:
        medians = []
        for poses in trajectories.values():
            _, angles = consecutive_motion(poses[::spacing][:FRAMES_PER_SCENE])
            medians.append(float(np.median(angles)))
        table[spacing] = float(np.median(medians))
    best = min(
        SPACING_CANDIDATES, key=lambda s: (abs(table[s] - HYPERSIM_CONSECUTIVE_ROTATION_DEG), s)
    )
    return best, table


# ---------------------------------------------------------------- media rules
def ray_distance(z, intrinsics):
    """Planar z-depth (m) to distance along each pixel ray, the project's depth semantics."""
    y, x = np.indices(z.shape)
    u = (x - intrinsics[0, 2]) / intrinsics[0, 0]
    v = (y - intrinsics[1, 2]) / intrinsics[1, 1]
    return z * np.sqrt(u**2 + v**2 + 1.0)


def crop(array):
    return array[CROP["y0"] : CROP["y0"] + CROP["height"], CROP["x0"] : CROP["x0"] + CROP["width"]]


def reprojection_error(z_a, pose_a, z_b, pose_b, intrinsics, stride=4):
    """Median |z_proj - z_b| / z_b of frame-a points landing in frame b (OpenCV poses)."""
    y, x = np.indices(z_a.shape)
    y, x = y[::stride, ::stride].ravel(), x[::stride, ::stride].ravel()
    z = z_a[y, x]
    keep = np.isfinite(z) & (z > 0)
    y, x, z = y[keep], x[keep], z[keep]
    points = np.stack(
        (
            (x - intrinsics[0, 2]) / intrinsics[0, 0] * z,
            (y - intrinsics[1, 2]) / intrinsics[1, 1] * z,
            z,
        ),
        -1,
    )
    world = points @ pose_a[:3, :3].T + pose_a[:3, 3]
    local = (world - pose_b[:3, 3]) @ pose_b[:3, :3]
    front = local[:, 2] > 1e-6
    u = np.round(local[front, 0] / local[front, 2] * intrinsics[0, 0] + intrinsics[0, 2]).astype(
        int
    )
    v = np.round(local[front, 1] / local[front, 2] * intrinsics[1, 1] + intrinsics[1, 2]).astype(
        int
    )
    inside = (u >= 0) & (u < z_b.shape[1]) & (v >= 0) & (v < z_b.shape[0])
    observed = z_b[v[inside], u[inside]]
    projected = local[front][inside, 2]
    valid = np.isfinite(observed) & (observed > 0)
    landing = float(valid.sum() / max(len(z), 1))
    if not valid.any():
        return None, landing
    return float(np.median(np.abs(projected[valid] - observed[valid]) / observed[valid])), landing


def role_rule():
    spec = importlib.util.spec_from_file_location("v2_data_rules", ROLE_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.camera_only_roles


def frozen_prior():
    bounds = read(BOUNDS)
    return FrozenTrainingPrior(bounds["near_m"], bounds["far_m"], bounds["source_sha256"])


# ---------------------------------------------------------------- stages
def build_lock(archive, reader):
    infos = {i.filename: i for i in archive.infolist()}
    params = json.loads(archive.read("Replica/cam_params.json"))
    source = source_intrinsics(params)
    trajectories = {s: trajectory(archive.read(f"Replica/{s}/traj.txt").decode()) for s in SCENES}
    spacing, table = choose_spacing(trajectories)
    selected = {}
    for scene, poses in trajectories.items():
        if len(poses) != 2000:
            raise ValueError(f"{scene}: expected 2000 poses")
        ids = [spacing * k for k in range(FRAMES_PER_SCENE)]
        members = {}
        for fid in ids:
            for kind, name in (("rgb", f"frame{fid:06d}.jpg"), ("depth", f"depth{fid:06d}.png")):
                path = f"Replica/{scene}/results/{name}"
                info = infos[path]
                members.setdefault(str(fid), {})[kind] = {
                    "name": path,
                    "bytes": info.file_size,
                    "crc32": info.CRC,
                }
        selected[scene] = {
            "frame_ids": ids,
            "members": members,
            "poses_c2w_opencv": {str(fid): poses[fid].tolist() for fid in ids},
        }
    return {
        "schema": "mcss.rgbd_replica.cohort.v1",
        "experiment": EXPERIMENT,
        "source": {
            "url": URL,
            "etag": reader.etag,
            "bytes": reader.size,
            "distribution": "NICE-SLAM rendering of the Replica dataset (Straub et al. 2019)",
            "license": "Replica research license: research use only, no redistribution; the "
            "download was approved by the user on 2026-09-29; the media never enter the git "
            "repository (outputs/ is ignored)",
            "central_directory_members": len(infos),
        },
        "scenes": list(SCENES),
        "cam_params": params,
        "frame_rule": {
            "frames_per_scene": FRAMES_PER_SCENE,
            "spacing": spacing,
            "frame_ids": [spacing * k for k in range(FRAMES_PER_SCENE)],
            "spacing_rule": "among 50/75/100/125, the spacing whose across-scene median "
            "consecutive camera rotation of the 16 frames is closest to the TRAIN72 Hypersim "
            f"first16 median ({HYPERSIM_CONSECUTIVE_ROTATION_DEG} deg); camera metadata only",
            "spacing_table_median_rotation_deg": {str(k): v for k, v in table.items()},
        },
        "camera_model": {
            "source_intrinsics": source.tolist(),
            "crop": CROP,
            "target_size": list(TARGET_SIZE),
            "target_intrinsics": target_intrinsics(source).tolist(),
            "pose_convention": "traj.txt rows are camera-to-world 4x4 in the OpenCV convention "
            "(x right, y down, z forward; NICE-SLAM flips y/z to reach OpenGL), the project's "
            "convention; checked by the geometry rule",
            "depth": "uint16 PNG / scale (cam_params) = planar z in metres, converted to ray "
            "distance at full resolution, then cropped and resized with the frozen "
            "resize_rgb_depth (bilinear RGB, validity-weighted bilinear depth)",
        },
        "validity_rule": {
            "frame": "frozen frame_quality on the cropped full-resolution frame (finite RGB, "
            "near-black <=.999, positive finite depth >=.95); camera max|R^T R - I|<=.001 and "
            "det>0",
            "geometry": "median over consecutive selected-frame pairs of the median relative "
            f"z error of reprojected full-resolution depth <= {GEOMETRY_MEDIAN_ABSREL_MAX}, with "
            f"median landing fraction >= {GEOMETRY_MIN_LANDING}",
            "roles": "frozen V2 camera_only_roles (scripts/prepare_geometry_carrier_data.py) with "
            "the frozen TRAIN prior bounds",
            "minimum_valid_scenes": MIN_VALID_SCENES,
            "replacement": "none; an invalid scene is excluded",
        },
        "selected": selected,
        "use": USE,
        "data_budget_bytes": BUDGET,
        "metadata_bytes_read": reader.transferred,
        "image_or_depth_member_read": False,
        "model_forward_run": False,
        "source_sha256": {
            str(p): sha(p)
            for p in (Path(__file__).resolve(), ROLE_SCRIPT.resolve(), BOUNDS.resolve())
        },
        "created_utc": datetime.datetime.now(datetime.UTC).isoformat(),
    }


def lock(root):
    root = Path(root).resolve()
    if root.name != EXPERIMENT:
        raise PermissionError("Exact experiment directory required")
    if (root / "candidate_lock.json").exists():
        raise FileExistsError("Candidate lock exists; never re-lock")
    reader, archive = open_archive()
    with archive:
        value = build_lock(archive, reader)
    write(root / "candidate_lock.json", value)
    print(
        "LOCKED spacing",
        value["frame_rule"]["spacing"],
        "metadata bytes",
        value["metadata_bytes_read"],
        flush=True,
    )


def prepare(root, opener=open_archive):
    root = Path(root).resolve()
    lock_ = read(root / "candidate_lock.json")
    if (root / "manifest_replica.json").exists():
        raise FileExistsError("Completed manifest_replica.json exists; no overwrite")
    for path, digest in lock_["source_sha256"].items():
        if sha(path) != digest:
            raise PermissionError(f"Locked source changed: {path}")
    reader, archive = opener(BUDGET)
    if getattr(reader, "etag", lock_["source"]["etag"]) != lock_["source"]["etag"]:
        raise PermissionError("Remote archive changed since the lock")
    source = np.array(lock_["camera_model"]["source_intrinsics"])
    target = np.array(lock_["camera_model"]["target_intrinsics"])
    scale = float(lock_["cam_params"]["camera"]["scale"])
    roles_of, prior = role_rule(), frozen_prior()
    records, failures, qualities, geometry, supports, hashes = [], {}, {}, {}, {}, {}
    with archive, (root / "access.jsonl").open("a") as log:
        for scene in lock_["scenes"]:
            entry = lock_["selected"][scene]
            prepared = root / "prepared" / scene
            prepared.mkdir(parents=True, exist_ok=True)
            frames, quality, depths, poses = [], {}, {}, {}
            for fid in entry["frame_ids"]:
                members = entry["members"][str(fid)]
                raw = {}
                for kind in ("rgb", "depth"):
                    member = members[kind]
                    data = archive.read(member["name"])
                    if len(data) != member["bytes"] or zlib.crc32(data) != member["crc32"]:
                        raise ValueError(f"{member['name']}: size/CRC differs from the lock")
                    log.write(
                        json.dumps(
                            {
                                "scene": scene,
                                "member": member["name"],
                                "purpose": "REPLICA_DATA_PREPARATION_NO_FORWARD",
                            }
                        )
                        + "\n"
                    )
                    raw[kind] = data
                rgb = np.asarray(Image.open(io.BytesIO(raw["rgb"])).convert("RGB"))
                z = np.asarray(Image.open(io.BytesIO(raw["depth"]))).astype(np.float64) / scale
                if rgb.shape[:2] != tuple(SOURCE_SIZE) or z.shape != tuple(SOURCE_SIZE):
                    raise ValueError(f"{scene}/{fid}: unexpected frame size")
                z = np.where(z > 0, z, np.nan)
                distance = crop(ray_distance(z, source))
                rgb_crop = crop(rgb)
                pose = np.array(entry["poses_c2w_opencv"][str(fid)])
                rotation = pose[:3, :3]
                quality[fid] = frame_quality(rgb_crop, distance)
                orth = float(np.max(np.abs(rotation.T @ rotation - np.eye(3))))
                det = float(np.linalg.det(rotation))
                quality[fid].update(camera_orthogonality_error=orth, camera_determinant=det)
                quality[fid]["eligible"] = quality[fid]["eligible"] and orth <= 1e-3 and det > 0
                depths[fid], poses[fid] = z, pose
                rgb_small, depth_small = resize_rgb_depth(
                    rgb_crop, np.nan_to_num(distance, nan=0.0), TARGET_SIZE
                )
                rgb_out, depth_out = prepared / f"{fid:04d}.png", prepared / f"{fid:04d}.npy"
                Image.fromarray(rgb_small).save(rgb_out)
                np.save(depth_out, depth_small.astype(np.float32))
                hashes[str(rgb_out)], hashes[str(depth_out)] = sha(rgb_out), sha(depth_out)
                frames.append(
                    {
                        "frame_id": fid,
                        "rgb": str(rgb_out),
                        "depth": str(depth_out),
                        "intrinsics": target.tolist(),
                        "c2w": pose.tolist(),
                    }
                )
            qualities[scene] = {str(k): v for k, v in quality.items()}
            errors, landings = [], []
            ids = entry["frame_ids"]
            for a, b in zip(ids[:-1], ids[1:], strict=True):
                error, landing = reprojection_error(
                    depths[a], poses[a], depths[b], poses[b], source
                )
                if error is not None:
                    errors.append(error)
                landings.append(landing)
            check = {
                "median_pair_absrel": float(np.median(errors)) if errors else None,
                "median_landing_fraction": float(np.median(landings)),
                "pairs": len(ids) - 1,
            }
            check["pass"] = bool(
                errors
                and check["median_pair_absrel"] <= GEOMETRY_MEDIAN_ABSREL_MAX
                and check["median_landing_fraction"] >= GEOMETRY_MIN_LANDING
            )
            geometry[scene] = check
            if not check["pass"]:
                failures[scene] = {"status": "INELIGIBLE_SOURCE_GEOMETRY", **check}
                print("INELIGIBLE", scene, check, flush=True)
                continue
            sid = f"replica_{scene}"
            try:
                roles, support = roles_of(sid, frames, quality, prior)
            except ValueError as error:
                failures[scene] = {"status": "INELIGIBLE_FRAME_ROLES", "error": str(error)}
                print("INELIGIBLE", scene, "INELIGIBLE_FRAME_ROLES", flush=True)
                continue
            supports[scene] = support
            records.append(
                {
                    "scene_id": sid,
                    "split": SPLIT,
                    "historically_exposed": False,
                    "historical_exposure_basis": lock_["use"],
                    "physical_scene_id": sid,
                    "dataset": "Replica (NICE-SLAM rendering)",
                    "roles": roles,
                    "frames": frames,
                }
            )
            print("PREPARED", scene, check, flush=True)
    write(root / "frame_quality.json", qualities)
    write(root / "geometry_checks.json", geometry)
    write(root / "role_support.json", supports)
    write(root / "scene_failures.json", failures)
    write(root / "hashes.json", hashes)
    transferred = getattr(reader, "transferred", 0)
    status = "PASS" if len(records) >= MIN_VALID_SCENES else "BLOCKED_INSUFFICIENT_VALID_SCENES"
    write(
        root / "manifest_replica.json",
        {
            "schema": "mcss.rgbd_replica.data.v1",
            "image_size": list(TARGET_SIZE),
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
            "network_bytes": transferred,
            "budget_bytes": BUDGET,
            "model_forward_run": False,
            "score_computed": False,
            "candidate_lock_sha256": sha(root / "candidate_lock.json"),
        },
    )
    print(status, len(records), "failures", len(failures), "bytes", transferred, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("lock", "prepare"), required=True)
    parser.add_argument("--root", type=Path, default=Path("outputs") / EXPERIMENT)
    args = parser.parse_args()
    (lock if args.stage == "lock" else prepare)(args.root)


if __name__ == "__main__":
    main()
