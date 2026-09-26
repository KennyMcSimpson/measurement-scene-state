"""Bounded, train-only Hypersim preparation for the frozen engineering pilot."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import time
import zipfile
from pathlib import Path

import h5py
import numpy as np
import requests
import torch
import torch.nn.functional as F
from PIL import Image

from mcss.data.download import HTTPRangeReader
from mcss.data.hypersim import convert_hypersim_pose
from mcss.mechanism_pilot.calibration import calibrated_intrinsics, published_scene_metadata

SCENES = ("ai_001_001", "ai_002_001", "ai_003_001")
ROLES = {
    "context_a": [0, 1, 2],
    "context_b": [0, 3, 4],
    "stream": [5, 6],
    "query": [12, 13, 14, 15],
    "ignored": [7, 8, 9, 10, 11],
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


class BoundedRangeReader(HTTPRangeReader):
    """Require exact range responses, streaming bounds and immutable remote identity."""

    def __init__(self, url, session, timeout_seconds=60, byte_budget=2 * 1024**3):
        super().__init__(url, session, timeout_seconds)
        self.byte_budget = byte_budget
        self.transferred = 0
        self.shared_budget = None
        response = session.head(url, allow_redirects=True, timeout=timeout_seconds)
        response.raise_for_status()
        self.etag = response.headers.get("ETag")
        response.close()
        if not self.etag:
            raise OSError("Remote archive lacks ETag")

    def read(self, size=-1):
        if size == 0 or self.offset >= self.size:
            return b""
        size = self.size - self.offset if size < 0 else min(size, self.size - self.offset)
        if size > 64 * 1024**2 or self.transferred + size > self.byte_budget:
            raise OSError("Range request exceeds bounded download budget")
        if self.shared_budget is not None:
            if self.shared_budget[0] + size > 2 * 1024**3:
                raise OSError("Shared transfer budget exceeded")
            self.shared_budget[0] += size
        end = self.offset + size - 1
        with self.session.get(
            self.url,
            headers={"Range": f"bytes={self.offset}-{end}", "If-Match": self.etag},
            stream=True,
            timeout=self.timeout_seconds,
        ) as response:
            response.raise_for_status()
            expected = f"bytes {self.offset}-{end}/{self.size}"
            if response.status_code != 206 or response.headers.get("Content-Range") != expected:
                raise OSError("Server did not honor exact range; full ZIP fallback prohibited")
            if response.headers.get("ETag") != self.etag:
                raise OSError("Archive changed during download")
            chunks, total = [], 0
            for chunk in response.iter_content(1024**2):
                total += len(chunk)
                if total > size:
                    raise OSError("Range response exceeded requested bytes")
                chunks.append(chunk)
            if total != size:
                raise OSError("Truncated range response")
        self.offset += size
        self.transferred += size
        return b"".join(chunks)


def select_members(scene, infos):
    if scene not in SCENES:
        raise ValueError("Only frozen train scene whitelist permitted")
    names = {x.filename: x for x in infos}
    prefix = f"{scene}/"
    rgb, depth = {}, {}
    for name in names:
        match = re.fullmatch(
            re.escape(prefix) + r"images/scene_cam_00_final_preview/frame\.(\d{4})\.color\.jpg",
            name,
        )
        if match:
            rgb[int(match[1])] = name
        match = re.fullmatch(
            re.escape(prefix)
            + r"images/scene_cam_00_geometry_hdf5/frame\.(\d{4})\.depth_meters\.hdf5",
            name,
        )
        if match:
            depth[int(match[1])] = name
    ids = sorted(rgb.keys() & depth.keys())[:16]
    if len(ids) != 16:
        raise ValueError("Insufficient common RGB/depth frames; no replacement allowed")
    metadata = [prefix + "_detail/metadata_scene.csv"] + [
        prefix + f"_detail/cam_00/camera_keyframe_{kind}.hdf5"
        for kind in ("frame_indices", "positions", "orientations")
    ]
    selected = metadata + [name for i in ids for name in (rgb[i], depth[i])]
    selected.append(prefix + f"images/scene_cam_00_geometry_hdf5/frame.{ids[0]:04d}.position.hdf5")
    if any(name not in names for name in selected):
        raise ValueError("Required camera metadata missing")
    compressed = sum(names[n].compress_size for n in selected)
    uncompressed = sum(names[n].file_size for n in selected)
    if max(compressed, uncompressed) > 1024**3:
        raise ValueError("Selected members exceed 1 GiB scene budget")
    return {
        "scene_id": scene,
        "frame_ids": ids,
        "members": selected,
        "compressed_bytes": compressed,
        "uncompressed_bytes": uncompressed,
        "member_metadata": [
            {"name": n, "crc32": names[n].CRC, "bytes": names[n].file_size} for n in selected
        ],
    }


def resize_rgb_depth(rgb, depth, image_size=(128, 160)):
    rgb = torch.tensor(np.asarray(rgb).copy(), dtype=torch.float32).permute(2, 0, 1)[None] / 255
    rgb = F.interpolate(rgb, size=image_size, mode="bilinear", align_corners=False)
    depth = torch.tensor(np.asarray(depth).copy(), dtype=torch.float32)[None, None]
    valid = torch.isfinite(depth) & (depth > 0)
    numerator = F.interpolate(
        torch.where(valid, depth, 0), size=image_size, mode="bilinear", align_corners=False
    )
    weight = F.interpolate(valid.float(), size=image_size, mode="bilinear", align_corners=False)
    depth = torch.where(weight > 0, numerator / weight.clamp_min(1e-12), 0)
    return (rgb[0].permute(1, 2, 0).numpy() * 255).round().clip(0, 255).astype(np.uint8), depth[
        0, 0
    ].numpy()


def check_geometry(scene, depth, position_asset_units, pose, intrinsics, scale):
    y, x = np.indices(depth.shape)
    pixels = np.stack([x, y, np.ones_like(x)], -1)
    rays = pixels @ np.linalg.inv(intrinsics).T
    rays /= np.linalg.norm(rays, axis=-1, keepdims=True)
    points = (rays * depth[..., None]) @ pose[:3, :3].T + pose[:3, 3]
    truth = position_asset_units * scale
    valid = np.isfinite(truth).all(-1) & np.isfinite(points).all(-1) & (depth > 0)
    if not valid.any():
        raise ValueError("No valid geometry calibration pixels")
    errors = np.linalg.norm(points[valid] - truth[valid], axis=-1)
    report = {
        "scene_id": scene,
        "valid_pixels": int(valid.sum()),
        "median_error_m": float(np.median(errors)),
        "p95_error_m": float(np.quantile(errors, 0.95)),
        "maximum_error_m": float(errors.max()),
        "predeclared_p95_limit_m": 0.01,
    }
    if report["p95_error_m"] > 0.01:
        raise ValueError(f"Native geometry calibration failed: {report}")
    return report


def prepare(output, calibration, partitions):
    output, calibration = Path(output), Path(calibration)
    if output.exists():
        raise FileExistsError(
            "Refusing overwrite; retain failed attempts and use explicit recovery"
        )
    with Path(partitions).open() as handle:
        split = {r["scene_name"]: r["protocol_partition"] for r in csv.DictReader(handle)}
    if any(split.get(s) != "train" for s in SCENES):
        raise ValueError("Frozen scene is not train")
    rows = published_scene_metadata(calibration)
    output.mkdir(parents=True)
    sessions, readers, archives, plans = [], [], [], []
    shared_budget = [0]
    try:
        for scene in SCENES:
            session = requests.Session()
            url = f"https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/{scene}.zip"
            reader = BoundedRangeReader(url, session)
            reader.shared_budget = shared_budget
            archive = zipfile.ZipFile(reader)
            plan = select_members(scene, archive.infolist())
            plan.update(url=url, etag=reader.etag, archive_bytes=reader.size)
            sessions.append(session)
            readers.append(reader)
            archives.append(archive)
            plans.append(plan)
        if sum(p["uncompressed_bytes"] + p["compressed_bytes"] for p in plans) > 2 * 1024**3:
            raise ValueError("Global selected data budget exceeds 2 GiB")
        lock = {
            "schema": "mcss.small_training.data_lock.v1",
            "created_unix": time.time(),
            "split": "train",
            "physical_identity": "UNVERIFIED_NO_CROSS_SPLIT",
            "scenes": plans,
            "image_size": [128, 160],
            "roles_by_index": ROLES,
            "calibration_sha256": sha(calibration),
            "partitions_sha256": sha(partitions),
            "rgb_source": "historical_preview_jpeg_engineering_only",
            "resampling": "bilinear_half_pixel_rgb_and_valid_weight_depth",
            "depth": "ray_distance_meters",
        }
        write_json(output / "data_lock.json", lock)  # BEFORE first media fetch/decode
        print("LOCKED " + str(output / "data_lock.json"), flush=True)
        scenes, hashes, geometry_checks = [], {}, []
        with (output / "access.jsonl").open("x") as access:

            def log(scene, member, operation):
                access.write(
                    json.dumps(
                        {
                            "unix": time.time(),
                            "scene": scene,
                            "split": "train",
                            "member": member,
                            "operation": operation,
                            "purpose": "engineering_training_preparation",
                        }
                    )
                    + "\n"
                )
                access.flush()

            for plan, archive, reader in zip(plans, archives, readers, strict=True):
                scene = plan["scene_id"]
                raw = output / "raw"
                for member in plan["members"]:
                    destination = raw / member
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    payload = archive.read(member)  # zipfile checks CRC
                    destination.write_bytes(payload)
                    hashes[str(destination.relative_to(output))] = sha(destination)
                    log(scene, member, "download_selected_member")
                source = raw / scene
                with (source / "_detail/metadata_scene.csv").open() as handle:
                    units = {
                        r["parameter_name"]: r["parameter_value"] for r in csv.DictReader(handle)
                    }
                scale = float(units["meters_per_asset_unit"])
                if not np.isclose(
                    scale, float(rows[scene]["settings_units_info_meters_scale"]), rtol=1e-6
                ):
                    raise ValueError("Published/archive metric scale mismatch")

                def hdf(path, scene=scene, raw=raw):
                    log(scene, str(path.relative_to(raw)), "decode_hdf5")
                    with h5py.File(path, "r") as handle:
                        return handle["dataset"][:]

                indices = hdf(source / "_detail/cam_00/camera_keyframe_frame_indices.hdf5").reshape(
                    -1
                )
                positions = hdf(source / "_detail/cam_00/camera_keyframe_positions.hdf5")
                rotations = hdf(source / "_detail/cam_00/camera_keyframe_orientations.hdf5")
                mapping = {int(x): i for i, x in enumerate(indices)}
                if len(mapping) != len(indices):
                    raise ValueError("Duplicate camera keyframes")
                frames = []
                dest = output / "prepared" / scene
                dest.mkdir(parents=True)
                for fid in plan["frame_ids"]:
                    rgb_path = (
                        source / f"images/scene_cam_00_final_preview/frame.{fid:04d}.color.jpg"
                    )
                    depth_path = (
                        source
                        / f"images/scene_cam_00_geometry_hdf5/frame.{fid:04d}.depth_meters.hdf5"
                    )
                    log(scene, str(rgb_path.relative_to(raw)), "decode_train_rgb")
                    with Image.open(rgb_path) as image:
                        rgb = np.asarray(image.convert("RGB"))
                    depth = hdf(depth_path)
                    if rgb.shape[:2] != depth.shape:
                        raise ValueError("RGB/depth dimensions disagree")
                    if fid == plan["frame_ids"][0]:
                        position = hdf(
                            source
                            / f"images/scene_cam_00_geometry_hdf5/frame.{fid:04d}.position.hdf5"
                        )
                        pose_native = convert_hypersim_pose(
                            rotations[mapping[fid]],
                            positions[mapping[fid]],
                            meters_per_asset_unit=scale,
                        )
                        geometry_checks.append(
                            check_geometry(
                                scene,
                                depth,
                                position,
                                pose_native,
                                calibrated_intrinsics(rows[scene], depth.shape),
                                scale,
                            )
                        )
                    rgb, depth = resize_rgb_depth(rgb, depth)
                    rgb_out, depth_out = dest / f"{fid:04d}.png", dest / f"{fid:04d}.npy"
                    Image.fromarray(rgb).save(rgb_out)
                    np.save(depth_out, depth)
                    for p in (rgb_out, depth_out):
                        hashes[str(p.relative_to(output))] = sha(p)
                    index = mapping[fid]
                    pose = convert_hypersim_pose(
                        rotations[index], positions[index], meters_per_asset_unit=scale
                    )
                    frames.append(
                        {
                            "frame_id": fid,
                            "rgb": str(rgb_out.resolve()),
                            "depth": str(depth_out.resolve()),
                            "intrinsics": calibrated_intrinsics(rows[scene], (128, 160)).tolist(),
                            "c2w": pose.tolist(),
                        }
                    )
                scenes.append(
                    {
                        "scene_id": scene,
                        "split": "train",
                        "physical_scene_id": None,
                        "frames": frames,
                        "roles": {
                            k: [plan["frame_ids"][i] for i in ids] for k, ids in ROLES.items()
                        },
                    }
                )
                print(f"PREPARED {scene}: {reader.transferred} HTTP bytes", flush=True)
        manifest = {
            "schema": "mcss.small_training.data.v1",
            "image_size": [128, 160],
            "data_lock_sha256": sha(output / "data_lock.json"),
            "scenes": scenes,
            "depth_semantics": "ray_distance_meters",
            "scientific_confirmation": False,
        }
        write_json(output / "geometry_checks.json", geometry_checks)
        write_json(output / "manifest.json", manifest)
        write_json(output / "hashes.json", hashes)
        write_json(
            output / "download_cost.json",
            {
                "network_bytes": sum(r.transferred for r in readers),
                "raw_bytes": sum(p["uncompressed_bytes"] for p in plans),
                "full_archives_downloaded": False,
            },
        )
        return manifest
    finally:
        for obj in archives + readers + sessions:
            obj.close()
