"""Hypersim subset conversion using official RGB, camera, depth, and normal files."""

from __future__ import annotations

import csv
import re
from pathlib import Path

import h5py
import numpy as np
from PIL import Image

from mcss.data.manifest import ManifestFrame, SceneManifest, save_manifest
from mcss.data.replica import _backproject_depth, _estimate_bounds, _resize_scalar


def convert_hypersim_pose(
    orientation: np.ndarray, position: np.ndarray, *, meters_per_asset_unit: float
) -> np.ndarray:
    """Convert Hypersim OpenGL camera axes (x-right/y-up/z-back) to OpenCV c2w axes."""

    rotation = np.asarray(orientation, dtype=np.float32)
    translation = np.asarray(position, dtype=np.float32)
    if rotation.shape != (3, 3) or translation.shape != (3,):
        raise ValueError("Hypersim orientation and position require shapes [3, 3] and [3]")
    if not np.isfinite(rotation).all() or not np.isfinite(translation).all():
        raise ValueError("Hypersim pose values must be finite")
    if meters_per_asset_unit <= 0:
        raise ValueError("meters_per_asset_unit must be positive")
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = rotation @ np.diag([1.0, -1.0, -1.0]).astype(np.float32)
    pose[:3, 3] = translation * meters_per_asset_unit
    return pose


def prepare_hypersim(
    raw_root: str | Path,
    output_root: str | Path,
    *,
    scenes: list[str] | None = None,
    camera: str = "cam_00",
    image_size: tuple[int, int] | None = None,
    frame_stride: int = 1,
    frame_limit: int | None = None,
    fov_x_degrees: float = 60.0,
) -> list[Path]:
    """Convert a selectively downloaded Hypersim subset to common manifests.

    The official preview images are 1024x768 with a 60 degree horizontal field of view. The
    default can be overridden if a future official camera calibration export is supplied.
    """

    if not re.fullmatch(r"cam_\d{2}", camera):
        raise ValueError("camera must match cam_DD")
    if frame_stride < 1:
        raise ValueError("frame_stride must be positive")
    root = Path(raw_root)
    available = [
        path
        for path in root.iterdir()
        if path.is_dir() and re.fullmatch(r"ai_\d{3}_\d{3}", path.name)
    ]
    if re.fullmatch(r"ai_\d{3}_\d{3}", root.name):
        available = [root]
    requested = set(scenes) if scenes is not None else {path.name for path in available}
    selected = [path for path in available if path.name in requested]
    missing = requested - {path.name for path in selected}
    if missing:
        raise FileNotFoundError(f"Hypersim scenes not found below raw root: {sorted(missing)}")
    return [
        _prepare_scene(
            scene, Path(output_root), camera, image_size, frame_stride, frame_limit, fov_x_degrees
        )
        for scene in sorted(selected)
    ]


def _prepare_scene(
    scene: Path,
    output_root: Path,
    camera: str,
    image_size: tuple[int, int] | None,
    frame_stride: int,
    frame_limit: int | None,
    fov_x_degrees: float,
) -> Path:
    orientations = _read_hdf5(scene / "_detail" / camera / "camera_keyframe_orientations.hdf5")
    positions = _read_hdf5(scene / "_detail" / camera / "camera_keyframe_positions.hdf5")
    scale = _meters_per_asset_unit(scene / "_detail" / "metadata_scene.csv")
    rgb_files = _frame_files(scene / "images" / f"scene_{camera}_final_preview", ".color.jpg")
    depth_files = _frame_files(
        scene / "images" / f"scene_{camera}_geometry_hdf5", ".depth_meters.hdf5"
    )
    normal_files = _frame_files(
        scene / "images" / f"scene_{camera}_geometry_hdf5", ".normal_world.hdf5"
    )
    frame_ids = sorted(set(rgb_files) & set(depth_files))[::frame_stride]
    if frame_limit is not None:
        frame_ids = frame_ids[:frame_limit]
    if not frame_ids:
        raise FileNotFoundError(
            f"no matching preview RGB and depth frames found for {scene.name}/{camera}"
        )
    if max(frame_ids) >= len(orientations) or max(frame_ids) >= len(positions):
        raise ValueError("Hypersim camera trajectory is shorter than selected frame ids")

    destination = output_root / scene.name
    (destination / "rgb").mkdir(parents=True, exist_ok=True)
    (destination / "depth").mkdir(parents=True, exist_ok=True)
    (destination / "normal").mkdir(parents=True, exist_ok=True)
    frames: list[ManifestFrame] = []
    point_sets: list[np.ndarray] = []
    for frame_id in frame_ids:
        image = Image.open(rgb_files[frame_id]).convert("RGB")
        original_size = (image.height, image.width)
        target_size = image_size or original_size
        if target_size != original_size:
            image = image.resize((target_size[1], target_size[0]), Image.Resampling.BICUBIC)
        rgb_relative = f"rgb/{frame_id:06d}.png"
        image.save(destination / rgb_relative)
        depth = _read_hdf5(depth_files[frame_id]).astype(np.float32)
        if depth.ndim != 2:
            raise ValueError(f"Hypersim depth must be HxW: {depth_files[frame_id]}")
        if target_size != depth.shape:
            depth = _resize_scalar(depth, target_size)
        depth[~np.isfinite(depth) | (depth <= 0)] = 0.0
        depth_relative = f"depth/{frame_id:06d}.npy"
        np.save(destination / depth_relative, depth)
        normal_relative = None
        if frame_id in normal_files:
            normal = _read_hdf5(normal_files[frame_id]).astype(np.float32)
            if normal.shape[-1] != 3:
                raise ValueError(f"Hypersim normal must end in channel 3: {normal_files[frame_id]}")
            if target_size != normal.shape[:2]:
                normal = _resize_vector(normal, target_size)
            normal = _normalize_vectors(normal)
            normal_relative = f"normal/{frame_id:06d}.npy"
            np.save(destination / normal_relative, normal)
        intrinsics = _intrinsics_for_size(target_size, fov_x_degrees)
        pose = convert_hypersim_pose(
            orientations[frame_id], positions[frame_id], meters_per_asset_unit=scale
        )
        point_sets.append(_backproject_depth(depth, intrinsics, pose, stride=16))
        frames.append(
            ManifestFrame(
                frame_id=frame_id,
                rgb=rgb_relative,
                intrinsics=intrinsics.tolist(),
                c2w=pose.tolist(),
                depth=depth_relative,
                normal=normal_relative,
            )
        )
    manifest = SceneManifest(
        scene_id=scene.name,
        image_size=target_size,
        bounds=_estimate_bounds(point_sets),
        frames=tuple(frames),
        root=destination,
    )
    return save_manifest(manifest, destination / "manifest.json")


def _read_hdf5(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)
    with h5py.File(path, "r") as handle:
        datasets: list[h5py.Dataset] = []
        handle.visititems(
            lambda _name, object_: (
                datasets.append(object_) if isinstance(object_, h5py.Dataset) else None
            )
        )
        if len(datasets) != 1:
            raise ValueError(f"expected exactly one dataset in {path}, found {len(datasets)}")
        return np.asarray(datasets[0])


def _frame_files(directory: Path, suffix: str) -> dict[int, Path]:
    if not directory.is_dir():
        return {}
    result: dict[int, Path] = {}
    pattern = re.compile(r"frame\.(\d{4})\.")
    for path in directory.glob(f"*{suffix}"):
        match = pattern.search(path.name)
        if match:
            result[int(match.group(1))] = path
    return result


def _meters_per_asset_unit(path: Path) -> float:
    if not path.is_file():
        return 1.0
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.reader(handle):
            normalized = [cell.strip() for cell in row]
            if any(cell == "meters_per_asset_unit" for cell in normalized):
                for cell in reversed(normalized):
                    try:
                        value = float(cell)
                    except ValueError:
                        continue
                    if value > 0:
                        return value
    return 1.0


def _intrinsics_for_size(image_size: tuple[int, int], fov_x_degrees: float) -> np.ndarray:
    height, width = image_size
    focal = (width - 1) / (2.0 * np.tan(np.deg2rad(fov_x_degrees) / 2.0))
    return np.array(
        [[focal, 0.0, (width - 1) / 2.0], [0.0, focal, (height - 1) / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )


def _resize_vector(values: np.ndarray, image_size: tuple[int, int]) -> np.ndarray:
    channels = [_resize_scalar(values[..., channel], image_size) for channel in range(3)]
    return np.stack(channels, axis=-1)


def _normalize_vectors(values: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(values, axis=-1, keepdims=True)
    valid = norm > 1e-8
    return np.where(valid, values / np.clip(norm, 1e-8, None), 0.0).astype(np.float32)
