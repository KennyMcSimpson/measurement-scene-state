"""Conversion of rendered Replica RGB-D trajectories to the common manifest."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
from PIL import Image

from mcss.data.manifest import ManifestFrame, SceneManifest, save_manifest


def prepare_replica(
    raw_root: str | Path,
    output_root: str | Path,
    *,
    frame_limit: int | None = None,
    image_size: tuple[int, int] | None = None,
    depth_scale: float | None = None,
) -> Path:
    """Convert a pre-rendered Replica trajectory (RGB, depth, traj.txt, K) to a manifest.

    The official Replica release contains assets, not a standard camera trajectory. This function
    intentionally requires an already rendered RGB-D sequence instead of silently inventing
    poses. A compatible layout has `rgb/frameNNNNNN.jpg`, `depth/depthNNNNNN.png`, `traj.txt`,
    and `cam_params.json` beneath ``raw_root``.
    """

    raw = Path(raw_root)
    if not raw.is_dir():
        raise FileNotFoundError(f"Replica raw root does not exist: {raw}")
    rgb_files = _indexed_files(raw, ("rgb", "results"), ("*.jpg", "*.jpeg", "*.png"))
    depth_files = _indexed_files(raw, ("depth", "results"), ("*.png", "*.npy"))
    shared_ids = sorted(set(rgb_files) & set(depth_files))
    if not shared_ids:
        raise FileNotFoundError(
            "Replica preparation requires a rendered RGB-D trajectory. Expected matching RGB and "
            "depth files beneath rgb/depth or results; official mesh assets need "
            "Habitat-Sim rendering first."
        )
    trajectories = _load_trajectories(_first_existing(raw, ("traj.txt", "results/traj.txt")))
    parameters = _load_camera_parameters(
        _first_existing(raw, ("cam_params.json", "results/cam_params.json"))
    )
    if frame_limit is not None:
        if frame_limit < 1:
            raise ValueError("frame_limit must be positive")
        shared_ids = shared_ids[:frame_limit]
    if len(trajectories) < len(shared_ids):
        raise ValueError("traj.txt has fewer poses than matched RGB-D frames")

    scene_dir = Path(output_root) / _safe_scene_name(raw.name)
    (scene_dir / "rgb").mkdir(parents=True, exist_ok=True)
    (scene_dir / "depth").mkdir(parents=True, exist_ok=True)
    frames: list[ManifestFrame] = []
    sampled_points: list[np.ndarray] = []
    for pose_index, frame_id in enumerate(shared_ids):
        rgb_image = Image.open(rgb_files[frame_id]).convert("RGB")
        original_width, original_height = rgb_image.size
        target_size = image_size or (original_height, original_width)
        if target_size != (original_height, original_width):
            rgb_image = rgb_image.resize((target_size[1], target_size[0]), Image.Resampling.BICUBIC)
        rgb_relative = f"rgb/{frame_id:06d}.png"
        rgb_image.save(scene_dir / rgb_relative)
        raw_depth = _load_depth(depth_files[frame_id])
        scale = depth_scale or float(parameters.get("depth_scale", 1000.0))
        if scale <= 0:
            raise ValueError("depth_scale must be positive")
        depth = raw_depth.astype(np.float32) / scale
        depth[~np.isfinite(depth) | (depth <= 0)] = 0.0
        if target_size != raw_depth.shape[:2]:
            depth = _resize_scalar(depth, target_size)
        depth_relative = f"depth/{frame_id:06d}.npy"
        np.save(scene_dir / depth_relative, depth)
        intrinsics = _scaled_intrinsics(parameters, (original_height, original_width), target_size)
        pose = trajectories[pose_index]
        frames.append(
            ManifestFrame(
                frame_id=frame_id,
                rgb=rgb_relative,
                intrinsics=intrinsics.tolist(),
                c2w=pose.tolist(),
                depth=depth_relative,
            )
        )
        sampled_points.append(_backproject_depth(depth, intrinsics, pose, stride=16))
    manifest = SceneManifest(
        scene_id=_safe_scene_name(raw.name),
        image_size=target_size,
        bounds=_estimate_bounds(sampled_points),
        frames=tuple(frames),
        root=scene_dir,
    )
    return save_manifest(manifest, scene_dir / "manifest.json")


def _indexed_files(
    root: Path, directories: tuple[str, ...], patterns: tuple[str, ...]
) -> dict[int, Path]:
    result: dict[int, Path] = {}
    for directory in directories:
        candidate = root / directory
        if not candidate.is_dir():
            continue
        for pattern in patterns:
            for path in candidate.glob(pattern):
                frame_id = _frame_id(path.name)
                if frame_id is not None:
                    result.setdefault(frame_id, path)
    return result


def _frame_id(name: str) -> int | None:
    matches = re.findall(r"(\d+)", name)
    return None if not matches else int(matches[-1])


def _first_existing(root: Path, candidates: tuple[str, ...]) -> Path:
    for candidate in candidates:
        path = root / candidate
        if path.is_file():
            return path
    raise FileNotFoundError(f"none of the required files exist: {candidates}")


def _load_trajectories(path: Path) -> np.ndarray:
    values = np.loadtxt(path, dtype=np.float32)
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if values.size % 16 != 0:
        raise ValueError("traj.txt must contain a whole number of 4x4 matrices")
    poses = values.reshape(-1, 4, 4)
    if not np.allclose(poses[:, 3], [0.0, 0.0, 0.0, 1.0], atol=1e-4):
        raise ValueError("Replica trajectory poses must be homogeneous c2w matrices")
    return poses


def _load_camera_parameters(path: Path) -> dict[str, float]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("cam_params.json must contain an object")
    nested = data.get("camera")
    values = nested if isinstance(nested, dict) else data
    required = ("fx", "fy", "cx", "cy")
    if any(name not in values for name in required):
        raise ValueError("cam_params.json must contain fx, fy, cx, and cy")
    return {name: float(values[name]) for name in (*required, "depth_scale") if name in values}


def _load_depth(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        array = np.load(path)
    else:
        array = np.asarray(Image.open(path))
    if array.ndim != 2:
        raise ValueError(f"depth image must be one-channel: {path}")
    return array


def _scaled_intrinsics(
    parameters: dict[str, float], original_size: tuple[int, int], target_size: tuple[int, int]
) -> np.ndarray:
    original_height, original_width = original_size
    target_height, target_width = target_size
    x_scale = target_width / original_width
    y_scale = target_height / original_height
    return np.array(
        [
            [parameters["fx"] * x_scale, 0.0, parameters["cx"] * x_scale],
            [0.0, parameters["fy"] * y_scale, parameters["cy"] * y_scale],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def _resize_scalar(values: np.ndarray, image_size: tuple[int, int]) -> np.ndarray:
    image = Image.fromarray(values.astype(np.float32), mode="F")
    return np.asarray(
        image.resize((image_size[1], image_size[0]), Image.Resampling.NEAREST), dtype=np.float32
    ).copy()


def _backproject_depth(
    depth: np.ndarray, intrinsics: np.ndarray, c2w: np.ndarray, *, stride: int
) -> np.ndarray:
    y, x = np.mgrid[0 : depth.shape[0] : stride, 0 : depth.shape[1] : stride]
    distances = depth[::stride, ::stride]
    valid = distances > 0
    if not valid.any():
        return np.empty((0, 3), dtype=np.float32)
    pixels = np.stack((x[valid], y[valid], np.ones(valid.sum())), axis=-1).astype(np.float32)
    directions = pixels @ np.linalg.inv(intrinsics).T
    directions /= np.linalg.norm(directions, axis=-1, keepdims=True).clip(min=1e-8)
    world_directions = directions @ c2w[:3, :3].T
    return c2w[:3, 3] + world_directions * distances[valid, None]


def _estimate_bounds(
    point_sets: list[np.ndarray],
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    points = [point_set for point_set in point_sets if len(point_set)]
    if not points:
        return ((-1.2, -1.2, -1.2), (1.2, 1.2, 1.2))
    stacked = np.concatenate(points, axis=0)
    lower = np.quantile(stacked, 0.01, axis=0) - 0.2
    upper = np.quantile(stacked, 0.99, axis=0) + 0.2
    return tuple(lower.astype(float)), tuple(upper.astype(float))  # type: ignore[return-value]


def _safe_scene_name(name: str) -> str:
    if not name or any(character in name for character in "\\/"):
        raise ValueError("scene root must have a simple directory name")
    return name
