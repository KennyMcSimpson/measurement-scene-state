"""Dataset loader for prepared scene manifests."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import h5py
import numpy as np
import torch
import torch.nn.functional as functional
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

from mcss.data.manifest import ManifestFrame, SceneManifest, load_manifest
from mcss.geometry import (
    generate_rays,
    intersect_aabb,
    transform_cameras,
    transform_directions,
    transform_points,
)
from mcss.types import Cameras, SceneExample


class ManifestSceneDataset(Dataset[SceneExample]):
    """Turn one or more prepared manifests into context/target scene examples."""

    def __init__(
        self,
        root: str | Path,
        *,
        context_views: int,
        target_views: int,
        image_size: tuple[int, int] | None = None,
        sample_stride: int = 1,
        scene_ids: tuple[str, ...] | None = None,
        spatial_protocol: Literal[
            "legacy_manifest_bounds", "context_local_metric"
        ] = "legacy_manifest_bounds",
        local_bounds_m: tuple[tuple[float, float, float], tuple[float, float, float]] | None = None,
    ) -> None:
        if context_views < 1 or target_views < 1:
            raise ValueError("context_views and target_views must be positive")
        if sample_stride < 1:
            raise ValueError("sample_stride must be positive")
        if spatial_protocol not in {"legacy_manifest_bounds", "context_local_metric"}:
            raise ValueError("unsupported spatial protocol")
        if spatial_protocol == "context_local_metric" and local_bounds_m is None:
            raise ValueError("local_bounds_m is required for context_local_metric")
        self.root = Path(root)
        manifest_paths = _find_manifests(self.root)
        if not manifest_paths:
            raise FileNotFoundError(f"no manifest.json found below {self.root}")
        manifests = tuple(load_manifest(path) for path in manifest_paths)
        if scene_ids is not None:
            invalid_scene_id = any(
                not isinstance(scene_id, str) or not scene_id for scene_id in scene_ids
            )
            if not scene_ids or invalid_scene_id:
                raise ValueError("scene_ids must contain non-empty strings")
            if len(set(scene_ids)) != len(scene_ids):
                raise ValueError("scene_ids must contain unique values")
            by_scene_id: dict[str, SceneManifest] = {}
            for manifest in manifests:
                if manifest.scene_id in by_scene_id:
                    raise ValueError(f"duplicate manifest scene_id: {manifest.scene_id}")
                by_scene_id[manifest.scene_id] = manifest
            missing = [scene_id for scene_id in scene_ids if scene_id not in by_scene_id]
            if missing:
                raise ValueError(f"requested scene_ids not found: {missing}")
            manifests = tuple(by_scene_id[scene_id] for scene_id in scene_ids)
        self.manifests = manifests
        self.context_views = context_views
        self.target_views = target_views
        self.image_size = image_size
        self.spatial_protocol = spatial_protocol
        self.local_bounds_m = local_bounds_m
        self.entries: list[tuple[int, int]] = []
        total_views = context_views + target_views
        for manifest_index, manifest in enumerate(self.manifests):
            if len(manifest.frames) < total_views:
                continue
            max_start = len(manifest.frames) - total_views
            self.entries.extend(
                (manifest_index, start) for start in range(0, max_start + 1, sample_stride)
            )
        if not self.entries:
            raise ValueError("no manifest has enough frames for the requested context/target split")

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> SceneExample:
        manifest_index, start = self.entries[index]
        manifest = self.manifests[manifest_index]
        frames = manifest.frames[start : start + self.context_views + self.target_views]
        context_frames = frames[: self.context_views]
        target_frames = frames[self.context_views :]
        target_size = self.image_size or manifest.image_size
        context_rgb = torch.stack(
            [_load_rgb(manifest, frame, target_size) for frame in context_frames]
        )
        target_rgb = torch.stack(
            [_load_rgb(manifest, frame, target_size) for frame in target_frames]
        )
        context_cameras = _make_cameras(manifest, context_frames, target_size)
        target_cameras = _make_cameras(manifest, target_frames, target_size)
        target_depth = _load_optional_scalar(manifest, target_frames, "depth", target_size)
        target_normal = _load_optional_normal(manifest, target_frames, target_size)
        target_visibility = _load_optional_scalar(
            manifest, target_frames, "visibility", target_size
        )
        if target_visibility is None and target_depth is not None:
            target_visibility = (target_depth > 0).to(target_depth.dtype)
        target_point = _load_optional_vector(manifest, target_frames, "point", target_size)
        if self.spatial_protocol == "context_local_metric":
            if self.local_bounds_m is None:
                raise RuntimeError("context-local protocol is missing local bounds")
            world_to_anchor = torch.linalg.inv(context_cameras.c2w[0])
            context_cameras = transform_cameras(context_cameras, world_to_anchor)
            target_cameras = transform_cameras(target_cameras, world_to_anchor)
            if target_normal is not None:
                normals = target_normal.permute(0, 2, 3, 1)
                target_normal = transform_directions(normals, world_to_anchor).permute(0, 3, 1, 2)
            if target_point is not None:
                points = target_point.permute(0, 2, 3, 1)
                target_point = transform_points(points, world_to_anchor).permute(0, 3, 1, 2)
            bounds = torch.tensor(self.local_bounds_m, dtype=torch.float32)
            target_support = _ray_support(target_cameras, bounds)
        else:
            bounds = torch.tensor(manifest.bounds, dtype=torch.float32)
            target_support = None
        if target_point is None and target_depth is not None:
            origins, directions = generate_rays(target_cameras)
            distances = target_depth[:, 0].unsqueeze(-1)
            point_world = origins + directions * distances
            target_point = point_world.permute(0, 3, 1, 2).contiguous()
        if self.spatial_protocol == "context_local_metric":
            if target_point is None or target_support is None:
                raise ValueError(
                    "context_local_metric requires target depth or point labels to define "
                    "local measurement visibility"
                )
            target_rgb, target_depth, target_normal, target_point, target_visibility = (
                _clip_targets_to_local_measurement(
                    target_rgb,
                    target_depth,
                    target_normal,
                    target_point,
                    target_visibility,
                    target_support,
                    bounds,
                )
            )
        return SceneExample(
            context_rgb=context_rgb,
            context_cameras=context_cameras,
            target_rgb=target_rgb,
            target_cameras=target_cameras,
            bounds=bounds,
            target_depth=target_depth,
            target_normal=target_normal,
            target_point=target_point,
            target_visibility=target_visibility,
            target_support=target_support,
            scene_id=manifest.scene_id,
        )


def _ray_support(cameras: Cameras, bounds: Tensor) -> Tensor:
    origins, directions = generate_rays(cameras)
    _, _, hit = intersect_aabb(origins, directions, bounds)
    return hit.unsqueeze(-3).to(dtype=torch.float32)


def _clip_targets_to_local_measurement(
    target_rgb: Tensor,
    target_depth: Tensor | None,
    target_normal: Tensor | None,
    target_point: Tensor,
    target_visibility: Tensor | None,
    target_support: Tensor,
    bounds: Tensor,
) -> tuple[Tensor, Tensor | None, Tensor | None, Tensor, Tensor]:
    points = target_point.permute(0, 2, 3, 1)
    finite = torch.isfinite(points).all(dim=-1)
    inside = ((points >= bounds[0]) & (points <= bounds[1])).all(dim=-1)
    visible = finite & inside & (target_support[:, 0] > 0.5)
    if target_visibility is not None:
        visible = visible & (target_visibility[:, 0] > 0.5)
    mask = visible.unsqueeze(1)
    target_rgb = torch.where(mask, target_rgb, torch.zeros_like(target_rgb))
    if target_depth is not None:
        target_depth = torch.where(mask, target_depth, torch.zeros_like(target_depth))
    if target_normal is not None:
        target_normal = torch.where(mask, target_normal, torch.zeros_like(target_normal))
    target_point = torch.where(mask, target_point, torch.zeros_like(target_point))
    return target_rgb, target_depth, target_normal, target_point, mask.to(target_rgb.dtype)


def _find_manifests(root: Path) -> list[Path]:
    if root.is_file() and root.name == "manifest.json":
        return [root]
    if (root / "manifest.json").is_file():
        return [root / "manifest.json"]
    direct_scene_manifests = sorted(
        path for path in root.glob("*/manifest.json") if path.is_file()
    )
    if direct_scene_manifests:
        return direct_scene_manifests
    return sorted(path for path in root.rglob("manifest.json") if path.is_file())


def _make_cameras(
    manifest: SceneManifest,
    frames: tuple[ManifestFrame, ...] | list[ManifestFrame],
    image_size: tuple[int, int],
) -> Cameras:
    original_height, original_width = manifest.image_size
    target_height, target_width = image_size
    x_scale = target_width / original_width
    y_scale = target_height / original_height
    intrinsics = []
    poses = []
    for frame in frames:
        matrix = torch.tensor(frame.intrinsics, dtype=torch.float32)
        matrix = matrix.clone()
        matrix[0, 0] *= x_scale
        matrix[0, 2] *= x_scale
        matrix[1, 1] *= y_scale
        matrix[1, 2] *= y_scale
        intrinsics.append(matrix)
        poses.append(torch.tensor(frame.c2w, dtype=torch.float32))
    return Cameras(torch.stack(intrinsics), torch.stack(poses), image_size)


def _load_rgb(manifest: SceneManifest, frame: ManifestFrame, image_size: tuple[int, int]) -> Tensor:
    path = _frame_path(manifest, frame.rgb)
    image = Image.open(path).convert("RGB")
    if (image.height, image.width) != image_size:
        image = image.resize((image_size[1], image_size[0]), Image.Resampling.BICUBIC)
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def _load_optional_scalar(
    manifest: SceneManifest,
    frames: tuple[ManifestFrame, ...] | list[ManifestFrame],
    field: str,
    image_size: tuple[int, int],
) -> Tensor | None:
    paths = [getattr(frame, field) for frame in frames]
    if all(path is None for path in paths):
        return None
    if any(path is None for path in paths):
        raise ValueError(f"mixed missing/present {field} paths in one target window")
    values = [_load_array(_frame_path(manifest, path), field) for path in paths]  # type: ignore[arg-type]
    resized = [_resize_scalar_tensor(value, image_size) for value in values]
    return torch.stack([value.unsqueeze(0) for value in resized])


def _load_optional_normal(
    manifest: SceneManifest,
    frames: tuple[ManifestFrame, ...] | list[ManifestFrame],
    image_size: tuple[int, int],
) -> Tensor | None:
    paths = [frame.normal for frame in frames]
    if all(path is None for path in paths):
        return None
    if any(path is None for path in paths):
        raise ValueError("mixed missing/present normal paths in one target window")
    values = [_load_array(_frame_path(manifest, path), "normal") for path in paths]  # type: ignore[arg-type]
    return torch.stack([_resize_vector_tensor(value, image_size) for value in values])


def _load_optional_vector(
    manifest: SceneManifest,
    frames: tuple[ManifestFrame, ...] | list[ManifestFrame],
    field: str,
    image_size: tuple[int, int],
) -> Tensor | None:
    paths = [getattr(frame, field) for frame in frames]
    if all(path is None for path in paths):
        return None
    if any(path is None for path in paths):
        raise ValueError(f"mixed missing/present {field} paths in one target window")
    values = [_load_array(_frame_path(manifest, path), field) for path in paths]  # type: ignore[arg-type]
    return torch.stack([_resize_point_tensor(value, image_size) for value in values])


def _load_array(path: Path, field: str) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        value = np.load(path)
    elif path.suffix.lower() in {".h5", ".hdf5"}:
        with h5py.File(path, "r") as handle:
            datasets: list[h5py.Dataset] = []
            handle.visititems(
                lambda _name, object_: (
                    datasets.append(object_) if isinstance(object_, h5py.Dataset) else None
                )
            )
            if len(datasets) != 1:
                raise ValueError(f"expected one dataset in {path}")
            value = np.asarray(datasets[0])
    else:
        value = np.asarray(Image.open(path))
    value = np.asarray(value, dtype=np.float32)
    if field in {"depth", "visibility"} and value.ndim != 2:
        raise ValueError(f"{field} file must contain HxW data: {path}")
    if field in {"normal", "point"} and (value.ndim != 3 or value.shape[-1] != 3):
        raise ValueError(f"{field} file must contain HxWx3 data: {path}")
    return value


def _resize_scalar_tensor(value: np.ndarray, image_size: tuple[int, int]) -> Tensor:
    tensor = torch.from_numpy(value).float()[None, None]
    if tuple(value.shape) != image_size:
        tensor = functional.interpolate(tensor, size=image_size, mode="nearest")
    return tensor[0, 0]


def _resize_vector_tensor(value: np.ndarray, image_size: tuple[int, int]) -> Tensor:
    tensor = torch.from_numpy(value).float().permute(2, 0, 1)[None]
    if tuple(value.shape[:2]) != image_size:
        tensor = functional.interpolate(
            tensor, size=image_size, mode="bilinear", align_corners=True
        )
    tensor = tensor[0].permute(1, 2, 0)
    norm = torch.linalg.vector_norm(tensor, dim=-1, keepdim=True)
    return torch.where(
        norm > 1e-8, tensor / norm.clamp_min(1e-8), torch.zeros_like(tensor)
    ).permute(2, 0, 1)


def _resize_point_tensor(value: np.ndarray, image_size: tuple[int, int]) -> Tensor:
    tensor = torch.from_numpy(value).float().permute(2, 0, 1)[None]
    if tuple(value.shape[:2]) != image_size:
        tensor = functional.interpolate(
            tensor, size=image_size, mode="bilinear", align_corners=True
        )
    return tensor[0]


def _frame_path(manifest: SceneManifest, relative_path: str) -> Path:
    if manifest.root is None:
        raise ValueError("manifest root is required to load frame files")
    path = manifest.root / relative_path
    if not path.is_file():
        raise FileNotFoundError(path)
    return path
