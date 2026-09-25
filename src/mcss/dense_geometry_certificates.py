"""Known-camera consistency certificates for dense supplier depth."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Self

import numpy as np

from mcss.geometry_supplier import GeometrySupplierResult

DENSE_CERTIFICATE_CACHE_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class DenseCertificateConfig:
    grid_stride: int = 4
    min_depth_m: float = 0.10
    max_depth_m: float = 20.0
    depth_absolute_tolerance_m: float = 0.10
    depth_relative_tolerance: float = 0.05
    roundtrip_threshold_px: float = 2.0
    surface_half_width_m: float = 0.05
    free_space_margin_m: float = 0.05
    minimum_supporting_views: int = 1
    maximum_conflicting_views: int = 0

    def __post_init__(self) -> None:
        if self.grid_stride < 1:
            raise ValueError("grid_stride must be positive")
        if not 0 < self.min_depth_m < self.max_depth_m:
            raise ValueError("depth range must be positive and increasing")
        values = (
            self.depth_absolute_tolerance_m,
            self.depth_relative_tolerance,
            self.roundtrip_threshold_px,
            self.surface_half_width_m,
            self.free_space_margin_m,
        )
        if min(values) < 0:
            raise ValueError("certificate tolerances must be nonnegative")
        if self.minimum_supporting_views < 1 or self.maximum_conflicting_views < 0:
            raise ValueError("support/conflict counts are invalid")


@dataclass
class DenseGeometryCertificates:
    frame_ids: tuple[int, ...]
    pixel_xy: np.ndarray
    depth_along_ray: np.ndarray
    world_points: np.ndarray
    raw_confidence: np.ndarray
    raw_valid: np.ndarray
    certified: np.ndarray
    support_count: np.ndarray
    conflict_count: np.ndarray
    occlusion_count: np.ndarray
    provenance: np.ndarray
    free_end_m: np.ndarray
    surface_near_m: np.ndarray
    surface_far_m: np.ndarray
    risk: np.ndarray

    def __post_init__(self) -> None:
        scalar_shape = self.depth_along_ray.shape
        if len(scalar_shape) != 3 or scalar_shape[0] != len(self.frame_ids):
            raise ValueError("dense certificate scalar fields require shape [V, Gh, Gw]")
        scalar_fields = (
            self.raw_confidence,
            self.raw_valid,
            self.certified,
            self.support_count,
            self.conflict_count,
            self.occlusion_count,
            self.free_end_m,
            self.surface_near_m,
            self.surface_far_m,
            self.risk,
        )
        if any(value.shape != scalar_shape for value in scalar_fields):
            raise ValueError("dense certificate scalar field shapes must match")
        if self.pixel_xy.shape != (*scalar_shape, 2):
            raise ValueError("pixel_xy must have shape [V, Gh, Gw, 2]")
        if self.world_points.shape != (*scalar_shape, 3):
            raise ValueError("world_points must have shape [V, Gh, Gw, 3]")
        if self.provenance.shape != (*scalar_shape, len(self.frame_ids)):
            raise ValueError("provenance must have one trailing source-view dimension")
        for value in (
            self.pixel_xy,
            self.depth_along_ray,
            self.world_points,
            self.raw_confidence,
            self.free_end_m,
            self.surface_near_m,
            self.surface_far_m,
            self.risk,
        ):
            if not np.isfinite(value).all():
                raise ValueError("dense certificate numeric fields must be finite")
        if np.any((self.risk < 0) | (self.risk > 1)):
            raise ValueError("risk must be in [0, 1]")

    def canonical(self) -> Self:
        order = np.argsort(np.asarray(self.frame_ids), kind="stable")
        if np.array_equal(order, np.arange(len(order))):
            return self
        provenance = self.provenance[order][..., order]
        values = {
            name: getattr(self, name)[order]
            for name in (
                "pixel_xy",
                "depth_along_ray",
                "world_points",
                "raw_confidence",
                "raw_valid",
                "certified",
                "support_count",
                "conflict_count",
                "occlusion_count",
                "free_end_m",
                "surface_near_m",
                "surface_far_m",
                "risk",
            )
        }
        return type(self)(
            tuple(self.frame_ids[int(index)] for index in order),
            provenance=provenance,
            **values,
        )


def build_dense_certificates(
    supplier: GeometrySupplierResult, config: DenseCertificateConfig
) -> DenseGeometryCertificates:
    supplier = supplier.canonical()
    view_count, height, width = supplier.depth_along_ray.shape
    y_values = np.arange(0, height, config.grid_stride, dtype=np.int64)
    x_values = np.arange(0, width, config.grid_stride, dtype=np.int64)
    grid_y, grid_x = np.meshgrid(y_values, x_values, indexing="ij")
    grid_shape = grid_y.shape
    pixel_grid = np.stack((grid_x, grid_y), axis=-1).astype(np.float32)
    pixel_xy = np.broadcast_to(pixel_grid, (view_count, *grid_shape, 2)).copy()

    depth = supplier.depth_along_ray[:, grid_y, grid_x]
    confidence = supplier.confidence[:, grid_y, grid_x]
    raw_mask = supplier.valid_mask[:, grid_y, grid_x]
    raw_valid = (
        raw_mask
        & np.isfinite(depth)
        & (depth >= config.min_depth_m)
        & (depth <= config.max_depth_m)
    )
    world_points = np.zeros((view_count, *grid_shape, 3), dtype=np.float32)
    directions = []
    for view in range(view_count):
        camera_directions = _camera_directions(pixel_grid, supplier.intrinsics[view])
        directions.append(camera_directions)
        world_directions = camera_directions @ supplier.c2w[view, :3, :3].T
        world_points[view] = (
            supplier.c2w[view, :3, 3]
            + world_directions * depth[view][..., None]
        )

    support = np.zeros((view_count, *grid_shape), dtype=np.int16)
    conflict = np.zeros_like(support)
    occlusion = np.zeros_like(support)
    provenance = np.zeros((view_count, *grid_shape, view_count), dtype=np.bool_)
    residual_sum = np.zeros((view_count, *grid_shape), dtype=np.float32)
    residual_count = np.zeros_like(support)
    for source in range(view_count):
        provenance[source, ..., source] = raw_valid[source]
        for target in range(view_count):
            if target == source:
                continue
            projected_xy, projected_distance, in_front = _project_world(
                world_points[source], supplier.intrinsics[target], supplier.c2w[target]
            )
            inside = (
                in_front
                & (projected_xy[..., 0] >= 0)
                & (projected_xy[..., 0] <= width - 1)
                & (projected_xy[..., 1] >= 0)
                & (projected_xy[..., 1] <= height - 1)
            )
            target_depth, target_valid = _sample_nearest(
                supplier.depth_along_ray[target],
                supplier.valid_mask[target],
                projected_xy,
            )
            comparable = (
                raw_valid[source]
                & inside
                & target_valid
                & np.isfinite(target_depth)
                & (target_depth >= config.min_depth_m)
                & (target_depth <= config.max_depth_m)
            )
            tolerance = (
                config.depth_absolute_tolerance_m
                + config.depth_relative_tolerance * projected_distance
            )
            difference = target_depth - projected_distance
            target_world = _world_from_pixels(
                projected_xy, target_depth, supplier.intrinsics[target], supplier.c2w[target]
            )
            roundtrip_xy, _, roundtrip_front = _project_world(
                target_world, supplier.intrinsics[source], supplier.c2w[source]
            )
            roundtrip_error = np.linalg.norm(roundtrip_xy - pixel_grid, axis=-1)
            agrees = (
                comparable
                & (np.abs(difference) <= tolerance)
                & roundtrip_front
                & (roundtrip_error <= config.roundtrip_threshold_px)
            )
            is_occlusion = comparable & (difference < -tolerance)
            is_conflict = comparable & ~agrees & ~is_occlusion
            support[source] += agrees.astype(np.int16)
            occlusion[source] += is_occlusion.astype(np.int16)
            conflict[source] += is_conflict.astype(np.int16)
            provenance[source, ..., target] = agrees
            normalized_residual = np.minimum(
                1.0,
                np.abs(difference) / np.maximum(tolerance, 1e-8),
            )
            residual_sum[source] += np.where(comparable, normalized_residual, 0.0)
            residual_count[source] += comparable.astype(np.int16)

    certified = (
        raw_valid
        & (support >= config.minimum_supporting_views)
        & (conflict <= config.maximum_conflicting_views)
    )
    free_end = np.where(certified, np.maximum(0.0, depth - config.free_space_margin_m), 0.0)
    surface_near = np.where(certified, np.maximum(0.0, depth - config.surface_half_width_m), 0.0)
    surface_far = np.where(certified, depth + config.surface_half_width_m, 0.0)
    possible_support = max(1, view_count - 1)
    support_deficit = 1.0 - np.minimum(support, possible_support) / possible_support
    conflict_rate = conflict / possible_support
    residual = residual_sum / np.maximum(residual_count, 1)
    confidence_uncertainty = 1.0 / (1.0 + np.maximum(confidence, 0.0))
    risk = np.clip(
        0.35 * support_deficit
        + 0.30 * residual
        + 0.20 * conflict_rate
        + 0.15 * confidence_uncertainty,
        0.0,
        1.0,
    ).astype(np.float32)

    return DenseGeometryCertificates(
        frame_ids=supplier.frame_ids,
        pixel_xy=pixel_xy,
        depth_along_ray=depth.astype(np.float32),
        world_points=world_points,
        raw_confidence=confidence.astype(np.float32),
        raw_valid=raw_valid,
        certified=certified,
        support_count=support,
        conflict_count=conflict,
        occlusion_count=occlusion,
        provenance=provenance,
        free_end_m=free_end.astype(np.float32),
        surface_near_m=surface_near.astype(np.float32),
        surface_far_m=surface_far.astype(np.float32),
        risk=risk,
    )


def write_dense_certificate_cache(
    destination: str | Path,
    certificates: DenseGeometryCertificates,
    config: DenseCertificateConfig,
    *,
    supplier_arrays_sha256: str,
) -> Path:
    """Seal dense certificates in an overwrite-refusing JSON+NPZ directory."""

    target = Path(destination).resolve()
    if target.exists():
        raise FileExistsError(f"refusing to overwrite dense certificate cache: {target}")
    if not _is_sha256(supplier_arrays_sha256):
        raise ValueError("supplier_arrays_sha256 must be a lowercase SHA-256 digest")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=str(target.parent))
    )
    canonical = certificates.canonical()
    try:
        arrays_path = temporary / "arrays.npz"
        arrays = _certificate_arrays(canonical)
        np.savez_compressed(arrays_path, **arrays)
        metadata = {
            "schema_version": DENSE_CERTIFICATE_CACHE_SCHEMA_VERSION,
            "frame_ids": list(canonical.frame_ids),
            "config": asdict(config),
            "supplier_arrays_sha256": supplier_arrays_sha256,
            "arrays_file": arrays_path.name,
            "arrays_sha256": _sha256_file(arrays_path),
            "arrays": {
                name: {"shape": list(value.shape), "dtype": str(value.dtype)}
                for name, value in arrays.items()
            },
        }
        (temporary / "metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temporary, target)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return target


def read_dense_certificate_cache(
    source: str | Path,
) -> tuple[DenseGeometryCertificates, dict[str, Any]]:
    root = Path(source).resolve()
    metadata_path = root / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(metadata_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if (
        not isinstance(metadata, dict)
        or metadata.get("schema_version") != DENSE_CERTIFICATE_CACHE_SCHEMA_VERSION
    ):
        raise ValueError("dense certificate cache metadata has an unsupported schema")
    arrays_path = root / str(metadata.get("arrays_file", ""))
    if not arrays_path.is_file():
        raise FileNotFoundError(arrays_path)
    if _sha256_file(arrays_path) != metadata.get("arrays_sha256"):
        raise ValueError("dense certificate cache arrays hash does not match metadata")
    with np.load(arrays_path, allow_pickle=False) as values:
        required = set(_certificate_array_names()) | {"frame_ids"}
        if set(values.files) != required:
            raise ValueError("dense certificate cache array keys do not match schema")
        frame_ids = tuple(int(value) for value in values["frame_ids"].tolist())
        certificates = DenseGeometryCertificates(
            frame_ids=frame_ids,
            **{name: values[name].copy() for name in _certificate_array_names()},
        )
    if list(frame_ids) != metadata.get("frame_ids"):
        raise ValueError("dense certificate metadata frame IDs do not match arrays")
    actual_schema = {
        name: {"shape": list(value.shape), "dtype": str(value.dtype)}
        for name, value in _certificate_arrays(certificates).items()
    }
    if actual_schema != metadata.get("arrays"):
        raise ValueError("dense certificate metadata array schema does not match arrays")
    return certificates, metadata


def _camera_directions(pixel_xy: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    homogeneous = np.concatenate(
        (pixel_xy, np.ones((*pixel_xy.shape[:-1], 1), dtype=np.float32)), axis=-1
    )
    directions = homogeneous @ np.linalg.inv(intrinsics).T
    return directions / np.maximum(np.linalg.norm(directions, axis=-1, keepdims=True), 1e-8)


def _world_from_pixels(
    pixel_xy: np.ndarray, depth: np.ndarray, intrinsics: np.ndarray, c2w: np.ndarray
) -> np.ndarray:
    camera_direction = _camera_directions(pixel_xy, intrinsics)
    world_direction = camera_direction @ c2w[:3, :3].T
    return c2w[:3, 3] + world_direction * depth[..., None]


def _project_world(
    world: np.ndarray, intrinsics: np.ndarray, c2w: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rotation = c2w[:3, :3]
    camera = (world - c2w[:3, 3]) @ rotation
    projected = camera @ intrinsics.T
    z = projected[..., 2]
    xy = projected[..., :2] / np.maximum(z[..., None], 1e-8)
    distance = np.linalg.norm(camera, axis=-1)
    return xy.astype(np.float32), distance.astype(np.float32), camera[..., 2] > 1e-8


def _sample_nearest(
    depth: np.ndarray, mask: np.ndarray, pixel_xy: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    height, width = depth.shape
    x = np.clip(np.rint(pixel_xy[..., 0]).astype(np.int64), 0, width - 1)
    y = np.clip(np.rint(pixel_xy[..., 1]).astype(np.int64), 0, height - 1)
    return depth[y, x], mask[y, x]


def _certificate_array_names() -> tuple[str, ...]:
    return (
        "pixel_xy",
        "depth_along_ray",
        "world_points",
        "raw_confidence",
        "raw_valid",
        "certified",
        "support_count",
        "conflict_count",
        "occlusion_count",
        "provenance",
        "free_end_m",
        "surface_near_m",
        "surface_far_m",
        "risk",
    )


def _certificate_arrays(certificates: DenseGeometryCertificates) -> dict[str, np.ndarray]:
    result = {
        name: np.asarray(getattr(certificates, name)) for name in _certificate_array_names()
    }
    result["frame_ids"] = np.asarray(certificates.frame_ids, dtype=np.int64)
    return result


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)
