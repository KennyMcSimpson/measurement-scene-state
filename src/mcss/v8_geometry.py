"""V8 model-supported geometry proposals and label-blind sealed caches.

This module is deliberately separate from the learned scene-state model.  A proposal is an
observation supplied by an external geometry model; it is not a measurement certificate and it
must not be silently promoted to a typed scene state.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

V8_PROPOSAL_SCHEMA_VERSION = 1
V8_CACHE_SCHEMA_VERSION = 1


@dataclass
class V8Proposal:
    """A dense, camera-frame metric proposal for a fixed context window."""

    supplier: str
    frame_ids: tuple[int, ...]
    depth_z_m: np.ndarray
    points_cam_m: np.ndarray
    native_confidence: np.ndarray
    native_risk: np.ndarray
    valid_mask: np.ndarray
    intrinsics: np.ndarray
    c2w: np.ndarray
    unit_provenance: str = "metric_m"
    coordinate_convention: str = "opencv_camera_frame"
    uncertainty_provenance: str = "native_rank_only"
    uncertainty_formula: str = "not_recorded"

    def __post_init__(self) -> None:
        self.supplier = str(self.supplier).strip()
        if not self.supplier:
            raise ValueError("supplier must be non-empty")
        self.frame_ids = tuple(int(value) for value in self.frame_ids)
        if not self.frame_ids or len(set(self.frame_ids)) != len(self.frame_ids):
            raise ValueError("frame_ids must be non-empty and unique")
        self.depth_z_m = np.asarray(self.depth_z_m, dtype=np.float32)
        self.points_cam_m = np.asarray(self.points_cam_m, dtype=np.float32)
        self.native_confidence = np.asarray(self.native_confidence, dtype=np.float32)
        self.native_risk = np.asarray(self.native_risk, dtype=np.float32)
        self.valid_mask = np.asarray(self.valid_mask, dtype=np.bool_)
        self.intrinsics = np.asarray(self.intrinsics, dtype=np.float32)
        self.c2w = np.asarray(self.c2w, dtype=np.float32)
        views = len(self.frame_ids)
        if self.depth_z_m.ndim != 3 or self.depth_z_m.shape[0] != views:
            raise ValueError("depth_z_m must have shape [V, H, W]")
        if self.points_cam_m.shape != (*self.depth_z_m.shape, 3):
            raise ValueError("points_cam_m must have shape [V, H, W, 3]")
        for name, value in {
            "native_confidence": self.native_confidence,
            "native_risk": self.native_risk,
            "valid_mask": self.valid_mask,
        }.items():
            if value.shape != self.depth_z_m.shape:
                raise ValueError(f"{name} must match depth_z_m shape")
        if self.intrinsics.shape != (views, 3, 3):
            raise ValueError("intrinsics must have shape [V, 3, 3]")
        if self.c2w.shape != (views, 4, 4):
            raise ValueError("c2w must have shape [V, 4, 4]")
        arrays = {
            "depth_z_m": self.depth_z_m,
            "points_cam_m": self.points_cam_m,
            "native_confidence": self.native_confidence,
            "native_risk": self.native_risk,
            "intrinsics": self.intrinsics,
            "c2w": self.c2w,
        }
        if not all(np.isfinite(value).all() for value in arrays.values()):
            raise ValueError("proposal arrays must contain only finite values")
        if (self.depth_z_m <= 0).any() and self.valid_mask.any():
            # Invalid pixels may be zero; valid pixels must be in front of the camera.
            if (self.depth_z_m[self.valid_mask] <= 0).any():
                raise ValueError("valid proposal depth must be positive")
        if self.valid_mask.any() and not np.allclose(
            self.points_cam_m[..., 2][self.valid_mask],
            self.depth_z_m[self.valid_mask],
            atol=1e-4,
            rtol=1e-4,
        ):
            raise ValueError("valid proposal point z coordinate must match depth_z_m")
        if not np.allclose(self.c2w[:, 3], [0.0, 0.0, 0.0, 1.0], atol=1e-5):
            raise ValueError("c2w must have homogeneous final rows")
        if self.coordinate_convention != "opencv_camera_frame":
            raise ValueError("unsupported proposal coordinate convention")
        if self.unit_provenance != "metric_m":
            raise ValueError("proposal units must be metric_m")

    @property
    def image_size(self) -> tuple[int, int]:
        return int(self.depth_z_m.shape[1]), int(self.depth_z_m.shape[2])

    def canonical(self) -> V8Proposal:
        order = np.argsort(np.asarray(self.frame_ids), kind="stable")
        if np.array_equal(order, np.arange(len(order))):
            return self
        return V8Proposal(
            supplier=self.supplier,
            frame_ids=tuple(self.frame_ids[int(index)] for index in order),
            depth_z_m=self.depth_z_m[order],
            points_cam_m=self.points_cam_m[order],
            native_confidence=self.native_confidence[order],
            native_risk=self.native_risk[order],
            valid_mask=self.valid_mask[order],
            intrinsics=self.intrinsics[order],
            c2w=self.c2w[order],
            unit_provenance=self.unit_provenance,
            coordinate_convention=self.coordinate_convention,
            uncertainty_provenance=self.uncertainty_provenance,
            uncertainty_formula=self.uncertainty_formula,
        )


def depth_z_to_points_cam(depth_z_m: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    """Backproject z-buffer depth to OpenCV camera-frame points."""

    depth = np.asarray(depth_z_m, dtype=np.float32)
    K = np.asarray(intrinsics, dtype=np.float32)
    if depth.ndim != 2 or K.shape != (3, 3):
        raise ValueError("depth_z_m and intrinsics have invalid shapes")
    if not np.isfinite(depth).all() or not np.isfinite(K).all():
        raise ValueError("depth and intrinsics must be finite")
    height, width = depth.shape
    yy, xx = np.meshgrid(
        np.arange(height, dtype=np.float32), np.arange(width, dtype=np.float32), indexing="ij"
    )
    pixels = np.stack([xx, yy, np.ones_like(xx)], axis=-1)
    rays = pixels @ np.linalg.inv(K).T
    return rays * depth[..., None]


def ray_distance_to_depth_z(
    ray_distance_m: np.ndarray, intrinsics: np.ndarray
) -> np.ndarray:
    """Convert Euclidean distance along a camera ray to OpenCV z-depth."""

    distance = np.asarray(ray_distance_m, dtype=np.float32)
    K = np.asarray(intrinsics, dtype=np.float32)
    if distance.ndim != 2 or K.shape != (3, 3):
        raise ValueError("ray_distance_m and intrinsics have invalid shapes")
    if not np.isfinite(distance).all() or not np.isfinite(K).all():
        raise ValueError("ray distance and intrinsics must be finite")
    height, width = distance.shape
    yy, xx = np.meshgrid(
        np.arange(height, dtype=np.float32),
        np.arange(width, dtype=np.float32),
        indexing="ij",
    )
    pixels = np.stack([xx, yy, np.ones_like(xx)], axis=-1)
    rays = pixels @ np.linalg.inv(K).T
    ray_norm = np.linalg.norm(rays, axis=-1)
    if (ray_norm <= 0).any():
        raise ValueError("camera rays must have positive norm")
    return distance * rays[..., 2] / ray_norm


def emvsnet_mm_to_m(value_mm: np.ndarray) -> np.ndarray:
    """Convert EMVSNet's DTU millimetre convention to metres."""

    value = np.asarray(value_mm, dtype=np.float32)
    if not np.isfinite(value).all():
        raise ValueError("EMVSNet depth must be finite before unit conversion")
    return value / np.float32(1000.0)


def classify_dual_proposals(
    first: V8Proposal,
    second: V8Proposal,
    *,
    disagreement_threshold_m: float,
    intervention_delta_m: float | None = None,
    joint_failure: bool = False,
) -> dict[str, Any]:
    """Assign an explicitly limited support status to two proposals.

    ``model-supported`` means only that the two proposal fields agree at the requested numerical
    tolerance.  It is intentionally distinct from measured/certified/correct.
    """

    if first.frame_ids != second.frame_ids or first.depth_z_m.shape != second.depth_z_m.shape:
        raise ValueError("dual proposals must share frame IDs and spatial shape")
    if disagreement_threshold_m <= 0:
        raise ValueError("disagreement_threshold_m must be positive")
    overlap = first.valid_mask & second.valid_mask
    if overlap.any():
        disagreement = np.abs(first.depth_z_m - second.depth_z_m)[overlap]
        max_disagreement = float(np.max(disagreement))
        median_disagreement = float(np.median(disagreement))
    else:
        max_disagreement = None
        median_disagreement = None
    status = "unknown"
    if max_disagreement is not None:
        status = "model-supported" if max_disagreement <= disagreement_threshold_m else "conflict"
        if joint_failure and status == "model-supported":
            status = "quiet_joint_failure"
    return {
        "status": status,
        "semantic_warning": (
            "agreement is supplier agreement only; it is not a measurement certificate "
            "or correctness proof"
        ),
        "overlap_fraction": float(overlap.mean()),
        "max_depth_disagreement_m": max_disagreement,
        "median_depth_disagreement_m": median_disagreement,
        "intervention_delta_m": intervention_delta_m,
        "quiet_joint_failure": bool(status == "quiet_joint_failure"),
    }


def write_v8_proposal_cache(
    destination: str | Path,
    proposal: V8Proposal,
    *,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Atomically write an immutable proposal cache and a hash seal."""

    target = Path(destination).resolve()
    if target.exists():
        raise FileExistsError(f"refusing to overwrite V8 proposal cache: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=str(target.parent)))
    canonical = proposal.canonical()
    try:
        arrays_path = temporary / "arrays.npz"
        np.savez_compressed(
            arrays_path,
            frame_ids=np.asarray(canonical.frame_ids, dtype=np.int64),
            depth_z_m=canonical.depth_z_m,
            points_cam_m=canonical.points_cam_m,
            native_confidence=canonical.native_confidence,
            native_risk=canonical.native_risk,
            valid_mask=canonical.valid_mask,
            intrinsics=canonical.intrinsics,
            c2w=canonical.c2w,
        )
        arrays_hash = _sha256_file(arrays_path)
        payload = {
            "schema_version": V8_CACHE_SCHEMA_VERSION,
            "proposal_schema_version": V8_PROPOSAL_SCHEMA_VERSION,
            "supplier": canonical.supplier,
            "frame_ids": list(canonical.frame_ids),
            "image_size": list(canonical.image_size),
            "unit_provenance": canonical.unit_provenance,
            "coordinate_convention": canonical.coordinate_convention,
            "uncertainty_provenance": canonical.uncertainty_provenance,
            "uncertainty_formula": canonical.uncertainty_formula,
            "arrays_file": arrays_path.name,
            "arrays_sha256": arrays_hash,
            "arrays": {
                name: {"shape": list(value.shape), "dtype": str(value.dtype)}
                for name, value in _arrays(canonical).items()
            },
            "metadata": metadata or {},
            "sealed_at_utc": datetime.now(UTC).isoformat(),
            "sealed": True,
        }
        (temporary / "metadata.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
        )
        seal = {
            "schema_version": V8_CACHE_SCHEMA_VERSION,
            "metadata_sha256": _sha256_file(temporary / "metadata.json"),
            "arrays_sha256": arrays_hash,
        }
        (temporary / "SEAL.json").write_text(
            json.dumps(seal, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temporary, target)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return target


def read_v8_proposal_cache(source: str | Path) -> tuple[V8Proposal, dict[str, Any]]:
    root = Path(source).resolve()
    metadata_path = root / "metadata.json"
    seal_path = root / "SEAL.json"
    if not metadata_path.is_file() or not seal_path.is_file():
        raise FileNotFoundError("V8 proposal cache requires metadata.json and SEAL.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    if metadata.get("schema_version") != V8_CACHE_SCHEMA_VERSION or not metadata.get("sealed"):
        raise ValueError("V8 proposal cache is not sealed or has an unsupported schema")
    if seal.get("metadata_sha256") != _sha256_file(metadata_path):
        raise ValueError("V8 proposal metadata seal does not match")
    arrays_path = root / str(metadata.get("arrays_file", ""))
    if not arrays_path.is_file() or seal.get("arrays_sha256") != _sha256_file(arrays_path):
        raise ValueError("V8 proposal arrays seal does not match")
    with np.load(arrays_path, allow_pickle=False) as arrays:
        required = set(_array_names())
        if set(arrays.files) != required:
            raise ValueError("V8 proposal array keys do not match schema")
        proposal = V8Proposal(
            supplier=str(metadata["supplier"]),
            frame_ids=tuple(int(value) for value in arrays["frame_ids"].tolist()),
            depth_z_m=arrays["depth_z_m"].copy(),
            points_cam_m=arrays["points_cam_m"].copy(),
            native_confidence=arrays["native_confidence"].copy(),
            native_risk=arrays["native_risk"].copy(),
            valid_mask=arrays["valid_mask"].copy(),
            intrinsics=arrays["intrinsics"].copy(),
            c2w=arrays["c2w"].copy(),
            unit_provenance=str(metadata["unit_provenance"]),
            coordinate_convention=str(metadata["coordinate_convention"]),
            uncertainty_provenance=str(metadata.get("uncertainty_provenance", "native_rank_only")),
            uncertainty_formula=str(metadata.get("uncertainty_formula", "not_recorded")),
        )
    if list(proposal.frame_ids) != metadata.get("frame_ids"):
        raise ValueError("V8 proposal frame IDs disagree with metadata")
    return proposal, metadata


def _array_names() -> tuple[str, ...]:
    return (
        "frame_ids",
        "depth_z_m",
        "points_cam_m",
        "native_confidence",
        "native_risk",
        "valid_mask",
        "intrinsics",
        "c2w",
    )


def _arrays(proposal: V8Proposal) -> dict[str, np.ndarray]:
    return {
        "depth_z_m": proposal.depth_z_m,
        "points_cam_m": proposal.points_cam_m,
        "native_confidence": proposal.native_confidence,
        "native_risk": proposal.native_risk,
        "valid_mask": proposal.valid_mask,
        "intrinsics": proposal.intrinsics,
        "c2w": proposal.c2w,
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
