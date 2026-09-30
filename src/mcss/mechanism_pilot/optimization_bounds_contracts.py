"""Typed, context-camera-only reproduction of the frozen GT-free AABB rule."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

from mcss.dynamic.types import hash_value
from mcss.geometry import generate_rays, transform_cameras
from mcss.types import Cameras

CURRENT_BOUNDS = ((-6.0, -4.0, -6.0), (6.0, 4.0, 6.0))


@dataclass(frozen=True)
class FrozenTrainingPrior:
    near_m: float
    far_m: float
    source_sha256: str

    def __post_init__(self):
        if not (0 < self.near_m < self.far_m < float("inf")):
            raise ValueError("Finite positive ordered frozen TRAIN prior required")
        if len(self.source_sha256) != 64:
            raise ValueError("Frozen prior artifact SHA256 required")


@dataclass(frozen=True)
class ContextCameraBundle:
    scene_id: str
    role: Literal["A", "B"]
    frame_ids: tuple[int, ...]
    cameras: tuple[Cameras, ...]
    anchor_c2w: torch.Tensor


def context_camera_bundle(record, role, image_size=(128, 160)):
    """Read only explicitly arrived A/B cameras; query media/cameras are never accessed."""
    if role not in ("A", "B"):
        raise PermissionError("Only fixed context A/B cameras permitted for bounds")
    roles = record["roles"]
    ids = tuple(roles["context_a" if role == "A" else "context_b"])
    if not ids or len(ids) != len(set(ids)) or set(ids) & set(roles["primary_query"]):
        raise PermissionError("Query camera cannot enter bounds context")
    if ids[0] != roles["context_a"][0]:
        raise ValueError("Contexts must share the old coordinate anchor")
    # Do not deep-copy or inspect non-context frame values.
    frames = {f["frame_id"]: f for f in record["frames"] if f["frame_id"] in ids}
    cameras = []
    for fid in ids:
        f = frames[fid]
        cameras.append(
            Cameras(
                torch.tensor(f["intrinsics"], dtype=torch.float64),
                torch.tensor(f["c2w"], dtype=torch.float64),
                tuple(image_size),
            )
        )
    return ContextCameraBundle(
        record["scene_id"], role, ids, tuple(cameras), cameras[0].c2w.clone()
    )


def frozen_gt_free_bounds(bundle: ContextCameraBundle, prior: FrozenTrainingPrior):
    """Exactly old alternative: all pixel unit-rays at TRAIN q01/q99; 5% each side."""
    if not isinstance(bundle, ContextCameraBundle) or not isinstance(prior, FrozenTrainingPrior):
        raise TypeError("Typed context cameras and frozen TRAIN prior required")
    if bundle.role not in ("A", "B") or len(bundle.frame_ids) != len(bundle.cameras):
        raise PermissionError("Invalid context-camera bundle")
    before = hash_value(bundle.cameras)
    points = []
    for camera in bundle.cameras:
        anchored = transform_cameras(camera, torch.linalg.inv(bundle.anchor_c2w))
        origins, rays = generate_rays(anchored)
        points.extend([(origins + rays * t).reshape(-1, 3) for t in (prior.near_m, prior.far_m)])
    points = torch.cat(points)
    lo, hi = points.amin(0), points.amax(0)
    center = (lo + hi) / 2
    extent = ((hi - lo) * 1.1).clamp_min(1.0)
    bounds = torch.stack((center - extent / 2, center + extent / 2))
    if not torch.isfinite(bounds).all():
        raise ValueError("Nonfinite context-derived bounds")
    assert hash_value(bundle.cameras) == before
    return bounds
