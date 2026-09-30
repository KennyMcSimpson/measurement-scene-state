"""RGB-D track bounds from measured context depth (a V7 factor; the frozen rule is untouched).

The box is the axis-aligned hull, in the anchor frame, of every valid back-projected context
depth sample and every context camera centre, padded by the frozen GT-free rule (extent x1.1,
at least 1 m per axis). Only context depth and context cameras are read; never a query.
"""

from __future__ import annotations

import torch

from mcss.geometry import generate_rays, transform_cameras
from mcss.mechanism_pilot.rgbd_evidence_carrier import RGBDContext, RGBDSceneData

PADDING, MIN_EXTENT = 1.1, 1.0


def context_depth_bounds(context, anchor_c2w, trim=0.0):
    """Hull of measured context depth and cameras; `trim` drops per-axis quantile tails.

    Camera centres are always inside the box, so every context ray starts inside it.
    """
    if not isinstance(context, RGBDContext):
        raise TypeError("RGBDContext required")
    inverse = torch.linalg.inv(anchor_c2w.to(torch.float64))
    points, centers = [], []
    for observation, depth in zip(context, context.depths, strict=True):
        camera = transform_cameras(observation.camera.to(dtype=torch.float64), inverse)
        origins, rays = generate_rays(camera)
        distance = depth.to(torch.float64)
        valid = torch.isfinite(distance) & (distance > 0)
        points.append(origins[valid] + rays[valid] * distance[valid, None])
        centers.append(camera.c2w[:3, 3][None])
    points, centers = torch.cat(points + centers), torch.cat(centers)
    if trim:
        lo = torch.quantile(points, trim, dim=0)
        hi = torch.quantile(points, 1.0 - trim, dim=0)
    else:
        lo, hi = points.amin(0), points.amax(0)
    lo, hi = torch.minimum(lo, centers.amin(0)), torch.maximum(hi, centers.amax(0))
    center = (lo + hi) / 2
    extent = ((hi - lo) * PADDING).clamp_min(MIN_EXTENT)
    bounds = torch.stack((center - extent / 2, center + extent / 2))
    if not torch.isfinite(bounds).all():
        raise ValueError("Nonfinite context-depth bounds")
    return bounds.to(torch.float32)


class _LazyDepthBounds:
    """bounds[role] from the full A/B context of that role (anchor states reuse it)."""

    def __init__(self, data):
        self._data, self._cache = data, {}

    def __getitem__(self, role):
        if role not in ("A", "B"):
            raise KeyError(role)
        if role not in self._cache:
            context = self._data.context(role)
            anchor = context[0].camera.c2w
            box = context_depth_bounds(context, anchor, self._data.TRIM)
            self._cache[role] = box.to(self._data.device)
        return self._cache[role]


class DepthBoundsSceneData(RGBDSceneData):
    """RGBDSceneData whose bounds come from the measured context depth of each role."""

    TRIM = 0.0

    def __init__(self, record, manifest, root, device, access):
        super().__init__(record, manifest, root, device, access)
        self.bounds = _LazyDepthBounds(self)


class TrimmedDepthBoundsSceneData(DepthBoundsSceneData):
    """Context-depth bounds with 1% per-axis quantile tails removed (finer voxels)."""

    TRIM = 0.01
