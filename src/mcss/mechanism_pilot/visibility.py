"""Evaluator-only approximate depth-consistency masks, never runtime features.

Nearest-pixel sampling with a depth tolerance is an approximation, not a proof of
surface visibility. All cameras must use the same world coordinate system and all
provided depths must be metric ray distances (not optical-axis z depths).
"""

from __future__ import annotations

from collections.abc import Sequence
from math import isfinite

import torch
from torch import Tensor

from mcss.geometry import generate_rays, project_world
from mcss.types import Cameras


def _depth(camera: Cameras, value: Tensor) -> Tensor:
    if camera.leading_shape:
        raise ValueError("visibility requires unbatched cameras")
    if value.shape == (1, *camera.image_size):
        value = value[0]
    if value.shape != camera.image_size or not value.is_floating_point():
        raise ValueError("ray-distance depth must be floating [H,W] or [1,H,W]")
    if value.device != camera.device:
        raise ValueError("depth and camera must share a device")
    return value.to(dtype=camera.dtype)


@torch.no_grad()
def common_visibility_masks(
    query_camera: Cameras,
    query_depth: Tensor,
    context_a: Sequence[tuple[Cameras, Tensor]],
    context_b: Sequence[tuple[Cameras, Tensor]],
    *,
    absolute_tolerance_m: float = 0.05,
    relative_tolerance: float = 0.01,
) -> dict[str, Tensor]:
    """Intersect group unions of frustum and approximate depth-consistent coverage.

    A query GT point is depth-consistent with a context when its Euclidean distance
    to that context camera agrees with the nearest context pixel's GT ray distance
    within ``absolute_tolerance_m + relative_tolerance * context_GT``. Each group
    unions its context masks; the returned masks intersect those two group unions.
    Invalid query depths are excluded from both masks. Invalid context depth affects
    depth consistency only. Query labels must be accessed only after state sealing.
    """

    if not all(isfinite(v) and v >= 0 for v in (absolute_tolerance_m, relative_tolerance)):
        raise ValueError("depth tolerances must be finite and nonnegative")
    if not context_a or not context_b:
        raise ValueError("both context groups must be nonempty")
    depth = _depth(query_camera, query_depth)
    query_valid = torch.isfinite(depth) & (depth > 0)
    origins, directions = generate_rays(query_camera)
    safe_depth = torch.where(query_valid, depth, torch.zeros_like(depth))
    points = (origins + directions * safe_depth[..., None]).reshape(-1, 3)
    groups = []
    for contexts in (context_a, context_b):
        group_frustum = torch.zeros_like(query_valid).reshape(-1)
        group_consistent = torch.zeros_like(group_frustum)
        for camera, context_depth in contexts:
            if camera.device != query_camera.device or camera.dtype != query_camera.dtype:
                raise ValueError("all cameras must share device and dtype")
            target = _depth(camera, context_depth)
            pixels, _, inside = project_world(points, camera)
            finite = torch.isfinite(pixels).all(dim=-1)
            inside = inside & finite & query_valid.reshape(-1)
            height, width = camera.image_size
            indices = torch.floor(torch.nan_to_num(pixels) + 0.5).to(torch.long)
            x = indices[:, 0].clamp(0, width - 1)
            y = indices[:, 1].clamp(0, height - 1)
            sampled = target[y, x]
            valid_target = torch.isfinite(sampled) & (sampled > 0)
            distance = torch.linalg.vector_norm(points - camera.c2w[:3, 3], dim=-1)
            tolerance = absolute_tolerance_m + relative_tolerance * sampled
            consistent = inside & valid_target & ((distance - sampled).abs() <= tolerance)
            group_frustum |= inside
            group_consistent |= consistent
        groups.append((group_frustum, group_consistent))
    return {
        "common_frustum": (groups[0][0] & groups[1][0]).reshape(depth.shape),
        "depth_consistent_common": (groups[0][1] & groups[1][1]).reshape(depth.shape),
    }
