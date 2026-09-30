"""Post-seal query-GT geometry labels; never optimizer features or supervision.

All input cameras are in world coordinates. Bounds alone are in the shared anchor
frame. Nearest-depth sampling reproduces visibility.py: floor(uv + .5), clamped
index, then explicit frustum/depth validity. The tolerance is frozen .05m + .01d.
"""

from itertools import combinations

import numpy as np
import torch

from mcss.dynamic.types import hash_value
from mcss.geometry import generate_rays, project_world, transform_points
from mcss.mechanism_pilot.direct_capacity_contracts import ObservationBatch, StateSealBarrier
from mcss.types import Cameras

ABSOLUTE_TOLERANCE_M = 0.05
RELATIVE_TOLERANCE = 0.01


def _camera(batch, view):
    return Cameras(
        batch.cameras.intrinsics[0, view],
        batch.cameras.c2w[0, view],
        batch.cameras.image_size,
    )


def _validate_batch(batch, n_views=None):
    if not isinstance(batch, ObservationBatch) or batch.depth is None:
        raise TypeError("Typed depth-bearing ObservationBatch required")
    n = len(batch.frame_ids)
    if not n or len(set(batch.frame_ids)) != n or (n_views is not None and n != n_views):
        raise ValueError("Nonempty unique frame IDs and correct view count required")
    if batch.cameras.leading_shape != (1, n):
        raise ValueError("Camera batch must have leading shape [1,V]")
    if batch.depth.shape != (1, n, 1, *batch.cameras.image_size):
        raise ValueError("Depth must be [1,V,1,H,W] metric ray distance")
    if batch.depth.device != batch.cameras.device or not batch.depth.is_floating_point():
        raise ValueError("Depth and cameras must share device, with floating metric depth")


def _angle(a, b):
    return torch.rad2deg(torch.acos((a * b).sum(-1).clamp(-1, 1)))


def _compute(query, context, bounds, anchor_c2w):
    camera = _camera(query, 0)
    height, width = camera.image_size
    depth = query.depth[0, 0, 0].to(camera.dtype)
    valid = torch.isfinite(depth) & (depth > 0)
    origins, directions = generate_rays(camera)
    world = origins + directions * torch.where(valid, depth, 0)[..., None]
    points = world.reshape(-1, 3)
    qvalid = valid.reshape(-1)
    nan = torch.full_like(depth, float("nan"))
    ys, xs = torch.meshgrid(
        torch.arange(height, device=depth.device),
        torch.arange(width, device=depth.device),
        indexing="ij",
    )
    result = {
        "scene_id": np.asarray(query.scene_id),
        "query_id": np.asarray(query.frame_ids[0]),
        "context_frame_ids": np.asarray(context.frame_ids),
        "query_valid": valid,
        "pixel_xy": torch.stack((xs, ys), -1),
        "world_xyz": torch.where(valid[..., None], world, float("nan")),
        "query_gt_depth": depth,
    }
    fields = {
        k: []
        for k in (
            "projected_uv",
            "projected_z",
            "in_front",
            "inside_image",
            "context_depth_valid",
            "nearest_pixel_xy",
            "projected_ray_distance",
            "context_gt_ray_distance",
            "depth_disagreement",
            "visible_support",
            "occluded_by_context_surface",
            "depth_conflict_front",
            "query_context_angle",
            "query_context_camera_baseline",
        )
    }
    point_directions, centers = [], []
    for index in range(len(context.frame_ids)):
        current = _camera(context, index)
        pixels, z, frustum = project_world(points, current)
        finite = torch.isfinite(pixels).all(-1)
        ch, cw = current.image_size
        indices = torch.floor(torch.nan_to_num(pixels) + 0.5).long()
        x, y = indices[:, 0].clamp(0, cw - 1), indices[:, 1].clamp(0, ch - 1)
        sampled = context.depth[0, index, 0][y, x].to(camera.dtype)
        inside = finite & (pixels[:, 0] >= 0) & (pixels[:, 0] <= cw - 1)
        inside &= (pixels[:, 1] >= 0) & (pixels[:, 1] <= ch - 1) & qvalid
        front = (z > torch.finfo(camera.dtype).eps) & qvalid
        depth_valid = torch.isfinite(sampled) & (sampled > 0) & inside & front
        vector = points - current.c2w[:3, 3]
        distance = vector.norm(dim=-1)
        difference = distance - sampled
        tolerance = ABSOLUTE_TOLERANCE_M + RELATIVE_TOLERANCE * sampled
        eligible = frustum & finite & qvalid & depth_valid
        visible = eligible & (difference.abs() <= tolerance)
        normalized = vector / distance[:, None].clamp_min(torch.finfo(camera.dtype).eps)
        angle = _angle(normalized, directions.reshape(-1, 3))
        values = {
            "projected_uv": pixels.reshape(height, width, 2),
            "projected_z": z.reshape(height, width),
            "nearest_pixel_xy": torch.stack((x, y), -1).reshape(height, width, 2),
            "in_front": front.reshape(height, width),
            "inside_image": inside.reshape(height, width),
            "context_depth_valid": depth_valid.reshape(height, width),
            "projected_ray_distance": distance.reshape(height, width),
            "context_gt_ray_distance": sampled.reshape(height, width),
            "depth_disagreement": difference.reshape(height, width),
            "visible_support": visible.reshape(height, width),
            "occluded_by_context_surface": (eligible & (difference > tolerance)).reshape(
                height, width
            ),
            "depth_conflict_front": (eligible & (difference < -tolerance)).reshape(height, width),
            "query_context_angle": torch.where(valid, angle.reshape(height, width), nan),
            "query_context_camera_baseline": torch.full_like(
                depth, (current.c2w[:3, 3] - camera.c2w[:3, 3]).norm()
            ),
        }
        for key, value in values.items():
            fields[key].append(value)
        point_directions.append(normalized)
        centers.append(current.c2w[:3, 3])
    result.update({key: torch.stack(values) for key, values in fields.items()})
    visible = result["visible_support"]
    count = visible.sum(0)
    result["visible_view_count"] = torch.where(valid, count, -1)
    result["obs_class"] = torch.where(valid, count.clamp_max(2), -1)
    result["nearest_context_angle"] = torch.where(valid, result["query_context_angle"].amin(0), nan)
    result["nearest_context_distance"] = torch.where(
        valid, result["projected_ray_distance"].amin(0), nan
    )
    angle_max, angle_median, baseline = nan.clone(), nan.clone(), nan.clone()
    supported = count >= 2
    pairs, baselines = [], []
    for i, j in combinations(range(len(context.frame_ids)), 2):
        pair_valid = visible[i] & visible[j]
        angle = _angle(point_directions[i], point_directions[j]).reshape(height, width)
        pairs.append(torch.where(pair_valid, angle, nan))
        baselines.append(torch.where(pair_valid, (centers[i] - centers[j]).norm(), nan))
    if pairs and supported.any():
        angles = torch.stack(pairs)[:, supported]
        angle_max[supported] = torch.nan_to_num(angles, nan=-1).amax(0)
        angle_median[supported] = torch.nanquantile(angles, 0.5, dim=0)
        baseline[supported] = torch.nan_to_num(torch.stack(baselines)[:, supported], nan=-1).amax(0)
    result["max_triangulation_angle"] = angle_max
    result["median_triangulation_angle"] = angle_median
    result["camera_baseline"] = baseline
    # Right-closed bins: [0,5], (5,15], (15,30], (30,180]. Undefined is -1.
    result["triangulation_angle_bin"] = torch.where(
        supported,
        (angle_max > 5).long() + (angle_max > 15).long() + (angle_max > 30).long(),
        -1,
    )
    result["max_triangulation_angle_bin"] = result["triangulation_angle_bin"].clone()
    result["median_triangulation_angle_bin"] = torch.where(
        supported,
        (angle_median > 5).long() + (angle_median > 15).long() + (angle_median > 30).long(),
        -1,
    )
    anchored = transform_points(world, torch.linalg.inv(anchor_c2w))
    result["bounds_inside"] = (
        valid & (anchored >= bounds[0]).all(-1) & (anchored <= bounds[1]).all(-1)
    )
    return {
        k: v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else v for k, v in result.items()
    }


@torch.no_grad()
def audit_context_observability(
    *,
    barrier,
    scene_id,
    query,
    context,
    bounds,
    anchor_c2w,
    allowed_scene_ids,
    holdout_scene_ids,
):
    """Return NPZ-friendly full-resolution arrays only after all planned states seal.

    This API is evaluator-only, not an OS sandbox. Query must come from the sealed
    evaluator; context must be an RGBD context batch. It never opens files, changes
    states, or passes labels to optimizers. OBS IDs 0/1/2 mean OBS0/OBS1/OBS2PLUS.
    """
    if not isinstance(barrier, StateSealBarrier):
        raise TypeError("StateSealBarrier required")
    allowed, forbidden = set(allowed_scene_ids), set(holdout_scene_ids)
    if not allowed or allowed & forbidden or scene_id not in allowed or scene_id in forbidden:
        raise PermissionError("Holdout/unknown observability scoring is forbidden")
    barrier.assert_ready()
    _validate_batch(query, 1)
    _validate_batch(context)
    if query.scene_id != scene_id or context.scene_id != scene_id:
        raise PermissionError("Both batches must belong to the sealed scene")
    if query.supervision != "SEALED_QUERY_EVALUATION" or context.supervision != "CONTEXT_ONLY_RGBD":
        raise PermissionError("Only sealed query diagnostic and legal RGBD context observations")
    if set(query.frame_ids) & set(context.frame_ids):
        raise PermissionError("Query cannot masquerade as a context view")
    if (
        query.cameras.device != context.cameras.device
        or query.cameras.dtype != context.cameras.dtype
    ):
        raise ValueError("All world cameras must share dtype/device")
    bounds = bounds.reshape(2, 3)
    if (
        bounds.device != query.cameras.device
        or anchor_c2w.device != query.cameras.device
        or anchor_c2w.shape != (4, 4)
        or not torch.isfinite(bounds).all()
        or not (bounds[1] > bounds[0]).all()
    ):
        raise ValueError("Finite ordered anchor bounds and consistent camera device required")
    if not torch.equal(anchor_c2w, context.cameras.c2w[0, 0]):
        raise PermissionError("Bounds anchor must be the first shared context camera")
    inputs = (query.cameras, query.depth, context.cameras, context.depth, bounds, anchor_c2w)
    before = hash_value(inputs)
    result = barrier.read(
        scene_id, query.frame_ids[0], lambda: _compute(query, context, bounds, anchor_c2w)
    )
    barrier.assert_ready()
    if hash_value(inputs) != before:
        raise RuntimeError("Observability diagnostic mutated input geometry")
    return result
