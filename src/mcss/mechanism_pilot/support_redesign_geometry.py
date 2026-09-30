"""Explicit support interventions, preserving voxel/candidate scatter consistency.

Runtime camera-only entry points cannot accept labels. Oracle geometry is a separate
API and is never a deployable scene representation or mathematical upper bound.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
import torch

from mcss.geometry import generate_rays, project_world, transform_cameras, transform_points


@dataclass(frozen=True)
class GeometryPlan:
    method: str
    bounds: torch.Tensor
    candidate_ids: torch.Tensor
    points: torch.Tensor
    normalized_xyz: torch.Tensor
    oracle: bool = False


def lattice(bounds, grid=(8, 8, 8)):
    depth, height, width = grid
    ids = torch.arange(depth * height * width, device=bounds.device)
    fractions = torch.stack(
        (
            (ids.remainder(width) + 0.5) / width,
            (torch.div(ids, width, rounding_mode="floor").remainder(height) + 0.5) / height,
            (torch.div(ids, height * width, rounding_mode="floor") + 0.5) / depth,
        ),
        -1,
    ).to(bounds)
    return bounds[0] + fractions * (bounds[1] - bounds[0]), fractions * 2 - 1


def view_counts(points, cameras):
    return torch.stack([project_world(points, c)[2] for c in cameras]).sum(0)


def nearest_distance(points, surfaces):
    if not len(surfaces):
        return torch.full((len(points),), float("inf"), device=points.device)
    minimum = torch.full((len(points),), float("inf"), device=points.device)
    for chunk in surfaces.split(4096):
        minimum = torch.minimum(minimum, torch.cdist(points, chunk.to(points)).amin(1))
    return minimum


def enclose(points, *, padding=0.05, minimum_extent=1.0):
    if points.ndim != 2 or points.shape[1] != 3 or not len(points):
        raise ValueError("Finite nonempty 3D points required")
    if not torch.isfinite(points).all():
        raise ValueError("Bounds cannot contain nonfinite geometry")
    low, high = points.amin(0), points.amax(0)
    center = (low + high) / 2
    extent = (high - low).clamp_min(minimum_extent) * (1 + 2 * padding)
    return torch.stack((center - extent / 2, center + extent / 2))


def adaptive_bounds(cameras, *, near: float, far: float):
    """Only arrived cameras and frozen TRAIN global ray-depth constants admitted."""
    if not 0 < near < far or not cameras:
        raise ValueError("Positive ordered train near/far and arrived cameras required")
    points = []
    for camera in cameras:
        origins, directions = generate_rays(camera)
        # Camera corners alone need not bound normalized-ray interior directions.
        # All image rays (same fixed resolution) avoid that geometric shortcut.
        for distance in (near, far):
            points.append((origins + directions * distance).reshape(-1, 3))
    return enclose(torch.cat(points))


def allocate_candidates(bounds, cameras, *, count=128):
    """Camera-frustum allocation only; no scene depth or feature/model score argument."""
    points, _ = lattice(bounds)
    if not 0 < count <= len(points) or not cameras:
        raise ValueError("Invalid candidate budget or missing context cameras")
    counts = view_counts(points, cameras).cpu().numpy()
    # Deterministic stable tie: voxel ID. No candidate budget increase.
    order = np.lexsort((np.arange(len(points)), -counts))[:count].copy()
    return torch.tensor(order, device=bounds.device, dtype=torch.long)


def oracle_candidates(bounds, cameras, context_surfaces, *, count=128):
    """Offline GT-near support selection from the legal frozen-resolution lattice."""
    points, _ = lattice(bounds)
    if count > len(points) or count < 1:
        raise ValueError("Invalid budget")
    counts = view_counts(points, cameras).cpu().numpy()
    distances = nearest_distance(points, context_surfaces).cpu().numpy()
    order = np.lexsort((np.arange(len(points)), -counts, distances, -(counts >= 2).astype(int)))
    return torch.tensor(order[:count].copy(), dtype=torch.long, device=bounds.device)


def make_plan(method, bounds, ids, *, oracle=False):
    points, normalized = lattice(bounds)
    if len(ids) != 128 or len(ids.unique()) != 128 or (ids < 0).any() or (ids >= 512).any():
        raise ValueError("Exactly128 unique scatter IDs in the8^3 lattice required")
    return GeometryPlan(method, bounds.clone(), ids.clone(), points[ids], normalized[ids], oracle)


def deployable_plan(method, original_bounds, original_ids, cameras, *, depth_prior=None):
    """Strict GT-free runtime entry. Oracle payload/unknown kwargs are rejected."""
    if method not in {"R0", "R1", "R2", "R3", "R12", "R123"}:
        raise PermissionError("Oracle/unknown method cannot enter deployable geometry")
    bounds = original_bounds
    if method in {"R1", "R12", "R123"}:
        if depth_prior is None or set(depth_prior) != {"near", "far"}:
            raise ValueError("Frozen TRAIN-only near/far constants required")
        bounds = adaptive_bounds(cameras, **depth_prior)
    ids = allocate_candidates(bounds, cameras) if method in {"R2", "R12", "R123"} else original_ids
    return make_plan(method, bounds, ids)


def oracle_plan(method, original_bounds, original_ids, cameras, *, query_bounds, context_surfaces):
    if method not in {"ORACLE_VOLUME", "ORACLE_SUPPORT", "ORACLE_VOLUME_SUPPORT"}:
        raise PermissionError("Diagnostic-only oracle method required")
    bounds = query_bounds if method != "ORACLE_SUPPORT" else original_bounds
    ids = (
        oracle_candidates(bounds, cameras, context_surfaces)
        if method != "ORACLE_VOLUME"
        else original_ids
    )
    return make_plan(method, bounds, ids, oracle=True)


def apply_plan(carrier, plan):
    """Change only explicit geometry buffers; never save this as the original checkpoint."""
    if tuple(carrier.config.grid_size) != (8, 8, 8) or carrier.config.token_count != 128:
        raise ValueError("Locked8^3/128 schema required")
    ids, bounds = plan.candidate_ids, plan.bounds
    if (
        ids.shape != (128,)
        or ids.dtype != torch.long
        or len(ids.unique()) != 128
        or (ids < 0).any()
        or (ids >= 512).any()
    ):
        raise ValueError("Exactly128 unique valid scatter IDs required")
    if (
        bounds.shape != (2, 3)
        or not torch.isfinite(bounds).all()
        or not (bounds[1] > bounds[0]).all()
    ):
        raise ValueError("Finite ordered2x3 bounds required")
    expected, normal = lattice(plan.bounds)
    if not torch.equal(plan.points, expected[plan.candidate_ids]):
        raise ValueError("Candidate points no longer match scatter voxel centers")
    if not torch.equal(plan.normalized_xyz, normal[plan.candidate_ids]):
        raise ValueError("Normalized candidate coordinates disagree")
    carrier._bounds.copy_(plan.bounds[None])
    carrier._candidate_ids.copy_(plan.candidate_ids)
    carrier._candidate_points.copy_(plan.points)
    carrier._candidate_normalized_xyz.copy_(plan.normalized_xyz)


def surface_points(camera, depth, anchor):
    origins, rays = generate_rays(camera)
    valid = torch.isfinite(depth) & (depth > 0)
    return transform_points(
        origins[valid] + rays[valid] * depth[valid, None], torch.linalg.inv(anchor)
    )


def context_selection(cameras_by_id, original_bounds, original_ids):
    """Only already-arrived cameras; caller must exclude every query/future frame."""
    ids = sorted(cameras_by_id)
    if len(ids) < 5:
        raise ValueError("Need five arrived frames")
    anchor = ids[0]
    inverse = torch.linalg.inv(cameras_by_id[anchor].c2w)
    cameras = {i: transform_cameras(c, inverse) for i, c in cameras_by_id.items()}
    points, _ = lattice(original_bounds)
    points = points[original_ids]
    options = []
    for pair in combinations(ids[1:], 2):
        counts = view_counts(points, [cameras[i] for i in (anchor, *pair)])
        baseline = sum(float(cameras[i].c2w[:3, 3].norm()) for i in pair) / 2
        options.append((pair, int((counts >= 2).sum()), baseline))
    best = None
    for a, b in combinations(options, 2):
        if set(a[0]) & set(b[0]):
            continue
        key = (-min(a[1], b[1]), -(a[2] + b[2]) / 2, a[0], b[0])
        if best is None or key < best[0]:
            best = (key, a[0], b[0])
    return {"context_a": [anchor, *best[1]], "context_b": [anchor, *best[2]]}
