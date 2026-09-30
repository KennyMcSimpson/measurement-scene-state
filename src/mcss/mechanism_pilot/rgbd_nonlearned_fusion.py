"""Non-learned RGB-D fusion baseline for the learned carrier (no network, four TRAIN-fit scalars).

The context depth of every view is voxelized into the same 16^3 grid inside the same frozen
GT-free bounds, and the state is read by the same fixed renderer as every learned carrier:
density logit = a * surface - b * free + c, from the frozen V5 depth statistics with
sigma = k * mean voxel edge; colour = per-view projected context RGB weighted by the per-view
surface likelihood. Nothing is learned beyond (k, a, b, c), chosen on TRAIN24 queries only.
"""

from __future__ import annotations

import math

import torch

from mcss.dynamic.feedback import anchored_observation
from mcss.geometry import project_world
from mcss.mechanism_pilot.rgbd_evidence_carrier import RGBDContext, depth_statistics
from mcss.types import SceneState

GRID = 16
GRID_SEARCH = {"k": (0.5, 1.0), "a": (8.0, 16.0, 32.0, 64.0), "b": (0.0, 4.0), "c": (-4.0, -2.0)}


def voxel_centers(bounds, grid=GRID):
    """Voxel centres [D*H*W, 3] in (d, h, w) order; x indexes W, y H, z D (renderer layout)."""
    lo, hi = bounds[0], bounds[1]
    steps = (torch.arange(grid, dtype=lo.dtype, device=lo.device) + 0.5) / grid
    z, y, x = torch.meshgrid(steps, steps, steps, indexing="ij")
    return lo + torch.stack((x, y, z), -1).reshape(-1, 3) * (hi - lo)


def _colors(observations, depths, points, sigma):
    """Surface-likelihood weighted projected context RGB; mid-gray where no view is valid."""
    total = torch.zeros(points.shape[0], 3, dtype=torch.float32)
    weight = torch.zeros(points.shape[0], dtype=torch.float32)
    for observation, depth in zip(observations, depths, strict=True):
        camera = observation.camera.to(dtype=torch.float32)
        height, width = camera.image_size
        pixels, _, inside = project_world(points, camera)
        finite = torch.isfinite(pixels).all(-1)
        safe = torch.where(finite.unsqueeze(-1), pixels, torch.zeros_like(pixels))
        u = safe[:, 0].round().clamp(0, width - 1).long()
        v = safe[:, 1].round().clamp(0, height - 1).long()
        measured = depth.to(torch.float32)[v, u]
        distance = (points - camera.c2w[:3, 3]).norm(dim=-1)
        valid = inside & finite & torch.isfinite(measured) & (measured > 0)
        likelihood = torch.exp(-0.5 * ((distance - measured) / sigma).square()) * valid
        rgb = observation.rgb.to(torch.float32)[:, v, u].T
        total += likelihood.unsqueeze(-1) * rgb
        weight += likelihood
    gray = torch.full_like(total, 0.5)
    return torch.where(
        weight.unsqueeze(-1) > 1e-6, total / weight.clamp_min(1e-6).unsqueeze(-1), gray
    )


@torch.no_grad()
def fused_state(context, bounds, params):
    """Query-independent SceneState from an RGBDContext; anchored at its first camera."""
    if not isinstance(context, RGBDContext):
        raise TypeError("RGBDContext required")
    anchor = context[0].camera.c2w
    observations = [anchored_observation(o, anchor) for o in context]
    bounds = torch.as_tensor(bounds, dtype=torch.float32).reshape(2, 3)
    points = voxel_centers(bounds)
    edge = ((bounds[1] - bounds[0]) / GRID).mean()
    stats = depth_statistics(
        observations,
        context.depths,
        points,
        torch.tensor(math.log(params["k"])),
        edge,
    )
    surface, free, _ = stats.unbind(-1)
    logits = params["a"] * surface - params["b"] * free + params["c"]
    colors = _colors(observations, context.depths, points, params["k"] * edge)
    shape = (1, 1, GRID, GRID, GRID)
    state = SceneState(
        density_logits=logits.reshape(shape),
        color=colors.T.reshape(1, 3, GRID, GRID, GRID).contiguous(),
        log_variance=torch.full(shape, -3.0),
        bounds=bounds.reshape(1, 2, 3),
    )
    return state, anchor.clone()
