"""Model-free geometry backend of MEASUREMENT_PLANNING_TASK_V1, made only of the frozen project
functions.

The frozen 1-pixel z-buffer keeps, per pixel, the smallest ray distance over all measured points, so
the reprojection of any set of measured frames equals the pixel-wise minimum of the single-frame
reprojections. Each measured frame is reprojected once per camera; a subset's reconstruction is that
minimum followed by the frozen harmonic fill (REPROJ_HARMONIC over the subset's frames). With no
extra frame this is exactly the sealed REPROJ_HARMONIC of the arms evaluator.
"""

from __future__ import annotations

import hashlib
import importlib.util
from functools import cache
from pathlib import Path

import numpy as np

from mcss.mechanism_pilot.rgbd_evidence_carrier import RGBDContext

REPO = Path(__file__).resolve().parents[3]
FROZEN = {
    "geometry": "scripts/run_rgbd_geometric_baselines.py",
    "inpainting": "scripts/run_rgbd_inpaint_baselines.py",
}


@cache
def frozen(name):
    """A frozen project script loaded as a module (never edited by this experiment)."""
    path = REPO / FROZEN[name]
    spec = importlib.util.spec_from_file_location(f"measurement_planning_frozen_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def frozen_hashes():
    return {rel: hashlib.sha256((REPO / rel).read_bytes()).hexdigest() for rel in FROZEN.values()}


def single_frames(context):
    """Each frame of an RGBDContext as its own one-frame RGBDContext, keyed by frame id."""
    return {o.frame_id: RGBDContext([o], context.depths[j : j + 1]) for j, o in enumerate(context)}


def frame_points(context):
    """World points [N, 3] (float64) of every valid depth sample, per frame (context_points)."""
    geo = frozen("geometry")
    return {fid: geo.context_points(single) for fid, single in single_frames(context).items()}


def reproject(points, camera):
    """Frozen 1-pixel z-buffer of ray distance at a camera: [H, W] float64, inf where empty."""
    depth, _ = frozen("geometry").reproject(points, camera)
    return depth


def union(maps):
    """Pixel-wise minimum of single-frame z-buffers = the z-buffer of all their points together."""
    maps = list(maps)
    return np.minimum.reduce(maps) if len(maps) > 1 else np.array(maps[0], copy=True)


def ray_cosine(camera):
    return frozen("inpainting").ray_cosine(camera)


def near_mask(hit):
    """Pixels within NEAR_PX (8) of a hit (frozen definition)."""
    return frozen("inpainting").near_mask(hit)


def harmonic(depth, cosine, constant):
    """Frozen REPROJ_HARMONIC of a z-buffer: hits kept, holes filled by the Laplace solve in inverse
    planar depth."""
    hit = np.isfinite(depth)
    return frozen("inpainting").harmonic_fill(depth, hit, cosine, constant), hit


def absrel(prediction, gt, mask):
    """The arms evaluator's AbsRel on a mask of GT-valid pixels; None when the mask is empty."""
    return float(np.mean(np.abs(prediction[mask] - gt[mask]) / gt[mask])) if mask.any() else None
