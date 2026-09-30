"""Geometric baselines: exact reprojection, z-buffer, hole filling and the evidence labels."""

import importlib.util
from pathlib import Path

import numpy as np
import torch

from mcss.dynamic.types import OnlineObservation
from mcss.geometry import generate_rays
from mcss.mechanism_pilot.rgbd_evidence_carrier import RGBDContext
from mcss.types import Cameras


def script():
    path = Path(__file__).parents[1] / "scripts" / "run_rgbd_geometric_baselines.py"
    spec = importlib.util.spec_from_file_location("run_rgbd_geometric_baselines", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def camera(tx=0.0, height=24, width=32, focal=20.0):
    intrinsics = torch.tensor(
        [[focal, 0.0, (width - 1) / 2], [0.0, focal, (height - 1) / 2], [0.0, 0.0, 1.0]]
    )
    c2w = torch.eye(4)
    c2w[0, 3] = tx
    return Cameras(intrinsics, c2w, (height, width))


def plane_distance(cam, z):
    """Ray distance from `cam` to the plane Z = z (cameras look along +Z)."""
    origins, rays = generate_rays(cam.to(dtype=torch.float64))
    return (z - origins[..., 2]) / rays[..., 2]


def test_reprojection_recovers_a_plane_seen_from_a_shifted_camera():
    m = script()
    source, target = camera(0.0), camera(0.15)
    depth = plane_distance(source, 3.0).to(torch.float32)
    context = RGBDContext([OnlineObservation("s", 0, torch.zeros(3, 24, 32), source)], depth[None])
    reprojected, hit = m.reproject(m.context_points(context), target)
    truth = plane_distance(target, 3.0).numpy()
    assert hit.mean() > 0.8
    assert (np.abs(reprojected[hit] - truth[hit]) / truth[hit]).max() < 0.02


def test_zbuffer_keeps_the_nearest_sample_and_the_fill_rules():
    m = script()
    points = torch.tensor([[0.0, 0.0, 2.0], [0.0, 0.0, 5.0]], dtype=torch.float64)
    depth, hit = m.reproject(points, camera())
    assert hit.sum() == 1 and np.isclose(depth[hit][0], 2.0)
    assert np.allclose(m.fill_nearest(depth, hit, 9.0), 2.0)
    constant = m.fill_constant(depth, hit, 9.0)
    assert np.isclose(constant[hit][0], 2.0) and np.allclose(constant[~hit], 9.0)
    empty = np.full((4, 4), np.inf)
    assert np.allclose(m.fill_nearest(empty, np.zeros((4, 4), bool), 9.0), 9.0)


def test_labels_follow_the_frozen_evidence_rule():
    m = script()
    base = {
        "mean": 0.1,
        "ci95": [0.05, 0.15],
        "positive": 8,
        "tied": 0,
        "negative": 0,
        "n_scenes": 8,
    }
    assert m.label(base) == "ABOVE"
    worse = {**base, "mean": -0.1, "ci95": [-0.15, -0.05], "positive": 0, "negative": 8}
    assert m.label(worse) == "BELOW"
    assert m.label({**base, "ci95": [-0.01, 0.2]}) == "NOT_DISTINGUISHABLE"
