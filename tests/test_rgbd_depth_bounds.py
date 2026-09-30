"""Context-depth bounds: contain every measured surface point and camera, frozen padding rule."""

import pytest
import torch
from test_geometry_carrier_evaluation import fixture as evaluation_fixture

from mcss.dynamic.types import OnlineObservation
from mcss.mechanism_pilot.rgbd_depth_bounds import (
    DepthBoundsSceneData,
    TrimmedDepthBoundsSceneData,
    context_depth_bounds,
)
from mcss.mechanism_pilot.rgbd_evidence_carrier import RGBDContext
from mcss.types import Cameras


def camera(x, height=16, width=20, focal=15.0):
    cx, cy = (width - 1) / 2, (height - 1) / 2
    pose = torch.eye(4)
    pose[0, 3] = x
    return Cameras(
        torch.tensor([[focal, 0, cx], [0, focal, cy], [0, 0, 1.0]]), pose, (height, width)
    )


def test_box_contains_far_measured_surfaces_that_the_frozen_prior_would_cut():
    cams = [camera(x) for x in (0.0, -0.5, 0.5)]
    obs = [OnlineObservation("s", i, torch.full((3, 16, 20), 0.5), c) for i, c in enumerate(cams)]
    depths = torch.full((3, 16, 20), 9.0)  # far wall at ~9 m ray distance, beyond the 5.67 m prior
    depths[:, 0, 0] = float("nan")  # invalid pixels are ignored
    bounds = context_depth_bounds(RGBDContext(obs, depths), cams[0].c2w)
    assert bounds[1, 2] > 9.0 * 0.8 and bounds[0, 2] <= 0.0
    extent = bounds[1] - bounds[0]
    assert torch.all(extent >= 1.0) and torch.isfinite(bounds).all()


def test_scene_data_bounds_are_lazy_per_role_and_reject_other_keys(tmp_path):
    manifest, _ = evaluation_fixture(tmp_path)
    record = next(r for r in manifest["scenes"] if r["split"] == "DEV")
    data = DepthBoundsSceneData(record, manifest, tmp_path, "cpu", [])
    box = data.bounds["A"]
    assert box.shape == (2, 3) and data.bounds["A"] is box
    with pytest.raises(KeyError):
        data.bounds["anchor"]


def test_trimming_shrinks_the_box_but_keeps_every_camera_inside():
    cams = [camera(x) for x in (0.0, -0.5, 0.5)]
    obs = [OnlineObservation("s", i, torch.full((3, 16, 20), 0.5), c) for i, c in enumerate(cams)]
    depths = torch.full((3, 16, 20), 3.0)
    depths[:, 0, :3] = 40.0  # a few very far samples (window) dominate the untrimmed hull
    context = RGBDContext(obs, depths)
    full = context_depth_bounds(context, cams[0].c2w)
    trimmed = context_depth_bounds(context, cams[0].c2w, trim=0.01)
    assert (trimmed[1] - trimmed[0]).prod() < (full[1] - full[0]).prod()
    for c in cams:
        center = c.c2w[:3, 3]
        assert torch.all(trimmed[0] <= center) and torch.all(center <= trimmed[1])
    assert TrimmedDepthBoundsSceneData.TRIM == 0.01 and DepthBoundsSceneData.TRIM == 0.0
