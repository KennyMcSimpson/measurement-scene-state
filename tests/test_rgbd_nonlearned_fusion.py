"""Non-learned RGB-D fusion: a measured plane is rendered back at its measured depth."""

import torch

from mcss.dynamic.types import OnlineObservation
from mcss.geometry import transform_cameras
from mcss.measurements import FixedMeasurementRenderer
from mcss.mechanism_pilot.rgbd_evidence_carrier import RGBDContext
from mcss.mechanism_pilot.rgbd_nonlearned_fusion import fused_state, voxel_centers
from mcss.types import Cameras


def camera(x, height=32, width=40, focal=30.0):
    cx, cy = (width - 1) / 2, (height - 1) / 2
    pose = torch.eye(4)
    pose[0, 3] = x
    intrinsics = torch.tensor([[focal, 0, cx], [0, focal, cy], [0, 0, 1.0]])
    return Cameras(intrinsics, pose, (height, width))


def ray_distance(cam, z):
    height, width = cam.image_size
    v, u = torch.meshgrid(
        torch.arange(height, dtype=torch.float32),
        torch.arange(width, dtype=torch.float32),
        indexing="ij",
    )
    directions = torch.stack((u, v, torch.ones_like(u)), -1) @ torch.linalg.inv(cam.intrinsics).T
    return z * directions.norm(dim=-1)


def test_voxel_centres_follow_the_renderer_layout():
    bounds = torch.tensor([[0.0, 0.0, 0.0], [16.0, 32.0, 48.0]])
    points = voxel_centers(bounds).reshape(16, 16, 16, 3)
    assert torch.allclose(points[0, 0, 0], torch.tensor([0.5, 1.0, 1.5]))
    assert torch.allclose(points[0, 0, 1], torch.tensor([1.5, 1.0, 1.5]))  # w -> x
    assert torch.allclose(points[0, 1, 0], torch.tensor([0.5, 3.0, 1.5]))  # h -> y
    assert torch.allclose(points[1, 0, 0], torch.tensor([0.5, 1.0, 4.5]))  # d -> z


def render_plane(z):
    cams = [camera(x) for x in (0.0, -0.3, 0.3)]
    obs = [OnlineObservation("s", i, torch.full((3, 32, 40), 0.4), c) for i, c in enumerate(cams)]
    depths = torch.stack([ray_distance(c, torch.full((32, 40), z)) for c in cams])
    bounds = torch.tensor([[-2.0, -2.0, 0.2], [2.0, 2.0, 4.2]])
    params = {"k": 0.5, "a": 64.0, "b": 4.0, "c": -4.0}
    state, anchor = fused_state(RGBDContext(obs, depths), bounds, params)
    query = transform_cameras(
        Cameras(cams[0].intrinsics[None, None], cams[0].c2w[None, None], (32, 40)),
        torch.linalg.inv(anchor),
    )
    pred = FixedMeasurementRenderer(n_samples=64)(state, query, ("depth", "visibility", "rgb"))
    truth = ray_distance(cams[0], torch.full((32, 40), z))[12:20, 16:24]
    return pred, truth


def test_fused_plane_renders_near_its_measured_depth_and_moves_with_it():
    near, truth_near = render_plane(2.0)
    far, truth_far = render_plane(3.0)
    depth_near = near["depth"][0, 0, 0][12:20, 16:24]
    depth_far = far["depth"][0, 0, 0][12:20, 16:24]
    # A sharp, opaque fused surface: within about one 0.25 m voxel of the measured depth.
    assert (depth_near - truth_near).abs().median() < 0.25
    assert (depth_far - truth_far).abs().median() < 0.25
    shift = (depth_far - depth_near).median()
    assert 0.8 < shift < 1.2
    assert near["visibility"][0, 0, 0][12:20, 16:24].min() > 0.97
    assert (near["rgb"][0, 0][:, 12:20, 16:24] - 0.4).abs().max() < 0.05
