import pytest
import torch

from mcss.geometry import make_intrinsics
from mcss.mechanism_pilot.static_geometry import (
    depth_to_ray_distance,
    support_audit,
    surface_geometry,
    volume_geometry,
)
from mcss.types import Cameras


def camera():
    return Cameras(
        make_intrinsics((8, 10), 60).double(), torch.eye(4, dtype=torch.float64), (8, 10)
    )


def test_projection_depth_roundtrip_uses_gt_unit_ray():
    c = camera()
    depth = torch.full((8, 10), 3.0, dtype=torch.float64)
    _, _, _, z, pe, de = surface_geometry(c, depth)
    assert pe.max() < 1e-10
    assert de.max() < 1e-10
    assert z.min() < 3


def test_camera_z_conversion_differs_off_axis():
    c = camera()
    z = torch.ones((8, 10), dtype=torch.float64)
    ray = depth_to_ray_distance(z, c, semantics="camera_z_meters")
    assert ray.min() > 1
    assert torch.equal(depth_to_ray_distance(ray, c, semantics="ray_distance_meters"), ray)
    with pytest.raises(ValueError):
        depth_to_ray_distance(z, c, semantics="unknown")


def test_volume_geometry_is_score_independent_and_empty_support_explicit():
    points = torch.tensor([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    centers = torch.tensor([[0.0, 0.0, 0.0]])
    bounds = torch.tensor([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]])
    outside, nearest, distance, covered = volume_geometry(
        points, centers, bounds, torch.tensor([False]), 0.5
    )
    assert outside.tolist() == [0.0, 2.0]
    assert nearest.tolist() == [0.0, 3.0]
    assert torch.isinf(distance).all()
    assert not covered.any()


def test_frustum_support_requires_two_views_and_world_anchor_translation():
    c = camera()
    c.c2w[0, 3] = 7
    p = torch.tensor([[0.0, 0.0, 2.0], [0.0, 0.0, -2.0]], dtype=torch.float64)
    bounds = torch.tensor([[-3.0, -3.0, -3.0], [3.0, 3.0, 3.0]], dtype=torch.float64)
    one, mask = support_audit(p, bounds, [(0, c)], c.c2w)
    assert not mask.any()
    two, mask = support_audit(p, bounds, [(0, c), (1, c)], c.c2w)
    assert mask.tolist() == [True, False]
    assert one["candidate_world_xyz"][0] == [7.0, 0.0, 2.0]
    assert two["support_count"] == [2, 0]
