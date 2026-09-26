import pytest
import torch

from mcss.mechanism_pilot.visibility import common_visibility_masks
from mcss.types import Cameras


def _camera(pose=None):
    return Cameras(
        torch.tensor([[2.0, 0.0, 1.0], [0.0, 2.0, 1.0], [0.0, 0.0, 1.0]]),
        torch.eye(4) if pose is None else pose,
        (3, 3),
    )


def test_identical_views_accept_ray_distance_not_axis_depth():
    camera = _camera()
    depth = torch.full((3, 3), 2.0)
    result = common_visibility_masks(camera, depth, [(camera, depth)], [(camera, depth)])
    assert result["common_frustum"].all()
    assert result["depth_consistent_common"].all()


def test_occlusion_and_group_union_then_intersection():
    camera = _camera()
    depth = torch.full((3, 3), 2.0)
    closer = torch.ones_like(depth)
    result = common_visibility_masks(camera, depth, [(camera, closer)], [(camera, depth)])
    assert result["common_frustum"].all()
    assert not result["depth_consistent_common"].any()
    result = common_visibility_masks(
        camera, depth, [(camera, closer), (camera, depth)], [(camera, depth)]
    )
    assert result["depth_consistent_common"].all()


def test_invalid_depth_masks_and_input_not_mutated():
    camera = _camera()
    depth = torch.full((3, 3), 2.0)
    depth[0] = torch.tensor([float("nan"), 0.0, -1.0])
    target = torch.full((3, 3), 2.0)
    target[1, 1] = float("inf")
    before = target.clone()
    result = common_visibility_masks(camera, depth[None], [(camera, target)], [(camera, target)])
    assert not result["common_frustum"][0].any()
    assert not result["depth_consistent_common"][0].any()
    assert result["common_frustum"][1, 1]
    assert not result["depth_consistent_common"][1, 1]
    assert torch.equal(before, target)


def test_nonidentity_pose_translation_uses_context_ray_distance():
    query_pose = torch.eye(4)
    query_pose[:3, :3] = torch.tensor([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
    query_pose[:3, 3] = torch.tensor([3.0, -2.0, 1.0])
    context_pose = query_pose.clone()
    context_pose[:3, 3] += query_pose[:3, 2]  # one metre forward along query optical axis
    query = _camera(query_pose)
    context = _camera(context_pose)
    depth = torch.zeros((3, 3))
    depth[1, 1] = 2.0
    target = torch.ones((3, 3))
    result = common_visibility_masks(query, depth, [(context, target)], [(context, target)])
    assert result["depth_consistent_common"].sum() == 1
    assert result["depth_consistent_common"][1, 1]


def test_depth_tolerance_uses_context_gt_and_is_not_free_visibility():
    camera = _camera()
    depth = torch.full((3, 3), 2.06)
    target = torch.full((3, 3), 2.0)
    assert common_visibility_masks(camera, depth, [(camera, target)], [(camera, target)])[
        "depth_consistent_common"
    ].all()
    result = common_visibility_masks(
        camera,
        depth,
        [(camera, target)],
        [(camera, target)],
        absolute_tolerance_m=0.0,
        relative_tolerance=0.01,
    )
    assert not result["depth_consistent_common"].any()
    with pytest.raises(ValueError, match="nonnegative"):
        common_visibility_masks(
            camera, depth, [(camera, target)], [(camera, target)], relative_tolerance=-1
        )
