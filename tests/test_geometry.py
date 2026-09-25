import torch

from mcss.geometry import (
    generate_rays,
    intersect_aabb,
    look_at,
    make_intrinsics,
    make_voxel_centers,
    project_world,
    transform_cameras,
    transform_directions,
    transform_points,
)
from mcss.types import Cameras


def test_identity_camera_center_ray_points_forward() -> None:
    intrinsics = torch.eye(3).view(1, 1, 3, 3)
    c2w = torch.eye(4).view(1, 1, 4, 4)
    cameras = Cameras(intrinsics, c2w, (1, 1))

    origins, directions = generate_rays(cameras)

    torch.testing.assert_close(origins[0, 0, 0, 0], torch.zeros(3))
    torch.testing.assert_close(directions[0, 0, 0, 0], torch.tensor([0.0, 0.0, 1.0]))


def test_projection_roundtrip_for_optical_axis() -> None:
    intrinsics = torch.tensor([[[[100.0, 0.0, 5.0], [0.0, 100.0, 4.0], [0.0, 0.0, 1.0]]]])
    cameras = Cameras(intrinsics, torch.eye(4).view(1, 1, 4, 4), (9, 11))
    points = torch.tensor([[[[0.0, 0.0, 2.0]]]])

    pixels, depth, valid = project_world(points, cameras)

    torch.testing.assert_close(pixels[0, 0, 0], torch.tensor([5.0, 4.0]))
    torch.testing.assert_close(depth[0, 0, 0], torch.tensor(2.0))
    assert valid.item()


def test_ray_aabb_intersection_marks_hits_and_misses() -> None:
    origins = torch.tensor([[0.0, 0.0, -3.0], [3.0, 0.0, -3.0]])
    directions = torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
    bounds = torch.tensor([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]])

    near, far, hit = intersect_aabb(origins, directions, bounds)

    torch.testing.assert_close(near[0], torch.tensor(2.0))
    torch.testing.assert_close(far[0], torch.tensor(4.0))
    assert hit.tolist() == [True, False]


def test_camera_and_voxel_helpers_have_expected_conventions() -> None:
    pose = look_at(torch.tensor([0.0, 0.0, -3.0]), torch.zeros(3))
    intrinsics = make_intrinsics((9, 11), 60.0)
    centers = make_voxel_centers(torch.tensor([[[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]]]), 2)

    torch.testing.assert_close(pose[:3, 2], torch.tensor([0.0, 0.0, 1.0]))
    torch.testing.assert_close(intrinsics[0, 2], torch.tensor(5.0))
    assert centers.shape == (1, 2, 2, 2, 3)
    torch.testing.assert_close(centers.amin(dim=(1, 2, 3))[0], torch.full((3,), -0.5))
    torch.testing.assert_close(centers.amax(dim=(1, 2, 3))[0], torch.full((3,), 0.5))


def test_rigid_transform_rebases_cameras_points_and_directions() -> None:
    rotation = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    anchor = torch.eye(4)
    anchor[:3, :3] = rotation
    anchor[:3, 3] = torch.tensor([10.0, 0.0, 0.0])
    second = anchor.clone()
    second[:3, 3] = torch.tensor([10.0, 1.0, 0.0])
    cameras = Cameras(torch.eye(3).repeat(2, 1, 1), torch.stack((anchor, second)), (1, 1))
    world_to_anchor = torch.linalg.inv(anchor)

    local_cameras = transform_cameras(cameras, world_to_anchor)
    local_points = transform_points(torch.tensor([[10.0, 2.0, 3.0]]), world_to_anchor)
    local_directions = transform_directions(torch.tensor([[0.0, 1.0, 0.0]]), world_to_anchor)

    torch.testing.assert_close(local_cameras.c2w[0], torch.eye(4))
    torch.testing.assert_close(local_cameras.c2w[1, :3, 3], torch.tensor([1.0, 0.0, 0.0]))
    torch.testing.assert_close(local_points, torch.tensor([[2.0, 0.0, 3.0]]))
    torch.testing.assert_close(local_directions, torch.tensor([[1.0, 0.0, 0.0]]))


def test_voxel_centers_accept_anisotropic_dhw_resolution() -> None:
    bounds = torch.tensor([[[-2.0, -3.0, 0.0], [2.0, 3.0, 8.0]]])

    centers = make_voxel_centers(bounds, (4, 3, 2))

    assert centers.shape == (1, 4, 3, 2, 3)
    torch.testing.assert_close(centers[0, 0, 0, 0], torch.tensor([-1.0, -2.0, 1.0]))
    torch.testing.assert_close(centers[0, -1, -1, -1], torch.tensor([1.0, 2.0, 7.0]))
