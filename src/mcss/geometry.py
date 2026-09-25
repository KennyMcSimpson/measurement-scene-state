"""Camera, ray, and bounded-volume geometry in the OpenCV convention."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import Tensor

from mcss.types import Cameras


def make_intrinsics(
    image_size: tuple[int, int], fov_x_degrees: float, *, device: torch.device | None = None
) -> Tensor:
    """Create an OpenCV pinhole matrix for an image with square pixels."""

    height, width = image_size
    if height <= 0 or width <= 0:
        raise ValueError("image_size values must be positive")
    if not 0.0 < fov_x_degrees < 179.0:
        raise ValueError("fov_x_degrees must be in (0, 179)")
    focal = (width - 1) / (2.0 * math.tan(math.radians(fov_x_degrees) / 2.0))
    return torch.tensor(
        [[focal, 0.0, (width - 1) / 2.0], [0.0, focal, (height - 1) / 2.0], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
        device=device,
    )


def look_at(eye: Tensor, target: Tensor, up: Tensor | None = None) -> Tensor:
    """Return camera-to-world pose where camera axes are x-right, y-down, z-forward."""

    if eye.shape != (3,) or target.shape != (3,):
        raise ValueError("look_at accepts eye and target tensors of shape [3]")
    if up is None:
        up = torch.tensor([0.0, 1.0, 0.0], dtype=eye.dtype, device=eye.device)
    if up.shape != (3,):
        raise ValueError("up must have shape [3]")
    forward = _normalize(target - eye)
    right = _normalize(torch.linalg.cross(up, forward))
    camera_up = _normalize(torch.linalg.cross(forward, right))
    pose = torch.eye(4, dtype=eye.dtype, device=eye.device)
    pose[:3, 0] = right
    pose[:3, 1] = -camera_up
    pose[:3, 2] = forward
    pose[:3, 3] = eye
    return pose


def pixel_grid(height: int, width: int, *, device: torch.device, dtype: torch.dtype) -> Tensor:
    """Return homogeneous pixel centers [H, W, 3] with integer OpenCV coordinates."""

    if height <= 0 or width <= 0:
        raise ValueError("height and width must be positive")
    y, x = torch.meshgrid(
        torch.arange(height, device=device, dtype=dtype),
        torch.arange(width, device=device, dtype=dtype),
        indexing="ij",
    )
    return torch.stack((x, y, torch.ones_like(x)), dim=-1)


def generate_rays(cameras: Cameras) -> tuple[Tensor, Tensor]:
    """Generate normalized world-space rays with shape ``[..., H, W, 3]``."""

    height, width = cameras.image_size
    pixels = pixel_grid(height, width, device=cameras.device, dtype=cameras.dtype)
    inverse_intrinsics = torch.linalg.inv(cameras.intrinsics)
    directions_camera = torch.einsum("...ij,hwj->...hwi", inverse_intrinsics, pixels)
    directions_camera = _normalize(directions_camera)
    rotation = cameras.c2w[..., :3, :3]
    directions_world = torch.einsum("...ij,...hwj->...hwi", rotation, directions_camera)
    directions_world = _normalize(directions_world)
    origins = cameras.c2w[..., :3, 3].unsqueeze(-2).unsqueeze(-2).expand_as(directions_world)
    return origins, directions_world


def project_world(points: Tensor, cameras: Cameras) -> tuple[Tensor, Tensor, Tensor]:
    """Project world points shaped ``[..., N, 3]`` through matching camera leading dimensions."""

    if points.ndim < 2 or points.shape[-1] != 3:
        raise ValueError("points must have shape [..., N, 3]")
    if points.shape[:-2] != cameras.leading_shape:
        raise ValueError("points leading dimensions must match cameras")
    homogeneous = torch.cat((points, torch.ones_like(points[..., :1])), dim=-1)
    w2c = torch.linalg.inv(cameras.c2w)
    camera_points = torch.einsum("...ij,...nj->...ni", w2c[..., :3, :], homogeneous)
    depth = camera_points[..., 2]
    projected = torch.einsum("...ij,...nj->...ni", cameras.intrinsics, camera_points)
    pixels = projected[..., :2] / projected[..., 2:].clamp_min(torch.finfo(points.dtype).eps)
    height, width = cameras.image_size
    valid = (
        (depth > torch.finfo(points.dtype).eps)
        & (pixels[..., 0] >= 0)
        & (pixels[..., 0] <= width - 1)
        & (pixels[..., 1] >= 0)
        & (pixels[..., 1] <= height - 1)
    )
    return pixels, depth, valid


def intersect_aabb(
    origins: Tensor, directions: Tensor, bounds: Tensor
) -> tuple[Tensor, Tensor, Tensor]:
    """Intersect normalized or non-normalized rays with one broadcastable axis-aligned box."""

    if origins.shape != directions.shape or origins.shape[-1] != 3:
        raise ValueError("origins and directions must share shape [..., 3]")
    if bounds.shape[-2:] != (2, 3):
        raise ValueError("bounds must have shape [..., 2, 3]")
    lower = bounds[..., 0, :]
    upper = bounds[..., 1, :]
    while lower.ndim < origins.ndim:
        lower = lower.unsqueeze(-2)
        upper = upper.unsqueeze(-2)
    eps = torch.finfo(origins.dtype).eps
    parallel = directions.abs() <= eps
    safe_directions = torch.where(parallel, torch.ones_like(directions), directions)
    t0 = (lower - origins) / safe_directions
    t1 = (upper - origins) / safe_directions
    minimum = torch.minimum(t0, t1)
    maximum = torch.maximum(t0, t1)
    inside_parallel_slab = (origins >= lower) & (origins <= upper)
    minimum = torch.where(
        parallel & inside_parallel_slab, torch.full_like(minimum, -torch.inf), minimum
    )
    maximum = torch.where(
        parallel & inside_parallel_slab, torch.full_like(maximum, torch.inf), maximum
    )
    minimum = torch.where(
        parallel & ~inside_parallel_slab, torch.full_like(minimum, torch.inf), minimum
    )
    maximum = torch.where(
        parallel & ~inside_parallel_slab, torch.full_like(maximum, -torch.inf), maximum
    )
    near_raw = minimum.amax(dim=-1)
    far = maximum.amin(dim=-1)
    hit = (far >= torch.maximum(near_raw, torch.zeros_like(near_raw))) & (far > 0)
    near = near_raw.clamp_min(0.0)
    near = torch.where(hit, near, torch.zeros_like(near))
    far = torch.where(hit, far, torch.zeros_like(far))
    return near, far, hit


def make_voxel_centers(bounds: Tensor, resolution: int | Sequence[int]) -> Tensor:
    """Create world-space center coordinates shaped ``[B, D, H, W, 3]``."""

    if bounds.ndim != 3 or bounds.shape[1:] != (2, 3):
        raise ValueError("bounds must have shape [B, 2, 3]")
    if isinstance(resolution, int):
        depth = height = width = resolution
    else:
        if len(resolution) != 3:
            raise ValueError("resolution sequence must contain D, H, W")
        depth, height, width = resolution
    if min(depth, height, width) <= 0:
        raise ValueError("resolution values must be positive")
    z, y, x = torch.meshgrid(
        (torch.arange(depth, device=bounds.device, dtype=bounds.dtype) + 0.5) / depth,
        (torch.arange(height, device=bounds.device, dtype=bounds.dtype) + 0.5) / height,
        (torch.arange(width, device=bounds.device, dtype=bounds.dtype) + 0.5) / width,
        indexing="ij",
    )
    fractions = torch.stack((x, y, z), dim=-1).unsqueeze(0)
    minimum = bounds[:, 0].view(-1, 1, 1, 1, 3)
    extent = (bounds[:, 1] - bounds[:, 0]).view(-1, 1, 1, 1, 3)
    return minimum + fractions * extent


def transform_cameras(cameras: Cameras, transform: Tensor) -> Cameras:
    """Express calibrated cameras in a new frame using a rigid old-to-new transform."""

    _validate_rigid_transform(transform)
    expanded = transform.to(device=cameras.device, dtype=cameras.dtype)
    while expanded.ndim < cameras.c2w.ndim:
        expanded = expanded.unsqueeze(-3)
    return Cameras(cameras.intrinsics.clone(), expanded @ cameras.c2w, cameras.image_size)


def transform_points(points: Tensor, transform: Tensor) -> Tensor:
    """Apply a broadcastable rigid transform to Euclidean points shaped ``[..., 3]``."""

    if points.ndim < 1 or points.shape[-1] != 3:
        raise ValueError("points must have shape [..., 3]")
    _validate_rigid_transform(transform)
    homogeneous = torch.cat((points, torch.ones_like(points[..., :1])), dim=-1)
    expanded = transform.to(device=points.device, dtype=points.dtype)
    while expanded.ndim < homogeneous.ndim + 1:
        expanded = expanded.unsqueeze(-3)
    return (homogeneous.unsqueeze(-2) @ expanded.transpose(-1, -2)).squeeze(-2)[..., :3]


def transform_directions(directions: Tensor, transform: Tensor) -> Tensor:
    """Rotate direction vectors without applying the transform translation."""

    if directions.ndim < 1 or directions.shape[-1] != 3:
        raise ValueError("directions must have shape [..., 3]")
    _validate_rigid_transform(transform)
    homogeneous = torch.cat((directions, torch.zeros_like(directions[..., :1])), dim=-1)
    expanded = transform.to(device=directions.device, dtype=directions.dtype)
    while expanded.ndim < homogeneous.ndim + 1:
        expanded = expanded.unsqueeze(-3)
    return (homogeneous.unsqueeze(-2) @ expanded.transpose(-1, -2)).squeeze(-2)[..., :3]


def _validate_rigid_transform(transform: Tensor) -> None:
    if transform.ndim < 2 or transform.shape[-2:] != (4, 4):
        raise ValueError("transform must have shape [..., 4, 4]")
    if not torch.isfinite(transform).all():
        raise ValueError("transform must contain only finite values")
    expected_row = torch.tensor(
        [0.0, 0.0, 0.0, 1.0], device=transform.device, dtype=transform.dtype
    )
    if not torch.allclose(transform[..., 3, :], expected_row, atol=1e-5, rtol=1e-5):
        raise ValueError("transform must have homogeneous final row [0, 0, 0, 1]")


def _normalize(vectors: Tensor) -> Tensor:
    return vectors / torch.linalg.vector_norm(vectors, dim=-1, keepdim=True).clamp_min(
        torch.finfo(vectors.dtype).eps
    )
