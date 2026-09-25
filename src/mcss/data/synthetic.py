"""Deterministic analytic sparse-view scenes for end-to-end verification."""

from __future__ import annotations

import math

import torch
from torch import Tensor
from torch.utils.data import Dataset

from mcss.geometry import generate_rays, intersect_aabb, look_at, make_intrinsics
from mcss.types import Cameras, SceneExample


class SyntheticSceneDataset(Dataset[SceneExample]):
    """Render colored spheres and boxes without external assets or learned code."""

    def __init__(
        self,
        *,
        length: int = 64,
        image_size: tuple[int, int] = (48, 64),
        context_views: int = 2,
        target_views: int = 1,
        seed: int = 0,
    ) -> None:
        if length < 1:
            raise ValueError("length must be positive")
        if context_views < 1 or target_views < 1:
            raise ValueError("context_views and target_views must be positive")
        if min(image_size) < 2:
            raise ValueError("image dimensions must be at least two")
        self.length = length
        self.image_size = image_size
        self.context_views = context_views
        self.target_views = target_views
        self.seed = seed

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> SceneExample:
        if not 0 <= index < self.length:
            raise IndexError(index)
        generator = torch.Generator().manual_seed(self.seed + index * 1_000_003)
        view_count = self.context_views + self.target_views
        cameras = self._make_cameras(view_count, index, generator)
        bounds = torch.tensor([[-1.2, -1.2, -1.2], [1.2, 1.2, 1.2]], dtype=torch.float32)
        if index % 2 == 0:
            rendered = _render_sphere(cameras, generator)
        else:
            rendered = _render_box(cameras, generator)
        context_indices = torch.arange(self.context_views)
        target_indices = torch.arange(self.context_views, view_count)
        return SceneExample(
            context_rgb=rendered["rgb"][context_indices],
            context_cameras=cameras.select_views(context_indices),
            target_rgb=rendered["rgb"][target_indices],
            target_cameras=cameras.select_views(target_indices),
            bounds=bounds,
            target_depth=rendered["depth"][target_indices],
            target_normal=rendered["normal"][target_indices],
            target_point=rendered["point"][target_indices],
            target_visibility=rendered["visibility"][target_indices],
            scene_id=f"synthetic_{index:06d}",
        )

    def _make_cameras(self, view_count: int, index: int, generator: torch.Generator) -> Cameras:
        phase = float(torch.rand((), generator=generator)) * 2.0 * math.pi
        elevation = -0.15 + float(torch.rand((), generator=generator)) * 0.5
        poses = []
        for view_index in range(view_count):
            angle = phase + 2.0 * math.pi * view_index / view_count
            radius = 2.8 + 0.15 * math.sin(index + view_index)
            eye = torch.tensor(
                [radius * math.sin(angle), elevation, -radius * math.cos(angle)],
                dtype=torch.float32,
            )
            poses.append(look_at(eye, torch.zeros(3)))
        intrinsics = make_intrinsics(self.image_size, 55.0)
        return Cameras(
            intrinsics.expand(view_count, 3, 3).clone(), torch.stack(poses), self.image_size
        )


def _render_sphere(cameras: Cameras, generator: torch.Generator) -> dict[str, Tensor]:
    origins, directions = generate_rays(cameras)
    center = (torch.rand(3, generator=generator) - 0.5) * torch.tensor([0.25, 0.2, 0.25])
    radius = 0.48 + float(torch.rand((), generator=generator)) * 0.18
    offset = origins - center
    half_b = (offset * directions).sum(dim=-1)
    constant = (offset * offset).sum(dim=-1) - radius**2
    discriminant = half_b.square() - constant
    square_root = discriminant.clamp_min(0.0).sqrt()
    near = -half_b - square_root
    far = -half_b + square_root
    distance = torch.where(near > 1e-5, near, far)
    visible = (discriminant >= 0.0) & (distance > 1e-5)
    distance = torch.where(visible, distance, torch.zeros_like(distance))
    point = origins + directions * distance.unsqueeze(-1)
    normal = (point - center) / radius
    normal = torch.where(visible.unsqueeze(-1), normal, torch.zeros_like(normal))
    base_color = 0.2 + 0.65 * torch.rand(3, generator=generator)
    rgb = _shade(point, normal, visible, base_color)
    return _channel_first(rgb, distance, normal, point, visible)


def _render_box(cameras: Cameras, generator: torch.Generator) -> dict[str, Tensor]:
    origins, directions = generate_rays(cameras)
    center = (torch.rand(3, generator=generator) - 0.5) * 0.25
    half_extent = 0.35 + 0.25 * torch.rand(3, generator=generator)
    bounds = torch.stack((center - half_extent, center + half_extent))
    distance, _, visible = intersect_aabb(origins, directions, bounds)
    point = origins + directions * distance.unsqueeze(-1)
    local = (point - center) / half_extent
    face_axis = local.abs().argmax(dim=-1)
    normal = torch.zeros_like(point)
    normal.scatter_(
        -1, face_axis.unsqueeze(-1), torch.gather(local.sign(), -1, face_axis.unsqueeze(-1))
    )
    normal = torch.where(visible.unsqueeze(-1), normal, torch.zeros_like(normal))
    base_color = 0.2 + 0.65 * torch.rand(3, generator=generator)
    rgb = _shade(point, normal, visible, base_color)
    return _channel_first(rgb, distance, normal, point, visible)


def _shade(point: Tensor, normal: Tensor, visible: Tensor, base_color: Tensor) -> Tensor:
    light = torch.tensor([0.4, -0.6, -0.7], dtype=point.dtype, device=point.device)
    light = light / torch.linalg.vector_norm(light)
    lambert = (normal * light).sum(dim=-1).clamp_min(0.0)
    texture = 0.9 + 0.1 * torch.sin(point[..., 0] * 8.0 + point[..., 2] * 5.0)
    surface = base_color * (0.25 + 0.75 * lambert).unsqueeze(-1) * texture.unsqueeze(-1)
    background = torch.full_like(surface, 0.02)
    return torch.where(visible.unsqueeze(-1), surface.clamp(0.0, 1.0), background)


def _channel_first(
    rgb: Tensor, depth: Tensor, normal: Tensor, point: Tensor, visible: Tensor
) -> dict[str, Tensor]:
    return {
        "rgb": rgb.permute(0, 3, 1, 2).contiguous(),
        "depth": depth.unsqueeze(1),
        "normal": normal.permute(0, 3, 1, 2).contiguous(),
        "point": point.permute(0, 3, 1, 2).contiguous(),
        "visibility": visible.to(rgb.dtype).unsqueeze(1),
    }
