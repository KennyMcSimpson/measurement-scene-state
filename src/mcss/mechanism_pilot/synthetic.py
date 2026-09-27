"""Analytic fixture, independent of the network, exclusively for engineering smoke.

An infinite textured front plane supplies exact normalized-ray distance. These
three procedural fixtures are not real scenes, trained scenes, or confirmation.
Query fixtures must only be requested by the evaluator after sealing all states.
"""

import torch

from mcss.dynamic.types import OnlineObservation
from mcss.geometry import generate_rays, make_intrinsics
from mcss.types import Cameras


def fixture_camera(frame_id: int, image_size=(16, 16)):
    """Generate calibration without generating any RGB/depth labels."""
    pose = torch.eye(4)
    pose[0, 3] = 0.12 * frame_id
    pose[1, 3] = 0.06 * (frame_id % 3)
    camera = Cameras(make_intrinsics(image_size, 60.0), pose, image_size)
    return camera


def fixture(scene_index: int, frame_id: int, image_size=(16, 16)):
    camera = fixture_camera(frame_id, image_size)
    origins, directions = generate_rays(camera)
    distance = (4.0 + 0.7 * scene_index - origins[..., 2]) / directions[..., 2]
    points = origins + distance[..., None] * directions
    rgb = torch.stack(
        [0.5 + 0.4 * torch.sin(points[..., i % 2] * (i + 1) + scene_index) for i in range(3)],
        dim=0,
    )
    return camera, rgb, distance, points


def arrived_observation(scene_index: int, frame_id: int):
    if frame_id not in range(8):
        raise ValueError("Query fixture cannot enter arrived observations")
    camera, rgb, _, _ = fixture(scene_index, frame_id)
    return OnlineObservation(f"synthetic-{scene_index}", frame_id, rgb, camera)
