"""V26: far-focus ray sampling (FAR pixels, the 75% share, the uniform fallback) and the branch."""

import importlib.util
import types
from pathlib import Path

import torch
from test_rgbd_stream_write_v3 import contexts

from mcss.types import Cameras


def script(name):
    path = Path(__file__).parents[1] / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def target(camera, depth=2.0):
    cameras = Cameras(camera.intrinsics[None, None], camera.c2w[None, None], camera.image_size)
    return types.SimpleNamespace(cameras=cameras, depth=torch.full((1, 1, 1, 8, 8), depth))


def test_far_pixels_are_valid_pixels_away_from_every_measured_hit():
    train = script("train_rgbd_far_focus_v26.py")
    warm, _, _, _ = contexts()
    seen = target(warm[0].camera)
    assert train.far_pixels(warm, seen, (8, 8)).numel() == 0
    away = warm[0].camera.c2w.clone()
    away[:3, :3] = away[:3, :3] @ torch.diag(torch.tensor([-1.0, 1.0, -1.0]))
    unseen = target(Cameras(warm[0].camera.intrinsics, away, warm[0].camera.image_size))
    assert train.far_pixels(warm, unseen, (8, 8)).tolist() == list(range(64))
    invalid = target(Cameras(warm[0].camera.intrinsics, away, warm[0].camera.image_size), 0.0)
    assert train.far_pixels(warm, invalid, (8, 8)).numel() == 0


def test_far_rays_are_distinct_far_pixels_up_to_three_quarters_and_none_without_them():
    train = script("train_rgbd_far_focus_v26.py")
    assert train.FAR_FRACTION == 0.75
    few = torch.tensor([3, 7, 11])
    indices = train.far_focus_indices(few, 64, 1024, torch.Generator().manual_seed(5))
    assert indices.shape == (1024,) and indices.dtype == torch.long
    assert sorted(indices[:3].tolist()) == [3, 7, 11]
    assert indices.min() >= 0 and indices.max() < 64
    again = train.far_focus_indices(few, 64, 1024, torch.Generator().manual_seed(5))
    assert torch.equal(indices, again)
    many = torch.arange(2000)
    indices = train.far_focus_indices(many, 4096, 1024, torch.Generator().manual_seed(5))
    assert len(set(indices[:768].tolist())) == 768 and indices[:768].max() < 2000
    uniform = train.far_focus_indices(few[:0], 64, 1024, torch.Generator().manual_seed(5))
    assert torch.equal(
        uniform, torch.randint(64, (1024,), generator=torch.Generator().manual_seed(5))
    )


def test_branch_needs_both_the_far_gain_and_the_hybrid_label():
    analysis = script("analyze_rgbd_far_focus_v26.py")
    assert analysis.branch("SUPPORTED", "ABOVE") == "A"
    assert analysis.branch("SUPPORTED", "NOT_DISTINGUISHABLE") == "B"
    assert analysis.branch("PARTIAL", "ABOVE") == "B"
    assert analysis.branch("NOT_ESTABLISHED", "BELOW") == "C"
