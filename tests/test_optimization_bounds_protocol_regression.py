"""End-to-end CPU regressions for query isolation and cross-bounds optimizer identity."""

import json
from copy import deepcopy
from dataclasses import asdict

import numpy as np
import torch
from PIL import Image

from mcss.dynamic.types import hash_scene_state
from mcss.mechanism_pilot.budget_state_optimization import run_budget_trajectory
from mcss.mechanism_pilot.direct_capacity_contracts import ContextOnlyLoader
from mcss.mechanism_pilot.direct_capacity_optimization import DirectOptimizationConfig
from mcss.mechanism_pilot.optimization_bounds_contracts import (
    FrozenTrainingPrior,
    context_camera_bundle,
    frozen_gt_free_bounds,
)


def synthetic_record(directory):
    directory.mkdir()
    record = {
        "scene_id": "synthetic",
        "roles": {"context_a": [0, 1], "context_b": [0, 2], "primary_query": [8, 9]},
        "frames": [],
    }
    for fid in (0, 1, 2, 8, 9):
        rgb, depth = directory / f"{fid}.png", directory / f"{fid}.npy"
        Image.fromarray(np.full((4, 4, 3), 40 + fid, np.uint8)).save(rgb)
        np.save(depth, np.full((4, 4), 2.0 + fid * 0.01, np.float32))
        pose = np.eye(4)
        pose[0, 3] = fid * 0.1
        record["frames"].append(
            {
                "frame_id": fid,
                "rgb": str(rgb),
                "depth": str(depth),
                "intrinsics": [[4.0, 0.0, 1.5], [0.0, 4.0, 1.5], [0.0, 0.0, 1.0]],
                "c2w": pose.tolist(),
            }
        )
    return record


def load_context(record, root, log):
    return ContextOnlyLoader(
        record,
        (4, 4),
        root,
        device="cpu",
        track="RGBD",
        access_log=log,
        capacity_scene_ids=("synthetic",),
        holdout_scene_ids=("protected",),
    ).context("A")


def optimize(batch, bounds, directory):
    config = DirectOptimizationConfig(steps=10, batch_rays=8, checkpoint_interval=10)
    run_budget_trajectory(
        batch,
        bounds,
        directory,
        seed=20260927,
        budgets=(10,),
        config=config,
        engineering_fixture=True,
    )
    state = torch.load(directory / "budget_10/FIXED_BUDGET/state.pt", weights_only=False)
    return state, json.loads((directory / "lock.json").read_text()), config


def test_query_rgb_depth_and_camera_perturbation_cannot_change_context_optimized_state(tmp_path):
    torch.set_num_threads(1)
    record = synthetic_record(tmp_path / "media")
    bounds = torch.tensor([[-6.0, -4.0, -6.0], [6.0, 4.0, 6.0]])
    original_log = []
    original_batch = load_context(record, tmp_path, original_log)
    original_state, original_lock, _ = optimize(original_batch, bounds, tmp_path / "original")

    changed = deepcopy(record)
    for frame in changed["frames"]:
        if frame["frame_id"] not in (8, 9):
            continue
        Image.fromarray(np.full((4, 4, 3), 250, np.uint8)).save(frame["rgb"])
        np.save(frame["depth"], np.full((4, 4), 200.0, np.float32))
        frame["intrinsics"] = [[120.0, 0.0, 20.0], [0.0, 90.0, -5.0], [0.0, 0.0, 1.0]]
        frame["c2w"][0][3] += 75
        frame["c2w"][2][3] -= 40
    changed_log = []
    changed_batch = load_context(changed, tmp_path, changed_log)
    changed_state, changed_lock, _ = optimize(changed_batch, bounds, tmp_path / "changed")

    assert all(event["frame_id"] in (0, 1) for event in original_log + changed_log)
    assert original_batch.frame_ids == changed_batch.frame_ids == (0, 1)
    torch.testing.assert_close(original_batch.rgb, changed_batch.rgb, rtol=0, atol=0)
    torch.testing.assert_close(original_batch.depth, changed_batch.depth, rtol=0, atol=0)
    torch.testing.assert_close(
        original_batch.cameras.c2w, changed_batch.cameras.c2w, rtol=0, atol=0
    )
    assert hash_scene_state(original_state) == hash_scene_state(changed_state)
    assert original_lock == changed_lock


def test_two_actual_bounds_runs_preserve_optimizer_loss_grid_and_renderer_configuration(tmp_path):
    torch.set_num_threads(1)
    record = synthetic_record(tmp_path / "media")
    batch = load_context(record, tmp_path, [])
    current = torch.tensor([[-6.0, -4.0, -6.0], [6.0, 4.0, 6.0]])
    alternative = frozen_gt_free_bounds(
        context_camera_bundle(record, "A", (4, 4)), FrozenTrainingPrior(0.2, 5.0, "a" * 64)
    )
    # Production runner materializes frozen JSON bounds in float32 for the renderer.
    alternative = alternative.to(dtype=batch.rgb.dtype)
    assert not torch.equal(current, alternative)
    current_state, current_lock, config = optimize(batch, current, tmp_path / "current")
    alternative_state, alternative_lock, _ = optimize(batch, alternative, tmp_path / "alternative")
    assert current_lock["config"] == alternative_lock["config"] == asdict(config)
    assert current_lock["grid"] == alternative_lock["grid"] == 16
    assert current_lock["samples"] == alternative_lock["samples"] == 64
    assert current_lock["parameter_count"] == alternative_lock["parameter_count"] == 4 * 16**3
    assert current_lock["supervision_frame_ids"] == alternative_lock["supervision_frame_ids"]
    assert current_lock["bounds"] != alternative_lock["bounds"]
    assert (
        current_state.density_logits.shape
        == alternative_state.density_logits.shape
        == (1, 1, 16, 16, 16)
    )
    for state in (current_state, alternative_state):
        assert torch.isfinite(state.density_logits).all()
        assert state.features is state.appearance is None
