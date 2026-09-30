from copy import deepcopy

import pytest
import torch

from mcss.mechanism_pilot.optimization_bounds_contracts import (
    FrozenTrainingPrior,
    context_camera_bundle,
    frozen_gt_free_bounds,
)


def record():
    return {
        "scene_id": "synthetic",
        "roles": {"context_a": [0, 1, 2], "context_b": [0, 3, 4], "primary_query": [8, 9]},
        "frames": [
            {
                "frame_id": i,
                "intrinsics": [[10.0, 0.0, 2.0], [0.0, 10.0, 2.0], [0.0, 0.0, 1.0]],
                "c2w": torch.eye(4).tolist(),
                "depth": "must_not_read",
                "rgb": "must_not_read",
            }
            for i in (0, 1, 2, 3, 4, 8, 9)
        ],
    }


def test_bounds_ignore_current_scene_depth_and_query_camera():
    r = record()
    prior = FrozenTrainingPrior(0.2, 5.0, "a" * 64)
    expected = frozen_gt_free_bounds(context_camera_bundle(r, "A", (5, 5)), prior)
    changed = deepcopy(r)
    for f in changed["frames"]:
        f["depth"] = object()
        f["rgb"] = object()
        if f["frame_id"] in (8, 9):
            f["c2w"] = object()
            f["intrinsics"] = object()
    actual = frozen_gt_free_bounds(context_camera_bundle(changed, "A", (5, 5)), prior)
    assert torch.equal(actual, expected)


def test_query_role_or_query_in_context_rejected():
    r = record()
    with pytest.raises(PermissionError):
        context_camera_bundle(r, "query")
    r["roles"]["context_a"] = [0, 1, 8]
    with pytest.raises(PermissionError):
        context_camera_bundle(r, "A")


def test_bounds_deterministic_and_camera_only_signature():
    bundle = context_camera_bundle(record(), "B", (5, 5))
    prior = FrozenTrainingPrior(0.2, 5.0, "b" * 64)
    assert torch.equal(frozen_gt_free_bounds(bundle, prior), frozen_gt_free_bounds(bundle, prior))
    with pytest.raises(TypeError):
        frozen_gt_free_bounds(bundle, prior, depth=torch.ones(5, 5))
    with pytest.raises(TypeError):
        frozen_gt_free_bounds(bundle, prior, query_camera=bundle.cameras[0])
    with pytest.raises(TypeError):
        frozen_gt_free_bounds(bundle.cameras, prior)


def test_minimum_extent_and_prior_constraints():
    bundle = context_camera_bundle(record(), "A", (5, 5))
    bounds = frozen_gt_free_bounds(bundle, FrozenTrainingPrior(0.2, 0.3, "c" * 64))
    assert torch.all(bounds[1] - bounds[0] >= 1.0)
    with pytest.raises(ValueError):
        FrozenTrainingPrior(2.0, 1.0, "d" * 64)
