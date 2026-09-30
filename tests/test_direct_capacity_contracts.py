"""Direct capacity context/query and oracle boundaries without carrier inference."""

from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest
import torch
from PIL import Image

from mcss.dynamic.types import hash_scene_state
from mcss.mechanism_pilot.direct_capacity_contracts import (
    CapacityEvaluator,
    ContextCheckpointSelector,
    ContextOnlyLoader,
    ExplicitQueryOracleLoader,
    FrozenOptimConfig,
    StateSealBarrier,
    validate_renderer_diagnostic,
    validate_resolution_sweep,
)
from mcss.types import SceneState


def scene(tmp_path):
    record = {
        "scene_id": "capacity0",
        "roles": {"context_a": [0, 1, 2], "context_b": [0, 3, 4], "primary_query": [8, 9]},
        "frames": [],
    }
    for fid in [0, 1, 2, 3, 4, 8, 9]:
        rgb, depth = tmp_path / f"{fid}.png", tmp_path / f"{fid}.npy"
        Image.fromarray(np.full((4, 4, 3), fid + 30, np.uint8)).save(rgb)
        np.save(depth, np.full((4, 4), 3.0, np.float32))
        record["frames"].append(
            {
                "frame_id": fid,
                "rgb": str(rgb),
                "depth": str(depth),
                "intrinsics": [[4.0, 0.0, 2.0], [0.0, 4.0, 2.0], [0.0, 0.0, 1.0]],
                "c2w": np.eye(4).tolist(),
            }
        )
    return record


def kwargs():
    return {"capacity_scene_ids": ("capacity0",), "holdout_scene_ids": ("held0",)}


def state():
    return SceneState(
        torch.zeros(1, 1, 2, 2, 2),
        torch.full((1, 3, 2, 2, 2), 0.5),
        torch.zeros(1, 1, 2, 2, 2),
        torch.tensor([[[-1.0, -1.0, 0.0], [1.0, 1.0, 3.0]]]),
    )


def test_rgbonly_cannot_access_depth_and_has_no_query_metadata(tmp_path, monkeypatch):
    record, access = scene(tmp_path), []
    loader = ContextOnlyLoader(
        record, (4, 4), tmp_path, track="RGB_ONLY", access_log=access, **kwargs()
    )

    def forbid(*args, **kw):
        raise AssertionError("Depth filesystem accessed by RGB_ONLY")

    monkeypatch.setattr(np, "load", forbid)
    batch = loader.context("A")
    assert batch.depth is None and batch.rgb.shape == (1, 3, 3, 4, 4)
    assert batch.frame_ids == (0, 1, 2)
    with pytest.raises(PermissionError):
        loader.context_depth(0)
    with pytest.raises(PermissionError):
        loader.context("A", track="RGBD")
    for method in [loader.query, loader.query_camera, loader.query_depth, loader.query_rgb]:
        with pytest.raises(PermissionError):
            method(8)
    assert all("depth" not in r["channels"] and r["frame_id"] < 5 for r in access)
    media = loader._ContextOnlyLoader__media
    assert set(media.frames) == {0, 1, 2, 3, 4}
    assert all("depth" not in f for f in media.frames.values())


def test_rgbd_reads_context_not_queries(tmp_path):
    loader = ContextOnlyLoader(scene(tmp_path), (4, 4), tmp_path, track="RGBD", **kwargs())
    assert loader.context("B").depth.shape == (1, 3, 1, 4, 4)
    assert loader.context("anchor").frame_ids == (0,)
    with pytest.raises(PermissionError):
        loader.context_depth(8)
    with pytest.raises(PermissionError):
        loader.context("query")


def test_all_planned_states_before_evaluator_and_immutable_multiquery_state(tmp_path):
    barrier = StateSealBarrier(["A", "B"])
    evaluator = CapacityEvaluator(scene(tmp_path), (4, 4), tmp_path, barrier=barrier, **kwargs())
    with pytest.raises(PermissionError):
        evaluator.query(8)
    original = state()
    barrier.seal("A", "capacity0", original)
    original.color.zero_()
    with pytest.raises(PermissionError):
        evaluator.query(8)
    barrier.seal("B", "capacity0", state())
    before = hash_scene_state(barrier.state("A"))
    assert evaluator.query(8).frame_ids == (8,)
    assert evaluator.query(9).frame_ids == (9,)
    assert hash_scene_state(barrier.state("A")) == before
    assert [e["event"] for e in barrier.events] == [
        "seal",
        "seal",
        "query_camera_GT",
        "query_camera_GT",
    ]
    barrier.state("A").color.zero_()
    with pytest.raises(PermissionError, match="mutated"):
        evaluator.query(8)


def test_query_oracle_requires_explicit_privileged_interface(tmp_path):
    record, access = scene(tmp_path), []
    with pytest.raises(PermissionError):
        ExplicitQueryOracleLoader(record, (4, 4), tmp_path, scope="RGBD", **kwargs())
    oracle = ExplicitQueryOracleLoader(
        record, (4, 4), tmp_path, scope="QUERY_SUPERVISED_ORACLE", access_log=access, **kwargs()
    )
    batch = oracle.diagnostic_query_supervision()
    assert batch.frame_ids == (8, 9)
    assert all("DIAGNOSTIC_QUERY_SUPERVISED_ORACLE" in r["purpose"] for r in access)


@pytest.mark.parametrize("kind", ["context", "evaluator", "oracle"])
def test_holdout_denied_before_any_media_or_optimization(tmp_path, kind):
    record = scene(tmp_path)
    record["scene_id"] = "held0"
    with pytest.raises(PermissionError, match="holdout"):
        if kind == "context":
            ContextOnlyLoader(record, (4, 4), tmp_path, track="RGBD", **kwargs())
        elif kind == "evaluator":
            CapacityEvaluator(record, (4, 4), tmp_path, barrier=StateSealBarrier(["A"]), **kwargs())
        else:
            ExplicitQueryOracleLoader(
                record, (4, 4), tmp_path, scope="QUERY_SUPERVISED_ORACLE", **kwargs()
            )


def test_context_checkpoint_selection_fixed_intervals_full_context_no_query_score():
    config = FrozenOptimConfig(max_steps=20, checkpoint_interval=10)
    selector = ContextCheckpointSelector(config, (0, 1, 2))
    assert selector.consider(0, context_objective=2.0, observed_ids=(0, 1, 2), state=state())
    assert selector.consider(10, context_objective=1.0, observed_ids=(0, 1, 2), state=state())
    assert not selector.consider(20, context_objective=1.5, observed_ids=(0, 1, 2), state=state())
    assert selector.best_step == 10
    with pytest.raises(TypeError):
        selector.consider(
            20, context_objective=0.1, observed_ids=(0, 1, 2), state=state(), query_score=0.0
        )
    with pytest.raises(PermissionError):
        selector.consider(20, context_objective=0.1, observed_ids=(0,), state=state())
    with pytest.raises(PermissionError):
        ContextCheckpointSelector(config, (0,)).consider(
            1, context_objective=1.0, observed_ids=(0,), state=state()
        )
    with pytest.raises(FrozenInstanceError):
        config.learning_rate = 0.1


def test_resolution_only_grid_and_renderer_only_samples():
    config = FrozenOptimConfig(scene_ids=("capacity0",), query_ids=(("capacity0", (8, 9)),))
    sweep = [replace(config, grid_size=g) for g in [8, 16, 32]]
    assert validate_resolution_sweep(sweep)
    for field, value in [
        ("learning_rate", 0.1),
        ("renderer_samples", 128),
        ("scene_ids", ("other",)),
        ("query_ids", (("capacity0", (7, 8)),)),
    ]:
        bad = [sweep[0], replace(sweep[1], **{field: value}), sweep[2]]
        with pytest.raises(PermissionError):
            validate_resolution_sweep(bad)
    assert validate_renderer_diagnostic(config, replace(config, renderer_samples=128))
    with pytest.raises(PermissionError):
        validate_renderer_diagnostic(config, replace(config, renderer_samples=128, grid_size=16))
