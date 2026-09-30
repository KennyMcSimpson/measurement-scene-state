"""Frozen budget trajectories preserve previous state semantics and optimizer prefix."""

import json
from dataclasses import replace

import pytest
import torch

from mcss.dynamic.types import hash_scene_state
from mcss.mechanism_pilot.budget_state_optimization import plateau_diagnostic, run_budget_trajectory
from mcss.mechanism_pilot.direct_capacity_contracts import ObservationBatch
from mcss.mechanism_pilot.direct_capacity_optimization import (
    DirectOptimizationConfig,
    optimize_state,
)
from mcss.types import Cameras


def batch(device="cpu"):
    camera = Cameras(
        torch.tensor([[[[2.0, 0.0, 1.0], [0.0, 2.0, 1.0], [0.0, 0.0, 1.0]]]], device=device),
        torch.eye(4, device=device)[None, None],
        (2, 2),
    )
    value = ObservationBatch(
        "synthetic",
        (0,),
        torch.full((1, 1, 3, 2, 2), 0.2, device=device),
        camera,
        torch.full((1, 1, 1, 2, 2), 2.0, device=device),
        "CONTEXT_ONLY_RGBD",
    )
    return value, torch.tensor([[-2.0, -2.0, 0.1], [2.0, 2.0, 4.0]], device=device)


def test_single_trajectory_prefix_and_context_only_selection(tmp_path):
    torch.set_num_threads(1)
    observed, bounds = batch()
    config = DirectOptimizationConfig(steps=20, batch_rays=8, checkpoint_interval=10)
    report = run_budget_trajectory(
        observed,
        bounds,
        tmp_path / "trajectory",
        seed=5,
        budgets=(10, 20),
        config=config,
        engineering_fixture=True,
    )
    old_state, old_summary = optimize_state(
        observed, bounds, 16, 64, tmp_path / "old", seed=5, config=replace(config, steps=10)
    )
    selected = torch.load(
        tmp_path / "trajectory/budget_10/CONTEXT_SELECTED/state.pt", weights_only=False
    )
    assert hash_scene_state(selected) == hash_scene_state(old_state)
    assert report["outputs"][1]["selected_step"] == old_summary["selected_step"]
    assert [c["step"] for c in report["full_objective_checkpoints"]] == [0, 10, 20]
    for budget in (10, 20):
        eligible = [c for c in report["full_objective_checkpoints"] if c["step"] <= budget]
        best = min(eligible, key=lambda c: (c["context_objective"], c["step"]))
        selected_summary = json.loads(
            (tmp_path / f"trajectory/budget_{budget}/CONTEXT_SELECTED/summary.json").read_text()
        )
        assert selected_summary["selected_step"] == best["step"]
        assert selected_summary["selected_context_metrics"]["step"] == best["step"]
        fixed = torch.load(
            tmp_path / f"trajectory/budget_{budget}/FIXED_BUDGET/state.pt", weights_only=False
        )
        snapshot = torch.load(
            tmp_path / f"trajectory/snapshots/step_{budget}.pt", weights_only=False
        )
        assert hash_scene_state(fixed) == hash_scene_state(snapshot)
        assert fixed.density_logits.shape == (1, 1, 16, 16, 16)
    assert report["single_trajectory"] and not report["query_selection"]
    with pytest.raises(TypeError):
        run_budget_trajectory(observed, bounds, tmp_path / "query", seed=5, query_score=0.0)


def test_1000_step_cpu_prefix_matches_previous_kernel_exactly(tmp_path):
    torch.set_num_threads(1)
    observed, bounds = batch("cpu")
    config = DirectOptimizationConfig(steps=1000, batch_rays=8)
    report = run_budget_trajectory(
        observed,
        bounds,
        tmp_path / "trajectory",
        seed=20260927,
        budgets=(1000,),
        config=config,
        engineering_fixture=True,
    )
    old_state, old_summary = optimize_state(
        observed, bounds, 16, 64, tmp_path / "old", seed=20260927, config=config
    )
    state = torch.load(
        tmp_path / "trajectory/budget_1000/CONTEXT_SELECTED/state.pt", weights_only=False
    )
    assert hash_scene_state(state) == hash_scene_state(old_state)
    assert report["outputs"][1]["selected_step"] == old_summary["selected_step"]
    for current, prior in zip(
        report["full_objective_checkpoints"], old_summary["full_objective_checkpoints"], strict=True
    ):
        assert all(current[key] == value for key, value in prior.items())


def test_formal_optimizer_and_budget_changes_are_forbidden(tmp_path):
    observed, bounds = batch()
    with pytest.raises(PermissionError):
        run_budget_trajectory(
            observed,
            bounds,
            tmp_path / "lr",
            seed=1,
            config=DirectOptimizationConfig(steps=10000, learning_rate=0.02),
        )
    with pytest.raises(ValueError):
        run_budget_trajectory(observed, bounds, tmp_path / "budget", seed=1, budgets=(1000, 2000))
    with pytest.raises(ValueError):
        run_budget_trajectory(
            observed, bounds, tmp_path / "mutable", seed=1, budgets=[1000, 3000, 10000]
        )


def test_plateau_frozen_last_ten_percent_median_gradient_rule():
    checks = [{"step": 900, "context_objective": 1.0}, {"step": 1000, "context_objective": 0.9995}]
    trace = [{"step": step, "gradient_norm_before_clip": 1.0} for step in range(900, 1001, 10)]
    result = plateau_diagnostic(
        checks, 1000, trace=trace, objective_threshold=0.001, gradient_threshold=0.10
    )
    assert result["plateau"] and result["n_trace_gradients"] == 11
    trace[-5:] = [{"step": r["step"], "gradient_norm_before_clip": 0.5} for r in trace[-5:]]
    result = plateau_diagnostic(
        checks, 1000, trace=trace, objective_threshold=0.001, gradient_threshold=0.10
    )
    assert not result["plateau"] and not result["gradient_stable"]
    assert not plateau_diagnostic(
        checks, 1000, trace=trace[:2], objective_threshold=0.001, gradient_threshold=0.10
    )["plateau"]


def test_secondary_single_budget_only_explicit_rgb_only(tmp_path):
    observed, bounds = batch()
    with pytest.raises(PermissionError, match="RGB_ONLY"):
        run_budget_trajectory(
            observed, bounds, tmp_path / "denied", seed=1, budgets=(10000,), secondary_sanity=True
        )
    with pytest.raises(ValueError, match="frozen main"):
        run_budget_trajectory(observed, bounds, tmp_path / "main", seed=1, budgets=(10000,))
    rgb_only = replace(observed, depth=None, supervision="CONTEXT_ONLY_RGB_ONLY")
    result = run_budget_trajectory(
        rgb_only,
        bounds,
        tmp_path / "secondary",
        seed=1,
        budgets=(10,),
        config=DirectOptimizationConfig(steps=10, batch_rays=8, checkpoint_interval=10),
        engineering_fixture=True,
        secondary_sanity=True,
    )
    assert len(result["outputs"]) == 2 and not result["outputs"][0]["has_depth"]
