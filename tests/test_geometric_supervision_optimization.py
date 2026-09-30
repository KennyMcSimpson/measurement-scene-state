"""Geometry supervision is context-only, matched, and does not rewrite ray semantics."""

from dataclasses import replace

import pytest
import torch
from test_budget_state_optimization import batch

from mcss.dynamic.types import hash_scene_state
from mcss.measurements import FixedMeasurementRenderer
from mcss.mechanism_pilot.budget_state_optimization import run_budget_trajectory
from mcss.mechanism_pilot.direct_capacity_optimization import (
    DirectOptimizationConfig,
    DirectState,
    cache_rays,
    render_rays,
)
from mcss.mechanism_pilot.geometric_supervision_optimization import (
    geometry_losses,
    optimize_geometric_state,
    ray_geometry,
    surface_tau,
)


def test_exact_ray_geometry_and_spacing():
    observed, bounds = batch()
    model, renderer = DirectState(bounds, 16), FixedMeasurementRenderer(n_samples=64)
    cache = cache_rays(observed, bounds)
    indices = torch.arange(4)
    t, alpha, weights, hit = ray_geometry(renderer, model.state(), cache, indices)
    pred = render_rays(renderer, model.state(), cache, indices)
    assert torch.equal(pred["visibility"], weights.sum(-1, keepdim=True))
    assert torch.equal(pred["depth"], (weights * t).sum(-1, keepdim=True))
    assert bool(hit.all()) and bool((alpha >= 0).all())
    assert surface_tau(bounds) == 0.5 * ((bounds[1] - bounds[0]) / 16).norm()


def test_free_gradient_only_before_band_and_valid_ray_denominator():
    t = torch.tensor([[[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]]])
    alpha = torch.full_like(t, 0.2, requires_grad=True)
    depth = torch.tensor([[[3.0], [0.5]]])
    free, _, counts = geometry_losses(
        t, alpha, alpha, torch.ones(1, 2, dtype=torch.bool), depth, 0.25
    )
    assert free.item() == pytest.approx(0.1)  # Empty free set in second ray counts in denominator.
    free.backward()
    assert torch.equal(alpha.grad, torch.tensor([[[0.25, 0.25, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]]))
    assert counts["free_empty_rays"] == 1


def test_surface_empty_band_and_miss_excluded_and_invalid_depth_safe():
    t = torch.tensor([[[1.0, 2.0, 3.0]]]).expand(1, 4, 3)
    weights = torch.full_like(t, 0.2, requires_grad=True)
    depth = torch.tensor([[[2.0], [10.0], [2.0], [float("nan")]]])
    free, surface, counts = geometry_losses(
        t, weights, weights, torch.tensor([[True, True, False, True]]), depth, 0.1
    )
    assert torch.isfinite(free) and surface.item() == pytest.approx(
        -torch.log(torch.tensor(0.2)).item()
    )
    surface.backward()
    assert weights.grad[0, 0, 1] < 0
    assert torch.count_nonzero(weights.grad) == 1
    assert counts["surface_eligible_rays"] == counts["empty_band_hit_rays"] == 1
    assert counts["valid_miss_rays"] == 1
    _, empty, _ = geometry_losses(
        t, weights, weights, torch.zeros(1, 4, dtype=torch.bool), depth, 0.1
    )
    assert empty.item() == 0


def test_matched_streams_initialization_and_context_firewall(tmp_path):
    torch.set_num_threads(1)
    observed, bounds = batch()
    config = DirectOptimizationConfig(steps=20, batch_rays=8, checkpoint_interval=10)
    summaries = []
    for variant in ("S0", "S1", "S2", "S3"):
        state, summary = optimize_geometric_state(
            observed,
            bounds,
            tmp_path / variant,
            variant=variant,
            seed=19,
            config=config,
            engineering_fixture=True,
        )
        assert summary["selected_step"] == 20 and summary["all_steps_finite"]
        assert state.density_logits.shape == (1, 1, 16, 16, 16)
        summaries.append(summary)
    assert len({s["initial_state_hash"] for s in summaries}) == 1
    assert len({s["ray_stream_sha256"] for s in summaries}) == 1
    assert len({s["state_hash"] for s in summaries}) == 4
    for supervision in ("QUERY_SUPERVISED_ORACLE", "CONTEXT_ONLY_RGB_ONLY"):
        with pytest.raises(PermissionError):
            optimize_geometric_state(
                replace(observed, supervision=supervision),
                bounds,
                tmp_path / supervision,
                variant="S3",
                seed=1,
            )
    with pytest.raises(PermissionError):
        optimize_geometric_state(
            observed, bounds, tmp_path / "short", variant="S0", seed=1, config=config
        )
    with pytest.raises(TypeError):
        optimize_geometric_state(
            observed, bounds, tmp_path / "weight", variant="S3", seed=1, lambda_surface=0.2
        )
    with pytest.raises(TypeError):
        optimize_geometric_state(
            observed, bounds, tmp_path / "query", variant="S3", seed=1, query_score=0.1
        )


def test_s0_cpu_1000_steps_exact_old_fixed_budget(tmp_path):
    torch.set_num_threads(1)
    observed, bounds = batch()
    config = DirectOptimizationConfig(steps=1000, batch_rays=8)
    run_budget_trajectory(
        observed,
        bounds,
        tmp_path / "old",
        seed=20260927,
        budgets=(1000,),
        config=config,
        engineering_fixture=True,
    )
    state, summary = optimize_geometric_state(
        observed,
        bounds,
        tmp_path / "new",
        variant="S0",
        seed=20260927,
        config=config,
        engineering_fixture=True,
    )
    previous = torch.load(tmp_path / "old/budget_1000/FIXED_BUDGET/state.pt", weights_only=False)
    assert hash_scene_state(state) == hash_scene_state(previous)
    assert summary["selection"] == "FIXED_BUDGET"
