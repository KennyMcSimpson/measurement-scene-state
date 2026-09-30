"""Independent fixed-kernel and context-only direct optimization contracts."""

import ast
import inspect
import json
from dataclasses import replace

import pytest
import torch

from mcss.dynamic.types import clone_scene_state, hash_scene_state
from mcss.measurements import FixedMeasurementRenderer
from mcss.mechanism_pilot import direct_capacity_optimization as optimization
from mcss.mechanism_pilot.direct_capacity_contracts import ObservationBatch
from mcss.types import Cameras, SceneState


def fixture_batch(depth=True):
    poses = torch.eye(4).repeat(1, 2, 1, 1)
    poses[0, 1, 0, 3] = 0.2
    k = torch.tensor([[4.0, 0.0, 2.0], [0.0, 4.0, 2.0], [0.0, 0.0, 1.0]])
    cameras = Cameras(k.repeat(1, 2, 1, 1), poses, (5, 5))
    bounds = torch.tensor([[-2.0, -2.0, 0.1], [2.0, 2.0, 4.0]])
    target = optimization.DirectState(bounds, 4)
    with torch.no_grad():
        target.density.fill_(-0.4)
        target.color_logits[:, 0].fill_(1.0)
        target.color_logits[:, 1].fill_(-0.6)
        rendered = FixedMeasurementRenderer(n_samples=16)(target.state(), cameras)
    batch = ObservationBatch(
        "synthetic",
        (0, 3),
        rendered["rgb"].detach(),
        cameras,
        rendered["depth"].detach() if depth else None,
        "CONTEXT_RGBD" if depth else "CONTEXT_RGB_ONLY",
    )
    return batch, bounds


def flatten_image(image):
    return image.permute(0, 1, 3, 4, 2).reshape(1, -1, image.shape[2])


@pytest.mark.parametrize("samples", [64, 128])
def test_cached_selected_rays_match_full_predictions_and_parameter_gradients(samples):
    batch, bounds = fixture_batch()
    model = optimization.DirectState(bounds, 4)
    with torch.no_grad():
        model.density.add_(torch.linspace(-0.1, 0.2, 64).reshape_as(model.density))
        model.color_logits.add_(torch.linspace(-1.0, 1.0, 192).reshape_as(model.color_logits))
    renderer = FixedMeasurementRenderer(n_samples=samples)
    cache = optimization.cache_rays(batch, bounds)
    # Both frames, boundary pixels and duplicate rays exercise flattening and accumulation.
    selected = torch.tensor([0, 4, 12, 24, 25, 31, 49, 12])
    full = renderer(model.state(), batch.cameras)
    sampled = optimization.render_rays(renderer, model.state(), cache, selected)
    for field in ("rgb", "depth", "visibility"):
        torch.testing.assert_close(
            sampled[field], flatten_image(full[field])[:, selected], rtol=0, atol=0
        )
    full_loss = sum(flatten_image(full[k])[:, selected].square().mean() for k in ("rgb", "depth"))
    ray_loss = sum(sampled[k].square().mean() for k in ("rgb", "depth"))
    full_grad = torch.autograd.grad(full_loss, tuple(model.parameters()))
    ray_grad = torch.autograd.grad(ray_loss, tuple(model.parameters()))
    for actual, expected in zip(ray_grad, full_grad, strict=True):
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=1e-7)
        assert actual.abs().sum() > 0


@pytest.mark.parametrize("with_depth", [True, False])
def test_optimizer_reduces_context_objective_selects_best_and_saves_detached_shared_state(
    tmp_path, with_depth
):
    batch, bounds = fixture_batch(depth=with_depth)
    config = optimization.DirectOptimizationConfig(steps=30, batch_rays=64, checkpoint_interval=5)
    output = tmp_path / "optimization"
    state, summary = optimization.optimize_state(
        batch, bounds, 4, 16, output, seed=91, config=config
    )
    checkpoints = summary["full_objective_checkpoints"]
    best = min(checkpoints, key=lambda row: row["context_objective"])
    assert summary["selected_step"] == best["step"]
    assert summary["selected"]["context_objective"] == pytest.approx(
        best["context_objective"], abs=1e-8
    )
    assert summary["selected"]["loss_rgb"] < summary["initial"]["loss_rgb"]
    if with_depth:
        assert summary["selected"]["loss_depth"] < summary["initial"]["loss_depth"]
    else:
        assert summary["selected"]["loss_depth"] is None
        assert optimization.cache_rays(batch, bounds)["depth"] is None
    assert summary["selected"]["context_objective"] < summary["initial"]["context_objective"]
    assert (
        summary["nonzero_gradient_observed"] and summary["all_steps_finite"] and summary["changed"]
    )
    assert summary["parameter_count"] == 4 * 4**3
    assert summary["shared_supervision_frame_ids"] == [0, 3]
    assert isinstance(state, SceneState)
    assert state.features is None
    for tensor in (state.density_logits, state.color, state.log_variance, state.bounds):
        assert tensor.device.type == "cpu" and not tensor.requires_grad and tensor.grad_fn is None
        assert torch.isfinite(tensor).all()
    torch.testing.assert_close(state.log_variance, torch.full_like(state.log_variance, -3.0))
    saved = torch.load(output / "state.pt", weights_only=False)
    assert hash_scene_state(saved) == summary["state_hash"] == hash_scene_state(state)
    before = hash_scene_state(state)
    renderer = FixedMeasurementRenderer(n_samples=16)
    initial = renderer(optimization.DirectState(bounds, 4).state(), batch.cameras)
    final = renderer(state, batch.cameras)
    assert not torch.equal(initial["rgb"], final["rgb"])
    assert final["rgb"].shape[:2] == (1, 2)
    assert hash_scene_state(state) == before  # one unchanged state serves both cameras
    lock = json.loads((output / "lock.json").read_text())
    assert not lock["carrier_loaded"]
    assert lock["has_depth"] == with_depth
    with pytest.raises(FileExistsError):
        optimization.optimize_state(batch, bounds, 4, 16, output, seed=91, config=config)


def test_zero_update_ties_select_initial_context_checkpoint(tmp_path):
    batch, bounds = fixture_batch()
    config = optimization.DirectOptimizationConfig(
        steps=2, learning_rate=0, checkpoint_interval=1, batch_rays=8
    )
    _, summary = optimization.optimize_state(
        batch, bounds, 4, 16, tmp_path / "ties", seed=1, config=config
    )
    assert summary["selected_step"] == 0
    assert not summary["changed"]
    assert len({r["context_objective"] for r in summary["full_objective_checkpoints"]}) == 1


@pytest.mark.parametrize("grid", [8, 16, 32])
def test_state_parameters_are_only_density_and_color(grid):
    _, bounds = fixture_batch()
    model = optimization.DirectState(bounds, grid)
    assert set(dict(model.named_parameters())) == {"density", "color_logits"}
    assert sum(p.numel() for p in model.parameters()) == 4 * grid**3
    state = model.state()
    assert state.density_logits is model.density
    assert state.features is state.evidence is state.appearance is None
    snapshot = clone_scene_state(state).to("cpu")
    with torch.no_grad():
        model.density.add_(1)
    assert not torch.equal(snapshot.density_logits, model.density)
    assert snapshot.density_logits.grad_fn is snapshot.color.grad_fn is None


@pytest.mark.parametrize("grid", [8, 16, 32])
def test_regularization_uses_physical_derivative_and_mean_density(grid):
    bounds = torch.tensor([[-1.0, -2.0, -3.0], [1.0, 2.0, 3.0]])
    model = optimization.DirectState(bounds, grid)
    state = model.state()
    # A color ramp with derivative 0.1 / metre along world x only.
    centers = -1 + (torch.arange(grid) + 0.5) * (2 / grid)
    state.color = (0.5 + 0.1 * centers).view(1, 1, 1, 1, grid).expand(1, 3, grid, grid, grid)
    config = optimization.DirectOptimizationConfig()
    expected = config.density_l2_weight * torch.nn.functional.softplus(torch.tensor(-2.0)).square()
    expected += config.tv_weight * 0.1 / 6
    torch.testing.assert_close(optimization.regularization(state, config), expected)
    doubled_bounds = clone_scene_state(state)
    doubled_bounds.bounds *= 2
    no_l2 = replace(config, density_l2_weight=0)
    torch.testing.assert_close(
        optimization.regularization(doubled_bounds, no_l2),
        optimization.regularization(state, no_l2) / 2,
    )


def test_optimizer_has_no_query_or_checkpoint_api_and_no_carrier_import():
    signature = inspect.signature(optimization.optimize_state)
    assert set(signature.parameters) == {
        "batch",
        "bounds",
        "grid",
        "samples",
        "output",
        "seed",
        "config",
    }
    assert all(p.kind != inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values())
    source = ast.parse(inspect.getsource(optimization))
    imports = [node.module for node in ast.walk(source) if isinstance(node, ast.ImportFrom)]
    assert not any(
        "carrier" in name or "checkpoint" in name or "writer" in name for name in imports
    )
    calls = [node for node in ast.walk(source) if isinstance(node, ast.Call)]
    assert not any(
        isinstance(node.func, ast.Attribute) and node.func.attr == "load" for node in calls
    )
