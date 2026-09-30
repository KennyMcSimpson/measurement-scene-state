"""Independent geometry contracts; no oracle score or deployed GT access."""

import inspect
from dataclasses import replace

import numpy as np
import pytest
import torch

from mcss.dynamic.carrier import DynamicSceneCarrier
from mcss.dynamic.config import CarrierConfig
from mcss.geometry import generate_rays
from mcss.mechanism_pilot import support_redesign_geometry as geometry
from mcss.types import Cameras


def camera(x=0.0):
    pose = torch.eye(4)
    pose[0, 3] = x
    return Cameras(torch.tensor([[2.0, 0.0, 2.0], [0.0, 2.0, 2.0], [0.0, 0.0, 1.0]]), pose, (5, 5))


def original():
    model = DynamicSceneCarrier(CarrierConfig())
    return model, model._bounds[0].clone(), model._candidate_ids.clone()


@pytest.mark.parametrize(
    "fn",
    [
        geometry.adaptive_bounds,
        geometry.allocate_candidates,
        geometry.deployable_plan,
        geometry.context_selection,
    ],
)
def test_runtime_api_has_no_labels_or_catchall_keyword_arguments(fn):
    parameters = inspect.signature(fn).parameters
    forbidden = {"depth", "gt_depth", "query_gt", "context_surfaces", "model_score", "loss"}
    assert not forbidden.intersection(parameters)
    assert all(p.kind != inspect.Parameter.VAR_KEYWORD for p in parameters.values())


def test_runtime_gt_injection_is_rejected():
    _, bounds, ids = original()
    cameras = [camera(), camera(0.1)]
    with pytest.raises(TypeError):
        geometry.adaptive_bounds(cameras, near=0.1, far=4.0, gt_depth=torch.ones(5, 5))
    with pytest.raises(TypeError):
        geometry.allocate_candidates(bounds, cameras, depth=torch.ones(5, 5))
    with pytest.raises(TypeError):
        geometry.context_selection(
            {i: camera(i / 10) for i in range(5)}, bounds, ids, model_score={"best": 1}
        )
    with pytest.raises(PermissionError):
        geometry.deployable_plan("ORACLE_SUPPORT", bounds, ids, cameras)
    with pytest.raises(ValueError):
        geometry.deployable_plan(
            "R1", bounds, ids, cameras, depth_prior={"near": 0.1, "far": 4.0, "gt_depth": 1}
        )


@pytest.mark.parametrize("method", ["R0", "R1", "R2", "R3", "R12", "R123"])
def test_runtime_plans_preserve_unique_budget_and_scatter_geometry(method):
    _, bounds, ids = original()
    plan = geometry.deployable_plan(
        method, bounds, ids, [camera(), camera(0.2)], depth_prior={"near": 0.1, "far": 4.0}
    )
    assert plan.candidate_ids.shape == (128,)
    assert plan.candidate_ids.dtype == torch.long
    assert plan.candidate_ids.unique().numel() == 128
    assert int(plan.candidate_ids.min()) >= 0 and int(plan.candidate_ids.max()) < 512
    assert not plan.oracle
    i = plan.candidate_ids
    xyz = torch.stack((i % 8, (i // 8) % 8, i // 64), -1).float()
    fractions = (xyz + 0.5) / 8
    assert torch.equal(plan.points, plan.bounds[0] + fractions * (plan.bounds[1] - plan.bounds[0]))
    assert torch.equal(plan.normalized_xyz, 2 * fractions - 1)


def test_adaptive_bounds_enclose_interior_rays_not_only_frustum_corners():
    cameras = [camera(), camera(0.2)]
    bounds = geometry.adaptive_bounds(cameras, near=0.1, far=4.0)
    for c in cameras:
        origin, rays = generate_rays(c)
        for depth in [0.1, 2.0, 4.0]:
            points = origin + depth * rays
            assert ((points >= bounds[0]) & (points <= bounds[1])).all()
    assert bounds[1, 2] >= 4  # central normalized ray extends beyond z of corner rays


def test_oracle_bounds_enclose_gt_and_are_marked_diagnostic_only():
    _, bounds, ids = original()
    surfaces = torch.tensor([[-10.0, -3.0, 0.1], [20.0, 2.0, 6.0]])
    query_bounds = geometry.enclose(surfaces)
    assert ((surfaces >= query_bounds[0]) & (surfaces <= query_bounds[1])).all()
    plan = geometry.oracle_plan(
        "ORACLE_VOLUME",
        bounds,
        ids,
        [camera()],
        query_bounds=query_bounds,
        context_surfaces=surfaces,
    )
    assert plan.oracle and torch.equal(plan.bounds, query_bounds)
    assert torch.equal(plan.candidate_ids, ids)
    with pytest.raises(ValueError):
        geometry.enclose(torch.tensor([[float("nan"), 0.0, 1.0]]))


def test_oracle_allocation_prefers_two_frusta_then_nearest_gt_and_stable_ids():
    bounds = torch.tensor([[-2.0, -2.0, 0.2], [2.0, 2.0, 4.2]])
    points, _ = geometry.lattice(bounds)
    cameras = [camera(), camera(0.2)]
    counts = geometry.view_counts(points, cameras)
    # Put surfaces at widely separated cells; derive ranking independently with tuples.
    surfaces = points[torch.tensor([0, 180, 511])].clone()
    distances = torch.linalg.vector_norm(points[:, None] - surfaces[None], dim=-1).amin(1)
    expected = sorted(
        range(512), key=lambda i: (-int(counts[i] >= 2), float(distances[i]), -int(counts[i]), i)
    )[:128]
    # Exact cell-center contacts and the broad ranking survive cdist rounding.
    actual = geometry.oracle_candidates(bounds, cameras, surfaces)
    assert actual.unique().numel() == 128
    assert np.array_equal(
        (counts[actual] >= 2).numpy(), (counts[torch.tensor(expected)] >= 2).numpy()
    )
    two_view = counts >= 2
    assert int(two_view[actual].sum()) == min(int(two_view.sum()), 128)
    actual_distances = geometry.nearest_distance(points, surfaces)
    independent = sorted(
        range(512),
        key=lambda i: (-int(counts[i] >= 2), float(actual_distances[i]), -int(counts[i]), i),
    )[:128]
    assert actual.tolist() == independent
    assert torch.equal(actual, geometry.oracle_candidates(bounds, cameras, surfaces))


def test_camera_context_selection_is_deterministic_disjoint_and_order_independent():
    _, bounds, ids = original()
    cameras = {i: camera(i / 10) for i in range(7)}
    first = geometry.context_selection(cameras, bounds, ids)
    second = geometry.context_selection(dict(reversed(list(cameras.items()))), bounds, ids)
    assert first == second
    assert len(first["context_a"]) == len(first["context_b"]) == 3
    assert set(first["context_a"]) & set(first["context_b"]) == {0}
    assert set(first["context_a"] + first["context_b"]) <= set(cameras)


def test_apply_plan_changes_only_buffers_not_learned_parameters_or_architecture():
    model, bounds, ids = original()
    parameters = {k: v.detach().clone() for k, v in model.named_parameters()}
    config = model.config
    plan = geometry.deployable_plan(
        "R12", bounds, ids, [camera(), camera(0.2)], depth_prior={"near": 0.1, "far": 4.0}
    )
    geometry.apply_plan(model, plan)
    assert model.config == config  # runtime diagnostic; not a rewritten old checkpoint
    assert all(torch.equal(parameters[k], v) for k, v in model.named_parameters())
    assert torch.equal(model._bounds[0], plan.bounds)
    assert torch.equal(model._candidate_ids, plan.candidate_ids)
    assert torch.equal(model._candidate_points, plan.points)
    assert torch.equal(model._candidate_normalized_xyz, plan.normalized_xyz)
    with pytest.raises(ValueError, match="scatter voxel centers"):
        geometry.apply_plan(model, replace(plan, points=plan.points + 0.01))
    with pytest.raises(ValueError, match="Normalized"):
        geometry.apply_plan(model, replace(plan, normalized_xyz=plan.normalized_xyz + 0.01))


def test_apply_plan_rejects_forged_duplicate_scatter_ids():
    model, bounds, _ = original()
    points, normalized = geometry.lattice(bounds)
    ids = torch.zeros(128, dtype=torch.long)
    forged = geometry.GeometryPlan("R2", bounds, ids, points[ids], normalized[ids])
    with pytest.raises(ValueError):
        geometry.apply_plan(model, forged)


def test_oracle_scarcity_keeps_budget_without_faking_supported_candidates():
    bounds = torch.tensor([[-2.0, -2.0, 0.2], [2.0, 2.0, 4.2]])
    points, _ = geometry.lattice(bounds)
    # One context can never supply >=2-view support, regardless of proximity to GT.
    ids = geometry.oracle_candidates(bounds, [camera()], points[:3])
    assert ids.numel() == ids.unique().numel() == 128
    assert (geometry.view_counts(points[ids], [camera()]) >= 2).sum() == 0


@pytest.mark.parametrize("invalid", ["reversed", "nonfinite", "shape"])
def test_apply_plan_rejects_invalid_metric_bounds(invalid):
    model, bounds, ids = original()
    plan = geometry.make_plan("R0", bounds, ids)
    bad = bounds.clone()
    if invalid == "reversed":
        bad = bad.flip(0)
    elif invalid == "nonfinite":
        bad[0, 0] = float("nan")
    else:
        bad = bad[None]
    with pytest.raises(ValueError):
        geometry.apply_plan(model, replace(plan, bounds=bad))
