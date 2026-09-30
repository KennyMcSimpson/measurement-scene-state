"""Matched geometry-aware carrier kernels use RGB-only state input and offline loss labels."""

import copy

import pytest
import torch
from test_mechanism_training import observation

from mcss.dynamic.carrier import DynamicSceneCarrier
from mcss.dynamic.config import CarrierConfig
from mcss.dynamic.types import hash_scene_state
from mcss.geometry import transform_cameras
from mcss.losses import MeasurementLoss
from mcss.measurements import FixedMeasurementRenderer
from mcss.mechanism_pilot.geometry_carrier import build_state, training_loss
from mcss.training.grounded import build_grounded_state
from mcss.types import Cameras


def fixture():
    torch.set_num_threads(1)
    torch.manual_seed(20260928)
    carrier = DynamicSceneCarrier(CarrierConfig(grid_size=(16, 16, 16)))
    context = [observation(i) for i in (0, 1, 2)]
    bounds = torch.tensor([[-2.0, -2.0, 0.1], [2.0, 2.0, 4.0]])
    q = observation(5).camera
    cameras = Cameras(q.intrinsics[None, None], q.c2w[None, None], (8, 8))
    cameras = transform_cameras(cameras, torch.linalg.inv(context[0].camera.c2w))
    return carrier, context, bounds, cameras


def test_scoped_geometry_buffers_and_old_equivalent_materialization():
    carrier, context, bounds, _ = fixture()
    original = {k: v.clone() for k, v in carrier.state_dict().items()}
    state, _ = build_state(carrier, context, bounds, "new")
    assert all(torch.equal(v, carrier.state_dict()[k]) for k, v in original.items())
    assert torch.equal(state.bounds, bounds[None])
    reference = DynamicSceneCarrier(
        CarrierConfig(
            grid_size=(16, 16, 16),
            local_bounds_m=tuple(tuple(float(x) for x in row) for row in bounds),
        )
    )
    parameters = dict(carrier.named_parameters())
    with torch.no_grad():
        for name, param in reference.named_parameters():
            param.copy_(parameters[name])
    old, _ = build_grounded_state(reference, context, "old")
    assert hash_scene_state(old) == hash_scene_state(state)
    assert sum(p.numel() for p in carrier.parameters()) == 5125
    assert torch.equal(reference._candidate_ids, carrier._candidate_ids)
    assert torch.equal(reference._candidate_normalized_xyz, carrier._candidate_normalized_xyz)


def test_loss_definition_matches_old_carrier_and_only_allowed_addition():
    carrier, context, bounds, cameras = fixture()
    state, _ = build_state(carrier, context, bounds, "loss")
    rgb = torch.full((1, 1, 3, 8, 8), 0.6)
    depth = torch.full((1, 1, 1, 8, 8), 2.0)
    depth[..., 0, 0] = 0
    indices = torch.arange(64)
    losses = {v: training_loss(state, cameras, rgb, depth, indices, v) for v in ("C0", "C1", "C2")}
    pred = FixedMeasurementRenderer(n_samples=64)(state, cameras, ("rgb", "depth"))
    expected, _ = MeasurementLoss({"rgb": 1.0, "depth": 1.0})(pred, {"rgb": rgb, "depth": depth})
    torch.testing.assert_close(losses["C0"][0], expected, atol=1e-7, rtol=1e-6)
    torch.testing.assert_close(
        losses["C1"][0], losses["C0"][0] + 0.1 * losses["C1"][1]["loss_surface"]
    )
    torch.testing.assert_close(
        losses["C2"][0], losses["C1"][0] + 0.1 * losses["C2"][1]["loss_free"]
    )
    assert all(t["regularization"].item() == 0 for _, t in losses.values())
    assert losses["C1"][1]["valid_depth_rays"] == 63
    assert losses["C1"][1]["surface_eligible_rays"] > 0


def test_surface_gradient_reaches_carrier_and_matched_initial_parameters():
    carrier, context, bounds, cameras = fixture()
    matched = copy.deepcopy(carrier)
    assert all(torch.equal(v, matched.state_dict()[k]) for k, v in carrier.state_dict().items())
    state, _ = build_state(carrier, context, bounds, "gradient")
    _, terms = training_loss(
        state,
        cameras,
        torch.full((1, 1, 3, 8, 8), 0.6),
        torch.full((1, 1, 1, 8, 8), 2.0),
        torch.arange(64),
        "C1",
    )
    terms["loss_surface"].backward()
    for name in (
        "image_encoder.0.weight",
        "fuse_down.weight",
        "complete_down.weight",
        "density_head.weight",
    ):
        grad = dict(carrier.named_parameters())[name].grad
        assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0
    # Geometry reverts before backward; gradients still reach original parameters.
    assert torch.equal(carrier._bounds, matched._bounds)
    assert torch.equal(carrier._candidate_points, matched._candidate_points)


def test_no_label_input_and_multiple_queries_share_state():
    carrier, context, bounds, cameras = fixture()
    with pytest.raises(TypeError):
        build_state(carrier, context, bounds, "illegal", depth=torch.ones(1))
    with pytest.raises(TypeError):
        build_state(carrier, context, bounds, "illegal", query_camera=cameras)
    state, _ = build_state(carrier, context, bounds, "shared")
    before = hash_scene_state(state)
    renderer = FixedMeasurementRenderer(n_samples=64)
    for offset in (0.0, 0.2):
        pose = cameras.c2w.clone()
        pose[..., 0, 3] += offset
        prediction = renderer(
            state, Cameras(cameras.intrinsics, pose, cameras.image_size), ("rgb", "depth")
        )
        assert torch.isfinite(prediction["depth"]).all()
        assert hash_scene_state(state) == before
    with pytest.raises(ValueError):
        training_loss(
            state,
            cameras,
            torch.zeros(1, 1, 3, 8, 8),
            torch.ones(1, 1, 1, 8, 8),
            torch.arange(64),
            "BEST",
        )
    with pytest.raises(ValueError):
        build_state(carrier, context, bounds.clone().requires_grad_(), "learnbounds")
