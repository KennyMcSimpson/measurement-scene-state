"""Residual control identity, state privacy and exact cached-readout regressions."""

from dataclasses import asdict

import pytest
import torch

from mcss.dynamic.types import hash_scene_state, hash_value
from mcss.measurements import FixedMeasurementRenderer
from mcss.mechanism_pilot.contracts import ResidualReadoutControl
from mcss.mechanism_pilot.static_residual import (
    SCHEMA,
    ResidualTrainingConfig,
    cache_readout,
    cached_forward,
    load_residual_checkpoint,
    supervised_loss,
    validate_training_manifest,
    zero_state_like,
)
from mcss.types import Cameras, SceneState


def fixture():
    state = SceneState(
        torch.zeros(1, 1, 2, 2, 2),
        torch.full((1, 3, 2, 2, 2), 0.4),
        torch.zeros(1, 1, 2, 2, 2),
        torch.tensor([[[-1.0, -1.0, 1.0], [1.0, 1.0, 3.0]]]),
        features=torch.randn(1, 8, 2, 2, 2),
    )
    camera = Cameras(
        torch.tensor([[[[4.0, 0.0, 2.0], [0.0, 4.0, 2.0], [0.0, 0.0, 1.0]]]]),
        torch.eye(4)[None, None],
        (4, 4),
    )
    model = ResidualReadoutControl(8, FixedMeasurementRenderer(n_samples=4))
    return state, camera, model


def test_cache_matches_nonzero_head_forward_and_preserves_state_camera():
    state, camera, model = fixture()
    torch.nn.init.normal_(model.head[-1].weight, std=0.01)
    before = hash_scene_state(state), hash_value(camera)
    cache = cache_readout(model, state, camera)
    direct, cached = model(state, camera), cached_forward(model, cache)
    assert all(torch.equal(direct[k], cached[k]) for k in direct)
    loss, _ = supervised_loss(
        cached, torch.zeros_like(cached["rgb"]), torch.ones_like(cached["depth"])
    )
    loss.backward()
    assert model.head[-1].weight.grad.abs().sum() > 0
    assert before == (hash_scene_state(state), hash_value(camera))
    assert state.features.grad is None


@pytest.mark.parametrize("bad", ["unseen", "dev", "query_context"])
def test_training_identity_firewall(bad):
    row = {
        "scene_id": "train1",
        "split": "train",
        "roles": {"query": [12, 13, 14, 15], "context_a": [0, 1, 2], "context_b": [0, 3, 4]},
    }
    if bad == "unseen":
        row["scene_id"] = "unseen"
    elif bad == "dev":
        row["split"] = "dev"
    else:
        row["roles"]["context_a"] = [0, 1, 8]
    with pytest.raises(PermissionError):
        validate_training_manifest({"scenes": [row]}, ["train1"])


def test_training_identity_accepts_only_explicit_allowlist():
    row = {
        "scene_id": "train1",
        "split": "train",
        "roles": {"query": [12, 13, 14, 15], "context_a": [0, 1, 2], "context_b": [0, 3, 4]},
    }
    validate_training_manifest({"scenes": [row]}, ["train1"])


def test_zero_state_is_empty_and_preserves_geometry():
    state, camera, model = fixture()
    before = hash_scene_state(state)
    zero = zero_state_like(state)
    assert torch.equal(zero.bounds, state.bounds)
    assert torch.count_nonzero(zero.features) == 0
    assert torch.all(zero.density_logits == -100)
    assert before == hash_scene_state(state)
    assert model(zero, camera)["depth"].max() < 1e-20


def test_checkpoint_restores_freezes_and_binds_carrier(tmp_path):
    state, camera, model = fixture()
    config = ResidualTrainingConfig(renderer_samples=4)
    path = tmp_path / "residual.pt"
    torch.save(
        {
            "schema": SCHEMA,
            "carrier_sha256": "bound",
            "config": asdict(config),
            "feature_dim": 8,
            "head_state_dict": model.head.state_dict(),
        },
        path,
    )
    loaded, _ = load_residual_checkpoint(path, carrier_sha256="bound")
    assert not loaded.training and all(not p.requires_grad for p in loaded.parameters())
    assert torch.equal(model(state, camera)["rgb"], loaded(state, camera)["rgb"])
    with pytest.raises(ValueError):
        load_residual_checkpoint(path, carrier_sha256="wrong")
