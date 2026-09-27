"""Protocol tests 9, 10, 13 and query/policy access boundaries, using synthetic tensors."""

import pytest
import torch

from mcss.dynamic.types import SealedScene, hash_scene_state, hash_value
from mcss.mechanism_pilot.contracts import (
    EvaluationLedger,
    ResidualReadoutControl,
    cross_scene_readout,
    validate_policy_inputs,
    validate_scene_splits,
)
from mcss.types import Cameras, SceneState


def state():
    return SceneState(
        torch.zeros(1, 1, 2, 2, 2),
        torch.full((1, 3, 2, 2, 2), 0.5),
        torch.zeros(1, 1, 2, 2, 2),
        torch.tensor([[[-1.0, -1.0, 1.0], [1.0, 1.0, 3.0]]]),
        features=torch.ones(1, 2, 2, 2, 2),
    )


def camera():
    return Cameras(torch.eye(3).reshape(1, 1, 3, 3), torch.eye(4).reshape(1, 1, 4, 4), (2, 2))


def sealed():
    s = state()
    return SealedScene(
        "episode",
        "scene",
        "test",
        "vault",
        s,
        hash_scene_state(s),
        (0, 1),
        "checkpoint",
        "config",
        "fast",
        torch.eye(4),
    )


class ToyRenderer(torch.nn.Module):
    def forward(self, scene, cameras, measurements):
        return {
            "rgb": scene.color.mean().expand(1, 1, 3, 2, 2),
            "depth": torch.ones(1, 1, 1, 2, 2),
            "visibility": torch.ones(1, 1, 1, 2, 2),
        }

    def render_features(self, scene, cameras):
        return scene.features.mean().expand(1, 1, 2, 2, 2)


def test_9_residual_training_cannot_modify_shared_state():
    s, c = state(), camera()
    s.features.requires_grad_()
    before = hash_scene_state(s)
    model = ResidualReadoutControl(2, ToyRenderer())
    assert model.training_status == "UNTRAINED_UNEVALUATED"
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    optimizer.zero_grad()
    predicted = model(s, c)
    loss = (predicted["rgb"] - 0.2).square().mean() + predicted["depth"].square().mean()
    loss.backward()
    assert s.features.grad is None
    optimizer.step()
    assert hash_scene_state(s) == before
    assert not torch.equal(model(s, c)["depth"], predicted["depth"])
    assert model.training_status == "UNTRAINED_UNEVALUATED"  # no inferred qualification


def test_10_cross_scene_keeps_original_query_camera():
    s, c = state(), camera()
    before = hash_value(c)

    class Recorder(ToyRenderer):
        def forward(self, scene, cameras, measurements):
            assert hash_value(cameras) == before
            assert cameras is not c
            return super().forward(scene, cameras, measurements)

    result = cross_scene_readout(Recorder(), s, c, "donor", "receiver")
    assert hash_value(c) == before
    assert set(result) == {"rgb", "depth", "visibility"}
    with pytest.raises(ValueError):
        cross_scene_readout(Recorder(), s, c, "same", "same")


def test_cross_scene_camera_mutation_is_rejected_and_isolated():
    s, c = state(), camera()
    before = hash_value(c)

    class Mutator(ToyRenderer):
        def forward(self, scene, cameras, measurements):
            cameras.c2w[..., 0, 3] += 1
            return super().forward(scene, cameras, measurements)

    with pytest.raises(RuntimeError):
        cross_scene_readout(Mutator(), s, c, "donor", "receiver")
    assert hash_value(c) == before


def row(scene, physical, split):
    return dict(
        scene_id=scene,
        physical_scene_id=physical,
        split=split,
        physical_scene_identity_status="VERIFIED",
    )


def test_13_physical_scene_repeated_scan_leakage_rejected():
    with pytest.raises(ValueError, match="leakage"):
        validate_scene_splits([row("scan1", "house", "train"), row("scan2", "house", "test")])
    assert (
        validate_scene_splits([row("scan1", "house", "train"), row("scan2", "house", "train")])[
            "train"
        ]
        == 1
    )
    with pytest.raises(ValueError):
        validate_scene_splits([{"scene_id": "unknown", "split": "test"}])


def test_all_candidates_required_before_any_query_loader():
    ledger = EvaluationLedger(["first", "second"])
    calls = []
    ledger.seal("first", sealed())
    for reader in (ledger.read_query_camera, ledger.read_query_ground_truth):
        with pytest.raises(PermissionError):
            reader("query", lambda: calls.append("leak"))
    assert not calls
    ledger.seal("second", sealed())
    assert ledger.read_query_camera("q", lambda: "camera") == "camera"
    assert ledger.read_query_ground_truth("q", lambda: "labels") == "labels"
    assert [e["event"] for e in ledger.events] == [
        "seal",
        "seal",
        "query_camera",
        "query_ground_truth",
    ]


def test_sealed_candidate_mutation_blocks_query_access():
    ledger = EvaluationLedger(["one"])
    candidate = sealed()
    ledger.seal("one", candidate)
    candidate.scene_state.color.add_(0.1)
    with pytest.raises(PermissionError):
        ledger.read_query_ground_truth("q", lambda: pytest.fail("must not call"))


@pytest.mark.parametrize(
    "payload",
    [
        {"depth": torch.ones(1)},
        {"query_camera": camera()},
        {"history_summary": {"branch_rewards": [1.0]}},
        {"state_summary": {"nested": {"oracle_action": 1}}},
    ],
)
def test_forbidden_policy_fields_rejected(payload):
    with pytest.raises(ValueError):
        validate_policy_inputs(payload)


def test_legal_policy_inputs():
    payload = {
        "arrived_rgb": torch.ones(3, 2, 2),
        "arrived_camera": camera(),
        "history_summary": {"previous_action": "OFF"},
    }
    assert validate_policy_inputs(payload) is payload
