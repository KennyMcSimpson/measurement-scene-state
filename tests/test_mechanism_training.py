"""Training-contract tests; random synthetic fixtures are not carrier qualification."""

from dataclasses import replace
from unittest.mock import patch

import pytest
import torch

from mcss.dynamic.carrier import DynamicSceneCarrier
from mcss.dynamic.config import CarrierConfig
from mcss.dynamic.types import OnlineObservation, hash_scene_state
from mcss.measurements import FixedMeasurementRenderer
from mcss.mechanism_pilot.training import common_anchor_measurement_loss
from mcss.types import Cameras


def observation(frame):
    pose = torch.eye(4)
    pose[0, 3] = 2 + frame * 0.025
    intrinsics = torch.tensor([[6.0, 0, 3.5], [0, 6.0, 3.5], [0, 0, 1.0]])
    return OnlineObservation(
        "scene", frame, torch.full((3, 8, 8), 0.1 + frame * 0.07),
        Cameras(intrinsics, pose, (8, 8)),
    )


@pytest.fixture
def batch():
    torch.manual_seed(21)
    carrier = DynamicSceneCarrier(CarrierConfig(
        feature_dim=4, hidden_dim=4, expansion_dim=8, grid_size=(4, 4, 4), token_count=16
    ))
    query = observation(5).camera
    return dict(
        carrier=carrier,
        first_context=[observation(i) for i in (0, 1, 2)],
        second_context=[observation(i) for i in (0, 3, 4)],
        query_frame_ids=[5],
        query_cameras=Cameras(query.intrinsics[None, None], query.c2w[None, None], (8, 8)),
        query_rgb=torch.full((1, 1, 3, 8, 8), 0.6),
        query_depth=torch.full((1, 1, 1, 8, 8), 3.0),
        renderer=FixedMeasurementRenderer(n_samples=8), split_id="train",
    )


def test_common_anchor_loss_backpropagates_without_fast_writes(batch):
    carrier = batch["carrier"]
    with patch.object(carrier, "materialize", wraps=carrier.materialize) as materialize:
        result = common_anchor_measurement_loss(**batch)
    assert materialize.call_count == 2
    for call in materialize.call_args_list:
        fast = call.args[1]
        assert fast.step == 0
        assert torch.count_nonzero(fast.delta_fuse) == 0
        assert torch.count_nonzero(fast.delta_complete) == 0
    assert materialize.call_args_list[0].args[0] is not materialize.call_args_list[1].args[0]
    assert materialize.call_args_list[0].args[1] is not materialize.call_args_list[1].args[1]
    assert torch.equal(result.first_anchor_c2w, result.second_anchor_c2w)
    assert torch.isfinite(result.loss)
    result.loss.backward()
    for name in ("image_encoder.0.weight", "fuse_down.weight", "complete_down.weight"):
        gradient = dict(carrier.named_parameters())[name].grad
        assert gradient is not None and torch.isfinite(gradient).all()
        assert torch.count_nonzero(gradient) > 0


def test_query_camera_and_labels_cannot_change_constructed_states(batch):
    first = common_anchor_measurement_loss(**batch)
    query = batch["query_cameras"]
    moved = query.c2w.clone()
    moved[..., 0, 3] += 1
    second = common_anchor_measurement_loss(**{
        **batch, "query_cameras": Cameras(query.intrinsics, moved, query.image_size),
        "query_rgb": batch["query_rgb"] * 0.5, "query_depth": batch["query_depth"] * 2,
    })
    assert hash_scene_state(first.first_state) == hash_scene_state(second.first_state)
    assert hash_scene_state(first.second_state) == hash_scene_state(second.second_state)
    events = []
    carrier, renderer = batch["carrier"], batch["renderer"]
    materialize, render = carrier.materialize, renderer.forward

    def record_state(*args, **kwargs):
        events.append("state")
        return materialize(*args, **kwargs)

    def record_query(*args, **kwargs):
        events.append("query")
        return render(*args, **kwargs)

    with patch.object(carrier, "materialize", side_effect=record_state):
        with patch.object(renderer, "forward", side_effect=record_query):
            common_anchor_measurement_loss(**batch)
    assert events == ["state", "state", "query", "query"]


@pytest.mark.parametrize("component", ["rgb", "pose", "intrinsics"])
def test_anchor_content_mismatch_rejected(batch, component):
    anchor = batch["second_context"][0]
    if component == "rgb":
        changed = replace(anchor, rgb=anchor.rgb + 0.1)
    else:
        pose, intrinsics = anchor.camera.c2w.clone(), anchor.camera.intrinsics.clone()
        if component == "pose":
            pose[0, 3] += 0.1
        else:
            intrinsics[0, 0] += 1
        changed = replace(anchor, camera=Cameras(intrinsics, pose, (8, 8)))
    batch["second_context"][0] = changed
    with pytest.raises(ValueError, match="Common anchor"):
        common_anchor_measurement_loss(**batch)


@pytest.mark.parametrize("split", ["dev", "test", "validation", ""])
def test_nontraining_supervision_rejected(batch, split):
    with pytest.raises(ValueError, match="split_id"):
        common_anchor_measurement_loss(**{**batch, "split_id": split})


def test_overlap_and_unequal_contexts_rejected(batch):
    with pytest.raises(ValueError, match="only at the common anchor"):
        common_anchor_measurement_loss(**{
            **batch, "second_context": [observation(i) for i in (0, 2, 4)]
        })
    with pytest.raises(ValueError, match="equal counts"):
        common_anchor_measurement_loss(**{**batch, "second_context": [observation(0)]})
    with pytest.raises(ValueError, match="disjoint"):
        common_anchor_measurement_loss(**{**batch, "query_frame_ids": [1]})


def test_invalid_depth_is_masked_and_empty_depth_rejected(batch):
    batch["query_depth"][..., 0, 0] = float("nan")
    batch["query_depth"][..., 0, 1] = float("inf")
    result = common_anchor_measurement_loss(**batch)
    assert torch.isfinite(result.loss)
    result.loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in batch["carrier"].parameters()
               if p.grad is not None)
    with pytest.raises(ValueError, match="valid metric depth"):
        common_anchor_measurement_loss(**{
            **batch, "query_depth": torch.zeros_like(batch["query_depth"])
        })
