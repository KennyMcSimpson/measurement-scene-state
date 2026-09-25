from dataclasses import replace

import pytest
import torch

from mcss.dynamic.cache import ObservationCache
from mcss.dynamic.carrier import DynamicSceneCarrier
from mcss.dynamic.config import CarrierConfig
from mcss.dynamic.types import FastWeights, OnlineObservation
from mcss.geometry import make_intrinsics
from mcss.types import Cameras


def _config() -> CarrierConfig:
    return CarrierConfig(
        feature_dim=4,
        hidden_dim=4,
        expansion_dim=6,
        grid_size=(4, 4, 4),
        token_count=16,
    )


def _camera(*, backward: bool = False) -> Cameras:
    pose = torch.eye(4)
    if backward:
        pose[0, 0] = -1.0
        pose[2, 2] = -1.0
    return Cameras(make_intrinsics((8, 8), 170.0), pose, (8, 8))


def _cache(carrier: DynamicSceneCarrier, *, backward: bool = False) -> ObservationCache:
    cache = ObservationCache(episode_id="episode-a", scene_id="scene-a")
    for frame_id in range(2):
        observation = OnlineObservation(
            scene_id="scene-a",
            frame_id=frame_id,
            rgb=torch.full((3, 8, 8), 0.25 + 0.1 * frame_id),
            camera=_camera(backward=backward),
        )
        cache = cache.append(observation, carrier.encode(observation))
    return cache


def test_carrier_materializes_finite_typed_scene_and_trace() -> None:
    torch.manual_seed(2)
    carrier = DynamicSceneCarrier(_config())
    cache = _cache(carrier)
    fast = carrier.initial_fast(cache.episode_id)

    trace = carrier.trace(cache, fast)
    state = carrier.materialize(cache, fast)

    assert state.density_logits.shape == (1, 1, 4, 4, 4)
    assert state.color.shape == (1, 3, 4, 4, 4)
    assert state.log_variance.shape == (1, 1, 4, 4, 4)
    assert torch.isfinite(state.density_logits).all()
    assert torch.isfinite(state.color).all()
    assert torch.isfinite(state.log_variance).all()
    assert trace.observed_feature_statistics.shape == (16, _config().statistics_dim)
    assert trace.fuse_activation.shape == (16, _config().expansion_dim)
    assert trace.complete_activation.shape == (16, _config().expansion_dim)
    assert trace.candidate_ids.shape == (16,)
    assert torch.all((trace.support_weights == 0) | (trace.support_weights == 1))
    assert torch.count_nonzero(trace.support_weights) > 0


def test_carrier_backpropagates_to_slow_parameters_and_fast_tensors() -> None:
    torch.manual_seed(3)
    carrier = DynamicSceneCarrier(_config())
    cache = _cache(carrier)
    initial = carrier.initial_fast(cache.episode_id)
    fast = FastWeights(
        episode_id=initial.episode_id,
        delta_fuse=torch.zeros_like(initial.delta_fuse, requires_grad=True),
        delta_complete=torch.zeros_like(initial.delta_complete, requires_grad=True),
    )

    state = carrier.materialize(cache, fast)
    (state.density_logits.mean() + state.color.mean() + state.log_variance.mean()).backward()

    assert fast.delta_fuse.grad is not None
    assert fast.delta_complete.grad is not None
    assert torch.isfinite(fast.delta_fuse.grad).all()
    assert torch.isfinite(fast.delta_complete.grad).all()
    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in carrier.parameters()
    )


def test_carrier_rejects_fast_weights_from_another_episode() -> None:
    carrier = DynamicSceneCarrier(_config())
    cache = _cache(carrier)
    wrong_episode = carrier.initial_fast("episode-b")

    with pytest.raises(ValueError, match="episode"):
        carrier.trace(cache, wrong_episode)
    with pytest.raises(ValueError, match="episode"):
        carrier.materialize(cache, wrong_episode)


def test_each_writable_down_projection_can_change_the_scene_independently() -> None:
    torch.manual_seed(4)
    carrier = DynamicSceneCarrier(_config())
    cache = _cache(carrier)
    initial = carrier.initial_fast(cache.episode_id)
    fuse_only = replace(initial, delta_fuse=torch.ones_like(initial.delta_fuse))
    complete_only = replace(initial, delta_complete=torch.ones_like(initial.delta_complete))

    baseline = carrier.materialize(cache, initial)
    fused = carrier.materialize(cache, fuse_only)
    completed = carrier.materialize(cache, complete_only)

    assert carrier.fuse_down.weight.shape == (4, 6)
    assert carrier.complete_down.weight.shape == (4, 6)
    assert not torch.allclose(fused.density_logits, baseline.density_logits)
    assert not torch.allclose(completed.density_logits, baseline.density_logits)


def test_zero_geometric_support_has_zero_write_weight_and_finite_scene() -> None:
    carrier = DynamicSceneCarrier(_config())
    cache = _cache(carrier, backward=True)
    fast = carrier.initial_fast(cache.episode_id)

    trace = carrier.trace(cache, fast)
    state = carrier.materialize(cache, fast)

    assert torch.count_nonzero(trace.support_weights) == 0
    assert torch.isfinite(trace.observed_feature_statistics).all()
    assert torch.isfinite(state.density_logits).all()
    assert torch.isfinite(state.color).all()
    assert torch.isfinite(state.log_variance).all()
