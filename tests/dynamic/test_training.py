import pytest
import torch

from mcss.dynamic.config import CarrierConfig, WriteConfig
from mcss.dynamic.types import Action, OnlineObservation
from mcss.dynamic.write_rule import DirectWriteRule
from mcss.geometry import make_intrinsics
from mcss.measurements import FixedMeasurementRenderer
from mcss.training.grounded import paired_context_measurement_loss
from mcss.training.write_unroll import unroll_observed_episode
from mcss.types import Cameras


def _config() -> CarrierConfig:
    return CarrierConfig(
        feature_dim=4,
        hidden_dim=4,
        expansion_dim=6,
        grid_size=(4, 4, 4),
        local_bounds_m=((-1.0, -1.0, 0.1), (1.0, 1.0, 3.0)),
        token_count=12,
    )


def _observation(frame_id: int, *, scene_id: str = "scene-a") -> OnlineObservation:
    camera_pose = torch.eye(4)
    camera_pose[0, 3] = 0.4 + 0.1 * frame_id
    camera = Cameras(make_intrinsics((8, 8), 100.0), camera_pose, (8, 8))
    return OnlineObservation(
        scene_id=scene_id,
        frame_id=frame_id,
        rgb=torch.full((3, 8, 8), 0.2 + 0.05 * frame_id),
        camera=camera,
    )


def _modules():
    from mcss.dynamic.carrier import DynamicSceneCarrier

    config = _config()
    return DynamicSceneCarrier(config), DirectWriteRule(
        config, WriteConfig(learning_rate=0.1, max_update_norm=0.2)
    )


@pytest.mark.parametrize("action", [Action.FUSE, Action.ALL])
def test_unroll_keeps_write_gradients_through_fixed_renderer(action: Action) -> None:
    torch.manual_seed(11)
    carrier, write_rule = _modules()
    renderer = FixedMeasurementRenderer(n_samples=8, ray_chunk_size=64)
    warmup = [_observation(0)]
    stream = [_observation(1)]

    state, fast, cache, anchor_c2w = unroll_observed_episode(
        carrier, write_rule, warmup, stream, [action], "episode-a"
    )
    raw_query = _observation(2).camera
    query_cameras = Cameras(
        raw_query.intrinsics.view(1, 1, 3, 3), raw_query.c2w.view(1, 1, 4, 4), (8, 8)
    )
    local_query = Cameras(
        query_cameras.intrinsics,
        (torch.linalg.inv(anchor_c2w) @ query_cameras.c2w),
        query_cameras.image_size,
    )
    with torch.no_grad():
        reference = renderer(state, local_query, {"rgb", "depth"})
    query_rgb = reference["rgb"].mul(0.4)
    query_depth = reference["depth"].add(0.1)

    from mcss.training.grounded import grounded_measurement_loss

    loss, terms = grounded_measurement_loss(
        state, query_cameras, query_rgb, query_depth, anchor_c2w, renderer
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert set(terms) == {"rgb", "depth"}
    assert cache.revision == 2
    assert torch.allclose(cache.entries[0].observation.camera.c2w, torch.eye(4))
    torch.testing.assert_close(anchor_c2w, warmup[0].camera.c2w)
    assert torch.count_nonzero(fast.delta_fuse) > 0
    if action == Action.ALL:
        assert torch.count_nonzero(fast.delta_complete) > 0
    else:
        assert torch.count_nonzero(fast.delta_complete) == 0
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for parameter in write_rule.parameters()
    )


def test_unroll_off_appends_and_episode_fast_states_are_isolated() -> None:
    carrier, write_rule = _modules()
    warmup = [_observation(0)]
    stream = [_observation(1)]
    _, written_fast, written_cache, _ = unroll_observed_episode(
        carrier, write_rule, warmup, stream, [Action.FUSE], "episode-a"
    )
    off_state, off_fast, off_cache, _ = unroll_observed_episode(
        carrier, write_rule, warmup, stream, [Action.OFF], "episode-b"
    )

    assert written_cache.episode_id == "episode-a"
    assert off_cache.episode_id == "episode-b"
    assert written_cache.revision == off_cache.revision == 2
    assert written_fast.delta_fuse.data_ptr() != off_fast.delta_fuse.data_ptr()
    assert torch.count_nonzero(off_fast.delta_fuse) == 0
    assert torch.count_nonzero(off_fast.delta_complete) == 0
    assert torch.isfinite(off_state.density_logits).all()


def test_paired_contexts_build_independently_and_reject_query_overlap() -> None:
    torch.manual_seed(12)
    carrier, _ = _modules()
    renderer = FixedMeasurementRenderer(n_samples=8, ray_chunk_size=64)
    first_context = [_observation(0)]
    second_context = [_observation(1)]
    query = _observation(2)
    query_cameras = Cameras(
        query.camera.intrinsics.view(1, 1, 3, 3), query.camera.c2w.view(1, 1, 4, 4), (8, 8)
    )
    query_rgb = query.rgb.view(1, 1, 3, 8, 8)
    query_depth = torch.ones(1, 1, 1, 8, 8)

    result = paired_context_measurement_loss(
        carrier,
        first_context,
        second_context,
        [2],
        query_cameras,
        query_rgb,
        query_depth,
        renderer,
    )
    result.loss.backward()

    assert torch.isfinite(result.loss)
    assert (
        result.first_state.density_logits.data_ptr()
        != result.second_state.density_logits.data_ptr()
    )
    assert not torch.allclose(result.first_anchor_c2w, result.second_anchor_c2w)
    assert any(parameter.grad is not None for parameter in carrier.parameters())
    with pytest.raises(ValueError, match="overlap"):
        paired_context_measurement_loss(
            carrier,
            first_context,
            second_context,
            [0],
            query_cameras,
            query_rgb,
            query_depth,
            renderer,
        )
