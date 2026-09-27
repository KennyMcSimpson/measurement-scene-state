"""Information-boundary and transaction tests; no scientific efficacy assertions."""

from unittest.mock import patch

import pytest
import torch

from mcss.dynamic.carrier import DynamicSceneCarrier
from mcss.dynamic.config import CarrierConfig, WriteConfig
from mcss.dynamic.types import Action, OnlineObservation, hash_scene_state, hash_value
from mcss.dynamic.write_rule import DirectWriteRule
from mcss.measurements import FixedMeasurementRenderer
from mcss.mechanism_pilot.runtime import FrameRoles, MechanismRuntime
from mcss.types import Cameras


def observation(frame):
    pose = torch.eye(4)
    pose[0, 3] = frame * 0.025
    intrinsics = torch.tensor([[6.0, 0, 3.5], [0, 6.0, 3.5], [0, 0, 1.0]])
    return OnlineObservation(
        "scene", frame, torch.full((3, 8, 8), 0.1 + frame * 0.07), Cameras(intrinsics, pose, (8, 8))
    )


@pytest.fixture
def runtime():
    torch.manual_seed(21)
    config = CarrierConfig(
        feature_dim=4, hidden_dim=4, expansion_dim=8, grid_size=(4, 4, 4), token_count=16
    )
    roles = FrameRoles((0, 1, 2), (0, 3, 4), (5, 6), (7,), (8, 9))
    return MechanismRuntime(
        DynamicSceneCarrier(config),
        DirectWriteRule(config, WriteConfig()),
        FixedMeasurementRenderer(n_samples=8),
        roles,
    )


def base(runtime):
    return runtime.build([observation(i) for i in (0, 1, 2)], episode_id="episode")


def test_off_appends_without_fast_write_and_feedback_precedes_append(runtime):
    initial = base(runtime)
    feedback = runtime.preview(initial, observation(5))
    child = runtime.step(initial, observation(5), Action.OFF)
    assert child.cache.revision == initial.cache.revision + 1
    assert torch.equal(child.fast.delta_fuse, initial.fast.delta_fuse)
    assert torch.equal(child.fast.delta_complete, initial.fast.delta_complete)
    assert child.fast.step == initial.fast.step
    assert child.history[-1]["pre_summary"]["observed_count"] == 3
    assert child.history[-1]["feedback"]["rgb_mse"] == feedback["rgb_mse"]
    assert child.history[-1]["pre_summary"]["state_hash"] == hash_scene_state(initial.state)


def test_all_uses_one_trace_and_synchronous_old_weight_proposals(runtime):
    initial = base(runtime)
    seen = []
    original = runtime.writer.propose

    def inspect(trace):
        result = original(trace)
        seen.append((trace, result))
        return result

    with patch.object(runtime.carrier, "trace", wraps=runtime.carrier.trace) as trace_call:
        with patch.object(runtime.writer, "propose", side_effect=inspect):
            result = runtime.step(initial, observation(5), Action.ALL)
    assert trace_call.call_count == len(seen) == 1
    trace, proposal = seen[0]
    assert trace.fast_step == initial.fast.step
    assert torch.equal(result.fast.delta_fuse, initial.fast.delta_fuse + proposal.delta_fuse)
    assert torch.equal(
        result.fast.delta_complete, initial.fast.delta_complete + proposal.delta_complete
    )


def test_candidates_own_cache_fast_scene_history_and_seal(runtime):
    histories = runtime.matched_histories(
        [observation(i) for i in (0, 1, 2)], [observation(5), observation(6)], episode_id="episode"
    )
    candidates = runtime.continuation_branches(histories, observation(7))
    left, right = candidates["FC"]["FUSE"], candidates["FC"]["COMPLETE"]
    right_hash = hash_scene_state(right.state)
    original_hash = hash_scene_state(histories["FC"].state)
    sealed = runtime.seal(left)
    left.fast.delta_fuse.add_(1)
    left.cache.entries[0].features.add_(1)
    left.cache.entries[0].observation.rgb.add_(1)
    left.state.density_logits.add_(1)
    left.history[0]["feedback"]["rgb_mse"] = 999
    assert hash_scene_state(right.state) == right_hash
    assert hash_scene_state(histories["FC"].state) == original_hash
    assert hash_scene_state(sealed.scene_state) == sealed.state_hash
    assert right.history[0]["feedback"]["rgb_mse"] != 999
    assert torch.all(right.cache.entries[0].observation.rgb < 1)
    assert left.fast.branch_id != right.fast.branch_id


def test_fc_cf_matched_observations_actions_sampling_and_compute(runtime):
    histories = runtime.matched_histories(
        [observation(i) for i in (0, 1, 2)], [observation(5), observation(6)], episode_id="episode"
    )
    fc, cf = histories.values()
    assert fc.summary["observed_ids"] == cf.summary["observed_ids"] == [0, 1, 2, 5, 6]
    assert sorted(r["action"] for r in fc.history) == sorted(r["action"] for r in cf.history)
    assert [r["action"] for r in fc.history] == ["FUSE", "COMPLETE"]
    for a, b in zip(fc.history, cf.history, strict=True):
        assert a["observation_hash"] == b["observation_hash"]
        assert a["candidate_pack_hash"] == b["candidate_pack_hash"]
        assert a["operations"] == b["operations"]
    assert fc.fast.step == cf.fast.step == 2


def test_query_rejected_from_construction_and_features(runtime):
    with pytest.raises(ValueError, match="Sealed query"):
        runtime.build([observation(8)], episode_id="bad")
    initial = base(runtime)
    with pytest.raises(ValueError, match="Sealed query"):
        runtime.preview(initial, observation(8))
    with pytest.raises(ValueError, match="Sealed query"):
        runtime.step(initial, observation(8), Action.ALL)


def test_depth_cannot_enter_runtime_policy_feedback(runtime):
    initial = base(runtime)
    with pytest.raises(TypeError):
        runtime.preview(initial, observation(5), depth=torch.ones(8, 8))
    with pytest.raises(TypeError):
        runtime.preview(initial, {"rgb": observation(5).rgb, "depth": torch.ones(8, 8)})
    assert not any("depth" in k or "reward" in k for k in runtime.preview(initial, observation(5)))


def test_last_commit_fresh_materialization_and_recomputed_summary(runtime):
    initial = base(runtime)
    before = initial.summary
    with patch.object(runtime.carrier, "materialize", wraps=runtime.carrier.materialize) as read:
        child = runtime.step(initial, observation(5), Action.ALL)
    assert read.call_count == 1
    assert read.call_args.args[0] is child.cache
    assert read.call_args.args[1] is child.fast
    assert child.summary["observed_count"] == 4
    assert child.summary["fast_step"] == 1
    assert child.summary["previous_action"] == "ALL"
    assert child.summary["state_hash"] == hash_scene_state(child.state)
    assert child.summary["fast_hash"] != before["fast_hash"]
    assert child.summary["fast_fuse_norm"] == float(child.fast.delta_fuse.norm())
    assert child.summary["previous_write_magnitude"] > 0


def test_independent_contexts_only_share_anchor_and_no_mutable_alias(runtime):
    left = base(runtime)
    right = runtime.build([observation(i) for i in (0, 3, 4)], episode_id="other")
    assert set(left.summary["observed_ids"]) & set(right.summary["observed_ids"]) == {0}
    assert torch.equal(left.anchor_c2w, right.anchor_c2w)
    original = hash_value(right.cache)
    left.cache.entries[0].features.add_(1)
    assert hash_value(right.cache) == original
    assert not torch.count_nonzero(left.fast.delta_fuse)
    assert not torch.count_nonzero(right.fast.delta_complete)
    with pytest.raises(ValueError, match="overlap"):
        FrameRoles((0, 1), (0, 2), (3, 4), (5,), (4, 6))
