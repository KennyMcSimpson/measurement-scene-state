import copy

import pytest
import torch

from mcss.dynamic.budget import WorkBudget
from mcss.dynamic.carrier import DynamicSceneCarrier
from mcss.dynamic.config import CarrierConfig, WriteConfig
from mcss.dynamic.policy import FixedPolicy, LearnedActionPolicy
from mcss.dynamic.runner import StreamingRunner
from mcss.dynamic.types import Action, OnlineObservation, hash_scene_state, hash_value
from mcss.dynamic.write_rule import DirectWriteRule
from mcss.measurements import FixedMeasurementRenderer
from mcss.types import Cameras


def observation(frame, brightness=0.2):
    pose = torch.eye(4)
    pose[0, 3] = 0.02 * frame
    intrinsics = torch.tensor([[6.0, 0, 4.5], [0, 6.0, 3.5], [0, 0, 1.0]])
    return OnlineObservation(
        "scene", frame, torch.full((3, 8, 10), brightness), Cameras(intrinsics, pose, (8, 10))
    )


def make_runner(action=Action.OFF, **kwargs):
    torch.manual_seed(8)
    config = CarrierConfig(
        feature_dim=4, hidden_dim=4, expansion_dim=8, grid_size=(4, 4, 4), token_count=16
    )
    return StreamingRunner(
        DynamicSceneCarrier(config),
        DirectWriteRule(config, WriteConfig()),
        FixedMeasurementRenderer(n_samples=8),
        FixedPolicy(action),
        **kwargs,
    )


def reset(runner, episode="first", steps=2):
    runner.reset(
        [observation(0), observation(1, 0.3)],
        episode_id=episode,
        scene_id="scene",
        split_id="dev",
        query_vault_id="vault",
        stream_steps=steps,
        query_count=1,
    )


def test_off_accepts_new_evidence_and_seal_has_no_mutable_alias():
    runner = make_runner()
    reset(runner)
    before = hash_scene_state(runner.scene_state)
    runner.step(observation(2, 0.8))
    assert runner.cache.revision == 3
    assert torch.count_nonzero(runner.fast.delta_fuse) == 0
    assert hash_scene_state(runner.scene_state) != before
    with pytest.raises(ValueError, match="consumed"):
        runner.seal()
    runner.step(observation(3, 0.9))
    sealed = runner.seal()
    assert sealed.observed_ids == (0, 1, 2, 3)
    assert runner.budget.calls["materialize"] == 3
    assert runner.budget.calls["encode"] == 4
    assert runner.budget.calls["trace"] == 0
    runner.scene_state.density_logits.add_(1)
    assert hash_scene_state(sealed.scene_state) == sealed.state_hash
    with pytest.raises(ValueError, match="accepting"):
        runner.step(observation(4))


def test_all_updates_private_fast_weights_and_reset_keeps_slow_checkpoint():
    runner = make_runner(Action.ALL)
    weights_before = hash_value(runner.carrier.state_dict())
    reset(runner)
    seen = []

    class InspectPolicy:
        def choose(self, control, feasible):
            seen.append((control.observed_count, runner.cache.revision))
            return Action.ALL

    runner.policy = InspectPolicy()
    runner.step(observation(2, 0.8))
    runner.step(observation(3, 0.9))
    assert seen == [(2, 2), (3, 3)]
    assert runner.fast.step == 2
    assert torch.count_nonzero(runner.fast.delta_fuse) > 0
    assert torch.count_nonzero(runner.fast.delta_complete) > 0
    assert hash_value(runner.carrier.state_dict()) == weights_before
    assert all(p.grad is None and not p.requires_grad for p in runner.carrier.parameters())
    first_fast = runner.fast
    runner.seal()
    reset(runner, "second")
    assert runner.fast.delta_fuse.data_ptr() != first_fast.delta_fuse.data_ptr()
    assert torch.count_nonzero(runner.fast.delta_fuse) == 0
    assert runner.fast.episode_id == "second"


def test_prefix_result_does_not_depend_on_unseen_future_and_budget_rejects_early():
    first = make_runner(Action.ALL)
    second = copy.deepcopy(first)
    reset(first)
    reset(second)
    first.step(observation(2, 0.6))
    second.step(observation(2, 0.6))
    assert hash_scene_state(first.scene_state) == hash_scene_state(second.scene_state)
    # Different future frames can change their successors, but not the stored common prefix.
    prefix = hash_scene_state(first.scene_state)
    first.step(observation(3, 0.1))
    assert hash_scene_state(second.scene_state) == prefix
    with pytest.raises(ValueError, match="Budget"):
        reset(make_runner(max_units=1))


def test_minimum_budget_forces_off_without_dropping_any_observation():
    runner = make_runner(Action.ALL)
    ledger = WorkBudget(runner.carrier.config, (8, 10), 8)
    mandatory = (2 * ledger.encode_units + ledger.materialize_units(2)
                 + ledger.base_future_units(2, 2, 1))
    runner.max_units = mandatory
    reset(runner)
    assert runner.step(observation(2, 0.8))["action"] == "OFF"
    assert runner.step(observation(3, 0.9))["action"] == "OFF"
    assert runner.cache.revision == 4
    assert runner.budget.remaining == ledger.render_units
    assert runner.seal().observed_ids == (0, 1, 2, 3)


def test_same_episode_reset_changes_branch_but_preserves_content_hashes():
    runner = make_runner(Action.ALL)
    seals, branches = [], []
    for _ in range(2):
        reset(runner)
        branches.append(runner.fast.branch_id)
        runner.step(observation(2, 0.8))
        runner.step(observation(3, 0.9))
        seals.append(runner.seal())
    assert branches[0] != branches[1]
    assert seals[0].state_hash == seals[1].state_hash
    assert seals[0].fast_state_hash == seals[1].fast_state_hash


def test_learned_policy_receives_prefix_control_and_is_frozen_for_deployment():
    policy = LearnedActionPolicy(hidden_dim=8, seed=7)
    runner = make_runner()
    runner.policy = policy
    # Reconstruct through the public constructor so the budget sees the MLP cost.
    runner = StreamingRunner(
        runner.carrier,
        runner.write_rule,
        runner.renderer,
        policy,
    )
    reset(runner, steps=1)

    record = runner.step(observation(2, 0.8))

    assert record["budget"]["policy_units"] == policy.declared_work_units
    assert set(record["control_input"]) == {
        "rgb_mse",
        "opacity_mean",
        "current_image_mean",
        "density_mean",
        "fast_norm",
        "observed_count",
        "previous_action",
        "remaining_budget",
        "image_feature_stats",
        "rgb_residual_4x4",
        "coverage",
        "valid_count",
        "camera_change",
        "fast_state_stats",
        "remaining_steps",
    }
    assert len(record["control_input"]["rgb_residual_4x4"]) == 48
    assert all(not parameter.requires_grad for parameter in policy.parameters())
