from dataclasses import replace

import pytest
import torch

from mcss.dynamic.config import CarrierConfig, WriteConfig
from mcss.dynamic.types import Action, FastWeights, WriteTrace
from mcss.dynamic.write_rule import DirectWriteRule, commit_write


def make_trace(fast, support=True):
    return WriteTrace(
        "episode",
        3,
        0,
        torch.randn(5, 16),
        torch.randn(5, 16),
        torch.randn(5, 23),
        torch.ones(5) if support else torch.zeros(5),
        torch.arange(5),
        branch_id=fast.branch_id,
        source_state_id=fast.state_id,
    )


def test_writes_update_selected_locations_without_mutating_source_and_keep_gradients():
    torch.manual_seed(9)
    rule = DirectWriteRule(CarrierConfig(), WriteConfig(max_update_norm=0.01))
    fast = FastWeights("episode", torch.zeros(8, 16), torch.zeros(8, 16))
    proposal = rule.propose(make_trace(fast))
    updated = commit_write(fast, proposal, Action.FUSE, cache_revision=3)
    assert torch.count_nonzero(updated.delta_fuse) > 0
    assert torch.equal(updated.delta_complete, fast.delta_complete)
    assert torch.count_nonzero(fast.delta_fuse) == 0
    assert proposal.delta_fuse.norm() <= 0.010001
    updated.delta_fuse.square().sum().backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in rule.parameters())
    both = commit_write(fast, proposal, Action.ALL, cache_revision=3)
    assert torch.equal(both.delta_fuse, updated.delta_fuse)
    assert torch.equal(both.delta_complete, proposal.delta_complete)
    with pytest.raises(ValueError, match="version"):
        commit_write(both, proposal, Action.ALL, cache_revision=3)
    with pytest.raises(ValueError, match="episode"):
        commit_write(
            FastWeights("other", torch.zeros(8, 16), torch.zeros(8, 16)),
            proposal,
            Action.ALL,
            cache_revision=3,
        )


def test_no_supported_locations_cannot_write_and_off_ignores_increment():
    rule = DirectWriteRule(CarrierConfig(), WriteConfig())
    fast = FastWeights("episode", torch.zeros(8, 16), torch.zeros(8, 16))
    unsupported = rule.propose(make_trace(fast, False))
    assert torch.count_nonzero(unsupported.delta_fuse) == 0
    assert torch.count_nonzero(unsupported.delta_complete) == 0
    proposal = rule.propose(make_trace(fast))
    off = commit_write(fast, proposal, Action.OFF, cache_revision=3)
    assert torch.equal(off.delta_fuse, fast.delta_fuse)
    assert torch.equal(off.delta_complete, fast.delta_complete)


def test_proposals_cannot_cross_same_episode_branches_or_resets():
    rule = DirectWriteRule(CarrierConfig(), WriteConfig())
    fast = FastWeights("episode", torch.zeros(8, 16), torch.zeros(8, 16))
    proposal = rule.propose(make_trace(fast))
    fork = fast.fork()
    fresh = FastWeights("episode", torch.zeros(8, 16), torch.zeros(8, 16))
    for other in (fork, fresh):
        assert other.branch_id != fast.branch_id
        with pytest.raises(ValueError, match="branch"):
            commit_write(other, proposal, Action.ALL, cache_revision=3)
    fork.delta_fuse.add_(1)
    assert torch.count_nonzero(fast.delta_fuse) == 0
    updated = commit_write(fast, proposal, Action.ALL, cache_revision=3)
    assert updated.branch_id == fast.branch_id


def test_fork_preserves_offline_gradient_path():
    source = torch.ones(8, 16, requires_grad=True)
    fast = FastWeights("episode", source, source * 2)
    fork = fast.fork()
    (fork.delta_fuse.sum() + fork.delta_complete.sum()).backward()
    assert torch.equal(source.grad, torch.full_like(source, 3))


def test_same_source_commits_cannot_cross_their_divergent_successors():
    rule = DirectWriteRule(CarrierConfig(), WriteConfig())
    fast = FastWeights("episode", torch.zeros(8, 16), torch.zeros(8, 16))
    proposal = rule.propose(make_trace(fast))
    fuse_only = commit_write(fast, proposal, Action.FUSE, cache_revision=3)
    complete_only = commit_write(fast, proposal, Action.COMPLETE, cache_revision=3)
    later = rule.propose(replace(make_trace(fuse_only), fast_step=1, cache_revision=4))
    commit_write(fuse_only, later, Action.ALL, cache_revision=4)
    with pytest.raises(ValueError, match="state"):
        commit_write(complete_only, later, Action.ALL, cache_revision=4)
    assert not torch.equal(fuse_only.delta_fuse, complete_only.delta_fuse)
