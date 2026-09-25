"""Differentiable offline, direct outer-product writing at deployment."""

import torch
from torch import nn

from mcss.dynamic.config import CarrierConfig, WriteConfig
from mcss.dynamic.types import Action, FastWeights, WriteProposal, WriteTrace


class DirectWriteRule(nn.Module):
    def __init__(self, carrier_config: CarrierConfig, write_config: WriteConfig):
        super().__init__()
        self.carrier_config = carrier_config
        self.write_config = write_config
        self.targets = nn.ModuleDict(
            {
                name: nn.Linear(carrier_config.statistics_dim, carrier_config.hidden_dim)
                for name in ("fuse", "complete")
            }
        )
        self.projections = nn.ParameterDict(
            {
                name: nn.Parameter(torch.eye(carrier_config.hidden_dim))
                for name in ("fuse", "complete")
            }
        )
        self.gates = nn.ParameterDict(
            {
                name: nn.Parameter(torch.linspace(-0.1, 0.1, carrier_config.hidden_dim))
                for name in ("fuse", "complete")
            }
        )

    def propose(self, trace: WriteTrace) -> WriteProposal:
        count = trace.candidate_ids.numel()
        config = self.carrier_config
        if count == 0 or trace.candidate_ids.unique().numel() != count:
            raise ValueError("Write candidates must be nonempty and unique")
        if trace.observed_feature_statistics.shape != (count, config.statistics_dim):
            raise ValueError("Write statistics shape mismatch")
        if trace.support_weights.shape != (count,) or (trace.support_weights < 0).any():
            raise ValueError("Invalid write support weights")
        tensors = (
            trace.fuse_activation,
            trace.complete_activation,
            trace.observed_feature_statistics,
            trace.support_weights,
        )
        if not all(torch.isfinite(t).all() for t in tensors):
            raise ValueError("Nonfinite write trace")
        q = trace.support_weights / trace.support_weights.sum().clamp_min(1)
        changes = {}
        for name, activation in (
            ("fuse", trace.fuse_activation),
            ("complete", trace.complete_activation),
        ):
            if activation.shape != (count, config.expansion_dim):
                raise ValueError("Write activation shape mismatch")
            target = torch.tanh(self.targets[name](trace.observed_feature_statistics))
            value = (self.gates[name] * target) @ self.projections[name]
            delta = self.write_config.learning_rate * torch.einsum(
                "ni,nj->ij", q[:, None] * value, activation
            )
            factor = (
                self.write_config.max_update_norm
                / delta.norm().clamp_min(torch.finfo(delta.dtype).eps)
            ).clamp(max=1)
            changes[name] = delta * factor
        return WriteProposal(
            trace.episode_id,
            trace.cache_revision,
            trace.fast_step,
            changes["fuse"],
            changes["complete"],
            branch_id=trace.branch_id,
            source_state_id=trace.source_state_id,
        )


def commit_write(
    fast: FastWeights, proposal: WriteProposal, action: Action, *, cache_revision: int
) -> FastWeights:
    action = Action(action)
    if proposal.episode_id != fast.episode_id:
        raise ValueError("Write proposal belongs to another episode")
    if proposal.branch_id != fast.branch_id:
        raise ValueError("Write proposal belongs to another trajectory branch")
    if proposal.fast_step != fast.step or proposal.cache_revision != cache_revision:
        raise ValueError("Stale write proposal version")
    if proposal.source_state_id != fast.state_id:
        raise ValueError("Write proposal was derived from another fast state")
    if proposal.delta_fuse.shape != fast.delta_fuse.shape or (
        proposal.delta_complete.shape != fast.delta_complete.shape
    ):
        raise ValueError("Write proposal matrix shape mismatch")
    fuse = fast.delta_fuse + (proposal.delta_fuse if action in (Action.FUSE, Action.ALL) else 0)
    complete = fast.delta_complete + (
        proposal.delta_complete if action in (Action.COMPLETE, Action.ALL) else 0
    )
    return FastWeights(
        fast.episode_id, fuse, complete, fast.step + int(action != Action.OFF), fast.branch_id
    )
