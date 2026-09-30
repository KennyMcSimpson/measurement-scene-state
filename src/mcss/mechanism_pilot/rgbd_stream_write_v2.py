"""Core B V2: learned fast-weight writes on the frozen V9 carrier (32^3, CONTEXT_DEPTH bounds).

Identical to Core B V1 except for the frozen slow carrier and the loss that follows its grid:
the V9 selection fixed before any V9 result by PLAN_BEFORE_V9_RESULTS.md (V9 C1 VARIABLE3TO7 if
the V9 VIEWCOUNT_GAIN point estimate is > 0, else V9 C0 FIXED3), a V8 resolution carrier at
32^3. The stream rule, policies, the DirectWriteRule and its offline TRAIN72 training are those
of Core B V1; OFF_CLAMP3 keeps its V1 meaning (the lifting support count clamped at 3).
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn.functional as functional

from mcss.dynamic.carrier import _CarrierComputation
from mcss.dynamic.config import CarrierConfig, WriteConfig
from mcss.dynamic.lifting import lift_cached_observations
from mcss.dynamic.types import Action, WriteTrace, hash_value
from mcss.dynamic.write_rule import DirectWriteRule, commit_write
from mcss.mechanism_pilot import rgbd_evidence_carrier as v5
from mcss.mechanism_pilot import rgbd_resolution_carrier as v8
from mcss.mechanism_pilot import rgbd_stream_write as v1
from mcss.mechanism_pilot.small_training import sha

EXPERIMENT = "EXP-3D-RGBD-DYNAMIC-WRITE-V2"
GRID = 32
STREAM_LENGTH = v1.STREAM_LENGTH
TRAINED_VIEW_COUNT = v1.TRAINED_VIEW_COUNT
POLICIES = v1.POLICIES
CONTROLS = v1.CONTROLS
ACTIONS = v1.ACTIONS
WRITE_SCHEMA = "mcss.rgbd_stream_write_rule.v2"
V8_LOSS_VARIANT = "C0"  # every V8 variant registers the frozen V2 C1 surface loss
stream_frame_ids = v1.stream_frame_ids
StreamRGBDLoader = v1.StreamRGBDLoader
rebuild_with = v1.rebuild_with
make_write_rule = v1.make_write_rule


class StreamResolutionCarrier(v8.ResolutionRGBDCarrier):
    """The frozen V9 (V8 resolution) carrier; OFF_CLAMP3 clamps the support-count statistic."""

    count_clamp = None

    def _compute(self, cache, fast):
        self._validate_fast(cache, fast)
        if self._pending_depth is None:
            raise PermissionError("RGB-D carrier states are built only through build_state")
        frame_ids, depths = self._pending_depth
        if tuple(entry.observation.frame_id for entry in cache.entries) != frame_ids:
            raise ValueError("Measured depth maps must align with the cached context views")
        lifted = lift_cached_observations(
            cache,
            self._candidate_points,
            self._candidate_normalized_xyz,
            feature_dim=self.config.feature_dim,
        )
        statistics = lifted.observed_feature_statistics.to(dtype=self.fuse_up.weight.dtype)
        if self.count_clamp is not None:
            column = 2 * self.config.feature_dim
            statistics = torch.cat(
                (
                    statistics[:, :column],
                    statistics[:, column : column + 1].clamp(max=float(self.count_clamp)),
                    statistics[:, column + 1 :],
                ),
                dim=-1,
            )
        support_weights = lifted.support_weights.to(dtype=statistics.dtype)
        fuse_activation = functional.gelu(self.fuse_up(statistics))
        fused_hidden = functional.linear(
            fuse_activation,
            self.fuse_down.weight + fast.delta_fuse,
            self.fuse_down.bias,
        )
        fused_hidden = fused_hidden * support_weights.unsqueeze(-1)
        grid = torch.tensor(
            tuple(self.config.grid_size), device=self._bounds.device, dtype=self._bounds.dtype
        )
        edge = ((self._bounds[:, 1] - self._bounds[:, 0]) / grid).mean()
        measured = v5.depth_statistics(
            [entry.observation for entry in cache.entries],
            depths,
            self._candidate_points,
            self.depth_log_sigma,
            edge.to(device=self._candidate_points.device, dtype=torch.float32),
        ).to(dtype=statistics.dtype)
        fused_hidden = fused_hidden + functional.linear(measured, self.depth_weight)
        refined = self.refinement(self._scatter_candidates(fused_hidden))
        complete_input = self._gather_candidates(refined)
        complete_activation = functional.gelu(self.complete_up(complete_input))
        trace = WriteTrace(
            episode_id=cache.episode_id,
            cache_revision=cache.revision,
            fast_step=fast.step,
            fuse_activation=fuse_activation,
            complete_activation=complete_activation,
            observed_feature_statistics=statistics,
            support_weights=support_weights,
            candidate_ids=self._candidate_ids.clone(),
            branch_id=fast.branch_id,
            source_state_id=fast.state_id,
        )
        return _CarrierComputation(trace, refined)


def load_frozen_carrier(path, device):
    """The sealed V9 weights in a StreamResolutionCarrier; every parameter frozen."""
    model, payload = v8.load_checkpoint(path, device)
    if tuple(model.config.grid_size) != (GRID, GRID, GRID):
        raise PermissionError("Core B V2 runs on the 32^3 V9 carrier only")
    carrier = StreamResolutionCarrier(model.config).to(device)
    carrier.load_state_dict(model.state_dict(), strict=True)
    if hash_value(carrier.state_dict()) != hash_value(model.state_dict()):
        raise RuntimeError("Frozen carrier transfer changed a tensor")
    carrier.eval()
    for parameter in carrier.parameters():
        parameter.requires_grad_(False)
    return carrier, payload


def unroll(carrier, rule, warmup, stream, bounds, policy, episode_id):
    """(final state, anchor c2w, final fast weights, episode) of one fixed-policy episode."""
    if policy not in POLICIES:
        raise ValueError(f"Unregistered policy {policy}")
    if not isinstance(warmup, v5.RGBDContext) or not isinstance(stream, v5.RGBDContext):
        raise TypeError("Warmup and stream must be RGBDContext values")
    if not isinstance(carrier, StreamResolutionCarrier):
        raise TypeError("Core B V2 requires the StreamResolutionCarrier")
    episode = v1._Episode(carrier, bounds, episode_id, warmup[0].scene_id)
    carrier.count_clamp = TRAINED_VIEW_COUNT if policy == "OFF_CLAMP3" else None
    try:
        fast = carrier.initial_fast(episode_id)
        for observation, depth in zip(warmup, warmup.depths, strict=True):
            episode.arrive(observation, depth)
        if policy != "NO_STREAM":
            for observation, depth in zip(stream, stream.depths, strict=True):
                episode.arrive(observation, depth)
                action = ACTIONS[policy]
                if action != Action.OFF:
                    # ALL is committed from one trace, so both matrices see identical old weights.
                    proposal = rule.propose(episode.run(fast, "trace"))
                    fast = commit_write(
                        fast, proposal, action, cache_revision=episode.cache.revision
                    )
        state = episode.run(fast, "materialize")
    finally:
        carrier.count_clamp = None
    return state, episode.anchor.clone(), fast, episode


def save_write_rule(path, rule, *, seed, step, lock_sha256, carrier_sha256):
    payload = {
        "schema": WRITE_SCHEMA,
        "carrier_config": asdict(rule.carrier_config),
        "write_config": asdict(rule.write_config),
        "rule": {k: v.detach().cpu() for k, v in rule.state_dict().items()},
        "seed": seed,
        "step": step,
        "lock_sha256": lock_sha256,
        "carrier_checkpoint_sha256": carrier_sha256,
        "slow_carrier_updated": False,
    }
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    torch.save(payload, path)
    restored, _ = load_write_rule(path, "cpu")
    if hash_value(restored.state_dict()) != hash_value(payload["rule"]):
        raise RuntimeError("Write-rule restore mismatch")
    return sha(path)


def load_write_rule(path, device):
    payload = torch.load(path, map_location=device, weights_only=True)
    if payload.get("schema") != WRITE_SCHEMA or payload.get("slow_carrier_updated"):
        raise PermissionError("Core B V2 write-rule schema required")
    carrier_config = CarrierConfig(**payload["carrier_config"])
    rule = DirectWriteRule(carrier_config, WriteConfig(**payload["write_config"])).to(device)
    rule.load_state_dict(payload["rule"], strict=True)
    return rule, payload


def training_loss(state, local_cameras, rgb, depth, indices):
    """The frozen V2 C1 loss at 32^3 (the loss the V9 carrier was trained with)."""
    return v8.training_loss(state, local_cameras, rgb, depth, indices, V8_LOSS_VARIANT)
