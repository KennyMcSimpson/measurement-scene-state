"""Core B V1: learned fast-weight writes over an RGB-D stream on the frozen qualified carrier.

An episode is one frozen V2 context role (the warmup, RGB-D) followed by STREAM_LENGTH stream
RGB-D frames chosen by a label-free frame-id rule. The slow carrier is a sealed V7 C1 selection
(Core A, qualified on FRESH-V2) and is never updated; only the project's DirectWriteRule
(WriteConfig defaults) is learned, offline on TRAIN72. Fixed policies per episode:

- NO_STREAM: the warmup state (the static Core A state);
- OFF: every stream frame is cached and the state rebuilt, nothing is written;
- OFF_CLAMP3: as OFF, with the lifting support-count statistic clamped at 3, the view count
  the carrier was trained with (a non-learned control for the view-count shift);
- FUSE / COMPLETE / ALL: as OFF, and each arrival's write proposal is committed to the fuse,
  the complete, or both fast matrices;
- ALL_UNTRAINED: ALL with the initial (seeded, never trained) write rule.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn.functional as functional
from torch import nn

from mcss.dynamic.cache import ObservationCache
from mcss.dynamic.carrier import _CarrierComputation
from mcss.dynamic.config import CarrierConfig, WriteConfig
from mcss.dynamic.feedback import anchored_observation
from mcss.dynamic.lifting import lift_cached_observations
from mcss.dynamic.types import Action, FastWeights, OnlineObservation, WriteTrace, hash_value
from mcss.dynamic.write_rule import DirectWriteRule, commit_write
from mcss.mechanism_pilot import rgbd_evidence_carrier as v5
from mcss.mechanism_pilot.direct_capacity_contracts import _Media
from mcss.mechanism_pilot.rgbd_bounds_carrier import load_checkpoint
from mcss.mechanism_pilot.small_training import sha
from mcss.types import Cameras

EXPERIMENT = "EXP-3D-RGBD-DYNAMIC-WRITE-V1"
STREAM_LENGTH = 4
TRAINED_VIEW_COUNT = 3
POLICIES = ("NO_STREAM", "OFF", "OFF_CLAMP3", "FUSE", "COMPLETE", "ALL", "ALL_UNTRAINED")
CONTROLS = ("ALL_WRONG_SCENE",)
ACTIONS = {
    "OFF": Action.OFF,
    "OFF_CLAMP3": Action.OFF,
    "FUSE": Action.FUSE,
    "COMPLETE": Action.COMPLETE,
    "ALL": Action.ALL,
    "ALL_UNTRAINED": Action.ALL,
}
WRITE_SCHEMA = "mcss.rgbd_stream_write_rule.v1"


def stream_frame_ids(record, role, length=STREAM_LENGTH):
    """`length` frames evenly spaced over the scene's free frames (in no context/query role).

    Frame ids only: no image, depth, pose or model output is consulted; both roles share the
    stream (`role` only names the episode). None when fewer free frames exist (the episode is
    then ineligible, never shortened). Frames arrive in increasing frame-id order.
    """
    if role not in ("A", "B") or length < 2:
        raise ValueError("Roles A/B and a stream of at least two frames required")
    roles = record["roles"]
    used = set(roles["context_a"]) | set(roles["context_b"]) | set(roles["primary_query"])
    used |= set(roles.get("query", ())) | set(roles.get("secondary_query", ()))
    free = sorted(f["frame_id"] for f in record["frames"] if f["frame_id"] not in used)
    if len(free) < length:
        return None
    return tuple(free[round(i * (len(free) - 1) / (length - 1))] for i in range(length))


class StreamRGBDLoader:
    """Stream frames only (RGB, measured ray distance, camera); no query frame is reachable."""

    def __init__(
        self, record, image_size, base, device, access_log, *, allowed_scene_ids, holdout_scene_ids
    ):
        allowed, forbidden = set(allowed_scene_ids), set(holdout_scene_ids)
        if not allowed or allowed & forbidden:
            raise PermissionError("Stream allowlist must be nonempty and disjoint from holdout")
        if record["scene_id"] not in allowed or record["scene_id"] in forbidden:
            raise PermissionError("Stream scene outside the allowed roster")
        self._streams = {role: stream_frame_ids(record, role) for role in ("A", "B")}
        ids = {i for stream in self._streams.values() if stream for i in stream}
        roles = record["roles"]
        queries = set(roles["primary_query"]) | set(roles.get("query", ()))
        if ids & queries:
            raise PermissionError("A query frame can never enter a stream")
        private = deepcopy(record)
        private["frames"] = [f for f in private["frames"] if f["frame_id"] in ids]
        self._media = _Media(private, image_size, base, device, access_log)

    def frame_ids(self, role):
        return self._streams[role]

    def stream(self, role):
        ids = self._streams[role]
        if ids is None:
            raise PermissionError("No eligible stream for this role")
        batch = self._media.read(ids, depth_allowed=True, purpose="STREAM_RGBD")
        observations = [
            OnlineObservation(
                batch.scene_id,
                fid,
                batch.rgb[0, j],
                Cameras(
                    batch.cameras.intrinsics[0, j],
                    batch.cameras.c2w[0, j],
                    batch.cameras.image_size,
                ),
            )
            for j, fid in enumerate(batch.frame_ids)
        ]
        return v5.RGBDContext(observations, batch.depth[0, :, 0])


class StreamRGBDCarrier(v5.RGBDEvidenceCarrier):
    """The frozen V5/V7 RGB-D carrier; OFF_CLAMP3 clamps the lifting support-count statistic."""

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
        grid = torch.tensor(v5.GRID, device=self._bounds.device, dtype=self._bounds.dtype)
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
    """The sealed V7 C1 weights in a StreamRGBDCarrier; every parameter frozen."""
    model, payload = load_checkpoint(path, device)
    if not any(k.startswith("depth_") for k in model.state_dict()):
        raise PermissionError("Core B V1 runs on the RGB-D (depth-bypass) carrier only")
    carrier = StreamRGBDCarrier(model.config).to(device)
    carrier.load_state_dict(model.state_dict(), strict=True)
    if hash_value(carrier.state_dict()) != hash_value(model.state_dict()):
        raise RuntimeError("Frozen carrier transfer changed a tensor")
    carrier.eval()
    for parameter in carrier.parameters():
        parameter.requires_grad_(False)
    return carrier, payload


class _Call(nn.Module):
    def __init__(self, carrier):
        super().__init__()
        self.carrier = carrier

    def forward(self, cache, fast, mode):
        if mode == "trace":
            return self.carrier.trace(cache, fast)
        return self.carrier.materialize(cache, fast)


class _Episode:
    """One scene-role episode: bounds-replaced carrier calls over a growing RGB-D cache."""

    def __init__(self, carrier, bounds, episode_id, scene_id):
        bounds = torch.as_tensor(bounds, device=carrier._bounds.device, dtype=carrier._bounds.dtype)
        if bounds.shape not in ((2, 3), (1, 2, 3)) or bounds.requires_grad:
            raise ValueError("Frozen nonlearned bounds must have shape [2,3] or [1,2,3]")
        bounds = bounds.reshape(1, 2, 3).detach().clone()
        if not torch.isfinite(bounds).all() or not (bounds[:, 1] > bounds[:, 0]).all():
            raise ValueError("Finite ordered bounds required")
        fractions = (carrier._candidate_normalized_xyz + 1) * 0.5
        points = bounds[:, 0] + fractions * (bounds[:, 1] - bounds[:, 0])
        self.carrier, self.call = carrier, _Call(carrier)
        self.replace = {"carrier._bounds": bounds, "carrier._candidate_points": points}
        self.cache = ObservationCache(episode_id=episode_id, scene_id=scene_id)
        self.frames, self.source_frames, self.depths, self.anchor = [], [], [], None

    def arrive(self, observation, depth):
        """Cache one arrival; cache ids are arrival indices (the carrier uses order only)."""
        if self.anchor is None:
            self.anchor = observation.camera.c2w
        arrival = OnlineObservation(
            observation.scene_id, len(self.frames), observation.rgb, observation.camera
        )
        arrived = anchored_observation(arrival, self.anchor)
        self.cache = self.cache.append(arrived, self.carrier.encode(arrived))
        self.frames.append(arrival.frame_id)
        self.source_frames.append(observation.frame_id)
        self.depths.append(depth)

    def run(self, fast, mode):
        depths = torch.stack(self.depths).to(
            device=self.carrier._bounds.device, dtype=torch.float32
        )
        self.carrier._pending_depth = (tuple(self.frames), depths)
        try:
            return torch.func.functional_call(
                self.call, self.replace, (self.cache, fast, mode), strict=False
            )
        finally:
            self.carrier._pending_depth = None


def unroll(carrier, rule, warmup, stream, bounds, policy, episode_id):
    """(final state, anchor c2w, final fast weights, episode) of one fixed-policy episode."""
    if policy not in POLICIES:
        raise ValueError(f"Unregistered policy {policy}")
    if not isinstance(warmup, v5.RGBDContext) or not isinstance(stream, v5.RGBDContext):
        raise TypeError("Warmup and stream must be RGBDContext values")
    if not isinstance(carrier, StreamRGBDCarrier):
        raise TypeError("Core B V1 requires the StreamRGBDCarrier")
    episode = _Episode(carrier, bounds, episode_id, warmup[0].scene_id)
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


def rebuild_with(episode, fast):
    """The episode's full cache materialized with another episode's fast matrices."""
    moved = FastWeights(
        episode.cache.episode_id,
        fast.delta_fuse.detach().clone(),
        fast.delta_complete.detach().clone(),
        fast.step,
    )
    return episode.run(moved, "materialize")


def make_write_rule(carrier, seed):
    torch.manual_seed(seed)
    return DirectWriteRule(carrier.config, WriteConfig()).to(carrier._bounds.device)


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
        raise PermissionError("Core B V1 write-rule schema required")
    carrier_config = CarrierConfig(**payload["carrier_config"])
    rule = DirectWriteRule(carrier_config, WriteConfig(**payload["write_config"])).to(device)
    rule.load_state_dict(payload["rule"], strict=True)
    return rule, payload


def training_loss(state, local_cameras, rgb, depth, indices):
    """The frozen V2 C1 loss (the loss the V7 C1 carrier was trained with)."""
    return v5.training_loss(state, local_cameras, rgb, depth, indices, "C1")
