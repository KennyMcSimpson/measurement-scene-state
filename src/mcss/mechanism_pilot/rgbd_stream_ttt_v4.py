"""Core B V4: gradient-based test-time writes (TTT) on the frozen V17 L step-24000 carrier.

PLAN_BEFORE_COREB_V4.md fixed this round. After the stream frames are cached (the OFF state), the
two fast matrices (delta_fuse, delta_complete) start at zero and take K Adam steps on the frozen V2
C1 loss of the stream frames' own RGB and measured depth, which are test-time inputs; every step
samples 256 rays per stream frame from a generator seeded by (seed, episode). The carrier, the
stream rule and the episode mechanics are Core B V3's; nothing of the carrier is updated.
"""

from __future__ import annotations

import hashlib

import torch

from mcss.dynamic.types import FastWeights
from mcss.geometry import transform_cameras
from mcss.mechanism_pilot import rgbd_stream_write as v1
from mcss.mechanism_pilot import rgbd_stream_write_v3 as v3
from mcss.types import Cameras

EXPERIMENT = "EXP-3D-RGBD-DYNAMIC-TTT-V4"
POLICIES = ("NO_STREAM", "OFF", "TTT")
CONTROLS = ("TTT_WRONG_SCENE",)
GRID = (
    {"steps": 10, "learning_rate": 1e-3},
    {"steps": 10, "learning_rate": 1e-2},
    {"steps": 50, "learning_rate": 1e-3},
    {"steps": 50, "learning_rate": 1e-2},
)
RAYS_PER_FRAME = 256
STREAM_LENGTH = v3.STREAM_LENGTH
StreamRGBDLoader = v3.StreamRGBDLoader
StreamResolutionCarrier = v3.StreamResolutionCarrier
load_frozen_carrier = v3.load_frozen_carrier
rebuild_with = v3.rebuild_with
stream_frame_ids = v3.stream_frame_ids
training_loss = v3.training_loss


def episode_generator(seed, episode_id):
    """The CPU generator of one episode's TTT rays (depends on the seed and the episode only)."""
    digest = hashlib.sha256(f"{seed}:{episode_id}".encode()).digest()
    return torch.Generator(device="cpu").manual_seed(int.from_bytes(digest[:8], "little"))


def cached_episode(carrier, warmup, stream, bounds, episode_id, with_stream):
    """The Core B episode with the warmup (and, unless NO_STREAM, the stream) cached."""
    if not isinstance(carrier, StreamResolutionCarrier):
        raise TypeError("Core B V4 requires the frozen StreamResolutionCarrier")
    episode = v1._Episode(carrier, bounds, episode_id, warmup[0].scene_id)
    for observation, depth in zip(warmup, warmup.depths, strict=True):
        episode.arrive(observation, depth)
    if with_stream:
        for observation, depth in zip(stream, stream.depths, strict=True):
            episode.arrive(observation, depth)
    return episode


def stream_targets(stream, anchor):
    """The stream frames as one [1, V] target batch in the episode's anchor frame."""
    observations = list(stream)
    intrinsics = torch.stack([o.camera.intrinsics for o in observations])[None]
    c2w = torch.stack([o.camera.c2w for o in observations])[None]
    cameras = Cameras(intrinsics, c2w, observations[0].camera.image_size)
    local = transform_cameras(cameras, torch.linalg.inv(anchor))
    rgb = torch.stack([o.rgb for o in observations])[None]
    depth = stream.depths[None, :, None]
    return local, rgb, depth


def ttt_fast(carrier, episode, stream, steps, learning_rate, generator):
    """K Adam steps on the zero-initialized fast matrices against the stream frames' RGB-D."""
    device = carrier._bounds.device
    local, rgb, depth = stream_targets(stream, episode.anchor)
    local = Cameras(local.intrinsics.to(device), local.c2w.to(device), local.image_size)
    rgb, depth = rgb.to(device), depth.to(device, torch.float32)
    views = rgb.shape[1]
    height, width = local.image_size
    zero = carrier.initial_fast(episode.cache.episode_id)
    delta_fuse = zero.delta_fuse.clone().requires_grad_(True)
    delta_complete = zero.delta_complete.clone().requires_grad_(True)
    optimizer = torch.optim.Adam([delta_fuse, delta_complete], lr=learning_rate)
    losses = []
    with torch.enable_grad():
        for _ in range(steps):
            fast = FastWeights(episode.cache.episode_id, delta_fuse, delta_complete)
            state = episode.run(fast, "materialize")
            offsets = torch.arange(views).repeat_interleave(RAYS_PER_FRAME) * (height * width)
            rays = torch.randint(height * width, (views * RAYS_PER_FRAME,), generator=generator)
            loss, _ = training_loss(state, local, rgb, depth, (offsets + rays).to(device))
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite test-time objective")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
    final = FastWeights(
        episode.cache.episode_id,
        delta_fuse.detach().clone(),
        delta_complete.detach().clone(),
        steps,
    )
    return final, losses


def unroll_ttt(carrier, warmup, stream, bounds, policy, episode_id, settings, generator):
    """(final state, anchor, fast weights, episode, TTT losses) of one fixed-policy episode."""
    if policy not in POLICIES:
        raise ValueError(f"Unregistered policy {policy}")
    episode = cached_episode(carrier, warmup, stream, bounds, episode_id, policy != "NO_STREAM")
    if policy == "TTT":
        fast, losses = ttt_fast(
            carrier, episode, stream, settings["steps"], settings["learning_rate"], generator
        )
    else:
        fast, losses = carrier.initial_fast(episode_id), []
    with torch.no_grad():
        state = episode.run(fast, "materialize")
    return state, episode.anchor.clone(), fast, episode, losses
