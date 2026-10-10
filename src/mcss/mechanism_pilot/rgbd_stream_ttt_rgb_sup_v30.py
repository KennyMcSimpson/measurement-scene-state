"""Core B V30: RGB-only stream frames used only as write supervision, never cached.

PLAN_BEFORE_V30.md fixed this round. Every state is built from the three RGB-D context frames only.
The stream frames are read as RGB + camera only (V29's loader; their depth files are never opened)
and serve only as targets of the test-time write: alone (RGB_SUP_100, Core B V7's `unroll_v7`
with an uncached stream) or together with the context RGB-D (CTX_RGB_SUP_100, `unroll_v30`). The
evaluation is Core B V11's `evaluate_v11` with `unroll_v30` and V29's loader in place for the call.
"""

from __future__ import annotations

import torch

from mcss.mechanism_pilot import rgbd_stream_ttt_v11 as v11
from mcss.mechanism_pilot.rgbd_evidence_carrier import RGBDContext
from mcss.mechanism_pilot.rgbd_stream_ttt_rgb_v29 import PURPOSE, RGBOnlyStreamLoader
from mcss.mechanism_pilot.rgbd_stream_ttt_v4 import cached_episode, ttt_fast
from mcss.mechanism_pilot.rgbd_stream_ttt_v7 import LEARNING_RATE, generator_for, unroll_v7

EXPERIMENT = "EXP-3D-RGBD-DYNAMIC-TTT-RGB-SUPERVISION-V30"
POLICY_SPECS = {
    "NO_STREAM": (False, None, 0),
    "CTX_TTT_100": (False, "context", 100),
    "RGB_SUP_100": (False, "stream", 100),
    "CTX_RGB_SUP_100": (False, "all", 100),
}
CONTROLS = {"CTX_RGB_SUP_100_WRONG_SCENE": "CTX_RGB_SUP_100"}
__all__ = [
    "CONTROLS",
    "EXPERIMENT",
    "LEARNING_RATE",
    "POLICY_SPECS",
    "PURPOSE",
    "evaluate_v30",
    "unroll_v30",
]


def unroll_v30(carrier, warmup, stream, bounds, policy, episode_id, seed, specs=None):
    """(final state, anchor, fast weights, episode, TTT losses) of one fixed-policy episode."""
    specs = POLICY_SPECS if specs is None else specs
    if policy not in specs:
        raise ValueError(f"Unregistered policy {policy}")
    with_stream, supervision, steps = specs[policy]
    if supervision != "all":
        return unroll_v7(carrier, warmup, stream, bounds, policy, episode_id, seed, specs)
    episode = cached_episode(carrier, warmup, stream, bounds, episode_id, with_stream)
    frames = RGBDContext([*warmup, *stream], torch.cat((warmup.depths, stream.depths)))
    generator = generator_for(seed, episode_id, policy)
    fast, losses = ttt_fast(carrier, episode, frames, steps, LEARNING_RATE, generator)
    with torch.no_grad():
        state = episode.run(fast, "materialize")
    return state, episode.anchor.clone(), fast, episode, losses


def evaluate_v30(carrier, cohorts, expected_split, out, seed, device, specs=None, controls=None):
    """Core B V11's sealed evaluation with this round's unroll and RGB-only stream frames."""
    specs = POLICY_SPECS if specs is None else specs
    controls = CONTROLS if controls is None else controls
    unroll, loader = v11.unroll_v11, v11.StreamRGBDLoader
    v11.unroll_v11, v11.StreamRGBDLoader = unroll_v30, RGBOnlyStreamLoader
    try:
        return v11.evaluate_v11(
            carrier, cohorts, expected_split, out, seed, device, specs, controls
        )
    finally:
        v11.unroll_v11, v11.StreamRGBDLoader = unroll, loader
