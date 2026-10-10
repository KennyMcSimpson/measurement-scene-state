"""Core B V9: does stream TTT keep improving past 100 steps (300 steps), on TRAIN-EXT.

PLAN_BEFORE_COREB_V9.md fixed this round. The policy table has Core B V7's format; every row but
TTT_300 is Core B V7's, and TTT_300 is TTT_100 with 300 Adam steps. Policies are unrolled by Core
B V7's `unroll_v7` (every policy here is seeded by (seed, episode, policy)) and evaluated by Core B
V8's `evaluate_v8` with this round's policy table and control set.
"""

from __future__ import annotations

from mcss.mechanism_pilot.rgbd_stream_ttt_v7 import LEARNING_RATE
from mcss.mechanism_pilot.rgbd_stream_ttt_v8 import evaluate_v8

EXPERIMENT = "EXP-3D-RGBD-DYNAMIC-TTT-DOSE-V9"
POLICY_SPECS = {
    "NO_STREAM": (False, None, 0),
    "OFF": (True, None, 0),
    "OFF_CTX_TTT_100": (True, "context", 100),
    "TTT_100": (True, "stream", 100),
    "TTT_300": (True, "stream", 300),
}
CONTROLS = {"TTT_300_WRONG_SCENE": "TTT_300"}
__all__ = ["CONTROLS", "EXPERIMENT", "LEARNING_RATE", "POLICY_SPECS", "evaluate_v9"]


def evaluate_v9(carrier, cohorts, expected_split, out, seed, device, specs=None, controls=None):
    """Core B V8's sealed evaluation with this round's policy table and control set."""
    specs = POLICY_SPECS if specs is None else specs
    controls = CONTROLS if controls is None else controls
    return evaluate_v8(carrier, cohorts, expected_split, out, seed, device, specs, controls)
