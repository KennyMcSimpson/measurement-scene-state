"""Core B V10: does the 100-step stream-TTT gain hold on other carriers (width x training data)?

PLAN_BEFORE_COREB_V10.md fixed this round. Four frozen step-24000 carriers (V17 L, V18 L, V19 L,
V20 L) are each read by their own experiment's `load_checkpoint` and moved into Core B's
StreamResolutionCarrier (as Core B V3's `load_frozen_carrier` does for V17 L). The policies are
Core B V7's NO_STREAM, OFF and TTT_100, unrolled by Core B V7's `unroll_v7` and evaluated by Core
B V8's `evaluate_v8` with this round's policy table and control set.
"""

from __future__ import annotations

from mcss.dynamic.types import hash_value
from mcss.mechanism_pilot import rgbd_long_data_carrier as v18
from mcss.mechanism_pilot import rgbd_long_scale_carrier as v20
from mcss.mechanism_pilot import rgbd_long_train_carrier as v17
from mcss.mechanism_pilot import rgbd_long_width_carrier as v19
from mcss.mechanism_pilot.rgbd_stream_ttt_v7 import LEARNING_RATE
from mcss.mechanism_pilot.rgbd_stream_ttt_v8 import evaluate_v8
from mcss.mechanism_pilot.rgbd_stream_write_v3 import GRID, StreamResolutionCarrier

EXPERIMENT = "EXP-3D-RGBD-DYNAMIC-TTT-CARRIER-V10"
CARRIER_STEP = 24000
# variant: (experiment, checkpoint module, hidden width, training set)
CARRIERS = {
    "V17_L": ("EXP-3D-RGBD-LONG-TRAIN-V17", v17, 32, "TRAIN72"),
    "V18_L": ("EXP-3D-RGBD-LONG-TRAIN-DATA-V18", v18, 32, "TRAIN127"),
    "V19_L": ("EXP-3D-RGBD-LONG-TRAIN-WIDTH-V19", v19, 64, "TRAIN72"),
    "V20_L": ("EXP-3D-RGBD-LONG-TRAIN-WIDTH-DATA-V20", v20, 64, "TRAIN127"),
}
POLICY_SPECS = {
    "NO_STREAM": (False, None, 0),
    "OFF": (True, None, 0),
    "TTT_100": (True, "stream", 100),
}
CONTROLS = {"TTT_100_WRONG_SCENE": "TTT_100"}
__all__ = [
    "CARRIERS",
    "CARRIER_STEP",
    "CONTROLS",
    "EXPERIMENT",
    "LEARNING_RATE",
    "POLICY_SPECS",
    "evaluate_v10",
    "load_stream_carrier",
]


def load_stream_carrier(variant, path, device):
    """One registered step-24000 checkpoint in a frozen StreamResolutionCarrier."""
    _, source, hidden, _ = CARRIERS[variant]
    model, payload = source.load_checkpoint(path, device)
    if tuple(model.config.grid_size) != (GRID, GRID, GRID) or payload["step"] != CARRIER_STEP:
        raise PermissionError("Core B V10 runs on 32^3 step-24000 carriers only")
    if model.config.hidden_dim != hidden:
        raise PermissionError(f"{variant} must have hidden width {hidden}")
    carrier = StreamResolutionCarrier(model.config).to(device)
    carrier.load_state_dict(model.state_dict(), strict=True)
    if hash_value(carrier.state_dict()) != hash_value(model.state_dict()):
        raise RuntimeError("Frozen carrier transfer changed a tensor")
    carrier.eval()
    for parameter in carrier.parameters():
        parameter.requires_grad_(False)
    return carrier, payload


def evaluate_v10(carrier, cohorts, expected_split, out, seed, device, specs=None, controls=None):
    """Core B V8's sealed evaluation with this round's policy table and control set."""
    specs = POLICY_SPECS if specs is None else specs
    controls = CONTROLS if controls is None else controls
    return evaluate_v8(carrier, cohorts, expected_split, out, seed, device, specs, controls)
