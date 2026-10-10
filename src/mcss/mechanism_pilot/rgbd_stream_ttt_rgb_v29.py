"""Core B V29: stream frames with RGB and camera only (no depth), cached and written.

PLAN_BEFORE_V29.md fixed this round. The stream frames are read without opening their depth files
(purpose STREAM_RGB_ONLY); their depth maps are all zero, which the carrier's depth statistics and
the frozen V2 C1 loss both treat as invalid, so caching uses their RGB only and a stream write is
supervised by their RGB only. Policies are rows in Core B V7's format, unrolled by Core B V7's
`unroll_v7`; the evaluation is Core B V8's `evaluate_v8` with this loader in place of the RGB-D
stream loader for the duration of the call.
"""

from __future__ import annotations

import torch

from mcss.dynamic.types import OnlineObservation
from mcss.mechanism_pilot import rgbd_stream_ttt_v8 as v8
from mcss.mechanism_pilot.rgbd_evidence_carrier import RGBDContext
from mcss.mechanism_pilot.rgbd_stream_ttt_v4 import StreamRGBDLoader
from mcss.mechanism_pilot.rgbd_stream_ttt_v7 import LEARNING_RATE
from mcss.types import Cameras

EXPERIMENT = "EXP-3D-RGBD-DYNAMIC-TTT-RGB-ONLY-V29"
POLICY_SPECS = {
    "NO_STREAM": (False, None, 0),
    "CTX_TTT_100": (False, "context", 100),
    "OFF_RGB": (True, None, 0),
    "TTT_RGB_100": (True, "stream", 100),
}
CONTROLS = {"TTT_RGB_100_WRONG_SCENE": "TTT_RGB_100"}
PURPOSE = "STREAM_RGB_ONLY"
__all__ = [
    "CONTROLS",
    "EXPERIMENT",
    "LEARNING_RATE",
    "POLICY_SPECS",
    "PURPOSE",
    "RGBOnlyStreamLoader",
    "evaluate_v29",
]


class RGBOnlyStreamLoader(StreamRGBDLoader):
    """The Core B stream frames as RGB + camera; their depth files are never opened."""

    def stream(self, role):
        ids = self._streams[role]
        if ids is None:
            raise PermissionError("No eligible stream for this role")
        batch = self._media.read(ids, depth_allowed=False, purpose=PURPOSE)
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
        height, width = batch.cameras.image_size
        zeros = torch.zeros(len(observations), height, width, device=batch.rgb.device)
        return RGBDContext(observations, zeros)


def evaluate_v29(carrier, cohorts, expected_split, out, seed, device, specs=None, controls=None):
    """Core B V8's sealed evaluation with RGB-only stream frames."""
    specs = POLICY_SPECS if specs is None else specs
    controls = CONTROLS if controls is None else controls
    original = v8.StreamRGBDLoader
    v8.StreamRGBDLoader = RGBOnlyStreamLoader
    try:
        return v8.evaluate_v8(carrier, cohorts, expected_split, out, seed, device, specs, controls)
    finally:
        v8.StreamRGBDLoader = original
