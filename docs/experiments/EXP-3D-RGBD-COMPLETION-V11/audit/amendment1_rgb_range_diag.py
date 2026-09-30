"""Engineering diagnostic of the V11 C1 seed 20260928 failure: range of rendered context RGB.

Loads the step-5000 checkpoint written just before the crash, builds the DEV A/B states from
context frames only (the frozen V11 build_state and CONTEXT_DEPTH bounds) and renders the context
cameras, as the frozen DEV-curve evaluation does before it computes metrics. Reports only the
range of the predicted RGB (min, max, number of values outside [0, 1]); no metric, no query frame
and no ground truth is computed or read.
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

from mcss.geometry import transform_cameras
from mcss.measurements import FixedMeasurementRenderer
from mcss.mechanism_pilot.rgbd_completion_carrier import build_state, load_checkpoint, scene_data

root = Path("outputs/EXP-3D-RGBD-COMPLETION-V11")
checkpoint = root / "checkpoints" / sys.argv[1] / f"step_{int(sys.argv[2]):06d}.pt"
torch.set_num_threads(1)
manifest = json.loads((root / "scene_split.json").read_text())
model, payload = load_checkpoint(checkpoint, "cpu")
model.eval()
renderer = FixedMeasurementRenderer(n_samples=64, ray_chunk_size=2048)
access, report = [], []
with torch.no_grad():
    for record in (r for r in manifest["scenes"] if r["split"] == "DEV"):
        data = scene_data("C1")(record, manifest, root, "cpu", access)
        for role in ("A", "B"):
            context = data.context(role)
            state, anchor = build_state(model, context, data.bounds[role], f"diag:{role}")
            inverse = torch.linalg.inv(anchor)
            for observation in context:
                camera = observation.camera
                batched = type(camera)(camera.intrinsics[None, None], camera.c2w[None, None], camera.image_size)
                rgb = renderer(state, transform_cameras(batched, inverse), ("rgb",))["rgb"].numpy()
                report.append(
                    {
                        "scene": record["scene_id"],
                        "role": role,
                        "frame": observation.frame_id,
                        "min": float(rgb.min()),
                        "max": float(rgb.max()),
                        "above_1": int((rgb > 1).sum()),
                        "below_0": int((rgb < 0).sum()),
                        "max_overshoot": float(max(rgb.max() - 1.0, -rgb.min(), 0.0)),
                    }
                )
bad = [r for r in report if r["above_1"] or r["below_0"]]
print(json.dumps({"checkpoint": str(checkpoint), "views": len(report), "views_out_of_range": len(bad),
                  "worst": max(report, key=lambda r: r["max_overshoot"]), "out_of_range": bad[:6]}, indent=1))
print("depth channels read:", sorted({e.get("purpose") for e in access if "depth" in e.get("channels", [])}))
