"""Exploratory TRAIN-only pilot: static RGB-D carrier trained with fixed 3 or variable 3-7 views.

Both arms: the V5 C1 carrier and loss with context-depth bounds (the V7 C1 rule), from the same
seeded initialization, TRAIN24 episodes, one role per step, 6000 steps. VARIABLE adds the first
m of the scene's 4 evenly spaced free frames to the context, m ~ U{0..4} (seeded). Scored on
the 48 TRAIN-X scenes with 3 views (NO_STREAM) and 7 views (OFF, OFF_CLAMP3). No DEV, FRESH
or protected scene is read. Usage: python viewcount_pilot.py SEED {FIXED3|VARIABLE} STEPS
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from mcss.geometry import transform_cameras
from mcss.measurements import FixedMeasurementRenderer
from mcss.mechanism_pilot import rgbd_evidence_carrier as v5
from mcss.mechanism_pilot.direct_capacity_evaluation import _metrics
from mcss.mechanism_pilot.rgbd_depth_bounds import DepthBoundsSceneData
from mcss.mechanism_pilot.rgbd_stream_write import (
    StreamRGBDCarrier,
    StreamRGBDLoader,
    make_write_rule,
    training_loss,
    unroll,
)

SEED, MODE, STEPS = int(sys.argv[1]), sys.argv[2], int(sys.argv[3])
torch.set_num_threads(1)
v7 = Path("/home/zonghan/measurement-scene-state/outputs/EXP-3D-RGBD-DEPTH-BOUNDS-V7")
manifest = json.loads((v7 / "scene_split.json").read_text())
sets = json.loads((v7 / "training_contract.json").read_text())["train_sets"]
access = []
train_ids = sorted(sets["TRAIN72"])
carrier = StreamRGBDCarrier(v5.carrier_config())
carrier.load_state_dict(v5.make_carrier("C1", SEED, "cpu").state_dict())
carrier.train()
rule = make_write_rule(carrier, SEED)  # never used for writes (OFF / NO_STREAM only)


def scenes(names):
    out = []
    for record in sorted(
        (r for r in manifest["scenes"] if r["scene_id"] in names), key=lambda r: r["scene_id"]
    ):
        data = DepthBoundsSceneData(record, manifest, v7, "cpu", access)
        stream = StreamRGBDLoader(
            record, manifest["image_size"], v7, "cpu", access,
            allowed_scene_ids=train_ids, holdout_scene_ids=[],
        )
        out.append((data, stream))
    return out


def extra_views(stream, m):
    return v5.RGBDContext(list(stream)[:m], stream.depths[:m])


train = scenes(set(sets["TRAIN24"]))
optimizer = torch.optim.Adam(carrier.parameters(), lr=1e-3, weight_decay=0)
generator = torch.Generator().manual_seed(SEED)
started, log = time.perf_counter(), []
for step in range(1, STEPS + 1):
    data, stream = train[(step - 1) % len(train)]
    role = "AB"[((step - 1) // len(train)) % 2]
    slot = ((step - 1) // (2 * len(train))) % 2
    m = int(torch.randint(0, 5, (1,), generator=generator)) if MODE == "VARIABLE" else 0
    warm = data.context(role)
    if m:
        state, anchor, _, _ = unroll(
            carrier, rule, warm, extra_views(stream.stream(role), m), data.bounds[role], "OFF", "t"
        )
    else:
        state, anchor, _, _ = unroll(carrier, rule, warm, warm, data.bounds[role], "NO_STREAM", "t")
    target = data.target(slot)
    cameras = transform_cameras(target.cameras, torch.linalg.inv(anchor))
    indices = torch.randint(128 * 160, (1024,), generator=generator)
    loss, terms = training_loss(state, cameras, target.rgb, target.depth, indices)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(carrier.parameters(), 1.0)
    optimizer.step()
    log.append(float(terms["loss_depth"].detach()))
    if step % 500 == 0:
        print(step, round(float(np.mean(log[-500:])), 4), round(time.perf_counter() - started), flush=True)

carrier.eval()
renderer = FixedMeasurementRenderer(n_samples=64, ray_chunk_size=2048)
names = ("NO_STREAM", "OFF", "OFF_CLAMP3")
scores = {n: {} for n in names}
with torch.no_grad():
    for data, stream in scenes(set(sets["TRAIN72"]) - set(sets["TRAIN24"])):
        sid = data.record["scene_id"]
        per = {n: [] for n in names}
        for role in ("A", "B"):
            warm, arrivals = data.context(role), stream.stream(role)
            states = {}
            for policy in names:
                states[policy], anchor, _, _ = unroll(
                    carrier, rule, warm, arrivals, data.bounds[role], policy, sid
                )
            for slot in range(2):
                target = data.target(slot)
                cameras = transform_cameras(target.cameras, torch.linalg.inv(anchor))
                for name, state in states.items():
                    pred = renderer(state, cameras, ("rgb", "depth", "visibility"))
                    per[name].append(_metrics(pred, target.rgb, target.depth)["depth_absrel"])
        for name in names:
            scores[name][sid] = float(np.mean(per[name]))
ids = sorted(scores["OFF"])


def paired(a, b):
    diff = np.array([scores[a][s] - scores[b][s] for s in ids])
    boot = diff[np.random.default_rng(0).integers(0, len(diff), (5000, len(diff)))].mean(1)
    return {"mean": float(diff.mean()), "ci95": [float(x) for x in np.percentile(boot, [2.5, 97.5])],
            "better": int((diff > 0).sum()), "n": len(diff)}


result = {
    "seed": SEED,
    "mode": MODE,
    "steps": STEPS,
    "absrel": {k: float(np.mean(list(v.values()))) for k, v in scores.items()},
    "more_views_gain_NO_STREAM_minus_OFF": paired("NO_STREAM", "OFF"),
    "more_views_clamped_NO_STREAM_minus_OFF_CLAMP3": paired("NO_STREAM", "OFF_CLAMP3"),
    "train_depth_last500": float(np.mean(log[-500:])),
    "seconds": time.perf_counter() - started,
    "exploratory_train_only": True,
}
Path(f"/tmp/claude_viewcount_pilot_{SEED}_{MODE}_{STEPS}.json").write_text(json.dumps(result, indent=1))
print(json.dumps(result), flush=True)
