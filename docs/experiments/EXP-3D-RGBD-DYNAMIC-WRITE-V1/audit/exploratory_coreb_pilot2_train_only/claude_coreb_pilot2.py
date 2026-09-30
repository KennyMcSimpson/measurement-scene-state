"""Exploratory TRAIN-only Core B pilot 2, run through the Core B V1 module (new stream rule).

Write rule trained on TRAIN24 episodes (ALL), scored on the 48 TRAIN-X scenes for every policy.
No DEV, FRESH or protected scene is read. Usage: python coreb_pilot2.py SEED STEPS
"""

import json
import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

from mcss.geometry import transform_cameras
from mcss.measurements import FixedMeasurementRenderer
from mcss.mechanism_pilot.direct_capacity_evaluation import _metrics
from mcss.mechanism_pilot.rgbd_depth_bounds import DepthBoundsSceneData
from mcss.mechanism_pilot.rgbd_stream_write import (
    POLICIES,
    StreamRGBDLoader,
    load_frozen_carrier,
    make_write_rule,
    rebuild_with,
    training_loss,
    unroll,
)

SEED, STEPS = int(sys.argv[1]), int(sys.argv[2])
torch.set_num_threads(1)
v7 = Path("/home/zonghan/measurement-scene-state/outputs/EXP-3D-RGBD-DEPTH-BOUNDS-V7")
manifest = json.loads((v7 / "scene_split.json").read_text())
sets = json.loads((v7 / "training_contract.json").read_text())["train_sets"]
selected = json.loads((v7 / "selected_checkpoints.json").read_text())
carrier, _ = load_frozen_carrier(selected["C1"][str(SEED)]["path"], "cpu")
access = []
train_ids = sorted(sets["TRAIN72"])


def scenes(names):
    out = []
    for record in sorted(
        (r for r in manifest["scenes"] if r["scene_id"] in names), key=lambda r: r["scene_id"]
    ):
        data = DepthBoundsSceneData(record, manifest, v7, "cpu", access)
        stream = StreamRGBDLoader(
            record,
            manifest["image_size"],
            v7,
            "cpu",
            access,
            allowed_scene_ids=train_ids,
            holdout_scene_ids=[],
        )
        out.append((data, stream))
    return out


rule = make_write_rule(carrier, SEED)
initial = deepcopy(rule)
optimizer = torch.optim.Adam(rule.parameters(), lr=1e-3)
train = scenes(set(sets["TRAIN24"]))
generator = torch.Generator().manual_seed(SEED)
started, log = time.perf_counter(), []
for step in range(1, STEPS + 1):
    data, stream = train[(step - 1) % len(train)]
    role = "AB"[((step - 1) // len(train)) % 2]
    slot = ((step - 1) // (2 * len(train))) % 2
    state, anchor, _, _ = unroll(
        carrier, rule, data.context(role), stream.stream(role), data.bounds[role], "ALL", f"t{step}"
    )
    target = data.target(slot)
    cameras = transform_cameras(target.cameras, torch.linalg.inv(anchor))
    indices = torch.randint(128 * 160, (1024,), generator=generator)
    loss, terms = training_loss(state, cameras, target.rgb, target.depth, indices)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(rule.parameters(), 1.0)
    optimizer.step()
    log.append(float(terms["loss_depth"].detach()))
    if step % 250 == 0:
        print(step, round(float(np.mean(log[-250:])), 4), round(time.perf_counter() - started), flush=True)

renderer = FixedMeasurementRenderer(n_samples=64, ray_chunk_size=2048)
names = (*POLICIES, "ALL_WRONG_SCENE")
scores = {n: {} for n in names}
held = scenes(set(sets["TRAIN72"]) - set(sets["TRAIN24"]))
with torch.no_grad():
    episodes = {}
    for data, stream in held:
        sid = data.record["scene_id"]
        for role in ("A", "B"):
            warm, arrivals = data.context(role), stream.stream(role)
            states = {}
            for policy in POLICIES:
                rule_used = initial if policy == "ALL_UNTRAINED" else rule
                state, anchor, fast, episode = unroll(
                    carrier, rule_used, warm, arrivals, data.bounds[role], policy, sid
                )
                states[policy] = state
                if policy == "ALL":
                    episodes[sid, role] = (episode, fast)
            episodes[sid, role, "states"] = (states, anchor)
    ids = sorted({k[0] for k in episodes if len(k) == 2})
    for data, _ in held:
        sid = data.record["scene_id"]
        per = {n: [] for n in names}
        for role in ("A", "B"):
            states, anchor = episodes[sid, role, "states"]
            donor = ids[(ids.index(sid) + 1) % len(ids)]
            states["ALL_WRONG_SCENE"] = rebuild_with(episodes[sid, role][0], episodes[donor, role][1])
            for slot in range(2):
                target = data.target(slot)
                cameras = transform_cameras(target.cameras, torch.linalg.inv(anchor))
                for name, state in states.items():
                    pred = renderer(state, cameras, ("rgb", "depth", "visibility"))
                    per[name].append(_metrics(pred, target.rgb, target.depth)["depth_absrel"])
        for name in names:
            scores[name][sid] = float(np.mean(per[name]))


def paired(a, b):
    diff = np.array([scores[a][s] - scores[b][s] for s in ids])
    boot = diff[np.random.default_rng(0).integers(0, len(diff), (5000, len(diff)))].mean(1)
    return {
        "mean": float(diff.mean()),
        "ci95": [float(x) for x in np.percentile(boot, [2.5, 97.5])],
        "better": int((diff > 0).sum()),
        "n": len(diff),
    }


result = {
    "seed": SEED,
    "steps": STEPS,
    "absrel": {k: float(np.mean(list(v.values()))) for k, v in scores.items()},
    "WRITE_GAIN_OFF_minus_ALL": paired("OFF", "ALL"),
    "STREAM_WRITE_GAIN_NO_STREAM_minus_ALL": paired("NO_STREAM", "ALL"),
    "BEYOND_CLAMP_OFF_CLAMP3_minus_ALL": paired("OFF_CLAMP3", "ALL"),
    "SPECIFICITY_WRONG_minus_ALL": paired("ALL_WRONG_SCENE", "ALL"),
    "STREAM_CACHE_NO_STREAM_minus_OFF": paired("NO_STREAM", "OFF"),
    "CLAMP_VS_STATIC_NO_STREAM_minus_OFF_CLAMP3": paired("NO_STREAM", "OFF_CLAMP3"),
    "train_depth_first250_last250": [float(np.mean(log[:250])), float(np.mean(log[-250:]))],
    "seconds": time.perf_counter() - started,
    "exploratory_train_only": True,
}
Path(f"/tmp/claude_coreb_pilot2_{SEED}_{STEPS}.json").write_text(json.dumps(result, indent=1))
print(json.dumps(result), flush=True)
