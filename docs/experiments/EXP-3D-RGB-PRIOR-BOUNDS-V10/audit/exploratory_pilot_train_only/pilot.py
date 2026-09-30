"""Exploratory TRAIN-only pilot for V10: RGB-only 32^3 carrier, frozen V2 prior vs TRAIN24 prior.

Trains on TRAIN24 only and scores the 48 held-out TRAIN-X scenes (the TRAIN72 scenes outside
TRAIN24). The pilot's own prior is fitted on TRAIN24 frames only, so TRAIN-X depth never sets
a box. No DEV, FRESH or protected scene is read. Usage: pilot.py VARIANT STEPS OUT_JSON, with
VARIANT in {FROZEN, TRAIN24}.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from mcss.geometry import transform_cameras
from mcss.measurements import FixedMeasurementRenderer
from mcss.mechanism_pilot.geometry_carrier_experiment import SceneData
from mcss.mechanism_pilot.optimization_bounds_contracts import (
    FrozenTrainingPrior,
    context_camera_bundle,
    frozen_gt_free_bounds,
)
from mcss.mechanism_pilot.readout_reanalysis import depth_metrics
from mcss.mechanism_pilot.rgb_prior_bounds_carrier import build_state, make_carrier, training_loss

REPO = Path.home() / "measurement-scene-state"
V2 = REPO / "outputs/EXP-3D-GEOMETRY-AWARE-CARRIER-V2"
V5 = REPO / "outputs/EXP-3D-RGBD-EVIDENCE-CARRIER-V5"
V6 = REPO / "outputs/EXP-3D-RGBD-TRAIN-SCALE-V6"
variant, steps, out = sys.argv[1], int(sys.argv[2]), Path(sys.argv[3])
torch.set_num_threads(1)
seed = 20260928
manifest = json.loads((V6 / "scene_split.json").read_text())
config = json.loads((V6 / "training_contract.json").read_text())
train24 = set(config["train_sets"]["TRAIN24"])
train = sorted((r for r in manifest["scenes"] if r["split"] == "TRAIN"), key=lambda r: r["scene_id"])
fit = [r for r in train if r["scene_id"] in train24]
held = [r for r in train if r["scene_id"] not in train24]
assert len(fit) == 24 and len(held) == 48
values = np.concatenate(
    [np.load(f["depth"]).reshape(-1) for r in fit for f in sorted(r["frames"], key=lambda f: f["frame_id"])]
)
values = values[np.isfinite(values) & (values > 0)]
t24 = [float(v) for v in np.quantile(values, [0.01, 0.99])]
frozen = json.loads((V2 / "train_depth_prior.json").read_text())
near, far = (frozen["near_m"], frozen["far_m"]) if variant == "FROZEN" else t24
prior = FrozenTrainingPrior(near, far, "0" * 64)


def loader(record, access):
    data = SceneData(record, manifest, V6, "cpu", access)
    data.bounds = {
        role: frozen_gt_free_bounds(
            context_camera_bundle(record, role, manifest["image_size"]), prior
        ).to(torch.float32)
        for role in ("A", "B")
    }
    return data


access = []
scenes = [loader(r, access) for r in fit]
generator = torch.Generator(device="cpu").manual_seed(seed)
order = torch.randperm(len(scenes), generator=generator).tolist()
carrier = make_carrier("C1", seed, "cpu")  # 32^3 RGB-only dense carrier
carrier.train()
optimizer = torch.optim.Adam(carrier.parameters(), lr=config["learning_rate"], weight_decay=0)
start = time.perf_counter()
losses = []
for step in range(1, steps + 1):
    visit = (step - 1) // len(scenes)
    scene = scenes[order[(step - 1) % len(scenes)]]
    slot = visit % len(scene.record["roles"]["primary_query"])
    optimizer.zero_grad(set_to_none=True)
    states = {
        role: build_state(carrier, scene.context(role), scene.bounds[role], f"{seed}:{step}:{role}")
        for role in ("A", "B")
    }
    target = scene.target(slot)
    indices = torch.randint(
        int(np.prod(manifest["image_size"])), (config["batch_rays"],), generator=generator
    )
    total = 0
    for role, (state, anchor) in states.items():
        cameras = transform_cameras(target.cameras, torch.linalg.inv(anchor))
        loss, _ = training_loss(state, cameras, target.rgb, target.depth, indices, "C1")
        total = total + loss * 0.5
    total.backward()
    torch.nn.utils.clip_grad_norm_(carrier.parameters(), config["gradient_clip"], error_if_nonfinite=True)
    optimizer.step()
    losses.append(float(total.detach()))
    if step % 250 == 0:
        print(variant, step, round(float(np.mean(losses[-250:])), 4), round(time.perf_counter() - start), flush=True)
carrier.eval()
renderer = FixedMeasurementRenderer(n_samples=64, ray_chunk_size=2048)
constant = json.loads((V6 / "decision_rules.json").read_text())["GEOMETRY_FREE_REFERENCE"]["values"]
per_scene, per_scene_constant, hits = {}, {}, {}
with torch.no_grad():
    for record in held:
        data = loader(record, access)
        rows, crow, hrow = [], [], []
        for role in ("A", "B"):
            state, anchor = build_state(carrier, data.context(role), data.bounds[role], f"pilot:{role}")
            for slot in range(len(record["roles"]["primary_query"])):
                target = data.target(slot)
                camera = transform_cameras(target.cameras, torch.linalg.inv(anchor))
                pred = renderer(state, camera, ("depth", "visibility"))
                gt = target.depth[0, 0, 0].numpy().astype(np.float64)
                rows.append(depth_metrics(pred["depth"][0, 0, 0].numpy().astype(np.float64), gt)["depth_absrel"])
                crow.append(depth_metrics(np.full_like(gt, constant["REF_TRAIN_ABSREL_OPTIMAL"]), gt)["depth_absrel"])
                hrow.append(float((pred["visibility"][0, 0, 0] > 1e-6).float().mean()))
        per_scene[record["scene_id"]] = float(np.mean(rows))
        per_scene_constant[record["scene_id"]] = float(np.mean(crow))
        hits[record["scene_id"]] = float(np.mean(hrow))
result = {
    "exploratory": True,
    "train_only": True,
    "variant": variant,
    "prior": {"near_m": near, "far_m": far, "fit": "frozen V2 prior" if variant == "FROZEN" else "TRAIN24 q01/q99"},
    "grid": 32,
    "steps": steps,
    "seed": seed,
    "train_scenes": len(fit),
    "heldout_scenes": len(held),
    "heldout_mean_absrel": float(np.mean(list(per_scene.values()))),
    "heldout_constant_absrel": float(np.mean(list(per_scene_constant.values()))),
    "heldout_scenes_better_than_constant": int(sum(per_scene[s] < per_scene_constant[s] for s in per_scene)),
    "mean_rendered_coverage": float(np.mean(list(hits.values()))),
    "per_scene_absrel": per_scene,
    "train_loss_last250": float(np.mean(losses[-250:])),
    "depth_reads_outside_training_targets": sum(
        1 for e in access if isinstance(e, dict) and "depth" in e.get("channels", []) and "TARGET" not in str(e.get("purpose", "")).upper()
    ),
    "seconds": time.perf_counter() - start,
}
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(result, indent=1))
print(json.dumps({k: v for k, v in result.items() if k != "per_scene_absrel"}), flush=True)
