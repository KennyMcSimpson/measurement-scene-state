"""Exploratory TRAIN-only diagnostic: non-learned RGB-D fusion floor by bounds rule x grid.

Fits the frozen NLF grid (k, a, b, c) on TRAIN24 primary queries, then scores the best fit on
the 48 TRAIN-X scenes (TRAIN72 minus TRAIN24, never used for the fit), next to the TRAIN-fit
AbsRel-optimal constant on the same views. No DEV, FRESH or protected scene is read, no
repository file is written. Usage: python nlf_resolution_diag.py {FROZEN|CONTEXT_DEPTH} GRID
"""

import itertools
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

from mcss.dynamic.feedback import anchored_observation
from mcss.geometry import transform_cameras
from mcss.measurements import FixedMeasurementRenderer
from mcss.mechanism_pilot.direct_capacity_evaluation import _metrics
from mcss.mechanism_pilot.rgbd_depth_bounds import DepthBoundsSceneData
from mcss.mechanism_pilot.rgbd_evidence_carrier import RGBDSceneData, depth_statistics
from mcss.mechanism_pilot.rgbd_nonlearned_fusion import GRID_SEARCH, _colors, voxel_centers
from mcss.types import SceneState

RULE, GRID = sys.argv[1], int(sys.argv[2])
CONSTANT = 2.3736  # REF_TRAIN_ABSREL_OPTIMAL (TRAIN24 primary-query fit, frozen in V5-V7)
torch.set_num_threads(1)
root = Path("outputs/EXP-3D-RGBD-DEPTH-BOUNDS-V7").resolve()
manifest = json.loads((root / "scene_split.json").read_text())
sets = json.loads((root / "training_contract.json").read_text())["train_sets"]
train24, train72 = set(sets["TRAIN24"]), set(sets["TRAIN72"])
cls = DepthBoundsSceneData if RULE == "CONTEXT_DEPTH" else RGBDSceneData
renderer = FixedMeasurementRenderer(n_samples=64, ray_chunk_size=2048)
access = []


def scenes(names):
    records = sorted(
        (r for r in manifest["scenes"] if r["scene_id"] in names and r["split"] == "TRAIN"),
        key=lambda r: r["scene_id"],
    )
    return [cls(r, manifest, root, "cpu", access) for r in records]


@torch.no_grad()
def statistics(context, bounds, k):
    anchor = context[0].camera.c2w
    observations = [anchored_observation(o, anchor) for o in context]
    bounds = torch.as_tensor(bounds, dtype=torch.float32).reshape(2, 3)
    points = voxel_centers(bounds, GRID)
    edge = ((bounds[1] - bounds[0]) / GRID).mean()
    stats = depth_statistics(observations, context.depths, points, torch.tensor(math.log(k)), edge)
    colors = _colors(observations, context.depths, points, k * edge)
    return stats, colors, bounds, anchor.clone()


def state(stats, colors, bounds, a, b, c):
    surface, free, _ = stats.unbind(-1)
    shape = (1, 1, GRID, GRID, GRID)
    return SceneState(
        density_logits=(a * surface - b * free + c).reshape(shape),
        color=colors.T.reshape(1, 3, GRID, GRID, GRID).contiguous(),
        log_variance=torch.full(shape, -3.0),
        bounds=bounds.reshape(1, 2, 3),
    )


def score(data, params):
    """{param index: [AbsRel per view]} plus constant AbsRel and opacity per view."""
    values = {i: [] for i in range(len(params))}
    opacity = {i: [] for i in range(len(params))}
    constant = []
    for scene in data:
        for role in ("A", "B"):
            cache = {}
            for slot in range(len(scene.record["roles"]["primary_query"])):
                target = scene.target(slot)
                depth = target.depth
                valid = torch.isfinite(depth) & (depth > 0)
                constant.append(float(((CONSTANT - depth).abs() / depth)[valid].mean()))
                for i, (k, a, b, c) in enumerate(params):
                    if k not in cache:
                        cache[k] = statistics(scene.context(role), scene.bounds[role], k)
                    stats, colors, bounds, anchor = cache[k]
                    cameras = transform_cameras(target.cameras, torch.linalg.inv(anchor))
                    with torch.no_grad():
                        pred = renderer(
                            state(stats, colors, bounds, a, b, c),
                            cameras,
                            ("rgb", "depth", "visibility"),
                        )
                    metrics = _metrics(pred, target.rgb, depth)
                    values[i].append(metrics["depth_absrel"])
                    opacity[i].append(metrics["opacity"])
    return values, opacity, constant


started = time.perf_counter()
grid = list(itertools.product(*(GRID_SEARCH[k] for k in ("k", "a", "b", "c"))))
fit, _, fit_constant = score(scenes(train24), grid)
ranked = sorted(range(len(grid)), key=lambda i: np.mean(fit[i]))
best = ranked[0]
held, held_opacity, held_constant = score(scenes(train72 - train24), [grid[best]])
result = {
    "rule": RULE,
    "grid": GRID,
    "best_params": dict(zip(("k", "a", "b", "c"), grid[best], strict=True)),
    "train24_fit_absrel": float(np.mean(fit[best])),
    "train24_constant_absrel": float(np.mean(fit_constant)),
    "trainx_absrel": float(np.mean(held[0])),
    "trainx_constant_absrel": float(np.mean(held_constant)),
    "trainx_opacity": float(np.mean(held_opacity[0])),
    "trainx_views": len(held[0]),
    "trainx_views_beating_constant": float(
        np.mean(np.array(held[0]) < np.array(held_constant))
    ),
    "top3_fit": [
        {**dict(zip(("k", "a", "b", "c"), grid[i], strict=True)), "absrel": float(np.mean(fit[i]))}
        for i in ranked[:3]
    ],
    "seconds": time.perf_counter() - started,
    "exploratory_train_only": True,
}
out = Path(f"/tmp/claude_nlf_resolution_{RULE}_{GRID}.json")
out.write_text(json.dumps(result, indent=1))
print(json.dumps(result), flush=True)
