"""Engineering-only CPU timing of one V7-recipe training step at 16^3 / 24^3 / 32^3 (TRAIN only).

In-process patches of three frozen 16-grid guards; no repository file is modified, no DEV read.
"""

import inspect
import json
import time
from pathlib import Path

import torch

from mcss.dynamic.config import CarrierConfig
from mcss.geometry import transform_cameras
from mcss.mechanism_pilot import dense_evidence_carrier as dense
from mcss.mechanism_pilot import geometry_carrier as gc
from mcss.mechanism_pilot import rgbd_evidence_carrier as v5
from mcss.mechanism_pilot.rgbd_depth_bounds import DepthBoundsSceneData
from mcss.mechanism_pilot.small_training import AccessLog

torch.set_num_threads(1)
source = inspect.getsource(gc.training_loss).replace(
    "tuple(state.spatial_shape) != (16, 16, 16)", "False"
)
namespace = dict(vars(gc))
exec(source, namespace)
loss_any_grid = namespace["training_loss"]

root = Path("outputs/EXP-3D-RGBD-DEPTH-BOUNDS-V7").resolve()
manifest = json.loads((root / "scene_split.json").read_text())
records = [r for r in manifest["scenes"] if r["split"] == "TRAIN"][:2]
access = AccessLog(Path("/tmp/claude_v8_bench_access.jsonl"))
scenes = [DepthBoundsSceneData(r, manifest, root, "cpu", access) for r in records]
generator = torch.Generator().manual_seed(0)
for g in (16, 24, 32):
    v5.GRID = (g, g, g)
    dense.GRID = (g, g, g)
    dense.TOKEN_COUNTS = (128, 4096, g**3)
    torch.manual_seed(20260928)
    carrier = v5.RGBDEvidenceCarrier(CarrierConfig(grid_size=(g, g, g), token_count=g**3))
    optimizer = torch.optim.Adam(carrier.parameters(), lr=1e-3)
    times = []
    for step in range(6):
        scene = scenes[step % 2]
        tick = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        states = {
            role: v5.build_state(carrier, scene.context(role), scene.bounds[role], f"b:{step}:{role}")
            for role in ("A", "B")
        }
        built = time.perf_counter() - tick
        target = scene.target(0)
        indices = torch.randint(128 * 160, (1024,), generator=generator)
        total = 0
        for state, anchor in states.values():
            cameras = transform_cameras(target.cameras, torch.linalg.inv(anchor))
            loss, _ = loss_any_grid(state, cameras, target.rgb, target.depth, indices, "C1")
            total = total + 0.5 * loss
        total.backward()
        optimizer.step()
        times.append((time.perf_counter() - tick, built))
    steady = times[1:]
    print(
        g,
        "step s %.3f" % (sum(t for t, _ in steady) / len(steady)),
        "build s %.3f" % (sum(b for _, b in steady) / len(steady)),
        "loss %.3f" % float(total),
        flush=True,
    )
