"""Exploratory TRAIN-only Core B pilot: learned fast-weight writes on the frozen V7 C1 carrier.

Episode: warmup = context_a (or context_b) RGB-D, then K stream RGB-D frames (camera-only
rule: the first K frames after the warmup that belong to no context or query role). The
DirectWriteRule (frozen project design, WriteConfig defaults) is trained on TRAIN24 with
ALL at every stream step and the frozen V2 C1 loss on a TRAIN primary query after the
stream; the slow carrier stays frozen. Scored on the 48 TRAIN-X scenes (never used for the
write rule). No DEV, FRESH or protected scene is read; nothing in the repository is written.
Usage: python coreb_pilot.py SEED STEPS
"""

import json
import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

from mcss.dynamic.cache import ObservationCache
from mcss.dynamic.config import WriteConfig
from mcss.dynamic.feedback import anchored_observation
from mcss.dynamic.types import Action, OnlineObservation
from mcss.dynamic.write_rule import DirectWriteRule, commit_write
from mcss.geometry import transform_cameras
from mcss.measurements import FixedMeasurementRenderer
from mcss.mechanism_pilot.direct_capacity_contracts import _Media
from mcss.mechanism_pilot.direct_capacity_evaluation import _metrics
from mcss.mechanism_pilot.rgbd_bounds_carrier import load_checkpoint, training_loss
from mcss.mechanism_pilot.rgbd_depth_bounds import DepthBoundsSceneData
from mcss.mechanism_pilot.rgbd_evidence_carrier import RGBDContext
from mcss.types import Cameras

SEED = int(sys.argv[1])
STEPS = int(sys.argv[2])
K = 4
torch.set_num_threads(1)
v7 = Path("outputs/EXP-3D-RGBD-DEPTH-BOUNDS-V7").resolve()
manifest = json.loads((v7 / "scene_split.json").read_text())
sets = json.loads((v7 / "training_contract.json").read_text())["train_sets"]
selected = json.loads((v7 / "selected_checkpoints.json").read_text())
carrier, _ = load_checkpoint(selected["C1"][str(SEED)]["path"], "cpu")
carrier.eval()
for parameter in carrier.parameters():
    parameter.requires_grad_(False)
access = []


def stream_ids(record, role):
    roles = record["roles"]
    warm = roles["context_a" if role == "A" else "context_b"]
    used = set(roles["context_a"]) | set(roles["context_b"]) | set(roles["primary_query"])
    used |= set(roles.get("query", []))
    ids = sorted(
        f["frame_id"] for f in record["frames"] if f["frame_id"] > max(warm) and f["frame_id"] not in used
    )
    return ids[:K]


def rgbd(record, ids):
    """Stream RGB-D frames only (TRAIN scenes in this pilot), read by frame id."""
    private = deepcopy(record)
    private["frames"] = [f for f in private["frames"] if f["frame_id"] in set(ids)]
    media = _Media(private, manifest["image_size"], v7, "cpu", access)
    batch = media.read(tuple(ids), depth_allowed=True, purpose="STREAM_RGBD_TRAIN_PILOT")
    observations = [
        OnlineObservation(
            batch.scene_id,
            fid,
            batch.rgb[0, j],
            Cameras(batch.cameras.intrinsics[0, j], batch.cameras.c2w[0, j], batch.cameras.image_size),
        )
        for j, fid in enumerate(batch.frame_ids)
    ]
    return RGBDContext(observations, batch.depth[0, :, 0])


class _Call(torch.nn.Module):
    def __init__(self, module):
        super().__init__()
        self.carrier = module

    def forward(self, cache, fast, mode):
        if mode == "trace":
            return self.carrier.trace(cache, fast)
        return self.carrier.materialize(cache, fast)


def unroll(rule, warm, stream, bounds, action, episode):
    """(state after warmup only, state after the stream) under one fixed action."""
    bounds = torch.as_tensor(bounds, dtype=carrier._bounds.dtype).reshape(1, 2, 3)
    fractions = (carrier._candidate_normalized_xyz + 1) * 0.5
    points = bounds[:, 0] + fractions * (bounds[:, 1] - bounds[:, 0])
    call, replace = _Call(carrier), {"carrier._bounds": bounds, "carrier._candidate_points": points}
    frames, depths = [], []

    def run(cache, fast, mode):
        carrier._pending_depth = (tuple(frames), torch.stack(depths).to(torch.float32))
        try:
            return torch.func.functional_call(call, replace, (cache, fast, mode), strict=False)
        finally:
            carrier._pending_depth = None

    anchor = warm[0].camera.c2w
    cache = ObservationCache(episode_id=episode, scene_id=warm[0].scene_id)
    fast = carrier.initial_fast(episode)
    for observation, depth in zip(warm, warm.depths, strict=True):
        arrived = anchored_observation(observation, anchor)
        cache = cache.append(arrived, carrier.encode(arrived))
        frames.append(observation.frame_id)
        depths.append(depth)
    first = run(cache, fast, "materialize")
    for observation, depth in zip(stream, stream.depths, strict=True):
        arrived = anchored_observation(observation, anchor)
        cache = cache.append(arrived, carrier.encode(arrived))
        frames.append(observation.frame_id)
        depths.append(depth)
        if action != Action.OFF:
            proposal = rule.propose(run(cache, fast, "trace"))
            fast = commit_write(fast, proposal, action, cache_revision=cache.revision)
    return first, run(cache, fast, "materialize"), anchor, fast


def scenes(names):
    records = sorted(
        (r for r in manifest["scenes"] if r["scene_id"] in names and r["split"] == "TRAIN"),
        key=lambda r: r["scene_id"],
    )
    return [DepthBoundsSceneData(r, manifest, v7, "cpu", access) for r in records]


torch.manual_seed(SEED)
rule = DirectWriteRule(carrier.config, WriteConfig())
initial = deepcopy(rule)
optimizer = torch.optim.Adam(rule.parameters(), lr=1e-3)
train = scenes(set(sets["TRAIN24"]))
generator = torch.Generator().manual_seed(SEED)
started, log = time.perf_counter(), []
for step in range(1, STEPS + 1):
    scene = train[(step - 1) % len(train)]
    role = "AB"[((step - 1) // len(train)) % 2]
    slot = ((step - 1) // (2 * len(train))) % len(scene.record["roles"]["primary_query"])
    ids = stream_ids(scene.record, role)
    if len(ids) < K:
        continue
    _, state, anchor, _ = unroll(
        rule, scene.context(role), rgbd(scene.record, ids), scene.bounds[role], Action.ALL, f"t{step}"
    )
    target = scene.target(slot)
    cameras = transform_cameras(target.cameras, torch.linalg.inv(anchor))
    indices = torch.randint(128 * 160, (1024,), generator=generator)
    loss, terms = training_loss(state, cameras, target.rgb, target.depth, indices, "C1")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(rule.parameters(), 1.0)
    optimizer.step()
    log.append(float(terms["loss_depth"]))
    if step % 100 == 0:
        print(step, round(float(np.mean(log[-100:])), 4), round(time.perf_counter() - started), flush=True)

renderer = FixedMeasurementRenderer(n_samples=64, ray_chunk_size=2048)
policies = {"NO_STREAM": None, "OFF": (initial, Action.OFF), "ALL_UNTRAINED": (initial, Action.ALL)}
policies["ALL_TRAINED"] = (rule, Action.ALL)
scores = {name: {} for name in policies}
norms = []
with torch.no_grad():
    for scene in scenes(set(sets["TRAIN72"]) - set(sets["TRAIN24"])):
        sid = scene.record["scene_id"]
        per = {name: [] for name in policies}
        for role in ("A", "B"):
            ids = stream_ids(scene.record, role)
            if len(ids) < K:
                continue
            warm, stream = scene.context(role), rgbd(scene.record, ids)
            states = {}
            for name, spec in policies.items():
                if spec is None:
                    continue
                first, final, anchor, fast = unroll(spec[0], warm, stream, scene.bounds[role], spec[1], sid)
                states[name] = final
                states["NO_STREAM"] = first
                if name == "ALL_TRAINED":
                    norms.append(float(fast.delta_fuse.norm() + fast.delta_complete.norm()))
            for slot in range(len(scene.record["roles"]["primary_query"])):
                target = scene.target(slot)
                cameras = transform_cameras(target.cameras, torch.linalg.inv(anchor))
                for name, state in states.items():
                    pred = renderer(state, cameras, ("rgb", "depth", "visibility"))
                    per[name].append(_metrics(pred, target.rgb, target.depth)["depth_absrel"])
        for name in policies:
            if per[name]:
                scores[name][sid] = float(np.mean(per[name]))
common = sorted(set.intersection(*(set(v) for v in scores.values())))


def paired(a, b):
    diff = np.array([scores[a][s] - scores[b][s] for s in common])
    rng = np.random.default_rng(0)
    boot = diff[rng.integers(0, len(diff), (5000, len(diff)))].mean(1)
    return {
        "mean": float(diff.mean()),
        "ci95": [float(x) for x in np.percentile(boot, [2.5, 97.5])],
        "better": int((diff > 0).sum()),
        "n": len(diff),
    }


result = {
    "seed": SEED,
    "steps": STEPS,
    "trainx_scenes": len(common),
    "absrel": {k: float(np.mean([v[s] for s in common])) for k, v in scores.items()},
    "stream_gain_NO_STREAM_minus_OFF": paired("NO_STREAM", "OFF"),
    "write_gain_OFF_minus_ALL_TRAINED": paired("OFF", "ALL_TRAINED"),
    "untrained_write_OFF_minus_ALL_UNTRAINED": paired("OFF", "ALL_UNTRAINED"),
    "trained_fast_norm_mean": float(np.mean(norms)) if norms else None,
    "train_depth_absrel_first100_last100": [float(np.mean(log[:100])), float(np.mean(log[-100:]))],
    "seconds": time.perf_counter() - started,
    "exploratory_train_only": True,
}
out = Path(f"/tmp/claude_coreb_pilot_{SEED}_{STEPS}.json")
out.write_text(json.dumps(result, indent=1))
print(json.dumps(result), flush=True)
