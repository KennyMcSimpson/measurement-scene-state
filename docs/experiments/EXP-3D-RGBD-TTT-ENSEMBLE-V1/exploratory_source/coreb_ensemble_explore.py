"""EXPLORATORY (in-sample on FRESH78, after the formal V1 analysis; not a test): does averaging the 7-frame DepthSplat
states with the model-free REPROJ7_HARMONIC make them consistently better than REPROJ7_HARMONIC (ALL region)?
Readouts per query (ray depth = depth_z / alpha / cos where alpha >= 0.5, else REPROJ7_HARMONIC):
  R7, CACHE, TTT, and the averages (R7 + CACHE) / 2, (R7 + TTT) / 2, plus the NEAR7 / FAR7 split for TTT.
Run through cv_host_run.py from the repository root (CUDA hidden):  python coreb_ensemble_explore.py V1_ROOT
"""
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from mcss.mechanism_pilot.direct_capacity_statistics import paired

root = Path(sys.argv[1]).resolve()


def load(item):
    if hashlib.sha256(Path(item["path"]).read_bytes()).hexdigest() != item["sha256"]:
        raise PermissionError(item["path"])
    return dict(np.load(item["path"]))


lock = json.loads((root / "frame_role_lock.json").read_text())["cohorts"]["FRESH78"]
records = {r["scene_id"]: r for r in json.loads(Path(lock["manifest"]).read_text())["scenes"]}
geometry = json.loads((root / "geometry/geometry_seal_FRESH78.json").read_text())["predictions"]
zs = {}
for part in sorted((root / "ds/FRESH78/ZS").glob("ds_seal_*of*.json")):
    zs.update(json.loads(part.read_text())["predictions"])
rows = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
for e in lock["episodes"]:
    frames = {f["frame_id"]: f for f in records[e["scene_id"]]["frames"]}
    for qid in e["queries"]:
        key = f"{e['scene_id']}/{e['role']}/{qid}"
        geo, arrays = load(geometry[key]), load(zs[key])
        gt = np.load(frames[qid]["depth"]).squeeze().astype(np.float64)
        valid = np.isfinite(gt) & (gt > 0)
        near7 = geo["near7"].astype(bool)
        r7, cosine = geo["REPROJ7_HARMONIC"], geo["cosine"]
        state = {}
        for arm in ("CACHE", "TTT_m1"):
            alpha = arrays[f"{arm}_alpha"].astype(np.float64)
            ray = arrays[f"{arm}_depth_z"].astype(np.float64) / np.maximum(alpha, 1e-12) / cosine
            state[arm] = np.where(alpha >= 0.5, ray, r7)
        readouts = {"R7": r7, "CACHE": state["CACHE"], "TTT": state["TTT_m1"],
                    "R7_CACHE": (r7 + state["CACHE"]) / 2, "R7_TTT": (r7 + state["TTT_m1"]) / 2,
                    "R7_TTT_GEO": np.sqrt(np.maximum(r7, 1e-6) * np.maximum(state["TTT_m1"], 1e-6))}
        for name, p in readouts.items():
            for region, mask in (("ALL", valid), ("NEAR7", valid & near7), ("FAR7", valid & ~near7)):
                if mask.any():
                    rows[f"{name}|{region}"][e["scene_id"]][e["role"]].append(
                        float(np.mean(np.abs(p[mask] - gt[mask]) / gt[mask])))
table = {m: {s: float(np.mean([np.mean(v) for v in roles.values()])) for s, roles in sc.items()} for m, sc in rows.items()}
print("means:", {m: round(float(np.mean(list(v.values()))), 4) for m, v in sorted(table.items())})
for name in ("CACHE", "TTT", "R7_CACHE", "R7_TTT", "R7_TTT_GEO"):
    for region in ("ALL", "NEAR7"):
        st = paired(table[f"R7|{region}"], table[f"{name}|{region}"], draws=10000, seed=20260928)
        print(f"R7 - {name:10s} {region:5s}: {st['mean']:+.4f} [{st['ci95'][0]:+.4f}, {st['ci95'][1]:+.4f}] "
              f"better/tied/worse {st['positive']}/{st['tied']}/{st['negative']} median {st['median']:+.4f}")
