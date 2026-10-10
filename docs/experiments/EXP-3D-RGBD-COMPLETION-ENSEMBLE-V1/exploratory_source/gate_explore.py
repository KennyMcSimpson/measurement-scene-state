"""EXPLORATORY (in-sample on FRESH78, after the formal V1 analysis; not a test): can a label-free agreement signal
decide where to trust learned completion in FAR?  Readouts on FAR pixels (3-frame NEAR/FAR):
  H3       REPROJ_HARMONIC
  J        reverse-JEPA DepthSplat (mean of the three seeds' ray depth where alpha >= 0.5, else H3)
  G1       the V21 G1 carrier (mean of three seeds)
  JG       mean of J and G1 where J covers, else G1
  GATE_t   where J covers and |log(J / G1)| < t: (J + G1) / 2, else H3     (agreement gate)
Scene values as in the formal analysis; contrasts H3 - readout with the project's paired bootstrap.
Run through cv_host_run.py from the repository root (CUDA hidden):  python gate_explore.py V1_ROOT
"""
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from mcss.mechanism_pilot.direct_capacity_statistics import paired

root = Path(sys.argv[1]).resolve()
TAUS = (0.05, 0.1, 0.2, 0.3)


def load(item):
    if hashlib.sha256(Path(item["path"]).read_bytes()).hexdigest() != item["sha256"]:
        raise PermissionError(item["path"])
    return dict(np.load(item["path"]))


lock = json.loads((root / "frame_role_lock.json").read_text())["cohorts"]["FRESH78"]
records = {r["scene_id"]: r for r in json.loads(Path(lock["manifest"]).read_text())["scenes"]}
arms = json.loads((root / "raw/eval/FRESH78/prediction_seal.json").read_text())
geometry = json.loads((root / "geometry/geometry_seal_FRESH78.json").read_text())["predictions"]
js = {}
for s in ("J1", "J2", "J3"):
    js[s] = {}
    for part in sorted((root / "ds/FRESH78" / s).glob("ds_seal_*of*.json")):
        js[s].update(json.loads(part.read_text())["predictions"])
rows = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
for e in lock["episodes"]:
    frames = {f["frame_id"]: f for f in records[e["scene_id"]]["frames"]}
    for qid in e["queries"]:
        key = f"{e['scene_id']}/{e['role']}/{qid}"
        geo, sealed = load(geometry[key]), load(arms["predictions"][f"FRESH78/{key}"])
        gt = np.load(frames[qid]["depth"]).squeeze().astype(np.float64)
        far = np.isfinite(gt) & (gt > 0) & ~geo["near"].astype(bool)
        if far.sum() < 20:
            continue
        h3, cosine = geo["REPROJ_HARMONIC"], geo["cosine"]
        g1 = np.mean([sealed[f"G1_{s}"] for s in arms["seeds"]], axis=0)
        depths, covers = [], []
        for s in js:
            a = load(js[s][key])
            alpha = a["CTX_alpha"].astype(np.float64)
            depths.append(a["CTX_depth_z"].astype(np.float64) / np.maximum(alpha, 1e-12) / cosine)
            covers.append(alpha >= 0.5)
        covers = np.array(covers)
        covered = covers.sum(axis=0) >= 2  # at least two of the three seeds cover the pixel
        pooled = np.where(covers, np.array(depths), 0.0).sum(axis=0) / np.maximum(covers.sum(axis=0), 1)
        j = np.where(covered, pooled, h3)  # mean over the covering seeds
        readouts = {"H3": h3, "J": j, "G1": g1, "JG": np.where(covered, (j + g1) / 2, g1),
                    "HG": (h3 + g1) / 2, "HJ": np.where(covered, (h3 + j) / 2, h3),
                    "HJG": np.where(covered, (h3 + j + g1) / 3, (h3 + g1) / 2),
                    "JG_GEO": np.where(covered, np.sqrt(np.maximum(j, 1e-6) * np.maximum(g1, 1e-6)), g1)}
        disagreement = np.abs(np.log(np.maximum(j, 1e-6) / np.maximum(g1, 1e-6)))
        for t in TAUS:
            readouts[f"GATE_{t}"] = np.where(covered & (disagreement < t), (j + g1) / 2, h3)
            rows[f"GATE_{t}|used"][e["scene_id"]][e["role"]].append(float((covered & (disagreement < t))[far].mean()))
        for name, p in readouts.items():
            rows[name][e["scene_id"]][e["role"]].append(float(np.mean(np.abs(p[far] - gt[far]) / gt[far])))
table = {m: {s: float(np.mean([np.mean(v) for v in roles.values()])) for s, roles in sc.items()} for m, sc in rows.items()}
print("FAR means:", {m: round(float(np.mean(list(v.values()))), 4) for m, v in table.items()})
for name in ("J", "G1", "JG", "JG_GEO", "HG", "HJ", "HJG", *[f"GATE_{t}" for t in TAUS]):
    st = paired(table["H3"], table[name], draws=10000, seed=20260928)
    print(f"H3 - {name:9s}: {st['mean']:+.4f} [{st['ci95'][0]:+.4f}, {st['ci95'][1]:+.4f}] "
          f"better/tied/worse {st['positive']}/{st['tied']}/{st['negative']} median {st['median']:+.4f}")
