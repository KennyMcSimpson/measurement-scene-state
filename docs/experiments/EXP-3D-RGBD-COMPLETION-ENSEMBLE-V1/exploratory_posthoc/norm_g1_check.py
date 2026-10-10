"""EXPLORATORY (post hoc): does a normalized, abstaining G1 readout remove the collapse and make the completion
ensemble work on real data? G1N = depth_G1 / opacity_G1 where opacity >= 0.5 (per seed), else H3; seeds averaged.
HGN = (H3 + G1N) / 2. FAR AbsRel per scene vs H3, next to the sealed G1 / HG and the 0.5 x H3 shrinkage control.
Run through cv_host_run.py from the repository root (CUDA hidden):  python norm_g1_check.py ROOT COHORT
"""
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from mcss.mechanism_pilot.direct_capacity_statistics import paired

root, cohort = Path(sys.argv[1]).resolve(), sys.argv[2]
lock = json.loads((root / "frame_role_lock.json").read_text())["cohorts"][cohort]
records = {r["scene_id"]: r for r in json.loads(Path(lock["manifest"]).read_text())["scenes"]}
arms = json.loads((root / f"raw/eval/{cohort}/prediction_seal.json").read_text())
rows = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
for e in lock["episodes"]:
    frames = {f["frame_id"]: f for f in records[e["scene_id"]]["frames"]}
    for qid in e["queries"]:
        item = arms["predictions"][f"{cohort}/{e['scene_id']}/{e['role']}/{qid}"]
        assert hashlib.sha256(Path(item["path"]).read_bytes()).hexdigest() == item["sha256"]
        p = dict(np.load(item["path"]))
        gt = np.load(frames[qid]["depth"]).squeeze().astype(np.float64)
        far = np.isfinite(gt) & (gt > 0) & ~p["near"].astype(bool)
        if far.sum() < 20:
            continue
        h3 = p["REPROJ_HARMONIC"]
        seeds = arms["seeds"]
        g1 = np.mean([p[f"G1_{s}"] for s in seeds], axis=0)
        normalized = []
        for s in seeds:
            opacity = p[f"OPACITY_G1_{s}"].astype(np.float64)
            normalized.append(np.where(opacity >= 0.5, p[f"G1_{s}"] / np.maximum(opacity, 1e-6), h3))
        g1n = np.mean(normalized, axis=0)
        preds = {"H3": h3, "G1": g1, "HG": (h3 + g1) / 2, "G1N": g1n, "HGN": (h3 + g1n) / 2, "H3x0.5": 0.5 * h3}
        for name, x in preds.items():
            rows[name][e["scene_id"]][e["role"]].append(float(np.mean(np.abs(x[far] - gt[far]) / gt[far])))
        rows["G1_abstain_share"][e["scene_id"]][e["role"]].append(
            float(np.mean([np.mean(p[f"OPACITY_G1_{s}"][far] < 0.5) for s in seeds])))
table = {k: {s: float(np.mean([np.mean(v) for v in r.values()])) for s, r in sc.items()} for k, sc in rows.items()}
print("FAR means:", {k: round(float(np.mean(list(v.values()))), 4) for k, v in sorted(table.items())})
for name in ("G1", "HG", "G1N", "HGN", "H3x0.5"):
    st = paired(table["H3"], table[name], draws=10000, seed=20260928)
    print(f"H3 - {name:6s}: {st['mean']:+.4f} [{st['ci95'][0]:+.4f}, {st['ci95'][1]:+.4f}] "
          f"{st['positive']}/{st['tied']}/{st['negative']} median {st['median']:+.4f}")
