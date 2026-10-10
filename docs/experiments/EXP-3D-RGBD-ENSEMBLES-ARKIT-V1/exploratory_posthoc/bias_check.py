"""EXPLORATORY (post hoc, after the formal analyses): is the completion ensemble's gain information from the learned
carrier G1, or a cancellation of opposite biases (harmonic too far, G1 collapsing toward zero depth because its
rendered depth is not divided by visibility)? On FAR pixels, per scene: signed log error log(pred / gt) of H3, G1, HG;
the share of FAR pixels where G1 < 0.25 x H3 ("collapsed"); and the AbsRel of pure-shrinkage controls of the harmonic
(0.5 x H3, 0.75 x H3) next to HG and G1. Uses the arms seal (H3, NEAR, G1 seeds) and the query depth.
Run through cv_host_run.py from the repository root (CUDA hidden):
  python bias_check.py ROOT COHORT
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
        g1 = np.mean([p[f"G1_{s}"] for s in arms["seeds"]], axis=0)
        preds = {"H3": h3, "G1": g1, "HG": (h3 + g1) / 2, "H3x0.5": 0.5 * h3, "H3x0.75": 0.75 * h3}
        put = lambda k, v: rows[k][e["scene_id"]][e["role"]].append(float(v))  # noqa: E731
        for name, x in preds.items():
            put(f"{name}|absrel", np.mean(np.abs(x[far] - gt[far]) / gt[far]))
            put(f"{name}|logbias", np.median(np.log(np.maximum(x[far], 1e-3) / gt[far])))
        put("G1_collapsed_share", np.mean(g1[far] < 0.25 * h3[far]))
table = {k: {s: float(np.mean([np.mean(v) for v in r.values()])) for s, r in sc.items()} for k, sc in rows.items()}
print("means:", {k: round(float(np.mean(list(v.values()))), 4) for k, v in sorted(table.items())})
for name in ("G1", "HG", "H3x0.5", "H3x0.75"):
    st = paired(table["H3|absrel"], table[f"{name}|absrel"], draws=10000, seed=20260928)
    print(f"H3 - {name:8s} (FAR AbsRel): {st['mean']:+.4f} [{st['ci95'][0]:+.4f}, {st['ci95'][1]:+.4f}] "
          f"{st['positive']}/{st['tied']}/{st['negative']} median {st['median']:+.4f}")
st = paired(table["H3x0.5|absrel"], table["HG|absrel"], draws=10000, seed=20260928)
print(f"H3x0.5 - HG: {st['mean']:+.4f} [{st['ci95'][0]:+.4f}, {st['ci95'][1]:+.4f}] "
      f"{st['positive']}/{st['tied']}/{st['negative']}")
