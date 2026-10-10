"""EXPLORATORY (post hoc; Hypersim cohorts already used many times): does the ARKit depth-spread rule (learned
completion where the context depth spread S = (p90 - p10) / median >= 1.2, else the harmonic fill) also hold on
synthetic data? For EVAL34 (EXP-3D-RGBD-COMPLETION-ENSEMBLE-V1) and FRESH78 (EXP-3D-RGBD-DEPTHSPLAT-JEPA-TTT-V1), with
the Hypersim-trained carrier G1 (normalized readout G1N, three seeds, from the arms seals): the S distribution, the
Spearman rho of the scene S with the scene gain H3 - G1N on FAR, and H3 - SEL_G1N at S >= 1.2 and at the cohort's own
median of S. Uses only sealed predictions (sha-checked) and query depth already scored.
Run through cv_host_run.py from the repository root (CUDA hidden):  python hypersim_selector_check.py
"""
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path.cwd() / "outputs/EXP-3D-RGBD-DEPTHSPLAT-JEPA-TTT-V1/code"))
from score_analyze import contrast, write  # noqa: E402

COHORTS = {"EVAL34": Path("outputs/EXP-3D-RGBD-COMPLETION-ENSEMBLE-V1"),
           "FRESH78": Path("outputs/EXP-3D-RGBD-DEPTHSPLAT-JEPA-TTT-V1")}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(item):
    if sha(item["path"]) != item["sha256"]:
        raise PermissionError(f"sealed prediction changed: {item['path']}")
    return dict(np.load(item["path"]))


def rank(x):
    return np.argsort(np.argsort(x, kind="stable"), kind="stable").astype(float)


def scene_table(rows, value):
    grouped = defaultdict(lambda: defaultdict(list))
    for r in rows:
        grouped[r["scene_id"]][r["role"]].append(value(r))
    return {s: float(np.mean([np.mean(v) for v in roles.values()])) for s, roles in grouped.items()}


result = {}
for cohort, root in COHORTS.items():
    lock = json.loads((root / "frame_role_lock.json").read_text())["cohorts"][cohort]
    records = {r["scene_id"]: r for r in json.loads(Path(lock["manifest"]).read_text())["scenes"]}
    arms = json.loads((root / f"raw/eval/{cohort}/prediction_seal.json").read_text())
    rows = []
    for e in lock["episodes"]:
        frames = {f["frame_id"]: f for f in records[e["scene_id"]]["frames"]}
        ctx = np.concatenate([d[np.isfinite(d) & (d > 0)] for d in
                              (np.load(frames[c]["depth"]).squeeze().astype(np.float64) for c in e["context"])])
        spread = float((np.percentile(ctx, 90) - np.percentile(ctx, 10)) / np.median(ctx))
        for qid in e["queries"]:
            p = load(arms["predictions"][f"{cohort}/{e['scene_id']}/{e['role']}/{qid}"])
            near = p["near"].astype(bool)
            h3 = p["REPROJ_HARMONIC"]
            g1n = np.mean([np.where(p[f"OPACITY_G1_{s}"] >= 0.5,
                                    p[f"G1_{s}"] / np.maximum(p[f"OPACITY_G1_{s}"].astype(np.float64), 1e-6), h3)
                           for s in arms["seeds"]], axis=0)
            gt = np.load(frames[qid]["depth"]).squeeze().astype(np.float64)
            far = np.isfinite(gt) & (gt > 0) & ~near
            if not far.any():
                continue
            err = lambda x: float(np.mean(np.abs(x[far] - gt[far]) / gt[far]))  # noqa: E731
            rows.append({"scene_id": e["scene_id"], "role": e["role"], "spread": spread, "h3": err(h3), "g1n": err(g1n)})
    spreads = np.array([r["spread"] for r in rows])
    median = float(np.median(spreads))
    table = {"H3|FAR": scene_table(rows, lambda r: r["h3"]), "G1N|FAR": scene_table(rows, lambda r: r["g1n"])}
    for name, t in (("1.2", 1.2), ("median", median)):
        table[f"SEL{name}|FAR"] = scene_table(rows, lambda r, t=t: r["g1n"] if r["spread"] >= t else r["h3"])
    gain = {s: table["H3|FAR"][s] - table["G1N|FAR"][s] for s in table["H3|FAR"]}
    scene_spread = scene_table(rows, lambda r: r["spread"])
    ids = sorted(gain)
    rho = float(np.corrcoef(rank(np.array([scene_spread[s] for s in ids])), rank(np.array([gain[s] for s in ids])))[0, 1])
    out = {"queries": len(rows), "spread_quantiles": {q: float(np.percentile(spreads, q)) for q in (10, 25, 50, 75, 90)},
           "share_selected_at_1.2": float(np.mean(spreads >= 1.2)), "spearman_spread_vs_gain": rho, "contrasts": {}}
    print(f"{cohort}: {len(rows)} queries; S quantiles", {q: round(v, 2) for q, v in out["spread_quantiles"].items()},
          f"selected at 1.2: {out['share_selected_at_1.2']:.2f}; rho(S, gain G1N) {rho:+.3f}")
    for name in ("G1N", "SEL1.2", "SELmedian"):
        s = contrast(table, "H3|FAR", f"{name}|FAR")
        out["contrasts"][name] = s
        print(f"  H3 - {name:9s}: {s['mean']:+.4f} [{s['ci95'][0]:+.4f}, {s['ci95'][1]:+.4f}] "
              f"{s['positive']}/{s['tied']}/{s['negative']} median {s['median']:+.4f} {s['status']}")
    result[cohort] = out
write(Path("outputs/EXP-3D-RGBD-COMPLETION-SELECTOR-ARKIT-V1/exploratory_posthoc/hypersim_selector_check.json"),
      {"note": "exploratory; Hypersim cohorts used many times", **result})
