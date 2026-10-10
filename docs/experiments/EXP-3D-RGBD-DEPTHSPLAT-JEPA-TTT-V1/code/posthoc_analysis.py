"""EXP-3D-RGBD-DEPTHSPLAT-JEPA-TTT-V1, EXPLORATORY post-hoc analysis (not preregistered; run only after the formal
score stage). Reads the sealed predictions (hash-checked) and the query depth.

1. Coverage-conditional FAR comparison: on the FAR pixels a builder covers (alpha >= 0.5), its ray depth versus
   REPROJ_HARMONIC and versus G1 (seed mean) on exactly those pixels; plus the share of FAR pixels covered.
   This separates "how much of FAR the state reaches" from "how good it is where it predicts".
2. Where TTT changes the state: per region (NEAR7 / FAR7) the AbsRel of ZS_CACHE_F7 and ZS_TTT_F7, the share of
   ALL pixels in each region, and the pixel-weighted contribution of each region to the ALL difference.
3. Depth-error structure of the covered FAR pixels: signed log error (bias) of each builder and of harmonic.
Scene values as in the formal analysis (role mean of query means; J = mean over seeds); scene bootstrap 10,000 draws.
Run through cv_host_run.py from the repository root (CUDA hidden):
  python posthoc_analysis.py ROOT [--cohort FRESH78] [--arms-seal SEAL] [--ds NAME=DIR ...]
"""

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from mcss.mechanism_pilot.direct_capacity_statistics import paired

ALPHA_MIN = 0.5
parser = argparse.ArgumentParser()
parser.add_argument("root", type=Path)
parser.add_argument("--cohort", default="FRESH78")
parser.add_argument("--arms-seal", type=Path)
parser.add_argument("--ds", nargs="*", default=[])
args = parser.parse_args()
root, cohort = args.root.resolve(), args.cohort
TTT = "TTT_m" + format(json.loads((root / "ttt_selection.json").read_text())["selected_mult"], "g")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(item):
    if sha(item["path"]) != item["sha256"]:
        raise PermissionError(f"sealed prediction changed: {item['path']}")
    return dict(np.load(item["path"]))


lock = json.loads((root / "frame_role_lock.json").read_text())["cohorts"][cohort]
records = {r["scene_id"]: r for r in json.loads(Path(lock["manifest"]).read_text())["scenes"]}
arms = json.loads((args.arms_seal or root / f"raw/eval/{cohort}/prediction_seal.json").read_text())
geometry = json.loads((root / f"geometry/geometry_seal_{cohort}.json").read_text())["predictions"]
directories = dict(spec.split("=", 1) for spec in args.ds) or {
    name: root / f"ds/{cohort}" / name for name in ("ZS", "ZS3", "J1", "J2", "J3")}
builders = {}
for name, directory in directories.items():
    parts = sorted(Path(directory).glob("ds_seal_*of*.json"))
    if parts:
        builders[name] = {}
        for part in parts:
            builders[name].update(json.loads(part.read_text())["predictions"])

rows = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))  # metric -> scene -> role -> values
for e in lock["episodes"]:
    frames = {f["frame_id"]: f for f in records[e["scene_id"]]["frames"]}
    for qid in e["queries"]:
        key = f"{e['scene_id']}/{e['role']}/{qid}"
        geo = load(geometry[key])
        sealed = load(arms["predictions"][f"{arms['cohort']}/{key}"])
        gt = np.load(frames[qid]["depth"]).squeeze().astype(np.float64)
        valid = np.isfinite(gt) & (gt > 0)
        near, near7 = geo["near"].astype(bool), geo["near7"].astype(bool)
        far, far7 = valid & ~near, valid & ~near7
        h3, h7, cosine = geo["REPROJ_HARMONIC"], geo["REPROJ7_HARMONIC"], geo["cosine"]
        g1 = np.mean([sealed[f"G1_{s}"] for s in arms["seeds"]], axis=0)

        def put(metric, value):
            if value is not None and np.isfinite(value):
                rows[metric][e["scene_id"]][e["role"]].append(float(value))

        for name, preds in builders.items():
            arrays = load(preds[key])
            alpha = arrays["CTX_alpha"].astype(np.float64)
            ray = arrays["CTX_depth_z"].astype(np.float64) / np.maximum(alpha, 1e-12) / cosine
            covered = far & (alpha >= ALPHA_MIN)
            if far.any():
                put(f"{name}|far_coverage", covered.sum() / far.sum())
            if covered.sum() >= 20:
                err = lambda p: np.abs(p[covered] - gt[covered]) / gt[covered]  # noqa: E731
                put(f"{name}|covered_far_ds", err(ray).mean())
                put(f"{name}|covered_far_harmonic", err(h3).mean())
                put(f"{name}|covered_far_g1", err(g1).mean())
                put(f"{name}|covered_far_ds_logbias", np.mean(np.log(ray[covered] / gt[covered])))
                put(f"{name}|covered_far_harmonic_logbias", np.mean(np.log(h3[covered] / gt[covered])))
            if name == "ZS":
                f7 = {}
                for arm in ("CACHE", TTT):
                    a = arrays[f"{arm}_alpha"].astype(np.float64)
                    r = arrays[f"{arm}_depth_z"].astype(np.float64) / np.maximum(a, 1e-12) / cosine
                    f7[arm] = np.where(a >= ALPHA_MIN, r, h7)
                n_all = valid.sum()
                for region, mask in (("NEAR7", valid & near7), ("FAR7", far7)):
                    put(f"share_{region}", mask.sum() / n_all)
                    if mask.any():
                        err = {arm: np.abs(p[mask] - gt[mask]) / gt[mask] for arm, p in f7.items()}
                        put(f"ttt_{region}_cache", err["CACHE"].mean())
                        put(f"ttt_{region}_ttt", err[TTT].mean())
                        # contribution of the region to ALL(CACHE) - ALL(TTT): pixel-weighted
                        put(f"ttt_{region}_contribution", (err["CACHE"].sum() - err[TTT].sum()) / n_all)
                        put(f"ttt_{region}_reproj7", (np.abs(h7[mask] - gt[mask]) / gt[mask]).mean())


def scenes(metric):
    return {s: float(np.mean([np.mean(v) for v in roles.values()])) for s, roles in rows[metric].items()}


table = {metric: scenes(metric) for metric in rows}
js = [b for b in ("J1", "J2", "J3") if b in builders]
for suffix in ("far_coverage", "covered_far_ds", "covered_far_harmonic", "covered_far_g1", "covered_far_ds_logbias",
               "covered_far_harmonic_logbias"):
    parts = [table.get(f"{b}|{suffix}") for b in js]
    if len(parts) >= 2 and all(parts):
        common = set.intersection(*(set(p) for p in parts))
        table[f"J|{suffix}"] = {s: float(np.mean([p[s] for p in parts])) for s in common}
out = {"note": "EXPLORATORY, not preregistered; after the formal score stage", "ttt_arm": TTT,
       "means": {m: {"mean": float(np.mean(list(v.values()))), "scenes": len(v)} for m, v in sorted(table.items())},
       "contrasts": {}}
for name in ("ZS", "ZS3", "J"):
    a, b, c = (table.get(f"{name}|covered_far_{k}") for k in ("harmonic", "ds", "g1"))
    if a and b:
        common = sorted(set(a) & set(b))
        stats = paired({s: a[s] for s in common}, {s: b[s] for s in common}, draws=10000, seed=20260928)
        stats.pop("per_scene", None), stats.pop("loso", None)
        out["contrasts"][f"{name}: harmonic - DS on covered FAR"] = stats
    if c and b:
        common = sorted(set(c) & set(b))
        stats = paired({s: c[s] for s in common}, {s: b[s] for s in common}, draws=10000, seed=20260928)
        stats.pop("per_scene", None), stats.pop("loso", None)
        out["contrasts"][f"{name}: G1 - DS on covered FAR"] = stats
(root / "results").mkdir(exist_ok=True)
(root / "results" / f"{cohort}_posthoc_exploratory.json").write_text(json.dumps(out, indent=1) + "\n")
for m, v in out["means"].items():
    print(f"{m:45s} {v['mean']:+.4f} ({v['scenes']})")
for label, s in out["contrasts"].items():
    print(f"{label}: {s['mean']:+.4f} [{s['ci95'][0]:+.4f}, {s['ci95'][1]:+.4f}] {s['improved_tied_worse']}")
