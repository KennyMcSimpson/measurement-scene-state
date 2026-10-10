"""EXP-3D-RGBD-DEPTHSPLAT-TTT-ABLATION-V1: preregistered contrasts and branch from the scene table that the locked
V1 score_analyze.py score stage wrote for this root (--ds ZS=<V1 ZS seals> NV=<this experiment's NV seals>).
Statistics are V1's (paired scene bootstrap, 10,000 draws, seed 20260928, V2 gate).
Run through cv_host_run.py from the repository root (CUDA hidden):
  python ablation_analyze.py ROOT COHORT --zs-mult 1 --nv-mult M
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "EXP-3D-RGBD-DEPTHSPLAT-JEPA-TTT-V1" / "code"))
from score_analyze import contrast, write  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("root", type=Path)
parser.add_argument("cohort")
parser.add_argument("--zs-mult", type=float, required=True)
parser.add_argument("--nv-mult", type=float, required=True)
args = parser.parse_args()
root = args.root.resolve()
table = json.loads((root / "scores" / args.cohort / "scene_table.json").read_text())
zs, nv = f"ZS_TTT_m{args.zs_mult:g}_F7", f"NV_TTT_m{args.nv_mult:g}_F7"
PRIMARY = {"P1_LEARNED_BASE_UNDER_TTT_ALL": (f"{nv}|ALL", f"{zs}|ALL"),
           "P2_LEARNED_BASE_CACHE_ALL": ("NV_CACHE_F7|ALL", "ZS_CACHE_F7|ALL")}
SECONDARY = {"P3_TTT_GAIN_ON_NAIVE_ALL": ("NV_CACHE_F7|ALL", f"{nv}|ALL"),
             "P4_NAIVE_SPLAT_VS_REPROJ7_ALL": ("REPROJ7_HARMONIC|ALL", "NV_CACHE_F7|ALL"),
             "P5_NAIVE_TTT_VS_REPROJ7_ALL": ("REPROJ7_HARMONIC|ALL", f"{nv}|ALL"),
             "P6a_LEARNED_BASE_UNDER_TTT_NEAR7": (f"{nv}|NEAR7", f"{zs}|NEAR7"),
             "P6b_LEARNED_BASE_UNDER_TTT_FAR7": (f"{nv}|FAR7", f"{zs}|FAR7"),
             "REF_ZS_TTT_GAIN_ALL": ("ZS_CACHE_F7|ALL", f"{zs}|ALL")}


def run(spec):
    return {label: contrast(table, left, right) if left in table and right in table else
            {"left": left, "right": right, "status": "NO_DATA"} for label, (left, right) in spec.items()}


primary, secondary = run(PRIMARY), run(SECONDARY)
p1, p3 = primary["P1_LEARNED_BASE_UNDER_TTT_ALL"], secondary["P3_TTT_GAIN_ON_NAIVE_ALL"]
if p1["status"] == "SUPPORTED":
    branch = "LEARNED_BASE_MATTERS"
elif p1["status"] == "HARMFUL":
    branch = "NAIVE_BETTER"
elif p1["ci95"][0] <= 0 <= p1["ci95"][1] and p3["status"] == "SUPPORTED":
    branch = "TTT_IS_GENERIC"
else:
    branch = "INCONCLUSIVE"
means = {}
for name in ("REPROJ7_HARMONIC", "ZS_CTX_F", "NV_CTX_F", "ZS_CACHE_F7", "NV_CACHE_F7", zs, nv):
    for region in ("ALL", "NEAR7", "FAR7"):
        key = f"{name}|{region}"
        if key in table:
            means[key] = {"mean": float(np.mean(list(table[key].values()))), "scenes": len(table[key])}
for key in table:
    if key.endswith("|COVERAGE") and key.startswith("NV_"):
        means[key] = {"mean": float(np.mean(list(table[key].values()))), "scenes": len(table[key])}
result = {"cohort": args.cohort, "branch": branch, "zs_mult": args.zs_mult, "nv_mult": args.nv_mult,
          "primary": primary, "secondary": secondary, "means": means}
write(root / "results" / f"{args.cohort}_ablation_results.json", result)
print("ABLATION", args.cohort, branch)
for label, s in {**primary, **secondary}.items():
    if s["status"] != "NO_DATA":
        print(f"  {label}: {s['mean']:+.4f} [{s['ci95'][0]:+.4f}, {s['ci95'][1]:+.4f}] "
              f"{s['positive']}/{s['tied']}/{s['negative']} {s['status']}")
for key, v in sorted(means.items()):
    print(f"  {key:32s} {v['mean']:.4f} ({v['scenes']})")
