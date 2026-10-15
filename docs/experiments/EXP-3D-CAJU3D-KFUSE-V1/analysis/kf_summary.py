"""EXP-3D-CAJU3D-KFUSE-V1 readable summary of the written results (no new computation on ground truth).
Run from the repository root:  python kf_summary.py ROOT > ROOT/results/summary.txt
"""

import json
import sys
from pathlib import Path

import numpy as np

root = Path(sys.argv[1]).resolve()


def line(c):
    i, t, w = c["improved_tied_worse"]
    top1 = c["positive_top1_contribution"]
    return (f"{c['mean']:+.5f} [{c['ci95'][0]:+.5f}, {c['ci95'][1]:+.5f}] {i}/{t}/{w} {c['status']}"
            f" loso [{c['loso_range'][0]:+.5f}, {c['loso_range'][1]:+.5f}]"
            f" top1 {'-' if top1 is None else f'{top1:.2f}'}")


r = json.loads((root / "results/confirm_results.json").read_text())
for cohort in ("CONF_SYSTEM3", "CONF_VAL"):
    x = r[cohort]
    print(f"{cohort}: {x['places']} places, {x['queries']} queries, {x['ground_truth_reads']} ground-truth reads")
    print("  means:", json.dumps({k: round(v, 5) for k, v in sorted(x["means"].items(), key=lambda kv: kv[1])}))
    print(f"  P  KF_BASE - KF_JEPA      {line(x['primary_base_minus_jepa'])}")
    for k, c in x["secondary"].items():
        print(f"     {k:22s} {line(c)}")
print("branch:", r["branch"])
f = json.loads((root / "results/fit_loo_results.json").read_text())
print(f"\nFIT leave-one-place-out ({f['places']} places, descriptive):",
      json.dumps({k: round(v, 5) for k, v in sorted(f["loo_means"].items(), key=lambda kv: kv[1])}))
for k, c in f["loo_contrasts"].items():
    print(f"     {k:22s} {line(c)}")
t = json.loads((root / "fusion_tables.json").read_text())
w = np.array(t["weight_grid"])
print("\nfrozen tables: edges", json.dumps({k: [round(e, 3) for e in v] for k, v in t["edges"].items()}))
print("  GEO_U", w[t["weights"]["GEO_U"]].tolist(), " KF_DIS", w[t["weights"]["KF_DIS"]].tolist())
for arm, shape, labels in (("KF_BASE", (8, 4, 2), "DIS x ALPHA x NEAR"), ("KF_JEPA", (8, 4, 2), "DIS x LOGP x NEAR")):
    table = w[t["weights"][arm]].reshape(shape)
    print(f"  {arm} ({labels}); rows DIS bins, columns the second feature's bins, [far, near]:")
    for i in range(shape[0]):
        print("    ", "  ".join(f"[{table[i, j, 0]:.2f} {table[i, j, 1]:.2f}]" for j in range(shape[1])))
