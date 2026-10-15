"""EXP-3D-READOUT-METHOD-V1, Amendment 1: statistics (CPU, deterministic: run twice and compare).

Joins results/amend1 (B1_FF, B7_FF, B1_REPLAY) with results/phase12 (B0, B1, B7, CORE_A, SYS_B, SYS_C) by query key.
Place = mean over episodes of the mean over queries; V1's paired place bootstrap (score_analyze.contrast).
Writes AMENDMENT_1_RESULTS.json and AMENDMENT_1_RESULTS.csv.  Run from the repository root:
  python rm_amend1_stats.py ROOT          (the amendment's analysis)
  python rm_amend1_stats.py ROOT smoke    (code check on the smoke fold FIT_SYSTEM2, writes into ROOT/smoke/amend1)
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(ROOT / "code"))
import rm_stats as RS  # noqa: E402  (reads sys.argv[1:] = ROOT [smoke]; its main() does not run)

NEW_DIR = ROOT / ("smoke/amend1" if RS.SMOKE else "results/amend1")
DEST = NEW_DIR if RS.SMOKE else ROOT

OLD = ("B0", "B1", "B7", "CORE_A", "SYS_B", "SYS_C")
NEW = ("B1_FF", "B7_FF", "B1_REPLAY")
PAIRS = (("B7_FF", "B7"), ("B1_FF", "B1"), ("SYS_B", "B7_FF"), ("B7_FF", "SYS_C"), ("B1_FF", "B7_FF"),
         ("CORE_A", "B0"), ("SYS_B", "SYS_C"))


def main():
    old = {e["key"]: e for f in RS.FOLDS
           for e in json.loads((RS.RES / f"fold_{f}.json").read_text())["queries"]}
    queries, logs, replay_diff, share = [], {}, 0.0, {"ff": [], "ttt": []}
    for f in RS.FOLDS:
        d = json.loads((NEW_DIR / f"fold_{f}.json").read_text())
        logs[f] = {"train_places": d["train_places"], "n_params": d["n_params"],
                   "best_epochs": [v["best_epoch"] for v in d["train_logs"].values()]}
        for e in d["queries"]:
            o = old[e["key"]]
            for s, (a, n) in e["methods"]["B1_REPLAY"].items():
                b, m = o["methods"]["B1"][s]
                replay_diff = max(replay_diff, abs(a - b) + abs(n - m))
            share["ff"].append(e["ff_covered_share"])
            share["ttt"].append(e["ttt_covered_share"])
            queries.append({**o, "methods": {**{m: o["methods"][m] for m in OLD},
                                             **{m: e["methods"][m] for m in NEW}}})
    methods = OLD + NEW
    table = RS.place_table(queries, methods)
    means = {m: float(np.mean(list(table[m].values()))) for m in methods}
    fold_q = {f: [e for e in queries if e["fold"] == f] for f in RS.FOLDS}
    fold_means = {}
    for f in RS.FOLDS:
        t = RS.place_table(fold_q[f], methods)
        fold_means[f] = {m: float(np.mean(list(t[m].values()))) for m in methods}
    contrasts = {f"{a}_minus_{b}": {**RS.short(RS.contrast(table, a, b)), **RS.summary(table, a, b)} for a, b in PAIRS}
    p1 = contrasts["B7_FF_minus_B7"]
    checks = {"ci_lower_gt_0": p1["ci95"][0] > 0, "mean_gain_ge_0.002": p1["mean"] >= 0.002,
              "median_gain_gt_0": p1["median_gain"] > 0,
              "top1_share_le_0.5": (p1["positive_top1_contribution"] or 0) <= 0.5}
    label = "SUPPORTED" if all(checks.values()) else ("HARMFUL" if p1["ci95"][1] < 0 else "NOT_ESTABLISHED")
    regional = {m: {s: RS.pixel_absrel(queries, m, s)[0] for s in RS.STRATA} for m in methods}
    w = RS.query_weights(queries)
    res = {"places": len(table["B7"]), "queries": len(queries), "means": means, "fold_means": fold_means,
           "contrasts": contrasts, "primary": {"contrast": p1, "checks": checks,
                                               "label": "TTT_ADDED_VALUE_IN_KF_V2_STRUCTURE=" + label},
           "pooled_pixel_absrel": regional,
           "contributions_vs_B7_FF": {"B7": RS.contribution(queries, w, "B7_FF", "B7")},
           "covered_share_mean": {k: float(np.mean(v)) for k, v in share.items()},
           "b1_replay_max_abs_difference": replay_diff, "logs": logs}
    (DEST / "AMENDMENT_1_RESULTS.json").write_text(json.dumps(res, indent=1, sort_keys=True) + "\n")
    rows = [[m, f"{means[m]:.5f}"] + [f"{fold_means[f][m]:.5f}" for f in RS.FOLDS] +
            [f"{regional[m][s]:.5f}" for s in ("ALL", "COV", "UNC_NEAR", "UNC_FILL", "ZERO_HIT")] for m in methods]
    rows += [[k, f"{c['mean']:+.5f}", f"[{c['ci95'][0]:+.5f}, {c['ci95'][1]:+.5f}]",
              "/".join(map(str, c["improved_tied_worse"])), c["status"], f"{c['median_gain']:+.5f}",
              f"{c['positive_top1_contribution'] or 0:.2f}", "", "", "", ""] for k, c in contrasts.items()]
    (DEST / "AMENDMENT_1_RESULTS.csv").write_text(RS.csv_text(
        ["row", "pooled_place_absrel_or_gain"] + [f"{f}_or_{x}" for f, x in zip(RS.FOLDS, ("ci95", "btw", "status",
                                                                                          "median"))] +
        ["pixel_ALL_or_top1", "pixel_COV", "pixel_UNC_NEAR", "pixel_UNC_FILL", "pixel_ZERO_HIT"], rows))
    print("AMEND1", res["primary"]["label"], json.dumps(checks), "replay", replay_diff)
    print(json.dumps({m: round(v, 5) for m, v in sorted(means.items(), key=lambda kv: kv[1])}))


if __name__ == "__main__":
    main()
