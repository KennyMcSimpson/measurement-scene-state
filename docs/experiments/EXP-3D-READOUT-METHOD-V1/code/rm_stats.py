"""EXP-3D-READOUT-METHOD-V1: statistics, the two gates and the result files, from results/phase12 only (CPU).

Place = mean over episodes of the mean over queries (V1's aggregation); pooled = the 350 places of the 4 held-out folds.
Writes BASELINE_MATRIX.csv, CALIBRATION_CONTROLS.csv, TTT_NECESSITY_RESULTS.csv, REGIONAL_ERROR_BREAKDOWN.csv and
BOOTSTRAP_RESULTS.json in ROOT. Deterministic (the bootstrap is seeded): run it twice and compare.
Run from the repository root:  python rm_stats.py ROOT
"""

import csv
import io
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(sys.argv[1]).resolve()
SMOKE = len(sys.argv) > 2 and sys.argv[2] == "smoke"  # code check on the smoke fold only, writes into ROOT/smoke
sys.path.insert(0, str(ROOT.parent / "EXP-3D-RGBD-DEPTHSPLAT-JEPA-TTT-V1" / "code"))
from score_analyze import contrast  # noqa: E402

FOLDS = ("FIT_SYSTEM2",) if SMOKE else ("CONF_SYSTEM3", "CONF_VAL", "CAJU", "ARKIT_SYSTEM4")
RES = ROOT / ("smoke" if SMOKE else "results/phase12")
DEST = ROOT / "smoke" if SMOKE else ROOT
SIMPLE = ("B2A", "B2B", "B3", "B4", "B5")
PHASE1 = ("B0", "B1") + SIMPLE + ("B6", "B6W", "B7")
PHASE2 = ("R7", "CORE_A", "B0", "SYS_A", "SYS_B", "SYS_C", "SYS_C_ON_B", "SYS_B_ON_C")
EXCLUSIVE = ("COV", "UNC_NEAR", "UNC_FILL")
STRATA = ("ALL", "COV", "UNC_NEAR", "UNC_FILL", "ZERO_HIT", "FILL_D0_8", "FILL_D8_32", "FILL_D32", "EDGE", "FAR")


def load():
    queries = []
    fits = {}
    logs = {}
    for f in FOLDS:
        d = json.loads((RES / f"fold_{f}.json").read_text())
        queries += d["queries"]
        fits[f] = d["fits"]
        logs[f] = {"train_places": d["train_places"], "n_params": d["n_params"],
                   "best_epochs": {k: [v["best_epoch"] for v in runs.values()] for k, runs in d["train_logs"].items()}}
    return queries, fits, logs


def qabsrel(e, m):
    s, n = e["methods"][m]["ALL"]
    return s / max(n, 1)


def place_table(queries, methods):
    grouped = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for e in queries:
        for m in methods:
            grouped[m][e["place"]][e["episode"]].append(qabsrel(e, m))
    return {m: {p: float(np.mean([np.mean(v) for v in eps.values()])) for p, eps in pl.items()}
            for m, pl in grouped.items()}


def query_weights(queries):
    places = defaultdict(lambda: defaultdict(list))
    for i, e in enumerate(queries):
        places[e["place"]][e["episode"]].append(i)
    w = np.zeros(len(queries))
    for eps in places.values():
        for members in eps.values():
            for i in members:
                w[i] = 1.0 / (len(places) * len(eps) * len(members))
    return w


def summary(table, base, m):
    places = sorted(table[base])
    gain = np.array([table[base][p] - table[m][p] for p in places])
    order = np.sort(gain)
    k = max(1, int(round(0.1 * len(gain))))
    return {"median_gain": float(np.median(gain)), "worst10_mean_gain": float(order[:k].mean()),
            "worse_places": int((gain < 0).sum()), "places": len(places)}


def short(c):
    return {k: c[k] for k in ("mean", "ci95", "improved_tied_worse", "status", "positive_top1_contribution",
                              "loso_range", "median") if k in c}


def pixel_absrel(queries, m, stratum):
    s = sum(e["methods"][m][stratum][0] for e in queries)
    n = sum(e["methods"][m][stratum][1] for e in queries)
    return (s / n if n else None), n


def contribution(queries, w, base, m):
    """Pooled place-level improvement of m over base, split by the exclusive regions and by zero-hit queries."""
    out = {r: 0.0 for r in EXCLUSIVE}
    zero = 0.0
    for e, wq in zip(queries, w):
        n = max(e["methods"][base]["ALL"][1], 1)
        for r in EXCLUSIVE:
            out[r] += wq * (e["methods"][base][r][0] - e["methods"][m][r][0]) / n
        if e["hit7"] == 0:
            zero += wq * (e["methods"][base]["ALL"][0] - e["methods"][m]["ALL"][0]) / n
    out["ZERO_HIT_QUERIES"] = zero
    return out


def csv_text(header, rows):
    buf = io.StringIO()
    wri = csv.writer(buf, lineterminator="\n")
    wri.writerow(header)
    wri.writerows(rows)
    return buf.getvalue()


def main():
    queries, fits, logs = load()
    methods = sorted(set(PHASE1) | set(PHASE2))
    table = place_table(queries, methods)
    means = {m: float(np.mean(list(table[m].values()))) for m in methods}
    fold_q = {f: [e for e in queries if e["fold"] == f] for f in FOLDS}
    fold_tables = {f: place_table(fold_q[f], methods) for f in FOLDS}
    fold_means = {f: {m: float(np.mean(list(fold_tables[f][m].values()))) for m in methods} for f in FOLDS}
    w = query_weights(queries)
    res = {"places": len(table["B0"]), "queries": len(queries), "means": means, "fold_means": fold_means,
           "contrasts": {}, "gates": {}, "logs": logs}
    # Phase 1
    s_star = min(SIMPLE, key=lambda m: (round(means[m], 12), SIMPLE.index(m)))
    pairs = [("B0", m) for m in PHASE1[1:]] + [("B1", m) for m in PHASE1[2:]] + [
        (s_star, "B7"), ("B6W", "B7"), ("B6", "B7"), ("B6", "B6W")] + [(m, "B7") for m in SIMPLE if m != s_star]
    pairs += [("SYS_B", "SYS_C"), ("SYS_A", "SYS_B"), ("SYS_A", "SYS_C"), ("SYS_C_ON_B", "SYS_C"),
              ("SYS_B", "SYS_B_ON_C"), ("R7", "SYS_A"), ("CORE_A", "SYS_B"), ("B0", "SYS_C"), ("R7", "CORE_A"),
              ("CORE_A", "B0"), ("SYS_C", "B7"), ("B7", "SYS_C")]
    for a, b in pairs:
        key = f"{a}_minus_{b}"
        if key not in res["contrasts"]:
            res["contrasts"][key] = {**short(contrast(table, a, b)), **summary(table, a, b)}
    # Gate A
    ca = res["contrasts"][f"{s_star}_minus_B7"]
    unc = {}
    for f in FOLDS:
        vals = {}
        for m in (s_star, "B7"):
            sn = sum(e["methods"][m]["UNC_NEAR"][0] + e["methods"][m]["UNC_FILL"][0] for e in fold_q[f])
            nn = sum(e["methods"][m]["UNC_NEAR"][1] + e["methods"][m]["UNC_FILL"][1] for e in fold_q[f])
            vals[m] = sn / max(nn, 1)
        unc[f] = vals
    checks_a = {"ci_lower_gt_0": ca["ci95"][0] > 0, "mean_gain_ge_0.005": ca["mean"] >= 0.005,
                "top1_share_le_0.5": (ca["positive_top1_contribution"] or 0) <= 0.5, "median_gain_gt_0": ca["median_gain"] > 0,
                "uncovered_lower_every_fold": all(unc[f]["B7"] < unc[f][s_star] for f in FOLDS)}
    res["gates"]["A"] = {"s_star": s_star, "contrast": ca, "uncovered_pixel_absrel_by_fold": unc, "checks": checks_a,
                         "label": "V2_METHOD_NECESSITY=" + ("SUPPORTED" if all(checks_a.values()) else "NOT_ESTABLISHED")}
    cb = res["contrasts"]["SYS_B_minus_SYS_C"]
    checks_b = {"ci_lower_gt_0": cb["ci95"][0] > 0, "mean_gain_ge_0.002": cb["mean"] >= 0.002,
                "median_gain_gt_0": cb["median_gain"] > 0,
                "top1_share_le_0.5": (cb["positive_top1_contribution"] or 0) <= 0.5}
    res["gates"]["B"] = {"contrast": cb, "checks": checks_b,
                         "label": "TTT_ADDED_VALUE=" + ("SUPPORTED" if all(checks_b.values()) else "NOT_ESTABLISHED")}
    res["contributions_vs_B1"] = {m: contribution(queries, w, "B1", m) for m in PHASE1[2:]}
    res["contributions_vs_R7"] = {m: contribution(queries, w, "R7", m) for m in ("SYS_A", "SYS_B", "SYS_C")}
    (DEST / "BOOTSTRAP_RESULTS.json").write_text(json.dumps(res, indent=1, sort_keys=True) + "\n")
    # CSV files
    rows = []
    for m in PHASE1:
        c = res["contrasts"].get(f"B0_minus_{m}")
        pa, _ = pixel_absrel(queries, m, "ALL")
        rows.append([m, f"{means[m]:.5f}"] + [f"{fold_means[f][m]:.5f}" for f in FOLDS] + [f"{pa:.5f}"] + (
            [f"{c['mean']:+.5f}", f"[{c['ci95'][0]:+.5f}, {c['ci95'][1]:+.5f}]", "/".join(map(str, c["improved_tied_worse"])),
             c["status"], f"{c['median_gain']:+.5f}", f"{c['worst10_mean_gain']:+.5f}",
             f"{c['positive_top1_contribution'] or 0:.2f}", f"[{c['loso_range'][0]:+.5f}, {c['loso_range'][1]:+.5f}]"]
            if c else ["", "", "", "", "", "", "", ""]))
    (DEST / "BASELINE_MATRIX.csv").write_text(csv_text(
        ["method", "pooled_place_absrel"] + [f"{f}_place_absrel" for f in FOLDS] + [
            "pooled_pixel_absrel", "gain_vs_B0", "ci95_vs_B0", "better/tied/worse_vs_B0", "status_vs_B0",
            "median_place_gain_vs_B0", "worst10pct_mean_gain_vs_B0", "top1_share_vs_B0", "loso_range_vs_B0"], rows))
    rows = []
    for f in FOLDS:
        for name in SIMPLE:
            p = fits[f][name]
            if name == "B5":
                desc = {"edges": p["edges"], "fitted_bins": sum(1 for q in p["params"] if q != [0.0, 0.0])}
            elif name == "B2A":
                desc = {"c_zero": p["c_zero"]}
            else:
                desc = p
            rows.append([f, name, json.dumps(desc, sort_keys=True), f"{fold_means[f][name]:.5f}"])
    (DEST / "CALIBRATION_CONTROLS.csv").write_text(csv_text(["fold", "control", "fitted_parameters", "fold_place_absrel"],
                                                            rows))
    rows = []
    for m in PHASE2:
        pa, _ = pixel_absrel(queries, m, "ALL")
        rows.append([m, f"{means[m]:.5f}"] + [f"{fold_means[f][m]:.5f}" for f in FOLDS] + [f"{pa:.5f}"])
    for key in ("SYS_B_minus_SYS_C", "SYS_A_minus_SYS_B", "SYS_A_minus_SYS_C", "SYS_C_ON_B_minus_SYS_C",
                "SYS_B_minus_SYS_B_ON_C", "CORE_A_minus_B0", "B7_minus_SYS_C"):
        c = res["contrasts"][key]
        rows.append([key, f"{c['mean']:+.5f}", f"[{c['ci95'][0]:+.5f}, {c['ci95'][1]:+.5f}]",
                     "/".join(map(str, c["improved_tied_worse"])), c["status"], f"{c['median_gain']:+.5f}",
                     f"{c['positive_top1_contribution'] or 0:.2f}"])
    (DEST / "TTT_NECESSITY_RESULTS.csv").write_text(csv_text(
        ["row", "pooled_place_absrel_or_gain", "CONF_SYSTEM3_or_ci95", "CONF_VAL_or_btw", "CAJU_or_status",
         "ARKIT_SYSTEM4_or_median", "pooled_pixel_absrel_or_top1"], rows))
    rows = []
    for scope in ("POOLED",) + FOLDS:
        qs = queries if scope == "POOLED" else fold_q[scope]
        for m in methods:
            for s in STRATA:
                v, n = pixel_absrel(qs, m, s)
                rows.append([scope, m, s, "" if v is None else f"{v:.5f}", n])
    (DEST / "REGIONAL_ERROR_BREAKDOWN.csv").write_text(csv_text(["scope", "method", "stratum", "pixel_absrel", "pixels"],
                                                                rows))
    print("GATE_A", res["gates"]["A"]["label"], "S*", s_star, json.dumps(checks_a))
    print("GATE_B", res["gates"]["B"]["label"], json.dumps(checks_b))
    print(json.dumps({m: round(v, 5) for m, v in sorted(means.items(), key=lambda kv: kv[1])}))


if __name__ == "__main__":
    main()
