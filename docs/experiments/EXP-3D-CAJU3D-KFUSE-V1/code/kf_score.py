"""EXP-3D-CAJU3D-KFUSE-V1: fitting the fusion tables and the preregistered confirmation (CPU).

  fit      FIT_SYSTEM + FIT_SYSTEM2 (160 places, one fitting set): quantile edges of DIS / ALPHA / LOGP on the covered
           pixels, per arm and bin the weight minimizing the place-weighted AbsRel; fusion_tables.json is written and
           its hash logged before any confirmation truth is read. Descriptive: leave-one-place-out estimates on FIT.
  confirm  CONF_SYSTEM3 (P1) and CONF_VAL (P2) with the frozen tables: P = KF_BASE - KF_JEPA (> 0: the H-JEPA's
           predicted variance is the better per-pixel cue than the TTT alpha); secondaries against CORE_B and GEO_U.
V1's paired place bootstrap + V2 gate (score_analyze.contrast); place = mean over episodes of the mean over queries.
Run through cv_host_run.py from the repository root (CUDA hidden):  python kf_score.py ROOT fit|confirm
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import kf_data as D  # noqa: E402  (sets the import paths)
import kf_logic as K  # noqa: E402
from score_analyze import contrast  # noqa: E402

S = D.S
FIT = ("FIT_SYSTEM", "FIT_SYSTEM2")
CONF = ("CONF_SYSTEM3", "CONF_VAL")
NAMES = ("CORE_B",) + tuple(K.ARMS)
SECONDARY = (("CORE_B", "KF_BASE"), ("CORE_B", "KF_JEPA"), ("CORE_B", "KF_FULL"), ("CORE_B", "GEO_U"),
             ("CORE_B", "KF_DIS"), ("GEO_U", "KF_BASE"), ("KF_BASE", "KF_FULL"), ("KF_DIS", "KF_BASE"))


def rows_of(queries, predict):
    """predict(q, arm) -> prediction on q's valid pixels; rows in engine_score's format (metric 'core_b')."""
    rows = []
    for q in queries:
        rows.append({"place": q["place"], "episode": q["episode"], "query": q["key"],
                     "core_b": {a: float(np.mean(np.abs(predict(q, a) - q["gt"]) / q["gt"])) for a in NAMES}})
    return rows


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=1, sort_keys=True) + "\n")


def stage_fit(root):
    for name in ("fusion_tables.json", "results/confirm_results.json"):
        if (root / name).exists():
            raise SystemExit(f"{name} exists already")
    access = []
    queries = [q for c in FIT for q in D.load_cohort(root, c, access)]
    covered = [K.covered_features(q)[1] for q in queries]
    edges = {n: K.quantile_edges(np.concatenate([f[n] for f in covered]), b).tolist()
             for n, b in K.FEATURE_BINS.items()}
    loss = K.accumulate(queries, edges)
    tables = {"edges": edges, "weights": {a: K.fit_weights(loss[a]).tolist() for a in K.ARMS},
              "weight_grid": K.WEIGHTS.tolist(), "arms": {a: list(v) for a, v in K.ARMS.items()},
              "fit_cohorts": list(FIT), "fit_places": len({q["place"] for q in queries})}
    # descriptive: leave-one-place-out on FIT
    by_place = {}
    for q in queries:
        by_place.setdefault(q["place"], []).append(q)
    loo_rows = []
    for place, members in sorted(by_place.items()):
        held = {"edges": edges, "weights": {a: K.fit_weights(loss[a], exclude=place).tolist() for a in K.ARMS}}
        loo_rows += rows_of(members, lambda q, a: K.readout(q, a, held))
    table = S.place_table(loo_rows)
    stats = {f"{l}_minus_{r}": contrast(table, l, r) for l, r in (("KF_BASE", "KF_JEPA"),) + SECONDARY}
    write(root / "fusion_tables.json", tables)
    (root / "audit").mkdir(exist_ok=True)
    with (root / "audit/fit_log.txt").open("a") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S %z')} {D.sha(root / 'fusion_tables.json')}\n")
    write(root / "results/fit_loo_results.json", {"places": len(by_place), "queries": len(queries),
                                                   "loo_means": {k: S.mean(v) for k, v in table.items()},
                                                   "loo_contrasts": stats, "ground_truth_reads": len(access)})
    print("KF_FIT", json.dumps({k: round(S.mean(v), 5) for k, v in table.items()}))


def stage_confirm(root):
    tables = json.loads((root / "fusion_tables.json").read_text())
    if D.sha(root / "fusion_tables.json") != (root / "audit/fit_log.txt").read_text().split()[-1]:
        raise PermissionError("fusion_tables.json changed since its hash was logged")
    results = {}
    for cohort in CONF:
        access = []
        queries = D.load_cohort(root, cohort, access)
        rows = rows_of(queries, lambda q, a: K.readout(q, a, tables))
        table = S.place_table(rows)
        results[cohort] = {"places": len(table["CORE_B"]), "queries": len(queries),
                           "means": {k: S.mean(v) for k, v in table.items()},
                           "primary_base_minus_jepa": contrast(table, "KF_BASE", "KF_JEPA"),
                           "secondary": {f"{l}_minus_{r}": contrast(table, l, r) for l, r in SECONDARY},
                           "ground_truth_reads": len(access)}
        write(root / f"scores/{cohort}/rows.json", rows)
    results["branch"] = K.branch(results["CONF_SYSTEM3"]["primary_base_minus_jepa"],
                                 results["CONF_VAL"]["primary_base_minus_jepa"])
    write(root / "results/confirm_results.json", results)
    print("CONFIRM", results["branch"])


def main():
    root, stage = Path(sys.argv[1]).resolve(), sys.argv[2]
    {"fit": stage_fit, "confirm": stage_confirm}[stage](root)


if __name__ == "__main__":
    main()
