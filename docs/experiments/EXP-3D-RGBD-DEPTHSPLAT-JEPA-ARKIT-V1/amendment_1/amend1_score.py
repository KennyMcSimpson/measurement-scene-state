"""EXP-3D-RGBD-DEPTHSPLAT-JEPA-ARKIT-V1, amendment 1 scoring. Runs only after the locked score stage (it requires
scores/score_info.json) and repeats its checks: every seal complete and unchanged before query depth is read.

H3CAL  the TRAIN-fitted calibration of the harmonic (amendment_1/h3_calibration_TRAIN.json, sha-checked against
       audit/amendment_1.json): a query without any context hit (empty NEAR) gets the constant c_zero_hit_m; a query
       with hits keeps H3 in NEAR and gets k_hit x H3 outside NEAR
JA_CAL JA where at least two of the three JA seeds cover the pixel (the locked pooling), else H3CAL
Contrasts (FAR, V2 labels; added by amendment 1 before any ARKIT160 query depth was read for this experiment;
they do not change R1, R2 or S1-S9):
  X1 H3CAL - JA_CAL   does real-data JEPA add information beyond the calibrated harmonic?
  X2 H3 - H3CAL       does the TRAIN-fitted calibration alone help on ARKIT160?
  X3 H3CAL - JA       the locked JA readout against the calibrated harmonic
  X4 H3CAL - HGN      the normalized G1 ensemble against the calibrated harmonic
  X5 H3CAL - JA       on the FAR pixels JA covers only (descriptive weight)
  X6 H3 - JA          on FAR of the queries with at least one context hit only (descriptive weight)
Run through cv_host_run.py from the repository root (CUDA hidden):  python amend1_score.py ROOT score | analyze
"""
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from jepa_arkit_score import BUILDERS, COHORT, contrast, ds_seal, load, pooled, ray_cosine, sha, write  # noqa: E402

CONTRASTS = {"X1_REAL_JEPA_BEYOND_CALIBRATED_HARMONIC_FAR": ("H3CAL|FAR", "JA_CAL|FAR"),
             "X2_CALIBRATION_ALONE_FAR": ("H3|FAR", "H3CAL|FAR"),
             "X3_LOCKED_JA_VS_CALIBRATED_HARMONIC_FAR": ("H3CAL|FAR", "JA|FAR"),
             "X4_G1N_ENSEMBLE_VS_CALIBRATED_HARMONIC_FAR": ("H3CAL|FAR", "HGN|FAR"),
             "X5_REAL_JEPA_VS_CALIBRATED_HARMONIC_JA_COVERED_FAR": ("H3CAL|FAR_JA_COVERED", "JA|FAR_JA_COVERED"),
             "X6_REAL_JEPA_VS_HARMONIC_FAR_OF_QUERIES_WITH_HITS": ("H3|FAR_HIT_QUERIES", "JA|FAR_HIT_QUERIES")}


def calibration(root):
    amendment = json.loads((root / "audit/amendment_1.json").read_text())
    path = root / "amendment_1/h3_calibration_TRAIN.json"
    if sha(path) != amendment["calibration_sha256"]:
        raise PermissionError("calibration changed since amendment 1")
    fitted = json.loads(path.read_text())
    if (fitted["k_hit"], fitted["c_zero_hit_m"]) != (amendment["k_hit"], amendment["c_zero_hit_m"]):
        raise PermissionError("calibration differs from amendment 1")
    return float(fitted["k_hit"]), float(fitted["c_zero_hit_m"])


def score(root):
    if not (root / "scores/score_info.json").exists():
        raise PermissionError("the locked score stage has not run")
    k_hit, c_zero = calibration(root)
    lock = json.loads((root / "frame_role_lock.json").read_text())["cohorts"][COHORT]
    if sha(lock["manifest"]) != lock["manifest_sha256"]:
        raise PermissionError("manifest changed since the frame-role lock")
    records = {r["scene_id"]: r for r in json.loads(Path(lock["manifest"]).read_text())["scenes"]}
    keys = [f"{e['scene_id']}/{e['role']}/{q}" for e in lock["episodes"] for q in e["queries"]]
    arms = json.loads((root / f"raw/eval/{COHORT}/prediction_seal.json").read_text())
    if arms["query_depth_read"] is not False or any(f"{COHORT}/{key}" not in arms["predictions"] for key in keys):
        raise PermissionError("arms seal incomplete or not certified")
    seals = {seed: ds_seal(root, seed, keys) for seed in BUILDERS["JA"]}
    rows, access, started = [], [], time.time()
    for e in lock["episodes"]:
        record = records[e["scene_id"]]
        frames = {f["frame_id"]: f for f in record["frames"]}
        for qid in e["queries"]:
            key = f"{e['scene_id']}/{e['role']}/{qid}"
            p = load(arms["predictions"][f"{COHORT}/{key}"])
            cosine = ray_cosine(record, qid)
            gt = np.load(frames[qid]["depth"]).squeeze().astype(np.float64)
            access.append({"key": key, "path": frames[qid]["depth"]})
            valid = np.isfinite(gt) & (gt > 0)
            near = p["near"].astype(bool)
            h3 = p["REPROJ_HARMONIC"]
            zero_hit = not near.any()
            h3cal = np.full(h3.shape, c_zero) if zero_hit else np.where(near, h3, k_hit * h3)
            g1n = np.mean([np.where(p[f"OPACITY_G1_{s}"] >= 0.5,
                                    p[f"G1_{s}"] / np.maximum(p[f"OPACITY_G1_{s}"].astype(np.float64), 1e-6), h3)
                           for s in arms["seeds"]], axis=0)
            arrays = [load(seals[s][key]) for s in BUILDERS["JA"]]
            ja, cover = pooled(arrays, cosine, h3)
            ja_cal, _ = pooled(arrays, cosine, h3cal)
            readouts = {"H3": h3, "H3CAL": h3cal, "JA": ja, "JA_CAL": ja_cal, "HGN": (h3 + g1n) / 2}
            far = valid & ~near
            values = {"JA_far_coverage": float(cover[far].mean()) if far.any() else None, "zero_hit": zero_hit}
            regions = {"FAR": far, "FAR_JA_COVERED": far & cover,
                       "FAR_HIT_QUERIES": far if not zero_hit else np.zeros_like(far)}
            for name, x in readouts.items():
                for region, mask in regions.items():
                    values[f"{name}|{region}"] = (float(np.mean(np.abs(x[mask] - gt[mask]) / gt[mask]))
                                                  if mask.any() else None)
            rows.append({"scene_id": e["scene_id"], "role": e["role"], "query": qid,
                         "source_cohort": record["source_cohort"], "values": values})
    grouped = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for row in rows:
        for name, v in row["values"].items():
            if v is not None and not isinstance(v, bool) and np.isfinite(v):
                grouped[name][row["scene_id"]][row["role"]].append(v)
    table = {n: {s: float(np.mean([np.mean(v) for v in r.values()])) for s, r in sc.items()} for n, sc in grouped.items()}
    out = root / "amendment_1"
    write(out / "query_rows.json", rows)
    write(out / "scene_table.json", table)
    write(out / "score_info.json", {"queries": len(rows), "k_hit": k_hit, "c_zero_hit_m": c_zero,
                                    "zero_hit_queries": sum(r["values"]["zero_hit"] for r in rows),
                                    "query_depth_reads": access, "seconds": round(time.time() - started, 1)})
    print("AMENDMENT_1_SCORED", len(rows), "queries; k_hit", k_hit, "c_zero_hit_m", c_zero)


def analyze(root):
    out = root / "amendment_1"
    table = json.loads((out / "scene_table.json").read_text())
    results = {name: contrast(table, a, b) for name, (a, b) in CONTRASTS.items()}
    rows = json.loads((out / "query_rows.json").read_text())
    cohort_of = {r["scene_id"]: r["source_cohort"] for r in rows}
    by_cohort = {c: {k: float(np.mean([v for s, v in table[k].items() if cohort_of[s] == c]))
                     for k in ("H3|FAR", "H3CAL|FAR", "JA|FAR", "JA_CAL|FAR", "HGN|FAR")}
                 for c in sorted(set(cohort_of.values()))}
    means = {k: {"mean": float(np.mean(list(v.values()))), "scenes": len(v)} for k, v in sorted(table.items())}
    k_hit, c_zero = calibration(root)
    write(out / "ARKIT160_amendment_1_results.json", {
        "cohort": COHORT, "k_hit": k_hit, "c_zero_hit_m": c_zero, "contrasts": results, "means": means,
        "by_cohort": by_cohort, "zero_hit_queries": sum(r["values"]["zero_hit"] for r in rows), "queries": len(rows)})
    print("AMENDMENT_1_ANALYZED k_hit", k_hit, "c_zero_hit_m", c_zero,
          "zero-hit queries", sum(r["values"]["zero_hit"] for r in rows), "of", len(rows))
    for name, s in results.items():
        print(f"  {name}: {s['mean']:+.4f} [{s['ci95'][0]:+.4f}, {s['ci95'][1]:+.4f}] "
              f"{s['positive']}/{s['tied']}/{s['negative']} ({s['scenes']} scenes) {s['status']}")
    for k, v in means.items():
        print(f"  {k:28s} {v['mean']:.4f} ({v['scenes']})")
    print("  by cohort:", json.dumps(by_cohort))


if __name__ == "__main__":
    root, stage = Path(sys.argv[1]).resolve(), sys.argv[2]
    {"score": score, "analyze": analyze}[stage](root)
