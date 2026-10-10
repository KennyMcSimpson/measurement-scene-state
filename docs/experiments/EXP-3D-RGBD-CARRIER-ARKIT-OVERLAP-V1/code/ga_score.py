"""EXP-3D-RGBD-CARRIER-ARKIT-OVERLAP-V1: scoring and the preregistered analysis on ARKIT160_OVERLAP.

score: the arms seal (REPROJ_HARMONIC, NEAR, depth and opacity of G1 and GA for three seeds each) must be complete and
unchanged before any query depth is read. Per query:
  H3      REPROJ_HARMONIC;  H3CAL: a query without any context hit (empty NEAR) gets 1.05 m everywhere, otherwise H3
  G1N, GAN  per seed depth / opacity where opacity >= 0.5, else H3; mean over the three seeds
  HGN = (H3 + G1N) / 2;  HGAN = (H3 + GAN) / 2;  GAN_CAL: as GAN with H3CAL as the fallback
Regions: FARH = FAR (valid pixels outside the 8-px NEAR) of the queries with at least one context hit; FAR of every
query; NEAR. Abstention = share of FARH pixels with opacity < 0.5, mean over seeds. Scene value = mean over roles of
the mean over queries (empty regions skipped).
analyze: V1's paired scene bootstrap (10,000 draws, seed 20260928) and V2 gate.
Run through cv_host_run.py from the repository root (CUDA hidden):  python ga_score.py ROOT score | analyze
"""
import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "EXP-3D-RGBD-DEPTHSPLAT-JEPA-TTT-V1" / "code"))
from score_analyze import contrast, write  # noqa: E402

COHORT, OPACITY_MIN, ZERO_HIT_CONSTANT = "ARKIT160_OVERLAP", 0.5, 1.05


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(item):
    if sha(item["path"]) != item["sha256"]:
        raise PermissionError(f"sealed prediction changed: {item['path']}")
    return dict(np.load(item["path"]))


def normalized(p, arm, seeds, fallback):
    return np.mean([np.where(p[f"OPACITY_{arm}_{s}"] >= OPACITY_MIN,
                             p[f"{arm}_{s}"] / np.maximum(p[f"OPACITY_{arm}_{s}"].astype(np.float64), 1e-6), fallback)
                    for s in seeds], axis=0)


def score(root):
    lock = json.loads((root / "frame_role_lock.json").read_text())["cohorts"][COHORT]
    if sha(lock["manifest"]) != lock["manifest_sha256"]:
        raise PermissionError("manifest changed since the frame-role lock")
    records = {r["scene_id"]: r for r in json.loads(Path(lock["manifest"]).read_text())["scenes"]}
    keys = [f"{e['scene_id']}/{e['role']}/{q}" for e in lock["episodes"] for q in e["queries"]]
    arms = json.loads((root / f"raw/eval/{COHORT}/prediction_seal.json").read_text())
    if arms["query_depth_read"] is not False or any(f"{COHORT}/{k}" not in arms["predictions"] for k in keys):
        raise PermissionError("arms seal incomplete or not certified")
    rows, access, started = [], [], time.time()
    for e in lock["episodes"]:
        record = records[e["scene_id"]]
        frames = {f["frame_id"]: f for f in record["frames"]}
        for qid in e["queries"]:
            key = f"{e['scene_id']}/{e['role']}/{qid}"
            p = load(arms["predictions"][f"{COHORT}/{key}"])
            near = p["near"].astype(bool)
            zero_hit = not near.any()
            h3 = p["REPROJ_HARMONIC"]
            h3cal = np.full(h3.shape, ZERO_HIT_CONSTANT) if zero_hit else h3
            g1n, gan = normalized(p, "G1", arms["seeds"], h3), normalized(p, "GA", arms["seeds"], h3)
            gt = np.load(frames[qid]["depth"]).squeeze().astype(np.float64)
            access.append({"key": key, "path": frames[qid]["depth"]})
            valid = np.isfinite(gt) & (gt > 0)
            readouts = {"H3": h3, "H3CAL": h3cal, "G1N": g1n, "GAN": gan, "HGN": (h3 + g1n) / 2,
                        "HGAN": (h3 + gan) / 2, "GAN_CAL": normalized(p, "GA", arms["seeds"], h3cal)}
            far = valid & ~near
            farh = far if not zero_hit else np.zeros_like(far)
            values = {"zero_hit": zero_hit}
            for arm in ("G1", "GA"):
                values[f"{arm}_farh_abstain"] = (float(np.mean([np.mean(p[f"OPACITY_{arm}_{s}"][farh] < OPACITY_MIN)
                                                                 for s in arms["seeds"]])) if farh.any() else None)
            for name, x in readouts.items():
                for region, mask in (("FARH", farh), ("FAR", far), ("NEAR", valid & near)):
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
    write(root / "scores/query_rows.json", rows)
    write(root / "scores/scene_table.json", table)
    write(root / "scores/score_info.json", {"queries": len(rows), "zero_hit_queries": sum(r["values"]["zero_hit"] for r in rows),
                                            "query_depth_reads": access, "seconds": round(time.time() - started, 1)})
    print("SCORED", len(rows), "queries;", sum(r["values"]["zero_hit"] for r in rows), "without a context hit")


PRIMARY = {"C1_REAL_CARRIER_VS_HARMONIC_FARH": ("H3|FARH", "GAN|FARH"),
           "C2_REAL_DATA_VS_HYPERSIM_CARRIER_FARH": ("G1N|FARH", "GAN|FARH")}
SECONDARY = {"S1_REAL_CARRIER_ENSEMBLE_VS_HARMONIC_FARH": ("H3|FARH", "HGAN|FARH"),
             "S2_REAL_VS_HYPERSIM_ENSEMBLE_FARH": ("HGN|FARH", "HGAN|FARH"),
             "S3_HYPERSIM_CARRIER_VS_HARMONIC_FARH": ("H3|FARH", "G1N|FARH"),
             "S4_HYPERSIM_ENSEMBLE_VS_HARMONIC_FARH": ("H3|FARH", "HGN|FARH"),
             "S5_REAL_CARRIER_VS_CALIBRATED_HARMONIC_FAR": ("H3CAL|FAR", "GAN_CAL|FAR"),
             "S6_REAL_CARRIER_VS_HARMONIC_NEAR": ("H3|NEAR", "GAN|NEAR")}


def label(status, names):
    return {"SUPPORTED": names[0], "PARTIAL": names[1], "HARMFUL": names[3]}.get(status, names[2])


def analyze(root):
    table = json.loads((root / "scores/scene_table.json").read_text())
    primary = {k: contrast(table, a, b) for k, (a, b) in PRIMARY.items()}
    secondary = {k: contrast(table, a, b) for k, (a, b) in SECONDARY.items()}
    branches = {
        "C1": label(primary["C1_REAL_CARRIER_VS_HARMONIC_FARH"]["status"],
                    ("REAL_CARRIER_BEATS_HARMONIC", "REAL_CARRIER_PARTIAL", "REAL_CARRIER_NOT_ESTABLISHED",
                     "REAL_CARRIER_HARMFUL")),
        "C2": label(primary["C2_REAL_DATA_VS_HYPERSIM_CARRIER_FARH"]["status"],
                    ("REAL_DATA_HELPS_CARRIER", "REAL_DATA_HELPS_CARRIER_PARTIAL", "REAL_DATA_CARRIER_NOT_ESTABLISHED",
                     "REAL_DATA_HARMS_CARRIER")),
    }
    rows = json.loads((root / "scores/query_rows.json").read_text())
    cohort_of = {r["scene_id"]: r["source_cohort"] for r in rows}
    by_cohort = {c: {k: float(np.mean([v for s, v in table[k].items() if cohort_of[s] == c]))
                     for k in ("H3|FARH", "G1N|FARH", "GAN|FARH", "HGN|FARH", "HGAN|FARH")}
                 for c in sorted(set(cohort_of.values()))}
    means = {k: {"mean": float(np.mean(list(v.values()))), "scenes": len(v)} for k, v in sorted(table.items())}
    write(root / "results/ARKIT160_OVERLAP_carrier_results.json", {"cohort": COHORT, "branches": branches,
                                                                   "primary": primary, "secondary": secondary,
                                                                   "means": means, "by_cohort": by_cohort})
    print("ANALYZED", branches)
    for name, s in {**primary, **secondary}.items():
        print(f"  {name}: {s['mean']:+.4f} [{s['ci95'][0]:+.4f}, {s['ci95'][1]:+.4f}] "
              f"{s['positive']}/{s['tied']}/{s['negative']} ({s['scenes']} scenes) {s['status']}")
    for k, v in means.items():
        if "|FARH" in k or "abstain" in k:
            print(f"  {k:22s} {v['mean']:.4f} ({v['scenes']})")
    print("  by cohort:", json.dumps(by_cohort))


if __name__ == "__main__":
    root, stage = Path(sys.argv[1]).resolve(), sys.argv[2]
    {"score": score, "analyze": analyze}[stage](root)
