"""EXP-3D-RGBD-DEPTHSPLAT-JEPA-ARKIT-OVERLAP-V1: scoring and the preregistered analysis on ARKIT160_OVERLAP (the 160
ARKIT160 scans, queries = frames 6 and 7, or the next free of 9, 11, 12 where a record already uses one of them).

score: every prediction must be sealed and unchanged (arms seal: REPROJ_HARMONIC, NEAR, G1 depth and opacity for
three seeds; DepthSplat CTX seals of JO1-JO3 (overlap-aware training, this experiment), JA1-JA3 (the original ARKit
recipe, EXP-3D-RGBD-DEPTHSPLAT-JEPA-ARKIT-V1), J1-J3 (Hypersim, V1) and ZS3 (released weights, scale cap 3.0)); only
then is query depth read. Per query:
  H3      REPROJ_HARMONIC;  H3CAL: the ARKit-TRAIN calibration of amendment 1 of JEPA-ARKIT-V1 (a query without any
          context hit gets 1.05 m everywhere, otherwise H3)
  JO, JA, J   where at least two of the three seeds cover the pixel (alpha >= 0.5): the mean ray depth (depth_z /
          alpha / cos) of the covering seeds; elsewhere H3          ZS3: alpha >= 0.5 -> ray depth, else H3
  G1N     per G1 seed depth / opacity where opacity >= 0.5, else H3, mean over seeds;  HGN = (H3 + G1N) / 2
  HJO     JO covered: (H3 + JO) / 2, else H3
  JO_CAL  JO covered: JO, else H3CAL
Regions: FARH = FAR (valid pixels outside the 8-px NEAR of the context reprojection) of the queries with at least one
context hit (NEAR not empty; known without query depth); FAR of every query; NEAR. Scene value = mean over roles of the
mean over queries (empty regions skipped).
analyze: V1's paired scene bootstrap (10,000 draws, seed 20260928) and V2 gate.
Run through cv_host_run.py from the repository root (CUDA hidden):  python overlap_score.py ROOT score | analyze
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

COHORT, ALPHA_MIN, ZERO_HIT_CONSTANT = "ARKIT160_OVERLAP", 0.5, 1.05
BUILDERS = {"JO": ("JO1", "JO2", "JO3"), "JA": ("JA1", "JA2", "JA3"), "J": ("J1", "J2", "J3"), "ZS3": ("ZS3",)}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(item):
    if sha(item["path"]) != item["sha256"]:
        raise PermissionError(f"sealed prediction changed: {item['path']}")
    return dict(np.load(item["path"]))


def ds_seal(root, name, keys):
    preds = {}
    for part in sorted((root / f"ds/{COHORT}" / name).glob("ds_seal_*of*.json")):
        seal = json.loads(part.read_text())
        if seal["query_depth_read"] is not False or seal["limit"] or seal["cohort"] != COHORT or seal["scale_max"] != 3.0:
            raise PermissionError(f"{part}: not a complete sealed run")
        preds.update(seal["predictions"])
    if set(preds) != set(keys):
        raise ValueError(f"{name}: sealed queries differ from the frame-role lock")
    return preds


def ray_cosine(record, qid):
    from mcss.measurement_planning import geometry as g
    from mcss.measurement_planning.information import TeacherAccess

    teacher = TeacherAccess(record, [qid], (128, 160), Path.cwd(), [])
    return np.asarray(g.ray_cosine(teacher.query_camera(qid)), dtype=np.float64)


def pooled(arrays, cosine, fallback):
    covers = np.array([a["CTX_alpha"].astype(np.float64) >= ALPHA_MIN for a in arrays])
    planar = np.array([a["CTX_depth_z"].astype(np.float64) / np.maximum(a["CTX_alpha"].astype(np.float64), 1e-12)
                       for a in arrays])
    count = covers.sum(axis=0)
    covered = count >= (2 if len(arrays) == 3 else 1)
    value = np.where(covers, planar, 0.0).sum(axis=0) / np.maximum(count, 1) / cosine
    return np.where(covered, value, fallback), covered


def score(root):
    lock = json.loads((root / "frame_role_lock.json").read_text())["cohorts"][COHORT]
    if sha(lock["manifest"]) != lock["manifest_sha256"]:
        raise PermissionError("manifest changed since the frame-role lock")
    records = {r["scene_id"]: r for r in json.loads(Path(lock["manifest"]).read_text())["scenes"]}
    keys = [f"{e['scene_id']}/{e['role']}/{q}" for e in lock["episodes"] for q in e["queries"]]
    arms = json.loads((root / f"raw/eval/{COHORT}/prediction_seal.json").read_text())
    if arms["query_depth_read"] is not False or any(f"{COHORT}/{k}" not in arms["predictions"] for k in keys):
        raise PermissionError("arms seal incomplete or not certified")
    seals = {seed: ds_seal(root, seed, keys) for seeds in BUILDERS.values() for seed in seeds}
    rows, access, started = [], [], time.time()
    for e in lock["episodes"]:
        record = records[e["scene_id"]]
        frames = {f["frame_id"]: f for f in record["frames"]}
        for qid in e["queries"]:
            key = f"{e['scene_id']}/{e['role']}/{qid}"
            p = load(arms["predictions"][f"{COHORT}/{key}"])
            cosine = ray_cosine(record, qid)
            near = p["near"].astype(bool)
            zero_hit = not near.any()
            h3 = p["REPROJ_HARMONIC"]
            h3cal = np.full(h3.shape, ZERO_HIT_CONSTANT) if zero_hit else h3
            g1n = np.mean([np.where(p[f"OPACITY_G1_{s}"] >= 0.5,
                                    p[f"G1_{s}"] / np.maximum(p[f"OPACITY_G1_{s}"].astype(np.float64), 1e-6), h3)
                           for s in arms["seeds"]], axis=0)
            built, cover = {}, {}
            for name, seeds in BUILDERS.items():
                built[name], cover[name] = pooled([load(seals[s][key]) for s in seeds], cosine, h3)
            gt = np.load(frames[qid]["depth"]).squeeze().astype(np.float64)
            access.append({"key": key, "path": frames[qid]["depth"]})
            valid = np.isfinite(gt) & (gt > 0)
            readouts = {"H3": h3, "H3CAL": h3cal, "JO": built["JO"], "JA": built["JA"], "J": built["J"],
                        "ZS3": built["ZS3"], "G1N": g1n, "HGN": (h3 + g1n) / 2,
                        "HJO": np.where(cover["JO"], (h3 + built["JO"]) / 2, h3),
                        "JO_CAL": np.where(cover["JO"], built["JO"], h3cal)}
            far = valid & ~near
            regions = {"FARH": far if not zero_hit else np.zeros_like(far), "FAR": far, "NEAR": valid & near}
            values = {"zero_hit": zero_hit}
            for n in BUILDERS:
                values[f"{n}_farh_coverage"] = float(cover[n][far].mean()) if far.any() and not zero_hit else None
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
    write(root / "scores/query_rows.json", rows)
    write(root / "scores/scene_table.json", table)
    write(root / "scores/score_info.json", {"queries": len(rows), "zero_hit_queries": sum(r["values"]["zero_hit"] for r in rows),
                                            "query_depth_reads": access, "seconds": round(time.time() - started, 1)})
    print("SCORED", len(rows), "queries;", sum(r["values"]["zero_hit"] for r in rows), "without a context hit")


PRIMARY = {"O1_OVERLAP_JEPA_VS_HARMONIC_FARH": ("H3|FARH", "JO|FARH"),
           "O2_OVERLAP_TRAINING_VS_ORIGINAL_RECIPE_FARH": ("JA|FARH", "JO|FARH")}
SECONDARY = {"S1_ORIGINAL_RECIPE_VS_HARMONIC_FARH": ("H3|FARH", "JA|FARH"),
             "S2_SYNTHETIC_JEPA_VS_HARMONIC_FARH": ("H3|FARH", "J|FARH"),
             "S3_OVERLAP_TRAINING_GAIN_SAME_CAP_FARH": ("ZS3|FARH", "JO|FARH"),
             "S4_G1N_ENSEMBLE_VS_HARMONIC_FARH": ("H3|FARH", "HGN|FARH"),
             "S5_OVERLAP_JEPA_ENSEMBLE_VS_HARMONIC_FARH": ("H3|FARH", "HJO|FARH"),
             "S6_OVERLAP_JEPA_VS_HARMONIC_NEAR": ("H3|NEAR", "JO|NEAR"),
             "S7_OVERLAP_JEPA_VS_CALIBRATED_HARMONIC_FAR_ALL_QUERIES": ("H3CAL|FAR", "JO_CAL|FAR"),
             "S8_OVERLAP_JEPA_VS_G1N_FARH": ("G1N|FARH", "JO|FARH")}


def label(status, names):
    return {"SUPPORTED": names[0], "PARTIAL": names[1], "HARMFUL": names[3]}.get(status, names[2])


def analyze(root):
    table = json.loads((root / "scores/scene_table.json").read_text())
    primary = {k: contrast(table, a, b) for k, (a, b) in PRIMARY.items()}
    secondary = {k: contrast(table, a, b) for k, (a, b) in SECONDARY.items()}
    branches = {
        "O1": label(primary["O1_OVERLAP_JEPA_VS_HARMONIC_FARH"]["status"],
                    ("OVERLAP_JEPA_BEATS_HARMONIC", "OVERLAP_JEPA_PARTIAL", "OVERLAP_JEPA_NOT_ESTABLISHED",
                     "OVERLAP_JEPA_HARMFUL")),
        "O2": label(primary["O2_OVERLAP_TRAINING_VS_ORIGINAL_RECIPE_FARH"]["status"],
                    ("OVERLAP_TRAINING_HELPS", "OVERLAP_TRAINING_PARTIAL", "OVERLAP_TRAINING_NOT_ESTABLISHED",
                     "OVERLAP_TRAINING_HARMFUL")),
    }
    rows = json.loads((root / "scores/query_rows.json").read_text())
    cohort_of = {r["scene_id"]: r["source_cohort"] for r in rows}
    by_cohort = {c: {k: float(np.mean([v for s, v in table[k].items() if cohort_of[s] == c]))
                     for k in ("H3|FARH", "JO|FARH", "JA|FARH", "J|FARH", "HGN|FARH", "HJO|FARH")}
                 for c in sorted(set(cohort_of.values()))}
    means = {k: {"mean": float(np.mean(list(v.values()))), "scenes": len(v)} for k, v in sorted(table.items())}
    write(root / "results/ARKIT160_OVERLAP_results.json", {"cohort": COHORT, "branches": branches, "primary": primary,
                                                           "secondary": secondary, "means": means,
                                                           "by_cohort": by_cohort})
    print("ANALYZED", branches)
    for name, s in {**primary, **secondary}.items():
        print(f"  {name}: {s['mean']:+.4f} [{s['ci95'][0]:+.4f}, {s['ci95'][1]:+.4f}] "
              f"{s['positive']}/{s['tied']}/{s['negative']} ({s['scenes']} scenes) {s['status']}")
    for k, v in means.items():
        if "|FARH" in k or "coverage" in k:
            print(f"  {k:22s} {v['mean']:.4f} ({v['scenes']})")
    print("  by cohort:", json.dumps(by_cohort))


if __name__ == "__main__":
    root, stage = Path(sys.argv[1]).resolve(), sys.argv[2]
    {"score": score, "analyze": analyze}[stage](root)
