"""EXP-3D-RGBD-TTT-ENSEMBLE-V1: scoring and the preregistered analysis on EVAL34 (7-frame setting).

score: the geometry seal (REPROJ7_HARMONIC, NEAR7, cosine; replayed against the arms seal) and the DepthSplat seal of
the released model (ZS: CACHE and TTT_m1) must be complete and unchanged; only then is query depth read. Per query:
  R7          REPROJ7_HARMONIC
  CACHE, TTT  ray depth (depth_z / alpha / cos) where alpha >= 0.5, else R7          (V1's _F7 readouts)
  R7_TTT_GEO  sqrt(R7 x TTT)          R7_TTT  (R7 + TTT) / 2
  R7_CACHE_GEO sqrt(R7 x CACHE)       R7_CACHE (R7 + CACHE) / 2
AbsRel on ALL (valid ground truth), NEAR7 and FAR7. Scene value = mean over roles of the mean over queries.
analyze: V1's paired scene bootstrap (10,000 draws, seed 20260928) and V2 gate.
Run through cv_host_run.py from the repository root (CUDA hidden):  python tttens_score.py ROOT score | analyze
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

COHORT, ALPHA_MIN = "EVAL34", 0.5


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(item):
    if sha(item["path"]) != item["sha256"]:
        raise PermissionError(f"sealed prediction changed: {item['path']}")
    return dict(np.load(item["path"]))


def score(root):
    lock = json.loads((root / "frame_role_lock.json").read_text())["cohorts"][COHORT]
    if sha(lock["manifest"]) != lock["manifest_sha256"]:
        raise PermissionError("manifest changed since the frame-role lock")
    records = {r["scene_id"]: r for r in json.loads(Path(lock["manifest"]).read_text())["scenes"]}
    keys = [f"{e['scene_id']}/{e['role']}/{q}" for e in lock["episodes"] for q in e["queries"]]
    geometry = json.loads((root / f"geometry/geometry_seal_{COHORT}.json").read_text())
    if not geometry["replay_vs_arms_seal"]["pass"] or set(geometry["predictions"]) != set(keys):
        raise PermissionError("geometry seal incomplete or its replay failed")
    zs = {}
    for part in sorted((root / f"ds/{COHORT}/ZS").glob("ds_seal_*of*.json")):
        seal = json.loads(part.read_text())
        if seal["query_depth_read"] is not False or seal["limit"] or seal["cohort"] != COHORT:
            raise PermissionError(f"{part}: not a complete sealed run")
        zs.update(seal["predictions"])
    if set(zs) != set(keys):
        raise ValueError("ZS seal differs from the frame-role lock")
    rows, access, started = [], [], time.time()
    for e in lock["episodes"]:
        frames = {f["frame_id"]: f for f in records[e["scene_id"]]["frames"]}
        for qid in e["queries"]:
            key = f"{e['scene_id']}/{e['role']}/{qid}"
            geo, arrays = load(geometry["predictions"][key]), load(zs[key])
            gt = np.load(frames[qid]["depth"]).squeeze().astype(np.float64)
            access.append({"key": key, "path": frames[qid]["depth"]})
            valid = np.isfinite(gt) & (gt > 0)
            near7 = geo["near7"].astype(bool)
            r7, cosine = geo["REPROJ7_HARMONIC"], geo["cosine"]
            state = {}
            for arm in ("CACHE", "TTT_m1"):
                alpha = arrays[f"{arm}_alpha"].astype(np.float64)
                ray = arrays[f"{arm}_depth_z"].astype(np.float64) / np.maximum(alpha, 1e-12) / cosine
                state[arm] = np.where(alpha >= ALPHA_MIN, ray, r7)
            geo_mean = lambda a, b: np.sqrt(np.maximum(a, 1e-6) * np.maximum(b, 1e-6))  # noqa: E731
            readouts = {"R7": r7, "CACHE": state["CACHE"], "TTT": state["TTT_m1"],
                        "R7_TTT_GEO": geo_mean(r7, state["TTT_m1"]), "R7_TTT": (r7 + state["TTT_m1"]) / 2,
                        "R7_CACHE_GEO": geo_mean(r7, state["CACHE"]), "R7_CACHE": (r7 + state["CACHE"]) / 2}
            values = {}
            for name, p in readouts.items():
                for region, mask in (("ALL", valid), ("NEAR7", valid & near7), ("FAR7", valid & ~near7)):
                    values[f"{name}|{region}"] = (float(np.mean(np.abs(p[mask] - gt[mask]) / gt[mask]))
                                                  if mask.any() else None)
            rows.append({"scene_id": e["scene_id"], "role": e["role"], "query": qid, "values": values})
    grouped = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for row in rows:
        for name, v in row["values"].items():
            if v is not None and np.isfinite(v):
                grouped[name][row["scene_id"]][row["role"]].append(v)
    table = {n: {s: float(np.mean([np.mean(v) for v in roles.values()])) for s, roles in sc.items()}
             for n, sc in grouped.items()}
    write(root / "scores/query_rows.json", rows)
    write(root / "scores/scene_table.json", table)
    write(root / "scores/score_info.json", {"queries": len(rows), "query_depth_reads": access,
                                            "seconds": round(time.time() - started, 1)})
    print("SCORED", len(rows), "queries")


PRIMARY = {"T1_TTT_ENSEMBLE_VS_REPROJ7_ALL": ("R7|ALL", "R7_TTT_GEO|ALL"),
           "T2_TTT_NEEDED_IN_ENSEMBLE_ALL": ("R7_CACHE_GEO|ALL", "R7_TTT_GEO|ALL")}
SECONDARY = {"S1_TTT_VS_REPROJ7_ALL": ("R7|ALL", "TTT|ALL"), "S2_TTT_BEYOND_CACHE_ALL": ("CACHE|ALL", "TTT|ALL"),
             "S3_ARITH_TTT_ENSEMBLE_VS_REPROJ7_ALL": ("R7|ALL", "R7_TTT|ALL"),
             "S4_CACHE_ENSEMBLE_VS_REPROJ7_ALL": ("R7|ALL", "R7_CACHE_GEO|ALL"),
             "S5_CACHE_VS_REPROJ7_ALL": ("R7|ALL", "CACHE|ALL"),
             "S6_TTT_ENSEMBLE_VS_REPROJ7_NEAR7": ("R7|NEAR7", "R7_TTT_GEO|NEAR7"),
             "S7_TTT_ENSEMBLE_VS_REPROJ7_FAR7": ("R7|FAR7", "R7_TTT_GEO|FAR7"),
             "S8_TTT_ENSEMBLE_VS_TTT_ALL": ("TTT|ALL", "R7_TTT_GEO|ALL")}


def analyze(root):
    table = json.loads((root / "scores/scene_table.json").read_text())
    run = lambda spec: {k: contrast(table, a, b) for k, (a, b) in spec.items()}  # noqa: E731
    primary, secondary = run(PRIMARY), run(SECONDARY)
    t1, t2 = primary["T1_TTT_ENSEMBLE_VS_REPROJ7_ALL"]["status"], primary["T2_TTT_NEEDED_IN_ENSEMBLE_ALL"]["status"]
    branch_t1 = {"SUPPORTED": "TTT_ENSEMBLE_CONSISTENT", "PARTIAL": "TTT_ENSEMBLE_PARTIAL",
                 "HARMFUL": "TTT_ENSEMBLE_HARMFUL"}.get(t1, "TTT_ENSEMBLE_NOT_ESTABLISHED")
    branch_t2 = {"SUPPORTED": "TTT_NEEDED", "PARTIAL": "TTT_NEEDED_PARTIAL", "HARMFUL": "TTT_HARMFUL"}.get(
        t2, "TTT_NOT_NEEDED")
    means = {k: {"mean": float(np.mean(list(v.values()))), "scenes": len(v)} for k, v in sorted(table.items())}
    result = {"cohort": COHORT, "branch_t1": branch_t1, "branch_t2": branch_t2, "primary": primary,
              "secondary": secondary, "means": means}
    write(root / "results/EVAL34_ttt_ensemble_results.json", result)
    print("ANALYZED", branch_t1, branch_t2)
    for label, s in {**primary, **secondary}.items():
        print(f"  {label}: {s['mean']:+.4f} [{s['ci95'][0]:+.4f}, {s['ci95'][1]:+.4f}] "
              f"{s['positive']}/{s['tied']}/{s['negative']} {s['status']}")
    for k, v in means.items():
        print(f"  {k:22s} {v['mean']:.4f} ({v['scenes']})")


if __name__ == "__main__":
    root, stage = Path(sys.argv[1]).resolve(), sys.argv[2]
    {"score": score, "analyze": analyze}[stage](root)
