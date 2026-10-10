"""EXP-3D-RGBD-DEPTHSPLAT-JEPA-BASE-TTT-ARKIT-V1: scoring and the preregistered analysis on ARKIT_VAL.

score: every prediction must be sealed and unchanged: reused (hashes in reused_seals.json) the 7-frame geometry seal
and the released-weights ZS seals (scale cap 0.1) of EXP-3D-RGBD-ENSEMBLES-ARKIT-V1; new the JA (JA1 of
EXP-3D-RGBD-DEPTHSPLAT-JEPA-ARKIT-V1) and ZS3 seals (scale cap 3.0). Only then is query depth read. Per query, the F7
readout of every state (CACHE, TTT_m1): ray depth (depth_z / alpha / cos) where alpha >= 0.5, else REPROJ7_HARMONIC
(R7); GEO_X = sqrt(R7 x X). AbsRel on ALL, NEAR7 and FAR7; coverage = share of valid pixels with alpha >= 0.5.
Scene value = mean over roles of the mean over queries (empty regions skipped).
analyze: V1's paired scene bootstrap (10,000 draws, seed 20260928) and V2 gate, plus a replication check of the
reused ZS / R7 means against EXP-3D-RGBD-ENSEMBLES-ARKIT-V1.
Run through cv_host_run.py from the repository root (CUDA hidden):  python jbt_score.py ROOT score | analyze
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

COHORT, ALPHA_MIN = "ARKIT_VAL", 0.5
CAPS = {"JA": 3.0, "ZS3": 3.0, "ZS": 0.1}
ARMS = {"CACHE": "CACHE", "TTT_m1": "TTT"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(item):
    if sha(item["path"]) != item["sha256"]:
        raise PermissionError(f"sealed prediction changed: {item['path']}")
    return dict(np.load(item["path"]))


def sealed(parts, keys, name):
    preds = {}
    for part in parts:
        seal = json.loads(Path(part).read_text())
        if seal["query_depth_read"] is not False or seal["limit"] or seal["cohort"] != COHORT:
            raise PermissionError(f"{part}: not a complete sealed run")
        if seal["scale_max"] != CAPS[name] or seal["ttt_mult"] != [1.0] or seal["ttt_steps"] != 100:
            raise PermissionError(f"{part}: settings differ from the protocol")
        preds.update(seal["predictions"])
    if set(preds) != set(keys):
        raise ValueError(f"{name}: sealed queries differ from the frame-role lock")
    return preds


def score(root):
    lock = json.loads((root / "frame_role_lock.json").read_text())["cohorts"][COHORT]
    if sha(lock["manifest"]) != lock["manifest_sha256"]:
        raise PermissionError("manifest changed since the frame-role lock")
    records = {r["scene_id"]: r for r in json.loads(Path(lock["manifest"]).read_text())["scenes"]}
    keys = [f"{e['scene_id']}/{e['role']}/{q}" for e in lock["episodes"] for q in e["queries"]]
    reused = json.loads((root / "reused_seals.json").read_text())
    for item in [reused["geometry_seal"], *reused["zs_seals"]]:
        if sha(item["path"]) != item["sha256"]:
            raise PermissionError(f"a reused seal changed: {item['path']}")
    geometry = json.loads(Path(reused["geometry_seal"]["path"]).read_text())
    if not geometry["replay_vs_arms_seal"]["pass"] or set(geometry["predictions"]) != set(keys):
        raise PermissionError("geometry seal incomplete or its replay failed")
    seals = {"ZS": sealed([i["path"] for i in reused["zs_seals"]], keys, "ZS")}
    for name in ("JA", "ZS3"):
        seals[name] = sealed(sorted((root / f"ds/{COHORT}/{name}").glob("ds_seal_*of*.json")), keys, name)
    gm = lambda *xs: np.exp(np.mean([np.log(np.maximum(x, 1e-6)) for x in xs], axis=0))  # noqa: E731
    rows, access, started = [], [], time.time()
    for e in lock["episodes"]:
        frames = {f["frame_id"]: f for f in records[e["scene_id"]]["frames"]}
        for qid in e["queries"]:
            key = f"{e['scene_id']}/{e['role']}/{qid}"
            geo = load(geometry["predictions"][key])
            r7, near7, cosine = geo["REPROJ7_HARMONIC"], geo["near7"].astype(bool), geo["cosine"]
            arrays = {name: load(seals[name][key]) for name in CAPS}
            gt = np.load(frames[qid]["depth"]).squeeze().astype(np.float64)
            access.append({"key": key, "path": frames[qid]["depth"]})
            valid = np.isfinite(gt) & (gt > 0)
            readouts, values = {"R7": r7}, {}
            for name, a in arrays.items():
                for arm, label in ARMS.items():
                    alpha = a[f"{arm}_alpha"].astype(np.float64)
                    ray = a[f"{arm}_depth_z"].astype(np.float64) / np.maximum(alpha, 1e-12) / cosine
                    readouts[f"{name}_{label}"] = np.where(alpha >= ALPHA_MIN, ray, r7)
                    values[f"{name}_{label}_coverage"] = float((alpha >= ALPHA_MIN)[valid].mean()) if valid.any() else None
                readouts[f"GEO_{name}_TTT"] = gm(r7, readouts[f"{name}_TTT"])
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
    table = {n: {s: float(np.mean([np.mean(v) for v in r.values()])) for s, r in sc.items()} for n, sc in grouped.items()}
    write(root / "scores/query_rows.json", rows)
    write(root / "scores/scene_table.json", table)
    write(root / "scores/score_info.json", {"queries": len(rows), "query_depth_reads": access,
                                            "seconds": round(time.time() - started, 1)})
    print("SCORED", len(rows), "queries")


PRIMARY = {"Q1_JEPA_BASE_VS_RELEASED_TTT_ALL": ("ZS_TTT|ALL", "JA_TTT|ALL"),
           "Q2_JEPA_TRAINING_HELPS_TTT_SAME_CAP_ALL": ("ZS3_TTT|ALL", "JA_TTT|ALL")}
SECONDARY = {"S1_ENSEMBLE_JEPA_BASE_VS_RELEASED_ALL": ("GEO_ZS_TTT|ALL", "GEO_JA_TTT|ALL"),
             "S2_ENSEMBLE_JEPA_BASE_VS_REPROJ7_ALL": ("R7|ALL", "GEO_JA_TTT|ALL"),
             "S3_TRAINING_EFFECT_BEFORE_TTT_ALL": ("ZS3_CACHE|ALL", "JA_CACHE|ALL"),
             "S4_TTT_GAIN_ON_JEPA_BASE_ALL": ("JA_CACHE|ALL", "JA_TTT|ALL"),
             "S5_JEPA_BASE_TTT_VS_REPROJ7_ALL": ("R7|ALL", "JA_TTT|ALL"),
             "S6_SCALE_CAP_RELEASED_TTT_ALL": ("ZS3_TTT|ALL", "ZS_TTT|ALL"),
             "S7_Q1_NEAR7": ("ZS_TTT|NEAR7", "JA_TTT|NEAR7"), "S8_Q1_FAR7": ("ZS_TTT|FAR7", "JA_TTT|FAR7"),
             "S9_Q2_NEAR7": ("ZS3_TTT|NEAR7", "JA_TTT|NEAR7")}
REPLICATION = {"ZS_TTT|ALL": "B:TTT|ALL", "ZS_CACHE|ALL": "B:CACHE|ALL", "R7|ALL": "B:R7|ALL",
               "GEO_ZS_TTT|ALL": "B:R7_TTT_GEO|ALL"}


def label(status, names):
    return {"SUPPORTED": names[0], "PARTIAL": names[1], "HARMFUL": names[3]}.get(status, names[2])


def analyze(root):
    table = json.loads((root / "scores/scene_table.json").read_text())
    primary = {k: contrast(table, a, b) for k, (a, b) in PRIMARY.items()}
    secondary = {k: contrast(table, a, b) for k, (a, b) in SECONDARY.items()}
    branches = {
        "Q1": label(primary["Q1_JEPA_BASE_VS_RELEASED_TTT_ALL"]["status"],
                    ("JEPA_BASE_BEATS_RELEASED_TTT", "JEPA_BASE_PARTIAL", "JEPA_BASE_NOT_ESTABLISHED",
                     "JEPA_BASE_HARMFUL")),
        "Q2": label(primary["Q2_JEPA_TRAINING_HELPS_TTT_SAME_CAP_ALL"]["status"],
                    ("JEPA_TRAINING_HELPS_TTT", "JEPA_TRAINING_HELPS_TTT_PARTIAL", "JEPA_TRAINING_TTT_NOT_ESTABLISHED",
                     "JEPA_TRAINING_HARMS_TTT")),
    }
    means = {k: {"mean": float(np.mean(list(v.values()))), "scenes": len(v)} for k, v in sorted(table.items())}
    reused = json.loads((root / "reused_seals.json").read_text())
    earlier = json.loads((Path(reused["source_experiment"]) / "results/ARKIT_VAL_ensembles_results.json").read_text())
    replication = {k: {"here": means[k]["mean"], "earlier": earlier["means"][v]["mean"],
                       "abs_diff": abs(means[k]["mean"] - earlier["means"][v]["mean"])} for k, v in REPLICATION.items()}
    write(root / "results/ARKIT_VAL_results.json", {"cohort": COHORT, "branches": branches, "primary": primary,
                                                    "secondary": secondary, "means": means,
                                                    "replication_of_reused_means": replication})
    print("ANALYZED", branches)
    for name, s in {**primary, **secondary}.items():
        print(f"  {name}: {s['mean']:+.4f} [{s['ci95'][0]:+.4f}, {s['ci95'][1]:+.4f}] "
              f"{s['positive']}/{s['tied']}/{s['negative']} {s['status']}")
    for k, v in means.items():
        if "|ALL" in k or "coverage" in k:
            print(f"  {k:24s} {v['mean']:.4f} ({v['scenes']})")
    print("  replication:", json.dumps({k: round(v["abs_diff"], 12) for k, v in replication.items()}))


if __name__ == "__main__":
    root, stage = Path(sys.argv[1]).resolve(), sys.argv[2]
    {"score": score, "analyze": analyze}[stage](root)
