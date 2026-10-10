"""EXP-3D-RGBD-DEPTHSPLAT-JEPA-CAP01-TTT-ARKIT-V1: scoring and the preregistered analysis on ARKIT_VAL.

score: every prediction must be sealed and unchanged: reused (hashes in reused_seals.json) the 7-frame geometry seal
and the ZS seals (released weights, cap 0.1) of EXP-3D-RGBD-ENSEMBLES-ARKIT-V1 and the JA / ZS3 seals (cap 3.0) of
EXP-3D-RGBD-DEPTHSPLAT-JEPA-BASE-TTT-ARKIT-V1; new the JC seals (this experiment's JC1, cap 0.1). Only then is query
depth read. Per query, the F7 readout of every state (CACHE, TTT_m1): ray depth (depth_z / alpha / cos) where
alpha >= 0.5, else REPROJ7_HARMONIC (R7); GEO_X = sqrt(R7 x X). AbsRel on ALL, NEAR7 and FAR7; coverage = share of
valid pixels with alpha >= 0.5. Scene value = mean over roles of the mean over queries (empty regions skipped).
analyze: V1's paired scene bootstrap (10,000 draws, seed 20260928) and V2 gate, plus a replication check of the reused
means against ENSEMBLES-ARKIT and JBT.
Run through cv_host_run.py from the repository root (CUDA hidden):  python jc_score.py ROOT score | analyze
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
CAPS = {"JC": 0.1, "ZS": 0.1, "JA": 3.0, "ZS3": 3.0}
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
    for item in [reused["geometry_seal"], *[p for parts in reused["seals"].values() for p in parts]]:
        if sha(item["path"]) != item["sha256"]:
            raise PermissionError(f"a reused seal changed: {item['path']}")
    geometry = json.loads(Path(reused["geometry_seal"]["path"]).read_text())
    if not geometry["replay_vs_arms_seal"]["pass"] or set(geometry["predictions"]) != set(keys):
        raise PermissionError("geometry seal incomplete or its replay failed")
    seals = {name: sealed([p["path"] for p in parts], keys, name) for name, parts in reused["seals"].items()}
    seals["JC"] = sealed(sorted((root / f"ds/{COHORT}/JC").glob("ds_seal_*of*.json")), keys, "JC")
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


PRIMARY = {"P1_JEPA_CAP01_VS_RELEASED_TTT_ALL": ("ZS_TTT|ALL", "JC_TTT|ALL"),
           "P2_JEPA_CAP01_VS_RELEASED_CACHE_ALL": ("ZS_CACHE|ALL", "JC_CACHE|ALL")}
SECONDARY = {"S1_ENSEMBLE_JC_VS_RELEASED_ALL": ("GEO_ZS_TTT|ALL", "GEO_JC_TTT|ALL"),
             "S2_ENSEMBLE_JC_VS_REPROJ7_ALL": ("R7|ALL", "GEO_JC_TTT|ALL"),
             "S3_TTT_GAIN_ON_JC_ALL": ("JC_CACHE|ALL", "JC_TTT|ALL"),
             "S4_JEPA_CAP01_VS_CAP3_TTT_ALL": ("JA_TTT|ALL", "JC_TTT|ALL"),
             "S5_P1_NEAR7": ("ZS_TTT|NEAR7", "JC_TTT|NEAR7"), "S6_P1_FAR7": ("ZS_TTT|FAR7", "JC_TTT|FAR7")}
REPLICATION = {"ENSEMBLES-ARKIT": {"ZS_TTT|ALL": "B:TTT|ALL", "ZS_CACHE|ALL": "B:CACHE|ALL", "R7|ALL": "B:R7|ALL"},
               "JBT": {"JA_TTT|ALL": "JA_TTT|ALL", "JA_CACHE|ALL": "JA_CACHE|ALL", "ZS3_TTT|ALL": "ZS3_TTT|ALL"}}


def label(status, names):
    return {"SUPPORTED": names[0], "PARTIAL": names[1], "HARMFUL": names[3]}.get(status, names[2])


def analyze(root):
    table = json.loads((root / "scores/scene_table.json").read_text())
    primary = {k: contrast(table, a, b) for k, (a, b) in PRIMARY.items()}
    secondary = {k: contrast(table, a, b) for k, (a, b) in SECONDARY.items()}
    branches = {
        "P1": label(primary["P1_JEPA_CAP01_VS_RELEASED_TTT_ALL"]["status"],
                    ("JEPA_CAP01_IMPROVES_TTT", "JEPA_CAP01_TTT_PARTIAL", "JEPA_CAP01_TTT_NOT_ESTABLISHED",
                     "JEPA_CAP01_HARMS_TTT")),
        "P2": label(primary["P2_JEPA_CAP01_VS_RELEASED_CACHE_ALL"]["status"],
                    ("JEPA_CAP01_IMPROVES_CACHE", "JEPA_CAP01_CACHE_PARTIAL", "JEPA_CAP01_CACHE_NOT_ESTABLISHED",
                     "JEPA_CAP01_HARMS_CACHE")),
    }
    means = {k: {"mean": float(np.mean(list(v.values()))), "scenes": len(v)} for k, v in sorted(table.items())}
    reused = json.loads((root / "reused_seals.json").read_text())
    earlier = {"ENSEMBLES-ARKIT": json.loads((Path(reused["source_experiment"]) /
                                              "results/ARKIT_VAL_ensembles_results.json").read_text())["means"],
               "JBT": json.loads((Path(reused["jbt_experiment"]) / "results/ARKIT_VAL_results.json").read_text())["means"]}
    replication = {f"{src}:{k}": abs(means[k]["mean"] - earlier[src][v]["mean"])
                   for src, pairs in REPLICATION.items() for k, v in pairs.items()}
    write(root / "results/ARKIT_VAL_results.json", {"cohort": COHORT, "branches": branches, "primary": primary,
                                                    "secondary": secondary, "means": means,
                                                    "replication_abs_diff": replication})
    print("ANALYZED", branches)
    for name, s in {**primary, **secondary}.items():
        print(f"  {name}: {s['mean']:+.4f} [{s['ci95'][0]:+.4f}, {s['ci95'][1]:+.4f}] "
              f"{s['positive']}/{s['tied']}/{s['negative']} median {s['median']:+.4f} {s['status']}")
    for k, v in means.items():
        if "|ALL" in k or "coverage" in k:
            print(f"  {k:24s} {v['mean']:.4f} ({v['scenes']})")
    print("  replication:", json.dumps({k: round(v, 12) for k, v in replication.items()}))


if __name__ == "__main__":
    root, stage = Path(sys.argv[1]).resolve(), sys.argv[2]
    {"score": score, "analyze": analyze}[stage](root)
