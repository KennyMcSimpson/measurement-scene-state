"""EXP-3D-RGBD-DEPTHSPLAT-JEPA-NORM-FEEDFORWARD-ARKIT-V1: scoring and the preregistered analysis on ARKIT_SYSTEM3.

score: every prediction must be sealed and unchanged before query depth is read: the reused 7-frame geometry seal of
EXP-3D-RGBD-FULL-SYSTEM-ARKIT-V3 (hash in reused_seals.json) and this experiment's CACHE seals (cap 0.1) of ZS
(released weights), JN1 and JK1 (OBJECTIVE-TTT checkpoints) and JN2, JN3 (trained here); every seal must name the
checkpoint hash recorded at the lock (ZS, JN1, JK1) or of the trained file (JN2, JN3). Per query, the F7 readout of
every CACHE state: ray depth (depth_z / alpha / cos) where alpha >= 0.5, else REPROJ7_HARMONIC (R7); GEO_X =
sqrt(R7 x X); JNS = the mean over JN1-JN3 of the per-query AbsRel. AbsRel on ALL, NEAR7 and FAR7; coverage = share of
valid pixels with alpha >= 0.5. Scene value = mean over roles of the mean over queries (empty regions skipped).
analyze: V1's paired scene bootstrap (10,000 draws, seed 20260928) and V2 gate.
Run through cv_host_run.py from the repository root (CUDA hidden):  python nf_score.py ROOT score | analyze
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

COHORT, ALPHA_MIN, CAP = "ARKIT_SYSTEM3", 0.5, 0.1
BUILDERS = ("ZS", "JN1", "JN2", "JN3", "JK1")
SEEDS = ("JN1", "JN2", "JN3")
REGIONS = ("ALL", "NEAR7", "FAR7")


def sha(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load(item):
    if sha(item["path"]) != item["sha256"]:
        raise PermissionError(f"sealed prediction changed: {item['path']}")
    return dict(np.load(item["path"]))


def sealed(root, name, keys, checkpoint_sha):
    preds = {}
    for part in sorted((root / f"ds/{COHORT}/{name}").glob("ds_seal_*of*.json")):
        seal = json.loads(part.read_text())
        if seal["query_depth_read"] is not False or seal["limit"] or seal["cohort"] != COHORT:
            raise PermissionError(f"{part}: not a complete sealed run")
        if seal["scale_max"] != CAP or seal["arms"] != ["CACHE"] or seal["checkpoint_sha256"] != checkpoint_sha:
            raise PermissionError(f"{part}: settings or checkpoint differ from the protocol")
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
    prereg = json.loads((root / "preregistration.json").read_text())
    reused = json.loads((root / "reused_seals.json").read_text())
    if sha(root / "reused_seals.json") != prereg["root_files"]["reused_seals.json"]:
        raise PermissionError("reused_seals.json changed since the lock")
    if sha(reused["geometry_seal"]["path"]) != reused["geometry_seal"]["sha256"]:
        raise PermissionError("the reused geometry seal changed")
    geometry = json.loads(Path(reused["geometry_seal"]["path"]).read_text())
    if not geometry["replay_vs_arms_seal"]["pass"] or set(geometry["predictions"]) != set(keys):
        raise PermissionError("geometry seal incomplete or its replay failed")
    checkpoints = {"ZS": prereg["released_weights_sha256"], **prereg["reused_checkpoints"],
                   **{s: sha(root / f"checkpoints/{s}/depthsplat_jepa_final.pth") for s in ("JN2", "JN3")}}
    seals = {name: sealed(root, name, keys, checkpoints[name]) for name in BUILDERS}
    gm = lambda *xs: np.exp(np.mean([np.log(np.maximum(x, 1e-6)) for x in xs], axis=0))  # noqa: E731
    rows, access, started = [], [], time.time()
    for e in lock["episodes"]:
        frames = {f["frame_id"]: f for f in records[e["scene_id"]]["frames"]}
        for qid in e["queries"]:
            key = f"{e['scene_id']}/{e['role']}/{qid}"
            geo = load(geometry["predictions"][key])
            r7, near7, cosine = geo["REPROJ7_HARMONIC"], geo["near7"].astype(bool), geo["cosine"]
            arrays = {name: load(seals[name][key]) for name in BUILDERS}
            gt = np.load(frames[qid]["depth"]).squeeze().astype(np.float64)
            access.append({"key": key, "path": frames[qid]["depth"]})
            valid = np.isfinite(gt) & (gt > 0)
            readouts, values = {"R7": r7}, {}
            for name, a in arrays.items():
                alpha = a["CACHE_alpha"].astype(np.float64)
                ray = a["CACHE_depth_z"].astype(np.float64) / np.maximum(alpha, 1e-12) / cosine
                readouts[f"{name}_CACHE"] = np.where(alpha >= ALPHA_MIN, ray, r7)
                readouts[f"GEO_{name}_CACHE"] = gm(r7, readouts[f"{name}_CACHE"])
                values[f"{name}_CACHE_coverage"] = float((alpha >= ALPHA_MIN)[valid].mean()) if valid.any() else None
            for name, p in readouts.items():
                for region, mask in zip(REGIONS, (valid, valid & near7, valid & ~near7)):
                    values[f"{name}|{region}"] = (float(np.mean(np.abs(p[mask] - gt[mask]) / gt[mask]))
                                                  if mask.any() else None)
            for suffix in [f"|{r}" for r in REGIONS] + ["_coverage"]:
                parts = [values[f"{s}_CACHE{suffix}"] for s in SEEDS]
                values[f"JNS_CACHE{suffix}"] = None if any(v is None for v in parts) else float(np.mean(parts))
            rows.append({"scene_id": e["scene_id"], "role": e["role"], "query": qid, "values": values})
    grouped = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for row in rows:
        for name, v in row["values"].items():
            if v is not None and np.isfinite(v):
                grouped[name][row["scene_id"]][row["role"]].append(v)
    table = {n: {s: float(np.mean([np.mean(v) for v in r.values()])) for s, r in sc.items()} for n, sc in grouped.items()}
    write(root / "scores/query_rows.json", rows)
    write(root / "scores/scene_table.json", table)
    write(root / "scores/score_info.json", {"queries": len(rows), "query_depth_reads": access, "checkpoints": checkpoints,
                                            "seconds": round(time.time() - started, 1)})
    print("SCORED", len(rows), "queries")


PRIMARY = {"P1_JN1_FEEDFORWARD_VS_RELEASED_ALL": ("ZS_CACHE|ALL", "JN1_CACHE|ALL"),
           "P2_RECIPE_FEEDFORWARD_VS_RELEASED_ALL": ("ZS_CACHE|ALL", "JNS_CACHE|ALL")}
SECONDARY = {"S1_JN1_VS_MODEL_FREE_ALL": ("R7|ALL", "JN1_CACHE|ALL"),
             "S2_RECIPE_VS_MODEL_FREE_ALL": ("R7|ALL", "JNS_CACHE|ALL"),
             "S3_RELEASED_VS_MODEL_FREE_ALL": ("R7|ALL", "ZS_CACHE|ALL"),
             "S4_JN1_NEAR7": ("ZS_CACHE|NEAR7", "JN1_CACHE|NEAR7"),
             "S5_JN1_FAR7": ("ZS_CACHE|FAR7", "JN1_CACHE|FAR7"),
             "S6_ENSEMBLE_JN1_VS_RELEASED_ALL": ("GEO_ZS_CACHE|ALL", "GEO_JN1_CACHE|ALL"),
             "S7_SEVEN_VIEW_JK1_ALL": ("ZS_CACHE|ALL", "JK1_CACHE|ALL"),
             "S8_SEED_JN2_ALL": ("ZS_CACHE|ALL", "JN2_CACHE|ALL"),
             "S9_SEED_JN3_ALL": ("ZS_CACHE|ALL", "JN3_CACHE|ALL")}


def label(status, names):
    return {"SUPPORTED": names[0], "PARTIAL": names[1], "HARMFUL": names[3]}.get(status, names[2])


def analyze(root):
    table = json.loads((root / "scores/scene_table.json").read_text())
    primary = {k: contrast(table, a, b) for k, (a, b) in PRIMARY.items()}
    secondary = {k: contrast(table, a, b) for k, (a, b) in SECONDARY.items()}
    branches = {
        "P1": label(primary["P1_JN1_FEEDFORWARD_VS_RELEASED_ALL"]["status"],
                    ("JN1_FEEDFORWARD_REPLICATES", "JN1_FEEDFORWARD_PARTIAL", "JN1_FEEDFORWARD_NOT_ESTABLISHED",
                     "JN1_FEEDFORWARD_HARMFUL")),
        "P2": label(primary["P2_RECIPE_FEEDFORWARD_VS_RELEASED_ALL"]["status"],
                    ("RECIPE_IMPROVES_FEEDFORWARD", "RECIPE_FEEDFORWARD_PARTIAL", "RECIPE_FEEDFORWARD_NOT_ESTABLISHED",
                     "RECIPE_HARMS_FEEDFORWARD")),
    }
    means = {k: {"mean": float(np.mean(list(v.values()))), "scenes": len(v)} for k, v in sorted(table.items())}
    write(root / "results/ARKIT_SYSTEM3_results.json", {"cohort": COHORT, "branches": branches, "primary": primary,
                                                        "secondary": secondary, "means": means})
    print("ANALYZED", branches)
    for name, s in {**primary, **secondary}.items():
        print(f"  {name}: {s['mean']:+.4f} [{s['ci95'][0]:+.4f}, {s['ci95'][1]:+.4f}] "
              f"{s['positive']}/{s['tied']}/{s['negative']} median {s['median']:+.4f} {s['status']}")
    for k, v in means.items():
        if "GEO_" not in k:
            print(f"  {k:24s} {v['mean']:.4f} ({v['scenes']})")


if __name__ == "__main__":
    root, stage = Path(sys.argv[1]).resolve(), sys.argv[2]
    {"score": score, "analyze": analyze}[stage](root)
