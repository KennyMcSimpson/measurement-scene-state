"""EXP-3D-RGBD-DEPTHSPLAT-JEPA-NORM-SCALE-ARKIT-V1: scoring and the preregistered analysis on ARKIT_SYSTEM3 and ARKIT_VAL.

score: every prediction must be sealed and unchanged before query depth is read: the reused geometry and DepthSplat
seals (hashes in reused_seals.json) and this experiment's CACHE seals (cap 0.1) of JL1-JL3 (both cohorts) and of JN2 /
JN3 (ARKIT_VAL), each naming the expected checkpoint hash (JL: the trained file; JN2 / JN3: NF's, recorded at the lock).
Per query, the F7 readout (ray depth = depth_z / alpha / cos where alpha >= 0.5, else REPROJ7_HARMONIC = R7) of every
CACHE state and of ZS's TTT_m1 state; CORE_B = sqrt(R7 x ZS_TTT_F7); JNS / JLS = the mean over the three seeds of the
per-query AbsRel. AbsRel on ALL, NEAR7 and FAR7; coverage = share of valid pixels with alpha >= 0.5. Scene value = mean
over roles of the mean over queries (empty regions skipped). Table keys are "<cohort>:<readout>|<region>".
analyze: V1's paired scene bootstrap (10,000 draws, seed 20260928) and V2 gate, plus a replication check of reused
means against NF, V3 and OBJECTIVE-TTT.
Run through cv_host_run.py from the repository root (CUDA hidden):  python ns_score.py ROOT score | analyze
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

ALPHA_MIN, CAP = 0.5, 0.1
COHORTS = {"ARKIT_SYSTEM3": "eval_SYSTEM3", "ARKIT_VAL": "eval_VAL"}
CACHE_BUILDERS = ("ZS", "JN1", "JN2", "JN3", "JL1", "JL2", "JL3")
NEW = {"ARKIT_SYSTEM3": ("JL1", "JL2", "JL3"), "ARKIT_VAL": ("JN2", "JN3", "JL1", "JL2", "JL3")}
SEED_GROUPS = {"JNS": ("JN1", "JN2", "JN3"), "JLS": ("JL1", "JL2", "JL3")}
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


def predictions(files, cohort, keys, checkpoint_sha=None):
    preds = {}
    for part in files:
        seal = json.loads(Path(part).read_text())
        if (seal["query_depth_read"] is not False or seal["limit"] or seal["cohort"] != cohort
                or seal["scale_max"] != CAP or "CACHE" not in seal["arms"]):
            raise PermissionError(f"{part}: not a complete CACHE run at cap {CAP}")
        if checkpoint_sha is not None and seal["checkpoint_sha256"] != checkpoint_sha:
            raise PermissionError(f"{part}: unexpected checkpoint")
        preds.update(seal["predictions"])
    if set(preds) != set(keys):
        raise ValueError(f"{files}: sealed queries differ from the frame-role lock")
    return preds


def f7(arrays, arm, r7, cosine):
    alpha = arrays[f"{arm}_alpha"].astype(np.float64)
    ray = arrays[f"{arm}_depth_z"].astype(np.float64) / np.maximum(alpha, 1e-12) / cosine
    return np.where(alpha >= ALPHA_MIN, ray, r7), alpha >= ALPHA_MIN


def score(root):
    prereg = json.loads((root / "preregistration.json").read_text())
    if sha(root / "reused_seals.json") != prereg["root_files"]["reused_seals.json"]:
        raise PermissionError("reused_seals.json changed since the lock")
    reused = json.loads((root / "reused_seals.json").read_text())
    checkpoints = {**{k: v["sha256"] for k, v in reused["checkpoints"].items()},
                   **{s: sha(root / f"checkpoints/{s}/depthsplat_jepa_final.pth") for s in SEED_GROUPS["JLS"]}}
    gm = lambda a, b: np.sqrt(np.maximum(a, 1e-6) * np.maximum(b, 1e-6))  # noqa: E731
    rows, access, started = [], [], time.time()
    for cohort, sub in COHORTS.items():
        lock_path = root / sub / "frame_role_lock.json"
        if sha(lock_path) != prereg["root_files"][f"{sub}/frame_role_lock.json"]:
            raise PermissionError(f"{cohort}: frame-role lock changed since the lock")
        lock = json.loads(lock_path.read_text())["cohorts"][cohort]
        if sha(lock["manifest"]) != lock["manifest_sha256"]:
            raise PermissionError(f"{cohort}: manifest changed since the frame-role lock")
        records = {r["scene_id"]: r for r in json.loads(Path(lock["manifest"]).read_text())["scenes"]}
        keys = [f"{e['scene_id']}/{e['role']}/{q}" for e in lock["episodes"] for q in e["queries"]]
        geo_item = reused["geometry"][cohort]
        if sha(geo_item["path"]) != geo_item["sha256"]:
            raise PermissionError(f"{cohort}: the reused geometry seal changed")
        geometry = json.loads(Path(geo_item["path"]).read_text())
        if not geometry["replay_vs_arms_seal"]["pass"] or set(geometry["predictions"]) != set(keys):
            raise PermissionError(f"{cohort}: geometry seal incomplete or its replay failed")
        seals = {}
        for name, parts in reused["seals"][cohort].items():
            for part in parts:
                if sha(part["path"]) != part["sha256"]:
                    raise PermissionError(f"a reused seal changed: {part['path']}")
            seals[name] = predictions([p["path"] for p in parts], cohort, keys)
        for name in NEW[cohort]:
            files = sorted((root / f"ds/{cohort}/{name}").glob("ds_seal_*of*.json"))
            seals[name] = predictions(files, cohort, keys, checkpoints[name])
        for e in lock["episodes"]:
            frames = {f["frame_id"]: f for f in records[e["scene_id"]]["frames"]}
            for qid in e["queries"]:
                key = f"{e['scene_id']}/{e['role']}/{qid}"
                geo = load(geometry["predictions"][key])
                r7, near7, cosine = geo["REPROJ7_HARMONIC"], geo["near7"].astype(bool), geo["cosine"]
                gt = np.load(frames[qid]["depth"]).squeeze().astype(np.float64)
                access.append({"cohort": cohort, "key": key, "path": frames[qid]["depth"]})
                valid = np.isfinite(gt) & (gt > 0)
                readouts, values = {"R7": r7}, {}
                for name in CACHE_BUILDERS:
                    readouts[f"{name}_CACHE"], covered = f7(load(seals[name][key]), "CACHE", r7, cosine)
                    values[f"{name}_CACHE_coverage"] = float(covered[valid].mean()) if valid.any() else None
                readouts["ZS_TTT_F7"], _ = f7(load(seals["ZS_TTT"][key]), "TTT_m1", r7, cosine)
                readouts["CORE_B"] = gm(r7, readouts["ZS_TTT_F7"])
                for name, p in readouts.items():
                    for region, mask in zip(REGIONS, (valid, valid & near7, valid & ~near7)):
                        values[f"{name}|{region}"] = (float(np.mean(np.abs(p[mask] - gt[mask]) / gt[mask]))
                                                      if mask.any() else None)
                for group, seeds in SEED_GROUPS.items():
                    for suffix in [f"|{r}" for r in REGIONS] + ["_coverage"]:
                        parts = [values[f"{s}_CACHE{suffix}"] for s in seeds]
                        values[f"{group}_CACHE{suffix}"] = None if any(v is None for v in parts) else float(np.mean(parts))
                rows.append({"cohort": cohort, "scene_id": e["scene_id"], "role": e["role"], "query": qid,
                             "values": values})
    grouped = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for row in rows:
        for name, v in row["values"].items():
            if v is not None and np.isfinite(v):
                grouped[f"{row['cohort']}:{name}"][row["scene_id"]][row["role"]].append(v)
    table = {n: {s: float(np.mean([np.mean(v) for v in r.values()])) for s, r in sc.items()} for n, sc in grouped.items()}
    write(root / "scores/query_rows.json", rows)
    write(root / "scores/scene_table.json", table)
    write(root / "scores/score_info.json", {"queries": len(rows), "query_depth_reads": access, "checkpoints": checkpoints,
                                            "seconds": round(time.time() - started, 1)})
    print("SCORED", len(rows), "queries")


S3, VAL = "ARKIT_SYSTEM3", "ARKIT_VAL"
PRIMARY = {"P1_MORE_DATA_SYSTEM3_ALL": (f"{S3}:JNS_CACHE|ALL", f"{S3}:JLS_CACHE|ALL"),
           "P2_SCALED_RECIPE_VS_RELEASED_SYSTEM3_ALL": (f"{S3}:ZS_CACHE|ALL", f"{S3}:JLS_CACHE|ALL")}
SECONDARY = {"S1_MORE_DATA_VAL_ALL": (f"{VAL}:JNS_CACHE|ALL", f"{VAL}:JLS_CACHE|ALL"),
             "S2_SCALED_RECIPE_VS_RELEASED_VAL_ALL": (f"{VAL}:ZS_CACHE|ALL", f"{VAL}:JLS_CACHE|ALL"),
             "S3_SCALED_RECIPE_VS_MODEL_FREE_SYSTEM3_ALL": (f"{S3}:R7|ALL", f"{S3}:JLS_CACHE|ALL"),
             "S4_SCALED_RECIPE_VS_CORE_B_SYSTEM3_ALL": (f"{S3}:CORE_B|ALL", f"{S3}:JLS_CACHE|ALL"),
             "S5_MORE_DATA_SYSTEM3_NEAR7": (f"{S3}:JNS_CACHE|NEAR7", f"{S3}:JLS_CACHE|NEAR7"),
             "S6_MORE_DATA_SYSTEM3_FAR7": (f"{S3}:JNS_CACHE|FAR7", f"{S3}:JLS_CACHE|FAR7"),
             "S7_SEED_JL1_SYSTEM3_ALL": (f"{S3}:ZS_CACHE|ALL", f"{S3}:JL1_CACHE|ALL"),
             "S8_SEED_JL2_SYSTEM3_ALL": (f"{S3}:ZS_CACHE|ALL", f"{S3}:JL2_CACHE|ALL"),
             "S9_SEED_JL3_SYSTEM3_ALL": (f"{S3}:ZS_CACHE|ALL", f"{S3}:JL3_CACHE|ALL")}
REPLICATION = {"NF": ("EXP-3D-RGBD-DEPTHSPLAT-JEPA-NORM-FEEDFORWARD-ARKIT-V1/results/ARKIT_SYSTEM3_results.json",
                      {f"{S3}:{k}": k for k in ("ZS_CACHE|ALL", "JN1_CACHE|ALL", "JN2_CACHE|ALL", "JN3_CACHE|ALL",
                                                 "JNS_CACHE|ALL", "R7|ALL")}),
               "V3": ("EXP-3D-RGBD-FULL-SYSTEM-ARKIT-V3/results/ARKIT_SYSTEM3_results.json",
                      {f"{S3}:CORE_B|ALL": "CORE_B|ALL", f"{S3}:ZS_TTT_F7|ALL": "ZS_TTT_F7|ALL"}),
               "OBJECTIVE-TTT": ("EXP-3D-RGBD-DEPTHSPLAT-JEPA-OBJECTIVE-TTT-ARKIT-V1/results/ARKIT_VAL_results.json",
                                 {f"{VAL}:ZS_CACHE|ALL": "ZS_CACHE|ALL", f"{VAL}:JN1_CACHE|ALL": "JN_CACHE|ALL",
                                  f"{VAL}:ZS_TTT_F7|ALL": "ZS_TTT|ALL", f"{VAL}:R7|ALL": "R7|ALL"})}


def label(status, names):
    return {"SUPPORTED": names[0], "PARTIAL": names[1], "HARMFUL": names[3]}.get(status, names[2])


def analyze(root):
    table = json.loads((root / "scores/scene_table.json").read_text())
    primary = {k: contrast(table, a, b) for k, (a, b) in PRIMARY.items()}
    secondary = {k: contrast(table, a, b) for k, (a, b) in SECONDARY.items()}
    branches = {
        "P1": label(primary["P1_MORE_DATA_SYSTEM3_ALL"]["status"],
                    ("MORE_DATA_HELPS", "MORE_DATA_HELPS_PARTIAL", "MORE_DATA_NOT_ESTABLISHED", "MORE_DATA_HURTS")),
        "P2": label(primary["P2_SCALED_RECIPE_VS_RELEASED_SYSTEM3_ALL"]["status"],
                    ("SCALED_RECIPE_IMPROVES_FEEDFORWARD", "SCALED_RECIPE_FEEDFORWARD_PARTIAL",
                     "SCALED_RECIPE_FEEDFORWARD_NOT_ESTABLISHED", "SCALED_RECIPE_HARMS_FEEDFORWARD")),
    }
    means = {k: {"mean": float(np.mean(list(v.values()))), "scenes": len(v)} for k, v in sorted(table.items())}
    outputs = Path(__file__).resolve().parents[2]
    replication = {}
    for source, (path, pairs) in REPLICATION.items():
        earlier = json.loads((outputs / path).read_text())["means"]
        for mine, theirs in pairs.items():
            replication[f"{source}:{mine}"] = abs(means[mine]["mean"] - earlier[theirs]["mean"])
    write(root / "results/NS_results.json", {"branches": branches, "primary": primary, "secondary": secondary,
                                             "means": means, "replication_abs_diff": replication})
    print("ANALYZED", branches)
    for name, s in {**primary, **secondary}.items():
        print(f"  {name}: {s['mean']:+.4f} [{s['ci95'][0]:+.4f}, {s['ci95'][1]:+.4f}] "
              f"{s['positive']}/{s['tied']}/{s['negative']} median {s['median']:+.4f} {s['status']}")
    for k, v in means.items():
        if any(t in k for t in ("ZS_CACHE", "JNS_CACHE", "JLS_CACHE", "JL1", "JL2", "JL3", "CORE_B|ALL", "R7|ALL")):
            print(f"  {k:36s} {v['mean']:.4f} ({v['scenes']})")
    print("  replication:", json.dumps({k: round(v, 12) for k, v in replication.items()}))


if __name__ == "__main__":
    root, stage = Path(sys.argv[1]).resolve(), sys.argv[2]
    {"score": score, "analyze": analyze}[stage](root)
