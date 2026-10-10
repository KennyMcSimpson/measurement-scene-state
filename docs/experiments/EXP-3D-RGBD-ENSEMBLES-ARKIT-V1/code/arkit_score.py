"""EXP-3D-RGBD-ENSEMBLES-ARKIT-V1: scoring and the preregistered analysis of both ensembles on ARKit (real sensors).

score: every prediction must be sealed and unchanged — the arms seal (REPROJ_HARMONIC, NEAR, G1 for three seeds),
the geometry seal (REPROJ7_HARMONIC, NEAR7, ray cosine; replayed against the arms seal) and the DepthSplat seals
(J1-J3: CTX; ZS: CACHE, TTT_m1); only then is query depth read.
  Completion ensemble (three context frames; FAR = valid pixels outside the 8-px NEAR of the context reprojection):
    H3 REPROJ_HARMONIC; G1 mean of three seeds; J where >= 2 of the 3 J seeds cover (alpha >= 0.5) the mean ray
    depth of the covering seeds, else H3; JG covered (J + G1) / 2 else G1; HG (H3 + G1) / 2;
    HJ covered (H3 + J) / 2 else H3; HJG covered (H3 + J + G1) / 3 else (H3 + G1) / 2.  ALL versions use H3 in NEAR.
  TTT ensemble (context + four stream frames): R7 REPROJ7_HARMONIC; CACHE, TTT ray depth where alpha >= 0.5 else R7;
    R7_TTT_GEO sqrt(R7 x TTT); R7_CACHE_GEO sqrt(R7 x CACHE); R7_TTT (R7 + TTT) / 2.  Regions ALL, NEAR7, FAR7.
Scene value = mean over roles of the mean over queries (empty regions skipped).
analyze: V1's paired scene bootstrap (10,000 draws, seed 20260928) and V2 gate.
Run through cv_host_run.py from the repository root (CUDA hidden):  python arkit_score.py ROOT score | analyze
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

COHORT, ALPHA_MIN, J_SEEDS = "ARKIT_VAL", 0.5, ("J1", "J2", "J3")


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
        if seal["query_depth_read"] is not False or seal["limit"] or seal["cohort"] != COHORT:
            raise PermissionError(f"{part}: not a complete sealed run")
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
    arms = json.loads((root / f"raw/eval/{COHORT}/prediction_seal.json").read_text())
    if arms["query_depth_read"] is not False or any(f"{COHORT}/{k}" not in arms["predictions"] for k in keys):
        raise PermissionError("arms seal incomplete or not certified")
    geometry = json.loads((root / f"geometry/geometry_seal_{COHORT}.json").read_text())
    if not geometry["replay_vs_arms_seal"]["pass"] or set(geometry["predictions"]) != set(keys):
        raise PermissionError("geometry seal incomplete or its replay failed")
    js = {s: ds_seal(root, s, keys) for s in J_SEEDS}
    zs = ds_seal(root, "ZS", keys)
    rows, access, started = [], [], time.time()
    gm = lambda *xs: np.exp(np.mean([np.log(np.maximum(x, 1e-6)) for x in xs], axis=0))  # noqa: E731
    for e in lock["episodes"]:
        frames = {f["frame_id"]: f for f in records[e["scene_id"]]["frames"]}
        for qid in e["queries"]:
            key = f"{e['scene_id']}/{e['role']}/{qid}"
            sealed, geo = load(arms["predictions"][f"{COHORT}/{key}"]), load(geometry["predictions"][key])
            jarr, zarr = [load(js[s][key]) for s in J_SEEDS], load(zs[key])
            gt = np.load(frames[qid]["depth"]).squeeze().astype(np.float64)
            access.append({"key": key, "path": frames[qid]["depth"]})
            valid = np.isfinite(gt) & (gt > 0)
            near, near7 = sealed["near"].astype(bool), geo["near7"].astype(bool)
            h3, r7, cosine = sealed["REPROJ_HARMONIC"], geo["REPROJ7_HARMONIC"], geo["cosine"]
            g1 = np.mean([sealed[f"G1_{s}"] for s in arms["seeds"]], axis=0)
            covers = np.array([a["CTX_alpha"].astype(np.float64) >= ALPHA_MIN for a in jarr])
            planar = np.array([a["CTX_depth_z"].astype(np.float64) / np.maximum(a["CTX_alpha"].astype(np.float64),
                                                                                1e-12) for a in jarr])
            count = covers.sum(axis=0)
            covered = count >= 2
            j = np.where(covered, (np.where(covers, planar, 0.0).sum(axis=0) / np.maximum(count, 1)) / cosine, h3)
            completion = {"H3": h3, "G1": g1, "J": j, "JG": np.where(covered, (j + g1) / 2, g1), "HG": (h3 + g1) / 2,
                          "HJ": np.where(covered, (h3 + j) / 2, h3),
                          "HJG": np.where(covered, (h3 + j + g1) / 3, (h3 + g1) / 2)}
            state = {}
            for arm in ("CACHE", "TTT_m1"):
                alpha = zarr[f"{arm}_alpha"].astype(np.float64)
                ray = zarr[f"{arm}_depth_z"].astype(np.float64) / np.maximum(alpha, 1e-12) / cosine
                state[arm] = np.where(alpha >= ALPHA_MIN, ray, r7)
            stream = {"R7": r7, "CACHE": state["CACHE"], "TTT": state["TTT_m1"],
                      "R7_TTT_GEO": gm(r7, state["TTT_m1"]), "R7_CACHE_GEO": gm(r7, state["CACHE"]),
                      "R7_TTT": (r7 + state["TTT_m1"]) / 2}
            absrel = lambda p, m: float(np.mean(np.abs(p[m] - gt[m]) / gt[m])) if m.any() else None  # noqa: E731
            values = {"J_far_coverage": float(covered[valid & ~near].mean()) if (valid & ~near).any() else None}
            for name, p in completion.items():
                values[f"A:{name}|FAR"] = absrel(p, valid & ~near)
                values[f"A:{name}|ALL"] = absrel(np.where(near, h3, p), valid)
            for name, p in stream.items():
                for region, mask in (("ALL", valid), ("NEAR7", valid & near7), ("FAR7", valid & ~near7)):
                    values[f"B:{name}|{region}"] = absrel(p, mask)
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


PRIMARY = {"A1_COMPLETION_ENSEMBLE_VS_HARMONIC_FAR": ("A:H3|FAR", "A:HJG|FAR"),
           "A2_JEPA_ADDS_TO_ENSEMBLE_FAR": ("A:HG|FAR", "A:HJG|FAR"),
           "B1_TTT_ENSEMBLE_VS_REPROJ7_ALL": ("B:R7|ALL", "B:R7_TTT_GEO|ALL"),
           "B2_TTT_NEEDED_IN_ENSEMBLE_ALL": ("B:R7_CACHE_GEO|ALL", "B:R7_TTT_GEO|ALL")}
SECONDARY = {"SA1_HG_VS_HARMONIC_FAR": ("A:H3|FAR", "A:HG|FAR"), "SA2_G1_VS_HARMONIC_FAR": ("A:H3|FAR", "A:G1|FAR"),
             "SA3_J_VS_HARMONIC_FAR": ("A:H3|FAR", "A:J|FAR"), "SA4_JG_VS_HARMONIC_FAR": ("A:H3|FAR", "A:JG|FAR"),
             "SA5_SHRINK_G1_FAR": ("A:G1|FAR", "A:HG|FAR"),
             "SA6_HJG_VS_HARMONIC_ALL": ("A:H3|ALL", "A:HJG|ALL"), "SA7_HG_VS_HARMONIC_ALL": ("A:H3|ALL", "A:HG|ALL"),
             "SB1_TTT_VS_REPROJ7_ALL": ("B:R7|ALL", "B:TTT|ALL"), "SB2_TTT_BEYOND_CACHE_ALL": ("B:CACHE|ALL", "B:TTT|ALL"),
             "SB3_ARITH_TTT_ENSEMBLE_ALL": ("B:R7|ALL", "B:R7_TTT|ALL"),
             "SB4_CACHE_ENSEMBLE_VS_REPROJ7_ALL": ("B:R7|ALL", "B:R7_CACHE_GEO|ALL"),
             "SB5_CACHE_VS_REPROJ7_ALL": ("B:R7|ALL", "B:CACHE|ALL"),
             "SB6_TTT_ENSEMBLE_NEAR7": ("B:R7|NEAR7", "B:R7_TTT_GEO|NEAR7"),
             "SB7_TTT_ENSEMBLE_FAR7": ("B:R7|FAR7", "B:R7_TTT_GEO|FAR7")}


def label(status, names):
    return {"SUPPORTED": names[0], "PARTIAL": names[1], "HARMFUL": names[3]}.get(status, names[2])


def analyze(root):
    table = json.loads((root / "scores/scene_table.json").read_text())
    run = lambda spec: {k: contrast(table, a, b) for k, (a, b) in spec.items()}  # noqa: E731
    primary, secondary = run(PRIMARY), run(SECONDARY)
    st = {k: v["status"] for k, v in primary.items()}
    branches = {
        "A1": label(st["A1_COMPLETION_ENSEMBLE_VS_HARMONIC_FAR"],
                    ("ENSEMBLE_CONSISTENT", "ENSEMBLE_PARTIAL", "ENSEMBLE_NOT_ESTABLISHED", "ENSEMBLE_HARMFUL")),
        "A2": label(st["A2_JEPA_ADDS_TO_ENSEMBLE_FAR"], ("JEPA_ADDS", "JEPA_ADDS_PARTIAL", "JEPA_NO_ADD", "JEPA_HARMFUL")),
        "B1": label(st["B1_TTT_ENSEMBLE_VS_REPROJ7_ALL"], ("TTT_ENSEMBLE_CONSISTENT", "TTT_ENSEMBLE_PARTIAL",
                                                          "TTT_ENSEMBLE_NOT_ESTABLISHED", "TTT_ENSEMBLE_HARMFUL")),
        "B2": label(st["B2_TTT_NEEDED_IN_ENSEMBLE_ALL"], ("TTT_NEEDED", "TTT_NEEDED_PARTIAL", "TTT_NOT_NEEDED",
                                                         "TTT_HARMFUL")),
    }
    means = {k: {"mean": float(np.mean(list(v.values()))), "scenes": len(v)} for k, v in sorted(table.items())}
    result = {"cohort": COHORT, "branches": branches, "primary": primary, "secondary": secondary, "means": means}
    write(root / "results/ARKIT_VAL_ensembles_results.json", result)
    print("ANALYZED", branches)
    for name, s in {**primary, **secondary}.items():
        print(f"  {name}: {s['mean']:+.4f} [{s['ci95'][0]:+.4f}, {s['ci95'][1]:+.4f}] "
              f"{s['positive']}/{s['tied']}/{s['negative']} {s['status']}")
    for k, v in means.items():
        print(f"  {k:24s} {v['mean']:.4f} ({v['scenes']})")


if __name__ == "__main__":
    root, stage = Path(sys.argv[1]).resolve(), sys.argv[2]
    {"score": score, "analyze": analyze}[stage](root)
