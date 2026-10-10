"""EXP-3D-RGBD-FULL-SYSTEM-ARKIT-V1: scoring and the preregistered analysis on ARKIT_SYSTEM (80 fresh scans).

score: every prediction must be sealed and unchanged before query depth is read: the arms seal (G1 and GA depth and
opacity, three seeds each; 3-context), the geometry seal (REPROJ7_HARMONIC = R7, NEAR7, ray cosine; replayed against
the arms seal), the released-weights DepthSplat seals (ZS, cap 0.1: CACHE, TTT_m1) and the selector seal. Per query:
  ZS_CACHE_F7, ZS_TTT_F7   ray depth where alpha >= 0.5, else R7;   CORE_B = sqrt(R7 x ZS_TTT_F7)
  GAN7 (G1N7)  per seed depth / opacity where opacity >= 0.5, else R7; mean over the three seeds
  R7S          GAN7 on FAR7 pixels (outside NEAR7) of selected episodes (S >= 1.2), else R7
  FULL         sqrt(R7S x TTT_F7S), TTT_F7S = TTT ray depth where alpha >= 0.5, else R7S
  FULL_G1      as FULL with G1N7;   FULL_ALL  as FULL with every episode selected
AbsRel on ALL (valid pixels), NEAR7 and FAR7. Scene value = mean over roles of the mean over queries.
analyze: V1's paired scene bootstrap (10,000 draws, seed 20260928) and V2 gate.
Run through cv_host_run.py from the repository root (CUDA hidden):  python sys_score.py ROOT score | analyze
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

COHORT, ALPHA_MIN, OPACITY_MIN = "ARKIT_SYSTEM", 0.5, 0.5


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
    prereg = json.loads((root / "preregistration.json").read_text())
    if sha(root / "selector_seal.json") != prereg["root_files"]["selector_seal.json"]:
        raise PermissionError("selector seal changed since the lock")
    selector = json.loads((root / "selector_seal.json").read_text())["episodes"]
    records = {r["scene_id"]: r for r in json.loads(Path(lock["manifest"]).read_text())["scenes"]}
    keys = [f"{e['scene_id']}/{e['role']}/{q}" for e in lock["episodes"] for q in e["queries"]]
    arms = json.loads((root / f"raw/eval/{COHORT}/prediction_seal.json").read_text())
    if arms["query_depth_read"] is not False or any(f"{COHORT}/{k}" not in arms["predictions"] for k in keys):
        raise PermissionError("arms seal incomplete or not certified")
    geometry = json.loads((root / f"geometry/geometry_seal_{COHORT}.json").read_text())
    if not geometry["replay_vs_arms_seal"]["pass"] or set(geometry["predictions"]) != set(keys):
        raise PermissionError("geometry seal incomplete or its replay failed")
    zs = {}
    for part in sorted((root / f"ds/{COHORT}/ZS").glob("ds_seal_*of*.json")):
        seal = json.loads(part.read_text())
        if (seal["query_depth_read"] is not False or seal["limit"] or seal["cohort"] != COHORT
                or seal["scale_max"] != 0.1 or seal["ttt_mult"] != [1.0] or seal["ttt_steps"] != 100):
            raise PermissionError(f"{part}: not the protocol's run")
        zs.update(seal["predictions"])
    if set(zs) != set(keys):
        raise ValueError("ZS seals differ from the frame-role lock")
    gm = lambda a, b: np.sqrt(np.maximum(a, 1e-6) * np.maximum(b, 1e-6))  # noqa: E731
    rows, access, started = [], [], time.time()
    for e in lock["episodes"]:
        record = records[e["scene_id"]]
        frames = {f["frame_id"]: f for f in record["frames"]}
        use = selector[f"{e['scene_id']}/{e['role']}"]["use_learned"]
        for qid in e["queries"]:
            key = f"{e['scene_id']}/{e['role']}/{qid}"
            p, geo, z = load(arms["predictions"][f"{COHORT}/{key}"]), load(geometry["predictions"][key]), load(zs[key])
            r7, near7, cosine = geo["REPROJ7_HARMONIC"], geo["near7"].astype(bool), geo["cosine"]
            far7 = ~near7
            ray = {arm: z[f"{arm}_depth_z"].astype(np.float64) / np.maximum(z[f"{arm}_alpha"].astype(np.float64), 1e-12)
                   / cosine for arm in ("CACHE", "TTT_m1")}
            cover = {arm: z[f"{arm}_alpha"].astype(np.float64) >= ALPHA_MIN for arm in ("CACHE", "TTT_m1")}
            gan7, g1n7 = normalized(p, "GA", arms["seeds"], r7), normalized(p, "G1", arms["seeds"], r7)

            def system(fill, selected):
                base = np.where(far7 & selected, fill, r7)
                return gm(base, np.where(cover["TTT_m1"], ray["TTT_m1"], base))

            zs_ttt = np.where(cover["TTT_m1"], ray["TTT_m1"], r7)
            readouts = {"R7": r7, "ZS_CACHE_F7": np.where(cover["CACHE"], ray["CACHE"], r7), "ZS_TTT_F7": zs_ttt,
                        "CORE_B": gm(r7, zs_ttt), "FULL": system(gan7, use), "FULL_G1": system(g1n7, use),
                        "FULL_ALL": system(gan7, True)}
            gt = np.load(frames[qid]["depth"]).squeeze().astype(np.float64)
            access.append({"key": key, "path": frames[qid]["depth"]})
            valid = np.isfinite(gt) & (gt > 0)
            values = {"far7_share": float(far7[valid].mean()) if valid.any() else None}
            for name, x in readouts.items():
                for region, mask in (("ALL", valid), ("NEAR7", valid & near7), ("FAR7", valid & far7)):
                    values[f"{name}|{region}"] = (float(np.mean(np.abs(x[mask] - gt[mask]) / gt[mask]))
                                                  if mask.any() else None)
            rows.append({"scene_id": e["scene_id"], "role": e["role"], "query": qid, "use_learned": use,
                         "values": values})
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


PRIMARY = {"P1_FULL_SYSTEM_VS_CORE_B_ALL": ("CORE_B|ALL", "FULL|ALL"),
           "P2_FULL_SYSTEM_VS_MODEL_FREE_ALL": ("R7|ALL", "FULL|ALL")}
SECONDARY = {"S1_FULL_SYSTEM_VS_CORE_B_FAR7": ("CORE_B|FAR7", "FULL|FAR7"),
             "S2_CORE_B_VS_MODEL_FREE_ALL": ("R7|ALL", "CORE_B|ALL"),
             "S3_TTT_GAIN_ALL": ("ZS_CACHE_F7|ALL", "ZS_TTT_F7|ALL"),
             "S4_FULL_WITH_HYPERSIM_CARRIER_ALL": ("CORE_B|ALL", "FULL_G1|ALL"),
             "S5_FULL_WITHOUT_SELECTION_ALL": ("CORE_B|ALL", "FULL_ALL|ALL"),
             "S6_SELECTION_EFFECT_ALL": ("FULL_ALL|ALL", "FULL|ALL")}


def label(status, names):
    return {"SUPPORTED": names[0], "PARTIAL": names[1], "HARMFUL": names[3]}.get(status, names[2])


def analyze(root):
    table = json.loads((root / "scores/scene_table.json").read_text())
    primary = {k: contrast(table, a, b) for k, (a, b) in PRIMARY.items()}
    secondary = {k: contrast(table, a, b) for k, (a, b) in SECONDARY.items()}
    branches = {
        "P1": label(primary["P1_FULL_SYSTEM_VS_CORE_B_ALL"]["status"],
                    ("FULL_SYSTEM_IMPROVES", "FULL_SYSTEM_IMPROVES_PARTIAL", "FULL_SYSTEM_IMPROVEMENT_NOT_ESTABLISHED",
                     "FULL_SYSTEM_HARMFUL")),
        "P2": label(primary["P2_FULL_SYSTEM_VS_MODEL_FREE_ALL"]["status"],
                    ("FULL_SYSTEM_BEATS_MODEL_FREE", "FULL_SYSTEM_BEATS_MODEL_FREE_PARTIAL",
                     "FULL_SYSTEM_VS_MODEL_FREE_NOT_ESTABLISHED", "FULL_SYSTEM_WORSE_THAN_MODEL_FREE")),
    }
    rows = json.loads((root / "scores/query_rows.json").read_text())
    episodes = {(r["scene_id"], r["role"]): r["use_learned"] for r in rows}
    descriptive = {"episodes": len(episodes), "episodes_selected": int(sum(episodes.values())),
                   "mean_far7_share": float(np.mean([r["values"]["far7_share"] for r in rows
                                                     if r["values"]["far7_share"] is not None]))}
    means = {k: {"mean": float(np.mean(list(v.values()))), "scenes": len(v)} for k, v in sorted(table.items())}
    write(root / "results/ARKIT_SYSTEM_results.json", {"cohort": COHORT, "branches": branches, "primary": primary,
                                                       "secondary": secondary, "means": means,
                                                       "descriptive": descriptive})
    print("ANALYZED", branches)
    for name, s in {**primary, **secondary}.items():
        print(f"  {name}: {s['mean']:+.4f} [{s['ci95'][0]:+.4f}, {s['ci95'][1]:+.4f}] "
              f"{s['positive']}/{s['tied']}/{s['negative']} median {s['median']:+.4f} ({s['scenes']} scenes) {s['status']}")
    for k, v in means.items():
        print(f"  {k:20s} {v['mean']:.4f} ({v['scenes']})")
    print("  descriptive:", json.dumps(descriptive))


if __name__ == "__main__":
    root, stage = Path(sys.argv[1]).resolve(), sys.argv[2]
    {"score": score, "analyze": analyze}[stage](root)
