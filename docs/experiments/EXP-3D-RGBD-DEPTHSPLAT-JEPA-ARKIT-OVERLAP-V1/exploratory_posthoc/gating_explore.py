"""EXPLORATORY (post hoc, in-sample, after the locked analysis of EXP-3D-RGBD-DEPTHSPLAT-JEPA-ARKIT-OVERLAP-V1): can
agreement gating make learned completion consistent on ARKit overlap queries? Learned completion lowered the mean FARH
AbsRel by ~25 % but improved only ~54 % of the scenes. Every variant falls back to the harmonic fill H3:
  JO, HJO, G1N, HGN       as in the locked analysis (reference)
  JO_SEED<t>              JO where at least two JO seeds cover and their ray depths agree ((max - min) / mean <= t)
  JOG1_AGREE<t>           (JO + G1N) / 2 where JO covers, G1N is confident (some seed opacity >= 0.5) and the two
                          agree (|JO - G1N| / mean <= t)
  HJOG                    JO covered: (H3 + JO + G1N) / 3, else (H3 + G1N) / 2
  JOG1_AVG                JO covered: (JO + G1N) / 2, else G1N
Contrasts H3 - X on FARH with the V2 labels, plus the share of FARH pixels where X differs from H3. Uses only sealed
predictions (sha-checked) and the query depth already scored. In-sample: a promising variant needs a fresh test.
Run through cv_host_run.py from the repository root (CUDA hidden):  python gating_explore.py ROOT
"""
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from overlap_score import ALPHA_MIN, COHORT, ds_seal, load, ray_cosine  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "EXP-3D-RGBD-DEPTHSPLAT-JEPA-TTT-V1" / "code"))
from score_analyze import contrast, write  # noqa: E402

SEEDS = ("JO1", "JO2", "JO3")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(root):
    lock = json.loads((root / "frame_role_lock.json").read_text())["cohorts"][COHORT]
    records = {r["scene_id"]: r for r in json.loads(Path(lock["manifest"]).read_text())["scenes"]}
    keys = [f"{e['scene_id']}/{e['role']}/{q}" for e in lock["episodes"] for q in e["queries"]]
    arms = json.loads((root / f"raw/eval/{COHORT}/prediction_seal.json").read_text())
    seals = {s: ds_seal(root, s, keys) for s in SEEDS}
    grouped = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for e in lock["episodes"]:
        record = records[e["scene_id"]]
        frames = {f["frame_id"]: f for f in record["frames"]}
        for qid in e["queries"]:
            key = f"{e['scene_id']}/{e['role']}/{qid}"
            p = load(arms["predictions"][f"{COHORT}/{key}"])
            near = p["near"].astype(bool)
            if not near.any():
                continue
            h3 = p["REPROJ_HARMONIC"]
            opac = np.array([p[f"OPACITY_G1_{s}"].astype(np.float64) for s in arms["seeds"]])
            g1_conf = (opac >= 0.5).any(axis=0)
            g1n = np.mean([np.where(p[f"OPACITY_G1_{s}"] >= 0.5,
                                    p[f"G1_{s}"] / np.maximum(p[f"OPACITY_G1_{s}"].astype(np.float64), 1e-6), h3)
                           for s in arms["seeds"]], axis=0)
            cosine = ray_cosine(record, qid)
            arrays = [load(seals[s][key]) for s in SEEDS]
            covers = np.array([a["CTX_alpha"].astype(np.float64) >= ALPHA_MIN for a in arrays])
            ray = np.array([a["CTX_depth_z"].astype(np.float64) / np.maximum(a["CTX_alpha"].astype(np.float64), 1e-12)
                            for a in arrays]) / cosine
            count = covers.sum(axis=0)
            covered = count >= 2
            jo_value = np.where(covers, ray, 0.0).sum(axis=0) / np.maximum(count, 1)
            jo = np.where(covered, jo_value, h3)
            hi = np.where(covers, ray, -np.inf).max(axis=0)
            lo = np.where(covers, ray, np.inf).min(axis=0)
            spread = np.where(covered, (hi - lo) / np.maximum(jo_value, 1e-6), np.inf)
            pair = np.abs(jo - g1n) / np.maximum((jo + g1n) / 2, 1e-6)
            variants = {"H3": h3, "JO": jo, "HJO": np.where(covered, (h3 + jo) / 2, h3), "G1N": g1n,
                        "HGN": (h3 + g1n) / 2, "HJOG": np.where(covered, (h3 + jo + g1n) / 3, (h3 + g1n) / 2),
                        "JOG1_AVG": np.where(covered, (jo + g1n) / 2, g1n)}
            for t in (0.05, 0.10, 0.20):
                variants[f"JO_SEED{t:.2f}"] = np.where(covered & (spread <= t), jo, h3)
            for t in (0.10, 0.20, 0.30):
                variants[f"JOG1_AGREE{t:.2f}"] = np.where(covered & g1_conf & (pair <= t), (jo + g1n) / 2, h3)
            gt = np.load(frames[qid]["depth"]).squeeze().astype(np.float64)
            farh = np.isfinite(gt) & (gt > 0) & ~near
            if not farh.any():
                continue
            for name, x in variants.items():
                grouped[f"{name}|FARH"][e["scene_id"]][e["role"]].append(
                    float(np.mean(np.abs(x[farh] - gt[farh]) / gt[farh])))
                grouped[f"{name}|USED"][e["scene_id"]][e["role"]].append(float(np.mean(x[farh] != h3[farh])))
    table = {n: {s: float(np.mean([np.mean(v) for v in r.values()])) for s, r in sc.items()} for n, sc in grouped.items()}
    names = sorted({k.split("|")[0] for k in table} - {"H3"})
    result = {}
    print(f"{'variant':16s} {'FARH':>7s} {'used':>6s}  H3 - X")
    for name in names:
        s = contrast(table, "H3|FARH", f"{name}|FARH")
        result[name] = {"mean_farh": float(np.mean(list(table[f"{name}|FARH"].values()))),
                        "used_share": float(np.mean(list(table[f"{name}|USED"].values()))), "contrast": s}
        print(f"{name:16s} {result[name]['mean_farh']:7.4f} {result[name]['used_share']:6.3f}  "
              f"{s['mean']:+.4f} [{s['ci95'][0]:+.4f}, {s['ci95'][1]:+.4f}] {s['positive']}/{s['tied']}/{s['negative']} "
              f"median {s['median']:+.4f} {s['status']}")
    print("H3 FARH", round(float(np.mean(list(table["H3|FARH"].values()))), 4), "scenes", len(table["H3|FARH"]))
    write(root / "exploratory_posthoc/gating_explore.json", {"note": "exploratory, in-sample", "results": result})


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve())
