"""EXPLORATORY (post hoc, in-sample on ARKIT_SYSTEM after its locked analysis): does a carrier built from all SEVEN
views (raw/eval7, eval_carrier_7view.py) add to Core B where the 3-context carrier did not? On FAR7 pixels (outside
the 8-px NEAR of the 7-frame reprojection), R7 is replaced by:
  GA7          the 7-view GA, normalized per seed (depth / opacity where opacity >= 0.5, else R7), mean over seeds
  GA3          the 3-context GA of the locked run (reference)
  GEO_GA7      sqrt(R7 x GA7)
under the selections ALL (every episode), S3 (3-context spread >= 1.2, the locked selector), S7 (7-frame spread
>= its in-sample median); the system is sqrt(R7S x TTT_F7S) as in the locked FULL. Reports FAR7 and ALL AbsRel and
H = CORE_B - X with the V2 labels (scene bootstrap). In-sample: a promising variant needs a fresh test.
Run through cv_host_run.py from the repository root (CUDA hidden):  python full7_explore.py ROOT
"""
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "EXP-3D-RGBD-DEPTHSPLAT-JEPA-TTT-V1" / "code"))
from score_analyze import contrast, write  # noqa: E402

COHORT = "ARKIT_SYSTEM"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(item):
    if sha(item["path"]) != item["sha256"]:
        raise PermissionError(f"sealed prediction changed: {item['path']}")
    return dict(np.load(item["path"]))


def normalized(p, arm, seeds, fallback):
    return np.mean([np.where(p[f"OPACITY_{arm}_{s}"] >= 0.5,
                             p[f"{arm}_{s}"] / np.maximum(p[f"OPACITY_{arm}_{s}"].astype(np.float64), 1e-6), fallback)
                    for s in seeds], axis=0)


def spread(depths):
    d = np.concatenate([x[np.isfinite(x) & (x > 0)] for x in depths])
    return float((np.percentile(d, 90) - np.percentile(d, 10)) / np.median(d))


def main(root):
    lock = json.loads((root / "frame_role_lock.json").read_text())["cohorts"][COHORT]
    records = {r["scene_id"]: r for r in json.loads(Path(lock["manifest"]).read_text())["scenes"]}
    arms = json.loads((root / f"raw/eval/{COHORT}/prediction_seal.json").read_text())
    seven = json.loads((root / f"raw/eval7/{COHORT}/prediction_seal.json").read_text())
    geometry = json.loads((root / f"geometry/geometry_seal_{COHORT}.json").read_text())
    selector = json.loads((root / "selector_seal.json").read_text())["episodes"]
    zs = {}
    for part in sorted((root / f"ds/{COHORT}/ZS").glob("ds_seal_*of*.json")):
        zs.update(json.loads(part.read_text())["predictions"])
    s7 = {}
    for e in lock["episodes"]:
        frames = {f["frame_id"]: f for f in records[e["scene_id"]]["frames"]}
        s7[(e["scene_id"], e["role"])] = spread([np.load(frames[c]["depth"]).squeeze().astype(np.float64)
                                                 for c in e["context"] + e["stream"]])
    s7_median = float(np.median(list(s7.values())))
    gm = lambda a, b: np.sqrt(np.maximum(a, 1e-6) * np.maximum(b, 1e-6))  # noqa: E731
    grouped = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for e in lock["episodes"]:
        frames = {f["frame_id"]: f for f in records[e["scene_id"]]["frames"]}
        selections = {"ALL": True, "S3": selector[f"{e['scene_id']}/{e['role']}"]["use_learned"],
                      "S7": s7[(e["scene_id"], e["role"])] >= s7_median}
        for qid in e["queries"]:
            key = f"{e['scene_id']}/{e['role']}/{qid}"
            geo, z = load(geometry["predictions"][key]), load(zs[key])
            p3, p7 = load(arms["predictions"][f"{COHORT}/{key}"]), load(seven["predictions"][f"{COHORT}/{key}"])
            r7, near7, cosine = geo["REPROJ7_HARMONIC"], geo["near7"].astype(bool), geo["cosine"]
            far7 = ~near7
            cover = z["TTT_m1_alpha"].astype(np.float64) >= 0.5
            ray = z["TTT_m1_depth_z"].astype(np.float64) / np.maximum(z["TTT_m1_alpha"].astype(np.float64), 1e-12) / cosine
            ga7, ga3 = normalized(p7, "GA", seven["seeds"], r7), normalized(p3, "GA", arms["seeds"], r7)
            g17 = normalized(p7, "G1", seven["seeds"], r7)
            fills = {"GA7": ga7, "GA3": ga3, "G17": g17, "GEO_GA7": gm(r7, ga7)}
            readouts = {"R7": r7, "CORE_B": gm(r7, np.where(cover, ray, r7))}
            for fname, fill in fills.items():
                readouts[f"FILL_{fname}"] = np.where(far7, fill, r7)
                for sname, use in selections.items():
                    base = np.where(far7 & use, fill, r7)
                    readouts[f"SYS_{fname}_{sname}"] = gm(base, np.where(cover, ray, base))
            gt = np.load(frames[qid]["depth"]).squeeze().astype(np.float64)
            valid = np.isfinite(gt) & (gt > 0)
            for name, x in readouts.items():
                for region, mask in (("ALL", valid), ("FAR7", valid & far7)):
                    if mask.any():
                        grouped[f"{name}|{region}"][e["scene_id"]][e["role"]].append(
                            float(np.mean(np.abs(x[mask] - gt[mask]) / gt[mask])))
    table = {k: {s: float(np.mean([np.mean(v) for v in r.values()])) for s, r in sc.items()} for k, sc in grouped.items()}
    result = {"s7_median": s7_median, "means": {k: float(np.mean(list(v.values()))) for k, v in sorted(table.items())},
              "contrasts": {}}
    print("S7 median", round(s7_median, 3))
    for k in ("R7", "CORE_B", "FILL_GA3", "FILL_GA7", "FILL_G17", "FILL_GEO_GA7"):
        print(f"  {k:18s} FAR7 {result['means'][k + '|FAR7']:.4f}  ALL {result['means'][k + '|ALL']:.4f}")
    for k in ("FILL_GA7", "FILL_G17", "FILL_GEO_GA7", "FILL_GA3"):
        s = contrast(table, "R7|FAR7", f"{k}|FAR7")
        result["contrasts"][f"R7-{k}|FAR7"] = s
        print(f"  R7 - {k:14s} FAR7 {s['mean']:+.4f} [{s['ci95'][0]:+.4f}, {s['ci95'][1]:+.4f}] "
              f"{s['positive']}/{s['tied']}/{s['negative']} {s['status']}")
    for fname in ("GA7", "G17", "GEO_GA7"):
        for sname in ("ALL", "S3", "S7"):
            k = f"SYS_{fname}_{sname}"
            s = contrast(table, "CORE_B|ALL", f"{k}|ALL")
            result["contrasts"][f"CORE_B-{k}|ALL"] = s
            print(f"  CORE_B - {k:18s} ALL {s['mean']:+.4f} [{s['ci95'][0]:+.4f}, {s['ci95'][1]:+.4f}] "
                  f"{s['positive']}/{s['tied']}/{s['negative']} {s['status']}  (ALL {result['means'][k + '|ALL']:.4f})")
    write(root / "exploratory_posthoc/full7_explore.json", {"note": "exploratory, in-sample", **result})


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve())
