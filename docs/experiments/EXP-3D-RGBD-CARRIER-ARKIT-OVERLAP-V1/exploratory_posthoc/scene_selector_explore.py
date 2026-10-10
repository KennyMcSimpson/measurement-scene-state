"""EXPLORATORY (post hoc, in-sample, after the locked analysis of EXP-3D-RGBD-CARRIER-ARKIT-OVERLAP-V1): which scenes
does learned completion make worse, and can they be recognised without query depth? On ARKit overlap queries GAN
(ARKit-trained carrier, normalized readout) lowers the mean FARH AbsRel by ~27 % but improves only 79/147 scenes.

Per query with a context hit, features that need no query depth:
  hit_share       share of query pixels hit by the context reprojection
  farh_share      share of query pixels outside NEAR
  ctx_depth_med   median measured context depth (m);  ctx_depth_spread = (p90 - p10) / median;  ctx_invalid share
  baseline        distance from the query camera to the nearest context camera / ctx_depth_med
  rotation_deg    rotation angle to the nearest context camera
  h3_far_ratio    median H3 depth outside NEAR / median hit depth
  disagree        median |log(GAN / H3)| outside NEAR
  ga_abstain      share of pixels outside NEAR where every GA seed abstains (opacity < 0.5)
  ga_spread       median relative spread of the confident GA seeds outside NEAR
Outcome (query depth, already scored): gain = AbsRel(H3) - AbsRel(X) on FARH for X = GAN and HGAN.
Scene value = mean over roles of the mean over queries. Reports Spearman rho of every feature with the scene gain
and, for each feature, the scene selectors "use X where feature >= / <= its 25/50/75th percentile, else H3" against
H3 (mean, positive/tied/negative). In-sample: any promising selector needs a fresh test.
Run through cv_host_run.py from the repository root (CUDA hidden):  python scene_selector_explore.py ROOT
"""
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "EXP-3D-RGBD-DEPTHSPLAT-JEPA-TTT-V1" / "code"))
from score_analyze import write  # noqa: E402

COHORT = "ARKIT160_OVERLAP"
FEATURES = ("hit_share", "farh_share", "ctx_depth_med", "ctx_depth_spread", "ctx_invalid", "baseline", "rotation_deg",
            "h3_far_ratio", "disagree", "ga_abstain", "ga_spread")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(item):
    if sha(item["path"]) != item["sha256"]:
        raise PermissionError(f"sealed prediction changed: {item['path']}")
    return dict(np.load(item["path"]))


def rotation_deg(a, b):
    r = np.asarray(a)[:3, :3].T @ np.asarray(b)[:3, :3]
    return float(np.degrees(np.arccos(np.clip((np.trace(r) - 1) / 2, -1.0, 1.0))))


def rank(x):
    order = np.argsort(np.argsort(x, kind="stable"), kind="stable").astype(float)
    return order


def spearman(x, y):
    return float(np.corrcoef(rank(np.asarray(x)), rank(np.asarray(y)))[0, 1])


def main(root):
    lock = json.loads((root / "frame_role_lock.json").read_text())["cohorts"][COHORT]
    records = {r["scene_id"]: r for r in json.loads(Path(lock["manifest"]).read_text())["scenes"]}
    arms = json.loads((root / f"raw/eval/{COHORT}/prediction_seal.json").read_text())
    seeds = arms["seeds"]
    per = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for e in lock["episodes"]:
        record = records[e["scene_id"]]
        frames = {f["frame_id"]: f for f in record["frames"]}
        ctx_depth = [np.load(frames[c]["depth"]).squeeze().astype(np.float64) for c in e["context"]]
        valid_ctx = np.concatenate([d[np.isfinite(d) & (d > 0)] for d in ctx_depth])
        ctx_med = float(np.median(valid_ctx))
        ctx_feats = {"ctx_depth_med": ctx_med,
                     "ctx_depth_spread": float((np.percentile(valid_ctx, 90) - np.percentile(valid_ctx, 10)) / ctx_med),
                     "ctx_invalid": float(1 - np.mean([np.mean(np.isfinite(d) & (d > 0)) for d in ctx_depth]))}
        for qid in e["queries"]:
            p = load(arms["predictions"][f"{COHORT}/{e['scene_id']}/{e['role']}/{qid}"])
            near, hit = p["near"].astype(bool), p["hit"].astype(bool)
            if not near.any():
                continue
            h3 = p["REPROJ_HARMONIC"]
            far_all = ~near
            opac = np.array([p[f"OPACITY_GA_{s}"].astype(np.float64) for s in seeds])
            depth = np.array([p[f"GA_{s}"] / np.maximum(p[f"OPACITY_GA_{s}"].astype(np.float64), 1e-6) for s in seeds])
            conf = opac >= 0.5
            gan = np.mean(np.where(conf, depth, h3), axis=0)
            n_conf = conf.sum(axis=0)
            mean_conf = np.where(conf, depth, 0).sum(axis=0) / np.maximum(n_conf, 1)
            spread = np.where(n_conf >= 2, (np.where(conf, depth, -np.inf).max(axis=0)
                                            - np.where(conf, depth, np.inf).min(axis=0)) / np.maximum(mean_conf, 1e-6),
                              np.nan)
            q_c2w = frames[qid]["c2w"]
            q_t = np.asarray(q_c2w)[:3, 3]
            dists = [np.linalg.norm(q_t - np.asarray(frames[c]["c2w"])[:3, 3]) for c in e["context"]]
            nearest = e["context"][int(np.argmin(dists))]
            feats = {**ctx_feats, "hit_share": float(hit.mean()), "farh_share": float(far_all.mean()),
                     "baseline": float(min(dists) / ctx_med), "rotation_deg": rotation_deg(frames[nearest]["c2w"], q_c2w),
                     "h3_far_ratio": float(np.median(h3[far_all]) / np.median(h3[hit])),
                     "disagree": float(np.median(np.abs(np.log(np.maximum(gan[far_all], 1e-6) / h3[far_all])))),
                     "ga_abstain": float(np.mean(n_conf[far_all] == 0)),
                     "ga_spread": float(np.nanmedian(spread[far_all])) if np.isfinite(spread[far_all]).any() else 0.0}
            gt = np.load(frames[qid]["depth"]).squeeze().astype(np.float64)
            farh = np.isfinite(gt) & (gt > 0) & far_all
            if not farh.any():
                continue
            absrel = lambda x: float(np.mean(np.abs(x[farh] - gt[farh]) / gt[farh]))  # noqa: E731
            outcome = {"gain_GAN": absrel(h3) - absrel(gan), "gain_HGAN": absrel(h3) - absrel((h3 + gan) / 2)}
            for k, v in {**feats, **outcome}.items():
                per[k][e["scene_id"]][e["role"]].append(v)
    table = {k: {s: float(np.mean([np.mean(v) for v in r.values()])) for s, r in sc.items()} for k, sc in per.items()}
    scenes = sorted(table["gain_GAN"])
    result = {"scenes": len(scenes), "features": {}}
    for target in ("gain_GAN", "gain_HGAN"):
        y = np.array([table[target][s] for s in scenes])
        print(f"{target}: mean {y.mean():+.4f}, positive {int((y > 0).sum())}/{len(y)}")
        for f in FEATURES:
            x = np.array([table[f][s] for s in scenes])
            rho = spearman(x, y)
            rules = {}
            for q in (25, 50, 75):
                t = float(np.percentile(x, q))
                for side, use in ((">=", x >= t), ("<=", x <= t)):
                    sel = np.where(use, y, 0.0)
                    rules[f"{side}p{q}"] = {"threshold": t, "mean": float(sel.mean()), "positive": int((sel > 0).sum()),
                                            "tied": int((sel == 0).sum()), "negative": int((sel < 0).sum())}
            best = max(rules.items(), key=lambda kv: ((kv[1]["positive"] + kv[1]["tied"]) / len(y) >= 0.75, kv[1]["mean"]))
            result["features"].setdefault(f, {})[target] = {"spearman": rho, "rules": rules}
            b = best[1]
            print(f"  {f:16s} rho {rho:+.3f}   best rule {best[0]:6s} (t={b['threshold']:.3f}): mean {b['mean']:+.4f} "
                  f"{b['positive']}/{b['tied']}/{b['negative']} non-worse {(b['positive'] + b['tied']) / len(y):.2f}")
    write(root / "exploratory_posthoc/scene_selector_explore.json", {"note": "exploratory, in-sample", **result})


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve())
