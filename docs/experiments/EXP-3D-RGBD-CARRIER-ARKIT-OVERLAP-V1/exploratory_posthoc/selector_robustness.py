"""EXPLORATORY (post hoc, in-sample) follow-up of scene_selector_explore.py: are the two signals robust?
For the per-query selector "use the learned completion X for this query when its feature >= threshold, else H3"
(decided per query, no query depth needed), with X in {GAN (ARKit carrier), G1N (Hypersim carrier), JO (overlap-aware
DepthSplat JEPA)}, features ctx_depth_spread and disagree (median |log(X / H3)| outside NEAR, computed for each X), and
thresholds at the 25/50/75th percentile of the per-query feature: FARH contrast H3 - SEL against H3 per source cohort
(ARKIT_VAL, ARKIT_CONFIRM) and pooled, plus Spearman rho of the scene feature with the scene gain per cohort.
Run through cv_host_run.py from the repository root (CUDA hidden):  python selector_robustness.py CARRIER_ROOT OVERLAP_ROOT
"""
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "EXP-3D-RGBD-DEPTHSPLAT-JEPA-TTT-V1" / "code"))
from score_analyze import write  # noqa: E402

COHORT, ALPHA_MIN = "ARKIT160_OVERLAP", 0.5


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


def ds_parts(root, name):
    preds = {}
    for part in sorted((root / f"ds/{COHORT}" / name).glob("ds_seal_*of*.json")):
        seal = json.loads(part.read_text())
        if seal["query_depth_read"] is not False or seal["limit"]:
            raise PermissionError(part)
        preds.update(seal["predictions"])
    return preds


def ray_cosine(record, qid):
    from mcss.measurement_planning import geometry as g
    from mcss.measurement_planning.information import TeacherAccess

    teacher = TeacherAccess(record, [qid], (128, 160), Path.cwd(), [])
    return np.asarray(g.ray_cosine(teacher.query_camera(qid)), dtype=np.float64)


def rank(x):
    return np.argsort(np.argsort(x, kind="stable"), kind="stable").astype(float)


def main(carrier, overlap):
    lock = json.loads((carrier / "frame_role_lock.json").read_text())["cohorts"][COHORT]
    records = {r["scene_id"]: r for r in json.loads(Path(lock["manifest"]).read_text())["scenes"]}
    arms = json.loads((carrier / f"raw/eval/{COHORT}/prediction_seal.json").read_text())
    jo = {s: ds_parts(overlap, s) for s in ("JO1", "JO2", "JO3")}
    rows = []
    for e in lock["episodes"]:
        record = records[e["scene_id"]]
        frames = {f["frame_id"]: f for f in record["frames"]}
        ctx = np.concatenate([d[np.isfinite(d) & (d > 0)] for d in
                              (np.load(frames[c]["depth"]).squeeze().astype(np.float64) for c in e["context"])])
        spread = float((np.percentile(ctx, 90) - np.percentile(ctx, 10)) / np.median(ctx))
        for qid in e["queries"]:
            key = f"{e['scene_id']}/{e['role']}/{qid}"
            p = load(arms["predictions"][f"{COHORT}/{key}"])
            near = p["near"].astype(bool)
            if not near.any():
                continue
            h3 = p["REPROJ_HARMONIC"]
            far = ~near
            cosine = ray_cosine(record, qid)
            arrays = [load(jo[s][key]) for s in jo]
            covers = np.array([a["CTX_alpha"].astype(np.float64) >= ALPHA_MIN for a in arrays])
            ray = np.array([a["CTX_depth_z"].astype(np.float64) / np.maximum(a["CTX_alpha"].astype(np.float64), 1e-12)
                            for a in arrays]) / cosine
            count = covers.sum(axis=0)
            preds = {"GAN": normalized(p, "GA", arms["seeds"], h3), "G1N": normalized(p, "G1", arms["seeds"], h3),
                     "JO": np.where(count >= 2, np.where(covers, ray, 0.0).sum(axis=0) / np.maximum(count, 1), h3)}
            gt = np.load(frames[qid]["depth"]).squeeze().astype(np.float64)
            farh = np.isfinite(gt) & (gt > 0) & far
            if not farh.any():
                continue
            absrel = lambda x: float(np.mean(np.abs(x[farh] - gt[farh]) / gt[farh]))  # noqa: E731
            row = {"scene_id": e["scene_id"], "role": e["role"], "cohort": record["source_cohort"],
                   "spread": spread, "h3": absrel(h3)}
            for name, x in preds.items():
                row[f"disagree_{name}"] = float(np.median(np.abs(np.log(np.maximum(x[far], 1e-6) / h3[far]))))
                row[f"err_{name}"] = absrel(x)
            rows.append(row)

    def scene_values(rows, value):
        grouped = defaultdict(lambda: defaultdict(list))
        for r in rows:
            grouped[r["scene_id"]][r["role"]].append(value(r))
        return {s: float(np.mean([np.mean(v) for v in roles.values()])) for s, roles in grouped.items()}

    result = {}
    cohort_of = {r["scene_id"]: r["cohort"] for r in rows}
    for model in ("GAN", "G1N", "JO"):
        for feature in ("spread", f"disagree_{model}"):
            values = np.array([r[feature] for r in rows])
            for q in (25, 50, 75):
                t = float(np.percentile(values, q))
                gain = scene_values(rows, lambda r, f=feature, t=t, m=model: r["h3"] - (r[f"err_{m}"] if r[f] >= t else r["h3"]))
                line = {}
                for cohort in ("ALL", "ARKIT_VAL", "ARKIT_CONFIRM"):
                    g = np.array([v for s, v in gain.items() if cohort in ("ALL", cohort_of[s])])
                    line[cohort] = {"mean": float(g.mean()), "positive": int((g > 1e-12).sum()),
                                    "tied": int((np.abs(g) <= 1e-12).sum()), "negative": int((g < -1e-12).sum()),
                                    "non_worse": float((g >= -1e-12).mean())}
                result[f"{model}|{feature}>=p{q}"] = {"threshold": t, **line}
                a, v, c = line["ALL"], line["ARKIT_VAL"], line["ARKIT_CONFIRM"]
                print(f"{model:4s} {feature:14s} >= p{q} (t={t:.3f}): ALL {a['mean']:+.4f} {a['positive']}/{a['tied']}/"
                      f"{a['negative']} nw {a['non_worse']:.2f} | VAL {v['mean']:+.4f} nw {v['non_worse']:.2f} | "
                      f"CONFIRM {c['mean']:+.4f} nw {c['non_worse']:.2f}")
            base = scene_values(rows, lambda r, m=model: r["h3"] - r[f"err_{m}"])
            feat = scene_values(rows, lambda r, f=feature: r[f])
            for cohort in ("ARKIT_VAL", "ARKIT_CONFIRM"):
                s_ids = [s for s in base if cohort_of[s] == cohort]
                rho = float(np.corrcoef(rank(np.array([feat[s] for s in s_ids])), rank(np.array([base[s] for s in s_ids])))[0, 1])
                result[f"{model}|{feature}|rho|{cohort}"] = rho
                print(f"     rho({feature}, gain {model}) in {cohort}: {rho:+.3f}")
    write(carrier / "exploratory_posthoc/selector_robustness.json", {"note": "exploratory, in-sample", **result})


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve())
