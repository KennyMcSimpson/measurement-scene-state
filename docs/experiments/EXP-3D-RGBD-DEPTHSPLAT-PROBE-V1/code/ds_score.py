"""Score DepthSplat beside harmonic filling and G1 on a probe's sealed queries (exploratory; one cohort).

Inputs: the arms-evaluator seal of the probe root (REPROJ_HARMONIC, NEAR mask, G1 x 3 seeds) and
ds_predict.py's renders at 128x160: DepthSplat's unnormalized planar depth (sum of alpha * T * z on a
zero background), the accumulated opacity alpha of that blend, and the model's context depths.
Query depth is read here only, after both prediction sets exist. Depths become ray distance with each
query's intrinsics.
  DS              the decoder's own depth render
  DSN             depth_z / alpha where alpha >= --alpha-min ("covered"), 0 elsewhere
  *_SCALED        times one scale per scene and role: the median ratio of measured to DepthSplat
                  context depth (both as ray distance at 128x160)
  DSN_OR_HARM     DSN where covered, REPROJ_HARMONIC elsewhere
  FAR_DSN_SCALED  REPROJ_HARMONIC in NEAR and in uncovered FAR, DSN_SCALED in covered FAR
  FAR_G1          REPROJ_HARMONIC in NEAR, G1 (mean over seeds) in FAR: the project's AHFILL8-style hybrid
  FAR_DSN_G1      REPROJ_HARMONIC in NEAR, DSN_SCALED in covered FAR, G1 (mean over seeds) in uncovered FAR
Regions: ALL, NEAR, FAR (the evaluator's NEAR mask), FAR_COV and FAR_UNCOV (FAR split by coverage).
Usage (repository root, project environment, through cv_host_run.py):
  python ds_score.py PROBE_ROOT DS_DIR [--alpha-min 0.5] [--cohort NAME (default: the contract's only cohort)]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REGIONS = ("ALL", "NEAR", "FAR", "FAR_COV", "FAR_UNCOV")
PAIRS = (
    ("REPROJ_HARMONIC", "DS", ("ALL", "FAR")),
    ("REPROJ_HARMONIC", "DSN_OR_HARM_SCALED", ("ALL", "NEAR", "FAR")),
    ("REPROJ_HARMONIC", "FAR_DSN_SCALED", ("ALL", "FAR")),
    ("G1", "FAR_DSN_SCALED", ("ALL", "FAR")),
    ("REPROJ_HARMONIC", "DSN_SCALED", ("FAR_COV",)),
    ("G1", "DSN_SCALED", ("FAR_COV",)),
    ("REPROJ_HARMONIC", "G1", ("ALL", "FAR", "FAR_COV", "FAR_UNCOV")),
    ("REPROJ_HARMONIC", "FAR_G1", ("ALL", "FAR")),
    ("FAR_G1", "FAR_DSN_G1", ("ALL", "FAR")),
    ("FAR_DSN_SCALED", "FAR_DSN_G1", ("ALL", "FAR")),
)


def ray_factor(intrinsics, shape):
    k = np.array(intrinsics, dtype=np.float64)
    y, x = np.indices(shape)
    return np.sqrt(((x - k[0, 2]) / k[0, 0]) ** 2 + ((y - k[1, 2]) / k[1, 1]) ** 2 + 1.0)


def absrel(value, gt, mask):
    return float(np.mean(np.abs(value[mask] - gt[mask]) / gt[mask])) if mask.any() else None


def context_scale(record, role_key, context_z):
    """Median measured/predicted ray distance over valid context pixels (128x160)."""
    rows = {f["frame_id"]: f for f in record["frames"]}
    z = torch.from_numpy(context_z)[:, None].float()
    z = F.interpolate(z, size=(128, 160), mode="bilinear", align_corners=False)[:, 0].numpy()
    ratios = []
    for j, fid in enumerate(record["roles"][role_key]):
        measured = np.load(rows[fid]["depth"]).squeeze().astype(np.float64)
        predicted = z[j] * ray_factor(rows[fid]["intrinsics"], measured.shape)
        valid = np.isfinite(measured) & (measured > 0) & (predicted > 1e-3)
        ratios.append(measured[valid] / predicted[valid])
    return float(np.median(np.concatenate(ratios)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root")
    parser.add_argument("ds_dir")
    parser.add_argument("--alpha-min", type=float, default=0.5)
    parser.add_argument("--cohort")
    args = parser.parse_args()
    root, ds_dir = Path(args.root), Path(args.ds_dir)
    contract = json.loads((root / "evaluation_contract.json").read_text())
    seeds = contract["seeds"]
    cohort = args.cohort or next(iter(contract["cohorts"]))
    manifest = json.loads(Path(contract["cohorts"][cohort]["manifest"]).read_text())
    seal = json.loads((root / f"raw/eval/{cohort}/prediction_seal.json").read_text())
    rows, scales, coverage = [], {}, []
    for record in manifest["scenes"]:
        sid, frames = record["scene_id"], {f["frame_id"]: f for f in record["frames"]}
        for role, key in (("A", "context_a"), ("B", "context_b")):
            for qid in record["roles"]["primary_query"]:
                arrays = dict(np.load(seal["predictions"][f"{cohort}/{sid}/{role}/{qid}"]["path"]))
                ds = dict(np.load(ds_dir / f"{sid}_{role}_{qid}.npz"))
                gt = np.load(frames[qid]["depth"]).squeeze().astype(np.float64)
                valid = np.isfinite(gt) & (gt > 0)
                near = arrays["near"].astype(bool)
                alpha = ds["alpha"].astype(np.float64)
                covered = alpha >= args.alpha_min
                masks = {"ALL": valid, "NEAR": valid & near, "FAR": valid & ~near,
                         "FAR_COV": valid & ~near & covered, "FAR_UNCOV": valid & ~near & ~covered}
                ray = ray_factor(frames[qid]["intrinsics"], gt.shape)
                ds_ray = ds["depth_z"].astype(np.float64) * ray
                dsn_ray = np.where(covered, ds["depth_z"] / np.maximum(alpha, 1e-6), 0.0) * ray
                if (sid, role) not in scales:
                    scales[sid, role] = context_scale(record, key, ds["context_depth_z"])
                scale = scales[sid, role]
                harmonic = arrays["REPROJ_HARMONIC"]
                g1 = np.mean([arrays[f"G1_{seed}"] for seed in seeds], axis=0)
                methods = {
                    "REPROJ_HARMONIC": harmonic,
                    "DS": ds_ray,
                    "DS_SCALED": ds_ray * scale,
                    "DSN": dsn_ray,
                    "DSN_SCALED": dsn_ray * scale,
                    "DSN_OR_HARM": np.where(covered, dsn_ray, harmonic),
                    "DSN_OR_HARM_SCALED": np.where(covered, dsn_ray * scale, harmonic),
                    "FAR_DSN_SCALED": np.where(near | ~covered, harmonic, dsn_ray * scale),
                    "FAR_G1": np.where(near, harmonic, g1),
                    "FAR_DSN_G1": np.where(near, harmonic, np.where(covered, dsn_ray * scale, g1)),
                }
                for seed in seeds:
                    methods[f"G1_{seed}"] = arrays[f"G1_{seed}"]
                for name, value in methods.items():
                    row = {"scene_id": sid, "role": role, "query_id": qid,
                           "method": "G1" if name.startswith("G1_") else name, "seed": name[3:] if name.startswith("G1_") else None}
                    for region, mask in masks.items():
                        row[region] = absrel(value, gt, mask)
                    rows.append(row)
                coverage.append({
                    "scene_id": sid, "role": role, "query_id": qid,
                    "near_covered": float(covered[masks["NEAR"]].mean()) if masks["NEAR"].any() else None,
                    "far_covered": float(covered[masks["FAR"]].mean()) if masks["FAR"].any() else None,
                    "far_fraction": float(masks["FAR"].sum() / max(valid.sum(), 1)),
                    "alpha_mean_near": float(alpha[masks["NEAR"]].mean()) if masks["NEAR"].any() else None,
                    "alpha_mean_far": float(alpha[masks["FAR"]].mean()) if masks["FAR"].any() else None,
                })
    scenes = sorted({r["scene_id"] for r in rows})
    methods = list(dict.fromkeys(r["method"] for r in rows))

    def scene_mean(method, region):
        out = []
        for s in scenes:
            values = [r[region] for r in rows if r["scene_id"] == s and r["method"] == method and r[region] is not None]
            out.append(np.mean(values) if values else np.nan)
        return np.array(out)

    def cov_mean(field):
        return np.nanmean([np.mean([c[field] for c in coverage if c["scene_id"] == s and c[field] is not None] or [np.nan])
                           for s in scenes])

    print(f"scenes {len(scenes)}  queries {len(coverage)}  alpha_min {args.alpha_min}")
    print(f"scale per scene/role: median {np.median(list(scales.values())):.3f} "
          f"range [{min(scales.values()):.3f}, {max(scales.values()):.3f}]")
    print(f"FAR share of valid pixels {cov_mean('far_fraction'):.3f}; covered share: NEAR {cov_mean('near_covered'):.3f} "
          f"FAR {cov_mean('far_covered'):.3f}; mean alpha: NEAR {cov_mean('alpha_mean_near'):.3f} "
          f"FAR {cov_mean('alpha_mean_far'):.3f}")
    print(f"{'method':22s} " + " ".join(f"{r:>9s}" for r in REGIONS))
    for m in methods:
        print(f"{m:22s} " + " ".join(f"{np.nanmean(scene_mean(m, r)):9.4f}" for r in REGIONS))
    print("paired scene differences (a - b; positive = b better), better/worse scene counts for b:")
    for a, b, regions in PAIRS:
        for region in regions:
            d = scene_mean(a, region) - scene_mean(b, region)
            print(f"  {a} - {b} [{region}]: {np.nanmean(d):+.4f}  {int((d > 0).sum())}/{int((d < 0).sum())}")
    (ds_dir / "scores.json").write_text(json.dumps({
        "alpha_min": args.alpha_min, "rows": rows, "coverage": coverage,
        "scales": {f"{k[0]}/{k[1]}": v for k, v in scales.items()},
    }))


if __name__ == "__main__":
    main()
