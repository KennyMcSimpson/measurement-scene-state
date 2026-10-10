"""Fuse the extra views' depth maps of a ds_predict.py run into the query views, without Gaussian rendering
(exploratory, CPU). Query depth is read only for scoring, after all predictions exist.

For every record, role and primary query: each extra (non-context) view's depth from the run (context_depth_z holds
every input view's final depth at the model's input size: DepthSplat's prediction for RGB-only views, the measured
depth where it was injected) is back-projected with that view's camera and z-buffered into the query camera
(nearest pixel, ray distance). A query pixel gets a fused depth when at least --min-views extra views agree within
--tol (relative); the front-most agreeing candidate cluster is averaged. Methods:
  FUSE        REPROJ_HARMONIC in NEAR and where no agreement exists, the fused depth in the remaining FAR pixels
  FUSE_ANY    the same with min-views 1 (plain z-buffer of all extra views, no consistency check)
Reports AbsRel per region, FAR coverage, and REPROJ_HARMONIC - method per scene (bootstrap 95%, 10,000 draws,
seed 20260928). Usage (project environment, through cv_host_run.py):
  python ds_fuse.py PROBE_ROOT DS_DIR [--tol 0.05] [--min-views 2]
"""

import argparse
import json
from pathlib import Path

import numpy as np


def ray_factor(intrinsics, shape):
    k = np.array(intrinsics, dtype=np.float64)
    y, x = np.indices(shape)
    return np.sqrt(((x - k[0, 2]) / k[0, 0]) ** 2 + ((y - k[1, 2]) / k[1, 1]) ** 2 + 1.0)


def splat(z_map, frame, query, shape=(128, 160)):
    """Back-project a planar depth map at its own size (normalized intrinsics of the 128x160 frame) into the query:
    per-pixel minimum ray distance, inf where nothing lands."""
    h, w = z_map.shape
    k = np.array(frame["intrinsics"], dtype=np.float64)
    fxn, fyn, cxn, cyn = k[0, 0] / 160, k[1, 1] / 128, (k[0, 2] + 0.5) / 160, (k[1, 2] + 0.5) / 128
    v, u = np.indices((h, w))
    xn, yn = ((u + 0.5) / w - cxn) / fxn, ((v + 0.5) / h - cyn) / fyn
    valid = np.isfinite(z_map) & (z_map > 1e-3)
    z = z_map[valid]
    cam = np.stack([xn[valid] * z, yn[valid] * z, z, np.ones_like(z)])
    world = np.array(frame["c2w"], dtype=np.float64) @ cam
    q = np.linalg.inv(np.array(query["c2w"], dtype=np.float64)) @ world
    kq = np.array(query["intrinsics"], dtype=np.float64)
    front = q[2] > 1e-3
    q = q[:, front]
    px = np.rint(kq[0, 0] * q[0] / q[2] + kq[0, 2]).astype(int)
    py = np.rint(kq[1, 1] * q[1] / q[2] + kq[1, 2]).astype(int)
    inside = (px >= 0) & (px < shape[1]) & (py >= 0) & (py < shape[0])
    dist = np.linalg.norm(q[:3, inside], axis=0)
    out = np.full(shape, np.inf)
    np.minimum.at(out, (py[inside], px[inside]), dist)
    return out


def fuse(candidates, tol, min_views):
    """candidates [K, H, W] (inf = none) -> front-most cluster of >= min_views agreeing values (mean), nan if none."""
    c = np.sort(candidates, axis=0)
    best = np.full(c.shape[1:], np.nan)
    for j in range(c.shape[0]):  # front to back: the first anchor with enough agreeing views wins
        anchor = c[j]
        agree = np.abs(c - anchor[None]) <= tol * anchor[None]
        count = agree.sum(axis=0)
        mean = np.where(agree, c, 0.0).sum(axis=0) / np.maximum(count, 1)
        take = np.isnan(best) & np.isfinite(anchor) & (count >= min_views)
        best[take] = mean[take]
    return best


def bootstrap(d, draws=10_000, seed=20260928):
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, len(d), size=(draws, len(d)))].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root")
    parser.add_argument("ds_dir")
    parser.add_argument("--tol", type=float, default=0.05)
    parser.add_argument("--min-views", type=int, default=2)
    args = parser.parse_args()
    root, ds_dir = Path(args.root), Path(args.ds_dir)
    contract = json.loads((root / "evaluation_contract.json").read_text())
    cohort = next(iter(contract["cohorts"]))
    manifest = json.loads(Path(contract["cohorts"][cohort]["manifest"]).read_text())
    seal = json.loads((root / f"raw/eval/{cohort}/prediction_seal.json").read_text())
    info = json.loads((ds_dir / "run_info.json").read_text())
    view_ids = {(r["scene_id"], r["role"]): r["views"] for r in info["records"]}
    rows = []
    for record in manifest["scenes"]:
        sid, frames = record["scene_id"], {f["frame_id"]: f for f in record["frames"]}
        for role, key in (("A", "context_a"), ("B", "context_b")):
            views = view_ids[sid, role]
            n_ctx = len(record["roles"][key])
            for qid in record["roles"]["primary_query"]:
                ds = dict(np.load(ds_dir / f"{sid}_{role}_{qid}.npz"))
                depths = ds["context_depth_z"]
                cands = np.stack([splat(depths[j], frames[views[j]], frames[qid]) for j in range(n_ctx, len(views))])
                arrays = dict(np.load(seal["predictions"][f"{cohort}/{sid}/{role}/{qid}"]["path"]))
                harmonic, near = arrays["REPROJ_HARMONIC"], arrays["near"].astype(bool)
                fused = fuse(cands, args.tol, args.min_views)
                fused_any = fuse(cands, args.tol, 1)
                gt = np.load(frames[qid]["depth"]).squeeze().astype(np.float64)
                valid = np.isfinite(gt) & (gt > 0)
                far = valid & ~near
                methods = {"REPROJ_HARMONIC": harmonic,
                           "FUSE": np.where(~near & np.isfinite(fused), fused, harmonic),
                           "FUSE_ANY": np.where(~near & np.isfinite(fused_any), fused_any, harmonic)}
                row = {"scene_id": sid, "role": role, "query_id": qid,
                       "far_covered": float(np.isfinite(fused)[far].mean()) if far.any() else None,
                       "far_covered_any": float(np.isfinite(fused_any)[far].mean()) if far.any() else None}
                for name, value in methods.items():
                    for region, mask in (("ALL", valid), ("NEAR", valid & near), ("FAR", far)):
                        row[f"{name}/{region}"] = (float(np.mean(np.abs(value[mask] - gt[mask]) / gt[mask]))
                                                   if mask.any() else None)
                rows.append(row)
    scenes = sorted({r["scene_id"] for r in rows})

    def per_scene(field):
        return np.array([np.mean([r[field] for r in rows if r["scene_id"] == s and r[field] is not None] or [np.nan])
                         for s in scenes])

    print(f"{ds_dir.name}: {len(scenes)} scenes, tol {args.tol}, min-views {args.min_views}; FAR covered "
          f"{np.nanmean(per_scene('far_covered')):.3f} (any {np.nanmean(per_scene('far_covered_any')):.3f})")
    for name in ("REPROJ_HARMONIC", "FUSE", "FUSE_ANY"):
        print(f"  {name:16s} " + " ".join(f"{region} {np.nanmean(per_scene(f'{name}/{region}')):.4f}"
                                          for region in ("ALL", "NEAR", "FAR")))
    result = {}
    for name in ("FUSE", "FUSE_ANY"):
        for region in ("FAR", "ALL"):
            d = per_scene(f"REPROJ_HARMONIC/{region}") - per_scene(f"{name}/{region}")
            lo, hi = bootstrap(d[np.isfinite(d)])
            result[f"{name}/{region}"] = {"mean": float(np.nanmean(d)), "ci95": [lo, hi],
                                          "better": int((d > 0).sum()), "worse": int((d < 0).sum())}
            print(f"  harmonic - {name} [{region}]: {np.nanmean(d):+.4f} [{lo:+.4f}, {hi:+.4f}] "
                  f"{int((d > 0).sum())}/{int((d < 0).sum())}")
    (ds_dir / f"fuse_tol{args.tol}_k{args.min_views}.json").write_text(
        json.dumps({"args": vars(args), "rows": rows, "summary": result}, indent=1))


if __name__ == "__main__":
    main()
