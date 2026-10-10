"""Sanity checks of a ds_predict.py run on a probe (exploratory, DEV): is DepthSplat's geometry wired correctly?

1. Context depth: DepthSplat's own context-view depth (planar z at the model's input size, resized to 128x160,
   converted to ray distance) against the measured context depth: AbsRel raw and with the per-(scene, role) scale.
2. Query colour: the rendered colour against the query RGB, PSNR over covered pixels (alpha >= 0.5) and over all.
   Wrong poses or intrinsics would misalign the render and give a low PSNR even where depth error is moderate.
3. Query depth on covered NEAR pixels only (normalized depth z / alpha), against REPROJ_HARMONIC on the same pixels.
Usage (project environment, through cv_host_run.py): python ds_diag.py PROBE_ROOT DS_DIR
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def ray_factor(intrinsics, shape):
    k = np.array(intrinsics, dtype=np.float64)
    y, x = np.indices(shape)
    return np.sqrt(((x - k[0, 2]) / k[0, 0]) ** 2 + ((y - k[1, 2]) / k[1, 1]) ** 2 + 1.0)


def psnr(a, b):
    mse = float(np.mean((a - b) ** 2))
    return 10 * np.log10(1.0 / max(mse, 1e-12))


def main():
    root, ds_dir = Path(sys.argv[1]), Path(sys.argv[2])
    contract = json.loads((root / "evaluation_contract.json").read_text())
    manifest = json.loads(Path(contract["cohorts"]["DEV"]["manifest"]).read_text())
    seal = json.loads((root / "raw/eval/DEV/prediction_seal.json").read_text())
    ctx_rows, query_rows = [], []
    for record in manifest["scenes"]:
        sid, frames = record["scene_id"], {f["frame_id"]: f for f in record["frames"]}
        for role, key in (("A", "context_a"), ("B", "context_b")):
            qids = record["roles"]["primary_query"]
            ds0 = dict(np.load(ds_dir / f"{sid}_{role}_{qids[0]}.npz"))
            z = torch.from_numpy(ds0["context_depth_z"])[:, None].float()
            z = F.interpolate(z, size=(128, 160), mode="bilinear", align_corners=False)[:, 0].numpy()
            pairs = []
            for j, fid in enumerate(record["roles"][key]):
                measured = np.load(frames[fid]["depth"]).squeeze().astype(np.float64)
                predicted = z[j] * ray_factor(frames[fid]["intrinsics"], measured.shape)
                valid = np.isfinite(measured) & (measured > 0)
                pairs.append((measured[valid], predicted[valid]))
            scale = float(np.median(np.concatenate([m / np.maximum(p, 1e-6) for m, p in pairs])))
            raw = np.mean([np.mean(np.abs(p - m) / m) for m, p in pairs])
            scaled = np.mean([np.mean(np.abs(p * scale - m) / m) for m, p in pairs])
            delta = np.mean([np.mean(np.maximum(p / m, m / np.maximum(p, 1e-6)) < 1.25) for m, p in pairs])
            ctx_rows.append({"scene_id": sid, "role": role, "absrel": raw, "absrel_scaled": scaled, "delta1": delta,
                             "scale": scale})
            for qid in qids:
                ds = dict(np.load(ds_dir / f"{sid}_{role}_{qid}.npz"))
                arrays = dict(np.load(seal["predictions"][f"DEV/{sid}/{role}/{qid}"]["path"]))
                rgb = np.asarray(Image.open(frames[qid]["rgb"]).convert("RGB"), dtype=np.float64) / 255.0
                color = np.transpose(ds["color"], (1, 2, 0)).astype(np.float64)
                alpha = ds["alpha"].astype(np.float64)
                covered = alpha >= 0.5
                gt = np.load(frames[qid]["depth"]).squeeze().astype(np.float64)
                valid = np.isfinite(gt) & (gt > 0)
                near = arrays["near"].astype(bool)
                dsn = ds["depth_z"] / np.maximum(alpha, 1e-6) * ray_factor(frames[qid]["intrinsics"], gt.shape)
                mask = valid & near & covered
                query_rows.append({
                    "scene_id": sid, "role": role, "query_id": qid,
                    "psnr_all": psnr(color, rgb),
                    "psnr_covered": psnr(color[covered], rgb[covered]) if covered.any() else None,
                    "covered": float(covered.mean()),
                    "near_cov_dsn": float(np.mean(np.abs(dsn[mask] - gt[mask]) / gt[mask])) if mask.any() else None,
                    "near_cov_dsn_scaled": float(np.mean(np.abs(dsn[mask] * scale - gt[mask]) / gt[mask]))
                    if mask.any() else None,
                    "near_cov_harmonic": float(np.mean(np.abs(arrays["REPROJ_HARMONIC"][mask] - gt[mask]) / gt[mask]))
                    if mask.any() else None,
                })
    print("context depth (DepthSplat vs measured, per scene/role):")
    for r in ctx_rows:
        print(f"  {r['scene_id']} {r['role']}: AbsRel {r['absrel']:.3f}  scaled {r['absrel_scaled']:.3f}  "
              f"delta1 {r['delta1']:.3f}  scale {r['scale']:.3f}")
    print(f"  mean: AbsRel {np.mean([r['absrel'] for r in ctx_rows]):.3f}  "
          f"scaled {np.mean([r['absrel_scaled'] for r in ctx_rows]):.3f}  "
          f"delta1 {np.mean([r['delta1'] for r in ctx_rows]):.3f}")
    print("query render:")
    for field in ("psnr_all", "psnr_covered", "covered", "near_cov_dsn", "near_cov_dsn_scaled", "near_cov_harmonic"):
        values = [r[field] for r in query_rows if r[field] is not None]
        print(f"  {field:22s} mean {np.mean(values):.3f}  min {np.min(values):.3f}  max {np.max(values):.3f}")
    (ds_dir / "diag.json").write_text(json.dumps({"context": ctx_rows, "query": query_rows}, indent=1))


if __name__ == "__main__":
    main()
