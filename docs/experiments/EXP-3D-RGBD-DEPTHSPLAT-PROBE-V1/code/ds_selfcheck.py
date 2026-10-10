"""Render DepthSplat's Gaussians back at their own context cameras (exploratory check of camera conventions).

With correct poses and intrinsics the render reproduces the context images closely and its normalized depth
(z / alpha) matches the encoder's own context depth. Optional --variant re-runs with a deliberately different
convention for comparison: "w2c" passes the inverse poses, "pixel_center" drops the +0.5 principal-point shift.
Run with the DepthSplat environment from the DepthSplat repository root (PYTHONPATH=.):
  python ds_selfcheck.py MANIFEST SPLIT [--variant none|w2c|pixel_center] [--queries]
--queries also renders the primary queries and reports their colour PSNR against the query images.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ds_predict import build, render_alpha, views  # noqa: E402


def apply_variant(batch, variant):
    if variant == "w2c":
        batch["extrinsics"] = torch.linalg.inv(batch["extrinsics"])
    elif variant == "pixel_center":
        batch["intrinsics"] = batch["intrinsics"].clone()
        batch["intrinsics"][..., 0, 2] -= 0.5 / 160
        batch["intrinsics"][..., 1, 2] -= 0.5 / 128
    return batch


def psnr(a, b):
    return 10 * np.log10(1.0 / max(float(((a - b) ** 2).mean()), 1e-12))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest")
    parser.add_argument("split")
    parser.add_argument("--variant", default="none", choices=("none", "w2c", "pixel_center"))
    parser.add_argument("--queries", action="store_true")
    args = parser.parse_args()
    device = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    encoder, decoder = build(device)
    records = [r for r in json.loads(Path(args.manifest).read_text())["scenes"] if r["split"] == args.split]
    ctx_rows, query_rows = [], []
    for record in records:
        for role, key in (("A", "context_a"), ("B", "context_b")):
            ids = record["roles"][key]
            context = apply_variant(views(record["frames"], ids, (256, 320), 0.5, 200.0, device), args.variant)
            same = apply_variant(views(record["frames"], ids, None, 0.5, 200.0, device), args.variant)
            with torch.no_grad():
                encoded = encoder(context, global_step=0, deterministic=True)
                gaussians = encoded["gaussians"]
                out = decoder.forward(gaussians, same["extrinsics"], same["intrinsics"], same["near"], same["far"],
                                      (128, 160), depth_mode="depth")
                alpha = render_alpha(gaussians, same, (128, 160))
                depth = encoded["depths"]
                while depth.dim() > 4:
                    depth = depth.squeeze(-1)
                depth = F.interpolate(depth[0][:, None], size=(128, 160), mode="bilinear", align_corners=False)[:, 0]
                for j, fid in enumerate(ids):
                    a = alpha[0, j]
                    covered = a >= 0.5
                    normalized = out.depth[0, j] / a.clamp_min(1e-6)
                    rel = ((normalized - depth[j]).abs() / depth[j].clamp_min(1e-6))[covered]
                    ctx_rows.append({
                        "scene_id": record["scene_id"], "role": role, "frame": fid,
                        "psnr": psnr(out.color[0, j], same["image"][0, j]), "alpha_mean": float(a.mean()),
                        "covered": float(covered.float().mean()),
                        "depth_vs_encoder_absrel": float(rel.mean()) if rel.numel() else None,
                    })
                if args.queries:
                    qids = record["roles"]["primary_query"]
                    target = apply_variant(views(record["frames"], qids, None, 0.5, 200.0, device), args.variant)
                    rendered = decoder.forward(gaussians, target["extrinsics"], target["intrinsics"], target["near"],
                                               target["far"], (128, 160))
                    q_alpha = render_alpha(gaussians, target, (128, 160))
                    for j, qid in enumerate(qids):
                        query_rows.append({"scene_id": record["scene_id"], "role": role, "frame": qid,
                                           "psnr": psnr(rendered.color[0, j], target["image"][0, j]),
                                           "covered": float((q_alpha[0, j] >= 0.5).float().mean())})
    print(f"variant {args.variant}: context self-render over {len(ctx_rows)} views")
    for field in ("psnr", "alpha_mean", "covered", "depth_vs_encoder_absrel"):
        values = [r[field] for r in ctx_rows if r[field] is not None]
        print(f"  {field:24s} mean {np.mean(values):.3f}  min {np.min(values):.3f}  max {np.max(values):.3f}")
    if query_rows:
        print(f"  query psnr mean {np.mean([r['psnr'] for r in query_rows]):.3f}  "
              f"covered mean {np.mean([r['covered'] for r in query_rows]):.3f}")
    print("SELFCHECK_DONE", json.dumps({"variant": args.variant, "context": ctx_rows, "query": query_rows}))


if __name__ == "__main__":
    main()
