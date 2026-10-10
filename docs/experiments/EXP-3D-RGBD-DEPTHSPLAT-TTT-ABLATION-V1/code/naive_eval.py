"""Model-free Gaussian states (no network) rendered and test-time optimized exactly like ds_eval.py's DepthSplat states
(no query image or depth is read; queries contribute their cameras only).

NAIVE state of a set of measured RGB-D frames: one isotropic Gaussian per valid measured pixel of the 256x320 input
grid (the measured planar depth nearest-upsampled, as injected into DepthSplat), centred on the pixel-centre ray at
that depth with DepthSplat's own normalized pinhole camera; standard deviation = c x the pixel footprint at that depth
(z / focal in pixels); opacity o; colour = the pixel RGB as the SH DC term (higher SH zero). Arms as in ds_eval.py:
CTX (3 context frames), CACHE (the 3 context + 4 stream frames), TTT_m<m> (CACHE + ds_eval.ttt, the same objective,
parameters, base learning rates and steps). Files and seal as in ds_eval.py.
Run with the DepthSplat environment from the DepthSplat repository root (PYTHONPATH=.):
  python naive_eval.py ROOT COHORT OUT --model NV --scale-c 1.0 --opacity 0.95 [--arms CTX CACHE TTT]
                       [--ttt-mult 1] [--ttt-steps 100] [--shard 0/1] [--limit N]
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

for _directory in (Path(__file__).resolve().parents[1], Path(__file__).resolve().parent):  # the helpers' directory
    sys.path.insert(0, str(_directory))
import ds_predict  # noqa: E402
from ds_eval import FAR, NEAR, TTT_BASE_LR, cameras, query_render, sha, ttt  # noqa: E402
from ds_predict import ORTHO, build, measured_planar, views  # noqa: E402
from src.model.types import Gaussians  # noqa: E402
from train_jepa import SIZE  # noqa: E402

SH_C0 = 0.28209479177387814
D_SH = 9  # DepthSplat's sh_degree 2


@torch.no_grad()
def naive_gaussians(frames, ids, device, scale_c, opacity):
    context = views(frames, ids, SIZE, NEAR, FAR, device)
    depth = measured_planar(frames, ids, SIZE, device)[0]  # [V, H, W], 0 = invalid
    h, w = SIZE
    ys, xs = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing="ij")
    pixels = torch.stack([(xs.reshape(-1) + 0.5) / w, (ys.reshape(-1) + 0.5) / h,
                          torch.ones(h * w, device=device)], dim=-1)  # normalized image coordinates
    means, sigmas, colours = [], [], []
    for v in range(len(ids)):
        k, c2w = context["intrinsics"][0, v], context["extrinsics"][0, v]
        z = depth[v].reshape(-1)
        valid = z > 0
        rays = pixels @ torch.linalg.inv(k).T
        rays = rays / rays[:, 2:3]
        means.append((c2w[:3, 3] + (rays @ c2w[:3, :3].T) * z[:, None])[valid])
        sigmas.append((scale_c * z / (k[0, 0] * w))[valid])
        colours.append(context["image"][0, v].reshape(3, -1).T[valid])
    means, sigmas, colours = torch.cat(means), torch.cat(sigmas), torch.cat(colours)
    count = means.shape[0]
    covariances = sigmas[:, None, None] ** 2 * torch.eye(3, device=device)
    harmonics = torch.zeros(count, 3, D_SH, device=device)
    harmonics[:, :, 0] = (colours - 0.5) / SH_C0
    opacities = torch.full((count,), float(opacity), device=device)
    return Gaussians(means[None], covariances[None], harmonics[None], opacities[None])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("cohort")
    parser.add_argument("out", type=Path)
    parser.add_argument("--model", default="NV")
    parser.add_argument("--scale-c", type=float, required=True)
    parser.add_argument("--opacity", type=float, required=True)
    parser.add_argument("--arms", nargs="+", default=["CTX", "CACHE", "TTT"], choices=["CTX", "CACHE", "TTT"])
    parser.add_argument("--ttt-mult", type=float, nargs="+", default=[1.0])
    parser.add_argument("--ttt-steps", type=int, default=100)
    parser.add_argument("--shard", default="0/1")
    parser.add_argument("--limit", type=int, default=0, help="rehearsal only: first N episodes")
    args = parser.parse_args()
    index, count = (int(x) for x in args.shard.split("/"))
    root = args.root.resolve()
    lock = json.loads((root / "frame_role_lock.json").read_text())["cohorts"][args.cohort]
    if sha(lock["manifest"]) != lock["manifest_sha256"]:
        raise PermissionError("manifest changed since the frame-role lock")
    records = {r["scene_id"]: r for r in json.loads(Path(lock["manifest"]).read_text())["scenes"]}
    episodes = lock["episodes"][: args.limit] if args.limit else lock["episodes"]
    episodes = episodes[index::count]
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    device = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(20261008)
    _, decoder = build(device)  # the decoder has no parameters; the encoder is not used
    started, predictions, ttt_logs, sizes = time.time(), {}, {}, []
    for n, e in enumerate(episodes):
        frames = records[e["scene_id"]]["frames"]
        query_ids, stream_ids = e["queries"], e["context"] + e["stream"]
        if set(query_ids) & set(stream_ids):
            raise PermissionError("a query frame would enter the state")
        target = cameras(frames, query_ids, device)
        arrays = {}
        if "CTX" in args.arms:
            arrays["CTX"] = query_render(decoder, naive_gaussians(frames, e["context"], device, args.scale_c,
                                                                  args.opacity), target)
        if "CACHE" in args.arms or "TTT" in args.arms:
            cached = naive_gaussians(frames, stream_ids, device, args.scale_c, args.opacity)
            sizes.append(int(cached.means.shape[1]))
            if "CACHE" in args.arms:
                arrays["CACHE"] = query_render(decoder, cached, target)
            if "TTT" in args.arms:
                for mult in args.ttt_mult:
                    name = f"TTT_m{mult:g}"
                    optimized, log = ttt(decoder, cached, frames, stream_ids, args.ttt_steps, mult, device)
                    arrays[name] = query_render(decoder, optimized, target)
                    ttt_logs[f"{e['scene_id']}/{e['role']}/{name}"] = log
                    del optimized
            del cached
        for j, qid in enumerate(query_ids):
            data = {}
            for arm, (depth, alpha) in arrays.items():
                data[f"{arm}_depth_z"] = depth[j]
                data[f"{arm}_alpha"] = alpha[j]
            path = out / f"{e['scene_id']}_{e['role']}_{qid}.npz"
            np.savez_compressed(path, **data)
            predictions[f"{e['scene_id']}/{e['role']}/{qid}"] = {"path": str(path), "sha256": sha(path)}
        if n == 0 or (n + 1) % 10 == 0:
            print(f"episode {n + 1}/{len(episodes)} {time.time() - started:.0f}s", flush=True)
    setting = json.dumps({"naive": True, "scale_c": args.scale_c, "opacity": args.opacity}, sort_keys=True)
    info = {"cohort": args.cohort, "model": args.model, "checkpoint": "NONE (model-free Gaussians)",
            "checkpoint_sha256": setting, "scale_max": None, "naive_scale_c": args.scale_c,
            "naive_opacity": args.opacity, "arms": args.arms, "ttt_mult": args.ttt_mult, "ttt_steps": args.ttt_steps,
            "ttt_base_lr": TTT_BASE_LR, "overrides": list(ds_predict.OVERRIDES), "shard": args.shard,
            "limit": args.limit, "episodes": len(episodes), "queries": len(predictions), "query_depth_read": False,
            "query_image_read": False, "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0),
            "seconds": round(time.time() - started, 1),
            "max_memory_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
            "cache_gaussians_mean": float(np.mean(sizes)) if sizes else None,
            "pose_orthonormalization_max_correction": ORTHO["max_correction"], "ttt": ttt_logs,
            "predictions": predictions}
    (out / f"ds_seal_{index}of{count}.json").write_text(json.dumps(info, indent=1))
    print("NAIVE_EVAL_DONE", args.model, args.cohort, len(predictions), "queries", info["seconds"], "s",
          info["max_memory_gb"], "GB", flush=True)


if __name__ == "__main__":
    main()
