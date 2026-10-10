"""EXP-3D-RGBD-DEPTHSPLAT-JEPA-TTT-V1: DepthSplat states of every episode rendered at its query cameras (no query
image or depth is read; queries contribute their cameras only).

For one state builder (released DepthSplat weights or a reverse-JEPA checkpoint, with its Gaussian scale cap) and
every episode of a cohort's frame-role lock:
  CTX:     the three context RGB-D views (measured depth injected where valid) -> Gaussians -> query render;
  CACHE:   the three context and four stream RGB-D views as one seven-view input -> Gaussians -> query render;
  TTT_m<m>: the CACHE Gaussians after --ttt-steps Adam steps on the seven measured frames with the training loss
           (relative L1 of the raw rendered depth on valid measured pixels + 0.5 x RGB L1); means, per-axis
           log-scales (in each Gaussian's own eigenbasis; rotations fixed), opacity logits and SH coefficients are
           optimized with m x the 3DGS default learning rates (means 1.6e-4 x the median measured planar depth of
           the seven frames, log-scales 5e-3, opacity logits 2.5e-2, SH DC 2.5e-3, higher SH 1.25e-4), eps 1e-15.
Each query file holds, per arm, depth_z (sum of alpha * T * planar z on a zero background) and alpha at 128x160.
Files are hashed into ds_seal_<shard>.json with the run settings.
Run with the DepthSplat environment from the DepthSplat repository root (PYTHONPATH=.):
  python ds_eval.py ROOT COHORT OUT --model NAME [--checkpoint PTH] [--scale-max 0.1] [--arms CTX CACHE TTT]
                    [--ttt-mult 1] [--ttt-steps 100] [--shard 0/1]
"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

for _directory in (Path(__file__).resolve().parents[1], Path(__file__).resolve().parent):  # the helpers' directory
    sys.path.insert(0, str(_directory))
import ds_predict  # noqa: E402
from ds_finetune import measured_z  # noqa: E402
from ds_predict import INJECTED, ORTHO, build, inject_measured_depth, measured_planar, orthonormal  # noqa: E402
from ds_predict import render_alpha, views  # noqa: E402
from src.model.types import Gaussians  # noqa: E402
from train_jepa import RENDER, SIZE, grounded_loss, render, set_scale_max  # noqa: E402

TTT_BASE_LR = {"means_per_metre": 1.6e-4, "log_scales": 5e-3, "opacity_logits": 2.5e-2, "sh_dc": 2.5e-3,
               "sh_rest": 1.25e-4}
NEAR, FAR = 0.5, 200.0


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cameras(frames, ids, device, image_size=(128, 160)):
    """Query cameras exactly as ds_predict.views builds them, without opening any query file."""
    rows = {f["frame_id"]: f for f in frames}
    h0, w0 = image_size
    intrinsics, extrinsics = [], []
    for i in ids:
        k = np.array(rows[i]["intrinsics"], dtype=np.float64)
        intrinsics.append(
            [[k[0, 0] / w0, 0.0, (k[0, 2] + 0.5) / w0], [0.0, k[1, 1] / h0, (k[1, 2] + 0.5) / h0], [0.0, 0.0, 1.0]]
        )
        extrinsics.append(orthonormal(np.array(rows[i]["c2w"], dtype=np.float64)))
    n = len(ids)
    return {
        "intrinsics": torch.tensor(np.array(intrinsics), dtype=torch.float32)[None].to(device),
        "extrinsics": torch.tensor(np.array(extrinsics), dtype=torch.float32)[None].to(device),
        "near": torch.full((1, n), NEAR, device=device),
        "far": torch.full((1, n), FAR, device=device),
    }


@torch.no_grad()
def encode(encoder, frames, ids, device):
    context = views(frames, ids, SIZE, NEAR, FAR, device)
    INJECTED["depth"] = measured_planar(frames, ids, SIZE, device)
    INJECTED["measured_views"] = len(ids)
    encoded = encoder(context, global_step=0, deterministic=True)
    gaussians = encoded["gaussians"] if isinstance(encoded, dict) else encoded
    return Gaussians(gaussians.means, gaussians.covariances, gaussians.harmonics, gaussians.opacities)


@torch.no_grad()
def query_render(decoder, gaussians, target):
    depth, _ = render(decoder, gaussians, target)
    alpha = render_alpha(gaussians, target, RENDER)[0]
    return depth.float().cpu().numpy(), alpha.float().cpu().numpy()


def ttt(decoder, gaussians, frames, ids, steps, mult, device):
    """Test-time optimization of the Gaussian state on the measured frames `ids`; returns it and a log."""
    target = views(frames, ids, None, NEAR, FAR, device)
    z = measured_z(frames, ids).to(device)
    rgb = target["image"][0]
    extent = float(z[z > 0].median())
    means0 = gaussians.means.detach()
    cov0 = gaussians.covariances.detach()
    evals, evecs = torch.linalg.eigh(cov0.double())
    log_s0 = (0.5 * evals.clamp_min(1e-30).log()).float()
    evecs = evecs.float()
    opacity0 = gaussians.opacities.detach().clamp(1e-6, 1 - 1e-6)
    logit0 = torch.log(opacity0) - torch.log1p(-opacity0)
    sh0 = gaussians.harmonics.detach()
    d_means = torch.zeros_like(means0, requires_grad=True)
    d_logs = torch.zeros_like(log_s0, requires_grad=True)
    d_logit = torch.zeros_like(logit0, requires_grad=True)
    d_dc = torch.zeros_like(sh0[..., :1], requires_grad=True)
    d_rest = torch.zeros_like(sh0[..., 1:], requires_grad=True)
    optimizer = torch.optim.Adam([
        {"params": [d_means], "lr": mult * TTT_BASE_LR["means_per_metre"] * extent},
        {"params": [d_logs], "lr": mult * TTT_BASE_LR["log_scales"]},
        {"params": [d_logit], "lr": mult * TTT_BASE_LR["opacity_logits"]},
        {"params": [d_dc], "lr": mult * TTT_BASE_LR["sh_dc"]},
        {"params": [d_rest], "lr": mult * TTT_BASE_LR["sh_rest"]},
    ], eps=1e-15)

    def state():
        scales2 = torch.exp(2 * (log_s0 + d_logs))
        covariances = (evecs * scales2[..., None, :]) @ evecs.transpose(-1, -2)
        return Gaussians(means0 + d_means, covariances, sh0 + torch.cat([d_dc, d_rest], dim=-1),
                         torch.sigmoid(logit0 + d_logit))

    with torch.no_grad():
        start = state()
        rebuild = float(((start.covariances - cov0).norm(dim=(-2, -1)) /
                         cov0.norm(dim=(-2, -1)).clamp_min(1e-20)).max())
    losses, failed = [], False
    for _ in range(steps):
        depth, color = render(decoder, state(), target)
        loss, depth_term = grounded_loss(depth, color, z, rgb)
        if not torch.isfinite(loss):  # preregistered: the episode keeps its CACHE state, the failure is recorded
            failed = True
            with torch.no_grad():
                for delta in (d_means, d_logs, d_logit, d_dc, d_rest):
                    delta.zero_()
            break
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append((float(loss), depth_term))
    with torch.no_grad():
        final = state()
        depth, color = render(decoder, final, target)
        loss, depth_term = grounded_loss(depth, color, z, rgb)
    log = {"extent_m": extent, "covariance_rebuild_max_rel_error": rebuild, "failed_reverted_to_cache": failed,
           "steps_done": len(losses), "loss_first": losses[0][0] if losses else None,
           "depth_first": losses[0][1] if losses else None, "loss_final": float(loss), "depth_final": depth_term,
           "mean_shift_max_m": float(d_means.detach().norm(dim=-1).max())}
    return Gaussians(*(t.detach() for t in (final.means, final.covariances, final.harmonics, final.opacities))), log


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("cohort")
    parser.add_argument("out", type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--checkpoint", default=str(ds_predict.CHECKPOINT))
    parser.add_argument("--scale-max", type=float, required=True)
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
    set_scale_max(args.scale_max)
    encoder, decoder = build(device, Path(args.checkpoint))
    inject_measured_depth(encoder)
    INJECTED["align"] = False
    started, predictions, ttt_logs = time.time(), {}, {}
    for n, e in enumerate(episodes):
        frames = records[e["scene_id"]]["frames"]
        query_ids, stream_ids = e["queries"], e["context"] + e["stream"]
        if set(query_ids) & set(stream_ids):
            raise PermissionError("a query frame would enter the state")
        target = cameras(frames, query_ids, device)
        arrays = {}
        if "CTX" in args.arms:
            arrays["CTX"] = query_render(decoder, encode(encoder, frames, e["context"], device), target)
        if "CACHE" in args.arms or "TTT" in args.arms:
            cached = encode(encoder, frames, stream_ids, device)
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
    info = {"cohort": args.cohort, "model": args.model, "checkpoint": str(args.checkpoint),
            "checkpoint_sha256": sha(args.checkpoint), "scale_max": args.scale_max, "arms": args.arms,
            "ttt_mult": args.ttt_mult, "ttt_steps": args.ttt_steps, "ttt_base_lr": TTT_BASE_LR,
            "overrides": list(ds_predict.OVERRIDES), "shard": args.shard, "limit": args.limit,
            "episodes": len(episodes), "queries": len(predictions), "query_depth_read": False,
            "query_image_read": False, "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0),
            "seconds": round(time.time() - started, 1), "max_memory_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
            "pose_orthonormalization_max_correction": ORTHO["max_correction"], "ttt": ttt_logs,
            "predictions": predictions}
    (out / f"ds_seal_{index}of{count}.json").write_text(json.dumps(info, indent=1))
    print("DS_EVAL_DONE", args.model, args.cohort, len(predictions), "queries", info["seconds"], "s",
          info["max_memory_gb"], "GB", flush=True)


if __name__ == "__main__":
    main()
