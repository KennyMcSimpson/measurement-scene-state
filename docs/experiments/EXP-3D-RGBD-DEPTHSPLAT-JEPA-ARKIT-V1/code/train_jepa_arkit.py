"""Reverse-JEPA training of DepthSplat on real sensor data (EXP-3D-RGBD-DEPTHSPLAT-JEPA-ARKIT-V1): the recipe of
EXP-3D-RGBD-DEPTHSPLAT-JEPA-TTT-V1/code/train_jepa.py with only the records changed: ARKitScenes TRAIN and DEV
(arkit_records.load_records) instead of Hypersim TRAIN72 + SEG-V1 and the Hypersim DEV.

Each step takes one TRAIN record and one context role. The role's
three RGB-D frames go into DepthSplat with their measured depth injected (planar z where valid), and the
resulting Gaussians (the scene state) are rendered at --targets frames drawn from the record's other frames.
Loss = relative L1 of the raw rendered depth (sum of alpha * T * z on a zero background, so an uncovered pixel
costs its full depth) against the target's measured planar depth on valid pixels, + 0.5 x L1 of the rendered
RGB. The whole encoder is trained from the released weights (its monocular ViT at a tenth of the learning
rate) in eval mode; the decoder has no parameters. Fixed step count, final step saved, no selection; DEV
primary-query depth is logged for information only.
Run with the DepthSplat environment from the DepthSplat repository root (PYTHONPATH=.):
  python train_jepa_arkit.py OUT --seed S [--steps 3000] [--lr 5e-6] [--targets 2] [--scale-max 3.0]
  python train_jepa_arkit.py OUT --eval-only [--scale-max 0.1] [--checkpoint PTH]   (DEV readout only)
"""

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

for _directory in (Path(__file__).resolve().parents[1], Path(__file__).resolve().parent):  # the helpers' directory
    sys.path.insert(0, str(_directory))
import ds_predict  # noqa: E402
from arkit_records import load_records  # noqa: E402
from ds_finetune import measured_z  # noqa: E402
from ds_predict import INJECTED, build, inject_measured_depth, local, measured_planar, render_alpha, views  # noqa: E402

SIZE = (256, 320)
RENDER = (128, 160)
RGB_WEIGHT = 0.5
ALPHA_MIN = 0.5
SCALE_KEY = "model.encoder.gaussian_adapter.gaussian_scale_max="


def set_scale_max(value):
    """DepthSplat's cap on Gaussian scales (world units); ds_predict's overrides use the README's 0.1."""
    index = [i for i, o in enumerate(ds_predict.OVERRIDES) if o.startswith(SCALE_KEY)]
    if len(index) != 1:
        raise RuntimeError("scale override not found exactly once")
    ds_predict.OVERRIDES[index[0]] = f"{SCALE_KEY}{value}"


def ray_factor(intrinsics, shape=RENDER):
    k = np.array(intrinsics, dtype=np.float64)
    y, x = np.indices(shape)
    return np.sqrt(((x - k[0, 2]) / k[0, 0]) ** 2 + ((y - k[1, 2]) / k[1, 1]) ** 2 + 1.0)


def encode(encoder, record, context_ids, step, device):
    """The scene state: Gaussians from the context RGB-D (measured depth injected)."""
    context = views(record["frames"], context_ids, SIZE, 0.5, 200.0, device)
    INJECTED["depth"] = measured_planar(record["frames"], context_ids, SIZE, device)
    INJECTED["measured_views"] = len(context_ids)
    encoded = encoder(context, global_step=step, deterministic=True)
    return encoded["gaussians"] if isinstance(encoded, dict) else encoded


def render(decoder, gaussians, target):
    out = decoder.forward(
        gaussians, target["extrinsics"], target["intrinsics"], target["near"], target["far"], RENDER,
        depth_mode="depth",
    )
    return out.depth[0], out.color[0]


def grounded_loss(depth, color, z, rgb):
    """Relative L1 of raw rendered depth on valid measured pixels + RGB_WEIGHT x RGB L1."""
    valid = z > 0
    depth_term = ((depth - z).abs() / z.clamp_min(1e-6))[valid].mean()
    return depth_term + RGB_WEIGHT * (color - rgb).abs().mean(), float(depth_term)


@torch.no_grad()
def dev_eval(encoder, decoder, dev, device):
    """DEV primary queries: AbsRel of the alpha-normalized depth on covered valid pixels, coverage, and the
    AbsRel of the raw (unnormalized) depth on all valid pixels."""
    absrel, coverage, raw = [], [], []
    for record in dev:
        frames = {f["frame_id"]: f for f in record["frames"]}
        qids = record["roles"]["primary_query"]
        for key in ("context_a", "context_b"):
            gaussians = encode(encoder, record, record["roles"][key], 0, device)
            target = views(record["frames"], qids, None, 0.5, 200.0, device)
            depth, _ = render(decoder, gaussians, target)
            alpha = render_alpha(gaussians, target, RENDER)[0]
            for j, qid in enumerate(qids):
                gt = np.load(local(frames[qid]["depth"])).squeeze().astype(np.float64)
                valid = np.isfinite(gt) & (gt > 0)
                a = alpha[j].cpu().numpy().astype(np.float64)
                z = depth[j].cpu().numpy().astype(np.float64)
                factor = ray_factor(frames[qid]["intrinsics"])
                covered = (a >= ALPHA_MIN) & valid
                normalized = z / np.maximum(a, 1e-6) * factor
                if covered.any():
                    absrel.append(float(np.mean(np.abs(normalized[covered] - gt[covered]) / gt[covered])))
                coverage.append(float(covered.sum() / max(valid.sum(), 1)))
                raw.append(float(np.mean(np.abs(z * factor - gt)[valid] / gt[valid])))
    return {"absrel_covered": float(np.mean(absrel)), "coverage": float(np.mean(coverage)),
            "absrel_raw": float(np.mean(raw))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("out")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--targets", type=int, default=2)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--scale-max", type=float, default=3.0)
    parser.add_argument("--checkpoint", default=str(ds_predict.CHECKPOINT))
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--no-save", action="store_true", help="smoke runs: skip the final checkpoint")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    device = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    set_scale_max(args.scale_max)
    records, dev = load_records()
    encoder, decoder = build(device, Path(args.checkpoint))
    inject_measured_depth(encoder)
    INJECTED["align"] = False
    encoder.eval()
    log = {"args": vars(args), "overrides": list(ds_predict.OVERRIDES), "train_records": len(records),
           "dev": [], "loss": []}
    started = time.time()
    log["dev"].append({"step": 0, **dev_eval(encoder, decoder, dev, device)})
    print("DEV0", log["dev"][-1], round(time.time() - started, 1), "s", flush=True)
    if args.eval_only:
        (out / "dev_eval.json").write_text(json.dumps(log, indent=1))
        return
    vit, rest = [], []
    for name, p in encoder.named_parameters():
        p.requires_grad_(True)
        (vit if name.startswith("depth_predictor.pretrained.") else rest).append(p)
    log["trainable"] = {"rest": sum(p.numel() for p in rest), "vit": sum(p.numel() for p in vit)}
    optimizer = torch.optim.AdamW(
        [{"params": rest, "lr": args.lr}, {"params": vit, "lr": args.lr / 10}], weight_decay=1e-4
    )
    base_lrs = [group["lr"] for group in optimizer.param_groups]
    print("TRAIN_RECORDS", len(records), "TRAINABLE", log["trainable"], flush=True)
    started, window, depth_window = time.time(), [], []
    for step in range(1, args.steps + 1):
        scale = min(1.0, step / args.warmup) * 0.5 * (1 + math.cos(math.pi * step / args.steps))
        for group, lr in zip(optimizer.param_groups, base_lrs):
            group["lr"] = lr * scale
        record = rng.choice(records)
        key = rng.choice(("context_a", "context_b"))
        context_ids = record["roles"][key]
        others = [f["frame_id"] for f in record["frames"] if f["frame_id"] not in context_ids]
        target_ids = rng.sample(others, args.targets)
        gaussians = encode(encoder, record, context_ids, step, device)
        target = views(record["frames"], target_ids, None, 0.5, 200.0, device)
        depth, color = render(decoder, gaussians, target)
        z = measured_z(record["frames"], target_ids).to(device)
        loss, depth_term = grounded_loss(depth, color, z, target["image"][0])
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at step {step}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(rest + vit, 1.0)
        optimizer.step()
        window.append(float(loss))
        depth_window.append(depth_term)
        if step % 50 == 0 or step == args.steps:
            log["loss"].append({"step": step, "loss": float(np.mean(window)), "depth": float(np.mean(depth_window))})
            print(f"step {step} loss {np.mean(window):.4f} depth {np.mean(depth_window):.4f} "
                  f"{time.time() - started:.0f}s", flush=True)
            window, depth_window = [], []
        if step % args.eval_every == 0 or step == args.steps:
            log["dev"].append({"step": step, **dev_eval(encoder, decoder, dev, device)})
            print("DEV", log["dev"][-1], flush=True)
    log["seconds"] = round(time.time() - started, 1)
    log["max_memory_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
    if not args.no_save:
        state = {f"encoder.{k}": v.detach().cpu() for k, v in encoder.state_dict().items()}
        torch.save({"state_dict": state, "source": "train_jepa_arkit.py", "args": vars(args)},
                   out / "depthsplat_jepa_final.pth")
    (out / "train_log.json").write_text(json.dumps(log, indent=1))
    print("JEPA_TRAIN_DONE", out, log["seconds"], "s", log["max_memory_gb"], "GB", flush=True)


if __name__ == "__main__":
    main()
