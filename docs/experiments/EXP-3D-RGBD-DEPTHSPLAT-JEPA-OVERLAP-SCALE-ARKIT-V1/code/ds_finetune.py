"""Fine-tune DepthSplat's depth predictor on the project's TRAIN records with measured depth (exploratory pilot).

Each step takes one TRAIN record (V7 TRAIN72 or an SEG-V1 segment of those scenes), draws 3 to 6 of its 16 frames,
runs DepthSplat's encoder in its depth-only mode on their RGB (prepared 128x160 frames upsampled to 256x320, as at
inference, with the records' cameras) and regresses every view's depth: mean |log z_pred - log z_measured| at
128x160 over valid measured pixels, the prediction resized bilinearly. Only the depth predictor is trained, its
monocular ViT at a tenth of the learning rate; the Gaussian head stays as released. The model stays in eval mode
(no dropout or normalization-statistics updates). No DEV or EVAL record is used for training; DEV context-frame
depth AbsRel is logged every --eval-every steps for information only, and the final step is saved (no selection).
Run with the DepthSplat environment from the DepthSplat repository root (PYTHONPATH=.):
  python ds_finetune.py OUT [--steps 4000] [--lr 2e-5] [--seed 20261006]
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
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ds_predict import build, local, views  # noqa: E402

REPO = Path("/home/xmum26_zonghan/zonghan_o_mirror/measurement-scene-state")
TRAIN_SOURCES = (REPO / "outputs/EXP-3D-RGBD-DEPTH-BOUNDS-V7/scene_split.json",
                 REPO / "outputs/EXP-3D-RGBD-SEGMENT-DATA-V1/manifest_seg.json")
DEV_MANIFEST = Path.home() / "cv_work/depthsplat/probe_dev/manifest_dev.json"
SIZE = (256, 320)


def load_records():
    split = json.loads(TRAIN_SOURCES[0].read_text())
    train = [r for r in split["scenes"] if r["split"] == "TRAIN"]
    dev_ids = {r["scene_id"] for r in split["scenes"] if r["split"] == "DEV"}
    protected = set(split["protected_scene_ids"])
    base_ids = {r["scene_id"] for r in train}
    segments = json.loads(TRAIN_SOURCES[1].read_text())["scenes"]
    for r in segments:
        if r["split"] != "TRAIN" or r["scene_id"].rsplit("_seg", 1)[0] not in base_ids:
            raise ValueError(f"segment {r['scene_id']} is not a TRAIN72 scene segment")
    records = train + segments
    for r in records:
        scene = r["scene_id"].rsplit("_seg", 1)[0]
        if scene in dev_ids or scene in protected:
            raise ValueError(f"{r['scene_id']} overlaps DEV or the protected holdout")
    dev = json.loads(DEV_MANIFEST.read_text())["scenes"]
    if {r["scene_id"] for r in dev} != dev_ids:
        raise ValueError("DEV manifest differs from the V7 DEV split")
    return records, dev


def measured_z(frames, ids):
    """Measured planar depth [V, 128, 160] (records store ray distance); 0 where invalid."""
    rows = {f["frame_id"]: f for f in frames}
    out = []
    for i in ids:
        depth = np.load(local(rows[i]["depth"])).squeeze().astype(np.float64)
        k = np.array(rows[i]["intrinsics"], dtype=np.float64)
        y, x = np.indices(depth.shape)
        ray = np.sqrt(((x - k[0, 2]) / k[0, 0]) ** 2 + ((y - k[1, 2]) / k[1, 1]) ** 2 + 1.0)
        out.append(np.where(np.isfinite(depth) & (depth > 0), depth / ray, 0.0))
    return torch.from_numpy(np.stack(out)).float()


def predict_depth(encoder, context, step):
    encoded = encoder(context, global_step=step, deterministic=True)
    depth = encoded["depths"]
    while depth.dim() > 4:  # [B, V, H, W, srf, s] -> [B, V, H, W]
        depth = depth.squeeze(-1)
    return F.interpolate(depth[0][:, None], size=(128, 160), mode="bilinear", align_corners=False)[:, 0]


def dev_absrel(encoder, dev, device):
    values = []
    with torch.no_grad():
        for record in dev:
            for key in ("context_a", "context_b"):
                ids = record["roles"][key]
                pred = predict_depth(encoder, views(record["frames"], ids, SIZE, 0.5, 200.0, device), 0).cpu()
                gt = measured_z(record["frames"], ids)
                valid = gt > 0
                values.append(float(((pred - gt).abs() / gt.clamp_min(1e-6))[valid].mean()))
    return float(np.mean(values))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("out")
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20261006)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    device = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    records, dev = load_records()
    encoder, _ = build(device)
    encoder.eval()
    encoder.cfg.train_depth_only = True  # DepthSplat's own depth-training path: returns depths, no Gaussians
    for p in encoder.parameters():
        p.requires_grad_(False)
    vit, rest = [], []
    for name, p in encoder.depth_predictor.named_parameters():
        p.requires_grad_(True)
        (vit if name.startswith("pretrained.") else rest).append(p)
    optimizer = torch.optim.AdamW([{"params": rest, "lr": args.lr}, {"params": vit, "lr": args.lr / 10}],
                                  weight_decay=1e-4)
    base_lrs = [g["lr"] for g in optimizer.param_groups]
    log = {"args": vars(args), "train_records": len(records), "trainable": {"rest": sum(p.numel() for p in rest),
           "vit": sum(p.numel() for p in vit)}, "dev": [], "loss": []}
    print("TRAIN_RECORDS", len(records), "TRAINABLE", log["trainable"], flush=True)
    log["dev"].append({"step": 0, "absrel": dev_absrel(encoder, dev, device)})
    print("DEV step 0 absrel", round(log["dev"][-1]["absrel"], 4), flush=True)
    started, window = time.time(), []
    for step in range(1, args.steps + 1):
        scale = min(1.0, step / args.warmup) * 0.5 * (1 + math.cos(math.pi * step / args.steps))
        for group, lr in zip(optimizer.param_groups, base_lrs):
            group["lr"] = lr * scale
        record = rng.choice(records)
        ids = rng.sample([f["frame_id"] for f in record["frames"]], rng.randint(3, 6))
        context = views(record["frames"], ids, SIZE, 0.5, 200.0, device)
        gt = measured_z(record["frames"], ids).to(device)
        valid = gt > 0
        pred = predict_depth(encoder, context, step)
        loss = (pred.clamp_min(1e-3).log() - gt.clamp_min(1e-3).log()).abs()[valid].mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(rest + vit, 1.0)
        optimizer.step()
        window.append(float(loss))
        if step % 50 == 0:
            log["loss"].append({"step": step, "loss": float(np.mean(window))})
            print(f"step {step} loss {np.mean(window):.4f} {time.time() - started:.0f}s", flush=True)
            window = []
        if step % args.eval_every == 0 or step == args.steps:
            log["dev"].append({"step": step, "absrel": dev_absrel(encoder, dev, device)})
            print("DEV step", step, "absrel", round(log["dev"][-1]["absrel"], 4), flush=True)
    state = {f"encoder.{k}": v.cpu() for k, v in encoder.state_dict().items()}
    torch.save({"state_dict": state, "source": "ds_finetune.py", "args": vars(args)}, out / "depthsplat_ft_final.pth")
    log["seconds"] = round(time.time() - started, 1)
    (out / "finetune_log.json").write_text(json.dumps(log, indent=1))
    print("FINETUNE_DONE", out, log["seconds"], flush=True)


if __name__ == "__main__":
    main()
