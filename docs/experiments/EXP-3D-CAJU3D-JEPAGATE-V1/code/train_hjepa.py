"""Train the geometry-aligned reverse JEPA (hjepa.GeometricReverseJEPA) on the ARKitScenes TRAIN places (GPU).

Data: outputs/EXP-3D-RGBD-ARKIT-DATA-V1/manifest_arkit_trainval.json, split TRAIN (182 places, 16 frames each), all
frames preloaded (RGB, measured ray depth, intrinsics, camera-to-world). Each sample = one scan, one target frame and
--sources other frames of the same scan (uniform without replacement); the model predicts the target's latent from the
sources warped into the target view. The 8 DEV places are the held-out monitor (every frame as target, fixed sources).
No evaluation cohort of any CAJU experiment (ARKIT_SYSTEM, SYSTEM2, SYSTEM3, VAL) is read.
Run with the DepthSplat environment (torch + CUDA) from the repository root:
  python train_hjepa.py OUT [--steps 30000] [--batch 8] [--sources 6] [--seed 20261016]
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hjepa as J  # noqa: E402

MANIFEST = Path("outputs/EXP-3D-RGBD-ARKIT-DATA-V1/manifest_arkit_trainval.json")
MIRROR = ("/home/zonghan/", "/home/xmum26_zonghan/zonghan_o_mirror/")


def local(path):
    return path.replace(*MIRROR) if path.startswith(MIRROR[0]) else path


def load_split(records):
    rgb, ray, k, c2w = [], [], [], []
    for r in records:
        frames = sorted(r["frames"], key=lambda f: f["frame_id"])
        rgb.append(np.stack([np.asarray(Image.open(local(f["rgb"])).convert("RGB")).transpose(2, 0, 1) for f in frames]))
        d = np.stack([np.load(local(f["depth"])).squeeze().astype(np.float32) for f in frames])
        ray.append(np.where(np.isfinite(d) & (d > 0), d, 0.0).astype(np.float32))
        k.append(np.stack([np.asarray(f["intrinsics"], dtype=np.float32) for f in frames]))
        c2w.append(np.stack([np.asarray(f["c2w"], dtype=np.float32) for f in frames]))
    return {"rgb": torch.from_numpy(np.stack(rgb)), "ray": torch.from_numpy(np.stack(ray)),
            "k": torch.from_numpy(np.stack(k)), "c2w": torch.from_numpy(np.stack(c2w))}


def orthonormal(c2w):
    """Nearest proper rotation (as ds_predict.orthonormal), batched."""
    u, _, vt = torch.linalg.svd(c2w[..., :3, :3])
    out = c2w.clone()
    out[..., :3, :3] = u @ vt
    return out


def batch_of(data, scans, targets, sources, device):
    src = {k: torch.stack([data[k][s, idx] for s, idx in zip(scans, sources)]) for k in data}
    tgt = {k: torch.stack([data[k][s, t] for s, t in zip(scans, targets)]) for k in data}
    to = lambda x: x.to(device, non_blocking=True)  # noqa: E731
    return (to(src["rgb"]).float() / 255, to(src["ray"]), to(src["k"]), orthonormal(to(src["c2w"])),
            to(tgt["rgb"]).float() / 255, to(tgt["ray"]), to(tgt["k"]), orthonormal(to(tgt["c2w"])))


@torch.no_grad()
def evaluate(model, data, n_sources, device):
    """Held-out DEV: every frame as target, the n_sources frames nearest in index as sources."""
    model.eval()
    nll, mse, corr = [], [], []
    n_scans, n_frames = data["rgb"].shape[:2]
    for s in range(n_scans):
        for t in range(n_frames):
            others = sorted((f for f in range(n_frames) if f != t), key=lambda f: (abs(f - t), f))[:n_sources]
            args = batch_of(data, [s], [t], [others], device)
            out = model(*args)
            err = ((out["z"] - out["z_hat"]) ** 2).mean(dim=1)
            nll.append(float((0.5 * err * torch.exp(-out["log_p"][:, 0]) + 0.5 * out["log_p"][:, 0]).mean()))
            mse.append(float(err.mean()))
            e, p = err.flatten(), out["log_p"].flatten()
            corr.append(float(torch.corrcoef(torch.stack([torch.log(e + 1e-8), p]))[0, 1]))
    model.train()
    return {"nll": float(np.mean(nll)), "mse": float(np.mean(mse)), "logerr_logp_corr": float(np.nanmean(corr))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("out", type=Path)
    parser.add_argument("--steps", type=int, default=30000)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--sources", type=int, default=6)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=20261016)
    parser.add_argument("--eval-every", type=int, default=2500)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit-scans", type=int, default=0, help="smoke runs only")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = args.device
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    records = json.loads(MANIFEST.read_text())["scenes"]
    limit = args.limit_scans or None
    train = load_split([r for r in records if r["split"] == "TRAIN"][:limit])
    dev = load_split([r for r in records if r["split"] == "DEV"][:limit])
    model = J.GeometricReverseJEPA(ema=0.996).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    n_scans, n_frames = train["rgb"].shape[:2]
    log = {"args": {k: str(v) for k, v in vars(args).items()}, "train_scans": int(n_scans),
           "dev_scans": int(dev["rgb"].shape[0]), "eval": [], "train": []}
    log["eval"].append({"step": 0, **evaluate(model, dev, args.sources, device)})
    print("EVAL0", log["eval"][-1], flush=True)
    started, window = time.time(), []
    for step in range(1, args.steps + 1):
        scale = min(1.0, step / 500) * 0.5 * (1 + math.cos(math.pi * step / args.steps))
        for group in optimizer.param_groups:
            group["lr"] = args.lr * scale
        scans = rng.integers(0, n_scans, args.batch).tolist()
        targets = rng.integers(0, n_frames, args.batch).tolist()
        sources = [rng.choice([f for f in range(n_frames) if f != t], args.sources, replace=False).tolist()
                   for t in targets]
        out = model(*batch_of(train, scans, targets, sources, device))
        loss, parts = J.jepa_loss(out)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()
        model.update_target()
        window.append(parts)
        if step % 250 == 0:
            mean = {k: float(np.mean([w[k] for w in window])) for k in window[0]}
            log["train"].append({"step": step, **mean, "seconds": round(time.time() - started, 1)})
            print(f"step {step} {mean} {time.time() - started:.0f}s", flush=True)
            window = []
        if step % args.eval_every == 0 or step == args.steps:
            log["eval"].append({"step": step, **evaluate(model, dev, args.sources, device)})
            print("EVAL", log["eval"][-1], flush=True)
    torch.save({"state_dict": model.state_dict(), "args": vars(args), "latent": J.LATENT},
               args.out / "hjepa_final.pth")
    log["seconds"] = round(time.time() - started, 1)
    log["max_memory_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2) if device == "cuda" else None
    (args.out / "train_log.json").write_text(json.dumps(log, indent=1))
    print("HJEPA_TRAIN_DONE", log["seconds"], "s", log["max_memory_gb"], "GB", flush=True)


if __name__ == "__main__":
    main()
