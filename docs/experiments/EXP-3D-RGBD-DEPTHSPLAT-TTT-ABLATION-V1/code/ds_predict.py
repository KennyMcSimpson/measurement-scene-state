"""DepthSplat predictions for project records (exploratory; reads no query depth).

For every record and role of a manifest, the three context frames (RGB only, the prepared
128x160 frames upsampled to --size) and their cameras go into the pretrained DepthSplat
(re10k+dl3dv base); its Gaussians are rendered at every primary query camera at 128x160
(planar z, the model's depth mode) and saved, with the model's own context-view depths.
The depth render is DepthSplat's sum of alpha * transmittance * z over Gaussians on a zero
background, so it is not normalized; the accumulated opacity (alpha) of the same blend is saved
too, and depth_z / alpha is the normalized depth wherever alpha is not small.
Intrinsics are normalized as DepthSplat expects (pixel-corner principal point / image size);
poses are the records' OpenCV camera-to-world matrices in metres.
Run with the DepthSplat environment from the DepthSplat repository root:
  python ds_predict.py MANIFEST SPLIT OUT [--size 256 320] [--near 0.5] [--far 200]
"""

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from hydra import compose, initialize_config_dir
from PIL import Image

from src.config import load_typed_root_config
from src.global_cfg import set_cfg
from src.model.decoder import get_decoder
from src.model.decoder.cuda_splatting import render_cuda
from src.model.encoder import get_encoder

CHECKPOINT = Path(
    "/home/xmum26_zonghan/zonghan_o_mirror/_models/depthsplat/"
    "depthsplat-gs-base-re10kdl3dv-448x768-randview2-6-f8ddd845.pth"
)
OVERRIDES = [
    "+experiment=dl3dv",
    "mode=test",
    "model.encoder.num_scales=2",
    "model.encoder.upsample_factor=4",
    "model.encoder.lowest_feature_resolution=8",
    "model.encoder.monodepth_vit_type=vitb",
    "model.encoder.gaussian_adapter.gaussian_scale_max=0.1",
    # Inference only: the final depth alone (the intermediate one is a training target).
    "model.encoder.supervise_intermediate_depth=false",
    "wandb.mode=disabled",
]
MIRROR = ("/home/zonghan/", "/home/xmum26_zonghan/zonghan_o_mirror/")
DINOV2_CODE = Path.home() / ".cache/torch/hub/facebookresearch_dinov2_main"
_hub_load = torch.hub.load


def hub_load_without_weights(repo, model, *args, **kwargs):
    """DepthSplat builds its monocular ViT with torch.hub.load("facebookresearch/dinov2", ...), which looks up the
    default branch on GitHub and downloads the ImageNet DINOv2 weights. The DepthSplat checkpoint holds all of those
    weights (build() fails on any missing encoder key), so the module is built without them, from the local code."""
    kwargs.setdefault("pretrained", False)
    if repo == "facebookresearch/dinov2":
        return _hub_load(str(DINOV2_CODE), model, *args, source="local", **kwargs)
    return _hub_load(repo, model, *args, **kwargs)


torch.hub.load = hub_load_without_weights


def local(path):
    return path.replace(*MIRROR) if path.startswith(MIRROR[0]) else path


def build(device, checkpoint=CHECKPOINT):
    with initialize_config_dir(config_dir=str(Path("config").resolve()), version_base=None):
        cfg_dict = compose(config_name="main", overrides=OVERRIDES)
    set_cfg(cfg_dict)
    cfg = load_typed_root_config(cfg_dict)
    encoder, _ = get_encoder(cfg.model.encoder)
    decoder = get_decoder(cfg.model.decoder, cfg.dataset)
    state = torch.load(checkpoint, map_location="cpu")
    state = state.get("state_dict", state)
    for prefix, module in (("encoder.", encoder), ("decoder.", decoder)):
        part = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
        missing, unexpected = module.load_state_dict(part, strict=False)
        if unexpected or (missing and prefix == "encoder."):
            raise RuntimeError(f"{prefix} missing {missing[:5]} unexpected {unexpected[:5]}")
    return encoder.to(device).eval(), decoder.to(device).eval()


ORTHO = {"max_correction": 0.0}


def orthonormal(c2w):
    """The nearest proper rotation (SVD) replaces the pose's rotation: DepthSplat's SH rotation asserts det(R) == 1
    to 1e-5, which a few recorded poses miss (e.g. EVAL-V3 ai_045_001 frame 0: |det - 1| = 2e-5, R^T R off by 0.006).
    The largest entry change is kept in ORTHO and reported in run_info.json."""
    u, _, vt = np.linalg.svd(c2w[:3, :3])
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        raise ValueError("c2w rotation is a reflection")
    ORTHO["max_correction"] = max(ORTHO["max_correction"], float(np.abs(rotation - c2w[:3, :3]).max()))
    out = c2w.copy()
    out[:3, :3] = rotation
    return out


def views(frames, ids, size, near, far, device, image_size=(128, 160)):
    rows = {f["frame_id"]: f for f in frames}
    images, intrinsics, extrinsics = [], [], []
    h0, w0 = image_size
    for i in ids:
        f = rows[i]
        rgb = np.asarray(Image.open(local(f["rgb"])).convert("RGB"), dtype=np.float32) / 255.0
        image = torch.from_numpy(rgb).permute(2, 0, 1)[None]
        if size is not None:
            image = F.interpolate(image, size=size, mode="bilinear", align_corners=False)
        images.append(image[0])
        k = np.array(f["intrinsics"], dtype=np.float64)
        intrinsics.append(
            [[k[0, 0] / w0, 0.0, (k[0, 2] + 0.5) / w0], [0.0, k[1, 1] / h0, (k[1, 2] + 0.5) / h0], [0.0, 0.0, 1.0]]
        )
        extrinsics.append(orthonormal(np.array(f["c2w"], dtype=np.float64)))
    n = len(ids)
    return {
        "image": torch.stack(images)[None].to(device),
        "intrinsics": torch.tensor(np.array(intrinsics), dtype=torch.float32)[None].to(device),
        "extrinsics": torch.tensor(np.array(extrinsics), dtype=torch.float32)[None].to(device),
        "near": torch.full((1, n), near, device=device),
        "far": torch.full((1, n), far, device=device),
    }


def spare_frames(record, n, min_query_gap=0):
    """Up to n frames that no role of the record uses (neither context nor query), evenly spaced in frame order;
    with min_query_gap, frames whose index is closer than that to any query frame are left out."""
    roles = record["roles"]
    used = set().union(*(set(roles.get(k, [])) for k in
                         ("context_a", "context_b", "query", "primary_query", "secondary_query", "stream")))
    queries = set().union(*(set(roles.get(k, [])) for k in ("query", "primary_query", "secondary_query")))
    spare = [f["frame_id"] for f in record["frames"] if f["frame_id"] not in used
             and all(abs(f["frame_id"] - q) >= min_query_gap for q in queries)]
    if n >= len(spare):
        return spare
    return [spare[round(i * (len(spare) - 1) / max(n - 1, 1))] for i in range(n)]


def measured_planar(frames, ids, size, device, keep=None):
    """Measured depth as planar z [1, V, H, W] (the records store ray distance), nearest-resized; 0 = invalid.
    Views outside `keep` (when given) get 0, so DepthSplat keeps its own depth there."""
    rows = {f["frame_id"]: f for f in frames}
    planes = []
    for i in ids:
        if keep is not None and i not in keep:
            planes.append(torch.zeros(np.load(local(rows[i]["depth"])).squeeze().shape))
            continue
        depth = np.load(local(rows[i]["depth"])).squeeze().astype(np.float64)
        k = np.array(rows[i]["intrinsics"], dtype=np.float64)
        y, x = np.indices(depth.shape)
        ray = np.sqrt(((x - k[0, 2]) / k[0, 0]) ** 2 + ((y - k[1, 2]) / k[1, 1]) ** 2 + 1.0)
        planes.append(torch.from_numpy(np.where(np.isfinite(depth) & (depth > 0), depth / ray, 0.0)).float())
    return F.interpolate(torch.stack(planes)[None], size=size, mode="nearest").to(device)


INJECTED = {}


def inject_measured_depth(encoder):
    """Make DepthSplat use INJECTED["depth"] (where valid) as its final context depth: the depth that feeds the
    Gaussian regressor and places the Gaussians. Its matching probabilities and features stay its own.
    Records how DepthSplat's own depth compares with the measured depth of the first INJECTED["measured_views"]
    views (median ratio and AbsRel); with INJECTED["align"], DepthSplat's depth of the remaining views is first
    multiplied by that median ratio (a test-time scale alignment to the measured views)."""
    original = encoder.depth_predictor.forward

    def forward(*args, **kwargs):
        result = original(*args, **kwargs)
        predicted = result["depth_preds"][-1]
        measured = INJECTED["depth"].to(predicted)
        n = INJECTED.get("measured_views", measured.shape[1])
        m, p = measured[:, :n], predicted[:, :n]
        valid = m > 0
        ratio = float(torch.median(m[valid] / p[valid].clamp_min(1e-6)))
        INJECTED["stats"] = {"median_ratio": ratio,
                             "context_absrel": float(((p - m).abs() / m.clamp_min(1e-6))[valid].mean())}
        if INJECTED.get("align"):
            predicted = predicted * ratio
        result["depth_preds"][-1] = torch.where(measured > 0, measured, predicted)
        return result

    encoder.depth_predictor.forward = forward


def render_alpha(gaussians, target, shape):
    """Accumulated opacity per target pixel: the total weight of render_depth_cuda's blend (zero background)."""
    b, v = target["extrinsics"].shape[:2]
    means = gaussians.means
    alpha = render_cuda(
        rearrange(target["extrinsics"], "b v i j -> (b v) i j"),
        rearrange(target["intrinsics"], "b v i j -> (b v) i j"),
        rearrange(target["near"], "b v -> (b v)"),
        rearrange(target["far"], "b v -> (b v)"),
        shape,
        torch.zeros((b * v, 3), device=means.device),
        repeat(means, "b g xyz -> (b v) g xyz", v=v),
        repeat(gaussians.covariances, "b g i j -> (b v) g i j", v=v),
        torch.ones((b * v, means.shape[1], 3, 1), device=means.device),
        repeat(gaussians.opacities, "b g -> (b v) g", v=v),
        use_sh=False,
    )
    return rearrange(alpha.mean(dim=1), "(b v) h w -> b v h w", b=b, v=v)


def git_head(path):
    try:
        return subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    except OSError:
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest")
    parser.add_argument("split")
    parser.add_argument("out")
    parser.add_argument("--size", type=int, nargs=2, default=[256, 320])
    parser.add_argument("--near", type=float, default=0.5)
    parser.add_argument("--far", type=float, default=200.0)
    parser.add_argument("--inject-measured-depth", action="store_true",
                        help="replace DepthSplat's final context depth by the measured context depth where valid")
    parser.add_argument("--extra-rgb", type=int, default=0,
                        help="add up to N frames no role uses as RGB-only views after the three context views")
    parser.add_argument("--min-query-gap", type=int, default=0,
                        help="leave out extra frames whose frame index is closer than this to any query frame")
    parser.add_argument("--inject-all", action="store_true",
                        help="with --inject-measured-depth, inject measured depth for the extra views too (RGB-D bound)")
    parser.add_argument("--checkpoint", default=str(CHECKPOINT), help="DepthSplat weights (default: the released base)")
    parser.add_argument("--align-extra-scale", action="store_true",
                        help="with --inject-measured-depth: scale DepthSplat's depth of the RGB-only views by the median "
                             "measured/predicted ratio on the measured views")
    args = parser.parse_args()
    if args.align_extra_scale and not args.inject_measured_depth:
        parser.error("--align-extra-scale needs --inject-measured-depth")
    INJECTED["align"] = args.align_extra_scale
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    device = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    started = time.time()
    encoder, decoder = build(device, Path(args.checkpoint))
    if args.inject_measured_depth:
        inject_measured_depth(encoder)
    manifest = json.loads(Path(args.manifest).read_text())
    records = [r for r in manifest["scenes"] if r["split"] == args.split]
    done, log = 0, []
    for record in records:
        for role, key in (("A", "context_a"), ("B", "context_b")):
            ctx_ids, query_ids = record["roles"][key], record["roles"]["primary_query"]
            extra_ids = spare_frames(record, args.extra_rgb, args.min_query_gap) if args.extra_rgb else []
            view_ids = list(ctx_ids) + extra_ids  # context views first: ds_score reads their depths by position
            context = views(record["frames"], view_ids, tuple(args.size), args.near, args.far, device)
            target = views(record["frames"], query_ids, None, args.near, args.far, device)
            if args.inject_measured_depth:
                keep = set(view_ids) if args.inject_all else set(ctx_ids)
                INJECTED["depth"] = measured_planar(record["frames"], view_ids, tuple(args.size), device, keep)
                INJECTED["measured_views"] = len(keep)
            with torch.no_grad():
                encoded = encoder(context, global_step=0, deterministic=True)
                gaussians = encoded["gaussians"] if isinstance(encoded, dict) else encoded
                rendered = decoder.forward(
                    gaussians, target["extrinsics"], target["intrinsics"], target["near"],
                    target["far"], (128, 160), depth_mode="depth",
                )
                alpha = render_alpha(gaussians, target, (128, 160))
            ctx_depth = encoded["depths"] if isinstance(encoded, dict) and "depths" in encoded else None
            while ctx_depth is not None and ctx_depth.dim() > 4:  # [B, V, H, W, srf, s] -> [B, V, H, W]
                ctx_depth = ctx_depth.squeeze(-1)
            entry = {"scene_id": record["scene_id"], "role": role, "views": view_ids,
                     "gaussians": int(gaussians.means.shape[1]),
                     "alpha_mean": float(alpha.mean()), "context_depth_shape": None if ctx_depth is None
                     else list(ctx_depth.shape), "injection": INJECTED.get("stats") if args.inject_measured_depth else None}
            log.append(entry)
            if len(log) == 1:
                print("FIRST", entry, flush=True)
            for j, qid in enumerate(query_ids):
                arrays = {
                    "depth_z": rendered.depth[0, j].float().cpu().numpy(),
                    "alpha": alpha[0, j].float().cpu().numpy(),
                    "color": rendered.color[0, j].float().cpu().numpy(),
                }
                if ctx_depth is not None:
                    arrays["context_depth_z"] = ctx_depth[0].float().cpu().numpy()
                np.savez_compressed(out / f"{record['scene_id']}_{role}_{qid}.npz", **arrays)
                done += 1
    info = {
        "manifest": str(Path(args.manifest).resolve()), "split": args.split, "size": args.size,
        "near": args.near, "far": args.far, "inject_measured_depth": args.inject_measured_depth,
        "extra_rgb": args.extra_rgb, "min_query_gap": args.min_query_gap, "inject_all": args.inject_all,
        "align_extra_scale": args.align_extra_scale,
        "checkpoint": str(args.checkpoint), "dinov2_code": str(DINOV2_CODE), "overrides": OVERRIDES,
        "depthsplat_commit": git_head(Path.cwd()), "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0), "queries": done, "seconds": round(time.time() - started, 1),
        "pose_orthonormalization_max_correction": ORTHO["max_correction"],
        "records": log,
    }
    (out / "run_info.json").write_text(json.dumps(info, indent=2))
    print("DS_PREDICTED", done, "queries", "size", args.size, "seconds", info["seconds"], flush=True)


if __name__ == "__main__":
    main()
