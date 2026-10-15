"""EXP-3D-CAJU3D-ENGINE-V1: the CAJU-3D TTT engine variants without any learned module (GPU replay).

Per episode, for one state builder (--model ZS = released DepthSplat, or JN1 = the accuracy-only reverse-JEPA
checkpoint), cap 0.1, measured depth injected:
  CACHE          the 7-view state (V1's encode), unchanged;
  RAW            V1's ds_eval.ttt unchanged (raw-depth relative L1 + 0.5 RGB L1 on valid measured pixels);
  NORM           alpha-normalized terms on covered valid pixels (engine_core.norm_terms), all parameters, V1's
                 optimizer (one Adam over the five groups, V1's learning rates);
  NORM_DEC       the same terms decoupled: means / log-scales / opacity logits get the depth term only, the SH the
                 RGB term only (two Adam optimizers, the same learning rates);
  NORM_DEC_INC   NORM_DEC with the depth term weighted by engine_core.incidence_weight of the measured depth and the
                 geometry gradient projected off the L2 anchor gradient when they conflict (engine_core.cone_project).
Every arm starts from the same CACHE state, runs 100 steps with a fresh optimizer and is rendered at the query
cameras (depth_z and alpha, 128x160). Logs per arm: losses, observed-view coverage / normalized error / raw error,
query coverage, Gaussian opacity, projections, failures, seconds. Every data file opened during an episode is
audited (the 7 arrived frames only).
Run with the DepthSplat environment from the DepthSplat repository root (PYTHONPATH=.):
  python engine_ttt.py ROOT OUT --model ZS|JN1 --arms CACHE NORM ... --cohorts A B [--shard k/n] [--limit N]
"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import PIL.Image
import torch

HERE = Path(__file__).resolve().parent
OUTPUTS = HERE.parents[1]
V1_CODE = OUTPUTS / "EXP-3D-RGBD-DEPTHSPLAT-JEPA-TTT-V1" / "code"
CAJU_CODE = OUTPUTS / "EXP-3D-CAJU-HEADROOM-V1" / "code"
for _path in (V1_CODE, CAJU_CODE, HERE):
    sys.path.insert(0, str(_path))
import ds_eval as v1  # noqa: E402  V1's locked file
import ds_predict  # noqa: E402
import engine_core as core  # noqa: E402
import engine_logic as L  # noqa: E402
from caju_logic import AccessAudit  # noqa: E402
from ds_finetune import measured_z  # noqa: E402
from ds_predict import INJECTED, build, inject_measured_depth, local, render_alpha, views  # noqa: E402
from src.model.types import Gaussians  # noqa: E402
from train_jepa import RENDER, render, set_scale_max  # noqa: E402

AUDIT = AccessAudit()
_np_load, _image_open = np.load, PIL.Image.open


def _audited_np_load(file, *args, **kwargs):
    if isinstance(file, (str, Path)):
        AUDIT.note(file)
    return _np_load(file, *args, **kwargs)


def _audited_image_open(fp, *args, **kwargs):
    if isinstance(fp, (str, Path)):
        AUDIT.note(fp)
    return _image_open(fp, *args, **kwargs)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sync():
    torch.cuda.synchronize()


def ttt_norm(decoder, gaussians, frames, ids, variant, device):
    """The NORM family; V1's parameterization (deltas of means, eigenbasis log-scales, opacity logits, SH)."""
    target = views(frames, ids, None, v1.NEAR, v1.FAR, device)
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
    lr = v1.TTT_BASE_LR
    groups = [{"params": [d_means], "lr": L.MULT * lr["means_per_metre"] * extent},
              {"params": [d_logs], "lr": L.MULT * lr["log_scales"]},
              {"params": [d_logit], "lr": L.MULT * lr["opacity_logits"]},
              {"params": [d_dc], "lr": L.MULT * lr["sh_dc"]},
              {"params": [d_rest], "lr": L.MULT * lr["sh_rest"]}]
    if variant == "NORM":
        optimizers = [torch.optim.Adam(groups, eps=1e-15)]
    else:
        optimizers = [torch.optim.Adam(groups[:3], eps=1e-15), torch.optim.Adam(groups[3:], eps=1e-15)]
    geom, app = [d_means, d_logs, d_logit], [d_dc, d_rest]
    weight = (z > 0).float()
    inc_share = None
    if variant == "NORM_DEC_INC":
        rows = {f["frame_id"]: f for f in frames}
        k = torch.tensor(np.array([rows[i]["intrinsics"] for i in ids], dtype=np.float64))
        weight, _, _ = core.incidence_weight(z, k, L.INC["cos_lo"], L.INC["cos_hi"], L.INC["edge_max"])
        inc_share = float((weight > 0).float().sum() / (z > 0).float().sum().clamp_min(1))
    conf = (opacity0 >= L.INC["anchor_conf_opacity"]).float()

    def state():
        scales2 = torch.exp(2 * (log_s0 + d_logs))
        covariances = (evecs * scales2[..., None, :]) @ evecs.transpose(-1, -2)
        return Gaussians(means0 + d_means, covariances, sh0 + torch.cat([d_dc, d_rest], dim=-1),
                         torch.sigmoid(logit0 + d_logit))

    log = {"depth_term": [], "rgb_term": [], "coverage": [], "projected_steps": 0, "failed_at_step": None,
           "incidence_weight_positive_share": inc_share, "extent_m": extent}
    for step in range(1, L.STEPS + 1):
        s = state()
        depth, color = render(decoder, s, target)
        alpha = render_alpha(s, target, RENDER)[0]
        depth_term, rgb_term, coverage = core.norm_terms(depth, color, alpha, z, rgb, weight, L.ALPHA_MIN)
        loss = depth_term + 0.5 * rgb_term
        if not torch.isfinite(loss):  # as V1: revert to CACHE and record the failure
            log["failed_at_step"] = step
            with torch.no_grad():
                for delta in geom + app:
                    delta.zero_()
            break
        if variant == "NORM":
            optimizers[0].zero_grad(set_to_none=True)
            loss.backward()
            optimizers[0].step()
        else:
            g_geom = torch.autograd.grad(depth_term, geom, retain_graph=True, allow_unused=True)
            g_app = torch.autograd.grad(rgb_term, app, allow_unused=True)
            g_geom = [torch.zeros_like(p) if g is None else g for g, p in zip(g_geom, geom)]
            g_app = [torch.zeros_like(p) if g is None else g for g, p in zip(g_app, app)]
            if variant == "NORM_DEC_INC":
                g_anchor = torch.autograd.grad(core.anchor_loss(d_means, d_logs, d_logit, conf), geom)
                g_geom, projected = core.cone_project(g_geom, g_anchor)
                log["projected_steps"] += int(projected)
            for p, g in zip(geom + app, g_geom + g_app):
                p.grad = g
            for optimizer in optimizers:
                optimizer.step()
        log["depth_term"].append(float(depth_term))
        log["rgb_term"].append(float(rgb_term))
        log["coverage"].append(float(coverage))
    with torch.no_grad():
        final = state()
    return Gaussians(*(t.detach() for t in (final.means, final.covariances, final.harmonics, final.opacities))), log


@torch.no_grad()
def observed_stats(decoder, gaussians, target, z):
    """Raw-depth relative L1 (V1's term), coverage and normalized error on covered valid pixels at the 7 views."""
    depth, _ = render(decoder, gaussians, target)
    alpha = render_alpha(gaussians, target, RENDER)[0]
    valid = z > 0
    covered = valid & (alpha >= L.ALPHA_MIN)
    norm = (depth / alpha.clamp_min(1e-6) - z).abs() / z.clamp_min(1e-6)
    return {"raw_depth_term": float(((depth - z).abs() / z.clamp_min(1e-6))[valid].mean()),
            "coverage": float(covered.sum() / valid.sum().clamp_min(1)),
            "normalized_error_covered": float(norm[covered].mean()) if bool(covered.any()) else None,
            "opacity_mean": float(gaussians.opacities.mean()),
            "opacity_share_ge_0_5": float((gaussians.opacities >= 0.5).float().mean())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--model", required=True, choices=sorted(L.CHECKPOINTS))
    parser.add_argument("--arms", nargs="+", required=True, choices=("CACHE",) + L.VARIANTS)
    parser.add_argument("--cohorts", nargs="+", required=True)
    parser.add_argument("--shard", default="0/1")
    parser.add_argument("--limit", type=int, default=0, help="rehearsal only: first N episodes")
    args = parser.parse_args()
    index, count = (int(x) for x in args.shard.split("/"))
    root = args.root.resolve()
    locks = json.loads((root / "frame_role_lock.json").read_text())["cohorts"]
    episodes, records = [], {}
    for cohort in args.cohorts:
        lock = locks[cohort]
        if sha(lock["manifest"]) != lock["manifest_sha256"]:
            raise PermissionError(f"{cohort}: manifest changed since the frame-role lock")
        for r in json.loads(Path(lock["manifest"]).read_text())["scenes"]:
            records[(cohort, r["scene_id"])] = r
        episodes += [dict(e, cohort=cohort) for e in lock["episodes"]]
    episodes = (episodes[: args.limit] if args.limit else episodes)[index::count]
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    device = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(L.SEED)
    set_scale_max(L.SCALE_MAX)
    checkpoint = Path(L.CHECKPOINTS[args.model])
    encoder, decoder = build(device, checkpoint)
    inject_measured_depth(encoder)
    INJECTED["align"] = False
    np.load, PIL.Image.open = _audited_np_load, _audited_image_open
    started, predictions, logs = time.time(), {}, {}
    for n, e in enumerate(episodes):
        frames = records[(e["cohort"], e["scene_id"])]["frames"]
        rows = {f["frame_id"]: f for f in frames}
        inputs, queries = e["context"] + e["stream"], e["queries"]
        if set(inputs) & set(queries):
            raise PermissionError("a query frame would enter the state")
        AUDIT.begin([local(rows[i][k]) for i in inputs for k in ("rgb", "depth")])
        cams = v1.cameras(frames, queries, device)
        obs_target = views(frames, inputs, None, v1.NEAR, v1.FAR, device)
        z = measured_z(frames, inputs).to(device)
        cached = v1.encode(encoder, frames, inputs, device)
        arrays, arm_logs = {}, {}
        for arm in args.arms:
            sync()
            t0 = time.perf_counter()
            if arm == "CACHE":
                state, arm_log = cached, {}
            elif arm == "RAW":
                state, arm_log = v1.ttt(decoder, cached, frames, inputs, L.STEPS, L.MULT, device)
            else:
                state, arm_log = ttt_norm(decoder, cached, frames, inputs, arm, device)
            sync()
            arm_log["seconds"] = time.perf_counter() - t0
            depth_z, alpha = v1.query_render(decoder, state, cams)
            arrays[arm] = (depth_z, alpha)
            arm_log["observed"] = observed_stats(decoder, state, obs_target, z)
            arm_log["query_coverage"] = float((alpha >= L.ALPHA_MIN).mean())
            arm_logs[arm] = arm_log
            if state is not cached:
                del state
        key = f"{e['scene_id']}/{e['role']}"
        for j, qid in enumerate(queries):
            data = {}
            for arm, (depth_z, alpha) in arrays.items():
                data[f"{arm}_depth_z"] = depth_z[j]
                data[f"{arm}_alpha"] = alpha[j]
            path = out / f"{e['scene_id']}_{e['role']}_{qid}.npz"
            np.savez_compressed(path, **data)
            predictions[f"{key}/{qid}"] = {"path": str(path), "sha256": sha(path), "cohort": e["cohort"]}
        violations = AUDIT.violations()
        if violations:
            raise PermissionError(f"{key}: files outside the arrived frames were opened: {violations[:3]}")
        logs[key] = {"cohort": e["cohort"], "place": e["place"], "arms": arm_logs, "files_opened": len(AUDIT.opened)}
        del cached
        if n == 0 or (n + 1) % 5 == 0:
            print(f"episode {n + 1}/{len(episodes)} {time.time() - started:.0f}s", flush=True)
    np.load, PIL.Image.open = _np_load, _image_open
    info = {"model": args.model, "checkpoint": str(checkpoint), "checkpoint_sha256": sha(checkpoint),
            "arms": args.arms, "cohorts": args.cohorts, "scale_max": L.SCALE_MAX, "ttt_steps": L.STEPS,
            "ttt_mult": L.MULT, "ttt_base_lr": v1.TTT_BASE_LR, "inc": L.INC, "seed": L.SEED,
            "overrides": list(ds_predict.OVERRIDES), "v1_ds_eval_sha256": sha(V1_CODE / "ds_eval.py"),
            "engine_ttt_sha256": sha(Path(__file__)), "engine_core_sha256": sha(HERE / "engine_core.py"),
            "shard": args.shard, "limit": args.limit, "episodes": len(episodes), "queries": len(predictions),
            "query_depth_read": False, "query_image_read": False, "access_violations": 0,
            "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0), "seconds": round(time.time() - started, 1),
            "max_memory_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
            "episodes_log": logs, "predictions": predictions}
    (out / f"engine_seal_{index}of{count}.json").write_text(json.dumps(info, indent=1))
    print("ENGINE_TTT_DONE", args.model, args.arms, len(episodes), "episodes", info["seconds"], "s",
          info["max_memory_gb"], "GB", flush=True)


if __name__ == "__main__":
    main()
