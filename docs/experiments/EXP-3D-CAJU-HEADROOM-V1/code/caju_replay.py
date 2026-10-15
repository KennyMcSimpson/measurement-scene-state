"""EXP-3D-CAJU-HEADROOM-V1: the counterfactual replay of OFF / SHORT / FULL from one CACHE state per episode (GPU).

Per episode (record x role) of the given cohorts:
  1. CACHE = V1's encode: released DepthSplat (cap 0.1) on the 3 context + 4 stream RGB-D views, measured depth
     injected. OFF is this state, unchanged.
  2. The pre-update features: the CACHE state rendered at the 7 arrived frames against their measured RGB-D (no query
     or history camera, image or depth is used).
  3. The TTT trajectory of V1's ds_eval.ttt, line for line (same parameterization, optimizer, learning rates, loss,
     failure rule). The state after optimizer step 10 is read out (no-grad renders only, the trajectory continues
     unchanged) as SHORT; the state after step 100 is FULL. A non-finite loss reverts to CACHE as in V1 (before step
     10: SHORT and FULL; later: FULL only).
  4. Every action's state is rendered at the query and history cameras (depth_z and alpha at 128x160, as V1) and at
     the 7 arrived frames (fit, observed-view damage).
Every data file opened during an episode is audited: it must be the RGB or measured depth of an arrived frame.
Segments are timed with CUDA synchronization. One npz per (episode, target frame) holds <ACTION>_depth_z and
<ACTION>_alpha; the shard's seal replay_seal_<k>of<n>.json holds the file hashes, settings and episode logs.
Run with the DepthSplat environment from the DepthSplat repository root (PYTHONPATH=.):
  python caju_replay.py ROOT OUT --cohorts CAJU_SEL CAJU_EVAL [--shard 0/1] [--limit N]
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
V1_CODE = HERE.parents[1] / "EXP-3D-RGBD-DEPTHSPLAT-JEPA-TTT-V1" / "code"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(V1_CODE))
import ds_eval as v1  # noqa: E402  V1's locked file (encode, cameras, query_render, ttt, learning rates)
import ds_predict  # noqa: E402
from caju_logic import ALPHA_MIN, RELIABLE_ERROR, STEPS, AccessAudit  # noqa: E402
from ds_finetune import measured_z  # noqa: E402
from ds_predict import INJECTED, build, inject_measured_depth, local, render_alpha, views  # noqa: E402
from src.model.types import Gaussians  # noqa: E402
from train_jepa import RENDER, grounded_loss, render, set_scale_max  # noqa: E402

SCALE_MAX, MULT, SEED = 0.1, 1.0, 20261008
SNAPSHOT, FULL_STEPS = STEPS["SHORT"], STEPS["FULL"]
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


def fingerprint(g):
    return {"gaussians": int(g.means.shape[1]),
            **{name: float(t.double().abs().sum()) for name, t in
               (("means", g.means), ("covariances", g.covariances), ("harmonics", g.harmonics),
                ("opacities", g.opacities))}}


def ttt_snapshots(decoder, gaussians, frames, ids, steps, snapshot, mult, device, on_state):
    """V1 ds_eval.ttt line for line, plus on_state(snapshot, state) after optimizer step `snapshot` and
    on_state(steps, state) at the end (detached states; renders only). Returns V1's log plus the per-step losses and
    the segment seconds."""
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
    optimizer = torch.optim.Adam([
        {"params": [d_means], "lr": mult * lr["means_per_metre"] * extent},
        {"params": [d_logs], "lr": mult * lr["log_scales"]},
        {"params": [d_logit], "lr": mult * lr["opacity_logits"]},
        {"params": [d_dc], "lr": mult * lr["sh_dc"]},
        {"params": [d_rest], "lr": mult * lr["sh_rest"]},
    ], eps=1e-15)

    def state():
        scales2 = torch.exp(2 * (log_s0 + d_logs))
        covariances = (evecs * scales2[..., None, :]) @ evecs.transpose(-1, -2)
        return Gaussians(means0 + d_means, covariances, sh0 + torch.cat([d_dc, d_rest], dim=-1),
                         torch.sigmoid(logit0 + d_logit))

    def detached():
        with torch.no_grad():
            s = state()
            return Gaussians(*(t.detach() for t in (s.means, s.covariances, s.harmonics, s.opacities)))

    with torch.no_grad():
        start = state()
        rebuild = float(((start.covariances - cov0).norm(dim=(-2, -1)) /
                         cov0.norm(dim=(-2, -1)).clamp_min(1e-20)).max())
    losses, failed, failed_at, taken, seconds = [], False, None, False, {}
    sync()
    started = time.perf_counter()
    for step in range(1, steps + 1):
        depth, color = render(decoder, state(), target)
        loss, depth_term = grounded_loss(depth, color, z, rgb)
        if not torch.isfinite(loss):  # V1's preregistered rule: the state reverts to CACHE, the failure is recorded
            failed, failed_at = True, step
            with torch.no_grad():
                for delta in (d_means, d_logs, d_logit, d_dc, d_rest):
                    delta.zero_()
            break
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append((float(loss), depth_term))
        if step == snapshot:
            sync()
            seconds["steps_1_to_snapshot"] = time.perf_counter() - started
            on_state(snapshot, detached())
            taken = True
            sync()
            started = time.perf_counter()
    sync()
    seconds["steps_after_snapshot" if taken else "steps_1_to_snapshot"] = time.perf_counter() - started
    if not taken:  # failed before the snapshot: a V1 run of `snapshot` steps would have reverted to CACHE too
        on_state(snapshot, detached())
    on_state(steps, detached())
    log = {"extent_m": extent, "covariance_rebuild_max_rel_error": rebuild, "failed_reverted_to_cache": failed,
           "failed_at_step": failed_at, "steps_done": len(losses),
           "loss_per_step": [x[0] for x in losses], "depth_term_per_step": [x[1] for x in losses],
           "mean_shift_max_m": float(d_means.detach().norm(dim=-1).max()), "seconds": seconds}
    return log


@torch.no_grad()
def observed(decoder, gaussians, target):
    depth, color = render(decoder, gaussians, target)
    alpha = render_alpha(gaussians, target, RENDER)[0]
    return depth.double(), color.double(), alpha.double()


def fit(depth, color, z, rgb):
    """V1's TTT loss on the arrived frames (relative L1 of raw depth on valid pixels + 0.5 x RGB L1)."""
    valid = z > 0
    depth_term = float(((depth - z).abs() / z.clamp_min(1e-6))[valid].mean())
    rgb_term = float((color - rgb).abs().mean())
    return {"loss": depth_term + 0.5 * rgb_term, "depth_term": depth_term, "rgb_l1": rgb_term}


def pre_update_features(depth, color, alpha, z, rgb, n_context, extrinsics):
    """Residual and state statistics of the CACHE state at the 7 arrived frames (context views first)."""
    valid = z > 0
    raw = ((depth - z).abs() / z.clamp_min(1e-6))
    covered = valid & (alpha >= ALPHA_MIN)
    norm = (depth / alpha.clamp_min(1e-6) - z).abs() / z.clamp_min(1e-6)
    rv = raw[valid].cpu().numpy()
    nc = norm[covered].cpu().numpy()
    zv = z[valid].cpu().numpy()
    per_view = [float(raw[v][valid[v]].mean()) if bool(valid[v].any()) else float("nan") for v in range(z.shape[0])]
    top = np.sort(rv)[::-1]
    k = max(1, int(round(0.1 * len(top))))
    centers = extrinsics[:, :3, 3].double().cpu().numpy()
    forward = extrinsics[:, :3, 2].double().cpu().numpy()
    forward /= np.linalg.norm(forward, axis=1, keepdims=True)
    angles = np.degrees(np.arccos(np.clip(forward @ forward.T, -1.0, 1.0)))
    baseline = np.linalg.norm(centers[:, None] - centers[None], axis=-1)

    def q(x, p):
        return float(np.percentile(x, p)) if len(x) else None

    p10, p50, p90 = (q(zv, p) for p in (10, 50, 90))
    return {
        "residual_raw_mean": float(rv.mean()), "residual_raw_p50": q(rv, 50), "residual_raw_p90": q(rv, 90),
        "residual_raw_p99": q(rv, 99), "rgb_l1": float((color - rgb).abs().mean()),
        "coverage": float(covered.sum() / max(int(valid.sum()), 1)),
        "residual_norm_covered_mean": float(nc.mean()) if len(nc) else None,
        "residual_norm_covered_p50": q(nc, 50), "residual_norm_covered_p90": q(nc, 90),
        "residual_norm_covered_p99": q(nc, 99),
        "valid_fraction": float(valid.double().mean()),
        "residual_top10_mass_share": float(top[:k].sum() / max(top.sum(), 1e-12)),
        "residual_view_max": float(np.nanmax(per_view)), "residual_view_min": float(np.nanmin(per_view)),
        "residual_view_std": float(np.nanstd(per_view)),
        "residual_context_mean": float(np.nanmean(per_view[:n_context])),
        "residual_stream_mean": float(np.nanmean(per_view[n_context:])),
        "measured_depth_p10": p10, "measured_depth_p50": p50, "measured_depth_p90": p90,
        "measured_depth_spread": (p90 - p10) / p50 if p50 else None,
        "camera_baseline_max_m": float(baseline.max()), "camera_angle_max_deg": float(angles.max()),
    }


def observed_damage(off, act, z):
    """Damage to the reliable observed region: pixels of the arrived frames where OFF is covered and its normalized
    depth is within RELIABLE_ERROR of the measurement. Mean error increase on the pixels the action still covers, and
    the share of reliable pixels the action breaks (uncovered or error above RELIABLE_ERROR)."""
    valid = z > 0

    def err(depth, alpha):
        return (depth / alpha.clamp_min(1e-6) - z).abs() / z.clamp_min(1e-6)

    e_off, e_act = err(off[0], off[2]), err(act[0], act[2])
    reliable = valid & (off[2] >= ALPHA_MIN) & (e_off <= RELIABLE_ERROR)
    kept = reliable & (act[2] >= ALPHA_MIN)
    broken = reliable & ((act[2] < ALPHA_MIN) | (e_act > RELIABLE_ERROR))
    n = int(reliable.sum())
    return {"reliable_pixels": n,
            "mean_error_increase": float(e_act[kept].mean() - e_off[kept].mean()) if bool(kept.any()) else None,
            "broken_share": float(broken.sum() / n) if n else None}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("out", type=Path)
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
    episodes = episodes[: args.limit] if args.limit else episodes
    episodes = episodes[index::count]
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    device = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(SEED)
    set_scale_max(SCALE_MAX)
    checkpoint = Path(ds_predict.CHECKPOINT)
    encoder, decoder = build(device, checkpoint)
    inject_measured_depth(encoder)
    INJECTED["align"] = False
    np.load, PIL.Image.open = _audited_np_load, _audited_image_open
    started, predictions, logs = time.time(), {}, {}
    for n, e in enumerate(episodes):
        record = records[(e["cohort"], e["scene_id"])]
        frames = record["frames"]
        rows = {f["frame_id"]: f for f in frames}
        inputs, targets = e["context"] + e["stream"], e["queries"] + e["history"]
        if set(inputs) & set(targets):
            raise PermissionError("a query or history frame would enter the state")
        AUDIT.begin([local(rows[i][k]) for i in inputs for k in ("rgb", "depth")])
        timing, arrays, maps = {}, {}, {}
        cams = v1.cameras(frames, targets, device)
        sync()
        t = time.perf_counter()
        cached = v1.encode(encoder, frames, inputs, device)
        sync()
        timing["encode"] = time.perf_counter() - t
        injection = dict(INJECTED.get("stats", {}))
        obs_target = views(frames, inputs, None, v1.NEAR, v1.FAR, device)
        z = measured_z(frames, inputs).to(device).double()
        rgb = obs_target["image"][0].double()
        names = {0: "OFF", SNAPSHOT: "SHORT", FULL_STEPS: "FULL"}

        def on_state(step, gaussians):
            name = names[step]
            sync()
            t0 = time.perf_counter()
            arrays[name] = v1.query_render(decoder, gaussians, cams)
            maps[name] = observed(decoder, gaussians, obs_target)
            sync()
            timing[f"readout_{name}"] = time.perf_counter() - t0

        on_state(0, cached)
        features = pre_update_features(*maps["OFF"], z, rgb, len(e["context"]), obs_target["extrinsics"][0])
        features.update({"gaussians": int(cached.means.shape[1]),
                         "depthsplat_vs_measured_median_ratio": injection.get("median_ratio"),
                         "depthsplat_vs_measured_absrel": injection.get("context_absrel")})
        ttt_log = ttt_snapshots(decoder, cached, frames, inputs, FULL_STEPS, SNAPSHOT, MULT, device, on_state)
        features["extent_m"] = ttt_log["extent_m"]
        timing.update(ttt_log.pop("seconds"))
        key = f"{e['scene_id']}/{e['role']}"
        for j, fid in enumerate(targets):
            data = {}
            for name, (depth_z, alpha) in arrays.items():
                data[f"{name}_depth_z"] = depth_z[j]
                data[f"{name}_alpha"] = alpha[j]
            path = out / f"{e['scene_id']}_{e['role']}_{fid}.npz"
            np.savez_compressed(path, **data)
            predictions[f"{key}/{fid}"] = {"path": str(path), "sha256": sha(path), "cohort": e["cohort"],
                                           "kind": "query" if fid in e["queries"] else "history"}
        violations = AUDIT.violations()
        if violations:
            raise PermissionError(f"{key}: files outside the arrived frames were opened: {violations[:3]}")
        logs[key] = {"cohort": e["cohort"], "place": e["place"], "cache_fingerprint": fingerprint(cached),
                     "features": features, "fit": {a: fit(m[0], m[1], z, rgb) for a, m in maps.items()},
                     "observed_damage": {a: observed_damage(maps["OFF"], maps[a], z) for a in ("SHORT", "FULL")},
                     "ttt": ttt_log, "timing_s": timing, "files_opened": len(AUDIT.opened),
                     "access_violations": violations}
        del cached
        if n == 0 or (n + 1) % 5 == 0:
            print(f"episode {n + 1}/{len(episodes)} {time.time() - started:.0f}s", flush=True)
    np.load, PIL.Image.open = _np_load, _image_open
    info = {"cohorts": args.cohorts, "checkpoint": str(checkpoint), "checkpoint_sha256": sha(checkpoint),
            "scale_max": SCALE_MAX, "ttt_mult": MULT, "ttt_steps": FULL_STEPS, "snapshot_step": SNAPSHOT,
            "ttt_base_lr": v1.TTT_BASE_LR, "seed": SEED, "overrides": list(ds_predict.OVERRIDES),
            "v1_ds_eval_sha256": sha(V1_CODE / "ds_eval.py"), "replay_sha256": sha(Path(__file__)),
            "shard": args.shard, "limit": args.limit, "episodes": len(episodes), "targets": len(predictions),
            "query_depth_read": False, "query_image_read": False, "history_depth_read": False,
            "history_image_read": False, "access_violations": 0, "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(0), "seconds": round(time.time() - started, 1),
            "max_memory_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
            "pose_orthonormalization_max_correction": ds_predict.ORTHO["max_correction"],
            "episodes_log": logs, "predictions": predictions}
    (out / f"replay_seal_{index}of{count}.json").write_text(json.dumps(info, indent=1))
    print("CAJU_REPLAY_DONE", args.cohorts, len(episodes), "episodes", len(predictions), "targets", info["seconds"],
          "s", info["max_memory_gb"], "GB", flush=True)


if __name__ == "__main__":
    main()
