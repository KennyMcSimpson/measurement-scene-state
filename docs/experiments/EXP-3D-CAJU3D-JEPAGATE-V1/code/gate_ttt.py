"""EXP-3D-CAJU3D-JEPAGATE-V1: Core B's TTT with a matched-budget pixel gate on the depth term (GPU replay).

Per episode (released DepthSplat, cap 0.1, the 7-view CACHE state of V1's encode):
  scores of every valid measured pixel of the 7 arrived views (gates.py): JEPA (the trained geometry-aligned reverse
  JEPA, leave-one-out normalized latent innovation), DEPTH (cross-view measured-depth disagreement), INC
  (1 - engine_core.incidence_weight), RANDOM (seeded);
  for each gate arm, the top SHARE of the episode's valid pixels by that score is dropped from the depth term of V1's
  TTT loss (weighted mean of the relative depth error over the kept valid pixels + 0.5 x RGB L1, unchanged); with no
  pixel dropped the loss is V1's exactly. 100 Adam steps, V1's parameterization, learning rates and failure rule.
  NONE (no pixel dropped) is run only in the rehearsal, to check the engine against the sealed Core B.
Every arm is rendered at the query cameras (depth_z, alpha). Logs: dropped-set overlaps between the scores, the
scores' summary, losses, seconds. Every data file opened during an episode is audited (arrived frames only).
Run with the DepthSplat environment from the DepthSplat repository root (PYTHONPATH=.):
  python gate_ttt.py ROOT OUT --cohorts A B --arms JEPA DEPTH INC RANDOM [--shard k/n] [--limit N]
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
ENGINE_CODE = OUTPUTS / "EXP-3D-CAJU3D-ENGINE-V1" / "code"
for _path in (V1_CODE, CAJU_CODE, ENGINE_CODE, HERE):
    sys.path.insert(0, str(_path))
import ds_eval as v1  # noqa: E402
import ds_predict  # noqa: E402
import engine_core  # noqa: E402
import gates as G  # noqa: E402
import hjepa as J  # noqa: E402
from caju_logic import AccessAudit  # noqa: E402
from ds_finetune import measured_z  # noqa: E402
from ds_predict import INJECTED, build, inject_measured_depth, local, views  # noqa: E402
from src.model.types import Gaussians  # noqa: E402
from train_jepa import RENDER, render, set_scale_max  # noqa: E402

SCALE_MAX, STEPS, MULT, SEED, SHARE = 0.1, 100, 1.0, 20261008, 0.10
INC = {"cos_lo": 0.1, "cos_hi": 0.4, "edge_max": 0.1}
ARMS = ("NONE", "JEPA", "DEPTH", "INC", "RANDOM")
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


def ttt_weighted(decoder, gaussians, frames, ids, weight, device):
    """V1's ds_eval.ttt with the depth term averaged over the kept valid pixels (weight 1) only."""
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
        {"params": [d_means], "lr": MULT * lr["means_per_metre"] * extent},
        {"params": [d_logs], "lr": MULT * lr["log_scales"]},
        {"params": [d_logit], "lr": MULT * lr["opacity_logits"]},
        {"params": [d_dc], "lr": MULT * lr["sh_dc"]},
        {"params": [d_rest], "lr": MULT * lr["sh_rest"]},
    ], eps=1e-15)
    w = weight.to(device) * (z > 0).float()

    def state():
        scales2 = torch.exp(2 * (log_s0 + d_logs))
        covariances = (evecs * scales2[..., None, :]) @ evecs.transpose(-1, -2)
        return Gaussians(means0 + d_means, covariances, sh0 + torch.cat([d_dc, d_rest], dim=-1),
                         torch.sigmoid(logit0 + d_logit))

    losses, failed = [], None
    for step in range(1, STEPS + 1):
        depth, color = render(decoder, state(), target)
        depth_term = ((depth - z).abs() / z.clamp_min(1e-6) * w).sum() / w.sum().clamp_min(1.0)
        loss = depth_term + 0.5 * (color - rgb).abs().mean()
        if not torch.isfinite(loss):
            failed = step
            with torch.no_grad():
                for delta in (d_means, d_logs, d_logit, d_dc, d_rest):
                    delta.zero_()
            break
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(depth_term))
    with torch.no_grad():
        final = state()
    out = Gaussians(*(t.detach() for t in (final.means, final.covariances, final.harmonics, final.opacities)))
    return out, {"depth_term_first": losses[0] if losses else None, "depth_term_last": losses[-1] if losses else None,
                 "failed_at_step": failed, "kept_valid_share": float(w.sum() / (z > 0).float().sum().clamp_min(1))}


def episode_scores(model, frames, inputs, device, generator):
    """Scores of every pixel of the 7 arrived views (JEPA, DEPTH, INC, RANDOM) and the valid mask."""
    rows = {f["frame_id"]: f for f in frames}
    rgb = views(frames, inputs, None, v1.NEAR, v1.FAR, device)["image"][0]  # [V, 3, H, W]
    planar = measured_z(frames, inputs).to(device)  # [V, H, W]
    k = torch.tensor(np.array([rows[i]["intrinsics"] for i in inputs], dtype=np.float32), device=device)
    c2w = torch.tensor(np.array([ds_predict.orthonormal(np.array(rows[i]["c2w"], dtype=np.float64))
                                 for i in inputs], dtype=np.float32), device=device)
    ray = planar * J.ray_factor(k, device=device)
    valid = planar > 0
    scores = {"JEPA": G.jepa_innovation(model, rgb, ray, k, c2w),
              "DEPTH": G.depth_disagreement(planar, k, c2w),
              "INC": 1.0 - engine_core.incidence_weight(planar, k, INC["cos_lo"], INC["cos_hi"], INC["edge_max"])[0],
              "RANDOM": torch.rand(planar.shape, generator=generator, device=device)}
    return scores, valid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--cohorts", nargs="+", required=True)
    parser.add_argument("--arms", nargs="+", required=True, choices=ARMS)
    parser.add_argument("--model", type=Path, required=True, help="the trained reverse JEPA (hjepa_final.pth)")
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
    torch.manual_seed(SEED)
    set_scale_max(SCALE_MAX)
    checkpoint = Path(ds_predict.CHECKPOINT)
    encoder, decoder = build(device, checkpoint)
    inject_measured_depth(encoder)
    INJECTED["align"] = False
    jepa = J.GeometricReverseJEPA()
    jepa.load_state_dict(torch.load(args.model, map_location="cpu", weights_only=False)["state_dict"])
    jepa = jepa.to(device).eval()
    np.load, PIL.Image.open = _audited_np_load, _audited_image_open
    started, predictions, logs = time.time(), {}, {}
    for n, e in enumerate(episodes):
        frames = records[(e["cohort"], e["scene_id"])]["frames"]
        rows = {f["frame_id"]: f for f in frames}
        inputs, queries = e["context"] + e["stream"], e["queries"]
        if set(inputs) & set(queries):
            raise PermissionError("a query frame would enter the state")
        key = f"{e['scene_id']}/{e['role']}"
        AUDIT.begin([local(rows[i][k]) for i in inputs for k in ("rgb", "depth")])
        generator = torch.Generator(device=device).manual_seed(int(hashlib.sha256(key.encode()).hexdigest()[:8], 16))
        cams = v1.cameras(frames, queries, device)
        cached = v1.encode(encoder, frames, inputs, device)
        sync()
        t0 = time.perf_counter()
        scores, valid = episode_scores(jepa, frames, inputs, device, generator)
        sync()
        gates = {arm: (torch.ones_like(valid, dtype=torch.float32) if arm == "NONE"
                       else G.matched_gate(scores[arm], valid, SHARE, generator)) for arm in args.arms}
        dropped = {arm: (valid & (g == 0)) for arm, g in gates.items()}
        overlap = {f"{a}|{b}": float((dropped[a] & dropped[b]).sum() / max(int(dropped[a].sum()), 1))
                   for i, a in enumerate(args.arms) for b in args.arms[i + 1:] if a != "NONE" and b != "NONE"}
        arrays, arm_logs = {}, {"scores_seconds": time.perf_counter() - t0, "dropped_overlap": overlap,
                                "valid_pixels": int(valid.sum()),
                                "score_summary": {a: [float(s[valid].mean()), float(s[valid].max())]
                                                  for a, s in scores.items()}}
        for arm in args.arms:
            sync()
            t1 = time.perf_counter()
            state, log = ttt_weighted(decoder, cached, frames, inputs, gates[arm], device)
            sync()
            log["seconds"] = time.perf_counter() - t1
            arrays[arm] = v1.query_render(decoder, state, cams)
            arm_logs[arm] = log
            del state
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
        logs[key] = {"cohort": e["cohort"], "place": e["place"], "arms": arm_logs}
        del cached
        if n == 0 or (n + 1) % 5 == 0:
            print(f"episode {n + 1}/{len(episodes)} {time.time() - started:.0f}s", flush=True)
    np.load, PIL.Image.open = _np_load, _image_open
    info = {"arms": args.arms, "cohorts": args.cohorts, "share": SHARE, "inc": INC, "scale_max": SCALE_MAX,
            "ttt_steps": STEPS, "ttt_mult": MULT, "seed": SEED, "checkpoint_sha256": sha(checkpoint),
            "jepa_model": str(args.model), "jepa_model_sha256": sha(args.model),
            "gate_ttt_sha256": sha(Path(__file__)), "gates_sha256": sha(HERE / "gates.py"),
            "hjepa_sha256": sha(HERE / "hjepa.py"), "v1_ds_eval_sha256": sha(V1_CODE / "ds_eval.py"),
            "shard": args.shard, "limit": args.limit, "episodes": len(episodes), "queries": len(predictions),
            "query_depth_read": False, "query_image_read": False, "access_violations": 0,
            "seconds": round(time.time() - started, 1),
            "max_memory_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
            "episodes_log": logs, "predictions": predictions}
    (out / f"gate_seal_{index}of{count}.json").write_text(json.dumps(info, indent=1))
    print("GATE_TTT_DONE", args.arms, len(episodes), "episodes", info["seconds"], "s", info["max_memory_gb"], "GB",
          flush=True)


if __name__ == "__main__":
    main()
