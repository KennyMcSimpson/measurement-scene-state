"""CAJU-3D step 2 (planned): per-pixel inconsistency scores of the 7 arrived views and the matched-budget gate
(torch only).

For each arrived view v and each valid measured pixel, a score says how much that measurement disagrees with the
other arrived views; the gate drops the top `share` of the episode's valid pixels from the TTT depth term. Scores:
  JEPA    normalized latent innovation of view v given the other six (hjepa leave-one-out): mean_c (z - z_hat)^2 / P,
          at the latent cell of the pixel;
  DEPTH   cross-view measured-depth disagreement: |D_v - D_rep| / D_v, D_rep = the nearest measured point of the other
          six views z-buffered into v (no hit: score 0, nothing contradicts the pixel);
  INC     1 - the incidence weight of engine_core (grazing angles and depth jumps score high);
  RANDOM  a seeded uniform score (the null control).
All inputs are arrived frames only; nothing of a query frame is used.
"""

import torch
import torch.nn.functional as F

import hjepa as J


def depth_disagreement(planar, k, c2w, min_depth=0.05):
    """planar [V, H, W] measured planar depth (0 = invalid), k [V, 3, 3], c2w [V, 4, 4] -> [V, H, W] score."""
    v_n, h, w = planar.shape
    rows = torch.arange(h, dtype=torch.float32, device=planar.device).view(h, 1)
    cols = torch.arange(w, dtype=torch.float32, device=planar.device).view(1, w)
    scores = torch.zeros_like(planar)
    pts, ok = [], []
    for s in range(v_n):
        d = planar[s]
        cam = torch.stack([(cols - k[s, 0, 2]) / k[s, 0, 0] * d, (rows - k[s, 1, 2]) / k[s, 1, 1] * d, d], dim=-1)
        pts.append(cam.reshape(-1, 3) @ c2w[s, :3, :3].T + c2w[s, :3, 3])
        ok.append((d > 0).reshape(-1))
    for v in range(v_n):
        others = [s for s in range(v_n) if s != v]
        p = torch.cat([pts[s] for s in others])
        valid = torch.cat([ok[s] for s in others])
        cam = (p - c2w[v, :3, 3]) @ c2w[v, :3, :3]
        z = cam[:, 2]
        zs = z.clamp_min(1e-6)
        u = torch.round(k[v, 0, 0] * cam[:, 0] / zs + k[v, 0, 2]).long()
        r = torch.round(k[v, 1, 1] * cam[:, 1] / zs + k[v, 1, 2]).long()
        good = valid & (z > min_depth) & (u >= 0) & (u < w) & (r >= 0) & (r < h)
        flat = (r.clamp(0, h - 1) * w + u.clamp(0, w - 1))
        key = torch.where(good, z, torch.full_like(z, float("inf")))
        rep = torch.full((h * w,), float("inf"), device=planar.device)
        rep = rep.scatter_reduce(0, flat, key, reduce="amin", include_self=True).view(h, w)
        d = planar[v]
        hit = torch.isfinite(rep) & (d > 0)
        scores[v] = torch.where(hit, (d - rep).abs() / d.clamp_min(1e-6), torch.zeros_like(d))
    return scores


@torch.no_grad()
def jepa_innovation(model, rgb, ray, k, c2w):
    """Leave-one-out normalized latent innovation of each of the V arrived views, upsampled to the pixels.
    rgb [V, 3, H, W] in [0, 1], ray [V, H, W], k [V, 3, 3], c2w [V, 4, 4] -> [V, H, W]."""
    v_n = rgb.shape[0]
    out = []
    for v in range(v_n):
        others = [s for s in range(v_n) if s != v]
        res = model(rgb[others][None], ray[others][None], k[others][None], c2w[others][None],
                    rgb[v][None], ray[v][None], k[v][None], c2w[v][None])
        nu = ((res["z"] - res["z_hat"]) ** 2).mean(dim=1, keepdim=True) * torch.exp(-res["log_p"])
        out.append(F.interpolate(nu, size=(J.H, J.W), mode="nearest")[0, 0])
    return torch.stack(out)


def matched_gate(score, valid, share, generator=None):
    """1 everywhere except the top `share` of the valid pixels by score (ties broken by a seeded jitter)."""
    flat = score[valid]
    n_drop = int(round(share * flat.numel()))
    gate = valid.float().clone()
    if n_drop == 0:
        return gate
    jitter = torch.rand(flat.shape, generator=generator, device=flat.device) * 1e-9 if generator is not None else 0.0
    top = torch.topk(flat + jitter, n_drop).indices
    idx = valid.nonzero(as_tuple=False)[top]
    gate[tuple(idx.T)] = 0.0
    return gate
