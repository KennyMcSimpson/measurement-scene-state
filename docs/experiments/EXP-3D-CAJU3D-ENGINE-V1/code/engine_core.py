"""EXP-3D-CAJU3D-ENGINE-V1: the math of the CAJU-3D TTT variants (torch only, no DepthSplat; unit-tested on CPU).

- incidence_weight: per-pixel weight of the measured depth, from the incidence cosine |n . ray| of the surface the
  measured points form (finite differences of the back-projected points) and the relative depth jump to the right and
  lower neighbours: w = clip((cos - cos_lo) / (cos_hi - cos_lo), 0, 1) x [jump < edge_max]; 0 where the pixel or a
  neighbour has no measurement (and on the last row / column).
- norm_terms: the alpha-normalized loss on covered valid pixels (alpha >= 0.5; the mask carries no gradient):
  depth term = weighted mean of |depth/alpha - z| / z, RGB term = mean of |color/alpha - rgb| (channel mean).
- anchor_loss / cone_project: the package's L2 anchor on the geometry deltas (positions weighted by the initial
  opacity >= 0.65) and the projection of the geometry gradient off the anchor gradient when their inner product is
  negative.
"""

import torch
import torch.nn.functional as F


def incidence_weight(z, intrinsics, cos_lo, cos_hi, edge_max):
    """z: [V, H, W] measured planar depth (0 = invalid); intrinsics: [V, 3, 3] in pixels of z (index coordinates, as
    ds_predict.measured_planar). Returns ([V, H, W] weight, [V, H, W] cosine, [V, H, W] jump)."""
    z = z.double()
    v, h, w = z.shape
    rows = torch.arange(h, dtype=torch.float64, device=z.device).view(1, h, 1)
    cols = torch.arange(w, dtype=torch.float64, device=z.device).view(1, 1, w)
    k = intrinsics.double().to(z.device)
    fx, fy = k[:, 0, 0].view(v, 1, 1), k[:, 1, 1].view(v, 1, 1)
    cx, cy = k[:, 0, 2].view(v, 1, 1), k[:, 1, 2].view(v, 1, 1)
    p = torch.stack([(cols - cx) * z / fx, (rows - cy) * z / fy, z], dim=1)  # [V, 3, H, W]
    dx, dy = torch.zeros_like(p), torch.zeros_like(p)
    dx[..., :, :-1] = p[..., :, 1:] - p[..., :, :-1]
    dy[..., :-1, :] = p[..., 1:, :] - p[..., :-1, :]
    normal = F.normalize(torch.cross(dx, dy, dim=1), dim=1, eps=1e-12)
    ray = F.normalize(p, dim=1, eps=1e-12)
    cosine = (normal * ray).sum(dim=1).abs()
    valid = z > 0
    ok = valid.clone()
    ok[..., :, :-1] &= valid[..., :, 1:]
    ok[..., :-1, :] &= valid[..., 1:, :]
    ok[..., :, -1] = False
    ok[..., -1, :] = False
    jx, jy = torch.zeros_like(z), torch.zeros_like(z)
    jx[..., :, :-1] = (z[..., :, 1:] - z[..., :, :-1]).abs() / z[..., :, :-1].clamp_min(0.1)
    jy[..., :-1, :] = (z[..., 1:, :] - z[..., :-1, :]).abs() / z[..., :-1, :].clamp_min(0.1)
    jump = torch.sqrt(jx ** 2 + jy ** 2)
    weight = ((cosine - cos_lo) / (cos_hi - cos_lo)).clamp(0, 1) * (jump < edge_max) * ok
    return weight.float(), torch.where(ok, cosine, torch.zeros_like(cosine)).float(), jump.float()


def norm_terms(depth, color, alpha, z, rgb, weight, alpha_min):
    """depth, alpha, z, weight: [V, H, W]; color, rgb: [V, 3, H, W]. Returns (depth term, RGB term, coverage)."""
    valid = z > 0
    covered = (valid & (alpha.detach() >= alpha_min)).float()
    a = alpha.clamp_min(1e-6)
    dz = (depth / a - z).abs() / z.clamp_min(1e-6)
    dc = (color / a.unsqueeze(1) - rgb).abs().mean(dim=1)
    wd = covered * weight
    depth_term = (wd * dz).sum() / wd.sum().clamp_min(1.0)
    rgb_term = (covered * dc).sum() / covered.sum().clamp_min(1.0)
    coverage = covered.sum() / valid.float().sum().clamp_min(1.0)
    return depth_term, rgb_term, coverage


def anchor_loss(d_means, d_logs, d_logit, conf):
    """The package's L2 anchor of the geometry deltas; conf: [B, G] (initial opacity >= 0.65)."""
    return (conf * (d_means ** 2).sum(dim=-1)).mean() + 0.1 * (d_logs ** 2).mean() + 0.05 * (d_logit ** 2).mean()


def cone_project(grads, anchor_grads):
    """Remove the component of grads along anchor_grads when <grads, anchor_grads> < 0 (summed over tensors).
    Returns (grads, projected?)."""
    dot = sum((g * a).sum() for g, a in zip(grads, anchor_grads))
    norm = sum((a * a).sum() for a in anchor_grads)
    if float(dot) < 0 and float(norm) > 1e-30:
        scale = dot / norm
        return [g - scale * a for g, a in zip(grads, anchor_grads)], True
    return list(grads), False
