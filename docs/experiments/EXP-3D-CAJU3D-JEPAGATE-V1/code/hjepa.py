"""CAJU-3D module 2A, corrected: a geometry-aligned reverse JEPA over RGB-D views (torch only).

Differences from the package's caju_h_jepa.py: the context is aligned to the target view geometrically before any
prediction (every source view's latent cells are back-projected with their measured depth, moved into the target
camera and z-buffered onto the target's latent grid), instead of mixing views by pixel index.
  online encoder E      RGB + log measured depth + validity (5 channels, 128x160) -> C x 64 x 80
  target encoder        EMA copy of E (stop-gradient), applied to the target view's own RGB-D
  warp                  source latent cells -> target latent grid (nearest point wins; ties averaged)
  predictor P           [warped features, hit mask, warped log depth, target Pluecker rays] -> (z_hat, log P)
  loss                  per-cell Gaussian NLL 0.5 |z - z_hat|^2 / P + 0.5 log P (channel mean) + a variance floor on
                        the online features (VICReg-style std >= 1 per channel)
Innovation at deployment: dz = z_target - z_hat, with the structural variance P.
Pixel conventions follow ds_predict.measured_planar: index coordinates, intrinsics in pixels of the 128x160 frames,
records store ray distance (planar z = ray / sqrt(((u - cx) / fx)^2 + ((v - cy) / fy)^2 + 1)).
"""

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

LATENT, H, W = 64, 128, 160
H2, W2 = H // 2, W // 2


def ray_factor(k, h=H, w=W, device=None):
    """sqrt(((u - cx)/fx)^2 + ((v - cy)/fy)^2 + 1) per pixel; k: [..., 3, 3] -> [..., h, w]."""
    v = torch.arange(h, dtype=torch.float32, device=device).view(h, 1)
    u = torch.arange(w, dtype=torch.float32, device=device).view(1, w)
    fx, fy = k[..., 0, 0, None, None], k[..., 1, 1, None, None]
    cx, cy = k[..., 0, 2, None, None], k[..., 1, 2, None, None]
    return torch.sqrt(((u - cx) / fx) ** 2 + ((v - cy) / fy) ** 2 + 1.0)


def encoder_input(rgb, ray_depth):
    """rgb [..., 3, H, W] in [0, 1], ray_depth [..., H, W] (0 = invalid) -> [..., 5, H, W]."""
    valid = (ray_depth > 0).float()
    logd = torch.log(ray_depth.clamp_min(1e-3)) * valid
    return torch.cat([rgb, logd.unsqueeze(-3), valid.unsqueeze(-3)], dim=-3)


def cell_points(planar_z, k, c2w):
    """World points of the latent cells of each view: planar_z [B, S, H, W] (0 = invalid), k [B, S, 3, 3],
    c2w [B, S, 4, 4] -> points [B, S, H2*W2, 3], valid [B, S, H2*W2]. A cell uses the depth of its top-left pixel
    and the ray through the cell centre (2j + 0.5, 2i + 0.5)."""
    b, s = planar_z.shape[:2]
    d = planar_z[..., 0::2, 0::2]  # [B, S, H2, W2]
    v = (torch.arange(H2, dtype=torch.float32, device=d.device) * 2 + 0.5).view(H2, 1)
    u = (torch.arange(W2, dtype=torch.float32, device=d.device) * 2 + 0.5).view(1, W2)
    fx, fy = k[..., 0, 0, None, None], k[..., 1, 1, None, None]
    cx, cy = k[..., 0, 2, None, None], k[..., 1, 2, None, None]
    cam = torch.stack([(u - cx) / fx * d, (v - cy) / fy * d, d], dim=-1).reshape(b, s, -1, 3)
    world = cam @ c2w[..., :3, :3].transpose(-1, -2) + c2w[..., None, :3, 3]
    return world, (d > 0).reshape(b, s, -1)


def warp_to_target(feat, points, valid, k_t, c2w_t, min_depth=0.05):
    """Z-buffer source cells onto the target latent grid.
    feat [B, S, C, H2, W2], points [B, S, N, 3], valid [B, S, N], k_t [B, 3, 3], c2w_t [B, 4, 4]
    -> warped [B, C, H2, W2], hit [B, 1, H2, W2], log depth of the nearest point [B, 1, H2, W2] (0 where no hit)."""
    b, s, c = feat.shape[:3]
    f = feat.reshape(b, s, c, -1).permute(0, 1, 3, 2).reshape(b, -1, c)  # [B, S*N, C]
    p = points.reshape(b, -1, 3)
    ok = valid.reshape(b, -1)
    rot, t = c2w_t[:, :3, :3], c2w_t[:, :3, 3]
    cam = (p - t[:, None, :]) @ rot  # world -> target camera: R^T (p - t)
    z = cam[..., 2]
    zs = z.clamp_min(1e-6)
    u = k_t[:, None, 0, 0] * cam[..., 0] / zs + k_t[:, None, 0, 2]
    v = k_t[:, None, 1, 1] * cam[..., 1] / zs + k_t[:, None, 1, 2]
    j = torch.floor((u + 0.5) / 2).long()
    i = torch.floor((v + 0.5) / 2).long()
    ok = ok & (z > min_depth) & (j >= 0) & (j < W2) & (i >= 0) & (i < H2)
    flat = (i.clamp(0, H2 - 1) * W2 + j.clamp(0, W2 - 1))
    big = torch.full_like(z, float("inf"))
    key = torch.where(ok, z, big)
    nearest = torch.full((b, H2 * W2), float("inf"), device=z.device, dtype=z.dtype)
    nearest = nearest.scatter_reduce(1, flat, key, reduce="amin", include_self=True)
    winner = ok & (key <= nearest.gather(1, flat) * (1 + 1e-6))
    w = winner.float()
    total = torch.zeros(b, H2 * W2, c, device=z.device, dtype=f.dtype)
    total = total.scatter_add(1, flat[..., None].expand(-1, -1, c), f * w[..., None])
    count = torch.zeros(b, H2 * W2, device=z.device, dtype=f.dtype).scatter_add(1, flat, w)
    hit = (count > 0).float()
    warped = (total / count.clamp_min(1.0)[..., None]).permute(0, 2, 1).reshape(b, c, H2, W2)
    logd = torch.where(hit > 0, torch.log(nearest.clamp_min(1e-3)), torch.zeros_like(nearest))
    return warped, hit.view(b, 1, H2, W2), logd.view(b, 1, H2, W2).float()


def pluecker(k, c2w, h=H2, w=W2, stride=2):
    """6-channel Pluecker rays (direction, moment) at the latent cell centres: k [B, 3, 3] (pixels of 128x160),
    c2w [B, 4, 4] -> [B, 6, h, w]."""
    v = (torch.arange(h, dtype=torch.float32, device=k.device) * stride + 0.5).view(h, 1)
    u = (torch.arange(w, dtype=torch.float32, device=k.device) * stride + 0.5).view(1, w)
    fx, fy = k[:, 0, 0, None, None], k[:, 1, 1, None, None]
    cx, cy = k[:, 0, 2, None, None], k[:, 1, 2, None, None]
    x = ((u - cx) / fx).expand(-1, h, w)
    y = ((v - cy) / fy).expand(-1, h, w)
    d_cam = torch.stack([x, y, torch.ones_like(x)], dim=-1)  # [B, h, w, 3]
    d = F.normalize(d_cam @ c2w[:, None, :3, :3].transpose(-1, -2), dim=-1)
    o = c2w[:, None, None, :3, 3].expand_as(d)
    return torch.cat([d, torch.cross(o, d, dim=-1)], dim=-1).permute(0, 3, 1, 2)


def conv_block(cin, cout, stride=1):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, stride=stride, padding=1), nn.GroupNorm(8, cout), nn.GELU())


class GeometricReverseJEPA(nn.Module):
    def __init__(self, latent=LATENT, hidden=128, ema=0.99):
        super().__init__()
        self.ema = ema
        self.encoder = nn.Sequential(conv_block(5, 32, 2), conv_block(32, latent), nn.Conv2d(latent, latent, 1))
        self.target_encoder = copy.deepcopy(self.encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad_(False)
        self.predictor = nn.Sequential(conv_block(latent + 8, hidden), conv_block(hidden, hidden),
                                       nn.Conv2d(hidden, hidden, 3, padding=2, dilation=2), nn.GroupNorm(8, hidden),
                                       nn.GELU(), conv_block(hidden, hidden), nn.Conv2d(hidden, latent + 1, 1))

    @torch.no_grad()
    def update_target(self):
        for p, q in zip(self.encoder.parameters(), self.target_encoder.parameters()):
            q.mul_(self.ema).add_(p.detach(), alpha=1 - self.ema)

    def forward(self, src_rgb, src_ray, src_k, src_c2w, tgt_rgb, tgt_ray, tgt_k, tgt_c2w):
        """src_*: [B, S, ...] (rgb [B, S, 3, H, W], ray depth [B, S, H, W], k [B, S, 3, 3], c2w [B, S, 4, 4]);
        tgt_*: [B, ...]. Returns z (target encoder, no grad), z_hat, log_p, the online features of the sources, hit."""
        b, s = src_rgb.shape[:2]
        feats = self.encoder(encoder_input(src_rgb, src_ray).reshape(b * s, 5, H, W)).reshape(b, s, -1, H2, W2)
        planar = src_ray / ray_factor(src_k, device=src_ray.device)
        points, valid = cell_points(planar, src_k, src_c2w)
        warped, hit, logd = warp_to_target(feats, points, valid, tgt_k, tgt_c2w)
        out = self.predictor(torch.cat([warped, hit, logd, pluecker(tgt_k, tgt_c2w)], dim=1))
        z_hat, log_p = out[:, :-1], out[:, -1:].clamp(-6.0, 4.0)
        with torch.no_grad():
            z = self.target_encoder(encoder_input(tgt_rgb, tgt_ray))
        return {"z": z, "z_hat": z_hat, "log_p": log_p, "feats": feats, "hit": hit}


def jepa_loss(out, var_weight=1.0):
    """Per-cell Gaussian NLL (channel mean) + a std floor of 1 per channel on the online source features."""
    err = ((out["z"] - out["z_hat"]) ** 2).mean(dim=1, keepdim=True)
    nll = (0.5 * err * torch.exp(-out["log_p"]) + 0.5 * out["log_p"]).mean()
    f = out["feats"].permute(0, 1, 3, 4, 2).reshape(-1, out["feats"].shape[2])
    var = F.relu(1.0 - torch.sqrt(f.var(dim=0) + 1e-4)).mean()
    return nll + var_weight * var, {"nll": float(nll), "var": float(var), "mse": float(err.mean())}
