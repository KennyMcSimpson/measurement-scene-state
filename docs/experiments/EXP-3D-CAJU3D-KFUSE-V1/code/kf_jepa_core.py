"""EXP-3D-CAJU3D-KFUSE-V1: the H-JEPA prediction at a camera with no frame behind it (torch only)."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "EXP-3D-CAJU3D-JEPAGATE-V1" / "code"))
import hjepa as J  # noqa: E402


@torch.no_grad()
def predict_at(model, src_rgb, src_ray, src_k, src_c2w, tgt_k, tgt_c2w):
    """GeometricReverseJEPA.forward without the target encoder: sources rgb [S, 3, H, W], ray depth [S, H, W],
    k [S, 3, 3], c2w [S, 4, 4]; target cameras k [T, 3, 3], c2w [T, 4, 4]
    -> z_hat [T, C, H2, W2], log_p [T, 1, H2, W2], hit [T, 1, H2, W2]."""
    s = src_rgb.shape[0]
    feats = model.encoder(J.encoder_input(src_rgb, src_ray)).reshape(1, s, -1, J.H2, J.W2)
    planar = src_ray / J.ray_factor(src_k, device=src_ray.device)
    points, valid = J.cell_points(planar[None], src_k[None], src_c2w[None])
    t = tgt_k.shape[0]
    warped, hit, logd = J.warp_to_target(feats.expand(t, -1, -1, -1, -1), points.expand(t, -1, -1, -1),
                                         valid.expand(t, -1, -1), tgt_k, tgt_c2w)
    out = model.predictor(torch.cat([warped, hit, logd, J.pluecker(tgt_k, tgt_c2w)], dim=1))
    return out[:, :-1], out[:, -1:].clamp(-6.0, 4.0), hit
