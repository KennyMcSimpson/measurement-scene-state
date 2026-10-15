"""EXP-3D-CAJU3D-FUSENET-V1: the learned per-pixel readout (torch only).

FuseNet maps the base image (8 channels, fn_data) and optionally the H-JEPA's query-cell prediction (66 channels at
64 x 80, projected 1 x 1 to 16 channels and upsampled) through a small dilated CNN to two maps:
  w = sigmoid(logit)        covered pixels (alpha >= 0.5): log pred = (1 - w) log R7 + w log F7   (Kalman-style weight)
  r = 0.7 tanh(raw)         uncovered pixels:               log pred = log R7 + r                (bounded correction)
The head starts at zero, so an untrained FuseNet reads exactly Core B (w = 0.5, r = 0).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

BASE_CH, JEPA_CH, JEPA_PROJ, WIDTH, DILATIONS, R_MAX = 8, 66, 16, 32, (1, 2, 4, 8, 1), 0.7


class FuseNet(nn.Module):
    def __init__(self, use_jepa):
        super().__init__()
        self.use_jepa = use_jepa
        self.project = nn.Sequential(nn.Conv2d(JEPA_CH, JEPA_PROJ, 1), nn.GELU()) if use_jepa else None
        layers, cin = [], BASE_CH + (JEPA_PROJ if use_jepa else 0)
        for d in DILATIONS:
            layers += [nn.Conv2d(cin, WIDTH, 3, padding=d, dilation=d), nn.GroupNorm(8, WIDTH), nn.GELU()]
            cin = WIDTH
        self.trunk = nn.Sequential(*layers)
        self.head = nn.Conv2d(WIDTH, 2, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, base, jepa=None):
        x = base
        if self.use_jepa:
            x = torch.cat([base, F.interpolate(self.project(jepa), size=base.shape[-2:], mode="nearest")], dim=1)
        out = self.head(self.trunk(x))
        return out[:, 0], R_MAX * torch.tanh(out[:, 1])


def log_prediction(base, logit, r, use_residual=True):
    """[B, H, W] log depth: the Kalman-style weight on the covered pixels, R7 (+ r) elsewhere."""
    log_r7, log_f7, covered = base[:, 0], base[:, 1], base[:, 6] > 0.5
    w = torch.sigmoid(logit)
    uncovered = log_r7 + r if use_residual else log_r7
    return torch.where(covered, (1 - w) * log_r7 + w * log_f7, uncovered)


def absrel_per_query(log_pred, gt):
    """[B] mean AbsRel over the valid ground-truth pixels (gt > 0) of each query."""
    valid = gt > 0
    err = torch.where(valid, (torch.exp(log_pred) - gt).abs() / gt.clamp_min(1e-6), torch.zeros_like(gt))
    return err.sum(dim=(1, 2)) / valid.sum(dim=(1, 2)).clamp_min(1)
