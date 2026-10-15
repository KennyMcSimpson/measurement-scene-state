"""EXP-3D-READOUT-METHOD-V1: the Phase 1 calibration controls (B2-B5), the learned readouts (B6, B6W, B7) and the
Phase 2 matched system readouts (SYS_A, SYS_B, SYS_C).

Phase 1: every method keeps the frozen KF_BASE table on the covered pixels (TTT alpha >= 0.5) and differs only on the
uncovered pixels, where the prediction starts at R7. Fits use the training places only; the objective is always the
place-weighted AbsRel (each query weighted as in the place-level mean, each of its valid pixels equally).
  B2A  zero-hit queries (no reprojection hit, R7 = the Hypersim TRAIN constant) -> one train-calibrated constant
  B2B  B2A + one scale for uncovered near7 pixels and one for uncovered fill pixels
  B3   B2A + a log-affine map a + b log R7 on the other uncovered pixels
  B4   B2A-free low-dimensional linear correction log R7 + theta . phi(x) on every uncovered pixel (12 features)
  B5   per-bin (near7 x zero-hit x alpha 6 x |log RAY - log R7| 4 x log R7 3 = 288 bins) blend of R7 and the TTT ray
       plus an offset, both from a grid
  B6   per-pixel MLP 8-32-32-1 on V2's 8 input channels (no spatial context), bounded correction 0.7 tanh
  B6W  per-pixel MLP 8-150-150-150-1 (about V2's parameter count, still no spatial context)
  B7   V2 (dilated CNN 1-2-4-8-16-1, width 32) retrained by SYSTEM4-HOLDOUT-V1's recipe
Phase 2: one readout family for all systems: log pred = log R7 + tanh(net) on EVERY pixel, net = V2's trunk;
SYS_A gets the 10 geometric channels, SYS_B adds the feed-forward render's 4 channels, SYS_C the TTT render's 4.
"""

import numpy as np
import torch
import torch.nn as nn

WEIGHT_GRID = np.round(np.linspace(0, 1, 11), 2)
OFFSET_GRID = np.round(np.linspace(-0.6, 0.6, 25), 3)
B_GRID = np.linspace(0.5, 1.5, 101)
DILATIONS, WIDTH = (1, 2, 4, 8, 16, 1), 32


def weighted_median(values, weights):
    order = np.argsort(values)
    cw = np.cumsum(weights[order])
    return float(values[order][np.searchsorted(cw, 0.5 * cw[-1])])


def best_scale(x, g, w):
    """AbsRel-optimal c for prediction c * x: weighted median of g / x with weights w x / g."""
    return weighted_median(g / x, w * x / g)


def fit_b2(unc):
    """unc: dict of 1-D arrays over the training uncovered pixels (r7, g, w, near, zero)."""
    z = unc["zero"]
    c_zero = best_scale(np.ones(z.sum()), unc["g"][z], unc["w"][z]) if z.any() else 2.3736329078674316
    nz = ~z
    s_near = best_scale(unc["r7"][nz & unc["near"]], unc["g"][nz & unc["near"]], unc["w"][nz & unc["near"]])
    s_fill = best_scale(unc["r7"][nz & ~unc["near"]], unc["g"][nz & ~unc["near"]], unc["w"][nz & ~unc["near"]])
    return {"c_zero": c_zero, "s_near": s_near, "s_fill": s_fill}


def fit_b3(unc, c_zero):
    nz = ~unc["zero"]
    lr, g, w = np.log(unc["r7"][nz]), unc["g"][nz], unc["w"][nz]
    best = None
    for b in B_GRID:
        x = np.exp(b * lr)
        c = best_scale(x, g, w)
        loss = float((w * np.abs(c * x - g) / g).sum())
        if best is None or loss < best[0]:
            best = (loss, float(b), float(np.log(c)))
    return {"a": best[2], "b": best[1], "c_zero": c_zero}


def b4_features(f):
    """12 features of an uncovered pixel set (dict of 1-D arrays)."""
    lr = np.log(f["r7"])
    return np.stack([np.ones_like(lr), lr, f["near"].astype(np.float64), f["alpha"], f["cosine"],
                     np.clip(np.log(f["ray"]) - lr, -2, 2), f["dist"], f["zero"].astype(np.float64), f["cov_share"],
                     f["near_share"], f["row"], f["col"]], axis=1)


def fit_b4(unc, device):
    phi = torch.tensor(b4_features(unc), dtype=torch.float32, device=device)
    lr = torch.tensor(np.log(unc["r7"]), dtype=torch.float32, device=device)
    g = torch.tensor(unc["g"], dtype=torch.float32, device=device)
    w = torch.tensor(unc["w"] / unc["w"].sum(), dtype=torch.float32, device=device)
    theta = torch.zeros(phi.shape[1], device=device, requires_grad=True)
    opt = torch.optim.Adam([theta], lr=0.01)
    for _ in range(600):
        pred = torch.exp(lr + (phi * theta).sum(1).clamp(-3, 3))
        loss = (w * (pred - g).abs() / g).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return {"theta": theta.detach().cpu().numpy().tolist()}


def b5_bins(f, edges):
    a_bin = np.digitize(f["alpha"], [0.05, 0.1, 0.2, 0.3, 0.4])
    d_bin = np.digitize(np.abs(np.log(f["ray"]) - np.log(f["r7"])), edges["dis"])
    r_bin = np.digitize(np.log(f["r7"]), edges["lr"])
    return ((((f["near"].astype(int) * 2 + f["zero"].astype(int)) * 6 + a_bin) * 4 + d_bin) * 3 + r_bin)


def fit_b5(unc):
    edges = {"dis": np.quantile(np.abs(np.log(unc["ray"]) - np.log(unc["r7"])), [0.25, 0.5, 0.75]).tolist(),
             "lr": np.quantile(np.log(unc["r7"]), [1 / 3, 2 / 3]).tolist()}
    idx = b5_bins(unc, edges)
    lr, lray, g, w = np.log(unc["r7"]), np.log(unc["ray"]), unc["g"], unc["w"]
    params = np.zeros((288, 2))
    for b in np.unique(idx):
        m = idx == b
        if m.sum() < 500:
            continue
        best = None
        for wb in WEIGHT_GRID:
            base = (1 - wb) * lr[m] + wb * lray[m]
            for c in OFFSET_GRID:
                loss = float((w[m] * np.abs(np.exp(base + c) - g[m]) / g[m]).sum())
                if best is None or loss < best[0]:
                    best = (loss, wb, c)
        params[b] = best[1:]
    return {"edges": edges, "params": params.tolist()}


def apply_b(name, p, f):
    """Uncovered-pixel log prediction of a fitted calibration control (f: dict of 1-D arrays)."""
    lr = np.log(f["r7"])
    z = f["zero"]
    if name == "B2A":
        return np.where(z, np.log(p["c_zero"]), lr)
    if name == "B2B":
        return np.where(z, np.log(p["c_zero"]), lr + np.log(np.where(f["near"], p["s_near"], p["s_fill"])))
    if name == "B3":
        return np.where(z, np.log(p["c_zero"]), p["a"] + p["b"] * lr)
    if name == "B4":
        return lr + np.clip(b4_features(f) @ np.asarray(p["theta"]), -3, 3)
    if name == "B5":
        q = np.asarray(p["params"])[b5_bins(f, p["edges"])]
        return (1 - q[:, 0]) * lr + q[:, 0] * np.log(f["ray"]) + q[:, 1]
    raise ValueError(name)


class PixelMLP(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        layers, cin = [], 8
        for h in hidden:
            layers += [nn.Conv2d(cin, h, 1), nn.GELU()]
            cin = h
        self.body = nn.Sequential(*layers)
        self.head = nn.Conv2d(cin, 1, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        return 0.7 * torch.tanh(self.head(self.body(x))[:, 0])


class SysNet(nn.Module):
    """V2's trunk on `cin` channels; the head starts at zero, so an untrained net reads R7."""
    def __init__(self, cin, r_max=1.0):
        super().__init__()
        layers = []
        for d in DILATIONS:
            layers += [nn.Conv2d(cin, WIDTH, 3, padding=d, dilation=d), nn.GroupNorm(8, WIDTH), nn.GELU()]
            cin = WIDTH
        self.trunk = nn.Sequential(*layers)
        self.head = nn.Conv2d(WIDTH, 1, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        self.r_max = r_max

    def forward(self, x):
        return self.r_max * torch.tanh(self.head(self.trunk(x))[:, 0])


def n_params(model):
    return int(sum(p.numel() for p in model.parameters()))
