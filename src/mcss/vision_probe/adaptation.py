"""A deliberately small observation-driven fast-weight diagnostic.

This is not the complete proposed model. A training-only PCA and RGB readout
expose a measurable feedback channel; two private residual maps test whether
that channel produces useful descriptor changes. RGB comes from the available
image, so this is pre-write reconstruction, not unseen-pixel prediction.
"""

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F

ACTIONS = ("OFF", "A", "B", "ALL")


@dataclass
class Readout:
    mean: Tensor
    basis: Tensor
    scale: Tensor
    decoder: Tensor
    bias: Tensor

    def project(self, features: Tensor) -> Tensor:
        return ((features - self.mean) @ self.basis) / self.scale

    def restore(self, features: Tensor, z: Tensor, updated: Tensor) -> Tensor:
        return features + ((updated - z) * self.scale) @ self.basis.T

    def predict(self, z: Tensor) -> Tensor:
        return z @ self.decoder + self.bias


@dataclass
class FastState:
    a: Tensor
    b: Tensor

    @classmethod
    def zero(cls, width: int) -> "FastState":
        return cls(torch.zeros(width, width), torch.zeros(width, width))

    def fork(self) -> "FastState":
        return FastState(self.a.clone(), self.b.clone())


def fit_readout(features: Tensor, rgb: Tensor, width: int = 64, ridge: float = 0.1) -> Readout:
    """Fit exclusively on training identities; no geometry or validation labels."""
    x = features.reshape(-1, features.shape[-1]).double()
    y = rgb.reshape(-1, 3).double()
    if len(x) != len(y) or len(x) < width + 1:
        raise ValueError("Need aligned training RGB and sufficient feature samples")
    mean = x.mean(0)
    covariance = (x - mean).T @ (x - mean) / (len(x) - 1)
    values, vectors = torch.linalg.eigh(covariance)
    basis = vectors[:, -width:].flip(1)
    scale = values[-width:].flip(0).clamp_min(1e-5).sqrt()
    z = (x - mean) @ basis / scale
    bias = y.mean(0)
    decoder = torch.linalg.solve(
        z.T @ z / len(z) + ridge * torch.eye(width, dtype=z.dtype),
        z.T @ (y - bias) / len(z),
    )
    return Readout(*(t.float() for t in (mean, basis, scale, decoder, bias)))


def neighborhood(z: Tensor) -> Tensor:
    """Replicated-border 3x3 mean; no geometry labels or external observations."""
    x = z.permute(2, 0, 1).unsqueeze(0)
    return F.avg_pool2d(F.pad(x, (1, 1, 1, 1), mode="replicate"), 3, 1)[0].permute(1, 2, 0)


def materialize(z: Tensor, state: FastState) -> Tensor:
    first = z + z @ state.a.T
    return first + neighborhood(first) @ state.b.T


def masks(height: int, width: int, step: int) -> tuple[Tensor, Tensor]:
    """Three fixed support partitions and one disjoint observed RGB selection set."""
    yy, xx = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
    partition = (yy + 2 * xx) % 4
    return partition == step % 3, partition == 3


def _clip(matrix: Tensor, limit: float) -> Tensor:
    return matrix * min(1.0, limit / max(float(matrix.norm()), 1e-12))


def proposals(
    z: Tensor,
    rgb: Tensor,
    readout: Readout,
    state: FastState,
    support: Tensor,
    eta: float,
    max_norm: float = 0.15,
) -> dict[str, FastState]:
    """All actions originate from the same old state; no inner backward pass.

    Direct normalized outer products project the observed RGB residual through
    the frozen decoder. B uses neighborhood activations after the old A map.
    This is a proxy write rule, not an optimal update or learned meta-optimizer.
    """
    if not support.any():
        raise ValueError("Empty update support")
    first = z + z @ state.a.T
    pred = readout.predict(materialize(z, state))
    residual = (rgb - pred)[support]
    v = residual @ readout.decoder.T / readout.decoder.square().sum().clamp_min(1e-6)
    increments = []
    for activations in (z, neighborhood(first)):
        u = activations[support]
        norm = u.square().sum(-1, keepdim=True).clamp_min(1e-6)
        delta = eta * (v.T @ (u / norm)) / len(u)
        increments.append(_clip(delta, max_norm))
    da, db = increments
    return {
        "OFF": state.fork(),
        "A": FastState(state.a + da, state.b.clone()),
        "B": FastState(state.a.clone(), state.b + db),
        "ALL": FastState(state.a + da, state.b + db),
    }


def probe_mse(z: Tensor, rgb: Tensor, readout: Readout, state: FastState, mask: Tensor) -> float:
    if not mask.any():
        raise ValueError("Empty observed selection set")
    return float((readout.predict(materialize(z, state))[mask] - rgb[mask]).square().mean())


def residual_selector(
    z: Tensor, rgb: Tensor, readout: Readout, candidates: dict[str, FastState], probe: Tensor
) -> str:
    """Select on disjoint observed pixels, never correspondence truth.

    Selection costs four candidate readouts. The probe is selection data and
    must never be reported as independent validation of the selected action.
    """
    scores = {a: probe_mse(z, rgb, readout, s, probe) for a, s in candidates.items()}
    return min(ACTIONS, key=lambda a: scores[a])
