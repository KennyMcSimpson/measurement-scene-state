"""V3 single-factor carrier contrast: one 16^3 carrier, sparse versus dense evidence candidates.

Only CarrierConfig.token_count differs between C0 (128 candidates, the V2 recipe) and C1/C2
(4096 candidates, i.e. every voxel). Parameters, initialization, the >=2-view lifting rule,
fusion, refinement, heads, renderer, bounds and training labels are shared; C2 only drops the
frozen V2 surface term. State construction receives context RGB/cameras and frozen bounds only.
"""

from __future__ import annotations

import hashlib

import torch
from torch import nn

from mcss.dynamic.carrier import DynamicSceneCarrier
from mcss.dynamic.config import CarrierConfig
from mcss.dynamic.types import OnlineObservation
from mcss.mechanism_pilot import geometry_carrier
from mcss.training.grounded import build_grounded_state

EXPERIMENT = "EXP-3D-DENSE-EVIDENCE-CARRIER-V3"
GRID = (16, 16, 16)
VARIANT_SPECS = {
    "C0": {"label": "SPARSE128_SURFACE", "token_count": 128, "loss": "C1"},
    "C1": {"label": "DENSE4096_SURFACE", "token_count": 4096, "loss": "C1"},
    "C2": {"label": "DENSE4096_NO_SURFACE", "token_count": 4096, "loss": "C0"},
}
VARIANTS = tuple(VARIANT_SPECS)
PRIMARY_PAIR = ("C0", "C1")
TOKEN_COUNTS = (128, 4096)


def carrier_config(variant):
    return CarrierConfig(grid_size=GRID, token_count=VARIANT_SPECS[variant]["token_count"])


def make_carrier(variant, seed, device):
    """Same per-seed parameter draw for every variant; token_count only changes buffers."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return DynamicSceneCarrier(carrier_config(variant)).to(device)


def parameter_hash(carrier):
    """Hash trainable parameters only; candidate buffers legitimately differ by variant."""
    digest = hashlib.sha256()
    for name, value in carrier.named_parameters():
        digest.update(name.encode())
        digest.update(value.detach().to("cpu", torch.float32).contiguous().numpy().tobytes())
    return digest.hexdigest()


class _Materializer(nn.Module):
    def __init__(self, carrier):
        super().__init__()
        self.carrier = carrier

    def forward(self, cache, fast):
        return self.carrier.materialize(cache, fast)


def build_state(carrier, observations, bounds, episode_id):
    """V2 build_state with the frozen candidate counts {128, 4096}; no labels or queries.

    Geometry buffers are scoped to this call with functional_call and never become learned
    parameters. No writes or nonzero fast offsets are used.
    """
    if not isinstance(carrier, DynamicSceneCarrier):
        raise TypeError("Existing DynamicSceneCarrier architecture required")
    config = carrier.config
    if (
        tuple(config.grid_size),
        config.feature_dim,
        config.hidden_dim,
        config.expansion_dim,
    ) != (GRID, 8, 8, 16) or config.token_count not in TOKEN_COUNTS:
        raise ValueError("Frozen matched 16-grid carrier with 128 or 4096 candidates required")
    observations = tuple(observations)
    if not observations or any(type(o) is not OnlineObservation for o in observations):
        raise TypeError("Only RGB-camera OnlineObservation values may construct state")
    bounds = torch.as_tensor(bounds, device=carrier._bounds.device, dtype=carrier._bounds.dtype)
    if bounds.shape not in ((2, 3), (1, 2, 3)) or bounds.requires_grad:
        raise ValueError("Frozen nonlearned bounds must have shape [2,3] or [1,2,3]")
    bounds = bounds.reshape(1, 2, 3).detach().clone()
    if not torch.isfinite(bounds).all() or not (bounds[:, 1] > bounds[:, 0]).all():
        raise ValueError("Finite ordered bounds required")
    fractions = (carrier._candidate_normalized_xyz + 1) * 0.5
    points = bounds[:, 0] + fractions * (bounds[:, 1] - bounds[:, 0])
    wrapped = _Materializer(carrier)
    replacements = {"carrier._bounds": bounds, "carrier._candidate_points": points}

    class ContextOnlyAdapter:
        encode = carrier.encode
        initial_fast = carrier.initial_fast

        @staticmethod
        def materialize(cache, fast):
            return torch.func.functional_call(wrapped, replacements, (cache, fast), strict=False)

    return build_grounded_state(ContextOnlyAdapter(), observations, episode_id)


def training_loss(state, local_cameras, rgb, depth, indices, variant):
    """Frozen V2 loss family: C0/C1 use the V2 surface recipe, C2 the V2 no-surface loss."""
    return geometry_carrier.training_loss(
        state, local_cameras, rgb, depth, indices, VARIANT_SPECS[variant]["loss"]
    )
