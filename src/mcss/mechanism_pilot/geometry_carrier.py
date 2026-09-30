"""Matched static carrier construction and offline surface supervision.

Carrier inputs are RGB observations and context-camera-derived bounds only. Loss
labels are accepted by a separate function after the shared state exists.
"""

from __future__ import annotations

import torch
from torch import nn

from mcss.dynamic.carrier import DynamicSceneCarrier
from mcss.dynamic.types import OnlineObservation
from mcss.measurements import FixedMeasurementRenderer
from mcss.mechanism_pilot.direct_capacity_contracts import ObservationBatch
from mcss.mechanism_pilot.direct_capacity_optimization import cache_rays, render_rays
from mcss.mechanism_pilot.geometric_supervision_optimization import (
    geometry_losses,
    ray_geometry,
    surface_tau,
)
from mcss.training.grounded import build_grounded_state


class _Materializer(nn.Module):
    def __init__(self, carrier):
        super().__init__()
        self.carrier = carrier

    def forward(self, cache, fast):
        return self.carrier.materialize(cache, fast)


def build_state(carrier, observations, bounds, episode_id):
    """Return (differentiable shared state, raw anchor pose); no labels/query arguments.

    The caller supplies only the frozen context-camera bounds rule. Geometry buffers
    are scoped to this call using functional_call, and do not enter learned parameters.
    No writes or nonzero fast offsets are used.
    """
    if not isinstance(carrier, DynamicSceneCarrier):
        raise TypeError("Existing DynamicSceneCarrier architecture required")
    config = carrier.config
    if (
        tuple(config.grid_size),
        config.token_count,
        config.feature_dim,
        config.hidden_dim,
        config.expansion_dim,
    ) != ((16, 16, 16), 128, 8, 8, 16):
        raise ValueError("Frozen matched 16-grid carrier architecture required")
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
    """Offline TRAIN labels only; caller enforces scene-role provenance.

    C0 retains the old carrier RGB Charbonnier (epsilon .001) plus masked depth
    AbsRel. Its explicit regularization was zero. C1 adds only .1 surface loss;
    optional C2 also adds .1 free-space. Index sampling is externally frozen.
    """
    if variant not in ("C0", "C1", "C2"):
        raise ValueError("Only preregistered C0/C1/C2 loss variants allowed")
    if tuple(state.spatial_shape) != (16, 16, 16):
        raise ValueError("Only 16-grid state allowed")
    if len(local_cameras.leading_shape) != 2 or local_cameras.leading_shape[0] != 1:
        raise ValueError("Single-scene cameras must have shape [1,V]")
    views = local_cameras.leading_shape[1]
    height, width = local_cameras.image_size
    if rgb.shape != (1, views, 3, height, width) or depth.shape != (1, views, 1, height, width):
        raise ValueError("Exact camera-aligned RGB/depth target shapes required")
    if not torch.isfinite(rgb).all() or (rgb < 0).any() or (rgb > 1).any():
        raise ValueError("Finite RGB targets in [0,1] required")
    if indices.ndim != 1 or indices.numel() == 0 or indices.dtype != torch.long:
        raise ValueError("Nonempty one-dimensional int64 sampled ray indices required")
    if indices.min() < 0 or indices.max() >= views * height * width:
        raise ValueError("Sampled ray index outside TRAIN labels")
    valid = torch.isfinite(depth) & (depth > 0)
    clean_depth = torch.where(valid, depth, torch.zeros_like(depth))
    batch = ObservationBatch(
        "offline_train",
        tuple(range(views)),
        rgb,
        local_cameras,
        clean_depth,
        "OFFLINE_TRAIN_LABELS_NOT_CARRIER_INPUT",
    )
    cache = cache_rays(batch, state.bounds)
    renderer = FixedMeasurementRenderer(n_samples=64, ray_chunk_size=2048).to(state.color.device)
    pred = render_rays(renderer, state, cache, indices)
    rgb_loss = torch.sqrt((pred["rgb"] - cache["rgb"][:, indices]).square() + 1e-6).mean()
    target = cache["depth"][:, indices]
    valid = target > 0
    relative = (pred["depth"] - target).abs() / target.abs().clamp_min(1e-3)
    depth_loss = (relative * valid).sum() / valid.sum().clamp_min(1)
    base = rgb_loss + depth_loss
    with torch.set_grad_enabled(torch.is_grad_enabled() and variant != "C0"):
        free, surface, counts = geometry_losses(
            *ray_geometry(renderer, state, cache, indices), target, surface_tau(state.bounds)
        )
    loss = base
    if variant in ("C1", "C2"):
        loss = loss + 0.1 * surface
    if variant == "C2":
        loss = loss + 0.1 * free
    terms = {
        "loss_rgb": rgb_loss,
        "loss_depth": depth_loss,
        "regularization": base.new_zeros(()),
        "base_loss": base,
        "loss_free": free,
        "loss_surface": surface,
        "loss": loss,
        "opacity": pred["visibility"].mean(),
        **counts,
    }
    return loss, terms
