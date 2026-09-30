"""V8: voxel-grid resolution factor for the V7 RGB-D recipe (16^3 vs 32^3; 24^3 secondary).

Every variant is the frozen V5 C1 model and loss trained on the frozen TRAIN72 set with the
V6/V7 budget and the bounds rule fixed by the V8 plan; only the voxel grid differs, and with it
the dense candidate count (one candidate per voxel). The 5150 shared parameters and their
per-seed draw do not depend on the grid, so C0 reproduces the V7 recipe exactly.

Three frozen 16-grid guards are generalized here, without touching the frozen modules:
the measured-depth likelihood scale uses the variant's own mean voxel edge (the V5 definition),
state construction accepts g^3 dense candidates, and the frozen V2 surface loss accepts a g^3
state while its surface band keeps the frozen 16-grid spacing, so the loss itself is unchanged.
"""

from __future__ import annotations

from dataclasses import asdict
from math import prod
from pathlib import Path

import torch
import torch.nn.functional as functional

from mcss.dynamic.carrier import _CarrierComputation
from mcss.dynamic.config import CarrierConfig
from mcss.dynamic.lifting import lift_cached_observations
from mcss.dynamic.types import OnlineObservation, WriteTrace, hash_value
from mcss.measurements import FixedMeasurementRenderer
from mcss.mechanism_pilot import rgbd_evidence_carrier as v5
from mcss.mechanism_pilot.dense_evidence_carrier import _Materializer
from mcss.mechanism_pilot.direct_capacity_contracts import ObservationBatch
from mcss.mechanism_pilot.direct_capacity_optimization import cache_rays, render_rays
from mcss.mechanism_pilot.geometric_supervision_optimization import (
    geometry_losses,
    ray_geometry,
    surface_tau,
)
from mcss.mechanism_pilot.rgbd_bounds_carrier import SCENE_DATA
from mcss.mechanism_pilot.small_training import sha
from mcss.training.grounded import build_grounded_state

EXPERIMENT = "EXP-3D-RGBD-RESOLUTION-V8"
SCHEMA = "mcss.rgbd_resolution_carrier.v8"
BOUNDS_RULE = "CONTEXT_DEPTH"
GRIDS = (16, 24, 32)
VARIANT_SPECS = {
    "C0": {"label": "GRID16", "grid": 16, "model": "C1", "train": "TRAIN72"},
    "C1": {"label": "GRID32", "grid": 32, "model": "C1", "train": "TRAIN72"},
    "C2": {"label": "GRID24", "grid": 24, "model": "C1", "train": "TRAIN72"},
}
VARIANTS = tuple(VARIANT_SPECS)
PRIMARY_PAIR = ("C0", "C1")
parameter_hash = v5.parameter_hash

__all__ = [
    "BOUNDS_RULE",
    "EXPERIMENT",
    "GRIDS",
    "PRIMARY_PAIR",
    "VARIANTS",
    "VARIANT_SPECS",
    "ResolutionRGBDCarrier",
    "build_state",
    "carrier_config",
    "load_checkpoint",
    "make_carrier",
    "parameter_hash",
    "save_checkpoint",
    "scene_data",
    "train_records",
    "training_loss",
]


class ResolutionRGBDCarrier(v5.RGBDEvidenceCarrier):
    """The V5 RGB-D evidence carrier whose depth scale follows its own voxel grid."""

    def _compute(self, cache, fast):
        self._validate_fast(cache, fast)
        if self._pending_depth is None:
            raise PermissionError("RGB-D carrier states are built only through build_state")
        frame_ids, depths = self._pending_depth
        if tuple(entry.observation.frame_id for entry in cache.entries) != frame_ids:
            raise ValueError("Measured depth maps must align with the cached context views")
        lifted = lift_cached_observations(
            cache,
            self._candidate_points,
            self._candidate_normalized_xyz,
            feature_dim=self.config.feature_dim,
        )
        statistics = lifted.observed_feature_statistics.to(dtype=self.fuse_up.weight.dtype)
        support_weights = lifted.support_weights.to(dtype=statistics.dtype)
        fuse_activation = functional.gelu(self.fuse_up(statistics))
        fused_hidden = functional.linear(
            fuse_activation,
            self.fuse_down.weight + fast.delta_fuse,
            self.fuse_down.bias,
        )
        fused_hidden = fused_hidden * support_weights.unsqueeze(-1)
        grid = torch.tensor(
            tuple(self.config.grid_size), device=self._bounds.device, dtype=self._bounds.dtype
        )
        edge = ((self._bounds[:, 1] - self._bounds[:, 0]) / grid).mean()
        measured = v5.depth_statistics(
            [entry.observation for entry in cache.entries],
            depths,
            self._candidate_points,
            self.depth_log_sigma,
            edge.to(device=self._candidate_points.device, dtype=torch.float32),
        ).to(dtype=statistics.dtype)
        fused_hidden = fused_hidden + functional.linear(measured, self.depth_weight)
        refined = self.refinement(self._scatter_candidates(fused_hidden))
        complete_input = self._gather_candidates(refined)
        complete_activation = functional.gelu(self.complete_up(complete_input))
        trace = WriteTrace(
            episode_id=cache.episode_id,
            cache_revision=cache.revision,
            fast_step=fast.step,
            fuse_activation=fuse_activation,
            complete_activation=complete_activation,
            observed_feature_statistics=statistics,
            support_weights=support_weights,
            candidate_ids=self._candidate_ids.clone(),
            branch_id=fast.branch_id,
            source_state_id=fast.state_id,
        )
        return _CarrierComputation(trace, refined)


def _check_config(config):
    grid = tuple(config.grid_size)
    if (
        grid not in {(g, g, g) for g in GRIDS}
        or (config.feature_dim, config.hidden_dim, config.expansion_dim) != (8, 8, 16)
        or config.token_count != prod(grid)
    ):
        raise ValueError("Frozen dense carrier at 16^3, 24^3 or 32^3 (one candidate per voxel)")


def carrier_config(variant):
    g = VARIANT_SPECS[variant]["grid"]
    return CarrierConfig(grid_size=(g, g, g), token_count=g**3)


def make_carrier(variant, seed, device, near=None, far=None):
    """Identical per-seed draw of every shared parameter; the depth bypass starts at zero."""
    del near, far
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return ResolutionRGBDCarrier(carrier_config(variant)).to(device)


def build_state(carrier, context, bounds, episode_id):
    """The frozen V3/V5 build_state for any registered grid; no labels or queries."""
    if not isinstance(context, v5.RGBDContext):
        raise TypeError("V8 states are built from an RGBDContext")
    if not isinstance(carrier, ResolutionRGBDCarrier):
        raise TypeError("V8 resolution carrier required")
    _check_config(carrier.config)
    observations = tuple(context)
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

    carrier._pending_depth = (
        context.frame_ids,
        context.depths.to(device=carrier._bounds.device, dtype=torch.float32),
    )
    try:
        return build_grounded_state(ContextOnlyAdapter(), observations, episode_id)
    finally:
        carrier._pending_depth = None


def training_loss(state, local_cameras, rgb, depth, indices, variant):
    """The frozen V2 C1 surface loss for a registered grid; the surface band stays 16-grid."""
    if VARIANT_SPECS[variant]["model"] != "C1":
        raise ValueError("Only the frozen V5 C1 loss is registered")
    if tuple(state.spatial_shape) not in {(g, g, g) for g in GRIDS}:
        raise ValueError("Only registered grid states allowed")
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
    free, surface, counts = geometry_losses(
        *ray_geometry(renderer, state, cache, indices), target, surface_tau(state.bounds)
    )
    loss = base + 0.1 * surface
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


def scene_data(variant):
    """Every V8 variant uses the one bounds rule fixed by the V8 plan."""
    del variant
    return SCENE_DATA[BOUNDS_RULE]


def train_records(manifest, config, variant):
    names = set(config["train_sets"][VARIANT_SPECS[variant]["train"]])
    records = sorted(
        (r for r in manifest["scenes"] if r["split"] == "TRAIN" and r["scene_id"] in names),
        key=lambda r: r["scene_id"],
    )
    if len(records) != len(names) or len(records) < 24:
        raise PermissionError("The variant's frozen TRAIN set is incomplete")
    return records


def save_checkpoint(path, carrier, *, variant, seed, step, lock_sha256):
    payload = {
        "schema": SCHEMA,
        "config": asdict(carrier.config),
        "carrier": {k: v.detach().cpu() for k, v in carrier.state_dict().items()},
        "depth_bypass": True,
        "variant": variant,
        "seed": seed,
        "step": step,
        "lock_sha256": lock_sha256,
        "test_time_input": "RGB+DEPTH+CAMERA",
        "writer_trained": False,
    }
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    torch.save(payload, path)
    restored, _ = load_checkpoint(path, "cpu")
    if hash_value(restored.state_dict()) != hash_value(payload["carrier"]):
        raise RuntimeError("Checkpoint restore mismatch")
    return sha(path)


def load_checkpoint(path, device):
    payload = torch.load(path, map_location=device, weights_only=True)
    if payload.get("schema") != SCHEMA or payload.get("writer_trained"):
        raise PermissionError("V8 static RGB-D resolution carrier schema required")
    config = CarrierConfig(**payload["config"])
    _check_config(config)
    if not payload["depth_bypass"] or not any(k.startswith("depth_") for k in payload["carrier"]):
        raise PermissionError("V8 checkpoints always carry the measured-depth bypass")
    model = ResolutionRGBDCarrier(config).to(device)
    model.load_state_dict(payload["carrier"], strict=True)
    return model, payload
