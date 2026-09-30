"""V11: carrier width for completion, on the V9 recipe fixed before any V9 result.

The base recipe is the V9 variant chosen by PLAN_BEFORE_V9_RESULTS.md (V9 C1 VARIABLE3TO7 if the
V9 VIEWCOUNT_GAIN point estimate is > 0, else V9 C0 FIXED3); the training contract records its
extra-view rule as `base_extra`. C0 retrains that recipe at width 8 and must equal the V9 variant
bit-for-bit. C1 is the same recipe (TRAIN72, 32^3, CONTEXT_DEPTH bounds, the frozen V2 C1 loss,
6000 steps, the base extra-view rule) with a wider carrier: hidden_dim 32 and expansion_dim 64
instead of 8 and 16. Every module width follows CarrierConfig, so the V8 computation, state
construction and loss are reused; only the frozen width guard is generalized to the two widths.
"""

from __future__ import annotations

from dataclasses import asdict
from math import prod
from pathlib import Path

import torch

from mcss.dynamic.config import CarrierConfig
from mcss.dynamic.types import OnlineObservation, hash_value
from mcss.mechanism_pilot import rgbd_evidence_carrier as v5
from mcss.mechanism_pilot import rgbd_resolution_carrier as v8
from mcss.mechanism_pilot import rgbd_viewcount_carrier as v9
from mcss.mechanism_pilot.dense_evidence_carrier import _Materializer
from mcss.mechanism_pilot.small_training import sha
from mcss.training.grounded import build_grounded_state

EXPERIMENT = "EXP-3D-RGBD-COMPLETION-V11"
SCHEMA = "mcss.rgbd_completion_carrier.v11"
GRID = 32
EXTRA_MAX = v9.EXTRA_MAX
EXTRA_SEED_OFFSET = v9.EXTRA_SEED_OFFSET
BASE_EXTRA = {"C0": "none", "C1": "uniform_0_4"}  # V9 variant -> its extra-view rule
VARIANT_SPECS = {
    "C0": {"label": "WIDTH8", "hidden": 8, "expansion": 16, "model": "C1", "train": "TRAIN72"},
    "C1": {"label": "WIDTH32", "hidden": 32, "expansion": 64, "model": "C1", "train": "TRAIN72"},
}
VARIANTS = tuple(VARIANT_SPECS)
PRIMARY_PAIR = ("C0", "C1")
V8_LOSS_VARIANT = "C0"  # every V8 variant registers the frozen V2 C1 surface loss
with_extra_views = v9.with_extra_views
parameter_hash = v5.parameter_hash
train_records = v8.train_records

__all__ = [
    "BASE_EXTRA",
    "EXPERIMENT",
    "EXTRA_MAX",
    "EXTRA_SEED_OFFSET",
    "GRID",
    "PRIMARY_PAIR",
    "VARIANTS",
    "VARIANT_SPECS",
    "build_state",
    "carrier_config",
    "extra_count",
    "load_checkpoint",
    "make_carrier",
    "parameter_hash",
    "save_checkpoint",
    "scene_data",
    "train_records",
    "training_loss",
    "with_extra_views",
]


def _check_config(config):
    widths = {(8, s["hidden"], s["expansion"]) for s in VARIANT_SPECS.values()}
    if (
        tuple(config.grid_size) != (GRID, GRID, GRID)
        or (config.feature_dim, config.hidden_dim, config.expansion_dim) not in widths
        or config.token_count != prod(config.grid_size)
    ):
        raise ValueError("A registered V11 width at 32^3 (one candidate per voxel) is required")


def carrier_config(variant):
    spec = VARIANT_SPECS[variant]
    return CarrierConfig(
        grid_size=(GRID, GRID, GRID),
        token_count=GRID**3,
        hidden_dim=spec["hidden"],
        expansion_dim=spec["expansion"],
    )


def make_carrier(variant, seed, device, near=None, far=None):
    """The V8 resolution carrier at the variant's width; at width 8 this is the V9 draw."""
    del near, far
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return v8.ResolutionRGBDCarrier(carrier_config(variant)).to(device)


def extra_count(base_extra, generator):
    """Extra training views of one step under the base V9 rule; same draws as V9."""
    if base_extra == "none":
        return 0
    if base_extra == "uniform_0_4":
        return int(torch.randint(0, EXTRA_MAX + 1, (1,), generator=generator))
    raise ValueError(f"Unregistered base extra-view rule {base_extra}")


def build_state(carrier, context, bounds, episode_id):
    """The V8 build_state with the width guard generalized to the registered V11 widths."""
    if not isinstance(context, v5.RGBDContext):
        raise TypeError("V11 states are built from an RGBDContext")
    if not isinstance(carrier, v8.ResolutionRGBDCarrier):
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
    """The frozen V2 C1 loss at 32^3 (surface band at the frozen 16-grid spacing)."""
    if variant not in VARIANT_SPECS:
        raise KeyError(variant)
    return v8.training_loss(state, local_cameras, rgb, depth, indices, V8_LOSS_VARIANT)


def scene_data(variant):
    del variant
    return v8.SCENE_DATA["CONTEXT_DEPTH"]


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
        raise PermissionError("V11 static RGB-D completion carrier schema required")
    config = CarrierConfig(**payload["config"])
    _check_config(config)
    if not payload["depth_bypass"] or not any(k.startswith("depth_") for k in payload["carrier"]):
        raise PermissionError("V11 checkpoints always carry the measured-depth bypass")
    model = v8.ResolutionRGBDCarrier(config).to(device)
    model.load_state_dict(payload["carrier"], strict=True)
    return model, payload
