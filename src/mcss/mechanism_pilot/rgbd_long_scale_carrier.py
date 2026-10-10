"""V20: four times longer training of the unchanged V15 C1 recipe (width 64, TRAIN127).

PLAN_BEFORE_V20_TRAINING.md fixed V20 before any V20 training. The single variant L is the V15
C1 recipe (the V11 C1 recipe at width 64, hidden 64 and expansion 128, on TRAIN72 plus
TRAIN-EXT, 127 scenes) trained for 24000 instead of 6000 steps. The learning rate is constant
and every random stream has its own seeded generator, so the first 6000 steps of each seed
must equal V15 C1 bit-for-bit (a built-in machinery control), and every step sees the data of
V18 L (V14 C1 and V15 C1 share one data stream per seed). The computation is V18's path
unchanged.
"""

from __future__ import annotations

from dataclasses import asdict
from math import prod
from pathlib import Path

import torch

from mcss.dynamic.config import CarrierConfig
from mcss.dynamic.types import OnlineObservation, hash_value
from mcss.mechanism_pilot import rgbd_completion_carrier as v11
from mcss.mechanism_pilot import rgbd_evidence_carrier as v5
from mcss.mechanism_pilot import rgbd_resolution_carrier as v8
from mcss.mechanism_pilot.dense_evidence_carrier import _Materializer
from mcss.mechanism_pilot.small_training import sha
from mcss.training.grounded import build_grounded_state

EXPERIMENT = "EXP-3D-RGBD-LONG-TRAIN-WIDTH-DATA-V20"
SCHEMA = "mcss.rgbd_long_scale_carrier.v20"
GRID = v11.GRID
EXTRA_MAX = v11.EXTRA_MAX
EXTRA_SEED_OFFSET = v11.EXTRA_SEED_OFFSET
VARIANT_SPECS = {
    "L": {
        "label": "W64_T127_24K",
        "hidden": 64,
        "expansion": 128,
        "model": "C1",
        "train": "TRAIN_EXT",
    },
}
VARIANTS = tuple(VARIANT_SPECS)
PRIMARY_PAIR = ("L",)
V8_LOSS_VARIANT = v11.V8_LOSS_VARIANT
extra_count = v11.extra_count
with_extra_views = v11.with_extra_views
parameter_hash = v5.parameter_hash
load_v11_checkpoint = v11.load_checkpoint

__all__ = [
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
    "load_v11_checkpoint",
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
        raise ValueError("The registered V20 width at 32^3 (one candidate per voxel) is required")


def train_records(manifest, config, variant):
    """Exactly the variant's own frozen TRAIN set (V20 table), sorted by scene id."""
    names = set(config["train_sets"][VARIANT_SPECS[variant]["train"]])
    records = sorted(
        (r for r in manifest["scenes"] if r["split"] == "TRAIN" and r["scene_id"] in names),
        key=lambda r: r["scene_id"],
    )
    if len(records) != len(names) or len(records) < 24:
        raise PermissionError("The variant's frozen TRAIN set is incomplete")
    return records


def carrier_config(variant):
    spec = VARIANT_SPECS[variant]
    return CarrierConfig(
        grid_size=(GRID, GRID, GRID),
        token_count=GRID**3,
        hidden_dim=spec["hidden"],
        expansion_dim=spec["expansion"],
    )


def make_carrier(variant, seed, device, near=None, far=None):
    """The V8 resolution carrier at width 64; for V20 L this is the V15 C1 (= V12 C1) draw."""
    del near, far
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return v8.ResolutionRGBDCarrier(carrier_config(variant)).to(device)


def build_state(carrier, context, bounds, episode_id):
    """The V8 build_state under the width guard of the registered V20 width."""
    if not isinstance(context, v5.RGBDContext):
        raise TypeError("V20 states are built from an RGBDContext")
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
        raise PermissionError("V20 static RGB-D long-training carrier schema required")
    config = CarrierConfig(**payload["config"])
    _check_config(config)
    if not payload["depth_bypass"] or not any(k.startswith("depth_") for k in payload["carrier"]):
        raise PermissionError("V20 checkpoints always carry the measured-depth bypass")
    model = v8.ResolutionRGBDCarrier(config).to(device)
    model.load_state_dict(payload["carrier"], strict=True)
    return model, payload
