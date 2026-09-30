"""V10: GT-free bounds prior for the RGB-only recipe (frozen 3-scene prior vs a TRAIN72 prior).

RGB-only track: every state is built from context RGB and cameras only, through the RGB-only
context loader, so no depth of any frame enters construction. Every variant is the frozen V5 C0
recipe (the dense evidence carrier without the measured-depth bypass and the frozen V2 C1
surface loss) trained on TRAIN72 with the V6-V9 budget. The box of every state is the frozen
GT-free rule: all context pixel rays cast to the near and far distances of a depth prior fitted
on TRAIN depth, hull x1.1, at least 1 m per axis. C0 and C1 differ only in that prior: the
frozen V2 prior (q01/q99 of 12 frames of 3 TRAIN scenes) versus the TRAIN72 prior (q01/q99 of
every prepared TRAIN72 frame). C2 is C1 at 16^3. The V8 grid generalizations are reused
unchanged: construction accepts g^3 dense candidates and the surface-loss band keeps the frozen
16-grid spacing, so at 16^3 construction and loss are exactly the V5 C0 ones.
"""

from __future__ import annotations

from dataclasses import asdict
from math import prod
from pathlib import Path

import torch

from mcss.dynamic.carrier import DynamicSceneCarrier
from mcss.dynamic.config import CarrierConfig
from mcss.dynamic.types import OnlineObservation, hash_value
from mcss.mechanism_pilot import rgbd_evidence_carrier as v5
from mcss.mechanism_pilot import rgbd_resolution_carrier as v8
from mcss.mechanism_pilot.dense_evidence_carrier import _Materializer
from mcss.mechanism_pilot.geometry_carrier_experiment import SceneData, read
from mcss.mechanism_pilot.optimization_bounds_contracts import (
    FrozenTrainingPrior,
    context_camera_bundle,
    frozen_gt_free_bounds,
)
from mcss.mechanism_pilot.small_training import sha
from mcss.training.grounded import build_grounded_state

EXPERIMENT = "EXP-3D-RGB-PRIOR-BOUNDS-V10"
SCHEMA = "mcss.rgb_prior_bounds_carrier.v10"
GRIDS = (16, 32)
PRIOR_FILES = {
    "FROZEN_V2_PRIOR": "train_depth_prior.json",
    "TRAIN72_PRIOR": "train72_depth_prior.json",
}
VARIANT_SPECS = {
    "C0": {
        "label": "FROZEN_PRIOR_GRID32",
        "grid": 32,
        "prior": "FROZEN_V2_PRIOR",
        "model": "V5_C0_RGB",
        "train": "TRAIN72",
    },
    "C1": {
        "label": "TRAIN72_PRIOR_GRID32",
        "grid": 32,
        "prior": "TRAIN72_PRIOR",
        "model": "V5_C0_RGB",
        "train": "TRAIN72",
    },
    "C2": {
        "label": "TRAIN72_PRIOR_GRID16",
        "grid": 16,
        "prior": "TRAIN72_PRIOR",
        "model": "V5_C0_RGB",
        "train": "TRAIN72",
    },
}
VARIANTS = tuple(VARIANT_SPECS)
PRIMARY_PAIR = ("C0", "C1")
# Every V8 variant registers the frozen V2 C1 surface loss; C0 names it.
V8_LOSS_VARIANT = "C0"
parameter_hash = v5.parameter_hash

__all__ = [
    "EXPERIMENT",
    "GRIDS",
    "PRIMARY_PAIR",
    "PRIOR_FILES",
    "SCENE_DATA",
    "VARIANTS",
    "VARIANT_SPECS",
    "Train72PriorSceneData",
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


class Train72PriorSceneData(SceneData):
    """The RGB-only SceneData whose GT-free camera-frustum bounds use the TRAIN72 prior."""

    def __init__(self, record, manifest, root, device, access):
        super().__init__(record, manifest, root, device, access)
        path = Path(root) / PRIOR_FILES["TRAIN72_PRIOR"]
        prior = read(path)
        prior = FrozenTrainingPrior(prior["near_m"], prior["far_m"], sha(path))
        self.bounds = {
            role: frozen_gt_free_bounds(
                context_camera_bundle(record, role, manifest["image_size"]), prior
            ).to(device=device, dtype=torch.float32)
            for role in ("A", "B")
        }


SCENE_DATA = {"FROZEN_V2_PRIOR": SceneData, "TRAIN72_PRIOR": Train72PriorSceneData}


def _check_config(config):
    grid = tuple(config.grid_size)
    if (
        grid not in {(g, g, g) for g in GRIDS}
        or (config.feature_dim, config.hidden_dim, config.expansion_dim) != (8, 8, 16)
        or config.token_count != prod(grid)
    ):
        raise ValueError("Frozen dense carrier at 16^3 or 32^3 (one candidate per voxel)")


def carrier_config(variant):
    g = VARIANT_SPECS[variant]["grid"]
    return CarrierConfig(grid_size=(g, g, g), token_count=g**3)


def make_carrier(variant, seed, device, near=None, far=None):
    """The V5 C0 per-seed draw: the dense carrier with no depth bypass."""
    del near, far  # the bounds prior enters only through the scene loader
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return DynamicSceneCarrier(carrier_config(variant)).to(device)


def build_state(carrier, context, bounds, episode_id):
    """The frozen V3 dense build_state for a registered grid, from RGB and cameras only."""
    if isinstance(context, v5.RGBDContext):
        raise TypeError("RGB-only states never receive a depth-bearing context")
    if type(carrier) is not DynamicSceneCarrier:
        raise TypeError("The RGB-only dense carrier without a depth bypass is required")
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

    return build_grounded_state(ContextOnlyAdapter(), observations, episode_id)


def training_loss(state, local_cameras, rgb, depth, indices, variant):
    """The frozen V2 C1 surface loss (the V5 C0 recipe) through the V8 grid generalization."""
    if variant not in VARIANT_SPECS:
        raise ValueError("Unregistered V10 variant")
    return v8.training_loss(state, local_cameras, rgb, depth, indices, V8_LOSS_VARIANT)


def scene_data(variant):
    """The RGB-only scene loader implementing the variant's bounds prior."""
    return SCENE_DATA[VARIANT_SPECS[variant]["prior"]]


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
        "depth_bypass": False,
        "variant": variant,
        "seed": seed,
        "step": step,
        "lock_sha256": lock_sha256,
        "test_time_input": "RGB+CAMERA",
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
        raise PermissionError("V10 static RGB-only carrier schema required")
    config = CarrierConfig(**payload["config"])
    _check_config(config)
    if payload["depth_bypass"] or any(k.startswith("depth_") for k in payload["carrier"]):
        raise PermissionError("V10 checkpoints never carry a depth bypass")
    model = DynamicSceneCarrier(config).to(device)
    model.load_state_dict(payload["carrier"], strict=True)
    return model, payload
