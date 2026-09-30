"""V9: training-time view count for the Core A RGB-D recipe (fixed 3 vs variable 3-7 views).

Every variant is the V8 carrier family at one grid (fixed by the pre-V8 plan) with the V7 C1
bounds rule, the frozen V5 C1 loss, TRAIN72 and 6000 steps from the same per-seed draw; only
the number of views in each training context differs. A context is the role's 3 frozen
context frames plus the first m of the scene's 4 evenly spaced free frames (the Core B stream
rule, frame ids only), in arrival order: m = 0 (C0), m ~ U{0..4} from a separate seeded
generator (C1), or m = 4 (C2). Bounds always come from the role's 3 context frames.
"""

from __future__ import annotations

import torch

from mcss.dynamic.config import CarrierConfig
from mcss.dynamic.types import OnlineObservation
from mcss.mechanism_pilot import rgbd_evidence_carrier as v5
from mcss.mechanism_pilot import rgbd_resolution_carrier as v8

EXPERIMENT = "EXP-3D-RGBD-VIEWCOUNT-V9"
GRID = 32
EXTRA_MAX = 4
EXTRA_SEED_OFFSET = 7919
VARIANT_SPECS = {
    "C0": {"label": "FIXED3", "extra": "none", "model": "C1", "train": "TRAIN72"},
    "C1": {"label": "VARIABLE3TO7", "extra": "uniform_0_4", "model": "C1", "train": "TRAIN72"},
    "C2": {"label": "FIXED7", "extra": "all_4", "model": "C1", "train": "TRAIN72"},
}
VARIANTS = tuple(VARIANT_SPECS)
PRIMARY_PAIR = ("C0", "C1")
build_state = v8.build_state
parameter_hash = v8.parameter_hash
save_checkpoint = v8.save_checkpoint
load_checkpoint = v8.load_checkpoint
train_records = v8.train_records

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
    "make_carrier",
    "parameter_hash",
    "save_checkpoint",
    "scene_data",
    "train_records",
    "training_loss",
    "with_extra_views",
]


def carrier_config(variant=None):
    del variant
    return CarrierConfig(grid_size=(GRID, GRID, GRID), token_count=GRID**3)


def make_carrier(variant, seed, device, near=None, far=None):
    """Identical per-seed draw for every variant (the V8 family at the planned grid)."""
    del near, far
    if variant not in VARIANT_SPECS:
        raise KeyError(variant)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return v8.ResolutionRGBDCarrier(carrier_config()).to(device)


def extra_count(variant, generator):
    """Views added to one training step's contexts; C1 draws from its own generator."""
    kind = VARIANT_SPECS[variant]["extra"]
    if kind == "none":
        return 0
    if kind == "all_4":
        return EXTRA_MAX
    return int(torch.randint(0, EXTRA_MAX + 1, (1,), generator=generator))


def with_extra_views(context, stream, count):
    """The context followed by the first `count` stream views, relabeled in arrival order.

    count == 0 returns the context unchanged, so C0 runs the exact V7/V8 code path. The carrier
    reads frame ids only for alignment and order, never as values.
    """
    if count == 0:
        return context
    if not 0 < count <= len(stream):
        raise ValueError("Extra view count outside the available stream")
    observations = [*context, *list(stream)[:count]]
    depths = torch.cat((context.depths, stream.depths[:count]))
    relabeled = [
        OnlineObservation(o.scene_id, i, o.rgb, o.camera) for i, o in enumerate(observations)
    ]
    return v5.RGBDContext(relabeled, depths)


def scene_data(variant):
    del variant
    return v8.SCENE_DATA["CONTEXT_DEPTH"]


def training_loss(state, local_cameras, rgb, depth, indices, variant):
    """The frozen V2 C1 loss for the planned grid (surface band at the frozen 16-grid spacing)."""
    if variant not in VARIANT_SPECS:
        raise KeyError(variant)
    return v8.training_loss(state, local_cameras, rgb, depth, indices, "C0")
