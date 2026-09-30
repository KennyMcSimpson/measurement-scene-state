"""V7: bounds factor for the V6 C1 recipe (frozen GT-free bounds vs measured-context-depth bounds).

Every variant is the frozen V5 C1 model and loss trained on the frozen TRAIN72 set with the V6
budget; only the rule that sets each state's bounds differs, in training and evaluation alike.
"""

from __future__ import annotations

from mcss.mechanism_pilot import rgbd_evidence_carrier as v5
from mcss.mechanism_pilot.rgbd_depth_bounds import (
    DepthBoundsSceneData,
    TrimmedDepthBoundsSceneData,
)

EXPERIMENT = "EXP-3D-RGBD-DEPTH-BOUNDS-V7"
VARIANT_SPECS = {
    "C0": {"label": "FROZEN_BOUNDS", "model": "C1", "train": "TRAIN72", "bounds": "FROZEN"},
    "C1": {"label": "DEPTH_BOUNDS", "model": "C1", "train": "TRAIN72", "bounds": "CONTEXT_DEPTH"},
    "C2": {
        "label": "DEPTH_BOUNDS_TRIM1",
        "model": "C1",
        "train": "TRAIN72",
        "bounds": "CONTEXT_DEPTH_TRIM1",
    },
}
VARIANTS = tuple(VARIANT_SPECS)
PRIMARY_PAIR = ("C0", "C1")
SCENE_DATA = {
    "FROZEN": v5.RGBDSceneData,
    "CONTEXT_DEPTH": DepthBoundsSceneData,
    "CONTEXT_DEPTH_TRIM1": TrimmedDepthBoundsSceneData,
}
build_state = v5.build_state
parameter_hash = v5.parameter_hash
save_checkpoint = v5.save_checkpoint
load_checkpoint = v5.load_checkpoint

__all__ = [
    "EXPERIMENT",
    "PRIMARY_PAIR",
    "SCENE_DATA",
    "VARIANTS",
    "VARIANT_SPECS",
    "build_state",
    "load_checkpoint",
    "make_carrier",
    "parameter_hash",
    "save_checkpoint",
    "scene_data",
    "train_records",
    "training_loss",
]


def scene_data(variant):
    """The scene loader implementing the variant's bounds rule."""
    return SCENE_DATA[VARIANT_SPECS[variant]["bounds"]]


def make_carrier(variant, seed, device, near=None, far=None):
    return v5.make_carrier(VARIANT_SPECS[variant]["model"], seed, device, near, far)


def training_loss(state, local_cameras, rgb, depth, indices, variant):
    return v5.training_loss(
        state, local_cameras, rgb, depth, indices, VARIANT_SPECS[variant]["model"]
    )


def train_records(manifest, config, variant):
    names = set(config["train_sets"][VARIANT_SPECS[variant]["train"]])
    records = sorted(
        (r for r in manifest["scenes"] if r["split"] == "TRAIN" and r["scene_id"] in names),
        key=lambda r: r["scene_id"],
    )
    if len(records) != len(names) or len(records) < 24:
        raise PermissionError("The variant's frozen TRAIN set is incomplete")
    return records
