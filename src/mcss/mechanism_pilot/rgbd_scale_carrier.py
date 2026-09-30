"""V6: training-scale factor for the frozen V5 RGB-D carrier recipe (24 vs 72 TRAIN scenes).

Every model, loss, bypass, state builder and checkpoint format is the frozen V5 one; a V6
variant only selects a V5 recipe and a frozen TRAIN set. Identical per-seed initialization for
all variants; the data stream differs only between TRAIN sets, by design.
"""

from __future__ import annotations

from mcss.mechanism_pilot import rgbd_evidence_carrier as v5

EXPERIMENT = "EXP-3D-RGBD-TRAIN-SCALE-V6"
VARIANT_SPECS = {
    "C0": {"label": "DEPTH_TRAIN24", "model": "C1", "train": "TRAIN24"},
    "C1": {"label": "DEPTH_TRAIN72", "model": "C1", "train": "TRAIN72"},
    "C2": {"label": "RGB_TRAIN72", "model": "C0", "train": "TRAIN72"},
}
VARIANTS = tuple(VARIANT_SPECS)
PRIMARY_PAIR = ("C0", "C1")
RGBDSceneData = v5.RGBDSceneData
build_state = v5.build_state
parameter_hash = v5.parameter_hash
save_checkpoint = v5.save_checkpoint
load_checkpoint = v5.load_checkpoint

__all__ = [
    "EXPERIMENT",
    "PRIMARY_PAIR",
    "VARIANTS",
    "VARIANT_SPECS",
    "RGBDSceneData",
    "build_state",
    "load_checkpoint",
    "make_carrier",
    "parameter_hash",
    "save_checkpoint",
    "train_records",
    "training_loss",
]


def make_carrier(variant, seed, device, near=None, far=None):
    """The frozen V5 recipe of the variant, drawn with the shared per-seed initialization."""
    return v5.make_carrier(VARIANT_SPECS[variant]["model"], seed, device, near, far)


def training_loss(state, local_cameras, rgb, depth, indices, variant):
    return v5.training_loss(
        state, local_cameras, rgb, depth, indices, VARIANT_SPECS[variant]["model"]
    )


def train_records(manifest, config, variant):
    """Exactly the frozen TRAIN set of the variant, sorted by scene id."""
    names = set(config["train_sets"][VARIANT_SPECS[variant]["train"]])
    records = sorted(
        (r for r in manifest["scenes"] if r["split"] == "TRAIN" and r["scene_id"] in names),
        key=lambda r: r["scene_id"],
    )
    if len(records) != len(names) or len(records) < 24:
        raise PermissionError("The variant's frozen TRAIN set is incomplete")
    return records
