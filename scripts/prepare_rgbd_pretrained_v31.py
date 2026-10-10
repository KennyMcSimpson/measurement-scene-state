#!/usr/bin/env python3
"""Freeze EXP-3D-RGBD-PRETRAINED-FEATURES-V31: arm P, the V17 L recipe with frozen DINOv2 features.

P runs the V17 L recipe in its own V17-style root outputs/EXP-3D-RGBD-PRETRAINED-FEATURES-V31/P/
EXP-3D-RGBD-LONG-TRAIN-V17, made by the frozen V17 prepare (as V21 made G1 and V26 made GF). That
prepare locks every src file, so the V31 carrier module is in the root's lock; the V31 training and
EVAL scripts (the V17 scripts with the carrier module swapped) and the backbone weights and code
archive are added to it. The contract records GPU 4, TF32 off and the backbone. GPU training is not
bit-reproducible, so the control file the frozen V17 EVAL code requires says NOT_APPLICABLE_GPU.
The comparison arm is V21's G1 (same recipe on GPU 4, no pretrained features), checked unchanged
here. Then V31's freeze record (freeze.json) holds every hash, before any V31 training.
Run from the repository root under cv_host_run.py (records store zonghan_o paths).
"""

from __future__ import annotations

import datetime
import hashlib
import importlib.util
import json
import shutil
from pathlib import Path

from mcss.mechanism_pilot import rgbd_pretrained_carrier as v31
from mcss.mechanism_pilot.geometry_carrier_experiment import verify_lock

V31 = Path("outputs/EXP-3D-RGBD-PRETRAINED-FEATURES-V31")
PLAN = V31 / "PLAN_BEFORE_V31.md"
PLAN_SHA256 = "82ff870d3a234445ddee9aae178338fd22e87fda98bd5fb2202480b6eb95b85a"
V17 = Path("outputs/EXP-3D-RGBD-LONG-TRAIN-V17")
V17_PREREGISTRATION_SHA256 = "36b8aa6e4e527025d3746aa7af5ffc060fbf7683fc1a26ff50a94e3c744c1d1f"
V21 = Path("outputs/EXP-3D-RGBD-SEGMENT-DIVERSITY-V21")
V17_NAME = "EXP-3D-RGBD-LONG-TRAIN-V17"
ARM = "P"
GPU = "4"
MODULE = Path("src/mcss/mechanism_pilot/rgbd_pretrained_carrier.py")
TRAIN_SCRIPT = Path("scripts/train_rgbd_pretrained_v31.py")
EVAL_SCRIPT = Path("scripts/evaluate_rgbd_pretrained_eval_v31.py")
BACKBONE_FILES = (
    v31.MODELS / v31.BACKBONE["weights"],
    v31.MODELS / f"dinov2_{v31.BACKBONE['code_commit']}.zip",
)
SCRIPTS = (
    Path("scripts/prepare_rgbd_pretrained_v31.py"),
    TRAIN_SCRIPT,
    EVAL_SCRIPT,
    MODULE,
    Path("scripts/run_rgbd_pretrained_v31.sh"),
    Path("scripts/eval_rgbd_pretrained_v31.sh"),
    Path("scripts/analyze_rgbd_pretrained_v31.py"),
    Path("scripts/train_rgbd_long_train_carrier.py"),
    Path("scripts/evaluate_rgbd_long_train_eval.py"),
    Path("scripts/run_rgbd_geometric_baselines.py"),
    Path("scripts/run_rgbd_inpaint_baselines.py"),
    Path.home() / "cv_tools/fp32_run.py",
    Path.home() / "cv_tools/cv_gpu_run.sh",
    Path.home() / "cv_tools/cv_host_run.py",
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def control_arm():
    """V21's G1: finished, unchanged since V21's freeze, with all four EVAL cohorts scored."""
    freeze = read(V21 / "freeze.json")
    root = Path(freeze["arms"]["G1"]["root"])
    if sha(root / "preregistration.json") != freeze["arms"]["G1"]["preregistration_sha256"]:
        raise PermissionError("V21 G1 lock changed")
    if read(V21 / "results.json").get("INTERPRETATION_BRANCH") is None:
        raise PermissionError("V21 must be analyzed")
    for seed in (20260928, 20260929, 20260930):
        summary = read(root / "checkpoints" / f"L_{seed}" / "summary.json")
        if summary["status"] != "COMPLETE" or summary["steps"] != 24000:
            raise PermissionError(f"V21 G1 seed {seed} incomplete")
    for cohort in ("EVAL_V3", "EVAL_V4", "EVAL_FAR", "FRESH_V2"):
        if not (root / "raw/eval" / cohort / "prediction_seal.json").is_file():
            raise PermissionError(f"V21 G1 lacks the {cohort} seal")
    return root


def make_arm(prepare_v17):
    root = (V31 / ARM / V17_NAME).resolve()
    if root.exists():
        raise FileExistsError(root)
    root.mkdir(parents=True)
    for name in ("PLAN_BEFORE_V17_TRAINING.md", "PROTOCOL.md"):
        shutil.copyfile(V17 / name, root / name)
    prepare_v17(root)
    contract = read(root / "training_contract.json")
    contract.update(
        device="cuda",
        gpu=GPU,
        tf32="off (NVIDIA_TF32_OVERRIDE=0 and the fp32 launcher)",
        v31_arm=ARM,
        pretrained_features={
            "backbone": v31.BACKBONE,
            "projected_channels": v31.PROJECTED,
            "projection_seed_offset": v31.PROJECTION_SEED_OFFSET,
            "rule": "per view: frozen DINOv2 patch tokens -> learned 384->24 projection -> "
            "bilinear resize to the frame, after the 8 learned channels; the lifted mean and "
            "variance of the 24 channels enter the fuse pre-activation through a zero-initialized "
            "bias-free 48->64 map (the V17 L draw at initialization)",
            "module": str(MODULE),
            "training_script": str(TRAIN_SCRIPT),
            "evaluation_script": str(EVAL_SCRIPT),
        },
        execution="three trainings concurrently on GPU 4 through cv_gpu_run.sh; EVAL on CPU, AVX2",
    )
    write(root / "training_contract.json", contract)
    write(
        root / "audit/v11_prefix_control.json",
        {
            "status": "PASS",
            "meaning": "NOT_APPLICABLE_GPU",
            "note": "V31 GPU arm: GPU training is not bit-reproducible (HIT-REPRO-V1), so the V17 "
            "built-in control cannot apply; this file only lets the V17 EVAL code run",
        },
    )
    write(root / "tests.json", read(V31 / "tests.json"))
    lock = read(root / "preregistration.json")
    key = str((root / "training_contract.json").resolve())
    if key not in lock["input_sha256"]:
        raise KeyError(key)
    if lock["source_sha256"].get(str(MODULE.resolve())) != sha(MODULE):
        raise PermissionError("The V17 prepare did not lock the V31 carrier module")
    lock["input_sha256"][key] = sha(root / "training_contract.json")
    for path in (TRAIN_SCRIPT, EVAL_SCRIPT):
        lock["source_sha256"][str(path.resolve())] = sha(path)
    for path in BACKBONE_FILES:
        lock["input_sha256"][str(path)] = sha(path)
    lock["v31"] = {
        "arm": ARM,
        "plan_sha256": PLAN_SHA256,
        "rewritten": ["training_contract.json"],
        "added_sources": [str(TRAIN_SCRIPT), str(EVAL_SCRIPT)],
        "added_inputs": [str(path) for path in BACKBONE_FILES],
    }
    write(root / "preregistration.json", lock)
    return root


def main():
    if sha(PLAN) != PLAN_SHA256:
        raise PermissionError("The V31 plan changed")
    if sha(V17 / "preregistration.json") != V17_PREREGISTRATION_SHA256:
        raise PermissionError("V17 is not the frozen V17")
    if read(V31 / "tests.json")["status"] != "PASS":
        raise PermissionError("The full suite must PASS before the V31 freeze")
    if (V31 / "freeze.json").exists():
        raise FileExistsError("V31 is frozen once")
    v31.load_backbone("cpu")  # every backbone hash is checked here
    g1 = control_arm()
    v17_prepare = module("v17_prepare", "scripts/prepare_rgbd_long_train_experiment.py")
    root = make_arm(v17_prepare.prepare)
    manifest, config = verify_lock(root)
    print(ARM, "train records", len(config["train_sets"]["TRAIN72"]), "steps", config["steps"])
    write(
        V31 / "freeze.json",
        {
            "status": "FROZEN_BEFORE_V31_TRAINING",
            "frozen_utc": datetime.datetime.now(datetime.UTC).isoformat(),
            "plan_sha256": PLAN_SHA256,
            "backbone": v31.BACKBONE,
            "projected_channels": v31.PROJECTED,
            "arms": {
                ARM: {
                    "root": str(root),
                    "preregistration_sha256": sha(root / "preregistration.json"),
                    "training_contract_sha256": sha(root / "training_contract.json"),
                },
                "G1": {
                    "root": str(g1),
                    "preregistration_sha256": sha(g1 / "preregistration.json"),
                    "source": "V21 (finished)",
                },
            },
            "v17_preregistration_sha256": V17_PREREGISTRATION_SHA256,
            "scripts_sha256": {str(p): sha(p) for p in SCRIPTS},
            "tests_sha256": sha(V31 / "tests.json"),
        },
    )
    print("V31_FROZEN", str(root))


if __name__ == "__main__":
    main()
