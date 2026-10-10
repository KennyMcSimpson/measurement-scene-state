#!/usr/bin/env python3
"""Freeze EXP-3D-RGBD-UNET-CONTEXT-V32: arm U, the V17 L recipe with a 3D U-Net context branch.

U runs the V17 L recipe in its own V17-style root outputs/EXP-3D-RGBD-UNET-CONTEXT-V32/U/
EXP-3D-RGBD-LONG-TRAIN-V17, made by the frozen V17 prepare (as V21 made G1, V26 GF and V31 P).
That prepare locks every src file, so the V32 carrier module is in the root's lock; the V32
training and EVAL scripts (the V17 scripts with the carrier module swapped) are added to it. The
contract records GPU 4, TF32 off and the branch. GPU training is not bit-reproducible, so the
control file the frozen V17 EVAL code requires says NOT_APPLICABLE_GPU. The comparison arm is
V21's G1 (same recipe on GPU 4, no branch), checked unchanged here. Then V32's freeze record
(freeze.json) holds every hash, before any V32 training. Plan amendment 1 moved the GPU smoke
into the GPU job (~/cv_work/v32/gpu_job.sh, registered here).
Run from the repository root under cv_host_run.py (records store zonghan_o paths).
"""

from __future__ import annotations

import datetime
import hashlib
import importlib.util
import json
import shutil
from pathlib import Path

from mcss.mechanism_pilot import rgbd_unet_carrier as v32
from mcss.mechanism_pilot.geometry_carrier_experiment import verify_lock

V32 = Path("outputs/EXP-3D-RGBD-UNET-CONTEXT-V32")
PLAN = V32 / "PLAN_BEFORE_V32.md"
PLAN_SHA256 = "4bbdacfafc8033b4f2eb9c40e956776e44978aca56f62970acd7463e56bfc2c5"
AMENDMENT = V32 / "PLAN_AMENDMENT_1.md"
AMENDMENT_SHA256 = "960bf302577e06a2d0b6652e5f032a9bbdae91a8ad6e5e5595ffbb5df25afe54"
V17 = Path("outputs/EXP-3D-RGBD-LONG-TRAIN-V17")
V17_PREREGISTRATION_SHA256 = "36b8aa6e4e527025d3746aa7af5ffc060fbf7683fc1a26ff50a94e3c744c1d1f"
V21 = Path("outputs/EXP-3D-RGBD-SEGMENT-DIVERSITY-V21")
V17_NAME = "EXP-3D-RGBD-LONG-TRAIN-V17"
ARM = "U"
GPU = "4"
MODULE = Path("src/mcss/mechanism_pilot/rgbd_unet_carrier.py")
TRAIN_SCRIPT = Path("scripts/train_rgbd_unet_v32.py")
EVAL_SCRIPT = Path("scripts/evaluate_rgbd_unet_eval_v32.py")
WORK = Path.home() / "cv_work/v32"
SCRIPTS = (
    Path("scripts/prepare_rgbd_unet_v32.py"),
    TRAIN_SCRIPT,
    EVAL_SCRIPT,
    MODULE,
    Path("scripts/run_rgbd_unet_v32.sh"),
    Path("scripts/eval_rgbd_unet_v32.sh"),
    Path("scripts/analyze_rgbd_unet_v32.py"),
    Path("scripts/train_rgbd_long_train_carrier.py"),
    Path("scripts/evaluate_rgbd_long_train_eval.py"),
    Path("scripts/run_rgbd_geometric_baselines.py"),
    Path("scripts/run_rgbd_inpaint_baselines.py"),
    WORK / "gpu_job.sh",
    WORK / "check_gpu_smoke.py",
    WORK / "smoke_v32.py",
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
    root = (V32 / ARM / V17_NAME).resolve()
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
        v32_arm=ARM,
        context_unet={
            "branch": v32.UNET,
            "parameters": 361856,
            "rule": "a residual whole-volume 3D U-Net next to the two-layer refinement, on the "
            "same scattered fused volume; its 1x1x1 output starts at zero (the V17 L draw at "
            "initialization); drawn from its own random stream (seed + 37)",
            "module": str(MODULE),
            "training_script": str(TRAIN_SCRIPT),
            "evaluation_script": str(EVAL_SCRIPT),
        },
        execution="GPU smoke then three trainings concurrently on GPU 4 through cv_gpu_run.sh "
        "(~/cv_work/v32/gpu_job.sh, plan amendment 1); EVAL on CPU, AVX2",
    )
    write(root / "training_contract.json", contract)
    write(
        root / "audit/v11_prefix_control.json",
        {
            "status": "PASS",
            "meaning": "NOT_APPLICABLE_GPU",
            "note": "V32 GPU arm: GPU training is not bit-reproducible (HIT-REPRO-V1), so the V17 "
            "built-in control cannot apply; this file only lets the V17 EVAL code run",
        },
    )
    write(root / "tests.json", read(V32 / "tests.json"))
    lock = read(root / "preregistration.json")
    key = str((root / "training_contract.json").resolve())
    if key not in lock["input_sha256"]:
        raise KeyError(key)
    if lock["source_sha256"].get(str(MODULE.resolve())) != sha(MODULE):
        raise PermissionError("The V17 prepare did not lock the V32 carrier module")
    lock["input_sha256"][key] = sha(root / "training_contract.json")
    for path in (TRAIN_SCRIPT, EVAL_SCRIPT):
        lock["source_sha256"][str(path.resolve())] = sha(path)
    lock["v32"] = {
        "arm": ARM,
        "plan_sha256": PLAN_SHA256,
        "amendment_1_sha256": AMENDMENT_SHA256,
        "rewritten": ["training_contract.json"],
        "added_sources": [str(TRAIN_SCRIPT), str(EVAL_SCRIPT)],
    }
    write(root / "preregistration.json", lock)
    return root


def main():
    if sha(PLAN) != PLAN_SHA256 or sha(AMENDMENT) != AMENDMENT_SHA256:
        raise PermissionError("The V32 plan or its amendment changed")
    if sha(V17 / "preregistration.json") != V17_PREREGISTRATION_SHA256:
        raise PermissionError("V17 is not the frozen V17")
    if read(V32 / "tests.json")["status"] != "PASS":
        raise PermissionError("The full suite must PASS before the V32 freeze")
    if (V32 / "freeze.json").exists():
        raise FileExistsError("V32 is frozen once")
    g1 = control_arm()
    v17_prepare = module("v17_prepare", "scripts/prepare_rgbd_long_train_experiment.py")
    root = make_arm(v17_prepare.prepare)
    manifest, config = verify_lock(root)
    print(ARM, "train records", len(config["train_sets"]["TRAIN72"]), "steps", config["steps"])
    write(
        V32 / "freeze.json",
        {
            "status": "FROZEN_BEFORE_V32_TRAINING",
            "frozen_utc": datetime.datetime.now(datetime.UTC).isoformat(),
            "plan_sha256": PLAN_SHA256,
            "amendment_1_sha256": AMENDMENT_SHA256,
            "context_unet": v32.UNET,
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
            "tests_sha256": sha(V32 / "tests.json"),
        },
    )
    print("V32_FROZEN", str(root))


if __name__ == "__main__":
    main()
