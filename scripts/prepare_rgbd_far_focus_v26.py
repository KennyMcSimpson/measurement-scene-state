#!/usr/bin/env python3
"""Freeze EXP-3D-RGBD-FAR-FOCUS-V26: the GF arm of the V17 L recipe with far-focus ray sampling.

GF runs the V17 L recipe in its own V17-style root outputs/EXP-3D-RGBD-FAR-FOCUS-V26/GF/
EXP-3D-RGBD-LONG-TRAIN-V17, made by the frozen V17 prepare (as V21 made G1 and G5). The contract
records GPU 4, TF32 off and the sampling rule; the V26 training script (the V17 script with
per-role far-focus sampling) is registered in the root's own lock. GPU training is not
bit-reproducible, so the control file the frozen V17 EVAL code requires says NOT_APPLICABLE_GPU.
The comparison arm is V21's G1 (same recipe on GPU 4, uniform sampling), checked unchanged here.
Then V26's freeze record (freeze.json) holds every hash, before any V26 training.
Run from the repository root under cv_host_run.py (records store zonghan_o paths).
"""

from __future__ import annotations

import datetime
import hashlib
import importlib.util
import json
import shutil
from pathlib import Path

from mcss.mechanism_pilot.geometry_carrier_experiment import verify_lock

V26 = Path("outputs/EXP-3D-RGBD-FAR-FOCUS-V26")
PLAN = V26 / "PLAN_BEFORE_V26.md"
PLAN_SHA256 = "0cce245568fc132e96adb2bcf9ec466008ab18a2a175a028440c9b47cbd23045"
AMENDMENT = V26 / "PLAN_AMENDMENT_1.md"
AMENDMENT_SHA256 = "08e6e904d18ab1f459f2daab58fcb7a647f4e8cf1696bf373338dc2fd34f91c3"
V17 = Path("outputs/EXP-3D-RGBD-LONG-TRAIN-V17")
V17_PREREGISTRATION_SHA256 = "36b8aa6e4e527025d3746aa7af5ffc060fbf7683fc1a26ff50a94e3c744c1d1f"
V21 = Path("outputs/EXP-3D-RGBD-SEGMENT-DIVERSITY-V21")
V17_NAME = "EXP-3D-RGBD-LONG-TRAIN-V17"
ARM = "GF"
GPU = "4"
FAR_FRACTION = 0.75
TRAIN_SCRIPT = Path("scripts/train_rgbd_far_focus_v26.py")
SCRIPTS = (
    Path("scripts/prepare_rgbd_far_focus_v26.py"),
    TRAIN_SCRIPT,
    Path("scripts/run_rgbd_far_focus_v26.sh"),
    Path("scripts/eval_rgbd_far_focus_v26.sh"),
    Path("scripts/analyze_rgbd_far_focus_v26.py"),
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
    root = (V26 / ARM / V17_NAME).resolve()
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
        v26_arm=ARM,
        ray_sampling={
            "rule": "per role (plan amendment 1): min(FAR_FRACTION * batch_rays, FAR pixel count) "
            "rays drawn without replacement from the valid query pixels farther than 8 px from the "
            "reprojected hits of every view used to build that role's state; the rest uniformly "
            "over the image; all uniform without FAR pixels",
            "far_fraction": FAR_FRACTION,
            "training_script": str(TRAIN_SCRIPT),
        },
        execution="three trainings concurrently on GPU 4 through cv_gpu_run.sh; EVAL on CPU, AVX2",
    )
    write(root / "training_contract.json", contract)
    write(
        root / "audit/v11_prefix_control.json",
        {
            "status": "PASS",
            "meaning": "NOT_APPLICABLE_GPU",
            "note": "V26 GPU arm: GPU training is not bit-reproducible (HIT-REPRO-V1), so the V17 "
            "built-in control cannot apply; this file only lets the frozen V17 EVAL code run",
        },
    )
    write(root / "tests.json", read(V26 / "tests.json"))
    lock = read(root / "preregistration.json")
    key = str((root / "training_contract.json").resolve())
    if key not in lock["input_sha256"]:
        raise KeyError(key)
    lock["input_sha256"][key] = sha(root / "training_contract.json")
    lock["source_sha256"][str(TRAIN_SCRIPT.resolve())] = sha(TRAIN_SCRIPT)
    lock["v26"] = {
        "arm": ARM,
        "plan_sha256": PLAN_SHA256,
        "rewritten": ["training_contract.json"],
        "added_source": str(TRAIN_SCRIPT),
    }
    write(root / "preregistration.json", lock)
    return root


def main():
    if sha(PLAN) != PLAN_SHA256 or sha(AMENDMENT) != AMENDMENT_SHA256:
        raise PermissionError("The V26 plan or its amendment changed")
    if sha(V17 / "preregistration.json") != V17_PREREGISTRATION_SHA256:
        raise PermissionError("V17 is not the frozen V17")
    if read(V26 / "tests.json")["status"] != "PASS":
        raise PermissionError("The full suite must PASS before the V26 freeze")
    if (V26 / "freeze.json").exists():
        raise FileExistsError("V26 is frozen once")
    g1 = control_arm()
    v17_prepare = module("v17_prepare", "scripts/prepare_rgbd_long_train_experiment.py")
    root = make_arm(v17_prepare.prepare)
    manifest, config = verify_lock(root)
    print(ARM, "train records", len(config["train_sets"]["TRAIN72"]), "steps", config["steps"])
    write(
        V26 / "freeze.json",
        {
            "status": "FROZEN_BEFORE_V26_TRAINING",
            "frozen_utc": datetime.datetime.now(datetime.UTC).isoformat(),
            "plan_sha256": PLAN_SHA256,
            "amendment_1_sha256": AMENDMENT_SHA256,
            "far_fraction": FAR_FRACTION,
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
            "tests_sha256": sha(V26 / "tests.json"),
        },
    )
    print("V26_FROZEN", str(root))


if __name__ == "__main__":
    main()
