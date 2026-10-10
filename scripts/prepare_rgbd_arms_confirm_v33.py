#!/usr/bin/env python3
"""Freeze EXP-3D-RGBD-ARMS-CONFIRM-V33 (CPU only, no training): the evaluation contract and lock.

Arms G1 (V21), U (V32) and P (V31) at step 24000 with seeds 20260928-30; cohorts EVAL_V5,
TRAIN_EXT and REPLICA (query role primary_query) with their manifest hashes, each disjoint from
every arm's TRAIN72; the NEAR radius, fill constant and abstention threshold of the V17 contract.
The lock (preregistration.json) holds the plan, the contract, every src file and evaluation or
analysis script, and each arm's lock and step-24000 checkpoints. Run from the repository root.
"""

from __future__ import annotations

import datetime
import hashlib
import json
from pathlib import Path

V33 = Path("outputs/EXP-3D-RGBD-ARMS-CONFIRM-V33")
PLAN = V33 / "PLAN_BEFORE_V33.md"
PLAN_SHA256 = "f4b5b84605642474702179829d11c1bd6c795c2afb23213052594f7fbf0ad0f8"
STEP = 24000
SEEDS = [20260928, 20260929, 20260930]
ARMS = {
    "G1": ("mcss.mechanism_pilot.rgbd_long_train_carrier", "EXP-3D-RGBD-SEGMENT-DIVERSITY-V21"),
    "P": ("mcss.mechanism_pilot.rgbd_pretrained_carrier", "EXP-3D-RGBD-PRETRAINED-FEATURES-V31"),
    "U": ("mcss.mechanism_pilot.rgbd_unet_carrier", "EXP-3D-RGBD-UNET-CONTEXT-V32"),
}
COHORTS = {
    "EVAL_V5": ("EXP-3D-RGBD-EVAL-V5-DATA/manifest_eval_v5.json", "EVAL_V5"),
    "TRAIN_EXT": ("EXP-3D-RGBD-TRAIN-EXT-DATA/manifest_train_ext.json", "TRAIN"),
    "REPLICA": ("EXP-3D-RGBD-REPLICA-DATA-V1/manifest_replica.json", "REPLICA"),
}
SCRIPTS = (
    "scripts/prepare_rgbd_arms_confirm_v33.py",
    "scripts/run_rgbd_arms_confirm_v33.sh",
    "scripts/evaluate_rgbd_carrier_arms.py",
    "scripts/analyze_rgbd_arms_confirm_v33.py",
    "scripts/evaluate_rgbd_long_train_eval.py",
    "scripts/evaluate_rgbd_completion_eval_v3.py",
    "scripts/run_rgbd_inpaint_baselines.py",
    "scripts/run_rgbd_geometric_baselines.py",
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main():
    if sha(PLAN) != PLAN_SHA256:
        raise PermissionError("The V33 plan changed")
    if (V33 / "preregistration.json").exists():
        raise FileExistsError("V33 is frozen once")
    v17 = read("outputs/EXP-3D-RGBD-LONG-TRAIN-V17/training_contract.json")["eval"]
    arms, checkpoints, trained = {}, {}, set()
    for name, (module, experiment) in ARMS.items():
        root = Path(read(Path("outputs") / experiment / "freeze.json")["arms"][name]["root"])
        trained |= set(read(root / "training_contract.json")["train_sets"]["TRAIN72"])
        arms[name] = {
            "module": module,
            "root": str(root),
            "step": STEP,
            "preregistration_sha256": sha(root / "preregistration.json"),
        }
        for seed in SEEDS:
            run = root / "checkpoints" / f"L_{seed}"
            if read(run / "summary.json")["status"] != "COMPLETE":
                raise PermissionError(f"{name} seed {seed} incomplete")
            item = next(c for c in read(run / "dev_curve.json") if int(c["step"]) == STEP)
            if sha(item["checkpoint"]) != item["checkpoint_sha256"]:
                raise PermissionError(f"{name} seed {seed} checkpoint changed")
            checkpoints[f"{name}_{seed}"] = {
                "path": item["checkpoint"],
                "sha256": item["checkpoint_sha256"],
            }
    cohorts = {}
    for name, (relative, split) in COHORTS.items():
        path = (Path("outputs") / relative).resolve()
        scenes = {r["scene_id"] for r in read(path)["scenes"]}
        if scenes & trained:
            raise PermissionError(f"{name} overlaps a training set")
        cohorts[name] = {
            "manifest": str(path),
            "manifest_sha256": sha(path),
            "split": split,
            "role": "primary_query",
            "scenes": len(scenes),
        }
    contract = {
        "arms": arms,
        "reference_arm": "G1",
        "seeds": SEEDS,
        "cohorts": cohorts,
        "near_px": v17["near_px"],
        "reprojection_fill_constant_m": v17["reprojection_fill_constant_m"],
        "abstain_opacity_threshold": v17["abstain_opacity_threshold"],
    }
    write(V33 / "evaluation_contract.json", contract)
    sources = sorted(Path("src/mcss").rglob("*.py")) + [Path(p) for p in SCRIPTS]
    write(
        V33 / "preregistration.json",
        {
            "status": "FROZEN_BEFORE_V33_EVALUATION",
            "frozen_utc": datetime.datetime.now(datetime.UTC).isoformat(),
            "plan_sha256": PLAN_SHA256,
            "evaluation_contract_sha256": sha(V33 / "evaluation_contract.json"),
            "run_condition": "V32 results.json P1_CONTEXT_GAIN_T24000.gain.mean > 0",
            "arm_checkpoints": checkpoints,
            "source_sha256": {str(p.resolve()): sha(p) for p in sources},
        },
    )
    print(
        "V33_FROZEN", len(checkpoints), "checkpoints", {k: v["scenes"] for k, v in cohorts.items()}
    )


if __name__ == "__main__":
    main()
