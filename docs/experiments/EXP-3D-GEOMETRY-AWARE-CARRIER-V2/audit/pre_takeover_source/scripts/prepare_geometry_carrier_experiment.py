#!/usr/bin/env python3
"""Assemble and freeze a matched protocol after metadata/data preparation and tests."""

from __future__ import annotations

import argparse
import importlib.metadata
import shutil
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import torch

from mcss.dynamic.config import CarrierConfig
from mcss.mechanism_pilot.geometry_carrier_experiment import read, validate_split
from mcss.mechanism_pilot.small_training import sha, write_json


def prepare(root):
    root = Path(root).resolve()
    if (root / "preregistration.json").exists():
        raise FileExistsError("No re-freezing")
    manifest = read(root / "data/manifest.json")
    validate_split(manifest)
    counts = {
        split: sum(r["split"] == split for r in manifest["scenes"]) for split in ("TRAIN", "DEV")
    }
    if counts != {"TRAIN": 24, "DEV": 8}:
        raise PermissionError(f"Incomplete fixed cohort: {counts}")
    write_json(root / "scene_split.json", manifest)
    write_json(
        root / "frame_role_lock.json",
        {
            "roles": {r["scene_id"]: r["roles"] for r in manifest["scenes"]},
            "manifest_sha256": sha(root / "scene_split.json"),
            "selection_uses_model_outputs": False,
            "no_frame_or_bounds_search_after_training": True,
        },
    )
    old = Path("outputs/EXP-3D-CONTEXT-OBSERVABILITY-SUPERVISION-V1")
    shutil.copyfile(old / "train_depth_prior.json", root / "train_depth_prior.json")
    write_json(
        root / "architecture_contract.json",
        {
            "carrier": asdict(CarrierConfig(grid_size=(16, 16, 16))),
            "parameter_count": 5125,
            "variants_identical": True,
            "token_count": 128,
            "candidate_semantics": "unchanged evenly-spaced flattened voxel IDs; "
            "normalized voxel centers",
            "candidate_fraction_old8": 0.25,
            "candidate_fraction_new16": 0.03125,
            "bounds": "same frozen_gt_free_bounds(context cameras, old TRAIN3 prior)",
            "prior_sha256": sha(root / "train_depth_prior.json"),
            "samples": 64,
            "state": "query-independent; all DEV states sealed before query load; "
            ">=2 queries/state",
            "anchor_control": "anchor RGB only, same full-role context-camera-derived bounds",
            "single_view_support_limit": "unchanged lifting >=2 view support; "
            "anchor depends on bias/prior",
            "test_time_input": "RGB+CAMERA",
            "test_time_depth_used": False,
            "inference_architecture_identical": True,
            "new_backbone_or_fusion": False,
        },
    )
    config = {
        "variants": ["C0", "C1"],
        "primary_method": "C1",
        "seeds": [20260928, 20260929, 20260930],
        "steps": 3000,
        "checkpoint_steps": [500, 1000, 1500, 2000, 2500, 3000],
        "learning_rate": 0.001,
        "optimizer": "Adam",
        "weight_decay": 0,
        "lr_schedule": "constant",
        "gradient_clip": 1.0,
        "batch_scenes": 1,
        "batch_rays": 1024,
        "precision": "FP32",
        "num_threads": 4,
        "augmentation": "none",
        "image_size": [128, 160],
        "train_targets": "TRAIN primary_query2 RGB+ray-distance depth; never carrier input",
        "context_roles": "independent A/B states, equal loss weights",
        "sampling": "seeded scene permutation cycled; primary query slot cycles per scene visit; "
        "uniform with-replacement rays, identical indices for A/B and C0/C1",
        "initialize": "from scratch with same per-seed Torch RNG; no B-final warm start",
        "secondary_C2": "NOT_RUN_SECONDARY_OPTIONAL",
        "budget_rationale": "3 independent seeds, 125 visits/train-scene, 3000 fixed steps; "
        "synthetic shared-GPU timing approx .6s/step; no DEV tuning of budget",
        "checkpoint_selection": "independent per variant/seed using same DEV rule; never best seed",
        "seed_handling": "all3 equally weighted within scene; report each seed; no cherry-picking",
        "writer_updates": False,
        "dynamic_ttt_run": False,
    }
    write_json(root / "training_contract.json", config)
    write_json(
        root / "loss_contract.json",
        {
            "C0": "mean_A_B[RGB Charbonnier(eps=.001)+masked depth AbsRel(clamp=.001)]",
            "regularization": 0,
            "regularization_authority": "old small_training phase A/MeasurementLoss; "
            "do not substitute direct-state density L2/TV or RGB MSE",
            "C1": "C0 + .1 mean_A_B surface loss",
            "C2": "optional not run",
            "surface_authority_sha256": sha(old / "supervision_contract.json"),
            "surface": "-log(sum rendering weights where abs(t-d)<=tau +1e-8)",
            "tau": ".5*norm((bounds_hi-bounds_lo)/16)",
            "lambda_surface": 0.1,
            "surface_normalization": "valid depth, hit ray, nonempty band only; count excluded "
            "miss/empty-band rays, keep all in original losses/metrics",
            "free_space_in_primary": False,
            "post_surface_unknown_never_empty": True,
            "surface_gradients": "includes upstream transmittance before surface band",
            "ray_sampling": "matched Monte Carlo estimate of old full-image loss",
            "training_track": "RGB+D offline utility track; not RGB-only training",
        },
    )
    write_json(
        root / "checkpoint_selection_lock.json",
        {
            "metric": "DEV scene-macro depth AbsRel of direct A/B states",
            "tie_tolerance": 1e-8,
            "tie_break": "earliest step",
            "independent_per_variant_seed": True,
            "eligible": "finite metrics and bounded RGB; per-scene coverage>=.99*ray_hitfraction; "
            "hit-weighted opacity>=.05 when hits exist; RGB MSE<=1.25*constant0.5RGB MSE",
            "no_hit": "retained in all GT-valid primary metrics; NO_HIT is not model collapse",
            "no_eligible": "NO_ELIGIBLE_CHECKPOINT, no fallback or gate relaxation",
            "context_query_gap": "reported and monitored; no posthoc threshold",
            "allowed_partitions": ["DEV"],
            "fresh_qualification_access": False,
        },
    )
    write_json(
        root / "decision_rules.json",
        {
            "bootstrap": {
                "unit": "physical scene",
                "draws": 10000,
                "seed": 20260928,
                "ci": "percentile95",
            },
            "surface_supported": "gain>0;CIlo>0;>=.75 nonworse; all leave-one-scene-out gains>0; "
            "top1 positive gain share<=.5; "
            "C1wrong and shuffle damage CIlo>0; opacity gate; state audits PASS",
            "static_supported": "C1 anchor-full gain>0 CIlo>0 >=.75 nonworse; wrong CIlo>0; "
            "multiple queries/same state audit; opacity gate",
            "opacity_gate": "each scene both coverage>=.99 hitfraction, C1drop<=.01; common-mask "
            "and depth/opacity-normalized gains CIlo>0 with >=75% scene coverage",
            "empty_mask": "null conditional error, coverage disclosed; full primary scene stays",
            "historical_design_information": "Before training, old S0 raw reveals "
            "fixed DEV ai_009_001 "
            "both-role query rays miss bounds. Preserve cohort and bounds; geometric miss "
            "distinguished from learned opacity collapse, no post-V2-outcome threshold change.",
            "fresh_open_gate": "SURFACE_SUPPORTED and STATIC_SUPPORTED and provenance VERIFIED; "
            "no PARTIAL opening exception",
            "fresh_status": manifest["fresh_status"],
            "fresh_opened": False,
            "final_static_status_if_no_fresh": "NOT_ESTABLISHED",
            "dynamic_ttt_next_stage_allowed": False,
        },
    )
    write_json(
        root / "audit/environment.json",
        {
            "python": sys.version,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
            "gpu_shared": True,
            "packages": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
            "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        },
    )
    if read(root / "data/final_data_audit.json")["status"] != "PASS":
        raise PermissionError("Final source/camera audit required")
    if not (root / "data_provenance.json").is_file():
        raise PermissionError("Data independence provenance required")
    paths = [
        root / name
        for name in (
            "scene_split.json",
            "frame_role_lock.json",
            "training_contract.json",
            "architecture_contract.json",
            "loss_contract.json",
            "checkpoint_selection_lock.json",
            "decision_rules.json",
            "train_depth_prior.json",
            "data_provenance.json",
            "USER_PROTOCOL.md",
        )
    ]
    # All data bytes are locked, including train/dev labels, independently of model inputs.
    data_hashes = {}
    for r in manifest["scenes"]:
        if r["split"] not in ("TRAIN", "DEV"):
            continue
        for f in r["frames"]:
            for kind in ("rgb", "depth"):
                p = Path(f[kind])
                data_hashes[str(p)] = sha(p)
    write_json(root / "audit/data_hashes.json", data_hashes)
    paths.append(root / "audit/data_hashes.json")
    source_paths = sorted(Path("src/mcss").rglob("*.py"))
    source_paths += sorted(Path("scripts").glob("*geometry_carrier*.py"))
    write_json(
        root / "preregistration.json",
        {
            "status": "FROZEN_BEFORE_FORMAL_TRAINING",
            "primary_method": "C1",
            "input_sha256": {str(p.resolve()): sha(p) for p in paths},
            "source_sha256": {str(p.resolve()): sha(p) for p in source_paths},
            "data_sha256": data_hashes,
            "fresh_status": manifest["fresh_status"],
            "formal_training_started": False,
            "qualification_opened": False,
        },
    )
    print("FROZEN", counts)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    a = p.parse_args()
    prepare(a.root)
