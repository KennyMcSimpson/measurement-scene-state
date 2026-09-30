#!/usr/bin/env python3
"""Freeze the V3 single-factor sparse/dense evidence protocol on the V2-locked TRAIN/DEV data."""

from __future__ import annotations

import argparse
import datetime
import importlib.metadata
import shutil
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import torch

from mcss.mechanism_pilot.dense_evidence_carrier import (
    EXPERIMENT,
    PRIMARY_PAIR,
    VARIANT_SPECS,
    carrier_config,
    make_carrier,
    parameter_hash,
)
from mcss.mechanism_pilot.geometry_carrier_experiment import read, validate_split
from mcss.mechanism_pilot.small_training import sha, write_json

V2 = Path("outputs/EXP-3D-GEOMETRY-AWARE-CARRIER-V2")
SEEDS = [20260928, 20260929, 20260930]
THREADS = {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}


def seal_previous(root):
    """Hash every earlier experiment artifact so finalization can prove nothing changed."""
    files = {}
    for base in (Path("outputs"), Path("docs/experiments")):
        for path in sorted(base.rglob("*")):
            if path.is_file() and EXPERIMENT not in path.parts:
                files[str(path)] = {"sha256": sha(path), "bytes": path.stat().st_size}
    write_json(
        root / "audit/previous_experiment_seal.json",
        {"created_utc": datetime.datetime.now(datetime.UTC).isoformat(), "files": files},
    )
    return len(files)


def prepare(root):
    root = Path(root).resolve()
    if root.name != EXPERIMENT:
        raise PermissionError("Exact V3 experiment directory required")
    if (root / "preregistration.json").exists():
        raise FileExistsError("No re-freezing")
    if not (root / "PROTOCOL.md").is_file():
        raise PermissionError("Frozen V3 protocol text required before freezing")
    if read(V2 / "integrity.json")["status"] != "PASS":
        raise PermissionError("V2 must be finalized before V3 reuses its locked cohort")
    manifest = read(V2 / "scene_split.json")
    validate_split(manifest)
    counts = {s: sum(r["split"] == s for r in manifest["scenes"]) for s in ("TRAIN", "DEV")}
    if counts != {"TRAIN": 24, "DEV": 8}:
        raise PermissionError(f"Unexpected V2 cohort: {counts}")
    write_json(root / "scene_split.json", manifest)
    write_json(
        root / "frame_role_lock.json",
        {
            "roles": {r["scene_id"]: r["roles"] for r in manifest["scenes"]},
            "manifest_sha256": sha(root / "scene_split.json"),
            "source": "V2 locked cohort, frame roles and GT-free bounds rule reused unchanged",
            "v2_scene_split_sha256": sha(V2 / "scene_split.json"),
            "dev_reused_from_v2": True,
            "selection_uses_model_outputs": False,
            "no_frame_or_bounds_search_after_training": True,
        },
    )
    for name in ("train_depth_prior.json", "data_provenance.json"):
        shutil.copyfile(V2 / name, root / name)
    hashes = {
        seed: {v: parameter_hash(make_carrier(v, seed, "cpu")) for v in VARIANT_SPECS}
        for seed in SEEDS
    }
    if any(len(set(per_seed.values())) != 1 for per_seed in hashes.values()):
        raise AssertionError("Variants must share identical initial parameters per seed")
    counts_by_variant = {
        v: sum(p.numel() for p in make_carrier(v, SEEDS[0], "cpu").parameters())
        for v in VARIANT_SPECS
    }
    if set(counts_by_variant.values()) != {5125}:
        raise AssertionError(f"Parameter counts changed: {counts_by_variant}")
    write_json(
        root / "architecture_contract.json",
        {
            "variants": {
                v: {**spec, "config": asdict(carrier_config(v))}
                for v, spec in VARIANT_SPECS.items()
            },
            "parameter_count": 5125,
            "initial_parameter_hash_by_seed": {str(s): h["C0"] for s, h in hashes.items()},
            "only_difference_C0_C1": "CarrierConfig.token_count 128 -> 4096: the candidate voxel "
            "IDs become every voxel of the 16^3 grid. Parameters, initialization, the >=2-view "
            "lifting rule, fusion, refinement, heads, renderer and bounds are unchanged.",
            "C2_difference_from_C1": "frozen V2 surface term disabled (V2 C0 loss)",
            "candidate_fraction": {"C0": 128 / 4096, "C1": 1.0, "C2": 1.0},
            "anchor_control": "anchor RGB only, same full-role context-camera-derived bounds; a "
            "single view never reaches the >=2-view support, so anchor states are prior-only",
            "state": "query-independent; all DEV states sealed before query load; "
            ">=2 queries/state",
            "test_time_input": "RGB+CAMERA",
            "test_time_depth_used": False,
            "new_backbone_or_fusion": False,
        },
    )
    config = {
        "experiment": EXPERIMENT,
        "variants": list(VARIANT_SPECS),
        "variant_specs": VARIANT_SPECS,
        "primary_method": "C1",
        "primary_pair": list(PRIMARY_PAIR),
        "primary_contrast": "DENSITY_GAIN = AbsRel(C0 sparse128+surface) - "
        "AbsRel(C1 dense4096+surface)",
        "secondary_contrast": "SURFACE_AT_DENSE_GAIN = AbsRel(C2) - AbsRel(C1); descriptive only",
        "seeds": SEEDS,
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
        "num_threads": 1,
        "thread_environment": THREADS,
        "cpu_affinity": list(range(8, 24)),
        "augmentation": "none",
        "image_size": [128, 160],
        "train_targets": "TRAIN primary_query2 RGB+ray-distance depth; never carrier input",
        "context_roles": "independent A/B states, equal loss weights",
        "sampling": "seeded scene permutation cycled; primary query slot cycles per scene visit; "
        "uniform with-replacement rays, identical indices for A/B and all variants",
        "initialize": "from scratch; identical per-seed parameters for all variants "
        "(token_count only changes buffers)",
        "parallel_workers": 3,
        "execution": "seeds sequential; the C0/C1/C2 runs of one seed concurrently "
        "(identical contention); each (re)start waits until the shared GPU has "
        "min_free_gpu_mib free",
        "min_free_gpu_mib": 3072,
        "infrastructure_retries": 2,
        "infrastructure_failure_policy": "a signal-terminated run, or a run whose log shows a "
        "fixed CUDA resource-allocation signature, is moved to audit/failed_attempts with an "
        "incident record and rerun from scratch with identical variant/seed/config, at most "
        "infrastructure_retries times per run counted across runner invocations; archived "
        "checkpoints and DEV curves are never used; any other Python error is never retried; "
        "a failed C0/C1 run stops the experiment",
        "checkpoint_selection": "independent per variant/seed using the frozen V2 DEV rule; "
        "never best seed",
        "seed_handling": "all seeds equally weighted within scene; each seed reported",
        "statistics": "frozen V2 estimator (geometry_carrier_statistics); its SURFACE_* fields "
        "denote the primary C0-C1 contrast, i.e. DENSITY in V3",
        "dev_reuse": "the 8 DEV scenes were evaluated in V2; V3 is a DEV mechanism diagnostic, "
        "not a qualification round",
        "fresh_qualification_included": False,
        "writer_updates": False,
        "dynamic_ttt_run": False,
    }
    write_json(root / "training_contract.json", config)
    write_json(
        root / "loss_contract.json",
        {
            "C0": "V2 C1 loss: mean_A_B[RGB Charbonnier(eps=.001) + masked depth AbsRel] + "
            ".1 mean_A_B surface loss",
            "C1": "identical to C0",
            "C2": "V2 C0 loss: mean_A_B[RGB Charbonnier(eps=.001) + masked depth AbsRel]",
            "authority": "mcss.mechanism_pilot.geometry_carrier.training_loss (frozen in V2)",
            "v2_loss_contract_sha256": sha(V2 / "loss_contract.json"),
            "lambda_surface": 0.1,
            "regularization": 0,
            "training_track": "RGB+D offline utility track; not RGB-only training",
        },
    )
    shutil.copyfile(V2 / "checkpoint_selection_lock.json", root / "checkpoint_selection_lock.json")
    write_json(
        root / "decision_rules.json",
        {
            "bootstrap": {
                "unit": "physical scene",
                "draws": 10000,
                "seed": 20260928,
                "ci": "percentile95",
            },
            "DENSITY_STATUS": "frozen V2 primary-contrast gate applied to C0-C1: gain>0; CIlo>0; "
            ">=.75 nonworse; all LOSO>0; positive top1 share<=.5; C1 wrong-scene and shuffle "
            "damage CIlo>0; C0/C1 common-mask opacity gate; state audits PASS",
            "SCENE_SPECIFICITY_STATUS": "C1 wrong-scene damage CIlo>0 and C1 spatial-shuffle "
            "damage CIlo>0 -> SUPPORTED, otherwise NOT_ESTABLISHED",
            "STATIC_DEV_STATUS": "frozen V2 static gate for C1",
            "SURFACE_AT_DENSE_GAIN": "AbsRel(C2)-AbsRel(C1), descriptive with CI, never a gate",
            "TRAIN_FIT": "last-100-step TRAIN depth AbsRel per variant, descriptive",
            "interpretation": {
                "A": "DENSITY_STATUS=SUPPORTED and SCENE_SPECIFICITY_STATUS=SUPPORTED",
                "B": "DENSITY_GAIN CIlo>0 or SCENE_SPECIFICITY_STATUS=SUPPORTED, not A",
                "C": "otherwise",
            },
            "fresh_qualification_included": False,
            "final_static_status": "NOT_ESTABLISHED",
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
    sealed = seal_previous(root)
    data_hashes = {}
    for r in manifest["scenes"]:
        for f in r["frames"]:
            for kind in ("rgb", "depth"):
                data_hashes[str(Path(f[kind]))] = sha(Path(f[kind]))
    write_json(root / "audit/data_hashes.json", data_hashes)
    names = (
        "scene_split.json",
        "frame_role_lock.json",
        "training_contract.json",
        "architecture_contract.json",
        "loss_contract.json",
        "checkpoint_selection_lock.json",
        "decision_rules.json",
        "train_depth_prior.json",
        "data_provenance.json",
        "PROTOCOL.md",
        "audit/data_hashes.json",
        "audit/previous_experiment_seal.json",
    )
    source_paths = sorted(Path("src/mcss").rglob("*.py"))
    source_paths += sorted(Path("scripts").glob("*dense_evidence*.py"))
    write_json(
        root / "preregistration.json",
        {
            "status": "FROZEN_BEFORE_FORMAL_TRAINING",
            "experiment": EXPERIMENT,
            "primary_method": "C1",
            "input_sha256": {str((root / n).resolve()): sha(root / n) for n in names},
            "source_sha256": {str(p.resolve()): sha(p) for p in source_paths},
            "data_sha256": data_hashes,
            "fresh_status": "NOT_INCLUDED_DEV_MECHANISM_ROUND",
            "formal_training_started": False,
            "qualification_opened": False,
        },
    )
    print("FROZEN", counts, "previous_files_sealed", sealed)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    a = p.parse_args()
    prepare(a.root)
