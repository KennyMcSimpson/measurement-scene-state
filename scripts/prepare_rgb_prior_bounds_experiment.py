#!/usr/bin/env python3
"""Freeze the V10 single-factor GT-free bounds-prior protocol (RGB-only, TRAIN72, CPU only)."""

from __future__ import annotations

import argparse
import datetime
import importlib.metadata
import shutil
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from mcss.mechanism_pilot.geometry_carrier_experiment import read, validate_split
from mcss.mechanism_pilot.readout_reanalysis import (
    PRIMARY_REFERENCE,
    READOUTS,
    REFERENCES,
    reference_values,
)
from mcss.mechanism_pilot.rgb_prior_bounds_carrier import (
    EXPERIMENT,
    PRIMARY_PAIR,
    PRIOR_FILES,
    VARIANT_SPECS,
    carrier_config,
    make_carrier,
    parameter_hash,
)
from mcss.mechanism_pilot.small_training import sha, write_json

V2 = Path("outputs/EXP-3D-GEOMETRY-AWARE-CARRIER-V2")
V3 = Path("outputs/EXP-3D-DENSE-EVIDENCE-CARRIER-V3")
V4 = Path("outputs/EXP-3D-PLANE-SWEEP-CARRIER-V4")
V5 = Path("outputs/EXP-3D-RGBD-EVIDENCE-CARRIER-V5")
V6 = Path("outputs/EXP-3D-RGBD-TRAIN-SCALE-V6")
V7 = Path("outputs/EXP-3D-RGBD-DEPTH-BOUNDS-V7")
V8 = Path("outputs/EXP-3D-RGBD-RESOLUTION-V8")
V9 = Path("outputs/EXP-3D-RGBD-VIEWCOUNT-V9")
DIAGNOSTIC = Path("outputs/EXP-3D-RGB-PRIOR-BOUNDS-DIAG")
COUNTS = {"C0": 5125, "C1": 5125, "C2": 5125}
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


def train72_prior(manifest, source):
    """Pooled finite positive TRAIN72 metric ray distance over every prepared frame, q01/q99."""
    access, values = [], []
    for record in sorted(
        (r for r in manifest["scenes"] if r["split"] == "TRAIN"), key=lambda r: r["scene_id"]
    ):
        for frame in sorted(record["frames"], key=lambda f: f["frame_id"]):
            path = Path(frame["depth"])
            access.append(
                {
                    "scene_id": record["scene_id"],
                    "frame_id": frame["frame_id"],
                    "path": str(path),
                    "sha256": sha(path),
                    "purpose": "frozen_global_training_depth_prior",
                }
            )
            depth = np.load(path).reshape(-1)
            values.append(depth[np.isfinite(depth) & (depth > 0)])
    values = np.concatenate(values)
    near, far = (float(v) for v in np.quantile(values, [0.01, 0.99]))
    if len({a["scene_id"] for a in access}) != 72 or not 0 < near < far:
        raise PermissionError("The TRAIN72 prior must pool exactly the 72 TRAIN scenes")
    return {
        "access": access,
        "near_m": near,
        "far_m": far,
        "population": "pooled finite positive TRAIN72 metric ray distance, every prepared frame",
        "quantiles": [0.01, 0.99],
        "sample_count": int(values.size),
        "train_manifest_sha256": sha(source),
    }


def prepare(root):
    root = Path(root).resolve()
    if root.name != EXPERIMENT:
        raise PermissionError("Exact V10 experiment directory required")
    if (root / "preregistration.json").exists():
        raise FileExistsError("No re-freezing")
    if not (root / "PROTOCOL.md").is_file():
        raise PermissionError("Frozen V10 protocol text required before freezing")
    for earlier in (V2, V3, V4, V5, V6, V7, V8, V9):
        if read(earlier / "integrity.json")["status"] != "PASS":
            raise PermissionError(f"{earlier.name} must be finalized before V10 freezes")
    prior = read(V2 / "train_depth_prior.json")
    near, far = prior["near_m"], prior["far_m"]
    manifest = read(V6 / "scene_split.json")
    v5_scenes = read(V5 / "scene_split.json")["scenes"]
    train24 = sorted(r["scene_id"] for r in v5_scenes if r["split"] == "TRAIN")
    train72 = sorted(r["scene_id"] for r in manifest["scenes"] if r["split"] == "TRAIN")
    validate_split(manifest)
    counts = {s: sum(r["split"] == s for r in manifest["scenes"]) for s in ("TRAIN", "DEV")}
    if counts != {"TRAIN": 72, "DEV": 8} or len(train24) != 24:
        raise PermissionError(f"Unexpected V2 cohort: {counts}")
    write_json(root / "scene_split.json", manifest)
    # Geometry-free constants from TRAIN supervision targets only (primary-query GT).
    train_depths, train_files = [], {}
    for record in sorted(
        (r for r in manifest["scenes"] if r["scene_id"] in train24), key=lambda r: r["scene_id"]
    ):
        frames = {f["frame_id"]: f for f in record["frames"]}
        for fid in record["roles"]["primary_query"]:
            path = Path(frames[fid]["depth"])
            train_files[str(path)] = sha(path)
            train_depths.append(np.load(path).squeeze().astype(np.float32))
    reference = reference_values(train_depths)
    write_json(
        root / "frame_role_lock.json",
        {
            "roles": {r["scene_id"]: r["roles"] for r in manifest["scenes"]},
            "manifest_sha256": sha(root / "scene_split.json"),
            "source": "V2 locked cohort and frame roles reused unchanged; the frozen GT-free "
            "bounds rule is reused with the variant's TRAIN depth prior",
            "v2_scene_split_sha256": sha(V2 / "scene_split.json"),
            "dev_reused_from_v2": True,
            "selection_uses_model_outputs": False,
            "no_frame_or_bounds_search_after_training": True,
        },
    )
    if PRIOR_FILES["FROZEN_V2_PRIOR"] != "train_depth_prior.json":
        raise AssertionError("C0 must use the frozen V2 prior file")
    for name in ("train_depth_prior.json", "data_provenance.json"):
        shutil.copyfile(V2 / name, root / name)
    write_json(
        root / PRIOR_FILES["TRAIN72_PRIOR"], train72_prior(manifest, V6 / "scene_split.json")
    )
    hashes = {
        seed: {v: parameter_hash(make_carrier(v, seed, "cpu", near, far)) for v in VARIANT_SPECS}
        for seed in SEEDS
    }
    if any(len(set(per_seed.values())) != 1 for per_seed in hashes.values()):
        raise AssertionError("Variants must share identical shared parameters per seed")
    v6_hashes = read(V6 / "architecture_contract.json")["shared_parameter_hash_by_seed"]
    if {str(s): h["C0"] for s, h in hashes.items()} != v6_hashes:
        raise AssertionError("V10 must reuse the V5-V9 per-seed shared-parameter draw")
    counts_by_variant = {
        v: sum(p.numel() for p in make_carrier(v, SEEDS[0], "cpu", near, far).parameters())
        for v in VARIANT_SPECS
    }
    if counts_by_variant != COUNTS:
        raise AssertionError(f"Parameter counts changed: {counts_by_variant}")
    write_json(
        root / "architecture_contract.json",
        {
            "variants": {
                v: {**spec, "config": asdict(carrier_config(v))}
                for v, spec in VARIANT_SPECS.items()
            },
            "parameter_count": COUNTS,
            "shared_parameter_hash_by_seed": {str(s): h["C0"] for s, h in hashes.items()},
            "only_difference_C0_C1": "C1 differs from C0 only by the depth prior that sets the "
            "GT-free camera-frustum bounds of every state (all context pixel rays cast to the "
            "prior's near and far distances, hull x1.1, >=1 m): the frozen V2 prior (q01/q99 "
            "of 12 frames of 3 TRAIN scenes) for C0, the TRAIN72 prior (q01/q99 of every "
            "prepared TRAIN72 frame) for C1, in training and in evaluation. The 5125 shared "
            "parameters and their per-seed draw (identical to V5-V9), the V5 C0 model without "
            "a depth bypass, the frozen V2 C1 surface loss, TRAIN72, 6000 steps and the 32^3 "
            "grid are identical. The surface-loss band follows each state's bounds with the "
            "frozen 16-grid spacing.",
            "C2_difference_from_C1": "16^3 grid (4096 candidates); secondary, never a gate",
            "priors": {
                name: {
                    "file": file,
                    "near_m": read(root / file)["near_m"],
                    "far_m": read(root / file)["far_m"],
                    "sha256": sha(root / file),
                }
                for name, file in PRIOR_FILES.items()
            },
            "depth_bypass": None,
            "anchor_control": "anchor RGB + camera, the same full-role GT-free bounds as the "
            "variant's A/B state; a single view never reaches the >=2-view lifting support, "
            "so an anchor state carries learned priors only",
            "state": "query-independent; all DEV states sealed before query load; "
            ">=2 queries/state",
            "test_time_input": "RGB+CAMERA for every variant",
            "test_time_depth_used": False,
            "new_backbone_or_learned_matching": False,
        },
    )
    config = {
        "experiment": EXPERIMENT,
        "variants": list(VARIANT_SPECS),
        "variant_specs": VARIANT_SPECS,
        "primary_method": "C1",
        "primary_pair": list(PRIMARY_PAIR),
        "primary_contrast": "PRIOR_BOUNDS_GAIN = AbsRel(C0 frozen V2 prior) - AbsRel(C1 "
        "TRAIN72 prior), both 32^3",
        "secondary_contrast": "GRID16_GAP = AbsRel(C2 TRAIN72 prior at 16^3) - AbsRel(C1); "
        "descriptive only",
        "bounds_rule": "the frozen GT-free camera-frustum rule with the variant's TRAIN depth "
        "prior (context pixel rays cast to its q01/q99, hull x1.1, >=1 m)",
        "bounds_prior_basis": "exploratory TRAIN-only diagnostic "
        "outputs/EXP-3D-RGB-PRIOR-BOUNDS-DIAG: the frozen V2 prior leaves 29.7% of TRAIN72 "
        "primary-query GT points outside the box, the TRAIN72 q01/q99 prior 2.2%; no DEV, "
        "FRESH or protected scene was read",
        "bounds_prior_basis_sha256": {
            str(p): sha(p) for p in sorted(DIAGNOSTIC.glob("*")) if p.is_file()
        },
        "train_sets": {"TRAIN24": train24, "TRAIN72": train72},
        "device": "cpu",
        "seeds": SEEDS,
        "steps": 6000,
        "checkpoint_steps": [100, 200, 300, 500, 750, 1000, 1500, 2000, 3000, 4000, 5000, 6000],
        "checkpoint_rationale": "identical to V6-V9",
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
        "context_input": "context RGB + cameras only (RGB-only track, RGB_ONLY context "
        "loader); no depth of any frame enters construction",
        "context_roles": "independent A/B states, equal loss weights",
        "sampling": "seeded scene permutation cycled; primary query slot cycles per scene visit; "
        "uniform with-replacement rays, identical indices for A/B and all variants",
        "initialize": "from scratch; identical per-seed shared parameters for all variants, "
        "equal to the V5-V9 per-seed draw; no depth bypass",
        "parallel_workers": 3,
        "execution": "seeds sequential; the C0/C1/C2 runs of one seed concurrently on "
        "CPU cores 8-23 (identical contention); no GPU is used or queried (GPU paused by "
        "the user); min_free_gpu_mib is unused",
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
        "statistics": "frozen V2 estimator (geometry_carrier_statistics) with V7 Amendment 1 "
        "(only the C0/C1 ray-hit-fraction equality assertion removed: the bounds prior "
        "changes the box by design); its SURFACE_* fields denote the primary C0-C1 contrast, "
        "i.e. PRIOR_BOUNDS in V10",
        "dev_reuse": "the 8 DEV scenes were evaluated in V2-V9; V10 is a DEV mechanism "
        "diagnostic of the GT-free bounds prior on the RGB-only track, not a qualification "
        "round; FRESH-V1 and FRESH-V2 are not used",
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
            "C2": "identical to C0",
            "authority": "mcss.mechanism_pilot.rgb_prior_bounds_carrier.training_loss -> "
            "mcss.mechanism_pilot.rgbd_resolution_carrier.training_loss: the frozen "
            "V2 C1 loss (mcss.mechanism_pilot.geometry_carrier.training_loss) accepting a "
            "registered g^3 state; identical at 16^3; the surface band keeps the frozen "
            "16-grid spacing for every grid",
            "v2_loss_contract_sha256": sha(V2 / "loss_contract.json"),
            "lambda_surface": 0.1,
            "regularization": 0,
            "training_track": "RGB-only track: context depth is never carrier input; TRAIN "
            "query depth is supervision only",
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
            "PRIOR_BOUNDS_STATUS": "frozen V2 primary-contrast gate applied to C0-C1: gain>0; "
            "CIlo>0; "
            ">=.75 nonworse; all LOSO>0; positive top1 share<=.5; C1 wrong-scene and shuffle "
            "damage CIlo>0; C0/C1 common-mask opacity gate; state audits PASS",
            "SCENE_SPECIFICITY_STATUS": "C1 wrong-scene damage CIlo>0 and C1 spatial-shuffle "
            "damage CIlo>0 -> SUPPORTED, otherwise NOT_ESTABLISHED",
            "STATIC_DEV_STATUS": "frozen V2 static gate for C1",
            "GRID16_GAP": "AbsRel(C2)-AbsRel(C1), descriptive with CI, never a gate",
            "RAY_HIT_FRACTIONS": "per variant and scene, descriptive, never a gate",
            "GEOMETRY_FREE_REFERENCE": {
                "values": reference,
                "train_files_sha256": train_files,
                "references": list(REFERENCES),
                "primary_reference": PRIMARY_REFERENCE,
                "readouts": list(READOUTS),
                "bootstrap": {"unit": "scene", "draws": 10000, "seed": 20260928},
                "labels": "ABOVE (frozen V2 evidence rule) / BELOW (mirror) / "
                "NOT_DISTINGUISHABLE per carrier, readout and metric",
                "role": "descriptive; never a gate; computed after all states are sealed",
            },
            "TRAIN_FIT": "last-100-step TRAIN depth AbsRel per variant, descriptive",
            "interpretation": {
                "A": "PRIOR_BOUNDS_STATUS=SUPPORTED and C1 RAW AbsRel ABOVE "
                "REF_TRAIN_ABSREL_OPTIMAL on ALL DEV scenes (the prior owns its no-hit "
                "failures): the RGB-only information-limit judgment is withdrawn; next: an "
                "independent qualification cohort",
                "B": "PRIOR_BOUNDS_GAIN CIlo>0, or C1 ABOVE the reference on all scenes or on "
                "the no-hit-excluded scenes, not A",
                "C": "otherwise: the RGB-only failure is not explained by bounds truncation "
                "at this scale",
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
        PRIOR_FILES["TRAIN72_PRIOR"],
        "audit/data_hashes.json",
        "audit/previous_experiment_seal.json",
    )
    source_paths = sorted(Path("src/mcss").rglob("*.py"))
    source_paths += sorted(Path("scripts").glob("*rgb_prior_bounds*.py"))
    source_paths.append(Path("scripts/analyze_rgbd_bounds_carrier_amend1.py"))
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
