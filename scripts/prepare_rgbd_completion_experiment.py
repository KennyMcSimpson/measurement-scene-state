#!/usr/bin/env python3
"""Freeze the V11 carrier-width protocol (TRAIN72 training, EVAL-V3 primary, CPU only)."""

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
from mcss.mechanism_pilot.rgbd_completion_carrier import (
    BASE_EXTRA,
    EXPERIMENT,
    EXTRA_MAX,
    EXTRA_SEED_OFFSET,
    GRID,
    PRIMARY_PAIR,
    VARIANT_SPECS,
    carrier_config,
    make_carrier,
    parameter_hash,
)
from mcss.mechanism_pilot.rgbd_evidence_carrier import LOG_SIGMA0
from mcss.mechanism_pilot.rgbd_stream_write import STREAM_LENGTH, stream_frame_ids
from mcss.mechanism_pilot.small_training import sha, write_json

V2 = Path("outputs/EXP-3D-GEOMETRY-AWARE-CARRIER-V2")
V3 = Path("outputs/EXP-3D-DENSE-EVIDENCE-CARRIER-V3")
V4 = Path("outputs/EXP-3D-PLANE-SWEEP-CARRIER-V4")
V5 = Path("outputs/EXP-3D-RGBD-EVIDENCE-CARRIER-V5")
V6 = Path("outputs/EXP-3D-RGBD-TRAIN-SCALE-V6")
V7 = Path("outputs/EXP-3D-RGBD-DEPTH-BOUNDS-V7")
V8 = Path("outputs/EXP-3D-RGBD-RESOLUTION-V8")
V9 = Path("outputs/EXP-3D-RGBD-VIEWCOUNT-V9")
EVAL_V3 = Path("outputs/EXP-3D-RGBD-EVAL-V3-DATA")
PLAN = "PLAN_BEFORE_V9_RESULTS.md"
PLAN_SHA256 = "fbfba0f0afbc7366a2964337b1eeba8683e95e80a946df2ab1946fc4c41ab091"
BOUNDS_TEXT = (
    "CONTEXT_DEPTH: hull of the measured depth and cameras of the role's 3 context frames, "
    "anchor frame, frozen x1.1 padding, >=1 m"
)
COUNTS = {"C0": 5150, "C1": 64238}
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
        raise PermissionError("Exact V11 experiment directory required")
    if (root / "preregistration.json").exists():
        raise FileExistsError("No re-freezing")
    if not (root / "PROTOCOL.md").is_file():
        raise PermissionError("Frozen V11 protocol text required before freezing")
    for earlier in (V2, V3, V4, V5, V6, V7, V8, V9):
        if read(earlier / "integrity.json")["status"] != "PASS":
            raise PermissionError(f"{earlier.name} must be finalized before V11 freezes")
    if sha(root / PLAN) != PLAN_SHA256:
        raise PermissionError("The pre-V9 plan fixing the base recipe must be intact")
    gain = float(read(V9 / "static_results.json")["surface_gain"]["mean"])
    base_variant = "C1" if gain > 0 else "C0"
    base_extra = BASE_EXTRA[base_variant]
    if GRID != read(V9 / "training_contract.json")["grid"]:
        raise PermissionError("V11 keeps the V9 grid")
    eval_integrity = read(EVAL_V3 / "preparation_integrity.json")
    eval_manifest = EVAL_V3 / "manifest_eval_v3.json"
    if eval_integrity["status"] != "PASS" or eval_integrity["candidate_lock_sha256"] != sha(
        EVAL_V3 / "candidate_lock.json"
    ):
        raise PermissionError("EVAL-V3 preparation must PASS against its frozen lock")
    eval_scenes = read(eval_manifest)["scenes"]
    if len(eval_scenes) < 30 or any(r["split"] != "EVAL_V3" for r in eval_scenes):
        raise PermissionError("EVAL-V3 needs at least 30 verified EVAL_V3 scenes")
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
    extra = {
        r["scene_id"]: stream_frame_ids(r, "A") for r in manifest["scenes"] if r["split"] == "TRAIN"
    }
    if any(ids is None or len(ids) != EXTRA_MAX for ids in extra.values()):
        raise PermissionError("Every TRAIN scene needs its 4 evenly spaced free frames")
    write_json(
        root / "frame_role_lock.json",
        {
            "roles": {r["scene_id"]: r["roles"] for r in manifest["scenes"]},
            "train_extra_view_frames": {k: list(v) for k, v in extra.items()},
            "extra_view_rule": f"the scene's {STREAM_LENGTH} free frames (in no context/query "
            "role) evenly spaced by frame id (the Core B V1 stream rule), shared by both roles; "
            "the first m arrive after the role's 3 context frames, relabeled in arrival order",
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
        seed: {v: parameter_hash(make_carrier(v, seed, "cpu", near, far)) for v in VARIANT_SPECS}
        for seed in SEEDS
    }
    v6_hashes = read(V6 / "architecture_contract.json")["shared_parameter_hash_by_seed"]
    if {str(s): h["C0"] for s, h in hashes.items()} != v6_hashes:
        raise AssertionError("V11 C0 must reuse the V5-V9 per-seed parameter draw")
    if any(h["C0"] == h["C1"] for h in hashes.values()):
        raise AssertionError("The width-32 draw must differ from the width-8 draw")
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
            "parameter_hash_by_seed": {str(s): h for s, h in hashes.items()},
            "only_difference_C0_C1": "C1 differs from C0 only by the carrier width: hidden_dim "
            "32 and expansion_dim 64 instead of 8 and 16 (64238 instead of 5150 parameters). "
            f"The recipe is the V9 {base_variant} one, fixed before any V9 result: TRAIN72, "
            f"the {GRID}^3 grid, the bounds rule ({BOUNDS_TEXT}), the frozen V2 C1 loss, 6000 "
            f"steps and the extra-view rule {base_extra!r} drawn from its own generator (seed "
            f"+ {EXTRA_SEED_OFFSET}); scene order, query slots and ray indices are identical. "
            "C0 retrains the V9 variant and must reproduce it bit-for-bit. DEV evaluation and "
            "checkpoint selection use the standard 3-view contexts.",
            "depth_bypass": {
                "log_sigma_init": LOG_SIGMA0,
                "sampling": "nearest pixel of the full-resolution measured ray-distance map",
                "placement": "after the >=2-view support gate: one measured view suffices",
                "inputs": "context RGB, measured context depth, context cameras; no query",
            },
            "anchor_control": "anchor RGB + anchor depth, the same full-role bounds as the "
            "variant's A/B state; a single view never reaches the >=2-view lifting support, "
            "so an anchor state carries the single-view depth bypass and learned priors only",
            "state": "query-independent; all DEV states sealed before query load; "
            ">=2 queries/state",
            "test_time_input": "RGB+DEPTH+CAMERA for every variant",
            "test_time_depth_used": True,
            "new_backbone_or_learned_matching": False,
        },
    )
    config = {
        "experiment": EXPERIMENT,
        "variants": list(VARIANT_SPECS),
        "variant_specs": VARIANT_SPECS,
        "primary_method": "C1",
        "primary_pair": list(PRIMARY_PAIR),
        "primary_contrast": "EVAL-V3 COMPLETION_GAIN = FAR AbsRel(C0 width 8) - FAR AbsRel(C1 "
        "width 32) and HYBRID_GAIN = AbsRel(REPROJ_NN) - AbsRel(FILL8 of C1)",
        "secondary_contrast": "DEV WIDTH_GAIN_DEV = AbsRel(C0) - AbsRel(C1), frozen V2 "
        "estimator, descriptive",
        "grid": GRID,
        "base_variant": base_variant,
        "base_extra": base_extra,
        "base_basis": f"{PLAN} (sha256 {PLAN_SHA256}), written before any V9 result: the V9 "
        f"C1 recipe if the V9 VIEWCOUNT_GAIN point estimate is >0, else V9 C0; "
        f"VIEWCOUNT_GAIN={gain!r}",
        "bounds_rule": "CONTEXT_DEPTH",
        "extra_views": {
            "max": EXTRA_MAX,
            "rule": base_extra,
            "generator_seed": f"seed + {EXTRA_SEED_OFFSET}",
        },
        "eval_v3": {
            "manifest": str(eval_manifest.resolve()),
            "manifest_sha256": sha(eval_manifest),
            "scenes": len(eval_scenes),
            "reprojection_fill_constant_m": reference["REF_TRAIN_ABSREL_OPTIMAL"],
            "near_px": 8,
            "role": "primary mechanism cohort: scene- and asset-disjoint from every used "
            "scene, volume-disjoint from DEV and FRESH, may share volumes with TRAIN72; "
            "never a qualification cohort",
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
        "augmentation": f"extra context views by the base rule ({base_extra}); nothing else",
        "image_size": [128, 160],
        "train_targets": "TRAIN primary_query2 RGB+ray-distance depth; never carrier input",
        "context_input": "context RGB + measured context ray distance + cameras (RGB-D "
        "track); query depth never enters construction",
        "context_roles": "independent A/B states, equal loss weights",
        "sampling": "seeded scene permutation cycled; primary query slot cycles per scene visit; "
        "uniform with-replacement rays, identical indices for A/B and all variants; the "
        "extra-view count comes from its own generator so the ray stream is unchanged",
        "initialize": "from scratch; torch.manual_seed(seed) then the width's carrier; C0 "
        "equals the V5-V9 draw; the depth bypass starts at zero without consuming the RNG",
        "parallel_workers": 6,
        "execution": "all six runs (2 widths x 3 seeds) concurrently, one core each, on CPU "
        "cores 8-23; no GPU is used or queried (GPU paused by the user); min_free_gpu_mib "
        "is unused",
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
        "denote the secondary DEV C0-C1 contrast, i.e. WIDTH_GAIN_DEV in V11",
        "dev_reuse": "the 8 DEV scenes were evaluated in V2-V10 and Core B V1/V2; in V11 "
        "DEV only selects checkpoints and gives descriptive results; the primary cohort is "
        "EVAL-V3; FRESH-V1 and FRESH-V2 are not used",
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
            "authority": "mcss.mechanism_pilot.rgbd_completion_carrier.training_loss: the V8 "
            "generalized frozen V2 C1 loss (identical to geometry_carrier.training_loss at "
            "16^3; the surface band keeps the frozen 16-grid spacing for every grid)",
            "v2_loss_contract_sha256": sha(V2 / "loss_contract.json"),
            "lambda_surface": 0.1,
            "regularization": 0,
            "training_track": "RGB-D track: context depth is carrier input for every variant; "
            "TRAIN query depth is supervision only",
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
            "WIDTH_DEV_STATUS": "secondary, DEV: frozen V2 primary-contrast gate applied to "
            "C0-C1: gain>0; "
            "CIlo>0; "
            ">=.75 nonworse; all LOSO>0; positive top1 share<=.5; C1 wrong-scene and shuffle "
            "damage CIlo>0; C0/C1 common-mask opacity gate; state audits PASS",
            "SCENE_SPECIFICITY_STATUS": "C1 wrong-scene damage CIlo>0 and C1 spatial-shuffle "
            "damage CIlo>0 -> SUPPORTED, otherwise NOT_ESTABLISHED",
            "STATIC_DEV_STATUS": "frozen V2 static gate for C1",
            "EVAL_V3_PRIMARY": {
                "COMPLETION_STATUS": "frozen V2 gate checks (mean>0, CIlo>0, >=.75 "
                "nonworse, all LOSO>0, positive top1 share<=.5) on COMPLETION_GAIN = FAR "
                "AbsRel(C0) - FAR AbsRel(C1) over EVAL-V3 scenes",
                "HYBRID_STATUS": "the same checks on HYBRID_GAIN = AbsRel(REPROJ_NN) - "
                "AbsRel(FILL8 of C1)",
                "FAR": "valid pixels farther than 8 px from any context-depth reprojection "
                "hit (1-px z-buffer of the 3 context depth maps)",
                "FILL8": "REPROJ_NN within 8 px of a hit, the carrier elsewhere",
                "aggregation": "per scene: seeds equal, then the mean of role means of "
                "query values; 10000-draw scene bootstrap, seed 20260928",
            },
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
                "A": "COMPLETION_STATUS and HYBRID_STATUS SUPPORTED on EVAL-V3: width improves "
                "completion and measurement preservation with the wide carrier beats direct "
                "reprojection; next: an independent (volume-disjoint) confirmation",
                "B": "exactly one of the two EVAL-V3 statuses SUPPORTED",
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
        PLAN,
        "audit/data_hashes.json",
        "audit/previous_experiment_seal.json",
    )
    source_paths = sorted(Path("src/mcss").rglob("*.py"))
    source_paths += sorted(Path("scripts").glob("*rgbd_completion*.py"))
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
