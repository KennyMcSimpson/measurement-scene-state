#!/usr/bin/env python3
"""Freeze the V15 width-at-scale protocol (width 32 vs 64 on TRAIN127, EVAL-V3/V4 primary)."""

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
from mcss.mechanism_pilot.rgbd_evidence_carrier import LOG_SIGMA0
from mcss.mechanism_pilot.rgbd_stream_write import STREAM_LENGTH, stream_frame_ids
from mcss.mechanism_pilot.rgbd_wide_scale_carrier import (
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
from mcss.mechanism_pilot.small_training import sha, write_json

V2 = Path("outputs/EXP-3D-GEOMETRY-AWARE-CARRIER-V2")
V3 = Path("outputs/EXP-3D-DENSE-EVIDENCE-CARRIER-V3")
V4 = Path("outputs/EXP-3D-PLANE-SWEEP-CARRIER-V4")
V5 = Path("outputs/EXP-3D-RGBD-EVIDENCE-CARRIER-V5")
V6 = Path("outputs/EXP-3D-RGBD-TRAIN-SCALE-V6")
V7 = Path("outputs/EXP-3D-RGBD-DEPTH-BOUNDS-V7")
V8 = Path("outputs/EXP-3D-RGBD-RESOLUTION-V8")
V9 = Path("outputs/EXP-3D-RGBD-VIEWCOUNT-V9")
V11 = Path("outputs/EXP-3D-RGBD-COMPLETION-V11")
V12 = Path("outputs/EXP-3D-RGBD-WIDTH-V12")
V14 = Path("outputs/EXP-3D-RGBD-DATA-SCALE-V14")
TRAIN_EXT = Path("outputs/EXP-3D-RGBD-TRAIN-EXT-DATA")
TRAIN_EXT_MIN = 40
# Evaluation cohorts: data directory, manifest, query role; EVAL-V3/V4 are pooled (primary).
EVAL = {
    "EVAL_V3": (Path("outputs/EXP-3D-RGBD-EVAL-V3-DATA"), "manifest_eval_v3.json", "primary_query"),
    "EVAL_V4": (Path("outputs/EXP-3D-RGBD-EVAL-V4-DATA"), "manifest_eval_v4.json", "primary_query"),
    "EVAL_FAR": (Path("outputs/EXP-3D-RGBD-EVAL-FAR-DATA"), "manifest_eval_far.json", "far_query"),
}
# The sealed V14 predictions of every cohort: V15 C0 must reproduce V14 C1 (width 32,
# TRAIN127); V14 C0 (width 32, TRAIN72 = V11 C1) is the descriptive 2x2 cell.
V14_SEAL = V14 / "raw/eval/prediction_seal.json"
V14_PLAN_SHA256 = "526c3623752317da05922d5368e341938ba9fc7b660b7e504746e910ae6eb1b6"
PLAN = "PLAN_BEFORE_V15_TRAINING.md"
PLAN_SHA256 = "4c0cc1b719b6dbc5459075ac36a89b637536b05d08b80867a5b485f31fb7beb6"
# Experiments that may legitimately write while V15 runs; never sealed, listed in the seal.
CONCURRENT = ()
BOUNDS_TEXT = (
    "CONTEXT_DEPTH: hull of the measured depth and cameras of the role's 3 context frames, "
    "anchor frame, frozen x1.1 padding, >=1 m"
)
COUNTS = {"C0": 64238, "C1": 250542}
SEEDS = [20260928, 20260929, 20260930]
THREADS = {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}


def seal_previous(root):
    """Hash every earlier experiment artifact (concurrent experiments excluded, listed)."""
    files = {}
    for base in (Path("outputs"), Path("docs/experiments")):
        for path in sorted(base.rglob("*")):
            if (
                path.is_file()
                and EXPERIMENT not in path.parts
                and not any(name in path.parts for name in CONCURRENT)
            ):
                files[str(path)] = {"sha256": sha(path), "bytes": path.stat().st_size}
    write_json(
        root / "audit/previous_experiment_seal.json",
        {
            "created_utc": datetime.datetime.now(datetime.UTC).isoformat(),
            "files": files,
            "excluded_concurrent_experiments": list(CONCURRENT),
        },
    )
    return len(files)


def prepare(root):
    root = Path(root).resolve()
    if root.name != EXPERIMENT:
        raise PermissionError("Exact V15 experiment directory required")
    if (root / "preregistration.json").exists():
        raise FileExistsError("No re-freezing")
    if not (root / "PROTOCOL.md").is_file():
        raise PermissionError("Frozen V15 protocol text required before freezing")
    for earlier in (V2, V3, V4, V5, V6, V7, V8, V9, V11, V12, V14):
        if read(earlier / "integrity.json")["status"] != "PASS":
            raise PermissionError(f"{earlier.name} must be finalized before V15 freezes")
    if sha(root / PLAN) != PLAN_SHA256:
        raise PermissionError("The V15 plan written before any V15 training must be intact")
    ext_lock = read(TRAIN_EXT / "candidate_lock.json")
    ext_integrity = read(TRAIN_EXT / "preparation_integrity.json")
    if (
        ext_lock["plan_sha256"] != V14_PLAN_SHA256
        or ext_integrity["status"] != "PASS"
        or ext_integrity["candidate_lock_sha256"] != sha(TRAIN_EXT / "candidate_lock.json")
    ):
        raise PermissionError("TRAIN-EXT must PASS against its frozen lock under the V14 plan")
    ext_records = read(TRAIN_EXT / "manifest_train_ext.json")["scenes"]
    if len(ext_records) < TRAIN_EXT_MIN or any(r["split"] != "TRAIN" for r in ext_records):
        raise PermissionError("V15 trains on the >= 40 valid TRAIN-EXT scenes of V14")
    v11_config = read(V11 / "training_contract.json")
    base_variant = "V11 C1"
    base_extra = v11_config["base_extra"]
    if GRID != v11_config["grid"]:
        raise PermissionError("V15 keeps the V11 grid")
    v11_selected = V11 / "selected_checkpoints.json"
    for seed, item in read(v11_selected)["C1"].items():
        if sha(Path(item["path"])) != item["sha256"]:
            raise PermissionError(f"V11 C1 selected checkpoint changed: {seed}")
    v12_selected = V12 / "selected_checkpoints.json"
    for seed, item in read(v12_selected)["C1"].items():
        if sha(Path(item["path"])) != item["sha256"]:
            raise PermissionError(f"V12 C1 selected checkpoint changed: {seed}")
    v14_seal = read(V14_SEAL)
    if v14_seal["query_depth_read"] or v14_seal["consistency"]["c0_minus_v11_c1_max_abs"]:
        raise PermissionError("The sealed V14 predictions (C0 = V11 C1) are required")
    cohorts = {}
    for name, (directory, filename, role) in EVAL.items():
        integrity = read(directory / "preparation_integrity.json")
        if integrity["status"] != "PASS" or integrity["candidate_lock_sha256"] != sha(
            directory / "candidate_lock.json"
        ):
            raise PermissionError(f"{name} preparation must PASS against its frozen lock")
        scenes = read(directory / filename)["scenes"]
        if len(scenes) < 20 or any(r["split"] != name for r in scenes):
            raise PermissionError(f"{name} needs at least 20 verified {name} scenes")
        sealed = [k for k in v14_seal["predictions"] if k.startswith(f"{name}/")]
        if len(sealed) != 4 * len(scenes):
            raise PermissionError(f"The sealed V14 {name} predictions are required")
        cohorts[name] = {
            "manifest": str((directory / filename).resolve()),
            "manifest_sha256": sha(directory / filename),
            "scenes": len(scenes),
            "role": role,
            "v14_seal": str(V14_SEAL.resolve()),
        }
    prior = read(V2 / "train_depth_prior.json")
    near, far = prior["near_m"], prior["far_m"]
    manifest = read(V6 / "scene_split.json")
    v5_scenes = read(V5 / "scene_split.json")["scenes"]
    train24 = sorted(r["scene_id"] for r in v5_scenes if r["split"] == "TRAIN")
    train72 = sorted(r["scene_id"] for r in manifest["scenes"] if r["split"] == "TRAIN")
    counts = {s: sum(r["split"] == s for r in manifest["scenes"]) for s in ("TRAIN", "DEV")}
    if counts != {"TRAIN": 72, "DEV": 8} or len(train24) != 24:
        raise PermissionError(f"Unexpected V2 cohort: {counts}")
    manifest["scenes"] = manifest["scenes"] + ext_records
    train_ext = sorted(r["scene_id"] for r in manifest["scenes"] if r["split"] == "TRAIN")
    validate_split(manifest)
    dev_volumes = {r["scene_id"].split("_")[1] for r in manifest["scenes"] if r["split"] == "DEV"}
    if len(train_ext) != 72 + len(ext_records) or any(
        r["scene_id"].split("_")[1] in dev_volumes for r in ext_records
    ):
        raise PermissionError("TRAIN-EXT must add new scenes outside every DEV volume")
    physical = {r["physical_scene_id"].casefold() for r in manifest["scenes"]}
    for name, entry in cohorts.items():
        for r in read(entry["manifest"])["scenes"]:
            if r["scene_id"] in train_ext or r["physical_scene_id"].casefold() in physical:
                raise PermissionError(f"{name} shares a scene or source asset with training")
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
            "source": "V2 locked cohort plus the TRAIN-EXT records (frozen camera-only role "
            "rule); the GT-free bounds rule is reused unchanged",
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
    v11_hashes = read(V11 / "architecture_contract.json")["parameter_hash_by_seed"]
    v12_hashes = read(V12 / "architecture_contract.json")["parameter_hash_by_seed"]
    if {str(s): h["C0"] for s, h in hashes.items()} != {s: h["C1"] for s, h in v11_hashes.items()}:
        raise AssertionError("V15 C0 must reuse the V11 C1 per-seed parameter draw")
    if {str(s): h["C1"] for s, h in hashes.items()} != {s: h["C1"] for s, h in v12_hashes.items()}:
        raise AssertionError("V15 C1 must reuse the V12 C1 (width 64) per-seed draw")
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
            "64 and expansion_dim 128 instead of 32 and 64 (250542 instead of 64238 "
            "parameters; the per-seed draws are V12 C1's and V11 C1's). Both train on TRAIN72 "
            f"plus {len(ext_records)} TRAIN-EXT scenes. "
            f"The recipe is the {base_variant} one: "
            f"the {GRID}^3 grid, the bounds rule ({BOUNDS_TEXT}), the frozen V2 C1 loss, 6000 "
            f"steps and the extra-view rule {base_extra!r} drawn from its own generator (seed "
            f"+ {EXTRA_SEED_OFFSET}); scene order, query slots and ray indices are identical. "
            "C0 retrains V14 C1 and must reproduce it bit-for-bit. DEV evaluation and "
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
        "primary_contrast": "pooled EVAL-V3 + EVAL-V4 WIDTH_AT_SCALE_GAIN = FAR AbsRel(C0 "
        "width 32) - FAR AbsRel(C1 width 64), both on TRAIN72+EXT (frozen V2 gate) and "
        "HARMONIC_HYBRID_GAIN = AbsRel(REPROJ_HARMONIC) - AbsRel(HFILL8 of C1) (INPAINT-V1 "
        "label rule)",
        "secondary_contrast": "DEV WIDTH_GAIN_DEV = AbsRel(C0) - AbsRel(C1), frozen V2 "
        "estimator, descriptive",
        "grid": GRID,
        "base_variant": base_variant,
        "base_extra": base_extra,
        "base_basis": f"{PLAN} (sha256 {PLAN_SHA256}), written before any V15 training: "
        "the V14 C1 recipe (V11 C1 on TRAIN72+EXT) at widths 32 and 64",
        "bounds_rule": "CONTEXT_DEPTH",
        "extra_views": {
            "max": EXTRA_MAX,
            "rule": base_extra,
            "generator_seed": f"seed + {EXTRA_SEED_OFFSET}",
        },
        "eval": {
            "cohorts": cohorts,
            "primary": ["EVAL_V3", "EVAL_V4"],
            "descriptive": ["EVAL_FAR"],
            "reprojection_fill_constant_m": v11_config["eval_v3"]["reprojection_fill_constant_m"],
            "near_px": v11_config["eval_v3"]["near_px"],
            "abstain_opacity_threshold": 0.5,
            "c0_reproduction_tolerance": 1e-6,
            "width64_train72": {
                "selected": str(v12_selected.resolve()),
                "selected_sha256": sha(v12_selected),
                "method": "V12 C1 selected checkpoints (width 64, TRAIN72), V12 loader; "
                "the descriptive 2x2 cell, rendered on the primary cohorts only",
            },
            "last_step": 6000,
            "role": "mechanism cohorts: scene- and asset-disjoint from TRAIN72 and "
            "TRAIN-EXT, volume-disjoint from DEV, may share volumes with the training "
            "scenes; EVAL-V3 and EVAL-V4 are disjoint and pooled for the primary "
            "contrasts; EVAL-FAR reuses their scenes with far queries and is descriptive; "
            "never a qualification cohort",
        },
        "train_ext": {
            "manifest": str((TRAIN_EXT / "manifest_train_ext.json").resolve()),
            "manifest_sha256": sha(TRAIN_EXT / "manifest_train_ext.json"),
            "candidate_lock_sha256": sha(TRAIN_EXT / "candidate_lock.json"),
            "scenes": len(ext_records),
        },
        "train_sets": {"TRAIN24": train24, "TRAIN72": train72, "TRAIN_EXT": train_ext},
        "device": "cpu",
        "seeds": SEEDS,
        "steps": 6000,
        "checkpoint_steps": [100, 200, 300, 500, 750, 1000, 1500, 2000, 3000, 4000, 5000, 6000],
        "checkpoint_rationale": "identical to V6-V11",
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
        "sampling": "seeded permutation of the shared TRAIN72+EXT set, cycled; primary query "
        "slot cycles per scene visit; uniform with-replacement rays, identical indices for "
        "A/B and both widths; the extra-view count comes from its own generator so the ray "
        "stream is unchanged",
        "initialize": "from scratch; torch.manual_seed(seed) then the width's carrier; C0 "
        "equals the V11 C1 draw and C1 the V12 C1 draw; the depth bypass starts at zero "
        "without consuming the RNG",
        "parallel_workers": 6,
        "execution": "all six runs (2 widths x 3 seeds) concurrently, one core each, on "
        "CPU cores 8-23; no GPU is used or queried (GPU paused by the user); "
        "min_free_gpu_mib is unused",
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
        "denote the secondary DEV C0-C1 contrast, i.e. WIDTH_GAIN_DEV in V15",
        "dev_reuse": "the 8 DEV scenes were evaluated in V2-V14 and Core B V1/V2; in V15 "
        "DEV only selects checkpoints and gives descriptive results; the primary cohort is "
        "the pooled EVAL-V3 + EVAL-V4; FRESH-V1 and FRESH-V2 are not used",
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
            "authority": "mcss.mechanism_pilot.rgbd_wide_scale_carrier.training_loss: the V8 "
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
            "EVAL_PRIMARY": {
                "WIDTH_AT_SCALE_STATUS": "frozen V2 gate checks (mean>0, CIlo>0, >=.75 "
                "nonworse, all LOSO>0, positive top1 share<=.5) on WIDTH_AT_SCALE_GAIN = "
                "FAR AbsRel(C0) - FAR AbsRel(C1) over the pooled EVAL-V3 + EVAL-V4 scenes",
                "HARMONIC_HYBRID_LABEL": "INPAINT-V1 label rule (ABOVE / "
                "NOT_DISTINGUISHABLE / BELOW) on AbsRel(REPROJ_HARMONIC) - AbsRel(HFILL8 "
                "of C1) over the same scenes",
                "HFILL8_WIDTH_STATUS": "frozen V2 gate checks on AbsRel(HFILL8 of C0) - "
                "AbsRel(HFILL8 of C1), secondary",
                "AHFILL8": "HFILL8 whose FAR pixels of rendered opacity < 0.5 also take "
                "REPROJ_HARMONIC (V13); descriptive, its threshold was chosen on EVAL-V3/V4",
                "C0_REPRODUCES_V14_C1": "C0 predictions equal the sealed V14 C1 predictions "
                "of every cohort (max abs <= 1e-6) before any query depth is read, else stop",
                "TWO_BY_TWO": "descriptive: width 32 / TRAIN72 (the sealed V14 C0 "
                "predictions) and width 64 / TRAIN72 (V12 C1 selected checkpoints) on the same "
                "scenes; interaction [FAR(32,72) - FAR(64,72)] - [FAR(32,127) - FAR(64,127)]",
                "LAST_STEP": "descriptive: the step-6000 checkpoints of C0 and C1",
                "FAR": "valid pixels farther than 8 px from any context-depth reprojection "
                "hit (1-px z-buffer of the 3 context depth maps)",
                "HFILL8": "REPROJ_HARMONIC within 8 px of a hit, the carrier elsewhere",
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
                "A": "WIDTH_AT_SCALE_STATUS SUPPORTED and HARMONIC_HYBRID_LABEL ABOVE on the "
                "pooled EVAL-V3 + EVAL-V4: with more training scenes, width improves "
                "completion and the wider carrier beats classical inpainting in the hybrid",
                "B": "exactly one of the two conditions holds",
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
    source_paths += sorted(Path("scripts").glob("*rgbd_wide_scale*.py"))
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
    sizes = {"TRAIN72": len(train72), "TRAIN_EXT": len(train_ext), "DEV": counts["DEV"]}
    print("FROZEN", sizes, "previous_files_sealed", sealed)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    a = p.parse_args()
    prepare(a.root)
