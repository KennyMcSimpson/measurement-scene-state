#!/usr/bin/env python3
"""Freeze the V19 long-training protocol (V12 C1 recipe, width 64, 24000 steps, hit CPU).

Derived from the V17 freeze: the single variant L reuses the V12 C1 per-seed parameter draw, and
the V12 C1 training logs and step-6000 checkpoints are hash-locked as the built-in control. The
frozen V17 (running concurrently, excluded from the previous-experiment seal like V18) is
recorded for the P2 width contrast. Run from the repository root under cv_host_run.py (records
from zonghan_o store /home/zonghan/... paths).
"""

from __future__ import annotations

import argparse
import datetime
import importlib.metadata
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import torch

from mcss.mechanism_pilot.geometry_carrier_experiment import read, validate_split
from mcss.mechanism_pilot.rgbd_evidence_carrier import LOG_SIGMA0
from mcss.mechanism_pilot.rgbd_long_width_carrier import (
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
V11 = Path("outputs/EXP-3D-RGBD-COMPLETION-V11")
V12 = Path("outputs/EXP-3D-RGBD-WIDTH-V12")
V14 = Path("outputs/EXP-3D-RGBD-DATA-SCALE-V14")
V15 = Path("outputs/EXP-3D-RGBD-WIDTH-DATA-V15")
# Evaluation cohorts: data directory, manifest, query role, the manifest's split label.
EVAL = {
    "EVAL_V3": (
        Path("outputs/EXP-3D-RGBD-EVAL-V3-DATA"),
        "manifest_eval_v3.json",
        "primary_query",
        "EVAL_V3",
    ),
    "EVAL_V4": (
        Path("outputs/EXP-3D-RGBD-EVAL-V4-DATA"),
        "manifest_eval_v4.json",
        "primary_query",
        "EVAL_V4",
    ),
    "EVAL_FAR": (
        Path("outputs/EXP-3D-RGBD-EVAL-FAR-DATA"),
        "manifest_eval_far.json",
        "far_query",
        "EVAL_FAR",
    ),
    "FRESH_V2": (
        Path("outputs/EXP-3D-RGBD-FRESH-V2-DATA"),
        "manifest_fresh_v2.json",
        "primary_query",
        "FRESH_QUALIFICATION",
    ),
}
PRIMARY_COHORTS = ["EVAL_V3", "EVAL_V4"]
PLAN = "PLAN_BEFORE_V19_TRAINING.md"
PLAN_SHA256 = "7649a2da96bc0cb68fab7bc1e4c612bb0a7b918a7bcbb71ce2205978be37ced4"
CONCURRENT = ("EXP-3D-RGBD-LONG-TRAIN-V17", "EXP-3D-RGBD-LONG-TRAIN-DATA-V18")
V17 = Path("outputs/EXP-3D-RGBD-LONG-TRAIN-V17")
V17_PREREGISTRATION_SHA256 = "36b8aa6e4e527025d3746aa7af5ffc060fbf7683fc1a26ff50a94e3c744c1d1f"
BOUNDS_TEXT = (
    "CONTEXT_DEPTH: hull of the measured depth and cameras of the role's 3 context frames, "
    "anchor frame, frozen x1.1 padding, >=1 m"
)
COUNTS = {"L": 250542}
SEEDS = [20260928, 20260929, 20260930]
STEPS = 24000
V11_CHECKPOINTS = [100, 200, 300, 500, 750, 1000, 1500, 2000, 3000, 4000, 5000, 6000]
CHECKPOINT_STEPS = V11_CHECKPOINTS + [8000, 10000, 12000, 16000, 20000, 24000]
CURVE_STEPS = [1000, 2000, 3000, 4000, 5000, 6000, 8000, 10000, 12000, 16000, 20000, 24000]
THREADS = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "ATEN_CPU_CAPABILITY": "avx2",
    "MKL_CBWR": "AVX2",
    "DNNL_MAX_CPU_ISA": "AVX2",
}
LAUNCHER = Path.home() / "cv_tools/cv_host_run.py"


def seal_previous(root):
    """Hash every earlier experiment artifact present on this host (concurrent excluded, listed)."""
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
            "host": platform.node(),
            "files": files,
            "excluded_concurrent_experiments": list(CONCURRENT),
        },
    )
    return len(files)


def control_references():
    """The V12 C1 training logs and step-6000 checkpoints that every V19 seed must reproduce."""
    refs = {}
    for seed in SEEDS:
        run = V12 / "checkpoints" / f"C1_{seed}"
        log, checkpoint = run / "training.jsonl", run / "step_006000.pt"
        if len(log.read_text().splitlines()) != 6000:
            raise PermissionError(f"V12 C1 {seed} training log must have 6000 steps")
        refs[str(seed)] = {
            "training_log": str(log.resolve()),
            "training_log_sha256": sha(log),
            "step_006000": str(checkpoint.resolve()),
            "step_006000_sha256": sha(checkpoint),
        }
    return refs


def prepare(root):
    root = Path(root).resolve()
    if root.name != EXPERIMENT:
        raise PermissionError("Exact V19 experiment directory required")
    if (root / "preregistration.json").exists():
        raise FileExistsError("No re-freezing")
    if not (root / "PROTOCOL.md").is_file():
        raise PermissionError("Frozen V19 protocol text required before freezing")
    for earlier in (V2, V3, V4, V5, V6, V7, V8, V9, V11, V12, V14, V15):
        if read(earlier / "integrity.json")["status"] != "PASS":
            raise PermissionError(f"{earlier.name} must be finalized before V19 freezes")
    if sha(root / PLAN) != PLAN_SHA256:
        raise PermissionError("The V19 plan written before any V19 training must be intact")
    if sha(LAUNCHER) != sha(Path("../_archives_cv/cv_host_run.py")):
        raise PermissionError("The path-mapping launcher must equal the migrated one")
    v11_config = read(V11 / "training_contract.json")
    base_extra = v11_config["base_extra"]
    if GRID != v11_config["grid"] or v11_config["checkpoint_steps"] != V11_CHECKPOINTS:
        raise PermissionError("V19 keeps the V11 grid and the V11 checkpoints up to 6000 steps")
    for key in ("learning_rate", "batch_rays", "gradient_clip", "seeds", "num_threads"):
        expected = {
            "learning_rate": 0.001,
            "batch_rays": 1024,
            "gradient_clip": 1.0,
            "seeds": SEEDS,
            "num_threads": 1,
        }[key]
        if v11_config[key] != expected:
            raise PermissionError(f"V11 {key} differs from the V19 recipe")
    for seed, item in read(V11 / "selected_checkpoints.json")["C1"].items():
        if sha(Path(item["path"])) != item["sha256"]:
            raise PermissionError(f"V11 C1 selected checkpoint changed: {seed}")
    for seed, item in read(V12 / "selected_checkpoints.json")["C1"].items():
        if sha(Path(item["path"])) != item["sha256"]:
            raise PermissionError(f"V12 C1 selected checkpoint changed: {seed}")
    v12_config = read(V12 / "training_contract.json")
    recipe_keys = ("hidden", "expansion", "model", "train")
    shared_keys = ("learning_rate", "batch_rays", "gradient_clip", "seeds", "num_threads")
    if (
        v12_config["checkpoint_steps"] != V11_CHECKPOINTS
        or v12_config["base_extra"] != base_extra
        or v12_config["grid"] != GRID
        or v12_config["steps"] != 6000
        or {k: v12_config["variant_specs"]["C1"][k] for k in recipe_keys}
        != {k: VARIANT_SPECS["L"][k] for k in recipe_keys}
        or any(v12_config[k] != v11_config[k] for k in shared_keys)
    ):
        raise PermissionError("V19 keeps the V12 C1 recipe")
    if sha(V17 / "preregistration.json") != V17_PREREGISTRATION_SHA256:
        raise PermissionError("The P2 width contrast needs the frozen V17")
    controls = control_references()
    cohorts = {}
    for name, (directory, filename, role, split) in EVAL.items():
        integrity = read(directory / "preparation_integrity.json")
        if integrity["status"] != "PASS" or integrity["candidate_lock_sha256"] != sha(
            directory / "candidate_lock.json"
        ):
            raise PermissionError(f"{name} preparation must PASS against its frozen lock")
        scenes = read(directory / filename)["scenes"]
        if len(scenes) < 10 or any(r["split"] != split for r in scenes):
            raise PermissionError(f"{name} needs at least 10 verified {split} scenes")
        cohorts[name] = {
            "manifest": str((directory / filename).resolve()),
            "manifest_sha256": sha(directory / filename),
            "scenes": len(scenes),
            "role": role,
            "split": split,
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
    validate_split(manifest)
    physical = {r["physical_scene_id"].casefold() for r in manifest["scenes"]}
    for name, entry in cohorts.items():
        for r in read(entry["manifest"])["scenes"]:
            if r["scene_id"] in train72 or r["physical_scene_id"].casefold() in physical:
                raise PermissionError(f"{name} shares a scene or source asset with TRAIN72 or DEV")
    write_json(root / "scene_split.json", manifest)
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
            "source": "V2 locked cohort (V6 scene split, TRAIN72 + DEV8), unchanged",
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
    v12_hashes = read(V12 / "architecture_contract.json")["parameter_hash_by_seed"]
    if {str(s): h["L"] for s, h in hashes.items()} != {s: h["C1"] for s, h in v12_hashes.items()}:
        raise AssertionError("V19 L must reuse the V12 C1 per-seed parameter draw")
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
            "only_difference_from_v12_c1": "24000 instead of 6000 training steps and six extra "
            f"checkpoints after step 6000; the recipe is V12 C1: width 64, the {GRID}^3 grid, the "
            f"bounds rule ({BOUNDS_TEXT}), the frozen V2 C1 loss and the extra-view rule "
            f"{base_extra!r} drawn from its own generator (seed + {EXTRA_SEED_OFFSET}). The "
            "learning rate is constant and the checkpoint schedule up to 6000 is V12's, so steps "
            "1-6000 must reproduce V12 C1 bit-for-bit, and each seed sees V17 L's data stream. "
            "DEV evaluation and checkpoint selection use the standard 3-view contexts.",
            "depth_bypass": {
                "log_sigma_init": LOG_SIGMA0,
                "sampling": "nearest pixel of the full-resolution measured ray-distance map",
                "placement": "after the >=2-view support gate: one measured view suffices",
                "inputs": "context RGB, measured context depth, context cameras; no query",
            },
            "state": "query-independent; all states sealed before query depth is read",
            "test_time_input": "RGB+DEPTH+CAMERA",
            "test_time_depth_used": True,
            "new_backbone_or_learned_matching": False,
        },
    )
    config = {
        "experiment": EXPERIMENT,
        "variants": list(VARIANT_SPECS),
        "variant_specs": VARIANT_SPECS,
        "primary_method": "L",
        "primary_pair": list(PRIMARY_PAIR),
        "primary_contrast": "pooled EVAL-V3 + EVAL-V4 LENGTH_GAIN = FAR AbsRel(L at step 6000) - "
        "FAR AbsRel(L at step 24000) (P1, frozen V2 gate) and WIDTH_GAIN_24K = FAR AbsRel(V17 L "
        "at step 24000) - FAR AbsRel(V19 L at step 24000) (P2, frozen V2 gate, paired by scene "
        "from both experiments' scored query rows after both finish)",
        "width_contrast": {
            "experiment": "EXP-3D-RGBD-LONG-TRAIN-V17",
            "directory": str(V17.resolve()),
            "preregistration_sha256": V17_PREREGISTRATION_SHA256,
            "script": "scripts/analyze_rgbd_long_width_vs_v17.py",
            "no_data_rule": "if the V17 built-in control fails (no T24000), WIDTH_STATUS is "
            "NO_DATA and the branch follows P1 alone",
        },
        "grid": GRID,
        "base_variant": "V12 C1",
        "base_extra": base_extra,
        "base_basis": f"{PLAN} (sha256 {PLAN_SHA256}), written before any V19 training",
        "bounds_rule": "CONTEXT_DEPTH",
        "extra_views": {
            "max": EXTRA_MAX,
            "rule": base_extra,
            "generator_seed": f"seed + {EXTRA_SEED_OFFSET}",
        },
        "eval": {
            "cohorts": cohorts,
            "primary": PRIMARY_COHORTS,
            "descriptive": ["EVAL_FAR", "FRESH_V2"],
            "reprojection_fill_constant_m": v11_config["eval_v3"]["reprojection_fill_constant_m"],
            "near_px": v11_config["eval_v3"]["near_px"],
            "abstain_opacity_threshold": 0.5,
            "checkpoint_labels": {
                "T<step>": "the L checkpoint at that step",
                "SEL": "the frozen V2 DEV rule over all 18 checkpoints (selection.json)",
            },
            "curve_steps_primary": CURVE_STEPS,
            "steps_descriptive_cohorts": [6000, 24000],
            "role": "mechanism cohorts: scene- and asset-disjoint from TRAIN72 and DEV; EVAL-V3 "
            "and EVAL-V4 are disjoint and pooled for the primary contrasts; EVAL-FAR reuses their "
            "scenes with far queries; FRESH-V2 (sixth use) is volume-disjoint from TRAIN72 and "
            "only a mechanism contrast; never a qualification cohort",
        },
        "control": {
            "rule": "for every seed, training.jsonl lines 1-6000 equal V12 C1 except 'seconds', "
            "and the step-6000 parameters equal V12 C1 step_006000.pt tensor by tensor; else stop",
            "references": controls,
        },
        "train_sets": {"TRAIN24": train24, "TRAIN72": train72},
        "device": "cpu",
        "seeds": SEEDS,
        "steps": STEPS,
        "checkpoint_steps": CHECKPOINT_STEPS,
        "checkpoint_rationale": "identical to V6-V15 up to 6000 steps, then 8000-24000",
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
        "cpu_affinity": list(range(24, 40)),
        "host_launcher": {"path": str(LAUNCHER), "sha256": sha(LAUNCHER)},
        "augmentation": f"extra context views by the base rule ({base_extra}); nothing else",
        "image_size": [128, 160],
        "train_targets": "TRAIN primary_query2 RGB+ray-distance depth; never carrier input",
        "context_input": "context RGB + measured context ray distance + cameras; query depth never "
        "enters construction",
        "context_roles": "independent A/B states, equal loss weights",
        "sampling": "seeded permutation of TRAIN72, cycled; primary query slot cycles per scene "
        "visit; uniform with-replacement rays, identical indices for A/B; the extra-view count "
        "comes from its own generator",
        "initialize": "from scratch; torch.manual_seed(seed) then the width-64 carrier (the V12 C1 "
        "draw); the depth bypass starts at zero without consuming the RNG",
        "parallel_workers": 3,
        "execution": "three runs (one per seed) concurrently, one core each, on CPU cores 24-39 of "
        "hit, AVX2 kernels only; no GPU is used",
        "infrastructure_retries": 2,
        "infrastructure_failure_policy": "a signal-terminated run is moved to "
        "audit/failed_attempts with an incident record and rerun from scratch with identical "
        "seed/config, at most infrastructure_retries times; any other error is never retried and "
        "a failed run stops the experiment",
        "checkpoint_selection": "independent per seed using the frozen V2 DEV rule; "
        "never best seed",
        "seed_handling": "all seeds equally weighted within scene; each seed reported",
        "dev_reuse": "DEV only selects checkpoints (SEL, descriptive); the primary cohort is the "
        "pooled EVAL-V3 + EVAL-V4",
        "fresh_qualification_included": False,
        "writer_updates": False,
        "dynamic_ttt_run": False,
    }
    write_json(root / "training_contract.json", config)
    write_json(
        root / "loss_contract.json",
        {
            "L": "V2 C1 loss: mean_A_B[RGB Charbonnier(eps=.001) + masked depth AbsRel] + "
            ".1 mean_A_B surface loss",
            "authority": "mcss.mechanism_pilot.rgbd_long_width_carrier.training_loss: the V8 "
            "generalized frozen V2 C1 loss",
            "v2_loss_contract_sha256": sha(V2 / "loss_contract.json"),
            "lambda_surface": 0.1,
            "regularization": 0,
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
            "EVAL_PRIMARY": {
                "LENGTH_STATUS": "frozen V2 gate checks (mean>0, CIlo>0, >=.75 nonworse, all "
                "LOSO>0, positive top1 share<=.5) on LENGTH_GAIN = FAR AbsRel(T6000) - FAR "
                "AbsRel(T24000) over the pooled EVAL-V3 + EVAL-V4 scenes",
                "WIDTH_STATUS": "frozen V2 gate checks on WIDTH_GAIN_24K = FAR AbsRel(V17 L at "
                "T24000) - FAR AbsRel(V19 L at T24000) over the same pooled scenes, paired by "
                "scene; NO_DATA if V17 has no T24000",
                "AHFILL8_LABEL_DESCRIPTIVE": "INPAINT-V1 label rule (ABOVE / NOT_DISTINGUISHABLE "
                "/ BELOW) on AbsRel(REPROJ_HARMONIC) - AbsRel(AHFILL8_T24000) over the same scenes",
                "INTERACTION_DESCRIPTIVE": "[FAR(V17 T6000) - FAR(V17 T24000)] - [FAR(V19 T6000) "
                "- FAR(V19 T24000)] per scene, bootstrapped",
                "FAR": "valid pixels farther than 8 px from any context-depth reprojection hit "
                "(1-px z-buffer of the 3 context depth maps)",
                "HFILL8": "REPROJ_HARMONIC within 8 px of a hit, the carrier elsewhere",
                "AHFILL8": "HFILL8 whose FAR pixels of rendered opacity < 0.5 also take "
                "REPROJ_HARMONIC (V13 rule, threshold 0.5)",
                "aggregation": "per scene: seeds equal, then the mean of role means of query "
                "values; 10000-draw scene bootstrap, seed 20260928",
            },
            "CONTROL": "every seed reproduces V12 C1 for steps 1-6000 (training log and step-6000 "
            "parameters) before any EVAL prediction is made, else stop",
            "interpretation": {
                "A": "LENGTH_STATUS SUPPORTED and WIDTH_STATUS SUPPORTED: with long training "
                "the wider carrier improves completion",
                "B": "exactly one of the two conditions holds",
                "C": "otherwise: even with long training, width is not the completion bottleneck",
            },
            "fresh_qualification_included": False,
            "dynamic_ttt_next_stage_allowed": False,
        },
    )
    write_json(
        root / "audit/environment.json",
        {
            "host": platform.node(),
            "cpu": next(
                (
                    line.split(":", 1)[1].strip()
                    for line in open("/proc/cpuinfo")
                    if line.startswith("model name")
                ),
                None,
            ),
            "python": sys.version,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "thread_environment": {k: os.environ.get(k) for k in THREADS},
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
    source_paths += sorted(Path("scripts").glob("*rgbd_long_width*.py"))
    source_paths += [LAUNCHER]
    write_json(
        root / "preregistration.json",
        {
            "status": "FROZEN_BEFORE_FORMAL_TRAINING",
            "experiment": EXPERIMENT,
            "primary_method": "L",
            "input_sha256": {str((root / n).resolve()): sha(root / n) for n in names},
            "source_sha256": {str(p.resolve()): sha(p) for p in source_paths},
            "data_sha256": data_hashes,
            "fresh_status": "NOT_INCLUDED_MECHANISM_ROUND",
            "formal_training_started": False,
            "qualification_opened": False,
        },
    )
    print(
        "FROZEN", {"TRAIN72": len(train72), "DEV": counts["DEV"]}, "previous_files_sealed", sealed
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    a = p.parse_args()
    prepare(a.root)
