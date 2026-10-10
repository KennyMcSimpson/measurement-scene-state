#!/usr/bin/env python3
"""Freeze the Core B V3 protocol (learned writes on the frozen V17 L step-24000 carrier, GPU 4).

Hand-written from the Core B V2 freeze. Only the frozen carrier changes: the V17 L checkpoint at
step 24000 of each seed, hash-checked against V17's own DEV-curve record, with V17's
preregistration and built-in control checked. The manifest is the V7 TRAIN72 + DEV8 split that
Core B V1/V2 used; V17 trained on the same split (checked record by record, paths aside). Run from
the repository root under cv_host_run.py (records from zonghan_o store /home/zonghan/... paths).
"""

from __future__ import annotations

import argparse
import datetime
import json
import shutil
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import torch

from mcss.dynamic.config import WriteConfig
from mcss.mechanism_pilot.geometry_carrier_experiment import read, validate_split
from mcss.mechanism_pilot.rgbd_stream_write_v3 import (
    CONTROLS,
    EXPERIMENT,
    POLICIES,
    STREAM_LENGTH,
    TRAINED_VIEW_COUNT,
    load_frozen_carrier,
    make_write_rule,
    stream_frame_ids,
)
from mcss.mechanism_pilot.small_training import sha, write_json

V7 = Path("outputs/EXP-3D-RGBD-DEPTH-BOUNDS-V7")
V17 = Path("outputs/EXP-3D-RGBD-LONG-TRAIN-V17")
V17_PREREGISTRATION_SHA256 = "36b8aa6e4e527025d3746aa7af5ffc060fbf7683fc1a26ff50a94e3c744c1d1f"
COREB_V2 = Path("outputs/EXP-3D-RGBD-DYNAMIC-WRITE-V2")
PLAN = "PLAN_BEFORE_COREB_V3.md"
PLAN_SHA256 = "66892c425e0c36f135c62cf049a513c19c406e09d4451d71d5573975818722ce"
QUALIFICATION = Path("outputs/EXP-3D-RGBD-FRESH-QUALIFICATION-V2")
CARRIER_STEP = 24000
THREADS = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "ATEN_CPU_CAPABILITY": "avx2",
    "MKL_CBWR": "AVX2",
    "DNNL_MAX_CPU_ISA": "AVX2",
    "NVIDIA_TF32_OVERRIDE": "0",
}
STEPS = 3000
GPU = "4"
LAUNCHER = Path.home() / "cv_tools/cv_host_run.py"
# Running or recently finished experiments whose directories may still change; not sealed.
CONCURRENT = (
    "EXP-3D-RGBD-LONG-TRAIN-WIDTH-V19",
    "EXP-3D-RGBD-LONG-TRAIN-WIDTH-DATA-V20",
    "EXP-3D-RGBD-SEGMENT-DATA-V1",
    "EXP-3D-RGBD-SEGMENT-DIVERSITY-V21",
)


def seal_previous(root):
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


def episodes(manifest, split):
    rows = {}
    for record in manifest["scenes"]:
        if record["split"] != split:
            continue
        for role in ("A", "B"):
            ids = stream_frame_ids(record, role)
            rows[f"{record['scene_id']}/{role}"] = list(ids) if ids else None
    return rows


def semantic(manifest):
    """Scene, split, roles and frame ids of every record (paths differ between hosts)."""
    return {
        r["scene_id"]: (
            r["split"],
            json.dumps(r["roles"], sort_keys=True),
            [f["frame_id"] for f in r["frames"]],
        )
        for r in manifest["scenes"]
    }


def v17_carriers():
    """The V17 L step-24000 checkpoint of every seed, checked against V17's own records."""
    if sha(V17 / "preregistration.json") != V17_PREREGISTRATION_SHA256:
        raise PermissionError("V17 is not the frozen V17")
    if read(V17 / "audit/v11_prefix_control.json")["status"] != "PASS":
        raise PermissionError("V17's built-in control must PASS")
    if read(V17 / "eval_results.json")["control"] != "PASS":
        raise PermissionError("V17 must have finished its analysis")
    config = read(V17 / "training_contract.json")
    carriers = {}
    for seed in config["seeds"]:
        run = V17 / "checkpoints" / f"L_{seed}"
        entry = next(e for e in read(run / "dev_curve.json") if e["step"] == CARRIER_STEP)
        path = Path(entry["checkpoint"])
        if sha(path) != entry["checkpoint_sha256"]:
            raise PermissionError(f"V17 step-{CARRIER_STEP} checkpoint changed: {seed}")
        carriers[str(seed)] = {
            "path": str(path.resolve()),
            "sha256": entry["checkpoint_sha256"],
            "step": CARRIER_STEP,
            "v17_dev_query_absrel": entry["query_absrel"],
        }
    return config["seeds"], carriers


def prepare(root):
    root = Path(root).resolve()
    if root.name != EXPERIMENT:
        raise PermissionError("Exact Core B V3 directory required")
    if (root / "preregistration.json").exists():
        raise FileExistsError("No re-freezing")
    if not (root / "PROTOCOL.md").is_file():
        raise PermissionError("Frozen protocol text required before freezing")
    if read(V7 / "integrity.json")["status"] != "PASS":
        raise PermissionError("V7 must be finalized")
    if read(COREB_V2 / "integrity.json")["status"] != "PASS":
        raise PermissionError("Core B V2 must be finalized")
    if sha(root / PLAN) != PLAN_SHA256:
        raise PermissionError("The Core B V3 plan must be intact")
    if sha(LAUNCHER) != sha(Path("../_archives_cv/cv_host_run.py")):
        raise PermissionError("The path-mapping launcher must equal the migrated one")
    qualification = read(QUALIFICATION / "fresh_qualification_results.json")
    if (
        qualification["FRESH_QUALIFICATION_STATUS"] != "QUALIFIED"
        or qualification["DYNAMIC_TTT_NEXT_STAGE_ALLOWED"] is not True
    ):
        raise PermissionError("Core B opens only after the static carrier is QUALIFIED")
    manifest = read(V7 / "scene_split.json")
    if semantic(read(V17 / "scene_split.json")) != semantic(manifest):
        raise PermissionError("V17 must have trained on the V7 TRAIN72 + DEV8 split")
    validate_split(manifest)
    write_json(root / "scene_split.json", manifest)
    dev, train = episodes(manifest, "DEV"), episodes(manifest, "TRAIN")
    if any(v is None for v in dev.values()):
        raise PermissionError(f"Every DEV episode needs a full stream: {dev}")
    eligible = sorted(k for k, v in train.items() if v is not None)
    v2_streams = read(COREB_V2 / "stream_roles.json")
    if v2_streams["DEV"] != dev or v2_streams["TRAIN"] != train:
        raise PermissionError("The stream roles must be Core B V2's")
    write_json(
        root / "stream_roles.json",
        {
            "rule": f"{STREAM_LENGTH} frame ids evenly spaced over the scene's free frames "
            "(in no context or query role), shared by both roles, arrival in increasing id "
            "order; frame ids only; an episode without a full stream is ineligible, never "
            "shortened (the Core B V1 rule, identical to Core B V2's roles)",
            "stream_length": STREAM_LENGTH,
            "DEV": dev,
            "TRAIN": train,
            "train_eligible_episodes": eligible,
            "selection_uses_media_or_model": False,
        },
    )
    seeds, carriers = v17_carriers()
    for seed in seeds:
        item = carriers[str(seed)]
        carrier, _ = load_frozen_carrier(item["path"], "cpu")
        rule = make_write_rule(carrier, seed)
        item["write_rule_parameters"] = sum(p.numel() for p in rule.parameters())
        item["carrier_trainable_parameters"] = sum(p.requires_grad for p in carrier.parameters())
        item["carrier_parameters"] = sum(p.numel() for p in carrier.parameters())
    shutil.copyfile(V7 / "train_depth_prior.json", root / "train_depth_prior.json")
    write_json(
        root / "carrier_lock.json",
        {
            "source": f"V17 L step-{CARRIER_STEP} checkpoints (the last step, not DEV-selected), "
            f"fixed by {PLAN} (sha256 {PLAN_SHA256})",
            "carrier_variant": f"V17_L_T{CARRIER_STEP}",
            "carrier_qualification": "the V7 C1 recipe (QUALIFIED on FRESH-V2) at width 32 trained "
            "for 24000 steps; not independently qualified",
            "v17_preregistration_sha256": V17_PREREGISTRATION_SHA256,
            "qualification_results_sha256": sha(QUALIFICATION / "fresh_qualification_results.json"),
            "bounds_rule": "CONTEXT_DEPTH (the V7 C1, V9 and V17 rule, from the warmup context "
            "only)",
            "carriers": carriers,
            "slow_carrier_updated": False,
        },
    )
    baseline_policy = "OFF"
    config = {
        "experiment": EXPERIMENT,
        "seeds": seeds,
        "policies": list(POLICIES),
        "controls": list(CONTROLS),
        "primary_contrast": f"BEYOND_BASELINE = AbsRel({baseline_policy}) - AbsRel(ALL)",
        "carrier_variant": f"V17_L_T{CARRIER_STEP}",
        "baseline_policy": baseline_policy,
        "baseline_rule": "OFF, as in Core B V2: the V17 carrier was trained with 3-7 views, so "
        "seven views are inside its training range",
        "stream_length": STREAM_LENGTH,
        "trained_view_count": TRAINED_VIEW_COUNT,
        "write_rule": "mcss.dynamic.write_rule.DirectWriteRule (frozen project design)",
        "write_config": asdict(WriteConfig()),
        "training_action": "ALL at every stream arrival",
        "training_loss": "frozen V2 C1 loss (RGB Charbonnier + masked depth AbsRel + .1 surface) "
        "through the V8 32^3 generalization, on 1024 uniform rays of one TRAIN primary query, "
        "rendered after the stream",
        "training_episodes": "seeded permutation of the eligible TRAIN72 episodes (scene, role), "
        "cycled; the primary-query slot cycles per visit",
        "steps": STEPS,
        "optimizer": "Adam",
        "learning_rate": 0.001,
        "gradient_clip": 1.0,
        "batch_rays": 1024,
        "num_threads": 1,
        "thread_environment": THREADS,
        "cpu_affinity": list(range(56, 64)),
        "device": "cuda",
        "gpu": GPU,
        "gpu_rule": "launched through ~/cv_tools/cv_gpu_run.sh (GPU 4 only, only when idle); TF32 "
        "off (NVIDIA_TF32_OVERRIDE=0); not bit-reproducible, every comparison is within this run",
        "host_launcher": {"path": str(LAUNCHER), "sha256": sha(LAUNCHER)},
        "parallel_workers": 3,
        "slow_carrier": "frozen (requires_grad False); only the write rule is optimized",
        "write_rule_initialization": "torch.manual_seed(seed) then DirectWriteRule(); the "
        "initial rule is saved and evaluated as ALL_UNTRAINED",
        "checkpoint_selection": "none: the final step is evaluated; no DEV forward during training",
        "dev_reuse": "the 8 DEV scenes were evaluated in V2-V10 and in Core B V1/V2, and V17 "
        "recorded its DEV curve at step 24000; Core B V3 is a DEV mechanism round, not a "
        "qualification; FRESH-V1/V2 are not used",
        "fresh_qualification_included": False,
    }
    write_json(root / "training_contract.json", config)
    write_json(
        root / "decision_rules.json",
        {
            "aggregation": "per seed: scene mean of role means of query AbsRel; seeds equally "
            "weighted within scene",
            "bootstrap": {"unit": "physical scene", "draws": 10000, "seed": 20260928},
            "gate": "frozen V2 primary-contrast checks: mean>0, CIlo>0, >=75% scenes non-worse; "
            "every LOSO mean >0; positive top1 share <=.5; sealed-state integrity PASS; status "
            "HARMFUL / SUPPORTED / PARTIAL / NOT_ESTABLISHED as in geometry_carrier_statistics",
            "BEYOND_BASELINE_STATUS": f"gate on BEYOND_BASELINE = {baseline_policy} - ALL "
            "(primary)",
            "SPECIFICITY_STATUS": "gate on WRITE_SPECIFICITY = ALL_WRONG_SCENE - ALL (primary)",
            "WRITE_STATUS": "gate on WRITE_GAIN = OFF - ALL",
            "STREAM_STATUS": "gate on STREAM_WRITE_GAIN = NO_STREAM - ALL",
            "BEYOND_CLAMP_STATUS": "gate on BEYOND_CLAMP_GAIN = OFF_CLAMP3 - ALL",
            "WRITE_SPECIFICITY_STATUS": "ALL_WRONG_SCENE - ALL has CIlo>0 -> SUPPORTED",
            "descriptive": "STREAM_CACHE_EFFECT = NO_STREAM - OFF; CLAMP_EFFECT = OFF - "
            "OFF_CLAMP3; FUSE and COMPLETE gains over OFF; TRAINING_GAIN = ALL_UNTRAINED - ALL; "
            "per-seed gains; fast-weight norms; delta1; the same contrasts of Core B V2",
            "interpretation": {
                "A": "BEYOND_BASELINE_STATUS and SPECIFICITY_STATUS SUPPORTED: on the stronger "
                "carrier, learned writes beat the preregistered non-learned baseline with "
                "scene-specific information; next an independent qualification of the carrier "
                "and the writes together",
                "B": "BEYOND_BASELINE point estimate >0, not A",
                "C": "otherwise: even on the stronger carrier the frozen write design does not "
                "beat caching the stream frames; Core B is not supported in its current design",
            },
            "final_holdout": "never opened",
        },
    )
    write_json(
        root / "audit/environment.json",
        {
            "python": sys.version,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        },
    )
    sealed = seal_previous(root)
    data_hashes = {
        str(Path(f[kind])): sha(Path(f[kind]))
        for r in manifest["scenes"]
        for f in r["frames"]
        for kind in ("rgb", "depth")
    }
    write_json(root / "audit/data_hashes.json", data_hashes)
    names = (
        "scene_split.json",
        "stream_roles.json",
        "carrier_lock.json",
        "training_contract.json",
        "decision_rules.json",
        "train_depth_prior.json",
        "PROTOCOL.md",
        PLAN,
        "audit/data_hashes.json",
        "audit/previous_experiment_seal.json",
    )
    sources = sorted(Path("src/mcss").rglob("*.py")) + sorted(
        Path("scripts").glob("*rgbd_stream_write_v3*.py")
    )
    sources.append(LAUNCHER)
    write_json(
        root / "preregistration.json",
        {
            "status": "FROZEN_BEFORE_FORMAL_TRAINING",
            "experiment": EXPERIMENT,
            "input_sha256": {str((root / n).resolve()): sha(root / n) for n in names},
            "source_sha256": {str(p.resolve()): sha(p) for p in sources},
            "data_sha256": data_hashes,
            "fresh_status": "NOT_INCLUDED_DEV_MECHANISM_ROUND",
            "formal_training_started": False,
        },
    )
    print(
        "FROZEN_CORE_B_V3",
        config["carrier_variant"],
        baseline_policy,
        len(dev),
        "DEV episodes",
        len(eligible),
        "TRAIN episodes",
        sealed,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    prepare(parser.parse_args().root)
