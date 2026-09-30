#!/usr/bin/env python3
"""Freeze the Core B V2 protocol (learned writes on the frozen V9 carrier, CPU only)."""

from __future__ import annotations

import argparse
import datetime
import shutil
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import torch

from mcss.dynamic.config import WriteConfig
from mcss.mechanism_pilot.geometry_carrier_experiment import read, validate_split
from mcss.mechanism_pilot.rgbd_stream_write_v2 import (
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
V9 = Path("outputs/EXP-3D-RGBD-VIEWCOUNT-V9")
PLAN = "PLAN_BEFORE_V9_RESULTS.md"
PLAN_SHA256 = "42c83ceca313ba4debe8befb9f5daaf06293978a912dea5fa6f23cd55a9b414a"
QUALIFICATION = Path("outputs/EXP-3D-RGBD-FRESH-QUALIFICATION-V2")
THREADS = {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
STEPS = 3000


def seal_previous(root):
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


def episodes(manifest, split):
    rows = {}
    for record in manifest["scenes"]:
        if record["split"] != split:
            continue
        for role in ("A", "B"):
            ids = stream_frame_ids(record, role)
            rows[f"{record['scene_id']}/{role}"] = list(ids) if ids else None
    return rows


def prepare(root):
    root = Path(root).resolve()
    if root.name != EXPERIMENT:
        raise PermissionError("Exact Core B V2 directory required")
    if (root / "preregistration.json").exists():
        raise FileExistsError("No re-freezing")
    if not (root / "PROTOCOL.md").is_file():
        raise PermissionError("Frozen protocol text required before freezing")
    if read(V7 / "integrity.json")["status"] != "PASS":
        raise PermissionError("V7 must be finalized")
    if read(V9 / "integrity.json")["status"] != "PASS":
        raise PermissionError("V9 must be finalized")
    if sha(root / PLAN) != PLAN_SHA256:
        raise PermissionError("The pre-V9 plan fixing the carrier must be intact")
    gain = float(read(V9 / "static_results.json")["surface_gain"]["mean"])
    variant = "C1" if gain > 0 else "C0"
    baseline_policy = "OFF" if variant == "C1" else "OFF_CLAMP3"
    qualification = read(QUALIFICATION / "fresh_qualification_results.json")
    if (
        qualification["FRESH_QUALIFICATION_STATUS"] != "QUALIFIED"
        or qualification["DYNAMIC_TTT_NEXT_STAGE_ALLOWED"] is not True
    ):
        raise PermissionError("Core B opens only after the static carrier is QUALIFIED")
    manifest = read(V7 / "scene_split.json")
    if sha(V9 / "scene_split.json") != sha(V7 / "scene_split.json"):
        raise PermissionError("V9 and Core B V1 must share the V7 manifest")
    validate_split(manifest)
    write_json(root / "scene_split.json", manifest)
    dev, train = episodes(manifest, "DEV"), episodes(manifest, "TRAIN")
    if any(v is None for v in dev.values()):
        raise PermissionError(f"Every DEV episode needs a full stream: {dev}")
    eligible = sorted(k for k, v in train.items() if v is not None)
    write_json(
        root / "stream_roles.json",
        {
            "rule": f"{STREAM_LENGTH} frame ids evenly spaced over the scene's free frames "
            "(in no context or query role), shared by both roles, arrival in increasing id "
            "order; frame ids only; an episode without a full stream is ineligible, never "
            "shortened (the Core B V1 rule)",
            "stream_length": STREAM_LENGTH,
            "DEV": dev,
            "TRAIN": train,
            "train_eligible_episodes": eligible,
            "selection_uses_media_or_model": False,
        },
    )
    v9_config = read(V9 / "training_contract.json")
    selected = read(V9 / "selected_checkpoints.json")[variant]
    carriers = {}
    for seed in v9_config["seeds"]:
        item = selected[str(seed)]
        if sha(Path(item["path"])) != item["sha256"]:
            raise PermissionError("Sealed V9 checkpoint changed")
        carrier, _ = load_frozen_carrier(item["path"], "cpu")
        rule = make_write_rule(carrier, seed)
        carriers[str(seed)] = {
            **item,
            "write_rule_parameters": sum(p.numel() for p in rule.parameters()),
            "carrier_trainable_parameters": sum(p.requires_grad for p in carrier.parameters()),
        }
    shutil.copyfile(V7 / "train_depth_prior.json", root / "train_depth_prior.json")
    write_json(
        root / "carrier_lock.json",
        {
            "source": f"sealed V9 {variant} per-seed DEV selections, chosen by {PLAN} "
            f"(sha256 {PLAN_SHA256}): V9 C1 if the V9 VIEWCOUNT_GAIN point estimate is >0, "
            f"else V9 C0; VIEWCOUNT_GAIN={gain!r}",
            "carrier_variant": variant,
            "carrier_qualification": "V9 carriers are DEV-round successors of the V7 C1 "
            "recipe (QUALIFIED on FRESH-V2); they are not independently qualified",
            "v9_preregistration_sha256": sha(V9 / "preregistration.json"),
            "qualification_results_sha256": sha(QUALIFICATION / "fresh_qualification_results.json"),
            "bounds_rule": "CONTEXT_DEPTH (the V7 C1 and V9 rule, from the warmup context only)",
            "carriers": carriers,
            "slow_carrier_updated": False,
        },
    )
    config = {
        "experiment": EXPERIMENT,
        "seeds": v9_config["seeds"],
        "policies": list(POLICIES),
        "controls": list(CONTROLS),
        "primary_contrast": f"BEYOND_BASELINE = AbsRel({baseline_policy}) - AbsRel(ALL)",
        "carrier_variant": variant,
        "baseline_policy": baseline_policy,
        "baseline_rule": f"{PLAN}: OFF for the V9 C1 carrier (seven views are inside its "
        "training range), OFF_CLAMP3 for the V9 C0 carrier; fixed before any V9 result",
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
        "cpu_affinity": list(range(8, 24)),
        "device": "cpu",
        "parallel_workers": 3,
        "slow_carrier": "frozen (requires_grad False); only the write rule is optimized",
        "write_rule_initialization": "torch.manual_seed(seed) then DirectWriteRule(); the "
        "initial rule is saved and evaluated as ALL_UNTRAINED",
        "checkpoint_selection": "none: the final step is evaluated; no DEV forward during training",
        "dev_reuse": "the 8 DEV scenes were evaluated in V2-V10; Core B V2 is a DEV mechanism "
        "round, not a qualification; FRESH-V1/V2 are not used",
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
            "per-seed gains; fast-weight norms; delta1",
            "interpretation": {
                "A": "BEYOND_BASELINE_STATUS and SPECIFICITY_STATUS SUPPORTED: learned writes "
                "beat the preregistered non-learned baseline with scene-specific information; "
                "next an independent qualification of the carrier and the writes together",
                "B": "BEYOND_BASELINE point estimate >0, not A",
                "C": "otherwise: the frozen write design does not beat the non-learned "
                "baseline on a view-count-robust carrier; next redesign the write",
            },
            "final_holdout": "never opened",
        },
    )
    write_json(
        root / "audit/environment.json",
        {
            "python": sys.version,
            "torch": torch.__version__,
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
        Path("scripts").glob("*rgbd_stream_write_v2*.py")
    )
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
        "FROZEN_CORE_B_V2",
        variant,
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
