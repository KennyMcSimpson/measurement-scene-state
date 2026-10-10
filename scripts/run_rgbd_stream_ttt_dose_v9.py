#!/usr/bin/env python3
"""Freeze, run and analyze Core B V9 (stream TTT dose 300 vs 100, TRAIN-EXT, GPU 4).

  --stage freeze  writes the contracts and the preregistration (CPU, CUDA hidden);
  --stage run     on GPU 4 through cv_gpu_run.sh: the sealed evaluation of every seed, an
                  independent recomputation of every metric, the normalized readout, the NEAR/FAR
                  split with model-free masks (checked against the GPU pilot's sealed TRAIN-EXT
                  masks), the comparison with Core B V6's NO_STREAM/OFF rows, the analysis and the
                  README.
Run from the repository root under cv_host_run.py (records store zonghan_o paths).
"""

from __future__ import annotations

import argparse
import datetime
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from mcss.dynamic.types import hash_scene_state
from mcss.mechanism_pilot.direct_capacity_statistics import describe, paired
from mcss.mechanism_pilot.geometry_carrier_experiment import read, verify_data
from mcss.mechanism_pilot.rgbd_stream_evaluation_v3 import preseal_depth_audit
from mcss.mechanism_pilot.rgbd_stream_ttt_context_v6 import cohort_records
from mcss.mechanism_pilot.rgbd_stream_ttt_eval_v5 import eval_context
from mcss.mechanism_pilot.rgbd_stream_ttt_v4 import (
    RAYS_PER_FRAME,
    STREAM_LENGTH,
    load_frozen_carrier,
    stream_frame_ids,
)
from mcss.mechanism_pilot.rgbd_stream_ttt_v9 import (
    CONTROLS,
    EXPERIMENT,
    LEARNING_RATE,
    POLICY_SPECS,
    evaluate_v9,
)
from mcss.mechanism_pilot.small_training import sha, write_json
from mcss.mechanism_pilot.statistics import measurement_metrics

COREB_V4 = Path("outputs/EXP-3D-RGBD-DYNAMIC-TTT-V4")
COREB_V6 = Path("outputs/EXP-3D-RGBD-DYNAMIC-TTT-CONTEXT-V6")
COREB_V7 = Path("outputs/EXP-3D-RGBD-DYNAMIC-TTT-FRESH-V7")
COREB_V8 = Path("outputs/EXP-3D-RGBD-DYNAMIC-TTT-SOURCE-V8")
V7_SPLIT = Path("outputs/EXP-3D-RGBD-DEPTH-BOUNDS-V7")
V17 = Path("outputs/EXP-3D-RGBD-LONG-TRAIN-V17")
V18 = Path("outputs/EXP-3D-RGBD-LONG-TRAIN-DATA-V18")
PILOT_SEAL = (
    Path.home()
    / "cv_work/gpu_pilot_v1/w32/EXP-3D-RGBD-LONG-TRAIN-V17/raw/eval/TRAIN_EXT/prediction_seal.json"
)
PLAN = "PLAN_BEFORE_COREB_V9.md"
PLAN_SHA256 = "8956b200a14d0f42907f692cbb36002acd87fc146832dc9920e1b9690ebb758d"
CARRIER_STEP = 24000
COHORT = "TRAIN_EXT"
EXPECTED_SPLIT = {COHORT: "TRAIN"}
USED_COHORTS = ("EVAL_FAR", "EVAL_V3", "EVAL_V4", "FRESH_V2")
SPECS = dict(POLICY_SPECS)
BASE_POLICIES = ("NO_STREAM", "OFF")
GPU = "4"
LAUNCHER = Path.home() / "cv_tools/cv_host_run.py"
THREADS = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "ATEN_CPU_CAPABILITY": "avx2",
    "MKL_CBWR": "AVX2",
    "DNNL_MAX_CPU_ISA": "AVX2",
    "NVIDIA_TF32_OVERRIDE": "0",
}
CONTRASTS = {
    "STREAM_DOSE_300": ("TTT_100", "TTT_300"),
    "SUPERVISION_SOURCE_100": ("OFF_CTX_TTT_100", "TTT_100"),
    "STREAM_TTT_100": ("OFF", "TTT_100"),
    "STREAM_TTT_300": ("OFF", "TTT_300"),
    "CONTEXT_TTT_WITH_STREAM_100": ("OFF", "OFF_CTX_TTT_100"),
    "TTT_300_SPECIFICITY": ("TTT_300_WRONG_SCENE", "TTT_300"),
    "STREAM_CACHE_EFFECT": ("NO_STREAM", "OFF"),
}
PRIMARY = {
    "P1_STREAM_DOSE_300_STATUS": "STREAM_DOSE_300",
    "P2_SUPERVISION_SOURCE_100_STATUS": "SUPERVISION_SOURCE_100",
}
DRAWS, BOOT_SEED = 10000, 20260928
METRICS = ("rgb_mse", "depth_absrel", "depth_rmse", "depth_delta1", "opacity", "coverage")
KEYS = ("seed", "cohort", "scene_id", "role", "query_id", "policy")
SCRIPTS = Path("scripts")


def module(name, filename):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / filename)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def v3_analysis():
    """Core B V3's scene aggregation and frozen V2 gate (as in Core B V4-V8)."""
    return module("coreb_v3_analysis_v9", "analyze_rgbd_stream_write_v3.py")


def resolve(path, base):
    path = Path(path)
    return path if path.is_absolute() else Path(base) / path


def load_cohorts(root):
    """{cohort: (manifest, base)} from the frozen cohort lock, the manifest hash-checked."""
    item = read(Path(root) / "cohort_lock.json")[COHORT]
    if sha(Path(item["manifest"])) != item["manifest_sha256"]:
        raise PermissionError(f"{COHORT} manifest changed")
    return {COHORT: (read(Path(item["manifest"])), Path(item["manifest"]).parent)}


def verify_lock_v9(root):
    """The preregistration's input and source hashes (verify_lock without a TRAIN/DEV split)."""
    lock = read(Path(root) / "preregistration.json")
    if lock["status"] != "FROZEN_BEFORE_FORMAL_TRAINING":
        raise PermissionError("Formal preregistration absent")
    for section in ("input_sha256", "source_sha256"):
        for name, digest in lock[section].items():
            if sha(Path(name)) != digest:
                raise PermissionError(f"Frozen input/source changed: {name}")
    return read(Path(root) / "evaluation_contract.json")


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


def used_scene_ids():
    """TRAIN72, DEV and the holdout (V7 split), V17's evaluation cohorts and EVAL-V5."""
    split = read(V7_SPLIT / "scene_split.json")
    used = {r["scene_id"] for r in split["scenes"]} | set(split.get("protected_scene_ids", []))
    contract = read(V17 / "training_contract.json")["eval"]["cohorts"]
    for name in USED_COHORTS:
        entry = contract[name]
        if sha(Path(entry["manifest"])) != entry["manifest_sha256"]:
            raise PermissionError(f"{name} manifest is not V17's")
        used |= {r["scene_id"] for r in read(Path(entry["manifest"]))["scenes"]}
    fresh = read(COREB_V7 / "cohort_lock.json")["EVAL_V5"]
    if sha(Path(fresh["manifest"])) != fresh["manifest_sha256"]:
        raise PermissionError("EVAL-V5 manifest changed")
    return used | {r["scene_id"] for r in read(Path(fresh["manifest"]))["scenes"]}


def freeze(root):
    root = Path(root).resolve()
    if root.name != EXPERIMENT or (root / "preregistration.json").exists():
        raise PermissionError("Exact, unfrozen Core B V9 directory required")
    if not (root / "PROTOCOL.md").is_file() or sha(root / PLAN) != PLAN_SHA256:
        raise PermissionError("The Core B V9 plan and protocol must be in place")
    for done in (COREB_V4, COREB_V6, COREB_V7, COREB_V8):
        if read(done / "integrity.json")["status"] != "PASS":
            raise PermissionError(f"{done.name} must be finalized")
    if read(COREB_V4 / "selection.json")["settings"]["learning_rate"] != LEARNING_RATE:
        raise PermissionError("TTT must use the learning rate Core B V4 selected on TRAIN")
    if sha(LAUNCHER) != sha(Path("../_archives_cv/cv_host_run.py")):
        raise PermissionError("The path-mapping launcher must equal the migrated one")
    carriers = read(COREB_V4 / "carrier_lock.json")
    for seed, item in carriers["carriers"].items():
        if sha(Path(item["path"])) != item["sha256"] or item["step"] != CARRIER_STEP:
            raise PermissionError(f"V17 step-{CARRIER_STEP} carrier changed: {seed}")
        load_frozen_carrier(item["path"], "cpu")
    write_json(root / "carrier_lock.json", carriers)
    entry = read(V18 / "training_contract.json")["train_ext"]
    if sha(Path(entry["manifest"])) != entry["manifest_sha256"]:
        raise PermissionError("TRAIN-EXT manifest is not the one V18 trained on")
    if read(COREB_V6 / "cohort_lock.json")[COHORT]["manifest_sha256"] != entry["manifest_sha256"]:
        raise PermissionError("TRAIN-EXT must be Core B V6's cohort")
    lock = {COHORT: {"manifest": entry["manifest"], "manifest_sha256": entry["manifest_sha256"]}}
    write_json(root / "cohort_lock.json", lock)
    entries = cohort_records(load_cohorts(root), EXPECTED_SPLIT)
    if used_scene_ids() & set(entries):
        raise PermissionError(
            "TRAIN-EXT must be disjoint from TRAIN72, DEV, the holdout, V17's evaluation cohorts "
            "and EVAL-V5"
        )
    episodes = {}
    for sid, (_, record, _, _) in sorted(entries.items()):
        for role in ("A", "B"):
            ids = stream_frame_ids(record, role)
            if ids is None:
                raise PermissionError(f"{sid}/{role} lacks a full stream")
            episodes[f"{sid}/{role}"] = list(ids)
    write_json(
        root / "stream_roles.json",
        {
            "rule": "Core B V1-V8 frame-id rule",
            "stream_length": STREAM_LENGTH,
            "episodes": episodes,
        },
    )
    config = {
        "experiment": EXPERIMENT,
        "seeds": sorted(int(s) for s in carriers["carriers"]),
        "policies": {name: list(spec) for name, spec in SPECS.items()},
        "controls": CONTROLS,
        "learning_rate": LEARNING_RATE,
        "settings_source": "learning rate: Core B V4 selection (TRAIN); 100 steps: V23 dose arms "
        "and Core B V7/V8; 300 steps: fixed by the plan before any 300-step computation; no "
        "selection on TRAIN-EXT",
        "rays_per_supervised_frame": RAYS_PER_FRAME,
        "stream_length": STREAM_LENGTH,
        "cohort": {
            "name": COHORT,
            "scenes": len(entries),
            "expected_split": EXPECTED_SPLIT[COHORT],
        },
        "near_masks": "V11 geometry + INPAINT-V1 near_mask from the role's context depth "
        "(model-free), checked against the GPU pilot's sealed TRAIN-EXT masks",
        "pilot_mask_check": {
            "path": str(PILOT_SEAL),
            "sha256": sha(PILOT_SEAL) if PILOT_SEAL.is_file() else None,
        },
        "num_threads": 1,
        "thread_environment": THREADS,
        "cpu_affinity": list(range(56, 64)),
        "device": "cuda",
        "gpu": GPU,
        "host_launcher": {"path": str(LAUNCHER), "sha256": sha(LAUNCHER)},
    }
    write_json(root / "evaluation_contract.json", config)
    write_json(
        root / "decision_rules.json",
        {
            "aggregation": "per seed: scene mean of role means of query AbsRel; seeds equally "
            "weighted",
            "bootstrap": {"unit": "scene", "draws": DRAWS, "seed": BOOT_SEED},
            "gate": "frozen V2 primary-contrast checks (as Core B V2-V8)",
            "P1_STREAM_DOSE_300_STATUS": "gate on AbsRel(TTT_100) - AbsRel(TTT_300)",
            "P2_SUPERVISION_SOURCE_100_STATUS": "gate on AbsRel(OFF_CTX_TTT_100) - AbsRel(TTT_100)",
            "interpretation": {
                "A": "both SUPPORTED",
                "B": "only P1 SUPPORTED",
                "C": "only P2 SUPPORTED",
                "D": "neither SUPPORTED",
            },
            "descriptive": list(CONTRASTS),
            "final_holdout": "never opened",
        },
    )
    sealed = seal_previous(root)
    data_hashes = {}
    for _, record, _, base in entries.values():
        for frame in record["frames"]:
            for key in ("rgb", "depth"):
                path = resolve(frame[key], base)
                data_hashes[str(path)] = sha(path)
    write_json(root / "audit/data_hashes.json", data_hashes)
    names = (
        "carrier_lock.json",
        "cohort_lock.json",
        "stream_roles.json",
        "evaluation_contract.json",
        "decision_rules.json",
        "PROTOCOL.md",
        PLAN,
        "audit/data_hashes.json",
        "audit/previous_experiment_seal.json",
    )
    sources = sorted(Path("src/mcss").rglob("*.py"))
    sources += sorted(SCRIPTS.glob("*rgbd_stream_ttt_dose_v9*.py"))
    sources += [
        SCRIPTS / "analyze_rgbd_stream_write_v3.py",
        SCRIPTS / "run_rgbd_geometric_baselines.py",
        SCRIPTS / "run_rgbd_inpaint_baselines.py",
        LAUNCHER,
    ]
    write_json(
        root / "preregistration.json",
        {
            "status": "FROZEN_BEFORE_FORMAL_TRAINING",
            "experiment": EXPERIMENT,
            "input_sha256": {str((root / n).resolve()): sha(root / n) for n in names},
            "source_sha256": {str(p.resolve()): sha(p) for p in sources},
            "data_sha256": data_hashes,
            "formal_training_started": False,
        },
    )
    print(
        "FROZEN_CORE_B_V9",
        len(entries),
        "scenes",
        len(episodes),
        "episodes",
        sealed,
        "files sealed",
    )


def analyze(rows, seeds, primary=True):
    """Every contrast with Core B V3's aggregation and the frozen V2 gate; the plan's branch."""
    v3 = v3_analysis()
    policies = sorted({r["policy"] for r in rows})
    absolute = {
        p: describe(v3.scene_values(rows, p), draws=DRAWS, seed=BOOT_SEED)["mean"] for p in policies
    }
    contrasts, statuses = {}, {}
    for name, (a, b) in CONTRASTS.items():
        stats = paired(
            v3.scene_values(rows, a), v3.scene_values(rows, b), draws=DRAWS, seed=BOOT_SEED
        )
        status, _ = v3.gate(stats, True)
        contrasts[name] = {
            "mean": stats["mean"],
            "ci95": stats["ci95"],
            "positive_tied_negative": [stats["positive"], stats["tied"], stats["negative"]],
            "gate_descriptive": status,
        }
        for key, value in PRIMARY.items():
            if value == name:
                statuses[key] = status
    p1 = statuses["P1_STREAM_DOSE_300_STATUS"] == "SUPPORTED"
    p2 = statuses["P2_SUPERVISION_SOURCE_100_STATUS"] == "SUPPORTED"
    branch = {(True, True): "A", (True, False): "B", (False, True): "C", (False, False): "D"}[
        p1, p2
    ]
    per_seed = {
        str(seed): {
            name: float(np.mean(list(v3.scene_values(rows, a, seed=seed).values())))
            - float(np.mean(list(v3.scene_values(rows, b, seed=seed).values())))
            for name, (a, b) in CONTRASTS.items()
        }
        for seed in seeds
    }
    return {
        "absolute": absolute,
        "contrasts": contrasts,
        "statuses": statuses if primary else {},
        "INTERPRETATION_BRANCH": branch if primary else None,
        "per_seed_means": per_seed,
    }


def near_masks(cohorts):
    """{(scene, role, query): NEAR mask} from each role's measured context depth (model-free)."""
    geo = module("v9_geometry", "run_rgbd_geometric_baselines.py")
    inp = module("v9_inpaint", "run_rgbd_inpaint_baselines.py")
    masks = {}
    for _, record, manifest, base in cohort_records(cohorts, EXPECTED_SPLIT).values():
        for role, context in eval_context(record, manifest, base, "cpu", []).items():
            points = geo.context_points(context)
            for qid in record["roles"]["primary_query"]:
                camera = geo.query_camera(record, qid, manifest["image_size"])
                _, hit = geo.reproject(points, camera)
                masks[record["scene_id"], role, qid] = inp.near_mask(hit)
    return masks


def pilot_mask_agreement(masks, frozen):
    """How many masks equal the GPU pilot's sealed TRAIN-EXT masks (all present ones must)."""
    if frozen["sha256"] is None:
        return {"compared": 0, "identical": 0}
    if sha(PILOT_SEAL) != frozen["sha256"]:
        raise PermissionError("The GPU pilot's TRAIN-EXT seal changed")
    seal = read(PILOT_SEAL)["predictions"]
    compared = identical = 0
    for (sid, role, qid), mask in masks.items():
        item = seal.get(f"{COHORT}/{sid}/{role}/{qid}")
        if item is None:
            continue
        if sha(Path(item["path"])) != item["sha256"]:
            raise PermissionError("A pilot sealed prediction changed")
        compared += 1
        identical += bool(np.array_equal(np.load(item["path"])["near"].astype(bool), mask))
    if identical != compared:
        raise AssertionError(f"Only {identical} of {compared} NEAR masks equal the pilot's")
    return {"compared": compared, "identical": identical}


def row_agreement(rows, theirs):
    """NO_STREAM/OFF AbsRel matched to another round's rows by seed, scene, role, query, policy."""

    def values(items):
        return {
            tuple(r[k] for k in KEYS): r["depth_absrel"]
            for r in items
            if r["policy"] in BASE_POLICIES
        }

    mine, other = values(rows), values(theirs)
    diffs = [abs(mine[k] - other[k]) for k in sorted(set(mine) & set(other))]
    return {
        "rows": len(mine),
        "rows_compared": len(diffs),
        "rows_identical": sum(d == 0 for d in diffs),
        "max_abs_difference": max(diffs, default=0.0),
    }


def v6_agreement(root, rows, seeds):
    """NO_STREAM and OFF (no TTT, Core B V6's episode ids) against Core B V6's rows and states."""
    path = COREB_V6 / "raw/eval_ttt_query_results.json"
    result = row_agreement(rows, read(path) if path.is_file() else [])
    compared = identical = 0
    for seed in seeds:
        old = COREB_V6 / "raw/eval_ttt" / f"seed_{seed}" / "state_hashes.json"
        if not old.is_file():
            continue
        theirs = read(old)
        index = read(Path(root) / "raw/eval_ttt" / f"seed_{seed}" / "state_hashes.json")
        for key, item in index.items():
            if item["policy"] in BASE_POLICIES and key in theirs:
                compared += 1
                identical += item["state_hash"] == theirs[key]["state_hash"]
    return {**result, "states_compared": compared, "states_identical": identical}


def recompute(root, cohorts, rows, masks):
    """Every metric from the saved prediction and GT; NORM and NEAR/FAR rows; the seals."""
    entries = cohort_records(cohorts, EXPECTED_SPLIT)
    worst = 0.0
    derived = {"NORM": [], "FAR": [], "NEAR": []}
    for row in rows:
        path = Path(root) / "raw/eval_ttt" / f"seed_{row['seed']}" / row["prediction_path"]
        if sha(path) != row["prediction_file_sha256"]:
            raise AssertionError("Prediction changed")
        pred = dict(np.load(path))
        _, record, _, base = entries[row["scene_id"]]
        frame = next(f for f in record["frames"] if f["frame_id"] == row["query_id"])
        with Image.open(resolve(frame["rgb"], base)) as image:
            rgb = np.array(image.convert("RGB"), dtype=np.float32) / 255
        gt = np.load(resolve(frame["depth"], base)).squeeze()
        metrics = measurement_metrics(pred["rgb"], rgb, pred["depth"], gt, pred["opacity"])
        for key in METRICS:
            worst = max(worst, abs(metrics[key] - row[key]))
        common = {k: row[k] for k in KEYS}
        depth = pred["depth"] / np.maximum(pred["opacity"], 1e-3)
        norm = measurement_metrics(pred["rgb"], rgb, depth, gt, pred["opacity"])
        derived["NORM"].append({**common, "depth_absrel": norm["depth_absrel"]})
        valid = np.isfinite(gt) & (gt > 0)
        near = masks[row["scene_id"], row["role"], row["query_id"]]
        for region, mask in (("FAR", valid & ~near), ("NEAR", valid & near)):
            if mask.any():
                value = np.abs(pred["depth"][mask] - gt[mask]) / gt[mask]
                derived[region].append({**common, "depth_absrel": float(value.mean())})
    if worst > 1e-6:
        raise AssertionError(f"Metric mismatch {worst}")
    records = {sid: item[1] for sid, item in entries.items()}
    for seed in {r["seed"] for r in rows}:
        out = Path(root) / "raw/eval_ttt" / f"seed_{seed}"
        audit = read(out / "integrity.json")
        if sha(out / "states.pt") != audit["states_file_sha256"]:
            raise AssertionError("Sealed states changed")
        states = torch.load(out / "states.pt", map_location="cpu", weights_only=False)
        index = read(out / "state_hashes.json")
        if any(hash_scene_state(s) != index[k]["state_hash"] for k, s in states.items()):
            raise AssertionError("State hash mismatch")
        access = read(out / "GT_access.json")
        marker = next(
            i for i, e in enumerate(access) if e.get("event") == "ALL_STREAM_STATES_SEALED"
        )
        preseal_depth_audit(access[:marker], records)
    return worst, derived


def ttt_descriptives(root, seeds):
    out = {}
    for policy, (_, supervision, _) in SPECS.items():
        if supervision is None:
            continue
        first, last, norms = [], [], []
        for seed in seeds:
            index = read(Path(root) / "raw/eval_ttt" / f"seed_{seed}" / "state_hashes.json")
            for item in index.values():
                if item["policy"] == policy:
                    first.append(item["ttt_loss_first"])
                    last.append(item["ttt_loss_last"])
                    norms.append(item["fast_norm"])
        drop = np.array(first) - np.array(last)
        out[policy] = {
            "episodes": len(first),
            "loss_first_median": float(np.median(first)),
            "loss_last_median": float(np.median(last)),
            "share_with_lower_loss": float((drop > 0).mean()),
            "fast_norm_median": float(np.median(norms)),
        }
    return out


def stat(value):
    lo, hi = value["ci95"]
    pos, tie, neg = value["positive_tied_negative"]
    return f"{value['mean']:+.4f} [{lo:+.4f}, {hi:+.4f}]，场景 {pos}/{tie}/{neg}"


def report(root, result):
    main = result["primary"]
    c, s = main["contrasts"], main["statuses"]
    v6 = result["core_b_v6_agreement"]
    lines = [
        f"# {EXPERIMENT}",
        "",
        "Core B 第九轮：流帧写入从 100 步加到 300 步是否继续改善，以及 100 步时新帧监督"
        "胜过上下文监督（Core B V8 的 P2）能否在另一个队列上重现。"
        "队列是 TRAIN-EXT（55 个场景，V17 L 没有在上面训练过；"
        "Core B V6 只在上面算过 10 步的策略）。"
        "计划见 [PLAN_BEFORE_COREB_V9.md](PLAN_BEFORE_COREB_V9.md)。"
        "全部在 GPU（4 号卡，关闭 TF32）上进行，不能逐位复现。",
        "",
        f"- **P1 STREAM_DOSE_300（TTT_100 − TTT_300）：** {stat(c['STREAM_DOSE_300'])}，"
        f"{s['P1_STREAM_DOSE_300_STATUS']}",
        "- **P2 SUPERVISION_SOURCE_100（OFF_CTX_TTT_100 − TTT_100）：** "
        f"{stat(c['SUPERVISION_SOURCE_100'])}，{s['P2_SUPERVISION_SOURCE_100_STATUS']}",
        f"- **分支 {main['INTERPRETATION_BRANCH']}**",
        f"- **超出只缓存的增益：** OFF − TTT_100 {stat(c['STREAM_TTT_100'])}；"
        f"OFF − TTT_300 {stat(c['STREAM_TTT_300'])}",
        f"- **上下文监督本身（OFF − OFF_CTX_TTT_100）：** {stat(c['CONTEXT_TTT_WITH_STREAM_100'])}",
        f"- **特异性（TTT_300_WRONG_SCENE − TTT_300）：** {stat(c['TTT_300_SPECIFICITY'])}",
        f"- **缓存新帧（NO_STREAM − OFF）：** {stat(c['STREAM_CACHE_EFFECT'])}",
        "",
        "| 策略 | query AbsRel | 归一化读出 | NEAR | FAR |",
        "|---|---|---|---|---|",
    ]
    for policy in (*SPECS, *CONTROLS):
        lines.append(
            f"| {policy} | {main['absolute'][policy]:.4f} | "
            f"{result['normalized_readout']['absolute'][policy]:.4f} | "
            f"{result['regions']['NEAR']['absolute'][policy]:.4f} | "
            f"{result['regions']['FAR']['absolute'][policy]:.4f} |"
        )
    lines += [
        "",
        "| 分组（描述性） | STREAM_DOSE_300 | SUPERVISION_SOURCE_100 | STREAM_TTT_300 |",
        "|---|---|---|---|",
    ]
    for name, part in (
        ("归一化读出", result["normalized_readout"]),
        ("NEAR", result["regions"]["NEAR"]),
        ("FAR", result["regions"]["FAR"]),
    ):
        pc = part["contrasts"]
        lines.append(
            f"| {name} | {stat(pc['STREAM_DOSE_300'])} | {stat(pc['SUPERVISION_SOURCE_100'])} | "
            f"{stat(pc['STREAM_TTT_300'])} |"
        )
    lines += [
        "",
        "- **每个 seed：** "
        + "；".join(
            f"{k}: STREAM_DOSE_300 {v['STREAM_DOSE_300']:+.4f}，"
            f"SUPERVISION_SOURCE_100 {v['SUPERVISION_SOURCE_100']:+.4f}"
            for k, v in main["per_seed_means"].items()
        ),
        "- **被监督帧自身的损失（中位数）：** "
        + "；".join(
            f"{p} {v['loss_first_median']:.4f} → {v['loss_last_median']:.4f}"
            f"（{100 * v['share_with_lower_loss']:.0f}% 下降）"
            for p, v in result["ttt"].items()
        ),
        f"- **NEAR 掩码：** 与 GPU 试点在 TRAIN-EXT 上封存的掩码比较 "
        f"{result['pilot_mask_agreement']['compared']} 个，"
        f"{result['pilot_mask_agreement']['identical']} 个完全相同。",
        f"- **与 Core B V6 的一致性（NO_STREAM、OFF，描述性）：** {v6['rows_compared']} 行中 "
        f"{v6['rows_identical']} 行 AbsRel 完全相同，最大差 {v6['max_abs_difference']:.2e}；"
        f"{v6['states_compared']} 个状态中 {v6['states_identical']} 个哈希相同。",
        "",
        "说明：TRAIN-EXT 在 V18、V20 中用作训练数据，在 GPU 试点与 Core B V6 中用作评价；"
        "V17 L 从未在上面训练过。受保护的 final holdout 未打开。",
    ]
    (Path(root) / "README.md").write_text("\n".join(lines) + "\n")


def run(root):
    root = Path(root).resolve()
    config = verify_lock_v9(root)
    verify_data(root)
    if read(root / "tests.json")["status"] != "PASS":
        raise PermissionError("Full suite must PASS before the run")
    if not os.sched_getaffinity(0) <= set(config["cpu_affinity"]):
        raise PermissionError("Run must be pinned to the frozen CPU cores (taskset)")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != config["gpu"]:
        raise PermissionError("Run must hold the frozen GPU through cv_gpu_run.sh")
    if sha(LAUNCHER) != config["host_launcher"]["sha256"] or not os.environ.get("CV_ROOT"):
        raise PermissionError("The frozen launcher and CV_ROOT are required")
    for key, value in config["thread_environment"].items():
        if os.environ.get(key) != value:
            raise PermissionError(f"Thread environment {key} must be {value}")
    torch.set_num_threads(1)
    lock = read(root / "carrier_lock.json")["carriers"]
    carriers = {int(s): load_frozen_carrier(item["path"], "cuda")[0] for s, item in lock.items()}
    cohorts = load_cohorts(root)
    rows = []
    for seed, carrier in sorted(carriers.items()):
        out = root / "raw/eval_ttt" / f"seed_{seed}"
        seed_rows = evaluate_v9(
            carrier, cohorts, EXPECTED_SPLIT, out, seed, "cuda", SPECS, CONTROLS
        )
        rows.extend(seed_rows)
        print("EVALUATED", seed, len(seed_rows), flush=True)
    write_json(root / "raw/eval_ttt_query_results.json", rows)
    masks = near_masks(cohorts)
    agreement = pilot_mask_agreement(masks, config["pilot_mask_check"])
    worst, derived = recompute(root, cohorts, rows, masks)
    seeds = sorted(carriers)
    result = {
        "primary": analyze(rows, seeds),
        "normalized_readout": analyze(derived["NORM"], seeds, primary=False),
        "regions": {
            region: analyze(derived[region], seeds, primary=False) for region in ("NEAR", "FAR")
        },
        "pilot_mask_agreement": agreement,
        "core_b_v6_agreement": v6_agreement(root, rows, seeds),
        "ttt": ttt_descriptives(root, seeds),
        "max_rgb_clamp_excess": max(r["rgb_clamp_max_excess"] for r in rows),
    }
    result["INTERPRETATION_BRANCH"] = result["primary"]["INTERPRETATION_BRANCH"]
    write_json(root / "ttt_dose_results.json", result)
    for name, values in derived.items():
        write_json(root / f"raw/{name.lower()}_rows.json", values)
    old = read(root / "audit/previous_experiment_seal.json")["files"]
    changed = [p for p, info in old.items() if sha(Path(p)) != info["sha256"]]
    if changed:
        raise AssertionError(f"Old experiment files changed: {changed[:3]}")
    write_json(
        root / "integrity.json",
        {
            "status": "PASS",
            "metrics_recomputed": len(rows),
            "max_metric_error": worst,
            "old_files_unchanged": len(old),
            "settings_reselected": False,
            "slow_carrier_updated": False,
            "final_holdout_touched": False,
            "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "finished_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        },
    )
    report(root, result)
    print(
        "CORE_B_V9_COMPLETE",
        result["INTERPRETATION_BRANCH"],
        json.dumps(result["primary"]["statuses"]),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--stage", choices=("freeze", "run"), required=True)
    args = parser.parse_args()
    if args.stage == "freeze":
        freeze(args.root)
    else:
        run(args.root)
    sys.exit(0)
