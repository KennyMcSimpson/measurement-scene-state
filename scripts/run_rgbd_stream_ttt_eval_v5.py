#!/usr/bin/env python3
"""Freeze, run and analyze Core B V5 (Core B V4's TTT on the independent EVAL-V3 + EVAL-V4 cohorts).

  --stage freeze  writes the contracts and the preregistration (CPU, CUDA hidden);
  --stage run     on GPU 4 through cv_gpu_run.sh: the sealed evaluation of every seed, the analysis
                  with Core B V4's contrasts and gates, an independent recomputation of every
                  metric,
                  the normalized-readout description and the README.
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
from mcss.mechanism_pilot.geometry_carrier_experiment import read, verify_data
from mcss.mechanism_pilot.rgbd_stream_evaluation_v3 import preseal_depth_audit
from mcss.mechanism_pilot.rgbd_stream_ttt_eval_v5 import (
    COHORTS,
    EXPERIMENT,
    cohort_records,
    evaluate_ttt_cohorts,
)
from mcss.mechanism_pilot.rgbd_stream_ttt_v4 import (
    CONTROLS,
    POLICIES,
    RAYS_PER_FRAME,
    STREAM_LENGTH,
    load_frozen_carrier,
    stream_frame_ids,
)
from mcss.mechanism_pilot.small_training import sha, write_json
from mcss.mechanism_pilot.statistics import measurement_metrics

COREB_V4 = Path("outputs/EXP-3D-RGBD-DYNAMIC-TTT-V4")
V7 = Path("outputs/EXP-3D-RGBD-DEPTH-BOUNDS-V7")
V17 = Path("outputs/EXP-3D-RGBD-LONG-TRAIN-V17")
PLAN = "PLAN_BEFORE_COREB_V5.md"
PLAN_SHA256 = "748c95e84078237d264fe931d0858ee7d79fd4697ee768b42127d4f8ce0f8e6a"
SETTINGS = {"steps": 10, "learning_rate": 1e-3}
CARRIER_STEP = 24000
GPU = "4"
LAUNCHER = Path.home() / "cv_tools/cv_host_run.py"
CONCURRENT = ("EXP-3D-RGBD-SEGMENT-DIVERSITY-V21",)
THREADS = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "ATEN_CPU_CAPABILITY": "avx2",
    "MKL_CBWR": "AVX2",
    "DNNL_MAX_CPU_ISA": "AVX2",
    "NVIDIA_TF32_OVERRIDE": "0",
}
METRICS = ("rgb_mse", "depth_absrel", "depth_rmse", "depth_delta1", "opacity", "coverage")
KEYS = ("seed", "cohort", "scene_id", "role", "query_id", "policy")


def coreb_v4_runner():
    """Core B V4's runner: its analyze() holds the frozen contrasts, gates and branch rule."""
    path = Path("scripts/run_rgbd_stream_ttt_v4.py")
    spec = importlib.util.spec_from_file_location("coreb_v4_runner", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def resolve(path, base):
    path = Path(path)
    return path if path.is_absolute() else Path(base) / path


def load_cohorts(root):
    """{cohort: (manifest, base)} from the frozen cohort lock, every manifest hash-checked."""
    lock = read(Path(root) / "cohort_lock.json")
    cohorts = {}
    for name in COHORTS:
        item = lock[name]
        if sha(Path(item["manifest"])) != item["manifest_sha256"]:
            raise PermissionError(f"{name} manifest changed")
        cohorts[name] = (read(Path(item["manifest"])), Path(item["manifest"]).parent)
    return cohorts


def verify_lock_v5(root):
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
            skip = EXPERIMENT in path.parts or any(n in path.parts for n in CONCURRENT)
            if path.is_file() and not skip:
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


def freeze(root):
    root = Path(root).resolve()
    if root.name != EXPERIMENT or (root / "preregistration.json").exists():
        raise PermissionError("Exact, unfrozen Core B V5 directory required")
    if not (root / "PROTOCOL.md").is_file() or sha(root / PLAN) != PLAN_SHA256:
        raise PermissionError("The Core B V5 plan and protocol must be in place")
    if read(COREB_V4 / "integrity.json")["status"] != "PASS":
        raise PermissionError("Core B V4 must be finalized")
    if read(COREB_V4 / "ttt_results.json")["INTERPRETATION_BRANCH"] != "A":
        raise PermissionError("Core B V5 confirms a branch-A Core B V4")
    if read(COREB_V4 / "selection.json")["settings"] != SETTINGS:
        raise PermissionError("The TTT setting must be the one Core B V4 selected on TRAIN")
    if sha(LAUNCHER) != sha(Path("../_archives_cv/cv_host_run.py")):
        raise PermissionError("The path-mapping launcher must equal the migrated one")
    carriers = read(COREB_V4 / "carrier_lock.json")
    for seed, item in carriers["carriers"].items():
        if sha(Path(item["path"])) != item["sha256"] or item["step"] != CARRIER_STEP:
            raise PermissionError(f"V17 step-{CARRIER_STEP} carrier changed: {seed}")
        load_frozen_carrier(item["path"], "cpu")
    write_json(root / "carrier_lock.json", carriers)
    contract = read(V17 / "training_contract.json")["eval"]["cohorts"]
    lock = {}
    for name in COHORTS:
        entry = contract[name]
        if sha(Path(entry["manifest"])) != entry["manifest_sha256"]:
            raise PermissionError(f"{name} manifest is not V17's")
        lock[name] = {"manifest": entry["manifest"], "manifest_sha256": entry["manifest_sha256"]}
    write_json(root / "cohort_lock.json", lock)
    entries = cohort_records(load_cohorts(root))
    split = read(V7 / "scene_split.json")
    used = {r["scene_id"] for r in split["scenes"]} | set(split.get("protected_scene_ids", []))
    if used & set(entries):
        raise PermissionError("EVAL scenes must be disjoint from TRAIN72, DEV and the holdout")
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
            "rule": "Core B V1-V4 frame-id rule",
            "stream_length": STREAM_LENGTH,
            "episodes": episodes,
        },
    )
    config = {
        "experiment": EXPERIMENT,
        "seeds": sorted(int(s) for s in carriers["carriers"]),
        "policies": list(POLICIES),
        "controls": list(CONTROLS),
        "settings": SETTINGS,
        "settings_source": "Core B V4 selection.json (TRAIN only); no new selection",
        "rays_per_stream_frame": RAYS_PER_FRAME,
        "stream_length": STREAM_LENGTH,
        "cohorts": {name: len([s for s in entries.values() if s[0] == name]) for name in COHORTS},
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
            "weighted (Core B V4's analyze)",
            "bootstrap": {"unit": "scene", "draws": 10000, "seed": 20260928},
            "gate": "frozen V2 primary-contrast checks (as Core B V2-V4)",
            "TTT_BEYOND_BASELINE_STATUS": "gate on AbsRel(OFF) - AbsRel(TTT) (primary)",
            "TTT_SPECIFICITY_STATUS": "gate on AbsRel(TTT_WRONG_SCENE) - AbsRel(TTT) (primary)",
            "interpretation": {
                "A": "both primary gates SUPPORTED",
                "B": "TTT_BEYOND_BASELINE point estimate > 0, not A",
                "C": "otherwise",
            },
            "descriptive": "per seed, per cohort, normalized readout depth / max(opacity, 1e-3)",
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
    sources = sorted(Path("src/mcss").rglob("*.py")) + sorted(
        Path("scripts").glob("*rgbd_stream_ttt_eval_v5*.py")
    )
    sources += [
        Path("scripts/run_rgbd_stream_ttt_v4.py"),
        Path("scripts/analyze_rgbd_stream_write_v3.py"),
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
        "FROZEN_CORE_B_V5",
        len(entries),
        "scenes",
        len(episodes),
        "episodes",
        sealed,
        "files sealed",
    )


def summary(result):
    """The compact form of one Core B V4 analysis."""
    return {
        "absolute": {p: v["mean"] for p, v in result["absolute"].items()},
        "contrasts": {
            k: {
                "mean": v["mean"],
                "ci95": v["ci95"],
                "positive_tied_negative": [v["positive"], v["tied"], v["negative"]],
            }
            for k, v in result["contrasts"].items()
        },
        "statuses": result["statuses"],
        "INTERPRETATION_BRANCH": result["INTERPRETATION_BRANCH"],
        "per_seed_means": result["per_seed_means"],
    }


def recompute(root, cohorts, rows):
    """Every metric from the saved prediction and GT frame; the normalized readout; the seals."""
    entries = cohort_records(cohorts)
    worst, normalized = 0.0, []
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
        depth = pred["depth"] / np.maximum(pred["opacity"], 1e-3)
        norm = measurement_metrics(pred["rgb"], rgb, depth, gt, pred["opacity"])
        normalized.append({**{k: row[k] for k in KEYS}, "depth_absrel": norm["depth_absrel"]})
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
    return worst, normalized


def ttt_descriptives(root, seeds):
    first, last, norms = [], [], []
    for seed in seeds:
        index = read(Path(root) / "raw/eval_ttt" / f"seed_{seed}" / "state_hashes.json")
        for item in index.values():
            if item["policy"] == "TTT":
                first.append(item["ttt_stream_loss_first"])
                last.append(item["ttt_stream_loss_last"])
                norms.append(item["fast_norm"])
    drop = np.array(first) - np.array(last)
    return {
        "episodes": len(first),
        "stream_loss_first_median": float(np.median(first)),
        "stream_loss_last_median": float(np.median(last)),
        "share_of_episodes_with_lower_stream_loss": float((drop > 0).mean()),
        "fast_norm_median": float(np.median(norms)),
    }


def stat(value):
    lo, hi = value["ci95"]
    pos, tie, neg = value["positive_tied_negative"]
    return f"{value['mean']:+.4f} [{lo:+.4f}, {hi:+.4f}]，场景 {pos}/{tie}/{neg}"


def report(root, result):
    main, norm = result["primary"], result["normalized_readout"]
    c, s = main["contrasts"], main["statuses"]
    lines = [
        f"# {EXPERIMENT}",
        "",
        "Core B 第五轮：在 Core B 从未使用过的 EVAL-V3 ∪ EVAL-V4（83 个场景）上，"
        "用 Core B V4 在 TRAIN 上选出的同一个 TTT 设置（10 步，学习率 1e-3，不重新选择），"
        "确认基于梯度的测试时写入能否胜过只缓存新帧。"
        "计划见 [PLAN_BEFORE_COREB_V5.md](PLAN_BEFORE_COREB_V5.md)。"
        "全部在 GPU（4 号卡，关闭 TF32）上进行，不能逐位复现。",
        "",
        f"- **TTT_BEYOND_BASELINE（OFF − TTT）：** {stat(c['TTT_BEYOND_BASELINE'])}，"
        f"{s['TTT_BEYOND_BASELINE_STATUS']}",
        f"- **TTT_SPECIFICITY（TTT_WRONG_SCENE − TTT）：** {stat(c['TTT_SPECIFICITY'])}，"
        f"{s['TTT_SPECIFICITY_STATUS']}",
        f"- **分支 {main['INTERPRETATION_BRANCH']}**",
        f"- **描述性：** 缓存新帧（NO_STREAM − OFF）{stat(c['STREAM_CACHE_EFFECT'])}；"
        f"NO_STREAM − TTT {stat(c['TTT_STREAM_GAIN'])}",
        "",
        "| 策略 | query AbsRel（记录的读出） | 归一化读出 |",
        "|---|---|---|",
    ]
    for policy in (*POLICIES, *CONTROLS):
        lines.append(
            f"| {policy} | {main['absolute'][policy]:.4f} | {norm['absolute'][policy]:.4f} |"
        )
    lines += ["", "| 队列 | OFF − TTT | TTT_WRONG_SCENE − TTT |", "|---|---|---|"]
    for name, part in result["per_cohort"].items():
        pc = part["contrasts"]
        lines.append(
            f"| {name} | {stat(pc['TTT_BEYOND_BASELINE'])} | {stat(pc['TTT_SPECIFICITY'])} |"
        )
    nc = norm["contrasts"]
    ttt = result["ttt"]
    lower = 100 * ttt["share_of_episodes_with_lower_stream_loss"]
    lines += [
        "",
        f"- **归一化读出（描述性）：** OFF − TTT {stat(nc['TTT_BEYOND_BASELINE'])}；"
        f"TTT_WRONG_SCENE − TTT {stat(nc['TTT_SPECIFICITY'])}",
        "- **每个 seed（OFF − TTT）：** "
        + "，".join(
            f"{k}: {v['TTT_BEYOND_BASELINE']:+.4f}" for k, v in main["per_seed_means"].items()
        ),
        f"- **TTT：** 流帧自身损失的中位数 {ttt['stream_loss_first_median']:.4f} → "
        f"{ttt['stream_loss_last_median']:.4f}，{lower:.0f}% 的 episode 下降；"
        f"增量范数中位数 {ttt['fast_norm_median']:.3f}",
        "",
        "说明：EVAL-V3/V4 在 Core A 中用作评价队列，但从未用于 Core B，"
        "也从未用于 carrier 的训练或选择；"
        "DEV、FRESH-V1/V2 未使用；受保护的 final holdout 未打开。",
    ]
    (Path(root) / "README.md").write_text("\n".join(lines) + "\n")


def run(root):
    root = Path(root).resolve()
    config = verify_lock_v5(root)
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
        seed_rows = evaluate_ttt_cohorts(carrier, config["settings"], cohorts, out, seed, "cuda")
        rows.extend(seed_rows)
        print("EVALUATED", seed, len(seed_rows), flush=True)
    write_json(root / "raw/eval_ttt_query_results.json", rows)
    v4 = coreb_v4_runner()
    seeds = sorted(carriers)
    worst, normalized = recompute(root, cohorts, rows)
    result = {
        "primary": summary(v4.analyze(rows, seeds)),
        "per_cohort": {
            name: summary(v4.analyze([r for r in rows if r["cohort"] == name], seeds))
            for name in COHORTS
        },
        "normalized_readout": summary(v4.analyze(normalized, seeds)),
        "ttt": ttt_descriptives(root, seeds),
        "settings": config["settings"],
        "max_rgb_clamp_excess": max(r["rgb_clamp_max_excess"] for r in rows),
    }
    result["INTERPRETATION_BRANCH"] = result["primary"]["INTERPRETATION_BRANCH"]
    write_json(root / "ttt_eval_results.json", result)
    write_json(root / "raw/normalized_readout_rows.json", normalized)
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
        "CORE_B_V5_COMPLETE",
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
