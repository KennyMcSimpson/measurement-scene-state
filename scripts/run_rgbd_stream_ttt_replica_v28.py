#!/usr/bin/env python3
"""Freeze, run and analyze V28 (stream TTT under cross-dataset shift, Replica, GPU 4 + CPU).

  --stage freeze  writes the contracts and the preregistration (CPU, CUDA hidden);
  --stage run     on GPU 4 through cv_gpu_run.sh: Core B V8's sealed evaluation of every seed,
                  an independent recomputation of every metric, the carrier-internal analysis, then
                  (CPU) V27's 3- and 7-frame baselines and hybrids on this round's sealed
                  predictions, the constant predictor, the check against REPLICA-TRANSFER-V1's rows,
                  and the README.
Run from the repository root under cv_host_run.py. Replica data stays on this host (research only).
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
from mcss.mechanism_pilot.rgbd_stream_ttt_v4 import (
    RAYS_PER_FRAME,
    STREAM_LENGTH,
    load_frozen_carrier,
    stream_frame_ids,
)
from mcss.mechanism_pilot.rgbd_stream_ttt_v7 import LEARNING_RATE
from mcss.mechanism_pilot.rgbd_stream_ttt_v8 import evaluate_v8
from mcss.mechanism_pilot.small_training import sha, write_json
from mcss.mechanism_pilot.statistics import measurement_metrics

EXPERIMENT = "EXP-3D-RGBD-DYNAMIC-TTT-REPLICA-V28"
COREB_V4 = Path("outputs/EXP-3D-RGBD-DYNAMIC-TTT-V4")
COREB_V8 = Path("outputs/EXP-3D-RGBD-DYNAMIC-TTT-SOURCE-V8")
V17 = Path("outputs/EXP-3D-RGBD-LONG-TRAIN-V17")
REPLICA = Path("outputs/EXP-3D-RGBD-REPLICA-DATA-V1")
TRANSFER = Path("outputs/EXP-3D-RGBD-REPLICA-TRANSFER-V1")
CONCURRENT = (
    "EXP-3D-RGBD-DYNAMIC-TTT-CARRIER-V10",
    "EXP-3D-RGBD-DYNAMIC-TTT-ONLINE-V11",
    "EXP-3D-RGBD-FAR-FOCUS-V26",
)
PLAN = "PLAN_BEFORE_V28.md"
PLAN_SHA256 = "8b154c923015e71bcdc8d35b7ab10cd48effe4cb9c2b9eb3f61bf84141c41174"
CARRIER_STEP = 24000
COHORT = "REPLICA"
EXPECTED_SPLIT = {COHORT: "REPLICA"}
POLICY_SPECS = {
    "NO_STREAM": (False, None, 0),
    "OFF": (True, None, 0),
    "OFF_CTX_TTT_100": (True, "context", 100),
    "TTT_100": (True, "stream", 100),
    "TTT_1000": (True, "stream", 1000),
}
CONTROLS = {"TTT_100_WRONG_SCENE": "TTT_100"}
SPECS = dict(POLICY_SPECS)
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
    "WRITE_UNDER_SHIFT": ("OFF", "TTT_100"),
    "STREAM_DOSE_1000": ("TTT_100", "TTT_1000"),
    "WRITE_1000_UNDER_SHIFT": ("OFF", "TTT_1000"),
    "CONTEXT_TTT_WITH_STREAM_100": ("OFF", "OFF_CTX_TTT_100"),
    "TTT_100_SPECIFICITY": ("TTT_100_WRONG_SCENE", "TTT_100"),
    "STREAM_CACHE_EFFECT": ("NO_STREAM", "OFF"),
}
HYBRID_CONTRASTS = {
    "HYBRID_VS_MODEL_FREE": ("REPROJ7_HARMONIC", "HFILL8_7_TTT_100"),
    "AHFILL8_VS_MODEL_FREE": ("REPROJ7_HARMONIC", "AHFILL8_7_TTT_100"),
    "HYBRID_NORM_VS_MODEL_FREE": ("REPROJ7_HARMONIC", "HFILL8_7N_TTT_100"),
    "TTT_GAIN_IN_HYBRID": ("HFILL8_7_OFF", "HFILL8_7_TTT_100"),
    "CARRIER_TTT_100_VS_MODEL_FREE": ("REPROJ7_HARMONIC", "CARRIER_TTT_100"),
    "CARRIER_TTT_100_VS_CONSTANT": ("CONSTANT", "CARRIER_TTT_100"),
    "MODEL_FREE_VS_CONSTANT": ("CONSTANT", "REPROJ7_HARMONIC"),
    "STREAM_FRAMES_FOR_MODEL_FREE": ("REPROJ3_HARMONIC", "REPROJ7_HARMONIC"),
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
    """Core B V3's scene aggregation and frozen V2 gate (as in Core B V4-V11 and V24-V27)."""
    return module("coreb_v3_analysis_v28", "analyze_rgbd_stream_write_v3.py")


def hybrid_module(root):
    """V27's frozen baseline and hybrid code with this round as its only source."""
    loaded = module("v27_hybrid_for_v28", "analyze_rgbd_hybrid_confirm_v27.py")
    loaded.ROUNDS = {COHORT: Path(root)}
    loaded.EXPECTED_SPLIT = dict(EXPECTED_SPLIT)
    loaded.V17 = V17
    return loaded


def resolve(path, base):
    path = Path(path)
    return path if path.is_absolute() else Path(base) / path


def load_cohorts(root):
    item = read(Path(root) / "cohort_lock.json")[COHORT]
    if sha(Path(item["manifest"])) != item["manifest_sha256"]:
        raise PermissionError(f"{COHORT} manifest changed")
    return {COHORT: (read(Path(item["manifest"])), Path(item["manifest"]).parent)}


def verify_lock_v28(root):
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
            if path.is_file() and not {EXPERIMENT, *CONCURRENT} & set(path.parts):
                files[str(path)] = {"sha256": sha(path), "bytes": path.stat().st_size}
    write_json(
        root / "audit/previous_experiment_seal.json",
        {
            "created_utc": datetime.datetime.now(datetime.UTC).isoformat(),
            "excluded_concurrent": list(CONCURRENT),
            "files": files,
        },
    )
    return len(files)


def replica_manifest():
    """The prepared Replica manifest, its preparation integrity PASS."""
    if read(REPLICA / "preparation_integrity.json")["status"] != "PASS":
        raise PermissionError("Replica preparation must PASS")
    path = (REPLICA / "manifest_replica.json").resolve()
    return {"manifest": str(path), "manifest_sha256": sha(path)}


def freeze(root):
    root = Path(root).resolve()
    if root.name != EXPERIMENT or (root / "preregistration.json").exists():
        raise PermissionError("Exact, unfrozen V28 directory required")
    if not (root / "PROTOCOL.md").is_file() or sha(root / PLAN) != PLAN_SHA256:
        raise PermissionError("The V28 plan and protocol must be in place")
    for done in (COREB_V4, COREB_V8):
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
    write_json(root / "cohort_lock.json", {COHORT: replica_manifest()})
    entries = cohort_records(load_cohorts(root), EXPECTED_SPLIT)
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
            "rule": "Core B V1-V11 frame-id rule",
            "stream_length": STREAM_LENGTH,
            "episodes": episodes,
        },
    )
    eval_contract = read(V17 / "training_contract.json")["eval"]
    config = {
        "experiment": EXPERIMENT,
        "seeds": sorted(int(s) for s in carriers["carriers"]),
        "policies": {name: list(spec) for name, spec in SPECS.items()},
        "controls": CONTROLS,
        "learning_rate": LEARNING_RATE,
        "rays_per_supervised_frame": RAYS_PER_FRAME,
        "stream_length": STREAM_LENGTH,
        "cohort": {"name": COHORT, "scenes": len(entries), "expected_split": "REPLICA"},
        "baselines": "V27's frozen 3-/7-frame reprojection + harmonic code on this round's seals",
        "reprojection_fill_constant_m": eval_contract["reprojection_fill_constant_m"],
        "near_px": eval_contract["near_px"],
        "transfer_rows": {
            "path": str((TRANSFER / "raw/query_rows.json").resolve()),
            "sha256": sha(TRANSFER / "raw/query_rows.json"),
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
            "weighted (model-free baselines take the same value in every seed)",
            "bootstrap": {"unit": "scene", "draws": DRAWS, "seed": BOOT_SEED},
            "gate": "frozen V2 primary-contrast checks (as Core B V2-V11 and V24-V27)",
            "P1_WRITE_UNDER_SHIFT_STATUS": "gate on AbsRel(OFF) - AbsRel(TTT_100)",
            "P2_HYBRID_VS_MODEL_FREE_STATUS": "gate on AbsRel(REPROJ7_HARMONIC) - "
            "AbsRel(HFILL8_7(TTT_100))",
            "interpretation": {
                "A": "both SUPPORTED",
                "B": "only P1 SUPPORTED",
                "C": "only P2 SUPPORTED",
                "D": "neither SUPPORTED",
            },
            "descriptive": [*CONTRASTS, *HYBRID_CONTRASTS],
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
    sources += [
        SCRIPTS / "run_rgbd_stream_ttt_replica_v28.py",
        SCRIPTS / "analyze_rgbd_hybrid_confirm_v27.py",
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
    print("FROZEN_V28", len(entries), "scenes", len(episodes), "episodes", sealed, "files sealed")


def gate_contrasts(rows, contrasts, seeds):
    v3 = v3_analysis()
    out, per_seed = {}, {str(s): {} for s in seeds}
    for name, (a, b) in contrasts.items():
        left, right = v3.scene_values(rows, a), v3.scene_values(rows, b)
        stats = paired(left, right, draws=DRAWS, seed=BOOT_SEED)
        status, _ = v3.gate(stats, True)
        out[name] = {
            "mean": stats["mean"],
            "ci95": stats["ci95"],
            "positive_tied_negative": [stats["positive"], stats["tied"], stats["negative"]],
            "gate_descriptive": status,
        }
        for seed in seeds:
            per_seed[str(seed)][name] = float(
                np.mean(list(v3.scene_values(rows, a, seed=seed).values()))
            ) - float(np.mean(list(v3.scene_values(rows, b, seed=seed).values())))
    absolute = {
        p: describe(v3.scene_values(rows, p), draws=DRAWS, seed=BOOT_SEED)["mean"]
        for p in sorted({r["policy"] for r in rows})
    }
    return {"absolute": absolute, "contrasts": out, "per_seed_means": per_seed}


def branch(p1, p2):
    first, second = p1 == "SUPPORTED", p2 == "SUPPORTED"
    return {(True, True): "A", (True, False): "B", (False, True): "C", (False, False): "D"}[
        first, second
    ]


def recompute(root, cohorts, rows):
    """Every metric from the saved prediction and GT; NORM rows; the seals."""
    entries = cohort_records(cohorts, EXPECTED_SPLIT)
    worst, norm_rows = 0.0, []
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
        norm_rows.append({**{k: row[k] for k in KEYS}, "depth_absrel": norm["depth_absrel"]})
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
    return worst, norm_rows


def hybrid_part(root, config, cohorts, seeds):
    """V27's baselines and hybrids on this round's seals, plus the constant predictor."""
    v27 = hybrid_module(root)
    cohorts27 = v27.load_cohorts()
    hybrid_root = Path(root) / "hybrid"
    seal = v27.compute_baselines(hybrid_root, config, cohorts27)
    rows27, shares, worst, _ = v27.score(cohorts27, seal)
    entries = cohort_records(cohorts, EXPECTED_SPLIT)
    constant = config["reprojection_fill_constant_m"]
    extra, transfer = [], {}
    for item in read(config["transfer_rows"]["path"]):
        if item["method"] in ("REPROJ_HARMONIC", "CONSTANT"):
            transfer[item["method"], item["scene_id"], item["role"], item["query_id"]] = item[
                "absrel_ALL"
            ]
    worst_transfer, compared = 0.0, 0
    for row in rows27["ALL"]:
        if row["policy"] != "REPROJ3_HARMONIC" or row["seed"] != seeds[0]:
            continue
        key = ("REPROJ_HARMONIC", row["scene_id"], row["role"], row["query_id"])
        if key in transfer:
            worst_transfer = max(worst_transfer, abs(transfer[key] - row["depth_absrel"]))
            compared += 1
    for sid, (_, record, _, base) in entries.items():
        for role in ("A", "B"):
            for qid in record["roles"]["primary_query"]:
                frame = next(f for f in record["frames"] if f["frame_id"] == qid)
                gt = np.load(resolve(frame["depth"], base)).squeeze()
                valid = np.isfinite(gt) & (gt > 0)
                value = float(np.mean(np.abs(constant - gt[valid]) / gt[valid]))
                key = ("CONSTANT", sid, role, qid)
                if key in transfer and seeds:
                    worst_transfer = max(worst_transfer, abs(transfer[key] - value))
                    compared += 1
                for seed in seeds:
                    extra.append(
                        {
                            "seed": seed,
                            "cohort": COHORT,
                            "scene_id": sid,
                            "role": role,
                            "query_id": qid,
                            "policy": "CONSTANT",
                            "depth_absrel": value,
                        }
                    )
    all_rows = rows27["ALL"] + extra
    write_json(Path(root) / "raw/hybrid_all_rows.json", all_rows)
    write_json(Path(root) / "raw/hybrid_far7_rows.json", rows27["FAR7"])
    far7 = gate_contrasts(
        rows27["FAR7"],
        {k: v for k, v in HYBRID_CONTRASTS.items() if "CONSTANT" not in v},
        seeds,
    )
    return {
        "all": gate_contrasts(all_rows, HYBRID_CONTRASTS, seeds),
        "far7": far7,
        "near7_share": float(np.mean([s["near7_share"] for s in shares])),
        "near3_share": float(np.mean([s["near3_share"] for s in shares])),
        "carrier_absrel_max_diff": worst,
        "transfer_check": {"compared": compared, "max_abs_difference": worst_transfer},
    }


def stat(value):
    lo, hi = value["ci95"]
    pos, tie, neg = value["positive_tied_negative"]
    return f"{value['mean']:+.4f} [{lo:+.4f}, {hi:+.4f}]，场景 {pos}/{tie}/{neg}"


def report(root, result):
    main, hyb = result["carrier"], result["hybrid"]
    c, h = main["contrasts"], hyb["all"]["contrasts"]
    lines = [
        f"# {EXPERIMENT}",
        "",
        "跨数据集（Replica，8 个场景）的流帧写入，并与只用同样 7 帧的无模型基线比较。"
        "carrier 是 V17 L 第 24000 步，只在 Hypersim 上训练过。"
        "计划见 [PLAN_BEFORE_V28.md](PLAN_BEFORE_V28.md)。"
        "状态构造与渲染在 GPU（4 号卡，关闭 TF32）上，基线与计分在 CPU 上。"
        "Replica 只允许研究用途，数据不进入仓库。",
        "",
        f"- **P1 WRITE_UNDER_SHIFT（OFF − TTT_100）：** {stat(c['WRITE_UNDER_SHIFT'])}，"
        f"{result['statuses']['P1_WRITE_UNDER_SHIFT_STATUS']}",
        "- **P2 HYBRID_VS_MODEL_FREE（REPROJ7_HARMONIC − HFILL8_7(TTT_100)）：** "
        f"{stat(h['HYBRID_VS_MODEL_FREE'])}，{result['statuses']['P2_HYBRID_VS_MODEL_FREE_STATUS']}",
        f"- **分支 {result['INTERPRETATION_BRANCH']}**",
        f"- **像素比例：** 3 帧 NEAR {100 * hyb['near3_share']:.1f}%，"
        f"7 帧 NEAR {100 * hyb['near7_share']:.1f}%。",
        "",
        "| carrier 内部的对比（正值：后者更好） | 全部像素 | 归一化读出 |",
        "|---|---|---|",
    ]
    for name in CONTRASTS:
        lines.append(
            f"| {name} | {stat(c[name])} | {stat(result['normalized']['contrasts'][name])} |"
        )
    lines += ["", "| 方法 | AbsRel |", "|---|---|"]
    for name, value in sorted({**main["absolute"], **hyb["all"]["absolute"]}.items()):
        lines.append(f"| {name} | {value:.4f} |")
    lines += ["", "| 与无模型基线的对比 | 全部像素 | FAR7 |", "|---|---|---|"]
    for name in HYBRID_CONTRASTS:
        far = hyb["far7"]["contrasts"].get(name)
        lines.append(f"| {name} | {stat(h[name])} | {stat(far) if far else '—'} |")
    check = hyb["transfer_check"]
    lines += [
        "",
        "- **每个 seed（P1）：** "
        + "；".join(
            f"{k}: {v['WRITE_UNDER_SHIFT']:+.4f}" for k, v in main["per_seed_means"].items()
        ),
        f"- **核对：** 3 帧调和插值与常数预测的 AbsRel 与 REPLICA-TRANSFER-V1 的结果行比较 "
        f"{check['compared']} 个，最大差 {check['max_abs_difference']:.1e}；"
        f"carrier 的 AbsRel 与本轮结果行最大差 {hyb['carrier_absrel_max_diff']:.1e}；"
        f"指标重新计算的最大误差 {result['max_metric_error']:.1e}。",
        "",
        "说明：这是 Replica 的第二次使用（第一次是 REPLICA-TRANSFER-V1 的静态迁移）；"
        "只有 8 个场景，功效有限。受保护的 Hypersim final holdout 未打开。",
    ]
    (Path(root) / "README.md").write_text("\n".join(lines) + "\n")


def run(root):
    root = Path(root).resolve()
    config = verify_lock_v28(root)
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
    if sha(Path(config["transfer_rows"]["path"])) != config["transfer_rows"]["sha256"]:
        raise PermissionError("REPLICA-TRANSFER-V1 rows changed")
    torch.set_num_threads(1)
    lock = read(root / "carrier_lock.json")["carriers"]
    carriers = {int(s): load_frozen_carrier(item["path"], "cuda")[0] for s, item in lock.items()}
    cohorts = load_cohorts(root)
    rows = []
    for seed, carrier in sorted(carriers.items()):
        out = root / "raw/eval_ttt" / f"seed_{seed}"
        seed_rows = evaluate_v8(
            carrier, cohorts, EXPECTED_SPLIT, out, seed, "cuda", SPECS, CONTROLS
        )
        rows.extend(seed_rows)
        print("EVALUATED", seed, len(seed_rows), flush=True)
    write_json(root / "raw/eval_ttt_query_results.json", rows)
    worst, norm_rows = recompute(root, cohorts, rows)
    seeds = sorted(carriers)
    carrier_part = gate_contrasts(rows, CONTRASTS, seeds)
    normalized = gate_contrasts(norm_rows, CONTRASTS, seeds)
    hybrid = hybrid_part(root, config, cohorts, seeds)
    statuses = {
        "P1_WRITE_UNDER_SHIFT_STATUS": carrier_part["contrasts"]["WRITE_UNDER_SHIFT"][
            "gate_descriptive"
        ],
        "P2_HYBRID_VS_MODEL_FREE_STATUS": hybrid["all"]["contrasts"]["HYBRID_VS_MODEL_FREE"][
            "gate_descriptive"
        ],
    }
    result = {
        "statuses": statuses,
        "INTERPRETATION_BRANCH": branch(*statuses.values()),
        "carrier": carrier_part,
        "normalized": normalized,
        "hybrid": hybrid,
        "max_metric_error": worst,
        "max_rgb_clamp_excess": max(r["rgb_clamp_max_excess"] for r in rows),
    }
    write_json(root / "ttt_replica_results.json", result)
    write_json(root / "raw/norm_rows.json", norm_rows)
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
    print("V28_COMPLETE", result["INTERPRETATION_BRANCH"], json.dumps(statuses), flush=True)


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
