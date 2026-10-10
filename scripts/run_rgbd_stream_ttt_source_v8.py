#!/usr/bin/env python3
"""Freeze, run and analyze Core B V8 (new-frame vs context supervision, EVAL-V3/V4, GPU 4).

  --stage freeze  writes the contracts and the preregistration (CPU, CUDA hidden);
  --stage run     on GPU 4 through cv_gpu_run.sh: the sealed evaluation of every seed, an
                  independent recomputation of every metric, the normalized readout, the NEAR/FAR
                  split with model-free masks (checked against V17's sealed ones), the analysis and
                  the README.
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
from mcss.mechanism_pilot.rgbd_stream_ttt_v8 import (
    CONTROLS,
    EXPERIMENT,
    LEARNING_RATE,
    POLICY_SPECS,
    evaluate_v8,
)
from mcss.mechanism_pilot.small_training import sha, write_json
from mcss.mechanism_pilot.statistics import measurement_metrics

COREB_V4 = Path("outputs/EXP-3D-RGBD-DYNAMIC-TTT-V4")
COREB_V7 = Path("outputs/EXP-3D-RGBD-DYNAMIC-TTT-FRESH-V7")
V7_SPLIT = Path("outputs/EXP-3D-RGBD-DEPTH-BOUNDS-V7")
V17 = Path("outputs/EXP-3D-RGBD-LONG-TRAIN-V17")
PLAN = "PLAN_BEFORE_COREB_V8.md"
PLAN_SHA256 = "2f04035b0eae82551ea59ae896d308e49a52198483d3dd479910599a1d1b17d8"
CARRIER_STEP = 24000
COHORTS = ("EVAL_V3", "EVAL_V4")
EXPECTED_SPLIT = {name: name for name in COHORTS}
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
    "STREAM_TTT_100": ("OFF", "TTT_100"),
    "SUPERVISION_SOURCE_100": ("OFF_CTX_TTT_100", "TTT_100"),
    "STREAM_TTT": ("OFF", "TTT"),
    "STREAM_DOSE": ("TTT", "TTT_100"),
    "CONTEXT_TTT_WITH_STREAM_100": ("OFF", "OFF_CTX_TTT_100"),
    "TTT_100_SPECIFICITY": ("TTT_100_WRONG_SCENE", "TTT_100"),
    "STREAM_CACHE_EFFECT": ("NO_STREAM", "OFF"),
}
PRIMARY = {
    "P1_STREAM_TTT_100_STATUS": "STREAM_TTT_100",
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
    """Core B V3's scene aggregation and frozen V2 gate (as in Core B V4-V7)."""
    return module("coreb_v3_analysis_v8", "analyze_rgbd_stream_write_v3.py")


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


def verify_lock_v8(root):
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


def freeze(root):
    root = Path(root).resolve()
    if root.name != EXPERIMENT or (root / "preregistration.json").exists():
        raise PermissionError("Exact, unfrozen Core B V8 directory required")
    if not (root / "PROTOCOL.md").is_file() or sha(root / PLAN) != PLAN_SHA256:
        raise PermissionError("The Core B V8 plan and protocol must be in place")
    for done in (COREB_V4, COREB_V7):
        if read(done / "integrity.json")["status"] != "PASS":
            raise PermissionError(f"{done.name} must be finalized")
    selected = read(COREB_V4 / "selection.json")["settings"]
    if selected != {"steps": POLICY_SPECS["TTT"][2], "learning_rate": LEARNING_RATE}:
        raise PermissionError("TTT must use the setting Core B V4 selected on TRAIN")
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
    entries = cohort_records(load_cohorts(root), EXPECTED_SPLIT)
    split = read(V7_SPLIT / "scene_split.json")
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
            "rule": "Core B V1-V7 frame-id rule",
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
        "settings_source": "10 steps: Core B V4 selection (TRAIN); 100 steps: V23 dose arms and "
        "Core B V7; no selection on EVAL-V3/V4 in this round",
        "rays_per_supervised_frame": RAYS_PER_FRAME,
        "stream_length": STREAM_LENGTH,
        "cohorts": {name: len([s for s in entries.values() if s[0] == name]) for name in COHORTS},
        "near_masks": "V11 geometry + INPAINT-V1 near_mask, checked against V17's sealed masks",
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
            "gate": "frozen V2 primary-contrast checks (as Core B V2-V7)",
            "P1_STREAM_TTT_100_STATUS": "gate on AbsRel(OFF) - AbsRel(TTT_100)",
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
    sources += sorted(SCRIPTS.glob("*rgbd_stream_ttt_source_v8*.py"))
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
        "FROZEN_CORE_B_V8",
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
    p1 = statuses["P1_STREAM_TTT_100_STATUS"] == "SUPPORTED"
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
    geo = module("v8_geometry", "run_rgbd_geometric_baselines.py")
    inp = module("v8_inpaint", "run_rgbd_inpaint_baselines.py")
    masks = {}
    for _, record, manifest, base in cohort_records(cohorts, EXPECTED_SPLIT).values():
        for role, context in eval_context(record, manifest, base, "cpu", []).items():
            points = geo.context_points(context)
            for qid in record["roles"]["primary_query"]:
                camera = geo.query_camera(record, qid, manifest["image_size"])
                _, hit = geo.reproject(points, camera)
                masks[record["scene_id"], role, qid] = inp.near_mask(hit)
    return masks


def v17_mask_agreement(masks):
    """Every NEAR mask present in V17's EVAL-V3/V4 seals must equal the sealed one."""
    seals = {}
    for name in COHORTS:
        path = V17 / "raw/eval" / name / "prediction_seal.json"
        if path.is_file():
            seals[name] = read(path)["predictions"]
    compared = 0
    for (sid, role, qid), mask in masks.items():
        found = [
            s[f"{n}/{sid}/{role}/{qid}"] for n, s in seals.items() if f"{n}/{sid}/{role}/{qid}" in s
        ]
        if not found:
            continue
        if sha(Path(found[0]["path"])) != found[0]["sha256"]:
            raise PermissionError("A V17 sealed prediction changed")
        if not np.array_equal(np.load(found[0]["path"])["near"].astype(bool), mask):
            raise AssertionError(f"NEAR mask differs from V17's: {sid}/{role}/{qid}")
        compared += 1
    return {"compared": compared, "identical": compared}


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
    lines = [
        f"# {EXPERIMENT}",
        "",
        "Core B 第八轮：新帧都已缓存时，用新帧监督的 100 步测试时写入是否胜过只用上下文监督。"
        "队列是 EVAL-V3 ∪ EVAL-V4（83 个场景），这两个队列从未计算过上下文监督的写入，"
        "也从未计算过 100 步的流帧写入。"
        "计划见 [PLAN_BEFORE_COREB_V8.md](PLAN_BEFORE_COREB_V8.md)。"
        "全部在 GPU（4 号卡，关闭 TF32）上进行，不能逐位复现。",
        "",
        f"- **P1 STREAM_TTT_100（OFF − TTT_100）：** {stat(c['STREAM_TTT_100'])}，"
        f"{s['P1_STREAM_TTT_100_STATUS']}",
        "- **P2 SUPERVISION_SOURCE_100（OFF_CTX_TTT_100 − TTT_100）：** "
        f"{stat(c['SUPERVISION_SOURCE_100'])}，{s['P2_SUPERVISION_SOURCE_100_STATUS']}",
        f"- **分支 {main['INTERPRETATION_BRANCH']}**",
        f"- **10 步：** OFF − TTT {stat(c['STREAM_TTT'])}；"
        f"剂量 TTT − TTT_100 {stat(c['STREAM_DOSE'])}",
        f"- **上下文监督本身（OFF − OFF_CTX_TTT_100）：** {stat(c['CONTEXT_TTT_WITH_STREAM_100'])}",
        f"- **特异性（TTT_100_WRONG_SCENE − TTT_100）：** {stat(c['TTT_100_SPECIFICITY'])}",
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
        "| 分组（描述性） | STREAM_TTT_100 | SUPERVISION_SOURCE_100 |",
        "|---|---|---|",
    ]
    for name, part in (
        ("NEAR", result["regions"]["NEAR"]),
        ("FAR", result["regions"]["FAR"]),
        *((cohort, part) for cohort, part in result["per_cohort"].items()),
    ):
        pc = part["contrasts"]
        lines.append(
            f"| {name} | {stat(pc['STREAM_TTT_100'])} | {stat(pc['SUPERVISION_SOURCE_100'])} |"
        )
    lines += [
        "",
        "- **每个 seed：** "
        + "；".join(
            f"{k}: STREAM_TTT_100 {v['STREAM_TTT_100']:+.4f}，"
            f"SUPERVISION_SOURCE_100 {v['SUPERVISION_SOURCE_100']:+.4f}"
            for k, v in main["per_seed_means"].items()
        ),
        "- **被监督帧自身的损失（中位数）：** "
        + "；".join(
            f"{p} {v['loss_first_median']:.4f} → {v['loss_last_median']:.4f}"
            f"（{100 * v['share_with_lower_loss']:.0f}% 下降）"
            for p, v in result["ttt"].items()
        ),
        f"- **NEAR 掩码：** 与 V17 封存的掩码比较 {result['v17_mask_agreement']['compared']} 个，"
        "全部相同。",
        "",
        "说明：EVAL-V3/V4 在 Core A 中反复使用，在 Core B V5 与 V23 中也使用过；"
        "受保护的 final holdout 未打开。",
    ]
    (Path(root) / "README.md").write_text("\n".join(lines) + "\n")


def run(root):
    root = Path(root).resolve()
    config = verify_lock_v8(root)
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
        seed_rows = evaluate_v8(
            carrier, cohorts, EXPECTED_SPLIT, out, seed, "cuda", SPECS, CONTROLS
        )
        rows.extend(seed_rows)
        print("EVALUATED", seed, len(seed_rows), flush=True)
    write_json(root / "raw/eval_ttt_query_results.json", rows)
    masks = near_masks(cohorts)
    agreement = v17_mask_agreement(masks)
    worst, derived = recompute(root, cohorts, rows, masks)
    seeds = sorted(carriers)
    result = {
        "primary": analyze(rows, seeds),
        "normalized_readout": analyze(derived["NORM"], seeds, primary=False),
        "regions": {
            region: analyze(derived[region], seeds, primary=False) for region in ("NEAR", "FAR")
        },
        "per_cohort": {
            name: analyze([r for r in rows if r["cohort"] == name], seeds, primary=False)
            for name in COHORTS
        },
        "v17_mask_agreement": agreement,
        "ttt": ttt_descriptives(root, seeds),
        "max_rgb_clamp_excess": max(r["rgb_clamp_max_excess"] for r in rows),
    }
    result["INTERPRETATION_BRANCH"] = result["primary"]["INTERPRETATION_BRANCH"]
    write_json(root / "ttt_source_results.json", result)
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
        "CORE_B_V8_COMPLETE",
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
