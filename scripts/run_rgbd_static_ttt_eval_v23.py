#!/usr/bin/env python3
"""Freeze, run and analyze V23 (context-only TTT of the static V17 L state, EVAL-V3/V4, GPU 4).

  --stage freeze  writes the contracts and the preregistration (CPU, CUDA hidden);
  --stage run     on GPU 4 through cv_gpu_run.sh: the sealed evaluation of every seed, an
                  independent recomputation of every metric, the model-free NEAR masks and harmonic
                  fills (checked against V17's sealed ones), the analysis and the README.
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
from mcss.mechanism_pilot.rgbd_static_ttt_v23 import (
    CONTROLS,
    EXPERIMENT,
    LEARNING_RATE,
    POLICIES,
    STEPS,
    evaluate_static,
)
from mcss.mechanism_pilot.rgbd_stream_evaluation_v3 import preseal_depth_audit
from mcss.mechanism_pilot.rgbd_stream_ttt_context_v6 import cohort_records
from mcss.mechanism_pilot.rgbd_stream_ttt_eval_v5 import eval_context
from mcss.mechanism_pilot.rgbd_stream_ttt_v4 import RAYS_PER_FRAME, load_frozen_carrier
from mcss.mechanism_pilot.small_training import sha, write_json
from mcss.mechanism_pilot.statistics import measurement_metrics

COREB_V4 = Path("outputs/EXP-3D-RGBD-DYNAMIC-TTT-V4")
COREB_V6 = Path("outputs/EXP-3D-RGBD-DYNAMIC-TTT-CONTEXT-V6")
V7 = Path("outputs/EXP-3D-RGBD-DEPTH-BOUNDS-V7")
V17 = Path("outputs/EXP-3D-RGBD-LONG-TRAIN-V17")
PLAN = "PLAN_BEFORE_V23.md"
PLAN_SHA256 = "8e0f54e2b9d9361d6554487464d0e520a3adf9d1681f9deb2bf7c3c10d700d4f"
CARRIER_STEP = 24000
COHORTS = ("EVAL_V3", "EVAL_V4")
EXPECTED_SPLIT = {name: name for name in COHORTS}
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
    "CONTEXT_TTT": ("NO_STREAM", "CTX_TTT"),
    "CONTEXT_TTT_SPECIFICITY": ("CTX_TTT_WRONG_SCENE", "CTX_TTT"),
    "CONTEXT_TTT_30": ("NO_STREAM", "CTX_TTT_30"),
    "CONTEXT_TTT_100": ("NO_STREAM", "CTX_TTT_100"),
    "DOSE_30_VS_10": ("CTX_TTT", "CTX_TTT_30"),
    "DOSE_100_VS_10": ("CTX_TTT", "CTX_TTT_100"),
}
PRIMARY = {
    "P1_CONTEXT_TTT_STATUS": "CONTEXT_TTT",
    "P2_SPECIFICITY_STATUS": "CONTEXT_TTT_SPECIFICITY",
}
BASELINE_CONTRASTS = {
    f"HARMONIC_MINUS_{p}": ("REPROJ_HARMONIC", p) for p in ("NO_STREAM", "CTX_TTT", "CTX_TTT_100")
}
HYBRID_CONTRASTS = {
    f"HARMONIC_MINUS_AHFILL8_{p}": ("REPROJ_HARMONIC", f"AHFILL8_{p}")
    for p in ("NO_STREAM", "CTX_TTT", "CTX_TTT_100")
}
ABSTAIN_OPACITY = 0.5
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
    """Core B V3's scene aggregation and frozen V2 gate (as in Core B V4-V6)."""
    return module("coreb_v3_analysis_v23", "analyze_rgbd_stream_write_v3.py")


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


def verify_lock_v23(root):
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
        raise PermissionError("Exact, unfrozen V23 directory required")
    if not (root / "PROTOCOL.md").is_file() or sha(root / PLAN) != PLAN_SHA256:
        raise PermissionError("The V23 plan and protocol must be in place")
    if read(COREB_V6 / "integrity.json")["status"] != "PASS":
        raise PermissionError("Core B V6 must be finalized")
    if read(COREB_V6 / "ttt_context_results.json")["INTERPRETATION_BRANCH"] not in ("A", "C"):
        raise PermissionError("V23 confirms Core B V6's supported context TTT")
    if read(COREB_V4 / "selection.json")["settings"] != {
        "steps": STEPS["CTX_TTT"],
        "learning_rate": LEARNING_RATE,
    }:
        raise PermissionError("CTX_TTT must use the setting Core B V4 selected on TRAIN")
    if sha(LAUNCHER) != sha(Path("../_archives_cv/cv_host_run.py")):
        raise PermissionError("The path-mapping launcher must equal the migrated one")
    carriers = read(COREB_V4 / "carrier_lock.json")
    for seed, item in carriers["carriers"].items():
        if sha(Path(item["path"])) != item["sha256"] or item["step"] != CARRIER_STEP:
            raise PermissionError(f"V17 step-{CARRIER_STEP} carrier changed: {seed}")
        load_frozen_carrier(item["path"], "cpu")
    write_json(root / "carrier_lock.json", carriers)
    contract = read(V17 / "training_contract.json")["eval"]
    lock = {}
    for name in COHORTS:
        entry = contract["cohorts"][name]
        if sha(Path(entry["manifest"])) != entry["manifest_sha256"]:
            raise PermissionError(f"{name} manifest is not V17's")
        lock[name] = {"manifest": entry["manifest"], "manifest_sha256": entry["manifest_sha256"]}
    write_json(root / "cohort_lock.json", lock)
    entries = cohort_records(load_cohorts(root), EXPECTED_SPLIT)
    split = read(V7 / "scene_split.json")
    used = {r["scene_id"] for r in split["scenes"]} | set(split.get("protected_scene_ids", []))
    if used & set(entries):
        raise PermissionError("EVAL scenes must be disjoint from TRAIN72, DEV and the holdout")
    config = {
        "experiment": EXPERIMENT,
        "seeds": sorted(int(s) for s in carriers["carriers"]),
        "policies": list(POLICIES),
        "ttt_steps": STEPS,
        "learning_rate": LEARNING_RATE,
        "controls": CONTROLS,
        "settings_source": "Core B V4 selection.json (TRAIN only) for CTX_TTT; dose arms "
        "descriptive",
        "rays_per_supervised_frame": RAYS_PER_FRAME,
        "cohorts": {name: len([s for s in entries.values() if s[0] == name]) for name in COHORTS},
        "baselines": "model-free NEAR mask and REPROJ_HARMONIC (V11 geometry + INPAINT-V1), "
        "checked "
        "against V17's sealed arrays",
        "reprojection_fill_constant_m": contract["reprojection_fill_constant_m"],
        "near_px": contract["near_px"],
        "abstain_opacity_threshold": ABSTAIN_OPACITY,
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
            "gate": "frozen V2 primary-contrast checks (as Core B V2-V6)",
            "P1_CONTEXT_TTT_STATUS": "gate on AbsRel(NO_STREAM) - AbsRel(CTX_TTT), all pixels",
            "P2_SPECIFICITY_STATUS": "gate on AbsRel(CTX_TTT_WRONG_SCENE) - AbsRel(CTX_TTT)",
            "interpretation": {"A": "both SUPPORTED", "B": "only P1 SUPPORTED", "C": "otherwise"},
            "descriptive": [*CONTRASTS, *BASELINE_CONTRASTS, *HYBRID_CONTRASTS, "NEAR/FAR", "NORM"],
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
        "evaluation_contract.json",
        "decision_rules.json",
        "PROTOCOL.md",
        PLAN,
        "audit/data_hashes.json",
        "audit/previous_experiment_seal.json",
    )
    sources = sorted(Path("src/mcss").rglob("*.py")) + sorted(
        SCRIPTS.glob("*rgbd_static_ttt_eval_v23*.py")
    )
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
    print("FROZEN_V23", len(entries), "scenes", sealed, "files sealed")


def analyze(rows, seeds, contrasts, primary=None):
    """Contrasts with Core B V3's aggregation and the frozen V2 gate; the plan's branch."""
    v3 = v3_analysis()
    policies = sorted({r["policy"] for r in rows})
    absolute = {
        p: describe(v3.scene_values(rows, p), draws=DRAWS, seed=BOOT_SEED)["mean"] for p in policies
    }
    out, statuses = {}, {}
    for name, (a, b) in contrasts.items():
        stats = paired(
            v3.scene_values(rows, a), v3.scene_values(rows, b), draws=DRAWS, seed=BOOT_SEED
        )
        status, _ = v3.gate(stats, True)
        out[name] = {
            "mean": stats["mean"],
            "ci95": stats["ci95"],
            "positive_tied_negative": [stats["positive"], stats["tied"], stats["negative"]],
            "gate_descriptive": status,
        }
        for key, value in (primary or {}).items():
            if value == name:
                statuses[key] = status
    result = {"absolute": absolute, "contrasts": out}
    if primary:
        p1 = statuses["P1_CONTEXT_TTT_STATUS"] == "SUPPORTED"
        p2 = statuses["P2_SPECIFICITY_STATUS"] == "SUPPORTED"
        result["statuses"] = statuses
        result["INTERPRETATION_BRANCH"] = "A" if p1 and p2 else ("B" if p1 else "C")
        result["per_seed_means"] = {
            str(seed): {
                name: float(np.mean(list(v3.scene_values(rows, a, seed=seed).values())))
                - float(np.mean(list(v3.scene_values(rows, b, seed=seed).values())))
                for name, (a, b) in contrasts.items()
            }
            for seed in seeds
        }
    return result


def baselines(cohorts, constant):
    """{(scene, role, query): (NEAR mask, harmonic fill)} from each role's context (model-free)."""
    geo = module("v23_geometry", "run_rgbd_geometric_baselines.py")
    inp = module("v23_inpaint", "run_rgbd_inpaint_baselines.py")
    out = {}
    for _, record, manifest, base in cohort_records(cohorts, EXPECTED_SPLIT).values():
        for role, context in eval_context(record, manifest, base, "cpu", []).items():
            points = geo.context_points(context)
            for qid in record["roles"]["primary_query"]:
                camera = geo.query_camera(record, qid, manifest["image_size"])
                depth, hit = geo.reproject(points, camera)
                harmonic = inp.harmonic_fill(depth, hit, inp.ray_cosine(camera), constant)
                out[record["scene_id"], role, qid] = (inp.near_mask(hit), harmonic)
    return out


def v17_agreement(arrays):
    """Every model-free baseline present in V17's EVAL-V3/V4 seals must equal the sealed one."""
    compared, worst = 0, 0.0
    seals = {}
    for name in COHORTS:
        path = V17 / "raw/eval" / name / "prediction_seal.json"
        if path.is_file():
            seals[name] = read(path)["predictions"]
    for (sid, role, qid), (near, harmonic) in arrays.items():
        found = [
            s[f"{n}/{sid}/{role}/{qid}"] for n, s in seals.items() if f"{n}/{sid}/{role}/{qid}" in s
        ]
        if not found:
            continue
        item = found[0]
        if sha(Path(item["path"])) != item["sha256"]:
            raise PermissionError("A V17 sealed prediction changed")
        sealed = np.load(item["path"])
        if not np.array_equal(sealed["near"].astype(bool), near):
            raise AssertionError(f"NEAR mask differs from V17's: {sid}/{role}/{qid}")
        worst = max(worst, float(np.max(np.abs(sealed["REPROJ_HARMONIC"] - harmonic))))
        compared += 1
    if worst > 1e-9:
        raise AssertionError(f"Harmonic fill differs from V17's: {worst}")
    return {"compared": compared, "near_identical": compared, "harmonic_max_abs_diff": worst}


def absrel(value, gt, mask):
    return float(np.mean(np.abs(value[mask] - gt[mask]) / gt[mask])) if mask.any() else None


def recompute(root, cohorts, rows, arrays):
    """Every metric from the saved prediction and GT; NORM, regions, baselines; the seals."""
    entries = cohort_records(cohorts, EXPECTED_SPLIT)
    worst = 0.0
    derived = {"NORM": [], "FAR": [], "NEAR": [], "BASE_FAR": [], "BASE_ALL": []}
    harmonic_done = set()
    for row in rows:
        path = Path(root) / "raw/eval_static" / f"seed_{row['seed']}" / row["prediction_path"]
        if sha(path) != row["prediction_file_sha256"]:
            raise AssertionError("Prediction changed")
        pred = dict(np.load(path))
        _, record, _, base = entries[row["scene_id"]]
        frame = next(f for f in record["frames"] if f["frame_id"] == row["query_id"])
        with Image.open(resolve(frame["rgb"], base)) as image:
            rgb = np.array(image.convert("RGB"), dtype=np.float32) / 255
        raw_gt = np.load(resolve(frame["depth"], base)).squeeze()
        metrics = measurement_metrics(pred["rgb"], rgb, pred["depth"], raw_gt, pred["opacity"])
        gt = raw_gt.astype(np.float64)
        for key in METRICS:
            worst = max(worst, abs(metrics[key] - row[key]))
        common = {k: row[k] for k in KEYS}
        depth = pred["depth"].astype(np.float64)
        norm = depth / np.maximum(pred["opacity"].astype(np.float64), 1e-3)
        derived["NORM"].append(
            {**common, "depth_absrel": absrel(norm, gt, np.isfinite(gt) & (gt > 0))}
        )
        near, harmonic = arrays[row["scene_id"], row["role"], row["query_id"]]
        valid = np.isfinite(gt) & (gt > 0)
        far = valid & ~near
        for region, mask in (("FAR", far), ("NEAR", valid & near)):
            value = absrel(depth, gt, mask)
            if value is not None:
                derived[region].append({**common, "depth_absrel": value})
        if row["policy"] in ("NO_STREAM", "CTX_TTT", "CTX_TTT_100"):
            value = absrel(depth, gt, far)
            if value is not None:
                derived["BASE_FAR"].append({**common, "depth_absrel": value})
            low = pred["opacity"] < ABSTAIN_OPACITY
            hybrid = np.where(near | low, harmonic, depth)
            derived["BASE_ALL"].append(
                {
                    **common,
                    "policy": f"AHFILL8_{row['policy']}",
                    "depth_absrel": absrel(hybrid, gt, valid),
                }
            )
            key = (row["seed"], row["scene_id"], row["role"], row["query_id"])
            if key not in harmonic_done:
                harmonic_done.add(key)
                base_common = {**common, "policy": "REPROJ_HARMONIC"}
                value = absrel(harmonic, gt, far)
                if value is not None:
                    derived["BASE_FAR"].append({**base_common, "depth_absrel": value})
                derived["BASE_ALL"].append(
                    {**base_common, "depth_absrel": absrel(harmonic, gt, valid)}
                )
    if worst > 1e-6:
        raise AssertionError(f"Metric mismatch {worst}")
    records = {sid: item[1] for sid, item in entries.items()}
    for seed in {r["seed"] for r in rows}:
        out = Path(root) / "raw/eval_static" / f"seed_{seed}"
        audit = read(out / "integrity.json")
        if sha(out / "states.pt") != audit["states_file_sha256"]:
            raise AssertionError("Sealed states changed")
        states = torch.load(out / "states.pt", map_location="cpu", weights_only=False)
        index = read(out / "state_hashes.json")
        if any(hash_scene_state(s) != index[k]["state_hash"] for k, s in states.items()):
            raise AssertionError("State hash mismatch")
        access = read(out / "GT_access.json")
        marker = next(
            i for i, e in enumerate(access) if e.get("event") == "ALL_STATIC_STATES_SEALED"
        )
        preseal_depth_audit(access[:marker], records)
    return worst, derived


def ttt_descriptives(root, seeds):
    out = {}
    for policy in POLICIES[1:]:
        first, last, norms = [], [], []
        for seed in seeds:
            index = read(Path(root) / "raw/eval_static" / f"seed_{seed}" / "state_hashes.json")
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
    region = result["regions"]
    lines = [
        f"# {EXPERIMENT}",
        "",
        "只用上下文的测试时精修（Core B V6 的 CTX_TTT，10 步，学习率 1e-3）"
        "能否改进静态补全 carrier。"
        "队列是 Core A 的主评价队列 EVAL-V3 ∪ EVAL-V4（83 个场景），carrier 是 V17 L 第 24000 步。"
        "计划见 [PLAN_BEFORE_V23.md](PLAN_BEFORE_V23.md)。"
        "全部在 GPU（4 号卡，关闭 TF32）上进行，不能逐位复现。",
        "",
        f"- **P1（NO_STREAM − CTX_TTT，全部像素）：** {stat(c['CONTEXT_TTT'])}，"
        f"{s['P1_CONTEXT_TTT_STATUS']}",
        f"- **P2（CTX_TTT_WRONG_SCENE − CTX_TTT）：** {stat(c['CONTEXT_TTT_SPECIFICITY'])}，"
        f"{s['P2_SPECIFICITY_STATUS']}",
        f"- **分支 {main['INTERPRETATION_BRANCH']}**",
        f"- **剂量（描述性）：** 30 步 {stat(c['CONTEXT_TTT_30'])}；"
        f"100 步 {stat(c['CONTEXT_TTT_100'])}",
        "",
        "| 策略 | 全部像素 | 归一化读出 | NEAR | FAR |",
        "|---|---|---|---|---|",
    ]
    for policy in (*POLICIES, *CONTROLS):
        lines.append(
            f"| {policy} | {main['absolute'][policy]:.4f} | "
            f"{result['normalized_readout']['absolute'][policy]:.4f} | "
            f"{region['NEAR']['absolute'][policy]:.4f} | {region['FAR']['absolute'][policy]:.4f} |"
        )
    lines += [
        "",
        "| 区域（描述性） | NO_STREAM − CTX_TTT | NO_STREAM − CTX_TTT_100 |",
        "|---|---|---|",
    ]
    for name in ("NEAR", "FAR"):
        rc = region[name]["contrasts"]
        lines.append(f"| {name} | {stat(rc['CONTEXT_TTT'])} | {stat(rc['CONTEXT_TTT_100'])} |")
    base, hybrid = result["baselines_far"]["contrasts"], result["hybrids_all"]["contrasts"]
    lines += [
        "",
        "| 与调和插值比较（描述性） | FAR：调和插值 − carrier | 全部像素：调和插值 − AHFILL8 |",
        "|---|---|---|",
    ]
    for policy in ("NO_STREAM", "CTX_TTT", "CTX_TTT_100"):
        lines.append(
            f"| {policy} | {stat(base[f'HARMONIC_MINUS_{policy}'])} | "
            f"{stat(hybrid[f'HARMONIC_MINUS_AHFILL8_{policy}'])} |"
        )
    agreement, ttt = result["v17_agreement"], result["ttt"]
    lines += [
        "",
        "- **每个 seed（NO_STREAM − CTX_TTT）：** "
        + "，".join(f"{k}: {v['CONTEXT_TTT']:+.4f}" for k, v in main["per_seed_means"].items()),
        "- **上下文帧自身的损失（中位数）：** "
        + "；".join(
            f"{p} {v['loss_first_median']:.4f} → {v['loss_last_median']:.4f}"
            f"（{100 * v['share_with_lower_loss']:.0f}% 下降）"
            for p, v in ttt.items()
        ),
        "- **与模型无关的基线：** 与 V17 封存的数组比较 "
        f"{agreement['compared']} 个，NEAR 掩码全部相同，"
        f"调和插值的最大差 {agreement['harmonic_max_abs_diff']:.1e}。",
        "",
        "说明：EVAL-V3/V4 在 Core A 中反复使用，在 Core B V5 中用于流帧 TTT，"
        "但从未计算过只用上下文的 TTT；"
        "剂量臂只是描述性的；受保护的 final holdout 未打开。",
    ]
    (Path(root) / "README.md").write_text("\n".join(lines) + "\n")


def run(root):
    root = Path(root).resolve()
    config = verify_lock_v23(root)
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
        out = root / "raw/eval_static" / f"seed_{seed}"
        seed_rows = evaluate_static(carrier, cohorts, EXPECTED_SPLIT, out, seed, "cuda")
        rows.extend(seed_rows)
        print("EVALUATED", seed, len(seed_rows), flush=True)
    write_json(root / "raw/eval_static_query_results.json", rows)
    arrays = baselines(cohorts, config["reprojection_fill_constant_m"])
    agreement = v17_agreement(arrays)
    worst, derived = recompute(root, cohorts, rows, arrays)
    seeds = sorted(carriers)
    result = {
        "primary": analyze(rows, seeds, CONTRASTS, PRIMARY),
        "normalized_readout": analyze(derived["NORM"], seeds, CONTRASTS),
        "regions": {name: analyze(derived[name], seeds, CONTRASTS) for name in ("NEAR", "FAR")},
        "baselines_far": analyze(derived["BASE_FAR"], seeds, BASELINE_CONTRASTS),
        "hybrids_all": analyze(derived["BASE_ALL"], seeds, HYBRID_CONTRASTS),
        "v17_agreement": agreement,
        "ttt": ttt_descriptives(root, seeds),
        "max_rgb_clamp_excess": max(r["rgb_clamp_max_excess"] for r in rows),
    }
    result["INTERPRETATION_BRANCH"] = result["primary"]["INTERPRETATION_BRANCH"]
    write_json(root / "static_ttt_results.json", result)
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
        "V23_COMPLETE",
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
