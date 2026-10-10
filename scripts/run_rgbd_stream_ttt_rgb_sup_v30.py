#!/usr/bin/env python3
"""Freeze, run and analyze V30 (RGB-only stream frames as write supervision, GPU 4 + CPU).

  --stage freeze  writes the contracts and the preregistration (CPU, CUDA hidden);
  --stage run     on GPU 4 through cv_gpu_run.sh: the sealed evaluation of every seed with
                  RGB-only stream frames, an independent recomputation of every metric, the check
                  that no stream depth was read before the seal, the carrier analysis; then (CPU)
                  V27's 3- and 7-frame baselines on this round, the 3-frame hybrids, the analysis
                  and the README.
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
from mcss.mechanism_pilot.rgbd_stream_ttt_rgb_sup_v30 import (
    CONTROLS,
    EXPERIMENT,
    LEARNING_RATE,
    POLICY_SPECS,
    PURPOSE,
    evaluate_v30,
)
from mcss.mechanism_pilot.rgbd_stream_ttt_v4 import (
    RAYS_PER_FRAME,
    STREAM_LENGTH,
    load_frozen_carrier,
    stream_frame_ids,
)
from mcss.mechanism_pilot.small_training import sha, write_json
from mcss.mechanism_pilot.statistics import measurement_metrics

COREB_V4 = Path("outputs/EXP-3D-RGBD-DYNAMIC-TTT-V4")
COREB_V8 = Path("outputs/EXP-3D-RGBD-DYNAMIC-TTT-SOURCE-V8")
V7_SPLIT = Path("outputs/EXP-3D-RGBD-DEPTH-BOUNDS-V7")
V17 = Path("outputs/EXP-3D-RGBD-LONG-TRAIN-V17")
CONCURRENT = (
    "EXP-3D-RGBD-DYNAMIC-TTT-ONLINE-V11",
    "EXP-3D-RGBD-FAR-FOCUS-V26",
    "EXP-3D-RGBD-DYNAMIC-TTT-REPLICA-V28",
)
PLAN = "PLAN_BEFORE_V30.md"
PLAN_SHA256 = "f3a0783cc1c60921f7a988ff8fe9537dd053b1adb4cc2a836175c900c22e7ea5"
CARRIER_STEP = 24000
COHORTS = ("EVAL_V3", "EVAL_V4")
EXPECTED_SPLIT = {name: name for name in COHORTS}
SPECS = dict(POLICY_SPECS)
GPU = "4"
LAUNCHER = Path.home() / "cv_tools/cv_host_run.py"
ABSTAIN_OPACITY = 0.5
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
    "RGB_SUPERVISION_GAIN": ("NO_STREAM", "RGB_SUP_100"),
    "RGB_ADDS_TO_CONTEXT": ("CTX_TTT_100", "CTX_RGB_SUP_100"),
    "CONTEXT_TTT_100": ("NO_STREAM", "CTX_TTT_100"),
    "CTX_RGB_SUP_GAIN": ("NO_STREAM", "CTX_RGB_SUP_100"),
    "RGB_SUP_VS_CONTEXT_TTT": ("CTX_TTT_100", "RGB_SUP_100"),
    "CTX_RGB_SUP_SPECIFICITY": ("CTX_RGB_SUP_100_WRONG_SCENE", "CTX_RGB_SUP_100"),
}
HYBRID_CONTRASTS = {
    "HYBRID_VS_MODEL_FREE_3": ("REPROJ3_HARMONIC", "HFILL8_3_CTX_RGB_SUP_100"),
    "STATIC_HYBRID_VS_MODEL_FREE_3": ("REPROJ3_HARMONIC", "HFILL8_3_NO_STREAM"),
    "CTX_HYBRID_VS_MODEL_FREE_3": ("REPROJ3_HARMONIC", "HFILL8_3_CTX_TTT_100"),
    "RGB_SUP_HYBRID_VS_MODEL_FREE_3": ("REPROJ3_HARMONIC", "HFILL8_3_RGB_SUP_100"),
    "HYBRID_GAIN_FROM_RGB_SUPERVISION": ("HFILL8_3_CTX_TTT_100", "HFILL8_3_CTX_RGB_SUP_100"),
    "AHFILL8_VS_MODEL_FREE_3": ("REPROJ3_HARMONIC", "AHFILL8_3_CTX_RGB_SUP_100"),
    "RGBD_REFERENCE_GAP": ("HFILL8_3_CTX_RGB_SUP_100", "REPROJ7_HARMONIC"),
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
    """Core B V3's scene aggregation and frozen V2 gate (as in Core B V4-V11 and V24-V28)."""
    return module("coreb_v3_analysis_v30", "analyze_rgbd_stream_write_v3.py")


def hybrid_module(root):
    """V27's frozen baseline code with this round (both cohorts) as its only source."""
    loaded = module("v27_hybrid_for_v30", "analyze_rgbd_hybrid_confirm_v27.py")
    loaded.ROUNDS = {name: Path(root) for name in COHORTS}
    loaded.EXPECTED_SPLIT = dict(EXPECTED_SPLIT)
    loaded.V17 = V17
    return loaded


def resolve(path, base):
    path = Path(path)
    return path if path.is_absolute() else Path(base) / path


def load_cohorts(root):
    lock = read(Path(root) / "cohort_lock.json")
    cohorts = {}
    for name in COHORTS:
        item = lock[name]
        if sha(Path(item["manifest"])) != item["manifest_sha256"]:
            raise PermissionError(f"{name} manifest changed")
        cohorts[name] = (read(Path(item["manifest"])), Path(item["manifest"]).parent)
    return cohorts


def verify_lock_v30(root):
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


def freeze(root):
    root = Path(root).resolve()
    if root.name != EXPERIMENT or (root / "preregistration.json").exists():
        raise PermissionError("Exact, unfrozen V30 directory required")
    if not (root / "PROTOCOL.md").is_file() or sha(root / PLAN) != PLAN_SHA256:
        raise PermissionError("The V30 plan and protocol must be in place")
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
    contract = read(V17 / "training_contract.json")["eval"]
    lock = {}
    for name in COHORTS:
        entry = contract["cohorts"][name]
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
            "rule": "Core B V1-V11 frame-id rule",
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
        "stream_frames": f"RGB + camera only ({PURPOSE}); depth all zero (invalid)",
        "rays_per_supervised_frame": RAYS_PER_FRAME,
        "stream_length": STREAM_LENGTH,
        "cohorts": {name: len([s for s in entries.values() if s[0] == name]) for name in COHORTS},
        "near_masks": "V11 geometry + INPAINT-V1 near_mask, checked against V17's sealed masks",
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
            "weighted (model-free baselines take the same value in every seed)",
            "bootstrap": {"unit": "scene", "draws": DRAWS, "seed": BOOT_SEED},
            "gate": "frozen V2 primary-contrast checks (as Core B V2-V11 and V24-V29)",
            "P1_RGB_SUPERVISION_GAIN_STATUS": "gate on AbsRel(NO_STREAM) - AbsRel(RGB_SUP_100)",
            "P2_RGB_ADDS_TO_CONTEXT_STATUS": "gate on AbsRel(CTX_TTT_100) - "
            "AbsRel(CTX_RGB_SUP_100)",
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
        SCRIPTS / "run_rgbd_stream_ttt_rgb_sup_v30.py",
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
    print("FROZEN_V30", len(entries), "scenes", len(episodes), "episodes", sealed, "files sealed")


def gate_contrasts(rows, contrasts, seeds):
    v3 = v3_analysis()
    out, per_seed = {}, {str(s): {} for s in seeds}
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


def rgb_only_audit(access, marker):
    """Before the seal: stream frames only as RGB_ONLY reads, never with depth, never RGB-D."""
    stream_reads = 0
    for event in access[:marker]:
        if event.get("purpose") == "STREAM_RGBD":
            raise AssertionError("An RGB-D stream read happened before the seal")
        if event.get("purpose") == PURPOSE:
            if "depth" in event.get("channels", []):
                raise AssertionError("An RGB-only stream read carried depth")
            stream_reads += 1
    return stream_reads


def near_masks(cohorts):
    """{(scene, role, query): 3-frame NEAR mask} from each role's measured context depth."""
    geo = module("v30_geometry", "run_rgbd_geometric_baselines.py")
    inp = module("v30_inpaint", "run_rgbd_inpaint_baselines.py")
    masks = {}
    for _, record, manifest, base in cohort_records(cohorts, EXPECTED_SPLIT).values():
        for role, context in eval_context(record, manifest, base, "cpu", []).items():
            points = geo.context_points(context)
            for qid in record["roles"]["primary_query"]:
                camera = geo.query_camera(record, qid, manifest["image_size"])
                _, hit = geo.reproject(points, camera)
                masks[record["scene_id"], role, qid] = inp.near_mask(hit)
    return masks


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
    stream_reads = 0
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
        stream_reads += rgb_only_audit(access, marker)
    return worst, derived, stream_reads


def hybrid_part(root, config, rows, seeds):
    """V27's baselines on this round (after the seal) and the 3-frame hybrids of every policy."""
    v27 = hybrid_module(root)
    cohorts27 = v27.load_cohorts()
    seal = v27.compute_baselines(Path(root) / "hybrid", config, cohorts27)
    entries = v27.entries(cohorts27)
    hybrid_rows = {"ALL": [], "FAR3": []}
    by_query = {}
    for row in rows:
        by_query.setdefault((row["scene_id"], row["role"], row["query_id"]), []).append(row)
    shares = []
    for (sid, role, qid), group in sorted(by_query.items()):
        item = seal[f"{sid}/{role}/{qid}"]
        if sha(Path(item["path"])) != item["sha256"]:
            raise AssertionError("A sealed baseline changed")
        arrays = dict(np.load(item["path"]))
        cohort, record, _, base = entries[sid]
        frame = next(f for f in record["frames"] if f["frame_id"] == qid)
        gt = np.load(resolve(frame["depth"], base)).squeeze()
        valid = np.isfinite(gt) & (gt > 0)
        near3 = arrays["near3"]
        far3 = valid & ~near3
        shares.append(float((valid & near3).sum() / valid.sum()))
        methods = {s: {"REPROJ3_HARMONIC": arrays["harmonic3"]} for s in seeds}
        for s in seeds:
            methods[s]["REPROJ7_HARMONIC"] = arrays["harmonic7"]
        for row in group:
            path = Path(root) / "raw/eval_ttt" / f"seed_{row['seed']}" / row["prediction_path"]
            pred = np.load(path)
            depth = pred["depth"].astype(np.float64)
            opacity = pred["opacity"].astype(np.float64)
            m, policy, harmonic = methods[row["seed"]], row["policy"], arrays["harmonic3"]
            m[f"CARRIER_{policy}"] = depth
            m[f"HFILL8_3_{policy}"] = np.where(near3, harmonic, depth)
            m[f"AHFILL8_3_{policy}"] = np.where(
                near3 | (opacity < ABSTAIN_OPACITY), harmonic, depth
            )
        for seed, m in methods.items():
            for name, value in m.items():
                common = {
                    "seed": seed,
                    "cohort": cohort,
                    "scene_id": sid,
                    "role": role,
                    "query_id": qid,
                    "policy": name,
                }
                error = float(np.mean(np.abs(value[valid] - gt[valid]) / gt[valid]))
                hybrid_rows["ALL"].append({**common, "depth_absrel": error})
                if far3.any():
                    error = float(np.mean(np.abs(value[far3] - gt[far3]) / gt[far3]))
                    hybrid_rows["FAR3"].append({**common, "depth_absrel": error})
    write_json(Path(root) / "raw/hybrid_all_rows.json", hybrid_rows["ALL"])
    write_json(Path(root) / "raw/hybrid_far3_rows.json", hybrid_rows["FAR3"])
    return {
        "all": gate_contrasts(hybrid_rows["ALL"], HYBRID_CONTRASTS, seeds),
        "far3": gate_contrasts(hybrid_rows["FAR3"], HYBRID_CONTRASTS, seeds),
        "near3_share": float(np.mean(shares)),
    }


def v17_mask_agreement(masks):
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


def stat(value):
    lo, hi = value["ci95"]
    pos, tie, neg = value["positive_tied_negative"]
    return f"{value['mean']:+.4f} [{lo:+.4f}, {hi:+.4f}]，场景 {pos}/{tie}/{neg}"


def report(root, result):
    carrier, hybrid = result["carrier"], result["hybrid"]
    c, h = carrier["contrasts"], hybrid["all"]["contrasts"]
    lines = [
        f"# {EXPERIMENT}",
        "",
        "流帧只有 RGB 与相机、没有深度时，不缓存它们、只把它们的 RGB 用作写入的监督，"
        "能否改进只由上下文构造的状态，能否胜过只用上下文监督的写入。"
        "队列是 EVAL-V3 ∪ EVAL-V4（83 个场景），carrier 是 V17 L 第 24000 步。"
        "计划见 [PLAN_BEFORE_V30.md](PLAN_BEFORE_V30.md)。"
        "状态构造与渲染在 GPU（4 号卡，关闭 TF32）上，基线与计分在 CPU 上。",
        "",
        "- **P1 RGB_SUPERVISION_GAIN（NO_STREAM − RGB_SUP_100）：** "
        f"{stat(c['RGB_SUPERVISION_GAIN'])}，"
        f"{result['statuses']['P1_RGB_SUPERVISION_GAIN_STATUS']}",
        "- **P2 RGB_ADDS_TO_CONTEXT（CTX_TTT_100 − CTX_RGB_SUP_100）：** "
        f"{stat(c['RGB_ADDS_TO_CONTEXT'])}，"
        f"{result['statuses']['P2_RGB_ADDS_TO_CONTEXT_STATUS']}",
        "- **混合方法（REPROJ3_HARMONIC − HFILL8_3(CTX_RGB_SUP_100)，描述性）：** "
        f"{stat(h['HYBRID_VS_MODEL_FREE_3'])}",
        f"- **分支 {result['INTERPRETATION_BRANCH']}**",
        f"- **像素比例：** 3 帧 NEAR 占有效像素 {100 * hybrid['near3_share']:.1f}%。",
        "",
        "| carrier 内部的对比（正值：后者更好） | 全部像素 | 归一化读出 | FAR | NEAR |",
        "|---|---|---|---|---|",
    ]
    for name in CONTRASTS:
        lines.append(
            f"| {name} | {stat(c[name])} | {stat(result['normalized']['contrasts'][name])} | "
            f"{stat(result['regions']['FAR']['contrasts'][name])} | "
            f"{stat(result['regions']['NEAR']['contrasts'][name])} |"
        )
    lines += ["", "| 方法 | 全部像素 AbsRel | FAR3 AbsRel |", "|---|---|---|"]
    far_abs = hybrid["far3"]["absolute"]
    for name, value in sorted(hybrid["all"]["absolute"].items()):
        far_text = f"{far_abs[name]:.4f}" if name in far_abs else "—"
        lines.append(f"| {name} | {value:.4f} | {far_text} |")
    lines += ["", "| 与无模型基线的对比 | 全部像素 | FAR3 |", "|---|---|---|"]
    for name in HYBRID_CONTRASTS:
        lines.append(f"| {name} | {stat(h[name])} | {stat(hybrid['far3']['contrasts'][name])} |")
    lines += [
        "",
        "- **每个 seed（P1 / P2）：** "
        + "；".join(
            f"{k}: {v['RGB_SUPERVISION_GAIN']:+.4f} / {v['RGB_ADDS_TO_CONTEXT']:+.4f}"
            for k, v in carrier["per_seed_means"].items()
        ),
        f"- **核对：** 封存前流帧只以 RGB 读取 {result['rgb_only_stream_reads']} 次，"
        "没有一次读取深度；"
        f"NEAR 掩码与 V17 封存的比较 {result['v17_mask_agreement']['compared']} 个，全部相同；"
        f"指标重新计算的最大误差 {result['max_metric_error']:.1e}。",
        "",
        "说明：REPROJ7_HARMONIC（假如流帧有深度）在全部状态封存之后才计算，只作参照。"
        "EVAL-V3/V4 在 Core A 与 Core B 中都使用过；受保护的 final holdout 未打开。",
    ]
    (Path(root) / "README.md").write_text("\n".join(lines) + "\n")


def run(root):
    root = Path(root).resolve()
    config = verify_lock_v30(root)
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
        seed_rows = evaluate_v30(
            carrier, cohorts, EXPECTED_SPLIT, out, seed, "cuda", SPECS, CONTROLS
        )
        rows.extend(seed_rows)
        print("EVALUATED", seed, len(seed_rows), flush=True)
    write_json(root / "raw/eval_ttt_query_results.json", rows)
    masks = near_masks(cohorts)
    agreement = v17_mask_agreement(masks)
    worst, derived, stream_reads = recompute(root, cohorts, rows, masks)
    seeds = sorted(carriers)
    carrier_part = gate_contrasts(rows, CONTRASTS, seeds)
    hybrid = hybrid_part(root, config, rows, seeds)
    statuses = {
        "P1_RGB_SUPERVISION_GAIN_STATUS": carrier_part["contrasts"]["RGB_SUPERVISION_GAIN"][
            "gate_descriptive"
        ],
        "P2_RGB_ADDS_TO_CONTEXT_STATUS": carrier_part["contrasts"]["RGB_ADDS_TO_CONTEXT"][
            "gate_descriptive"
        ],
    }
    result = {
        "statuses": statuses,
        "INTERPRETATION_BRANCH": branch(*statuses.values()),
        "carrier": carrier_part,
        "normalized": gate_contrasts(derived["NORM"], CONTRASTS, seeds),
        "regions": {
            region: gate_contrasts(derived[region], CONTRASTS, seeds) for region in ("NEAR", "FAR")
        },
        "hybrid": hybrid,
        "v17_mask_agreement": agreement,
        "rgb_only_stream_reads": stream_reads,
        "max_metric_error": worst,
        "max_rgb_clamp_excess": max(r["rgb_clamp_max_excess"] for r in rows),
    }
    write_json(root / "ttt_rgb_supervision_results.json", result)
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
    print("V30_COMPLETE", result["INTERPRETATION_BRANCH"], json.dumps(statuses), flush=True)


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
