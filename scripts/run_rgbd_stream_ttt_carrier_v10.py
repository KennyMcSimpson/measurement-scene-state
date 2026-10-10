#!/usr/bin/env python3
"""Freeze, run and analyze Core B V10 (100-step stream TTT on four carriers, EVAL-V3/V4, GPU 4).

  --stage freeze  writes the contracts and the preregistration (CPU, CUDA hidden);
  --stage run     on GPU 4 through cv_gpu_run.sh: the sealed evaluation of every carrier and seed,
                  an independent recomputation of every metric, the normalized readout, the
                  NEAR/FAR split with model-free masks (checked against V17's sealed ones), the
                  comparison of the V17 L rows with Core B V8's, the analysis and the README.
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
from mcss.mechanism_pilot.rgbd_stream_ttt_v4 import RAYS_PER_FRAME, STREAM_LENGTH, stream_frame_ids
from mcss.mechanism_pilot.rgbd_stream_ttt_v10 import (
    CARRIER_STEP,
    CARRIERS,
    CONTROLS,
    EXPERIMENT,
    LEARNING_RATE,
    POLICY_SPECS,
    evaluate_v10,
    load_stream_carrier,
)
from mcss.mechanism_pilot.small_training import sha, write_json
from mcss.mechanism_pilot.statistics import measurement_metrics

COREB_V4 = Path("outputs/EXP-3D-RGBD-DYNAMIC-TTT-V4")
COREB_V8 = Path("outputs/EXP-3D-RGBD-DYNAMIC-TTT-SOURCE-V8")
V7_SPLIT = Path("outputs/EXP-3D-RGBD-DEPTH-BOUNDS-V7")
V17 = Path("outputs/EXP-3D-RGBD-LONG-TRAIN-V17")
V18 = Path("outputs/EXP-3D-RGBD-LONG-TRAIN-DATA-V18")
CONCURRENT = ("EXP-3D-RGBD-DYNAMIC-TTT-DOSE-V9",)
PLAN = "PLAN_BEFORE_COREB_V10.md"
PLAN_SHA256 = "e8081221a93ea15e9bc18347685b500793ca502ad510ff811a9e6b9108610540"
SEEDS = (20260928, 20260929, 20260930)
COHORTS = ("EVAL_V3", "EVAL_V4")
EXPECTED_SPLIT = {name: name for name in COHORTS}
SPECS = dict(POLICY_SPECS)
REFERENCE = "V17_L"
PRIMARY = {
    "P1_STREAM_TTT_100_V18L_STATUS": "V18_L",
    "P2_STREAM_TTT_100_V19L_STATUS": "V19_L",
    "P3_STREAM_TTT_100_V20L_STATUS": "V20_L",
}
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
    "TTT_100_SPECIFICITY": ("TTT_100_WRONG_SCENE", "TTT_100"),
    "STREAM_CACHE_EFFECT": ("NO_STREAM", "OFF"),
}
# Effects on the per-scene gain OFF - TTT_100 (positive: wider / more-data carriers gain more).
FACTORS = {
    "WIDTH_EFFECT_ON_GAIN": ((("V19_L", 0.5), ("V20_L", 0.5)), (("V17_L", 0.5), ("V18_L", 0.5))),
    "DATA_EFFECT_ON_GAIN": ((("V18_L", 0.5), ("V20_L", 0.5)), (("V17_L", 0.5), ("V19_L", 0.5))),
    "INTERACTION_ON_GAIN": ((("V20_L", 1.0), ("V17_L", 1.0)), (("V19_L", 1.0), ("V18_L", 1.0))),
}
DRAWS, BOOT_SEED = 10000, 20260928
METRICS = ("rgb_mse", "depth_absrel", "depth_rmse", "depth_delta1", "opacity", "coverage")
KEYS = ("carrier", "seed", "cohort", "scene_id", "role", "query_id", "policy")
SCRIPTS = Path("scripts")


def module(name, filename):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / filename)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def v3_analysis():
    """Core B V3's scene aggregation and frozen V2 gate (as in Core B V4-V9)."""
    return module("coreb_v3_analysis_v10", "analyze_rgbd_stream_write_v3.py")


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


def verify_lock_v10(root):
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
    """Every earlier experiment file, except this round's and the concurrently running ones."""
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


def carrier_lock():
    """Every carrier's step-24000 checkpoint per seed, hash-checked against its own DEV curve."""
    carriers = {}
    for variant, (experiment, _, hidden, train) in CARRIERS.items():
        runs = {}
        for seed in SEEDS:
            run_dir = Path("outputs") / experiment / "checkpoints" / f"L_{seed}"
            entry = next(e for e in read(run_dir / "dev_curve.json") if e["step"] == CARRIER_STEP)
            path = (run_dir / f"step_{CARRIER_STEP:06d}.pt").resolve()
            if Path(entry["checkpoint"]).resolve() != path:
                raise PermissionError(f"{variant}/{seed}: the DEV curve names another checkpoint")
            if sha(path) != entry["checkpoint_sha256"]:
                raise PermissionError(f"{variant}/{seed}: checkpoint changed since its DEV curve")
            carrier, _ = load_stream_carrier(variant, path, "cpu")
            runs[str(seed)] = {
                "path": str(path),
                "sha256": entry["checkpoint_sha256"],
                "step": CARRIER_STEP,
                "carrier_parameters": sum(p.numel() for p in carrier.parameters()),
                "carrier_trainable_parameters": sum(
                    p.numel() for p in carrier.parameters() if p.requires_grad
                ),
                "hidden_width": hidden,
                "train_set": train,
                "dev_query_absrel": entry["query_absrel"],
            }
        carriers[variant] = runs
    reference = read(COREB_V4 / "carrier_lock.json")["carriers"]
    if {s: item["sha256"] for s, item in reference.items()} != {
        s: item["sha256"] for s, item in carriers[REFERENCE].items()
    }:
        raise PermissionError("V17 L must be the carrier of Core B V4-V9")
    return carriers


def freeze(root):
    root = Path(root).resolve()
    if root.name != EXPERIMENT or (root / "preregistration.json").exists():
        raise PermissionError("Exact, unfrozen Core B V10 directory required")
    if not (root / "PROTOCOL.md").is_file() or sha(root / PLAN) != PLAN_SHA256:
        raise PermissionError("The Core B V10 plan and protocol must be in place")
    for done in (COREB_V4, COREB_V8):
        if read(done / "integrity.json")["status"] != "PASS":
            raise PermissionError(f"{done.name} must be finalized")
    if read(COREB_V4 / "selection.json")["settings"]["learning_rate"] != LEARNING_RATE:
        raise PermissionError("TTT must use the learning rate Core B V4 selected on TRAIN")
    if sha(LAUNCHER) != sha(Path("../_archives_cv/cv_host_run.py")):
        raise PermissionError("The path-mapping launcher must equal the migrated one")
    write_json(
        root / "carrier_lock.json",
        {
            "carrier_step": CARRIER_STEP,
            "rule": "each experiment's checkpoints/L_<seed>/dev_curve.json step-24000 record",
            "carriers": carrier_lock(),
        },
    )
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
    extension = read(V18 / "training_contract.json")["train_ext"]
    if sha(Path(extension["manifest"])) != extension["manifest_sha256"]:
        raise PermissionError("TRAIN-EXT manifest is not the one V18 and V20 trained on")
    used |= {r["scene_id"] for r in read(Path(extension["manifest"]))["scenes"]}
    if used & set(entries):
        raise PermissionError("EVAL scenes must be disjoint from TRAIN127, DEV and the holdout")
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
            "rule": "Core B V1-V9 frame-id rule",
            "stream_length": STREAM_LENGTH,
            "episodes": episodes,
        },
    )
    config = {
        "experiment": EXPERIMENT,
        "seeds": list(SEEDS),
        "carriers": {v: {"hidden_width": c[2], "train_set": c[3]} for v, c in CARRIERS.items()},
        "reference_carrier": REFERENCE,
        "policies": {name: list(spec) for name, spec in SPECS.items()},
        "controls": CONTROLS,
        "learning_rate": LEARNING_RATE,
        "settings_source": "learning rate: Core B V4 selection (TRAIN); 100 steps: V23 dose arms "
        "and Core B V7/V8; no selection on EVAL-V3/V4 or on any carrier in this round",
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
            "aggregation": "per carrier and seed: scene mean of role means of query AbsRel; seeds "
            "equally weighted",
            "bootstrap": {"unit": "scene", "draws": DRAWS, "seed": BOOT_SEED},
            "gate": "frozen V2 primary-contrast checks (as Core B V2-V9)",
            **{
                key: f"gate on AbsRel(OFF) - AbsRel(TTT_100) with carrier {variant}"
                for key, variant in PRIMARY.items()
            },
            "interpretation": {
                "A": "all three SUPPORTED",
                "B": "one or two SUPPORTED",
                "C": "none SUPPORTED",
            },
            "descriptive": [*CONTRASTS, *FACTORS, f"{REFERENCE} contrasts"],
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
    sources += sorted(SCRIPTS.glob("*rgbd_stream_ttt_carrier_v10*.py"))
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
        "FROZEN_CORE_B_V10",
        len(CARRIERS),
        "carriers",
        len(entries),
        "scenes",
        len(episodes),
        "episodes",
        sealed,
        "files sealed",
    )


def summary(stats):
    return {
        "mean": stats["mean"],
        "ci95": stats["ci95"],
        "positive_tied_negative": [stats["positive"], stats["tied"], stats["negative"]],
    }


def analyze(rows, seeds, primary=True):
    """Every contrast per carrier (Core B V3's aggregation, frozen V2 gate); the plan's branch."""
    v3 = v3_analysis()
    carriers, gains, statuses = {}, {}, {}
    for variant in CARRIERS:
        part = [r for r in rows if r["carrier"] == variant]
        policies = sorted({r["policy"] for r in part})
        absolute = {
            p: describe(v3.scene_values(part, p), draws=DRAWS, seed=BOOT_SEED)["mean"]
            for p in policies
        }
        contrasts = {}
        for name, (a, b) in CONTRASTS.items():
            left, right = v3.scene_values(part, a), v3.scene_values(part, b)
            stats = paired(left, right, draws=DRAWS, seed=BOOT_SEED)
            status, _ = v3.gate(stats, True)
            contrasts[name] = {**summary(stats), "gate_descriptive": status}
            if name == "STREAM_TTT_100":
                gains[variant] = {s: left[s] - right[s] for s in left}
        per_seed = {
            str(seed): {
                name: float(np.mean(list(v3.scene_values(part, a, seed=seed).values())))
                - float(np.mean(list(v3.scene_values(part, b, seed=seed).values())))
                for name, (a, b) in CONTRASTS.items()
            }
            for seed in seeds
        }
        carriers[variant] = {"absolute": absolute, "contrasts": contrasts, "per_seed": per_seed}
    for key, variant in PRIMARY.items():
        statuses[key] = carriers[variant]["contrasts"]["STREAM_TTT_100"]["gate_descriptive"]
    supported = sum(status == "SUPPORTED" for status in statuses.values())
    branch = "A" if supported == len(PRIMARY) else "C" if supported == 0 else "B"
    factors = {}
    for name, sides in FACTORS.items():
        left, right = (
            {s: sum(w * gains[v][s] for v, w in side) for s in gains[REFERENCE]} for side in sides
        )
        factors[name] = summary(paired(left, right, draws=DRAWS, seed=BOOT_SEED))
    return {
        "carriers": carriers,
        "factors_on_gain": factors,
        "statuses": statuses if primary else {},
        "INTERPRETATION_BRANCH": branch if primary else None,
    }


def near_masks(cohorts):
    """{(scene, role, query): NEAR mask} from each role's measured context depth (model-free)."""
    geo = module("v10_geometry", "run_rgbd_geometric_baselines.py")
    inp = module("v10_inpaint", "run_rgbd_inpaint_baselines.py")
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


def row_agreement(rows, theirs):
    """Per policy: this round's V17 L AbsRel matched to another round's rows of the same queries."""
    keys = ("seed", "cohort", "scene_id", "role", "query_id")

    def values(items, policy):
        return {
            tuple(r[k] for k in keys): r["depth_absrel"] for r in items if r["policy"] == policy
        }

    mine = [r for r in rows if r["carrier"] == REFERENCE]
    out = {}
    for policy in (*SPECS, *CONTROLS):
        a, b = values(mine, policy), values(theirs, policy)
        diffs = [abs(a[k] - b[k]) for k in sorted(set(a) & set(b))]
        out[policy] = {
            "rows": len(a),
            "compared": len(diffs),
            "identical": sum(d == 0 for d in diffs),
            "max_abs_difference": max(diffs, default=0.0),
            "mean_abs_difference": float(np.mean(diffs)) if diffs else 0.0,
        }
    return out


def v8_agreement(rows):
    """The V17 L rows against Core B V8's (same carrier, episodes and generators; GPU)."""
    path = COREB_V8 / "raw/eval_ttt_query_results.json"
    return row_agreement(rows, read(path) if path.is_file() else [])


def prediction_dir(root, row):
    return Path(root) / "raw/eval_ttt" / row["carrier"] / f"seed_{row['seed']}"


def recompute(root, cohorts, rows, masks):
    """Every metric from the saved prediction and GT; NORM and NEAR/FAR rows; the seals."""
    entries = cohort_records(cohorts, EXPECTED_SPLIT)
    worst = 0.0
    derived = {"NORM": [], "FAR": [], "NEAR": []}
    for row in rows:
        path = prediction_dir(root, row) / row["prediction_path"]
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
    for carrier, seed in {(r["carrier"], r["seed"]) for r in rows}:
        out = Path(root) / "raw/eval_ttt" / carrier / f"seed_{seed}"
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
    for variant in CARRIERS:
        first, last, norms = [], [], []
        for seed in seeds:
            index = read(
                Path(root) / "raw/eval_ttt" / variant / f"seed_{seed}" / "state_hashes.json"
            )
            for item in index.values():
                if item["policy"] == "TTT_100":
                    first.append(item["ttt_loss_first"])
                    last.append(item["ttt_loss_last"])
                    norms.append(item["fast_norm"])
        drop = np.array(first) - np.array(last)
        out[variant] = {
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
    lines = [
        f"# {EXPERIMENT}",
        "",
        "Core B 第十轮：100 步流帧写入超出只缓存的增益，是否在另外三个 carrier 上也成立。"
        "四个 carrier 都训练到第 24000 步，构成宽度（32/64）× 训练数据（TRAIN72/TRAIN127）的 2×2；"
        "V17 L 是 Core B V4–V9 所用的 carrier，作为本轮内的参照。"
        "队列是 EVAL-V3 ∪ EVAL-V4（83 个场景）。"
        "计划见 [PLAN_BEFORE_COREB_V10.md](PLAN_BEFORE_COREB_V10.md)。"
        "全部在 GPU（4 号卡，关闭 TF32）上进行，不能逐位复现。",
        "",
    ]
    for key, variant in PRIMARY.items():
        c = main["carriers"][variant]["contrasts"]["STREAM_TTT_100"]
        lines.append(
            f"- **{key[:2]} {variant}（OFF − TTT_100）：** {stat(c)}，{main['statuses'][key]}"
        )
    reference = main["carriers"][REFERENCE]["contrasts"]["STREAM_TTT_100"]
    lines += [
        f"- **分支 {main['INTERPRETATION_BRANCH']}**",
        f"- **参照 {REFERENCE}（描述性）：** {stat(reference)}",
        "",
        "| carrier | 宽度 | 训练集 | NO_STREAM | OFF | TTT_100 | WRONG_SCENE | OFF − TTT_100 "
        "| 特异性 | FAR：OFF − TTT_100 |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for variant, (_, _, hidden, train) in CARRIERS.items():
        part = main["carriers"][variant]
        a, c = part["absolute"], part["contrasts"]
        far = result["regions"]["FAR"]["carriers"][variant]["contrasts"]["STREAM_TTT_100"]
        lines.append(
            f"| {variant} | {hidden} | {train} | {a['NO_STREAM']:.4f} | {a['OFF']:.4f} | "
            f"{a['TTT_100']:.4f} | {a['TTT_100_WRONG_SCENE']:.4f} | {stat(c['STREAM_TTT_100'])} | "
            f"{stat(c['TTT_100_SPECIFICITY'])} | {stat(far)} |"
        )
    lines += [
        "",
        "| 对增益的作用（描述性） | 全部像素 | NEAR | FAR | 归一化读出 |",
        "|---|---|---|---|---|",
    ]
    for name in FACTORS:
        lines.append(
            f"| {name} | {stat(main['factors_on_gain'][name])} | "
            f"{stat(result['regions']['NEAR']['factors_on_gain'][name])} | "
            f"{stat(result['regions']['FAR']['factors_on_gain'][name])} | "
            f"{stat(result['normalized_readout']['factors_on_gain'][name])} |"
        )
    v8 = result["core_b_v8_agreement"]
    lines += [
        "",
        "- **每个 seed（OFF − TTT_100）：** "
        + "；".join(
            f"{variant} "
            + "/".join(f"{v['STREAM_TTT_100']:+.4f}" for v in part["per_seed"].values())
            for variant, part in main["carriers"].items()
        ),
        "- **被监督帧自身的损失（TTT_100，中位数）：** "
        + "；".join(
            f"{p} {v['loss_first_median']:.4f} → {v['loss_last_median']:.4f}"
            f"（{100 * v['share_with_lower_loss']:.0f}% 下降）"
            for p, v in result["ttt"].items()
        ),
        f"- **与 Core B V8 的一致性（{REFERENCE}，描述性）：** "
        + "；".join(
            f"{p} {v['identical']}/{v['compared']} 行相同，最大差 {v['max_abs_difference']:.2e}"
            for p, v in v8.items()
        ),
        f"- **NEAR 掩码：** 与 V17 封存的掩码比较 {result['v17_mask_agreement']['compared']} 个，"
        "全部相同。",
        "",
        "说明：EVAL-V3/V4 在 Core A 中反复使用，在 Core B V5、V23 与 V8 中也使用过；"
        "受保护的 final holdout 未打开。Core B V9 与本轮的冻结同时运行，不在本轮的封存范围内。",
    ]
    (Path(root) / "README.md").write_text("\n".join(lines) + "\n")


def run(root):
    root = Path(root).resolve()
    config = verify_lock_v10(root)
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
    cohorts = load_cohorts(root)
    rows = []
    for variant in CARRIERS:
        for seed in SEEDS:
            item = lock[variant][str(seed)]
            if sha(Path(item["path"])) != item["sha256"]:
                raise PermissionError(f"{variant}/{seed}: carrier changed")
            carrier = load_stream_carrier(variant, item["path"], "cuda")[0]
            out = root / "raw/eval_ttt" / variant / f"seed_{seed}"
            seed_rows = evaluate_v10(
                carrier, cohorts, EXPECTED_SPLIT, out, seed, "cuda", SPECS, CONTROLS
            )
            rows.extend({**row, "carrier": variant} for row in seed_rows)
            print("EVALUATED", variant, seed, len(seed_rows), flush=True)
            del carrier
    write_json(root / "raw/eval_ttt_query_results.json", rows)
    masks = near_masks(cohorts)
    agreement = v17_mask_agreement(masks)
    worst, derived = recompute(root, cohorts, rows, masks)
    seeds = list(SEEDS)
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
        "core_b_v8_agreement": v8_agreement(rows),
        "ttt": ttt_descriptives(root, seeds),
        "max_rgb_clamp_excess": max(r["rgb_clamp_max_excess"] for r in rows),
    }
    result["INTERPRETATION_BRANCH"] = result["primary"]["INTERPRETATION_BRANCH"]
    write_json(root / "ttt_carrier_results.json", result)
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
        "CORE_B_V10_COMPLETE",
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
