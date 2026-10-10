#!/usr/bin/env python3
"""Freeze, run and analyze Core B V4 (gradient-based test-time writes on the V17 carrier, GPU 4).

  --stage freeze  writes the contracts and the preregistration (CPU, CUDA hidden);
  --stage run     on GPU 4 through cv_gpu_run.sh: the TRAIN-only selection of the TTT setting
                  (written before any DEV computation), the sealed DEV evaluation of every seed,
                  the analysis, an independent recomputation of every DEV metric and the README.
Run from the repository root under cv_host_run.py (records store zonghan_o paths).
"""

from __future__ import annotations

import argparse
import datetime
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from mcss.dynamic.types import hash_scene_state
from mcss.mechanism_pilot.direct_capacity_statistics import describe, paired
from mcss.mechanism_pilot.geometry_carrier_experiment import (
    read,
    validate_split,
    verify_data,
    verify_lock,
)
from mcss.mechanism_pilot.rgbd_stream_evaluation_v3 import preseal_depth_audit
from mcss.mechanism_pilot.rgbd_stream_ttt_evaluation_v4 import (
    SELECTION_SEED,
    evaluate_ttt,
    select_on_train,
    selection_episodes,
)
from mcss.mechanism_pilot.rgbd_stream_ttt_v4 import (
    CONTROLS,
    EXPERIMENT,
    GRID,
    POLICIES,
    RAYS_PER_FRAME,
    STREAM_LENGTH,
    load_frozen_carrier,
    stream_frame_ids,
)
from mcss.mechanism_pilot.small_training import sha, write_json
from mcss.mechanism_pilot.statistics import measurement_metrics

V7 = Path("outputs/EXP-3D-RGBD-DEPTH-BOUNDS-V7")
V17 = Path("outputs/EXP-3D-RGBD-LONG-TRAIN-V17")
V17_PREREGISTRATION_SHA256 = "36b8aa6e4e527025d3746aa7af5ffc060fbf7683fc1a26ff50a94e3c744c1d1f"
COREB_V2 = Path("outputs/EXP-3D-RGBD-DYNAMIC-WRITE-V2")
COREB_V3 = Path("outputs/EXP-3D-RGBD-DYNAMIC-WRITE-V3")
PLAN = "PLAN_BEFORE_COREB_V4.md"
PLAN_SHA256 = "3c7229ca405e0f9d050414e5f616bf0c54482706f9f168ad959cbdc8f7902db5"
QUALIFICATION = Path("outputs/EXP-3D-RGBD-FRESH-QUALIFICATION-V2")
CARRIER_STEP = 24000
SELECTION_COUNT = 32
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
CONTRASTS = {
    "TTT_BEYOND_BASELINE": ("OFF", "TTT"),
    "TTT_SPECIFICITY": ("TTT_WRONG_SCENE", "TTT"),
    "STREAM_CACHE_EFFECT": ("NO_STREAM", "OFF"),
    "TTT_STREAM_GAIN": ("NO_STREAM", "TTT"),
}
PRIMARY = {
    "TTT_BEYOND_BASELINE_STATUS": "TTT_BEYOND_BASELINE",
    "TTT_SPECIFICITY_STATUS": "TTT_SPECIFICITY",
}
DRAWS, BOOT_SEED = 10000, 20260928
METRICS = ("rgb_mse", "depth_absrel", "depth_rmse", "depth_delta1", "opacity", "coverage")


def v3_analysis():
    spec = importlib.util.spec_from_file_location(
        "coreb_v3_analysis", Path("scripts/analyze_rgbd_stream_write_v3.py")
    )
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def semantic(manifest):
    return {
        r["scene_id"]: (
            r["split"],
            json.dumps(r["roles"], sort_keys=True),
            [f["frame_id"] for f in r["frames"]],
        )
        for r in manifest["scenes"]
    }


def seal_previous(root):
    files = {}
    for base in (Path("outputs"), Path("docs/experiments")):
        for path in sorted(base.rglob("*")):
            if (
                path.is_file()
                and EXPERIMENT not in path.parts
                and not any(n in path.parts for n in CONCURRENT)
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


def freeze(root):
    root = Path(root).resolve()
    if root.name != EXPERIMENT or (root / "preregistration.json").exists():
        raise PermissionError("Exact, unfrozen Core B V4 directory required")
    if not (root / "PROTOCOL.md").is_file() or sha(root / PLAN) != PLAN_SHA256:
        raise PermissionError("The Core B V4 plan and protocol must be in place")
    if read(COREB_V3 / "integrity.json")["status"] != "PASS":
        raise PermissionError("Core B V3 must be finalized")
    if sha(LAUNCHER) != sha(Path("../_archives_cv/cv_host_run.py")):
        raise PermissionError("The path-mapping launcher must equal the migrated one")
    qualification = read(QUALIFICATION / "fresh_qualification_results.json")
    if qualification["FRESH_QUALIFICATION_STATUS"] != "QUALIFIED":
        raise PermissionError("Core B opens only after the static carrier is QUALIFIED")
    if sha(V17 / "preregistration.json") != V17_PREREGISTRATION_SHA256:
        raise PermissionError("V17 is not the frozen V17")
    manifest = read(V7 / "scene_split.json")
    if semantic(read(V17 / "scene_split.json")) != semantic(manifest):
        raise PermissionError("V17 must have trained on the V7 split")
    validate_split(manifest)
    write_json(root / "scene_split.json", manifest)
    streams = read(COREB_V2 / "stream_roles.json")
    for split in ("DEV", "TRAIN"):
        for record in (r for r in manifest["scenes"] if r["split"] == split):
            for role in ("A", "B"):
                ids = stream_frame_ids(record, role)
                if streams[split][f"{record['scene_id']}/{role}"] != (list(ids) if ids else None):
                    raise PermissionError("The stream roles must be Core B V2's")
    write_json(root / "stream_roles.json", streams)
    shutil.copyfile(V7 / "train_depth_prior.json", root / "train_depth_prior.json")
    if sha(root / "train_depth_prior.json") != sha(COREB_V3 / "train_depth_prior.json"):
        raise PermissionError("The TRAIN depth prior must be V7's, as in Core B V3")
    carriers = read(COREB_V3 / "carrier_lock.json")
    for seed, item in carriers["carriers"].items():
        if sha(Path(item["path"])) != item["sha256"] or item["step"] != CARRIER_STEP:
            raise PermissionError(f"V17 step-{CARRIER_STEP} carrier changed: {seed}")
        load_frozen_carrier(item["path"], "cpu")
    write_json(root / "carrier_lock.json", carriers)
    episodes = selection_episodes(streams["train_eligible_episodes"], SELECTION_COUNT)
    config = {
        "experiment": EXPERIMENT,
        "seeds": sorted(int(s) for s in carriers["carriers"]),
        "policies": list(POLICIES),
        "controls": list(CONTROLS),
        "grid": list(GRID),
        "rays_per_stream_frame": RAYS_PER_FRAME,
        "stream_length": STREAM_LENGTH,
        "optimizer": "Adam on (delta_fuse, delta_complete), both from zero; frozen V2 C1 loss "
        "of the stream frames' own RGB and measured depth; rays from sha256(seed:episode)-"
        "seeded generators",
        "selection": {
            "seed": SELECTION_SEED,
            "count": SELECTION_COUNT,
            "episodes": [f"{s}/{r}" for s, r in episodes],
            "criterion": "mean over carriers, episodes and primary-query slots of AbsRel(OFF) - "
            "AbsRel(TTT) on the offline TRAIN labels; ties: fewer steps, then smaller rate",
        },
        "primary_contrasts": {
            k: f"AbsRel({a}) - AbsRel({b})"
            for k, (a, b) in CONTRASTS.items()
            if k in PRIMARY.values()
        },
        "num_threads": 1,
        "thread_environment": THREADS,
        "cpu_affinity": list(range(56, 64)),
        "device": "cuda",
        "gpu": GPU,
        "host_launcher": {"path": str(LAUNCHER), "sha256": sha(LAUNCHER)},
        "fresh_qualification_included": False,
    }
    write_json(root / "training_contract.json", config)
    write_json(
        root / "decision_rules.json",
        {
            "aggregation": "per seed: scene mean of role means of query AbsRel; seeds equally "
            "weighted",
            "bootstrap": {"unit": "physical scene", "draws": DRAWS, "seed": BOOT_SEED},
            "gate": "frozen V2 primary-contrast checks (as Core B V2/V3)",
            "TTT_BEYOND_BASELINE_STATUS": "gate on AbsRel(OFF) - AbsRel(TTT) (primary)",
            "TTT_SPECIFICITY_STATUS": "gate on AbsRel(TTT_WRONG_SCENE) - AbsRel(TTT) (primary)",
            "interpretation": {
                "A": "both primary gates SUPPORTED",
                "B": "TTT_BEYOND_BASELINE point estimate > 0, not A",
                "C": "otherwise",
            },
            "final_holdout": "never opened",
        },
    )
    sealed = seal_previous(root)
    data_hashes = {
        str(Path(f[k])): sha(Path(f[k]))
        for r in manifest["scenes"]
        for f in r["frames"]
        for k in ("rgb", "depth")
    }
    write_json(root / "audit/data_hashes.json", data_hashes)
    names = (
        "scene_split.json",
        "stream_roles.json",
        "carrier_lock.json",
        "train_depth_prior.json",
        "training_contract.json",
        "decision_rules.json",
        "PROTOCOL.md",
        PLAN,
        "audit/data_hashes.json",
        "audit/previous_experiment_seal.json",
    )
    sources = sorted(Path("src/mcss").rglob("*.py")) + sorted(
        Path("scripts").glob("*rgbd_stream_ttt_v4*.py")
    )
    sources += [Path("scripts/analyze_rgbd_stream_write_v3.py"), LAUNCHER]
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
    print("FROZEN_CORE_B_V4", len(episodes), "selection episodes", sealed, "files sealed")


def analyze(rows, seeds):
    v3 = v3_analysis()
    absolute = {
        p: describe(v3.scene_values(rows, p), draws=DRAWS, seed=BOOT_SEED)
        for p in sorted({r["policy"] for r in rows})
    }
    contrasts = {
        name: paired(
            v3.scene_values(rows, a), v3.scene_values(rows, b), draws=DRAWS, seed=BOOT_SEED
        )
        for name, (a, b) in CONTRASTS.items()
    }
    statuses, checks = {}, {}
    for status, name in PRIMARY.items():
        statuses[status], checks[status] = v3.gate(contrasts[name], True)
    if all(statuses[k] == "SUPPORTED" for k in PRIMARY):
        branch = "A"
    elif contrasts["TTT_BEYOND_BASELINE"]["mean"] > 0:
        branch = "B"
    else:
        branch = "C"
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
        "statuses": statuses,
        "gate_checks": checks,
        "INTERPRETATION_BRANCH": branch,
        "per_seed_means": per_seed,
    }


def recompute(root, manifest, rows):
    """Every DEV metric from the saved prediction and the GT frame; state hashes and seal order."""
    records = {r["scene_id"]: r for r in manifest["scenes"] if r["split"] == "DEV"}
    worst = 0.0
    for row in rows:
        out = Path(root) / "raw/dev_ttt" / f"seed_{row['seed']}"
        path = out / row["prediction_path"]
        if sha(path) != row["prediction_file_sha256"]:
            raise AssertionError("Prediction changed")
        pred = dict(np.load(path))
        frame = next(
            f for f in records[row["scene_id"]]["frames"] if f["frame_id"] == row["query_id"]
        )
        with Image.open(frame["rgb"]) as image:
            rgb = np.array(image.convert("RGB"), dtype=np.float32) / 255
        metrics = measurement_metrics(
            pred["rgb"], rgb, pred["depth"], np.load(frame["depth"]).squeeze(), pred["opacity"]
        )
        for key in METRICS:
            worst = max(worst, abs(metrics[key] - row[key]))
    if worst > 1e-6:
        raise AssertionError(f"Metric mismatch {worst}")
    for seed in {r["seed"] for r in rows}:
        out = Path(root) / "raw/dev_ttt" / f"seed_{seed}"
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
    return worst


def stat(value):
    lo, hi = value["ci95"]
    counts = f"{value['positive']}/{value['tied']}/{value['negative']}"
    return f"{value['mean']:+.4f} [{lo:+.4f}, {hi:+.4f}]，场景 {counts}"


def report(root, selection, result, config):
    a = result["absolute"]
    c, s = result["contrasts"], result["statuses"]
    lines = [
        f"# {EXPERIMENT}",
        "",
        "Core B 第四轮：在 V17 L 第 24000 步的 carrier（冻结）上，"
        "用新帧自己的 RGB 与测得深度，对快速权重做基于梯度的测试时写入（TTT），"
        "检验它能否胜过只缓存新帧。"
        "计划见 [PLAN_BEFORE_COREB_V4.md](PLAN_BEFORE_COREB_V4.md)。"
        "全部在 GPU（4 号卡，关闭 TF32）上进行，不能逐位复现。",
        "",
        f"- **TRAIN 上选出的设置：** {selection['settings']}；各组的平均增益 "
        + "，".join(f"{GRID[int(k)]}: {v:+.4f}" for k, v in selection["gains"].items()),
        f"- **TTT_BEYOND_BASELINE（OFF − TTT）：** {stat(c['TTT_BEYOND_BASELINE'])}，"
        f"{s['TTT_BEYOND_BASELINE_STATUS']}",
        f"- **TTT_SPECIFICITY（TTT_WRONG_SCENE − TTT）：** {stat(c['TTT_SPECIFICITY'])}，"
        f"{s['TTT_SPECIFICITY_STATUS']}",
        f"- **分支 {result['INTERPRETATION_BRANCH']}**",
        f"- **描述性：** 缓存新帧（NO_STREAM − OFF）{stat(c['STREAM_CACHE_EFFECT'])}；"
        f"NO_STREAM − TTT {stat(c['TTT_STREAM_GAIN'])}",
        "",
        "| 策略 | DEV query AbsRel（95% CI） |",
        "|---|---|",
    ]
    for policy in (*POLICIES, *CONTROLS):
        v = a[policy]
        lines.append(f"| {policy} | {v['mean']:.4f} [{v['ci95'][0]:.4f}, {v['ci95'][1]:.4f}] |")
    lines += [
        "",
        "说明：DEV 的 8 个场景在 V2–V10 与 Core B V1–V3 中反复使用，本轮是 DEV 机制诊断；"
        "V17 的 carrier 本身未经独立资格验证；"
        "FRESH-V1/V2 未使用；受保护的 final holdout 未打开。",
    ]
    (Path(root) / "README.md").write_text("\n".join(lines) + "\n")


def run(root):
    root = Path(root).resolve()
    manifest, config = verify_lock(root)
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
    episodes = [tuple(e.split("/")) for e in config["selection"]["episodes"]]
    if (root / "selection.json").exists():
        raise FileExistsError("The selection is made once")
    selection = select_on_train(carriers, manifest, root, episodes, GRID, "cuda")
    write_json(root / "selection.json", {k: v for k, v in selection.items() if k != "rows"})
    write_json(root / "raw/selection_rows.json", selection["rows"])
    print("SELECTED", selection["settings"], selection["gains"], flush=True)
    rows = []
    for seed, carrier in sorted(carriers.items()):
        out = root / "raw/dev_ttt" / f"seed_{seed}"
        seed_rows = evaluate_ttt(carrier, selection["settings"], manifest, root, out, seed, "cuda")
        rows.extend(seed_rows)
        print("DEV_EVALUATED", seed, len(seed_rows), flush=True)
    write_json(root / "raw/dev_ttt_query_results.json", rows)
    result = analyze(rows, sorted(carriers))
    result["selection"] = {k: v for k, v in selection.items() if k not in ("rows", "access")}
    result["max_rgb_clamp_excess"] = max(r["rgb_clamp_max_excess"] for r in rows)
    write_json(root / "ttt_results.json", result)
    worst = recompute(root, manifest, rows)
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
            "selection_before_dev": True,
            "slow_carrier_updated": False,
            "final_holdout_touched": False,
            "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "finished_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        },
    )
    report(root, selection, result, config)
    print(
        "CORE_B_V4_COMPLETE",
        result["INTERPRETATION_BRANCH"],
        json.dumps(result["statuses"]),
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
