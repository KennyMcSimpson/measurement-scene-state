#!/usr/bin/env python3
"""V27: the hybrids and the MAX rule on TRAIN-EXT (Core B V9) and EVAL-V5 (Core B V7), CPU only.

  --stage freeze  hash-locks the plan, the protocol, both rounds' records and the sources;
  --stage run     computes the 3- and 7-frame reprojection + harmonic baselines (no GT), seals
                  them, rebuilds the hybrids from the rounds' sealed predictions, scores everything
                  against the query depth and writes the analysis and the README.
Run from the repository root under cv_host_run.py with CUDA hidden. No model is run.
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

from mcss.mechanism_pilot.direct_capacity_statistics import describe, paired
from mcss.mechanism_pilot.geometry_carrier_experiment import read
from mcss.mechanism_pilot.rgbd_evidence_carrier import RGBDContext
from mcss.mechanism_pilot.rgbd_stream_evaluation_v3 import preseal_depth_audit
from mcss.mechanism_pilot.rgbd_stream_ttt_context_v6 import cohort_records
from mcss.mechanism_pilot.rgbd_stream_ttt_eval_v5 import eval_context
from mcss.mechanism_pilot.rgbd_stream_ttt_v4 import StreamRGBDLoader
from mcss.mechanism_pilot.small_training import sha, write_json

EXPERIMENT = "EXP-3D-RGBD-HYBRID-CONFIRM-V27"
ROUNDS = {
    "TRAIN_EXT": Path("outputs/EXP-3D-RGBD-DYNAMIC-TTT-DOSE-V9"),
    "EVAL_V5": Path("outputs/EXP-3D-RGBD-DYNAMIC-TTT-FRESH-V7"),
}
EXPECTED_SPLIT = {"TRAIN_EXT": "TRAIN", "EVAL_V5": "EVAL_V5"}
V17 = Path("outputs/EXP-3D-RGBD-LONG-TRAIN-V17")
PILOT_SEAL = (
    Path.home()
    / "cv_work/gpu_pilot_v1/w32/EXP-3D-RGBD-LONG-TRAIN-V17/raw/eval/TRAIN_EXT/prediction_seal.json"
)
CONCURRENT = (
    "EXP-3D-RGBD-DYNAMIC-TTT-CARRIER-V10",
    "EXP-3D-RGBD-DYNAMIC-TTT-ONLINE-V11",
    "EXP-3D-RGBD-FAR-FOCUS-V26",
)
PLAN = "PLAN_BEFORE_V27.md"
PLAN_SHA256 = "d991381fb4a2ff8e1985e448bff3d41e4be6c64ee8ebafa66c023dd023e77708"
POLICIES = ("NO_STREAM", "OFF", "OFF_CTX_TTT_100", "TTT_100")
STREAM_POLICIES = POLICIES[1:]
ABSTAIN_OPACITY = 0.5
CPU_CORES = set(range(32, 40))
THREADS = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "ATEN_CPU_CAPABILITY": "avx2",
    "MKL_CBWR": "AVX2",
    "DNNL_MAX_CPU_ISA": "AVX2",
    "CUDA_VISIBLE_DEVICES": "",
}
CONTRASTS = {
    "MAXFILL_VS_MODEL_FREE": ("REPROJ7_HARMONIC", "MAXFILL8_7_TTT_100"),
    "HYBRID_VS_MODEL_FREE": ("REPROJ7_HARMONIC", "HFILL8_7_TTT_100"),
    "MAXFILL_VS_HYBRID": ("HFILL8_7_TTT_100", "MAXFILL8_7_TTT_100"),
    "TTT_GAIN_IN_HYBRID": ("HFILL8_7_OFF", "HFILL8_7_TTT_100"),
    "TTT_GAIN_IN_MAXFILL": ("MAXFILL8_7_OFF", "MAXFILL8_7_TTT_100"),
    "OFF_MAXFILL_VS_MODEL_FREE": ("REPROJ7_HARMONIC", "MAXFILL8_7_OFF"),
    "AHFILL8_VS_MODEL_FREE": ("REPROJ7_HARMONIC", "AHFILL8_7_TTT_100"),
    "HYBRID_NORM_VS_MODEL_FREE": ("REPROJ7_HARMONIC", "HFILL8_7N_TTT_100"),
    "MAXFILL_NORM_VS_MODEL_FREE": ("REPROJ7_HARMONIC", "MAXFILL8_7N_TTT_100"),
    "CARRIER_TTT_100_VS_MODEL_FREE": ("REPROJ7_HARMONIC", "CARRIER_TTT_100"),
    "CARRIER_TTT_GAIN": ("CARRIER_OFF", "CARRIER_TTT_100"),
    "CONTEXT_TTT_IN_MAXFILL": ("MAXFILL8_7_OFF", "MAXFILL8_7_OFF_CTX_TTT_100"),
    "STREAM_FRAMES_FOR_MODEL_FREE": ("REPROJ3_HARMONIC", "REPROJ7_HARMONIC"),
    "CORE_A_HYBRID": ("REPROJ3_HARMONIC", "HFILL8_3_NO_STREAM"),
}
PRIMARY = {
    "P1_MAXFILL_VS_MODEL_FREE_STATUS": "MAXFILL_VS_MODEL_FREE",
    "P2_HYBRID_VS_MODEL_FREE_STATUS": "HYBRID_VS_MODEL_FREE",
}
DRAWS, BOOT_SEED = 10000, 20260928
SCRIPTS = Path("scripts")


def module(name, filename):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / filename)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def v3_analysis():
    """Core B V3's scene aggregation and frozen V2 gate (as in Core B V4-V11, V24 and V25)."""
    return module("coreb_v3_analysis_v27", "analyze_rgbd_stream_write_v3.py")


def resolve(path, base):
    path = Path(path)
    return path if path.is_absolute() else Path(base) / path


def hfill(near, harmonic, carrier):
    """HFILL8: the harmonic fill on NEAR pixels, the carrier everywhere else."""
    return np.where(near, harmonic, carrier)


def maxfill(near, harmonic, carrier):
    """MAXFILL8: the harmonic fill on NEAR pixels, the farther of carrier and harmonic elsewhere."""
    return np.where(near, harmonic, np.maximum(carrier, harmonic))


def ahfill(near, opacity, harmonic, carrier, threshold=ABSTAIN_OPACITY):
    """AHFILL8: as HFILL8, and the harmonic fill also where the carrier's opacity is low."""
    return np.where(near | (opacity < threshold), harmonic, carrier)


def absrel(value, gt, mask):
    return float(np.mean(np.abs(value[mask] - gt[mask]) / gt[mask])) if mask.any() else None


def load_cohorts():
    """{cohort: (manifest, base, round)} from each round's cohort lock, manifests hash-checked."""
    cohorts = {}
    for name, round_dir in ROUNDS.items():
        item = read(round_dir / "cohort_lock.json")[name]
        if sha(Path(item["manifest"])) != item["manifest_sha256"]:
            raise PermissionError(f"{name} manifest changed")
        cohorts[name] = (read(Path(item["manifest"])), Path(item["manifest"]).parent, round_dir)
    return cohorts


def entries(cohorts):
    """{scene: (cohort, record, manifest, base)} over both cohorts."""
    return cohort_records({k: (m, b) for k, (m, b, _) in cohorts.items()}, EXPECTED_SPLIT)


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


def check_environment():
    for key, value in THREADS.items():
        if os.environ.get(key) != value:
            raise PermissionError(f"Environment {key} must be {value!r}")
    if not os.sched_getaffinity(0) <= CPU_CORES:
        raise PermissionError("V27 must be pinned to cores 32-39 (taskset)")


def freeze(root):
    root = Path(root).resolve()
    if root.name != EXPERIMENT or (root / "preregistration.json").exists():
        raise PermissionError("Exact, unfrozen V27 directory required")
    if not (root / "PROTOCOL.md").is_file() or sha(root / PLAN) != PLAN_SHA256:
        raise PermissionError("The V27 plan and protocol must be in place")
    check_environment()
    for round_dir in ROUNDS.values():
        if read(round_dir / "integrity.json")["status"] != "PASS":
            raise PermissionError(f"{round_dir.name} must be finalized")
    contract = read(V17 / "training_contract.json")["eval"]
    write_json(
        root / "analysis_contract.json",
        {
            "experiment": EXPERIMENT,
            "rounds": {k: str(v) for k, v in ROUNDS.items()},
            "policies": list(POLICIES),
            "reprojection_fill_constant_m": contract["reprojection_fill_constant_m"],
            "near_px": contract["near_px"],
            "abstain_opacity_threshold": ABSTAIN_OPACITY,
            "pilot_mask_check": {
                "path": str(PILOT_SEAL),
                "sha256": sha(PILOT_SEAL) if PILOT_SEAL.is_file() else None,
            },
            "thread_environment": THREADS,
            "cpu_affinity": sorted(CPU_CORES),
            "device": "cpu",
        },
    )
    write_json(
        root / "decision_rules.json",
        {
            "aggregation": "per seed: scene mean of role means of query AbsRel; seeds equally "
            "weighted (model-free baselines take the same value in every seed); TRAIN-EXT and "
            "EVAL-V5 pooled",
            "bootstrap": {"unit": "scene", "draws": DRAWS, "seed": BOOT_SEED},
            "gate": "frozen V2 primary-contrast checks (as Core B V2-V11, V24, V25)",
            "P1_MAXFILL_VS_MODEL_FREE_STATUS": "gate on AbsRel(REPROJ7_HARMONIC) - "
            "AbsRel(MAXFILL8_7(TTT_100))",
            "P2_HYBRID_VS_MODEL_FREE_STATUS": "gate on AbsRel(REPROJ7_HARMONIC) - "
            "AbsRel(HFILL8_7(TTT_100))",
            "interpretation": {
                "A": "both SUPPORTED",
                "B": "only P1 SUPPORTED",
                "C": "only P2 SUPPORTED",
                "D": "neither SUPPORTED",
            },
            "descriptive": [c for c in CONTRASTS if c not in PRIMARY.values()],
            "final_holdout": "never opened",
        },
    )
    sealed = seal_previous(root)
    names = ("analysis_contract.json", "decision_rules.json", "PROTOCOL.md", PLAN)
    inputs = [V17 / "training_contract.json", root / "audit/previous_experiment_seal.json"]
    for round_dir in ROUNDS.values():
        inputs += [
            round_dir / "integrity.json",
            round_dir / "preregistration.json",
            round_dir / "cohort_lock.json",
            round_dir / "stream_roles.json",
            round_dir / "raw/eval_ttt_query_results.json",
        ]
    sources = sorted(Path("src/mcss").rglob("*.py"))
    sources += [
        SCRIPTS / "analyze_rgbd_hybrid_confirm_v27.py",
        SCRIPTS / "analyze_rgbd_stream_write_v3.py",
        SCRIPTS / "run_rgbd_geometric_baselines.py",
        SCRIPTS / "run_rgbd_inpaint_baselines.py",
        Path.home() / "cv_tools/cv_host_run.py",
    ]
    write_json(
        root / "preregistration.json",
        {
            "status": "FROZEN_BEFORE_ANALYSIS",
            "experiment": EXPERIMENT,
            "input_sha256": {
                **{str((root / n).resolve()): sha(root / n) for n in names},
                **{str(p.resolve()): sha(p) for p in inputs},
            },
            "source_sha256": {str(p.resolve()): sha(p) for p in sources},
        },
    )
    print("FROZEN_V27", sealed, "files sealed")


def verify_lock(root):
    lock = read(Path(root) / "preregistration.json")
    if lock["status"] != "FROZEN_BEFORE_ANALYSIS":
        raise PermissionError("Preregistration absent")
    for section in ("input_sha256", "source_sha256"):
        for name, digest in lock[section].items():
            if sha(Path(name)) != digest:
                raise PermissionError(f"Frozen input/source changed: {name}")
    return read(Path(root) / "analysis_contract.json")


def compute_baselines(root, config, cohorts):
    """3- and 7-frame reprojection + harmonic fill for every query; no query depth is read."""
    geo = module("v27_geometry", "run_rgbd_geometric_baselines.py")
    inp = module("v27_inpaint", "run_rgbd_inpaint_baselines.py")
    if inp.NEAR_PX != config["near_px"]:
        raise PermissionError("NEAR radius differs from V17's")
    out_dir = Path(root) / "raw/baselines"
    out_dir.mkdir(parents=True)
    seal, access = {}, []
    found = entries(cohorts)
    for sid, (cohort, record, manifest, base) in found.items():
        roles = read(cohorts[cohort][2] / "stream_roles.json")["episodes"]
        ids = tuple(r["scene_id"] for r in manifest["scenes"])
        streams = StreamRGBDLoader(
            record,
            manifest["image_size"],
            base,
            "cpu",
            access,
            allowed_scene_ids=ids,
            holdout_scene_ids=tuple(manifest.get("protected_scene_ids", [])),
        )
        for role, context in eval_context(record, manifest, base, "cpu", access).items():
            stream = streams.stream(role)
            if [o.frame_id for o in stream] != roles[f"{sid}/{role}"]:
                raise PermissionError(f"{sid}/{role}: stream frames differ from the round's")
            frames = {
                "3": context,
                "7": RGBDContext([*context, *stream], torch.cat((context.depths, stream.depths))),
            }
            points = {k: geo.context_points(v) for k, v in frames.items()}
            for qid in record["roles"]["primary_query"]:
                camera = geo.query_camera(record, qid, manifest["image_size"])
                cosine = inp.ray_cosine(camera)
                arrays = {}
                for k, value in points.items():
                    depth, hit = geo.reproject(value, camera)
                    arrays[f"near{k}"] = inp.near_mask(hit)
                    arrays[f"harmonic{k}"] = inp.harmonic_fill(
                        depth, hit, cosine, config["reprojection_fill_constant_m"]
                    )
                if (arrays["near3"] & ~arrays["near7"]).any():
                    raise AssertionError("The 7-frame NEAR mask must contain the 3-frame one")
                path = out_dir / f"{sid}_{role}_{qid}.npz"
                np.savez_compressed(path, **arrays)
                seal[f"{sid}/{role}/{qid}"] = {"path": str(path), "sha256": sha(path)}
    preseal_depth_audit(access, {sid: item[1] for sid, item in found.items()})
    write_json(Path(root) / "raw/baseline_seal.json", {"query_depth_read": False, "files": seal})
    write_json(Path(root) / "raw/input_access.json", access)
    return seal


def pilot_agreement(seal, frozen):
    """Every TRAIN-EXT 3-frame NEAR mask present in the GPU pilot's seal must equal it."""
    if frozen["sha256"] is None:
        return {"compared": 0, "identical": 0}
    if sha(PILOT_SEAL) != frozen["sha256"]:
        raise PermissionError("The GPU pilot's TRAIN-EXT seal changed")
    pilot = read(PILOT_SEAL)["predictions"]
    compared = 0
    for key, item in seal.items():
        found = pilot.get(f"TRAIN_EXT/{key}")
        if found is None:
            continue
        if sha(Path(found["path"])) != found["sha256"]:
            raise PermissionError("A pilot sealed prediction changed")
        if not np.array_equal(
            np.load(found["path"])["near"].astype(bool), np.load(item["path"])["near3"]
        ):
            raise AssertionError(f"3-frame NEAR mask differs from the pilot's: {key}")
        compared += 1
    return {"compared": compared, "identical": compared}


def score(cohorts, seal):
    """Rows of every method on ALL valid pixels and on FAR7, from sealed files only."""
    found = entries(cohorts)
    rows = {"ALL": [], "FAR7": []}
    shares, worst, seeds = [], 0.0, set()
    for cohort, (_, _, round_dir) in cohorts.items():
        by_key = {}
        for row in read(round_dir / "raw/eval_ttt_query_results.json"):
            if row["policy"] in POLICIES and row["cohort"] == cohort:
                by_key.setdefault((row["scene_id"], row["role"], row["query_id"]), []).append(row)
        for (sid, role, qid), group in sorted(by_key.items()):
            seeds |= {r["seed"] for r in group}
            item = seal[f"{sid}/{role}/{qid}"]
            if sha(Path(item["path"])) != item["sha256"]:
                raise AssertionError("A sealed baseline changed")
            arrays = dict(np.load(item["path"]))
            _, record, _, base = found[sid]
            frame = next(f for f in record["frames"] if f["frame_id"] == qid)
            gt = np.load(resolve(frame["depth"], base)).squeeze()
            valid = np.isfinite(gt) & (gt > 0)
            near3, near7 = arrays["near3"], arrays["near7"]
            far7 = valid & ~near7
            shares.append(
                {
                    "cohort": cohort,
                    "near3_share": float((valid & near3).sum() / valid.sum()),
                    "near7_share": float((valid & near7).sum() / valid.sum()),
                }
            )
            methods = {}
            for row in group:
                path = round_dir / "raw/eval_ttt" / f"seed_{row['seed']}" / row["prediction_path"]
                if sha(path) != row["prediction_file_sha256"]:
                    raise AssertionError("A sealed round prediction changed")
                pred = np.load(path)
                depth = pred["depth"].astype(np.float64)
                opacity = pred["opacity"].astype(np.float64)
                worst = max(worst, abs(absrel(depth, gt, valid) - row["depth_absrel"]))
                m = methods.setdefault(
                    row["seed"],
                    {
                        "REPROJ3_HARMONIC": arrays["harmonic3"],
                        "REPROJ7_HARMONIC": arrays["harmonic7"],
                    },
                )
                policy = row["policy"]
                m[f"CARRIER_{policy}"] = depth
                if policy == "NO_STREAM":
                    m["HFILL8_3_NO_STREAM"] = hfill(near3, arrays["harmonic3"], depth)
                    continue
                harmonic = arrays["harmonic7"]
                normalized = depth / np.maximum(opacity, 1e-3)
                m[f"HFILL8_7_{policy}"] = hfill(near7, harmonic, depth)
                m[f"MAXFILL8_7_{policy}"] = maxfill(near7, harmonic, depth)
                m[f"AHFILL8_7_{policy}"] = ahfill(near7, opacity, harmonic, depth)
                m[f"HFILL8_7N_{policy}"] = hfill(near7, harmonic, normalized)
                m[f"MAXFILL8_7N_{policy}"] = maxfill(near7, harmonic, normalized)
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
                    rows["ALL"].append({**common, "depth_absrel": absrel(value, gt, valid)})
                    if far7.any():
                        rows["FAR7"].append({**common, "depth_absrel": absrel(value, gt, far7)})
    if worst > 1e-6:
        raise AssertionError(f"Carrier AbsRel differs from the rounds' rows: {worst}")
    return rows, shares, worst, sorted(seeds)


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
    p1 = statuses["P1_MAXFILL_VS_MODEL_FREE_STATUS"] == "SUPPORTED"
    p2 = statuses["P2_HYBRID_VS_MODEL_FREE_STATUS"] == "SUPPORTED"
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


def stat(value):
    lo, hi = value["ci95"]
    pos, tie, neg = value["positive_tied_negative"]
    return f"{value['mean']:+.4f} [{lo:+.4f}, {hi:+.4f}]，场景 {pos}/{tie}/{neg}"


def report(root, result):
    main, far = result["primary"], result["far7"]
    c, s = main["contrasts"], main["statuses"]
    lines = [
        f"# {EXPERIMENT}",
        "",
        "在两个从未用于这个问题的队列上（TRAIN-EXT 55 个场景，取 Core B V9 封存的预测；"
        "EVAL-V5 23 个场景，取 Core B V7 封存的预测），"
        "把混合方法与只用同样 7 帧的无模型基线（REPROJ7_HARMONIC）比较，"
        "并第一次预注册检验探索性分析得到的 MAXFILL 规则"
        "（NEAR7 用调和插值，其余取 carrier 与调和插值中更远的深度）。"
        "只在 CPU 上重新计分，不运行任何模型。计划见 [PLAN_BEFORE_V27.md](PLAN_BEFORE_V27.md)。",
        "",
        "- **P1 MAXFILL_VS_MODEL_FREE（REPROJ7_HARMONIC − MAXFILL8_7(TTT_100)）：** "
        f"{stat(c['MAXFILL_VS_MODEL_FREE'])}，{s['P1_MAXFILL_VS_MODEL_FREE_STATUS']}",
        "- **P2 HYBRID_VS_MODEL_FREE（REPROJ7_HARMONIC − HFILL8_7(TTT_100)）：** "
        f"{stat(c['HYBRID_VS_MODEL_FREE'])}，{s['P2_HYBRID_VS_MODEL_FREE_STATUS']}",
        f"- **分支 {main['INTERPRETATION_BRANCH']}**",
        "- **像素比例：** "
        + "；".join(
            f"{k} 的 3 帧 NEAR {100 * v['near3_share']:.1f}%、"
            f"7 帧 NEAR {100 * v['near7_share']:.1f}%"
            for k, v in result["region_shares"].items()
        ),
        "",
        "| 方法 | 全部像素 AbsRel | FAR7 AbsRel |",
        "|---|---|---|",
    ]
    for name in sorted(main["absolute"]):
        value = far["absolute"].get(name)
        far_text = f"{value:.4f}" if value is not None else "—"
        lines.append(f"| {name} | {main['absolute'][name]:.4f} | {far_text} |")
    lines += ["", "| 对比（正值：后者更好） | 全部像素 | FAR7 |", "|---|---|---|"]
    for name in CONTRASTS:
        lines.append(f"| {name} | {stat(c[name])} | {stat(far['contrasts'][name])} |")
    lines += [
        "",
        "- **各队列（P1 / P2）：** "
        + "；".join(
            f"{cohort} {stat(part['contrasts']['MAXFILL_VS_MODEL_FREE'])} / "
            f"{stat(part['contrasts']['HYBRID_VS_MODEL_FREE'])}"
            for cohort, part in result["per_cohort"].items()
        ),
        "- **每个 seed（P1 / P2）：** "
        + "；".join(
            f"{k}: {v['MAXFILL_VS_MODEL_FREE']:+.4f} / {v['HYBRID_VS_MODEL_FREE']:+.4f}"
            for k, v in main["per_seed_means"].items()
        ),
        f"- **核对：** TRAIN-EXT 的 3 帧 NEAR 掩码与 GPU 试点的封存比较 "
        f"{result['pilot_agreement']['compared']} 个，完全相同；carrier 的 AbsRel 与两轮的结果行"
        f"最大差 {result['carrier_absrel_max_diff']:.1e}。",
        "",
        "说明：MAXFILL 规则来自 V24、V25 数据上的探索性分析，本轮是它的第一次预注册检验。"
        "TRAIN-EXT 在 V18、V20 中用作训练数据，在 GPU 试点与 Core B V6、V9 中用作评价；"
        "EVAL-V5 在 Core B V7 中用过。受保护的 final holdout 未打开。",
    ]
    (Path(root) / "README.md").write_text("\n".join(lines) + "\n")


def run(root):
    root = Path(root).resolve()
    config = verify_lock(root)
    check_environment()
    torch.set_num_threads(1)
    cohorts = load_cohorts()
    seal = compute_baselines(root, config, cohorts)
    agreement = pilot_agreement(seal, config["pilot_mask_check"])
    rows, shares, worst, seeds = score(cohorts, seal)
    result = {
        "primary": analyze(rows["ALL"], seeds),
        "far7": analyze(rows["FAR7"], seeds, primary=False),
        "per_cohort": {
            name: analyze([r for r in rows["ALL"] if r["cohort"] == name], seeds, primary=False)
            for name in ROUNDS
        },
        "region_shares": {
            name: {
                k: float(np.mean([s[k] for s in shares if s["cohort"] == name]))
                for k in ("near3_share", "near7_share")
            }
            for name in ROUNDS
        },
        "pilot_agreement": agreement,
        "carrier_absrel_max_diff": worst,
    }
    result["INTERPRETATION_BRANCH"] = result["primary"]["INTERPRETATION_BRANCH"]
    write_json(root / "v27_results.json", result)
    for name, values in rows.items():
        write_json(root / f"raw/{name.lower()}_rows.json", values)
    old = read(root / "audit/previous_experiment_seal.json")["files"]
    changed = [p for p, info in old.items() if sha(Path(p)) != info["sha256"]]
    if changed:
        raise AssertionError(f"Old experiment files changed: {changed[:3]}")
    write_json(
        root / "integrity.json",
        {
            "status": "PASS",
            "rows_all": len(rows["ALL"]),
            "rows_far7": len(rows["FAR7"]),
            "old_files_unchanged": len(old),
            "model_run": False,
            "final_holdout_touched": False,
            "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "finished_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        },
    )
    report(root, result)
    print(
        "V27_COMPLETE",
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
