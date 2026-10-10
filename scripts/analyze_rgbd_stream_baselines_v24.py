#!/usr/bin/env python3
"""V24: Core B V8's sealed EVAL-V3/V4 predictions against model-free baselines on the same frames.

  --stage freeze  hash-locks the plan, the protocol, Core B V8's records and the sources (CPU);
  --stage run     computes the 3- and 7-frame reprojection + harmonic baselines (no GT), seals
                  them, rebuilds the hybrids from V8's sealed predictions, scores everything
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

EXPERIMENT = "EXP-3D-RGBD-STREAM-BASELINES-V24"
COREB_V8 = Path("outputs/EXP-3D-RGBD-DYNAMIC-TTT-SOURCE-V8")
V17 = Path("outputs/EXP-3D-RGBD-LONG-TRAIN-V17")
CONCURRENT = (
    "EXP-3D-RGBD-DYNAMIC-TTT-DOSE-V9",
    "EXP-3D-RGBD-DYNAMIC-TTT-CARRIER-V10",
    "EXP-3D-RGBD-DYNAMIC-TTT-ONLINE-V11",
)
PLAN = "PLAN_BEFORE_V24.md"
PLAN_SHA256 = "ff8ea7e0208bd00b22984da1f5e81699b8bd354655fd1b8add9665a63f18472f"
COHORTS = ("EVAL_V3", "EVAL_V4")
EXPECTED_SPLIT = {name: name for name in COHORTS}
V8_POLICIES = ("NO_STREAM", "OFF", "OFF_CTX_TTT_100", "TTT", "TTT_100", "TTT_100_WRONG_SCENE")
STREAM_POLICIES = V8_POLICIES[1:]
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
    "TTT_GAIN_IN_HYBRID": ("HFILL8_7_OFF", "HFILL8_7_TTT_100"),
    "HYBRID_VS_MODEL_FREE": ("REPROJ7_HARMONIC", "HFILL8_7_TTT_100"),
    "OFF_HYBRID_VS_MODEL_FREE": ("REPROJ7_HARMONIC", "HFILL8_7_OFF"),
    "CARRIER_TTT_100_VS_MODEL_FREE": ("REPROJ7_HARMONIC", "CARRIER_TTT_100"),
    "CARRIER_OFF_VS_MODEL_FREE": ("REPROJ7_HARMONIC", "CARRIER_OFF"),
    "CARRIER_TTT_GAIN": ("CARRIER_OFF", "CARRIER_TTT_100"),
    "CONTEXT_TTT_IN_HYBRID": ("HFILL8_7_OFF", "HFILL8_7_OFF_CTX_TTT_100"),
    "TTT10_IN_HYBRID": ("HFILL8_7_OFF", "HFILL8_7_TTT"),
    "SPECIFICITY_IN_HYBRID": ("HFILL8_7_TTT_100_WRONG_SCENE", "HFILL8_7_TTT_100"),
    "TTT_GAIN_IN_AHFILL8": ("AHFILL8_7_OFF", "AHFILL8_7_TTT_100"),
    "AHFILL8_VS_MODEL_FREE": ("REPROJ7_HARMONIC", "AHFILL8_7_TTT_100"),
    "TTT_GAIN_IN_HYBRID_NORM": ("HFILL8_7N_OFF", "HFILL8_7N_TTT_100"),
    "HYBRID_NORM_VS_MODEL_FREE": ("REPROJ7_HARMONIC", "HFILL8_7N_TTT_100"),
    "STREAM_FRAMES_FOR_MODEL_FREE": ("REPROJ3_HARMONIC", "REPROJ7_HARMONIC"),
    "CORE_A_HYBRID": ("REPROJ3_HARMONIC", "HFILL8_3_NO_STREAM"),
}
PRIMARY = {
    "P1_TTT_GAIN_IN_HYBRID_STATUS": "TTT_GAIN_IN_HYBRID",
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
    """Core B V3's scene aggregation and frozen V2 gate (as in Core B V4-V11)."""
    return module("coreb_v3_analysis_v24", "analyze_rgbd_stream_write_v3.py")


def resolve(path, base):
    path = Path(path)
    return path if path.is_absolute() else Path(base) / path


def hfill(near, harmonic, carrier):
    """HFILL8: the harmonic fill on NEAR pixels, the carrier everywhere else."""
    return np.where(near, harmonic, carrier)


def ahfill(near, opacity, harmonic, carrier, threshold=ABSTAIN_OPACITY):
    """AHFILL8: as HFILL8, and the harmonic fill also where the carrier's opacity is low."""
    return np.where(near | (opacity < threshold), harmonic, carrier)


def absrel(value, gt, mask):
    return float(np.mean(np.abs(value[mask] - gt[mask]) / gt[mask])) if mask.any() else None


def load_cohorts():
    """{cohort: (manifest, base)} from Core B V8's cohort lock, every manifest hash-checked."""
    lock = read(COREB_V8 / "cohort_lock.json")
    cohorts = {}
    for name in COHORTS:
        item = lock[name]
        if sha(Path(item["manifest"])) != item["manifest_sha256"]:
            raise PermissionError(f"{name} manifest changed")
        cohorts[name] = (read(Path(item["manifest"])), Path(item["manifest"]).parent)
    return cohorts


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
        raise PermissionError("V24 must be pinned to cores 32-39 (taskset)")


def freeze(root):
    root = Path(root).resolve()
    if root.name != EXPERIMENT or (root / "preregistration.json").exists():
        raise PermissionError("Exact, unfrozen V24 directory required")
    if not (root / "PROTOCOL.md").is_file() or sha(root / PLAN) != PLAN_SHA256:
        raise PermissionError("The V24 plan and protocol must be in place")
    check_environment()
    if read(COREB_V8 / "integrity.json")["status"] != "PASS":
        raise PermissionError("Core B V8 must be finalized")
    contract = read(V17 / "training_contract.json")["eval"]
    config = {
        "experiment": EXPERIMENT,
        "source_round": str(COREB_V8),
        "policies": list(V8_POLICIES),
        "reprojection_fill_constant_m": contract["reprojection_fill_constant_m"],
        "near_px": contract["near_px"],
        "abstain_opacity_threshold": ABSTAIN_OPACITY,
        "frames": "3-frame: the role's warmup context; 7-frame: warmup context + 4 stream frames",
        "thread_environment": THREADS,
        "cpu_affinity": sorted(CPU_CORES),
        "device": "cpu",
    }
    write_json(root / "analysis_contract.json", config)
    write_json(
        root / "decision_rules.json",
        {
            "aggregation": "per seed: scene mean of role means of query AbsRel; seeds equally "
            "weighted (model-free baselines take the same value in every seed)",
            "bootstrap": {"unit": "scene", "draws": DRAWS, "seed": BOOT_SEED},
            "gate": "frozen V2 primary-contrast checks (as Core B V2-V11)",
            "P1_TTT_GAIN_IN_HYBRID_STATUS": "gate on AbsRel(HFILL8_7(OFF)) - "
            "AbsRel(HFILL8_7(TTT_100))",
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
    inputs = [
        COREB_V8 / "integrity.json",
        COREB_V8 / "preregistration.json",
        COREB_V8 / "cohort_lock.json",
        COREB_V8 / "stream_roles.json",
        COREB_V8 / "raw/eval_ttt_query_results.json",
        V17 / "training_contract.json",
        root / "audit/previous_experiment_seal.json",
    ]
    sources = sorted(Path("src/mcss").rglob("*.py"))
    sources += [
        SCRIPTS / "analyze_rgbd_stream_baselines_v24.py",
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
    print("FROZEN_V24", sealed, "files sealed")


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
    geo = module("v24_geometry", "run_rgbd_geometric_baselines.py")
    inp = module("v24_inpaint", "run_rgbd_inpaint_baselines.py")
    if inp.NEAR_PX != config["near_px"]:
        raise PermissionError("NEAR radius differs from V17's")
    roles = read(COREB_V8 / "stream_roles.json")["episodes"]
    out_dir = Path(root) / "raw/baselines"
    out_dir.mkdir(parents=True)
    seal, access = {}, []
    entries = cohort_records(cohorts, EXPECTED_SPLIT)
    for _, record, manifest, base in entries.values():
        sid = record["scene_id"]
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
                raise PermissionError(f"{sid}/{role}: stream frames differ from Core B V8's")
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
    preseal_depth_audit(access, {sid: item[1] for sid, item in entries.items()})
    write_json(Path(root) / "raw/baseline_seal.json", {"query_depth_read": False, "files": seal})
    write_json(Path(root) / "raw/input_access.json", access)
    return seal


def v17_agreement(seal):
    """Every 3-frame NEAR mask and harmonic fill must equal V17's sealed EVAL-V3/V4 arrays."""
    seals = {}
    for name in COHORTS:
        path = V17 / "raw/eval" / name / "prediction_seal.json"
        if path.is_file():
            seals[name] = read(path)["predictions"]
    compared, worst = 0, 0.0
    for key, item in seal.items():
        found = [s[f"{n}/{key}"] for n, s in seals.items() if f"{n}/{key}" in s]
        if not found:
            continue
        if sha(Path(found[0]["path"])) != found[0]["sha256"]:
            raise PermissionError("A V17 sealed prediction changed")
        theirs = np.load(found[0]["path"])
        mine = np.load(item["path"])
        if not np.array_equal(theirs["near"].astype(bool), mine["near3"]):
            raise AssertionError(f"3-frame NEAR mask differs from V17's: {key}")
        worst = max(worst, float(np.max(np.abs(theirs["REPROJ_HARMONIC"] - mine["harmonic3"]))))
        compared += 1
    if worst > 1e-9:
        raise AssertionError(f"3-frame harmonic fill differs from V17's: {worst}")
    return {"compared": compared, "near_identical": compared, "harmonic_max_abs_diff": worst}


def score(root, cohorts, seal):
    """Rows of every method on ALL valid pixels and on FAR7, from sealed files only."""
    entries = cohort_records(cohorts, EXPECTED_SPLIT)
    v8_rows = read(COREB_V8 / "raw/eval_ttt_query_results.json")
    by_key = {}
    for row in v8_rows:
        by_key.setdefault((row["scene_id"], row["role"], row["query_id"]), []).append(row)
    seeds = sorted({r["seed"] for r in v8_rows})
    rows = {"ALL": [], "FAR7": []}
    shares, worst = [], 0.0
    for (sid, role, qid), group in sorted(by_key.items()):
        item = seal[f"{sid}/{role}/{qid}"]
        if sha(Path(item["path"])) != item["sha256"]:
            raise AssertionError("A sealed baseline changed")
        base_arrays = dict(np.load(item["path"]))
        cohort, record, _, base = entries[sid]
        frame = next(f for f in record["frames"] if f["frame_id"] == qid)
        gt = np.load(resolve(frame["depth"], base)).squeeze()
        valid = np.isfinite(gt) & (gt > 0)
        near3, near7 = base_arrays["near3"], base_arrays["near7"]
        far7 = valid & ~near7
        shares.append(
            {
                "near3_share": float((valid & near3).sum() / valid.sum()),
                "near7_share": float((valid & near7).sum() / valid.sum()),
            }
        )
        methods = {}
        for seed in seeds:
            methods[seed] = {
                "REPROJ3_HARMONIC": base_arrays["harmonic3"],
                "REPROJ7_HARMONIC": base_arrays["harmonic7"],
            }
        for row in group:
            path = COREB_V8 / "raw/eval_ttt" / f"seed_{row['seed']}" / row["prediction_path"]
            if sha(path) != row["prediction_file_sha256"]:
                raise AssertionError("A Core B V8 prediction changed")
            pred = np.load(path)
            depth = pred["depth"].astype(np.float64)
            opacity = pred["opacity"].astype(np.float64)
            worst = max(worst, abs(absrel(depth, gt, valid) - row["depth_absrel"]))
            policy, m = row["policy"], methods[row["seed"]]
            m[f"CARRIER_{policy}"] = depth
            if policy == "NO_STREAM":
                m["HFILL8_3_NO_STREAM"] = hfill(near3, base_arrays["harmonic3"], depth)
                continue
            harmonic = base_arrays["harmonic7"]
            m[f"HFILL8_7_{policy}"] = hfill(near7, harmonic, depth)
            m[f"AHFILL8_7_{policy}"] = ahfill(near7, opacity, harmonic, depth)
            normalized = depth / np.maximum(opacity, 1e-3)
            m[f"HFILL8_7N_{policy}"] = hfill(near7, harmonic, normalized)
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
        raise AssertionError(f"Carrier AbsRel differs from Core B V8's rows: {worst}")
    return rows, shares, worst, seeds


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
    p1 = statuses["P1_TTT_GAIN_IN_HYBRID_STATUS"] == "SUPPORTED"
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
    share = result["region_shares"]
    lines = [
        f"# {EXPERIMENT}",
        "",
        "把 Core B V8 在 EVAL-V3 ∪ EVAL-V4（83 个场景）上封存的预测，与使用同样帧的无模型基线比较："
        "把 3 个上下文帧与 4 个流帧（共 7 帧）的测得深度重投影到 query，再用调和插值补洞"
        "（REPROJ7_HARMONIC）。混合方法 HFILL8_7(p) 在 7 帧 NEAR 像素上用调和插值，"
        "其余像素用策略 p 的 carrier 深度。只在 CPU 上重新计分，不运行任何模型。"
        "计划见 [PLAN_BEFORE_V24.md](PLAN_BEFORE_V24.md)。",
        "",
        f"- **P1 TTT_GAIN_IN_HYBRID（HFILL8_7(OFF) − HFILL8_7(TTT_100)）：** "
        f"{stat(c['TTT_GAIN_IN_HYBRID'])}，{s['P1_TTT_GAIN_IN_HYBRID_STATUS']}",
        f"- **P2 HYBRID_VS_MODEL_FREE（REPROJ7_HARMONIC − HFILL8_7(TTT_100)）：** "
        f"{stat(c['HYBRID_VS_MODEL_FREE'])}，{s['P2_HYBRID_VS_MODEL_FREE_STATUS']}",
        f"- **分支 {main['INTERPRETATION_BRANCH']}**",
        f"- **像素比例：** 3 帧 NEAR 占有效像素 {100 * share['near3_share']:.1f}%，"
        f"7 帧 NEAR 占 {100 * share['near7_share']:.1f}%。",
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
            f"{cohort} {stat(part['contrasts']['TTT_GAIN_IN_HYBRID'])} / "
            f"{stat(part['contrasts']['HYBRID_VS_MODEL_FREE'])}"
            for cohort, part in result["per_cohort"].items()
        ),
        "- **每个 seed（P1 / P2）：** "
        + "；".join(
            f"{k}: {v['TTT_GAIN_IN_HYBRID']:+.4f} / {v['HYBRID_VS_MODEL_FREE']:+.4f}"
            for k, v in main["per_seed_means"].items()
        ),
        f"- **核对：** 3 帧的 NEAR 掩码与调和插值与 V17 封存的数组比较 "
        f"{result['v17_agreement']['compared']} 个，完全相同；carrier 的 AbsRel 与 Core B V8 "
        f"的结果行最大差 {result['carrier_absrel_max_diff']:.1e}。",
        "",
        "说明：EVAL-V3/V4 在 Core A 与 Core B V5、V8、V10、V11、V23 中都使用过；"
        "受保护的 final holdout 未打开。",
    ]
    (Path(root) / "README.md").write_text("\n".join(lines) + "\n")


def run(root):
    root = Path(root).resolve()
    config = verify_lock(root)
    check_environment()
    torch.set_num_threads(1)
    cohorts = load_cohorts()
    seal = compute_baselines(root, config, cohorts)
    agreement = v17_agreement(seal)
    rows, shares, worst, seeds = score(root, cohorts, seal)
    result = {
        "primary": analyze(rows["ALL"], seeds),
        "far7": analyze(rows["FAR7"], seeds, primary=False),
        "per_cohort": {
            name: analyze([r for r in rows["ALL"] if r["cohort"] == name], seeds, primary=False)
            for name in COHORTS
        },
        "region_shares": {k: float(np.mean([s[k] for s in shares])) for k in shares[0]},
        "v17_agreement": agreement,
        "carrier_absrel_max_diff": worst,
    }
    result["INTERPRETATION_BRANCH"] = result["primary"]["INTERPRETATION_BRANCH"]
    write_json(root / "v24_results.json", result)
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
        "V24_COMPLETE",
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
