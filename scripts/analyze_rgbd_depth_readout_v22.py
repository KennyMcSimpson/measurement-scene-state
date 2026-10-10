#!/usr/bin/env python3
"""V22: is the FAR length gain of V17 the unnormalized depth readout? (EVAL-FAR, re-scoring only)

PLAN_BEFORE_V22.md fixed this round. The renderer's depth is sum_i w_i t_i, not divided by the
visibility sum_i w_i. V17-V20's sealed EVAL-FAR predictions (hash-checked against their seals) are
re-scored with RAW (the recorded depth) and NORM = depth / max(visibility, 1e-3), on V11's FAR,
NEAR and ALL regions, with V11's aggregation, contrast and frozen V2 gate. The RAW rows must
reproduce each run's recorded EVAL-FAR rows. No model runs; only already-read query depth is read.

  --stage freeze  writes the preregistration (plan, inputs and sources hashed);
  --stage run     re-scores, checks the reproduction, decides (V17) and describes (V18-V20).
Run from the repository root under cv_host_run.py (records store zonghan_o paths).
"""

from __future__ import annotations

import argparse
import datetime
import importlib.util
import subprocess
from pathlib import Path

import numpy as np

from mcss.mechanism_pilot.geometry_carrier_experiment import read
from mcss.mechanism_pilot.small_training import sha, write_json

EXPERIMENT = "EXP-3D-RGBD-DEPTH-READOUT-V22"
PLAN = "PLAN_BEFORE_V22.md"
PLAN_SHA256 = "022fe661dce5ee6fecd5ac9c4d3378c36ce3282904a3ae6a15863c9fb39c8051"
RUNS = {
    "V17": "EXP-3D-RGBD-LONG-TRAIN-V17",
    "V18": "EXP-3D-RGBD-LONG-TRAIN-DATA-V18",
    "V19": "EXP-3D-RGBD-LONG-TRAIN-WIDTH-V19",
    "V20": "EXP-3D-RGBD-LONG-TRAIN-WIDTH-DATA-V20",
}
PRIMARY_RUN = "V17"
COHORT = "EVAL_FAR"
LABELS = ("T6000", "T24000")
REGIONS = ("FAR", "NEAR", "ALL")
EPS = 1e-3
HOST = Path("scripts/evaluate_rgbd_completion_eval_v3.py")
SCRIPT = Path("scripts/analyze_rgbd_depth_readout_v22.py")
CONTRASTS = {
    "LENGTH_RAW": ("T6000", "T24000"),
    "LENGTH_NORM": ("T6000_NORM", "T24000_NORM"),
    "NORM_GAIN_6K": ("T6000", "T6000_NORM"),
    "NORM_GAIN_24K": ("T24000", "T24000_NORM"),
    "HARMONIC_MINUS_T24000": ("REPROJ_HARMONIC", "T24000"),
    "HARMONIC_MINUS_T24000_NORM": ("REPROJ_HARMONIC", "T24000_NORM"),
}


def host():
    spec = importlib.util.spec_from_file_location("v11_eval_host_v22", HOST)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def absrel(value, gt, mask):
    return float(np.mean(np.abs(value[mask] - gt[mask]) / gt[mask])) if mask.any() else None


def run_inputs(name):
    root = Path("outputs") / name
    return {
        "contract": root / "training_contract.json",
        "seal": root / "raw/eval" / COHORT / "prediction_seal.json",
        "rows": root / "raw/eval/query_rows.json",
        "results": root / "eval_results.json",
    }


def rescore(name):
    """(rows, FAR opacity rows) of one run's sealed EVAL-FAR predictions, RAW and NORM."""
    paths = run_inputs(name)
    config = read(paths["contract"])
    if read(paths["results"])["control"] != "PASS":
        raise PermissionError(f"{name}: built-in control not PASS")
    entry = config["eval"]["cohorts"][COHORT]
    if sha(Path(entry["manifest"])) != entry["manifest_sha256"]:
        raise PermissionError(f"{name}: {COHORT} manifest changed")
    manifest = read(Path(entry["manifest"]))
    seal = read(paths["seal"])
    rows, opacity = [], []
    for record in manifest["scenes"]:
        sid = record["scene_id"]
        frames = {f["frame_id"]: f for f in record["frames"]}
        for role in ("A", "B"):
            for qid in record["roles"][entry["role"]]:
                item = seal["predictions"][f"{COHORT}/{sid}/{role}/{qid}"]
                if sha(Path(item["path"])) != item["sha256"]:
                    raise PermissionError("A sealed prediction changed")
                arrays = dict(np.load(item["path"]))
                gt = np.load(frames[qid]["depth"]).squeeze().astype(np.float64)
                valid = np.isfinite(gt) & (gt > 0)
                near = arrays["near"].astype(bool)
                masks = {"ALL": valid, "NEAR": valid & near, "FAR": valid & ~near}
                common = {"cohort": COHORT, "scene_id": sid, "role": role, "query_id": qid}

                def scored(method, seed, value, common=common, masks=masks, gt=gt):
                    regions = {f"absrel_{r}": absrel(value, gt, m) for r, m in masks.items()}
                    return {**common, "method": method, "seed": seed, **regions}

                rows.append(scored("REPROJ_HARMONIC", 0, arrays["REPROJ_HARMONIC"]))
                for seed in config["seeds"]:
                    for label in LABELS:
                        raw = arrays[f"{label}_{seed}"].astype(np.float64)
                        visibility = arrays[f"OPACITY_{label}_{seed}"].astype(np.float64)
                        rows.append(scored(label, seed, raw))
                        rows.append(
                            scored(f"{label}_NORM", seed, raw / np.maximum(visibility, EPS))
                        )
                        if masks["FAR"].any():
                            # FAR mean opacity, stored under absrel_FAR to reuse V11's scene_values.
                            far_opacity = float(visibility[masks["FAR"]].mean())
                            opacity.append(
                                {**common, "method": label, "seed": seed, "absrel_FAR": far_opacity}
                            )
    return rows, opacity


def reproduction_error(name, rows):
    """Largest difference between the re-scored RAW rows and the run's recorded EVAL-FAR rows."""
    recorded = {
        (r["scene_id"], r["role"], r["query_id"], r["method"], r["seed"]): r
        for r in read(run_inputs(name)["rows"])
        if r["cohort"] == COHORT and r["method"] in (*LABELS, "REPROJ_HARMONIC")
    }
    worst, matched = 0.0, 0
    for row in rows:
        key = (row["scene_id"], row["role"], row["query_id"], row["method"], row["seed"])
        if key not in recorded:
            continue
        matched += 1
        for region in REGIONS:
            a, b = row[f"absrel_{region}"], recorded[key][f"absrel_{region}"]
            if (a is None) != (b is None):
                raise AssertionError("Region coverage differs from the recorded rows")
            if a is not None:
                worst = max(worst, abs(a - b))
    if matched != len(recorded):
        raise AssertionError(f"{name}: {matched} of {len(recorded)} recorded rows re-scored")
    return worst


def compact(stats):
    if stats is None:
        return None
    keys = ("mean", "ci95", "positive", "tied", "negative", "n_scenes")
    return {k: stats[k] for k in keys}


def decide(length_raw, length_norm, norm_gain_status):
    """The plan's rule: EXPLAINED iff the NORM residual's CI upper bound < half the RAW gain."""
    explained = length_norm["ci95"][1] < 0.5 * length_raw["mean"]
    if explained and norm_gain_status == "SUPPORTED":
        branch = "A"
    elif explained:
        branch = "B"
    else:
        branch = "C"
    return ("EXPLAINED" if explained else "NOT_EXPLAINED"), branch


def analyze(rows, opacity, v11):
    def values(method, region):
        return v11.scene_values(rows, method, f"absrel_{region}")

    contrasts = {}
    for region in REGIONS:
        for name, (a, b) in CONTRASTS.items():
            stats = v11.contrast(values(a, region), values(b, region))
            gate, checks = v11.gate(stats) if stats is not None else ("UNDEFINED", {})
            contrasts[f"{name}_{region}"] = {
                "stats": compact(stats),
                "gate": gate,
                "gate_checks": checks,
            }
    absolute = {
        f"{m}_{region}": float(np.mean(list(values(m, region).values())))
        for m in (*LABELS, *(f"{label}_NORM" for label in LABELS), "REPROJ_HARMONIC")
        for region in REGIONS
    }
    far_opacity = {
        label: float(np.mean(list(v11.scene_values(opacity, label, "absrel_FAR").values())))
        for label in LABELS
    }
    status, branch = decide(
        contrasts["LENGTH_RAW_FAR"]["stats"],
        contrasts["LENGTH_NORM_FAR"]["stats"],
        contrasts["NORM_GAIN_6K_FAR"]["gate"],
    )
    return {
        "absolute_scene_mean": absolute,
        "contrasts": contrasts,
        "far_mean_opacity": far_opacity,
        "EXPLANATION_STATUS": status,
        "NORM_GAIN_6K_STATUS": contrasts["NORM_GAIN_6K_FAR"]["gate"],
        "INTERPRETATION_BRANCH": branch,
    }


def freeze(root):
    root = Path(root).resolve()
    if root.name != EXPERIMENT or (root / "preregistration.json").exists():
        raise PermissionError("Exact, unfrozen V22 directory required")
    if sha(root / PLAN) != PLAN_SHA256:
        raise PermissionError("The V22 plan must be in place")
    manifests = set()
    for name in RUNS.values():
        entry = read(run_inputs(name)["contract"])["eval"]["cohorts"][COHORT]
        manifests.add((entry["manifest"], entry["manifest_sha256"]))
    if len(manifests) != 1:
        raise PermissionError("V17-V20 must share one EVAL-FAR manifest")
    inputs = [root / PLAN] + [p for name in RUNS.values() for p in run_inputs(name).values()]
    inputs.append(Path(next(iter(manifests))[0]))
    sources = sorted(Path("src/mcss").rglob("*.py")) + [SCRIPT, HOST]
    write_json(
        root / "preregistration.json",
        {
            "status": "FROZEN_BEFORE_ANALYSIS",
            "experiment": EXPERIMENT,
            "created_utc": datetime.datetime.now(datetime.UTC).isoformat(),
            "input_sha256": {str(p.resolve()): sha(p) for p in inputs},
            "source_sha256": {str(p.resolve()): sha(p) for p in sources},
            "readout_epsilon": EPS,
        },
    )
    print("FROZEN_V22", len(inputs), "inputs", len(sources), "sources")


def verify(root):
    lock = read(Path(root) / "preregistration.json")
    if lock["status"] != "FROZEN_BEFORE_ANALYSIS":
        raise PermissionError("V22 preregistration absent")
    for section in ("input_sha256", "source_sha256"):
        for path, digest in lock[section].items():
            if sha(Path(path)) != digest:
                raise PermissionError(f"Frozen input/source changed: {path}")


def stat(entry):
    s = entry["stats"]
    lo, hi = s["ci95"]
    return (
        f"{s['mean']:+.4f} [{lo:+.4f}, {hi:+.4f}]，场景 {s['positive']}/{s['tied']}/{s['negative']}"
    )


def report(root, results):
    p = results["runs"][PRIMARY_RUN]
    c = p["contrasts"]
    a = p["absolute_scene_mean"]
    lines = [
        f"# {EXPERIMENT}",
        "",
        "长训练的补全增益是否来自未归一化的深度读出。只在 CPU 上重新计分 V17–V20 已封存的 "
        "EVAL-FAR 预测（79 个场景，远处 query），不运行任何模型。"
        "计划见 [PLAN_BEFORE_V22.md](PLAN_BEFORE_V22.md)。",
        "",
        f"- **V17 长训练增益（FAR，第 6000 − 24000 步）：** RAW {stat(c['LENGTH_RAW_FAR'])}；"
        f"NORM {stat(c['LENGTH_NORM_FAR'])}",
        f"- **主判定一（解释）：** {p['EXPLANATION_STATUS']}"
        f"（NORM 残差的 CI 上界 {c['LENGTH_NORM_FAR']['stats']['ci95'][1]:+.4f}，"
        f"RAW 增益的一半 {0.5 * c['LENGTH_RAW_FAR']['stats']['mean']:+.4f}）",
        f"- **主判定二（第 6000 步的归一化增益）：** {stat(c['NORM_GAIN_6K_FAR'])}，"
        f"{p['NORM_GAIN_6K_STATUS']}",
        f"- **分支 {p['INTERPRETATION_BRANCH']}**",
        f"- **FAR 平均不透明度：** 第 6000 步 {p['far_mean_opacity']['T6000']:.3f}，"
        f"第 24000 步 {p['far_mean_opacity']['T24000']:.3f}",
        "",
        "| V17 FAR AbsRel | RAW | NORM |",
        "|---|---|---|",
        f"| 第 6000 步 | {a['T6000_FAR']:.4f} | {a['T6000_NORM_FAR']:.4f} |",
        f"| 第 24000 步 | {a['T24000_FAR']:.4f} | {a['T24000_NORM_FAR']:.4f} |",
        f"| 调和插值 | {a['REPROJ_HARMONIC_FAR']:.4f} | — |",
        "",
        "| run（描述性） | 长训练增益 RAW | 长训练增益 NORM | 第 6000 步 NORM 增益 | 分支 |",
        "|---|---|---|---|---|",
    ]
    for label, part in results["runs"].items():
        pc = part["contrasts"]
        lines.append(
            f"| {label} | {stat(pc['LENGTH_RAW_FAR'])} | {stat(pc['LENGTH_NORM_FAR'])} | "
            f"{stat(pc['NORM_GAIN_6K_FAR'])} | {part['INTERPRETATION_BRANCH']} |"
        )
    lines += [
        "",
        "- **调和插值 − carrier（FAR，第 24000 步）：** "
        f"RAW {stat(c['HARMONIC_MINUS_T24000_FAR'])}；"
        f"NORM {stat(c['HARMONIC_MINUS_T24000_NORM_FAR'])}",
        f"- **复现：** RAW 与各 run 记录的 EVAL-FAR 计分的最大差 "
        f"{max(results['reproduction_max_error'].values()):.2e}",
        "",
        "说明：这一假设是在 EVAL-V3/V4 上事后发现的；"
        "EVAL-FAR 的 query 深度在 V17–V20 计分时已经读过，但从未用于归一化读出。"
        "只有 V17 用于判定，V18–V20 是描述性的重复。受保护的 final holdout 未打开。",
    ]
    (Path(root) / "README.md").write_text("\n".join(lines) + "\n")


def run(root):
    root = Path(root).resolve()
    verify(root)
    v11 = host()
    results = {"runs": {}, "reproduction_max_error": {}}
    for label, name in RUNS.items():
        rows, opacity = rescore(name)
        results["reproduction_max_error"][label] = reproduction_error(name, rows)
        if results["reproduction_max_error"][label] > 1e-9:
            raise AssertionError(f"{label}: RAW does not reproduce the recorded rows")
        results["runs"][label] = analyze(rows, opacity, v11)
        write_json(root / f"raw/{label}_rows.json", rows)
        print(label, results["runs"][label]["INTERPRETATION_BRANCH"], flush=True)
    primary = results["runs"][PRIMARY_RUN]
    results["INTERPRETATION_BRANCH"] = primary["INTERPRETATION_BRANCH"]
    results["EXPLANATION_STATUS"] = primary["EXPLANATION_STATUS"]
    results["NORM_GAIN_6K_STATUS"] = primary["NORM_GAIN_6K_STATUS"]
    write_json(root / "results.json", results)
    write_json(
        root / "integrity.json",
        {
            "status": "PASS",
            "reproduction_max_error": results["reproduction_max_error"],
            "models_run": False,
            "final_holdout_touched": False,
            "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "finished_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        },
    )
    report(root, results)
    print(
        "V22_COMPLETE", results["INTERPRETATION_BRANCH"], results["EXPLANATION_STATUS"], flush=True
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
