#!/usr/bin/env python3
"""Report the frozen FRESH-V2 qualification of the V7 RGB-D carriers (saved results only)."""
# ruff: noqa: E501 -- Chinese scientific report prose

import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np


def read(root, name, default=None):
    path = root / (name + ".json")
    if not path.exists() and default is not None:
        return default
    return json.loads(path.read_text())


def stat(value):
    if value is None or value.get("mean") is None:
        return "UNDEFINED"
    lo, hi = value["ci95"]
    return f"{value['mean']:+.4f}，95% CI [{lo:+.4f}, {hi:+.4f}]"


def per_scene(root):
    rows = read(root, "raw/fresh_readout_rows", [])
    refs = read(root, "raw/fresh_reference_rows", [])
    table = defaultdict(dict)
    for variant in ("C0", "C1", "C2"):
        values = defaultdict(list)
        for r in rows:
            if r["variant"] == variant and r["method"] == "direct" and r["readout"] == "RAW":
                values[r["scene_id"]].append(r["depth_absrel"])
        for scene, v in values.items():
            table[scene][variant] = float(np.mean(v))
    constant = defaultdict(list)
    for r in refs:
        if r["reference"] == "REF_TRAIN_ABSREL_OPTIMAL" and r["readout"] == "RAW":
            constant[r["scene_id"]].append(r["depth_absrel"])
    for scene, v in constant.items():
        table[scene]["constant"] = float(np.mean(v))
    return dict(table)


def volume_of(scene):
    """Hypersim volume (ai_VVV_SSS -> VVV); any other id is its own volume."""
    parts = scene.split("_")
    return parts[1] if len(parts) == 3 and parts[0] == "ai" else scene


def volume_bootstrap(per_scene_values, draws=10000, seed=20260928):
    """Descriptive: mean of volume means, percentile CI over resampled volumes."""
    volumes = defaultdict(list)
    for scene, value in per_scene_values.items():
        volumes[volume_of(scene)].append(value)
    means = np.array([np.mean(v) for _, v in sorted(volumes.items())])
    rng = np.random.default_rng(seed)
    samples = means[rng.integers(0, len(means), (draws, len(means)))].mean(axis=1)
    lo, hi = np.percentile(samples, [2.5, 97.5])
    return {
        "mean": float(means.mean()),
        "ci95": [float(lo), float(hi)],
        "n_volumes": len(means),
        "volumes_positive": int((means > 0).sum()),
        "unit": "volume (descriptive only)",
        "draws": draws,
        "seed": seed,
    }


def run(root, docs=None):
    root = Path(root)
    results = read(root, "static_results")
    qualification = read(root, "qualification_results")
    controls = read(root, "wrong_scene_results")["contrasts"]
    obs = read(root, "observability_analysis")
    reference = read(root, "reference_results", {}) or None
    contract = read(root, "qualification_contract")
    v7 = Path(contract["v7_root"])
    v7_summary = (v7 / "terminal_summary.txt").read_text().splitlines()
    v7_fields = dict(line.split("=", 1) for line in v7_summary if "=" in line)
    if results["primary"] != "C1_SURFACE16":
        raise PermissionError("Frozen V2 estimator primary label expected (C1 = V7 primary)")
    groups = results["groups"]
    bounds = results["surface_gain"]
    wrong, shuffle = controls["C1|wrong_scene"], controls["C1|spatial_shuffle"]
    bounds_status = qualification["SURFACE_TRAINING_STATUS"]
    specificity = (
        "SUPPORTED" if wrong["ci95"][0] > 0 and shuffle["ci95"][0] > 0 else "NOT_ESTABLISHED"
    )
    static = qualification["STATIC_DEV_STATUS"]
    c1_reference = reference["carriers"]["V7_C1"]["RAW"]["depth_absrel"] if reference else None
    reference_gate = (
        "SUPPORTED"
        if c1_reference is not None and c1_reference["reference_label"] == "ABOVE"
        else "NOT_ESTABLISHED"
    )
    gates = (bounds_status, specificity, static, reference_gate)
    verdict = "QUALIFIED" if all(g == "SUPPORTED" for g in gates) else "NOT_QUALIFIED"
    reference_status = (
        reference["summary"]["CARRIER_REFERENCE_STATUS"] if reference else "NOT_RECORDED"
    )
    versus = c1_reference["vs_REF_TRAIN_ABSREL_OPTIMAL"] if c1_reference else None
    sensitivity = reference.get("sensitivity_excluding_no_hit_scenes", {}) if reference else {}
    volume_bounds = volume_bootstrap(bounds["per_scene"])
    volume_reference = volume_bootstrap(versus["per_scene"]) if versus else None
    table = per_scene(root)
    fields = {
        "FRESH_SCENES": len(results["scene_ids"]),
        "FRESH_VOLUMES": len({volume_of(s) for s in results["scene_ids"]}),
        "FRESH_SCENE_IDS": results["scene_ids"],
        "FROZEN_BOUNDS_QUERY_ABSREL_FRESH": round(
            groups["C0|direct"]["query_metrics"]["depth_absrel"]["mean"], 6
        ),
        "DEPTH_BOUNDS_QUERY_ABSREL_FRESH": round(
            groups["C1|direct"]["query_metrics"]["depth_absrel"]["mean"], 6
        ),
        "DEPTH_BOUNDS_TRIM1_QUERY_ABSREL_FRESH": round(
            groups["C2|direct"]["query_metrics"]["depth_absrel"]["mean"], 6
        )
        if "C2|direct" in groups
        else "NOT_EVALUATED",
        "BOUNDS_GAIN_FRESH": round(bounds["mean"], 6),
        "BOUNDS_GAIN_FRESH_CI": bounds["ci95"],
        "BOUNDS_GAIN_FRESH_IMPROVED_TIED_WORSE": [
            bounds["positive"],
            bounds["tied"],
            bounds["negative"],
        ],
        "BOUNDS_GAIN_FRESH_VOLUME_BOOTSTRAP": volume_bounds,
        "BOUNDS_GAIN_DEV_V7": v7_fields.get("BOUNDS_GAIN"),
        "C1_VS_CONSTANT_FRESH": round(versus["mean"], 6) if versus else None,
        "C1_VS_CONSTANT_FRESH_CI": versus["ci95"] if versus else None,
        "C1_VS_CONSTANT_FRESH_IMPROVED_TIED_WORSE": versus["improved_tied_worse"]
        if versus
        else None,
        "C1_VS_CONSTANT_FRESH_VOLUME_BOOTSTRAP": volume_reference,
        "C1_WRONG_SCENE_DAMAGE_FRESH": round(wrong["mean"], 6),
        "C1_WRONG_SCENE_DAMAGE_FRESH_CI": wrong["ci95"],
        "C1_SHUFFLE_DAMAGE_FRESH": round(shuffle["mean"], 6),
        "C1_SHUFFLE_DAMAGE_FRESH_CI": shuffle["ci95"],
        "C1_FULL_CONTEXT_GAIN_FRESH": round(results["full_context_gain"]["C1"]["mean"], 6),
        "C1_FULL_CONTEXT_GAIN_FRESH_CI": results["full_context_gain"]["C1"]["ci95"],
        "BOUNDS_STATUS_FRESH": bounds_status,
        "SCENE_SPECIFICITY_STATUS_FRESH": specificity,
        "STATIC_STATUS_FRESH": static,
        "REFERENCE_STATUS_FRESH": reference_gate,
        "SEED_ROBUSTNESS_FRESH": qualification["SEED_ROBUSTNESS"],
        "CARRIER_REFERENCE_STATUS_FRESH": reference_status,
        "C1_REFERENCE_LABEL_FRESH": c1_reference["reference_label"] if c1_reference else None,
        "C1_REFERENCE_LABEL_FRESH_HIT_SCENES": sensitivity.get("carriers", {})
        .get("V7_C1", {})
        .get("RAW", {})
        .get("depth_absrel", {})
        .get("reference_label", "NO_NO_HIT_SCENE"),
        "FRESH_QUALIFICATION_STATUS": verdict,
        "CORE_A_STATIC_CARRIER_RGBD_TRACK": verdict,
        "DYNAMIC_TTT_NEXT_STAGE_ALLOWED": verdict == "QUALIFIED",
        "FINAL_HOLDOUT_OPENED": False,
        "TRAINING_RUN": False,
        "DEVICE": "cpu",
    }
    lines = [
        "# EXP-3D-RGBD-FRESH-QUALIFICATION-V2",
        "",
        f"`FRESH_QUALIFICATION_STATUS={verdict}`。",
        "",
        "对 V7 已封存的 RGB-D carrier（每个 seed 各自的 DEV 选择，不重新训练、不重新选择）在第二个独立队列 FRESH-V2 上做预注册资格验证。"
        f"FRESH-V2 共 {fields['FRESH_SCENES']} 个场景、{fields['FRESH_VOLUMES']} 个 volume：来自官方 train 划分、从未被观察、从未被任何实验使用，volume 与 TRAIN72 和 DEV 都不重叠，每个 volume 至多 2 个。"
        "每个变体用自己的 bounds 规则；所有 fresh 状态封存之后才读取 query GT；统计用 V7 修订 1 的估计器（冻结的 V2 估计器去掉一条不适用的命中比例断言）；受保护的官方 final holdout 未打开。",
        "",
        "| 项目 | FRESH-V2 | V7 DEV（参照） |",
        "|---|---|---|",
        f"| C0 冻结 bounds，query AbsRel | {fields['FROZEN_BOUNDS_QUERY_ABSREL_FRESH']:.4f} | {v7_fields.get('FROZEN_BOUNDS_QUERY_ABSREL')} |",
        f"| C1 测得深度 bounds，query AbsRel | {fields['DEPTH_BOUNDS_QUERY_ABSREL_FRESH']:.4f} | {v7_fields.get('DEPTH_BOUNDS_QUERY_ABSREL')} |",
        f"| BOUNDS_GAIN | {stat(bounds)} | {v7_fields.get('BOUNDS_GAIN')} |",
        f"| C1 相对 AbsRel 最优常数 | {stat(versus)} | 见 V7 报告 |",
        f"| C1 wrong-scene 损伤 | {stat(wrong)} | {v7_fields.get('C1_WRONG_SCENE_DAMAGE')} |",
        f"| C1 shuffle 损伤 | {stat(shuffle)} | {v7_fields.get('C1_SHUFFLE_DAMAGE')} |",
        f"| 3 视角相对 anchor | {stat(results['full_context_gain']['C1'])} | {v7_fields.get('C1_FULL_CONTEXT_GAIN')} |",
        f"| BOUNDS_STATUS | {bounds_status} | {v7_fields.get('BOUNDS_STATUS')} |",
        f"| SCENE_SPECIFICITY_STATUS | {specificity} | {v7_fields.get('SCENE_SPECIFICITY_STATUS')} |",
        f"| STATIC_STATUS | {static} | {v7_fields.get('STATIC_DEV_STATUS')} |",
        f"| REFERENCE_STATUS（C1 在全部场景上优于常数） | {reference_gate} | {'SUPPORTED' if v7_fields.get('C1_ABOVE_REFERENCE_ALL_SCENES') == 'true' else 'NOT_ESTABLISHED'} |",
        "",
        "## 逐场景（RAW AbsRel，seed 与角色平均）",
        "",
        "| 场景 | volume | 常数（TRAIN 拟合） | C0 | C1 | C2 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for scene in sorted(table):
        t = table[scene]
        lines.append(
            f"| {scene} | {volume_of(scene)} | {t.get('constant', float('nan')):.3f} | {t.get('C0', float('nan')):.3f} | "
            f"{t.get('C1', float('nan')):.3f} | {t.get('C2', float('nan')):.3f} |"
        )
    lines += [
        "",
        "## 描述性补充",
        "",
        f"- 按 volume 聚类的 bootstrap（同一 volume 内的场景可能相关）：BOUNDS_GAIN {stat(volume_bounds)}，{volume_bounds['volumes_positive']}/{volume_bounds['n_volumes']} 个 volume 为正；"
        + (
            f"C1 相对常数 {stat(volume_reference)}，{volume_reference['volumes_positive']}/{volume_reference['n_volumes']} 个 volume 为正。"
            if volume_reference
            else "C1 相对常数未记录。"
        ),
        f"- CARRIER_REFERENCE_STATUS={reference_status}；排除 NO_HIT 场景后 C1 的标签：{fields['C1_REFERENCE_LABEL_FRESH_HIT_SCENES']}。",
        "- 区域诊断 C0−C1 条件 AbsRel："
        + "；".join(f"{r}={stat(obs['gains'][r])}" for r in ("OBS0", "OBS1", "OBS2PLUS"))
        + "。",
        "",
        "## 结论",
        "",
        (
            "四个冻结 gate 在第二个独立队列上全部成立：Core A 的静态 carrier 在 RGB-D 轨道上通过资格验证（首次同时满足机制复现与绝对精度优于平凡基线）。按项目规则，Core B（动态 TTT 写入）的下一阶段可以开启；官方 final holdout 仍保持关闭。"
            if verdict == "QUALIFIED"
            else "冻结 gate 未在第二个独立队列上全部成立：Core A 在 RGB-D 轨道上尚未通过资格验证，Core B 保持关闭。未通过的 gate 见上表。"
        ),
    ]
    summary = (
        "RGBD FRESH QUALIFICATION V2\n"
        + "\n".join(
            f"{k}={str(v).lower() if isinstance(v, bool) else v}" for k, v in fields.items()
        )
        + "\n"
    )
    (root / "README.md").write_text("\n".join(lines) + "\n")
    (root / "STATUS.md").write_text("```text\n" + summary + "```\n")
    (root / "terminal_summary.txt").write_text(summary)
    (root / "fresh_qualification_results.json").write_text(
        json.dumps({**fields, "per_scene": table}, indent=2, ensure_ascii=False) + "\n"
    )
    if docs:
        docs = Path(docs)
        docs.mkdir(parents=True, exist_ok=True)
        for p in root.iterdir():
            if p.is_file() and p.suffix in (".json", ".md", ".txt"):
                shutil.copyfile(p, docs / p.name)
    print(summary)
    return fields


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--docs", type=Path)
    args = parser.parse_args()
    run(args.root, args.docs)
