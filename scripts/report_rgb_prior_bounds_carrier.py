#!/usr/bin/env python3
"""Report saved V10 bounds-prior DEV results without opening fresh data or models."""
# ruff: noqa: E501 -- Chinese scientific report prose

import argparse
import json
import math
from pathlib import Path

import numpy as np

LABELS = {"C0": "FROZEN_PRIOR_GRID32", "C1": "TRAIN72_PRIOR_GRID32", "C2": "TRAIN72_PRIOR_GRID16"}


def read(root, name, default=None):
    path = root / (name + ".json")
    if not path.exists() and default is not None:
        return default
    return json.loads(path.read_text())


def number(value):
    return "UNDEFINED" if value is None else f"{value:.6f}"


def stat(value):
    if value is None or value.get("mean") is None:
        return "UNDEFINED"
    lo, hi = value["ci95"]
    return f"{value['mean']:.6f}，95% CI [{lo:.6f}, {hi:.6f}]"


def negated(value):
    if value is None:
        return None
    lo, hi = value["ci95"]
    return {
        **value,
        "mean": -value["mean"],
        "ci95": [-hi, -lo],
        "positive": value["negative"],
        "negative": value["positive"],
    }


def train_fit(curves, steps):
    rows = curves.get("training", [])
    fit = {}
    for variant in sorted({r["variant"] for r in rows}):
        per_seed = []
        for seed in sorted({r["seed"] for r in rows if r["variant"] == variant}):
            tail = [
                (r["terms"]["A"]["loss_depth"] + r["terms"]["B"]["loss_depth"]) / 2
                for r in rows
                if r["variant"] == variant and r["seed"] == seed and r["step"] > steps - 100
            ]
            if tail:
                per_seed.append(sum(tail) / len(tail))
        fit[variant] = sum(per_seed) / len(per_seed) if per_seed else None
    return fit


def last_step_dev(curves, steps):
    """Selection-free descriptive DEV AbsRel at the final checkpoint (seed mean)."""
    rows = [r for r in curves.get("dev", []) if r["step"] == steps]
    return {
        v: float(np.mean([r["query_absrel"] for r in rows if r["variant"] == v]))
        for v in sorted({r["variant"] for r in rows})
    }


def learned_sigma(root):
    sigmas = {}
    for variant, name in (
        ("C0", "frozen_prior_grid32_training"),
        ("C1", "train72_prior_grid32_training"),
        ("C2", "train72_prior_grid16_training"),
    ):
        rows = read(root, name, [])
        values = [
            math.exp(r["depth_log_sigma"]) for r in rows if r.get("depth_log_sigma") is not None
        ]
        sigmas[variant] = [round(v, 6) for v in values]
    return sigmas


def reference_labels(block, variants):
    return {
        v: {
            f"{readout}:{metric}": block[f"V10_{v}"][readout][metric]["reference_label"]
            for readout in ("RAW", "OPACITY_NORMALIZED", "MEDIAN_SCALED")
            for metric in ("depth_absrel", "depth_delta1")
        }
        for v in variants
        if f"V10_{v}" in block
    }


def run(root, docs=None):
    root, docs = Path(root), Path(docs) if docs else Path(root)
    results = read(root, "static_results")
    qualification = read(root, "qualification_results")
    controls = read(root, "wrong_scene_results")["contrasts"]
    obs = read(root, "observability_analysis")
    contract = read(root, "training_contract", {"steps": 6000})
    curves = read(root, "training_curves", {"training": [], "dev": []})
    selected = read(root, "selected_checkpoints", {"status": "NOT_RECORDED"})
    roster = read(root, "evaluated_variants", {"secondary_status": {}})
    incidents = read(root, "audit/incidents", [])
    reference = read(root, "reference_results", {}) or None
    tests, integrity = (read(root, n, {"status": "PENDING"}) for n in ("tests", "integrity"))
    if results["primary"] != "C1_SURFACE16":
        raise PermissionError("Frozen V2 estimator primary label expected (C1 = V10 primary)")
    if (
        qualification["FRESH_QUALIFICATION_OPENED"]
        or qualification["DYNAMIC_TTT_NEXT_STAGE_ALLOWED"]
    ):
        raise PermissionError("V10 is a DEV mechanism round; fresh and Dynamic TTT stay closed")
    groups = results["groups"]
    scale = results["surface_gain"]
    secondary = "C2|direct" in groups
    variants = ("C0", "C1", "C2") if secondary else ("C0", "C1")
    depth_at_scale = negated(results.get("free_space_extra_gain"))
    c2_status = roster["secondary_status"].get("C2", "NOT_RUN_SECONDARY_OPTIONAL")
    wrong, shuffle = controls["C1|wrong_scene"], controls["C1|spatial_shuffle"]
    specificity = (
        "SUPPORTED" if wrong["ci95"][0] > 0 and shuffle["ci95"][0] > 0 else "NOT_ESTABLISHED"
    )
    scale_status = qualification["SURFACE_TRAINING_STATUS"]
    labels_all = reference_labels(reference["carriers"], variants) if reference else {}
    sensitivity = (reference or {}).get("sensitivity_excluding_no_hit_scenes", {})
    labels_hit = (
        reference_labels(sensitivity["carriers"], variants)
        if "carriers" in sensitivity
        else labels_all
    )
    above_all = labels_all.get("C1", {}).get("RAW:depth_absrel") == "ABOVE"
    above = labels_hit.get("C1", {}).get("RAW:depth_absrel") == "ABOVE"
    if scale_status == "SUPPORTED" and above_all:
        branch = "A"
    elif scale["ci95"][0] > 0 or above_all or above:
        branch = "B"
    else:
        branch = "C"
    next_step = {
        "A": "改用 TRAIN72 拟合的深度先验确定 bounds 后，RGB-only carrier 在全部 DEV 场景上稳定超过平凡基线，此前关于 RGB-only 已到信息上限的判断撤回。下一轮需要新的独立队列做资格验证；仍不开 Dynamic TTT。",
        "B": "bounds 先验有部分作用，但 RGB-only 仍未稳定超过平凡基线。下一轮检验光度证据（分辨率或多视角匹配）；仍不开 Dynamic TTT。",
        "C": "修正 bounds 截断没有让 RGB-only 稳定改善：RGB-only 的失败不是 bounds 造成的。维持 RGB-D 轨道；仍不开 Dynamic TTT。",
    }[branch]
    fit = train_fit(curves, contract["steps"])
    last = last_step_dev(curves, contract["steps"])
    fields = {
        "N_TRAIN_SCENES_BY_VARIANT": {"C0": 72, "C1": 72, "C2": 72},
        "GRID_BY_VARIANT": {"C0": 32, "C1": 32, "C2": 16},
        "PRIOR_BY_VARIANT": {"C0": "FROZEN_V2_PRIOR", "C1": "TRAIN72_PRIOR", "C2": "TRAIN72_PRIOR"},
        "MEAN_RAY_HIT_FRACTION": {
            v: number(float(np.mean(list(per.values()))))
            for v, per in read(root, "ray_hit_fractions", {"per_variant_scene": {}})[
                "per_variant_scene"
            ].items()
        },
        "BOUNDS_RULE": contract.get("bounds_rule", "NOT_RECORDED"),
        "N_DEV_SCENES": len(results["scene_ids"]),
        "DEVICE": "cpu",
        "STEPS": contract["steps"],
        "SECONDARY_C2_STATUS": c2_status,
        "FROZEN_PRIOR_GRID32_QUERY_ABSREL": number(
            groups["C0|direct"]["query_metrics"]["depth_absrel"]["mean"]
        ),
        "TRAIN72_PRIOR_GRID32_QUERY_ABSREL": number(
            groups["C1|direct"]["query_metrics"]["depth_absrel"]["mean"]
        ),
        "TRAIN72_PRIOR_GRID16_QUERY_ABSREL": number(
            groups["C2|direct"]["query_metrics"]["depth_absrel"]["mean"]
        )
        if secondary
        else "NOT_RUN_SECONDARY_OPTIONAL",
        "PRIOR_BOUNDS_GAIN": number(scale["mean"]),
        "PRIOR_BOUNDS_GAIN_CI": scale["ci95"],
        "GRID16_GAP": number(depth_at_scale["mean"]) if depth_at_scale else "NOT_RUN",
        "GRID16_GAP_CI": depth_at_scale["ci95"] if depth_at_scale else "NOT_RUN",
        "LAST_STEP_DEV_ABSREL": {v: number(x) for v, x in last.items()},
        "SELECTED_STEPS": {
            v: {s: x["step"] for s, x in seeds.items()} for v, seeds in selected.items()
        }
        if isinstance(selected, dict) and "status" not in selected
        else selected,
        "TRAIN_DEPTH_ABSREL_LAST100": {v: number(fit.get(v)) for v in variants},
        "C1_WRONG_SCENE_DAMAGE": number(wrong["mean"]),
        "C1_WRONG_SCENE_DAMAGE_CI": wrong["ci95"],
        "C1_SHUFFLE_DAMAGE": number(shuffle["mean"]),
        "C1_SHUFFLE_DAMAGE_CI": shuffle["ci95"],
        "C1_FULL_CONTEXT_GAIN": number(results["full_context_gain"]["C1"]["mean"]),
        "OBS2PLUS_PRIOR_BOUNDS_GAIN": number(obs["gains"]["OBS2PLUS"]["mean"]),
        "CARRIER_REFERENCE_STATUS": reference["summary"]["CARRIER_REFERENCE_STATUS"]
        if reference
        else "NOT_RECORDED",
        "REFERENCE_LABELS_ALL_SCENES": labels_all,
        "REFERENCE_LABELS_HIT_SCENES": labels_hit,
        "C1_ABOVE_REFERENCE_HIT_SCENES": above,
        "C1_ABOVE_REFERENCE_ALL_SCENES": above_all,
        "PRIOR_BOUNDS_STATUS": scale_status,
        "SCENE_SPECIFICITY_STATUS": specificity,
        "STATIC_DEV_STATUS": qualification["STATIC_DEV_STATUS"],
        "SEED_ROBUSTNESS": qualification["SEED_ROBUSTNESS"],
        "INTERPRETATION_BRANCH": branch,
        "FRESH_QUALIFICATION_INCLUDED": False,
        "FINAL_STATIC_STATUS": "NOT_ESTABLISHED",
        "DYNAMIC_TTT_NEXT_STAGE_ALLOWED": False,
        "TEST_TIME_INPUT": "RGB+CAMERA",
        "FINAL_HOLDOUT_TOUCHED": False,
        "INFRASTRUCTURE_RETRIES": len(incidents),
        "TESTS": tests["status"],
        "FINAL_INTEGRITY": integrity["status"],
        "REPORT_DIR": str(docs),
    }
    lines = [
        "# EXP-3D-RGB-PRIOR-BOUNDS-V10",
        "",
        "单因素机制诊断：RGB-only 轨道上 GT-free bounds 的深度先验。三个变体都是 V5 C0 的 RGB-only 配方（不含深度旁路的稠密 carrier、冻结的 V2 表面 loss、72 个训练场景、6000 步、与 V5–V9 相同的逐 seed 初始化），测试时只输入 RGB 与相机。bounds 都按冻结的 GT-free 规则由上下文相机射线投到先验的近、远距离确定，只有先验不同：C0 用冻结的 V2 先验（3 个训练场景 12 帧拟合），C1 用 TRAIN72 全部帧拟合的先验，二者都是 32³；C2 为 C1 的 16³ 版本（次要、不进入 gate）。全程 CPU。"
        + f"PRIOR_BOUNDS_GAIN=AbsRel(C0)−AbsRel(C1)={stat(scale)}；PRIOR_BOUNDS_STATUS={scale_status}；解读分支={branch}。",
        "",
        "DEV 的 8 个场景已在 V2–V9 中被评价，本轮是 DEV 机制诊断；FRESH-V1、FRESH-V2 都未使用。checkpoint 仍按 V2 冻结的 DEV 规则选择，另报不做选择的最后一步 DEV AbsRel。",
        "",
        "| Variant | 训练场景 | Query AbsRel (95% CI) | 最后一步 DEV AbsRel | 训练末100步深度AbsRel |",
        "|---|---:|---|---:|---:|",
    ]
    for v in variants:
        g = groups[v + "|direct"]
        lines.append(
            f"| {v} {LABELS[v]} | {fields['N_TRAIN_SCENES_BY_VARIANT'][v]} | {stat(g['query_metrics']['depth_absrel'])} | {number(last.get(v))} | {number(fit.get(v))} |"
        )
    answers = [
        (
            "改用 TRAIN72 拟合的先验确定 bounds，RGB-only 的 DEV 几何是否改善？",
            f"PRIOR_BOUNDS_GAIN={stat(scale)}；PRIOR_BOUNDS_STATUS={scale_status}（V2 冻结主对比 gate）。",
        ),
        (
            "是否跨场景与 seed？",
            f"improved/tied/worse={scale['positive']}/{scale['tied']}/{scale['negative']}；LOSO 范围={scale['loso_range']}；SEED_ROBUSTNESS={qualification['SEED_ROBUSTNESS']}。",
        ),
        (
            "不做 checkpoint 选择时差多少？",
            "最后一步（6000 步）DEV AbsRel="
            + json.dumps({v: number(x) for v, x in last.items()}, ensure_ascii=False)
            + "；选中步="
            + json.dumps(fields["SELECTED_STEPS"], ensure_ascii=False)
            + "。",
        ),
        (
            "状态是否场景专属、多视角是否优于单视角？",
            f"C1 wrong-scene={stat(wrong)}；shuffle={stat(shuffle)}；SCENE_SPECIFICITY_STATUS={specificity}；3 视角相对 anchor={stat(results['full_context_gain']['C1'])}；STATIC_DEV_STATUS={qualification['STATIC_DEV_STATUS']}。",
        ),
        (
            "TRAIN72 先验下 16³ 与 32³ 差多少（C2 对 C1）？",
            (
                f"GRID16_GAP=AbsRel(C2 16³)−AbsRel(C1 32³)={stat(depth_at_scale)}；正值表示 32³ 优于 16³。仅描述。"
                if depth_at_scale
                else f"C2 未纳入评价（{c2_status}）。"
            ),
        ),
        (
            "与无几何常数相比如何？",
            "全部场景标签="
            + json.dumps(labels_all.get("C1", {}), ensure_ascii=False)
            + "；排除 NO_HIT 场景后="
            + json.dumps(labels_hit.get("C1", {}), ensure_ascii=False)
            + f"；C1_ABOVE_REFERENCE_ALL_SCENES={above_all}"
            + f"；C1_ABOVE_REFERENCE_HIT_SCENES={above}。",
        ),
        (
            "训练集拟合如何？",
            "最后100步训练深度AbsRel="
            + json.dumps({v: number(fit.get(v)) for v in variants}, ensure_ascii=False)
            + "；射线命中比例（场景平均）="
            + json.dumps(fields["MEAN_RAY_HIT_FRACTION"], ensure_ascii=False)
            + "。",
        ),
        (
            "区域诊断？",
            "C0−C1 条件 AbsRel："
            + "；".join(f"{r}={stat(obs['gains'][r])}" for r in ("OBS0", "OBS1", "OBS2PLUS"))
            + "。",
        ),
        (
            "结论属于哪个预注册分支？",
            f"分支{branch}：{next_step} DYNAMIC_TTT_NEXT_STAGE_ALLOWED=false；FINAL_STATIC_STATUS=NOT_ESTABLISHED。",
        ),
    ]
    for i, (question, answer) in enumerate(answers, 1):
        lines.extend(["", f"**{i}. {question}**", "", answer])
    summary = (
        "RGB PRIOR BOUNDS V10 FINAL\n"
        + "\n".join(
            f"{k}={str(v).lower() if isinstance(v, bool) else v}" for k, v in fields.items()
        )
        + "\n"
    )
    (root / "README.md").write_text("\n".join(lines) + "\n")
    (root / "STATUS.md").write_text("```text\n" + summary + "```\n")
    (root / "terminal_summary.txt").write_text(summary)
    print(summary)
    return fields


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--docs", type=Path)
    a = p.parse_args()
    run(a.root, a.docs)
