#!/usr/bin/env python3
"""Report saved V5 RGB-D evidence DEV results without opening fresh data or models."""
# ruff: noqa: E501 -- Chinese scientific report prose

import argparse
import json
import math
from pathlib import Path

LABELS = {
    "C0": "DENSE_SURFACE",
    "C1": "DEPTH_SURFACE",
    "C2": "DEPTH_NO_SURFACE",
}


def read(root, name, default=None):
    path = root / (name + ".json")
    if not path.exists() and default is not None:
        return default
    return json.loads(path.read_text())


def number(value):
    return "UNDEFINED" if value is None else f"{value:.6f}"


def stat(value):
    if value is None or value.get("mean") is None:
        return "UNDEFINED（空区域未填零）"
    ci = value["ci95"]
    return f"{value['mean']:.6f}，95% CI [{ci[0]:.6f}, {ci[1]:.6f}]"


def negated(value):
    """Exact sign flip of a paired scene contrast: mean, CI and counts."""
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


def learned_sigma(root):
    sigmas = {}
    for variant, name in (("C1", "depth_training"), ("C2", "depth_no_surface_training")):
        rows = read(root, name, [])
        values = [
            math.exp(r["depth_log_sigma"]) for r in rows if r.get("depth_log_sigma") is not None
        ]
        sigmas[variant] = [round(v, 6) for v in values]
    return sigmas


def reference_summary(reference, variants):
    if reference is None:
        return "NOT_RECORDED", {}
    labels = {}
    for v in variants:
        carrier = reference["carriers"].get(f"V5_{v}")
        if carrier is None:
            continue
        labels[v] = {
            f"{readout}:{metric}": carrier[readout][metric]["reference_label"]
            for readout in ("RAW", "OPACITY_NORMALIZED", "MEDIAN_SCALED")
            for metric in ("depth_absrel", "depth_delta1")
        }
    return reference["summary"]["CARRIER_REFERENCE_STATUS"], labels


def run(root, docs=None):
    root, docs = Path(root), Path(docs) if docs else Path(root)
    results = read(root, "static_results")
    qualification = read(root, "qualification_results")
    state = read(root, "state_use_results")
    controls = read(root, "wrong_scene_results")["contrasts"]
    obs = read(root, "observability_analysis")
    split = read(root, "scene_split")
    contract = read(root, "training_contract", {"steps": 2000})
    curves = read(root, "training_curves", {"training": [], "dev": []})
    selected = read(root, "selected_checkpoints", {"status": "NOT_RECORDED"})
    roster = read(root, "evaluated_variants", {"secondary_status": {}})
    correlation = read(root, "state_correlation", {"per_variant_mean_over_seeds": {}})
    incidents = read(root, "audit/incidents", [])
    reference = read(root, "reference_results", {}) or None
    tests, integrity = (read(root, n, {"status": "PENDING"}) for n in ("tests", "integrity"))
    if results["primary"] != "C1_SURFACE16":
        raise PermissionError("Frozen V2 estimator primary label expected (C1 = V5 primary)")
    if (
        qualification["FRESH_QUALIFICATION_OPENED"]
        or qualification["DYNAMIC_TTT_NEXT_STAGE_ALLOWED"]
    ):
        raise PermissionError("V5 is a DEV mechanism round; fresh and Dynamic TTT stay closed")
    groups = results["groups"]
    depth = results["surface_gain"]
    secondary = "C2|direct" in groups
    surface_with_depth = negated(results.get("free_space_extra_gain"))
    c2_status = roster["secondary_status"].get("C2", "NOT_RUN_SECONDARY_OPTIONAL")
    wrong, shuffle = controls["C1|wrong_scene"], controls["C1|spatial_shuffle"]
    specificity = (
        "SUPPORTED" if wrong["ci95"][0] > 0 and shuffle["ci95"][0] > 0 else "NOT_ESTABLISHED"
    )
    depth_status = qualification["SURFACE_TRAINING_STATUS"]
    variants = ("C0", "C1", "C2") if secondary else ("C0", "C1")
    reference_status, reference_labels = reference_summary(reference, variants)
    # V2-V4 carriers never beat the geometry-free constant, so A also requires C1 to.
    above_reference = reference_labels.get("C1", {}).get("RAW:depth_absrel") == "ABOVE"
    if depth_status == "SUPPORTED" and specificity == "SUPPORTED" and above_reference:
        branch = "A"
    elif depth["ci95"][0] > 0 or specificity == "SUPPORTED" or above_reference:
        branch = "B"
    else:
        branch = "C"
    next_step = {
        "A": "测得的上下文深度让learned carrier形成场景专属几何（RGB-D轨道）。下一轮在RGB-D轨道上建立经过独立性审计的fresh队列做资格验证，并准备Core B的前置检查；仍不开Dynamic TTT。",
        "B": "深度证据有部分作用。下一轮检验carrier容量（grid分辨率）或训练规模；仍不开Dynamic TTT。",
        "C": "即使输入测得的上下文深度，16³ carrier在24个TRAIN场景上仍不能在DEV上泛化：瓶颈在carrier或训练规模，而不只是证据。下一轮另写协议；仍不开Dynamic TTT。",
    }[branch]
    fit = train_fit(curves, contract["steps"])
    sigmas = learned_sigma(root)
    corr = correlation["per_variant_mean_over_seeds"]
    fields = {
        "N_TRAIN_SCENES": sum(r.get("split") == "TRAIN" for r in split["scenes"]),
        "N_DEV_SCENES": len(results["scene_ids"]),
        "DEV_REUSED_FROM_V2_V3_V4": True,
        "DENSE_SURFACE_CHECKPOINT": selected.get("C0", "NOT_RECORDED"),
        "DEPTH_SURFACE_CHECKPOINT": selected.get("C1", "NOT_RECORDED"),
        "DEPTH_NO_SURFACE_CHECKPOINT": selected.get("C2", c2_status),
        "SECONDARY_C2_STATUS": c2_status,
        "DENSE_QUERY_ABSREL": number(groups["C0|direct"]["query_metrics"]["depth_absrel"]["mean"]),
        "DEPTH_QUERY_ABSREL": number(groups["C1|direct"]["query_metrics"]["depth_absrel"]["mean"]),
        "DEPTH_NO_SURFACE_QUERY_ABSREL": number(
            groups["C2|direct"]["query_metrics"]["depth_absrel"]["mean"]
        )
        if secondary
        else "NOT_RUN_SECONDARY_OPTIONAL",
        "DEPTH_GAIN": number(depth["mean"]),
        "DEPTH_GAIN_CI": depth["ci95"],
        "SURFACE_WITH_DEPTH_GAIN": number(surface_with_depth["mean"])
        if surface_with_depth
        else "NOT_RUN_SECONDARY_OPTIONAL",
        "SURFACE_WITH_DEPTH_GAIN_CI": surface_with_depth["ci95"]
        if surface_with_depth
        else "NOT_RUN_SECONDARY_OPTIONAL",
        "DENSE_FULL_CONTEXT_GAIN": number(results["full_context_gain"]["C0"]["mean"]),
        "DEPTH_FULL_CONTEXT_GAIN": number(results["full_context_gain"]["C1"]["mean"]),
        "DEPTH_FULL_CONTEXT_GAIN_CI": results["full_context_gain"]["C1"]["ci95"],
        "DENSE_WRONG_SCENE_DAMAGE": number(controls["C0|wrong_scene"]["mean"]),
        "DEPTH_WRONG_SCENE_DAMAGE": number(wrong["mean"]),
        "DEPTH_WRONG_SCENE_DAMAGE_CI": wrong["ci95"],
        "DENSE_SHUFFLE_DAMAGE": number(controls["C0|spatial_shuffle"]["mean"]),
        "DEPTH_SHUFFLE_DAMAGE": number(shuffle["mean"]),
        "DEPTH_SHUFFLE_DAMAGE_CI": shuffle["ci95"],
        "TRAIN_DEPTH_ABSREL_LAST100": {v: number(fit.get(v)) for v in variants},
        "LEARNED_SIGMA_SCALE_BY_SEED": sigmas,
        "WITHIN_VS_ACROSS_SCENE_STATE_CORR": {
            v: [
                number(corr.get(v, {}).get("within_scene_A_B")),
                number(corr.get(v, {}).get("across_scene_A")),
            ]
            for v in variants
        },
        "OBS2PLUS_DENSE": number(obs["groups"]["C0"]["OBS2PLUS"]["conditional_absrel"]["mean"]),
        "OBS2PLUS_DEPTH": number(obs["groups"]["C1"]["OBS2PLUS"]["conditional_absrel"]["mean"]),
        "OBS2PLUS_DEPTH_GAIN": number(obs["gains"]["OBS2PLUS"]["mean"]),
        "CARRIER_REFERENCE_STATUS": reference_status,
        "C1_ABOVE_GEOMETRY_FREE_REFERENCE": above_reference,
        "REFERENCE_LABELS": reference_labels,
        "DEPTH_STATUS": depth_status,
        "SCENE_SPECIFICITY_STATUS": specificity,
        "STATIC_DEV_STATUS": qualification["STATIC_DEV_STATUS"],
        "SEED_ROBUSTNESS": qualification["SEED_ROBUSTNESS"],
        "INTERPRETATION_BRANCH": branch,
        "FRESH_QUALIFICATION_INCLUDED": False,
        "FINAL_STATIC_STATUS": "NOT_ESTABLISHED",
        "DYNAMIC_TTT_NEXT_STAGE_ALLOWED": False,
        "TEST_TIME_INPUT": "RGB+DEPTH+CAMERA",
        "TEST_TIME_DEPTH_USED": True,
        "FINAL_HOLDOUT_TOUCHED": False,
        "DYNAMIC_TTT_RUN": False,
        "INFRASTRUCTURE_RETRIES": len(incidents),
        "TESTS": tests["status"],
        "FINAL_INTEGRITY": integrity["status"],
        "REPORT_DIR": str(docs),
    }
    lines = [
        "# EXP-3D-RGBD-EVIDENCE-CARRIER-V5",
        "",
        "单因素机制诊断（RGB-D轨道）：V3/V4的稠密16³ learned carrier，C1只多一条零初始化的测得深度旁路（每个上下文视角的测距给出体素的表面与自由空间似然）。"
        + "DEPTH_GAIN=AbsRel(C0 DENSE_SURFACE)−AbsRel(C1 DEPTH_SURFACE)="
        + stat(depth)
        + f"；DEPTH_STATUS={depth_status}；SCENE_SPECIFICITY_STATUS={specificity}；STATIC_DEV_STATUS={qualification['STATIC_DEV_STATUS']}；CARRIER_REFERENCE_STATUS={reference_status}；解读分支={branch}。",
        "",
        "Test-time input = RGB + 测得的上下文深度 + calibrated cameras（C0只读RGB与相机）。上下文深度只经由只含上下文帧的RGBD loader进入状态构建；query深度只用于TRAIN loss和全部状态seal之后的DEV评价。所有场景都是历史曝光场景，8个DEV场景已在V2、V3、V4中被评价：本轮是DEV机制诊断，不是资格轮，没有fresh队列。",
        "",
        "共享参数（5125个）、同seed初始值、4096个稠密证据候选、lifting、融合、refinement、heads、renderer、bounds、数据、预算与checkpoint规则全部相同；深度旁路共25个参数（3→8投影与log_sigma），加在≥2视角支持门之后，零初始化保证训练开始时C1与C0完全相同。C2只去掉表面监督，是次要变体，不进入任何gate。统计复用V2冻结的估计器，JSON中的SURFACE_*字段在本轮表示主对比C0−C1（测得深度）。",
        "",
        "| Variant | Query AbsRel (95% CI) | Context AbsRel | Query−context gap | 训练末100步深度AbsRel | Opacity |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for v in variants:
        g = groups[v + "|direct"]
        lines.append(
            f"| {v} {LABELS[v]} | {stat(g['query_metrics']['depth_absrel'])} | {number(g['context_metrics']['depth_absrel']['mean'])} | {number(g['generalization_gap']['mean'])} | {number(fit.get(v))} | {number(g['query_metrics']['opacity']['mean'])} |"
        )
    if reference is not None:
        values = reference["reference_values"]
        reference_text = (
            f"TRAIN拟合的无几何常数：AbsRel最优常数={values['REF_TRAIN_ABSREL_OPTIMAL']:.4f} m，中位数常数={values['REF_TRAIN_MEDIAN']:.4f} m。"
            + "；".join(
                f"{v}: RAW AbsRel相对参考="
                + stat(
                    reference["carriers"][f"V5_{v}"]["RAW"]["depth_absrel"][
                        f"vs_{reference['primary_reference']['depth_absrel']}"
                    ]
                )
                + f"（{reference_labels[v]['RAW:depth_absrel']}），RAW delta1相对参考="
                + stat(
                    reference["carriers"][f"V5_{v}"]["RAW"]["depth_delta1"][
                        f"vs_{reference['primary_reference']['depth_delta1']}"
                    ]
                )
                + f"（{reference_labels[v]['RAW:depth_delta1']}）"
                for v in variants
                if f"V5_{v}" in reference["carriers"]
            )
            + f"。CARRIER_REFERENCE_STATUS={reference_status}。正数表示carrier更好；只作描述，不参与gate。"
        )
    else:
        reference_text = "无几何参考未记录（CARRIER_REFERENCE_STATUS=NOT_RECORDED）。"
    answers = [
        (
            "测得的上下文深度是否改善DEV query几何？",
            f"DEPTH_GAIN={stat(depth)}；DEPTH_STATUS={depth_status}。gate沿用V2主对比：均值、CI、75%场景不差、所有LOSO>0、正向top1≤0.5、C1 wrong-scene与shuffle损伤CI>0、opacity与状态审计。",
        ),
        (
            "是否跨多个场景和seed？",
            f"improved/tied/worse={depth['positive']}/{depth['tied']}/{depth['negative']}；LOSO范围={depth['loso_range']}；positive top1/top3={depth['positive_top1_contribution']}/{depth['positive_top3_contribution']}；seeds={results['seeds']}；SEED_ROBUSTNESS={qualification['SEED_ROBUSTNESS']}。每个seed的结果见static_results.json。",
        ),
        (
            "状态是否变得场景专属？",
            f"C1 wrong-scene损伤={stat(wrong)}；C1 spatial shuffle损伤={stat(shuffle)}；SCENE_SPECIFICITY_STATUS={specificity}。对照C0：wrong-scene={stat(controls['C0|wrong_scene'])}，shuffle={stat(controls['C0|spatial_shuffle'])}。wrong-scene同时替换donor bounds，shuffle保留recipient bounds与相机。",
        ),
        (
            "full context是否稳定优于anchor？",
            "C0="
            + stat(results["full_context_gain"]["C0"])
            + "；C1="
            + stat(results["full_context_gain"]["C1"])
            + "。anchor只有一个视角，达不到≥2视角lifting支持；C1/C2的anchor状态仍含单视角深度旁路，所以这是3个RGB-D视角对1个RGB-D视角的融合检验。",
        ),
        (
            "训练集拟合与泛化差距如何？",
            "最后100步训练深度AbsRel（A/B平均、seed平均）="
            + json.dumps({v: number(fit.get(v)) for v in variants}, ensure_ascii=False)
            + "；query−context gap："
            + "；".join(
                f"{v}={stat(groups[v + '|direct']['generalization_gap'])}" for v in variants
            )
            + "。只是描述，不是gate。",
        ),
        (
            "模型学到了怎样的深度似然宽度？",
            "exp(log_sigma)（按seed，单位为平均体素边长）="
            + json.dumps(sigmas, ensure_ascii=False)
            + "；初值0.5。越小表示越依赖精确的表面位置。",
        ),
        (
            "有了测得深度后，表面监督是否还有作用？",
            (
                "SURFACE_WITH_DEPTH_GAIN=AbsRel(C2)−AbsRel(C1)="
                + stat(surface_with_depth)
                + f"（正值表示表面监督有益；improved/tied/worse={surface_with_depth['positive']}/{surface_with_depth['tied']}/{surface_with_depth['negative']}）。仅描述，不参与gate，不改变PRIMARY_METHOD=C1。"
            )
            if surface_with_depth
            else f"C2未纳入评价（SECONDARY_C2_STATUS={c2_status}）。",
        ),
        (
            "改善是否出现在多视角可观测区域？",
            "C0−C1条件AbsRel："
            + "；".join(f"{r}={stat(obs['gains'][r])}" for r in ("OBS0", "OBS1", "OBS2PLUS"))
            + "。测得深度应当主要改善OBS1/OBS2PLUS；这是seal后的区域诊断，不替代全图主指标。",
        ),
        (
            "状态相关性诊断说明了什么？",
            "同场景A/B与跨场景A状态的密度相关（seed平均）="
            + json.dumps(
                {
                    v: {
                        "within": number(corr.get(v, {}).get("within_scene_A_B")),
                        "across": number(corr.get(v, {}).get("across_scene_A")),
                        "direct_vs_anchor": number(corr.get(v, {}).get("direct_A_vs_anchor_A")),
                    }
                    for v in variants
                },
                ensure_ascii=False,
            )
            + "。仅描述，不参与gate。",
        ),
        (
            "DEV是否达到static qualification？",
            qualification["STATIC_DEV_STATUS"]
            + "；检查="
            + json.dumps(qualification["static_checks"], ensure_ascii=False)
            + "；主对比检查="
            + json.dumps(qualification["surface_checks"], ensure_ascii=False)
            + "。即使通过，本轮DEV已曝光，也不构成最终资格。",
        ),
        (
            "与TRAIN拟合的无几何常数相比如何？",
            reference_text,
        ),
        (
            "结论属于哪个预注册解读分支，下一步是什么？现在能否重开Dynamic TTT？",
            f"分支{branch}：{next_step} DYNAMIC_TTT_NEXT_STAGE_ALLOWED=false；DYNAMIC_TTT_RUN=false；FINAL_STATIC_STATUS=NOT_ESTABLISHED。",
        ),
    ]
    for i, (question, answer) in enumerate(answers, 1):
        lines.extend(["", f"**{i}. {question}**", "", answer])
    lines.extend(
        [
            "",
            "**opacity与状态审计**",
            "",
            "C0/C1 opacity checks="
            + json.dumps(state["opacity_checks"], ensure_ascii=False)
            + "；C1 static opacity checks="
            + json.dumps(state["static_opacity_checks"], ensure_ascii=False)
            + "；NO_HIT scenes="
            + str(state["no_hit_scenes"])
            + "；shared-state/seal审计="
            + json.dumps(state["input_audit"], ensure_ascii=False)
            + "。",
            "",
            "**选择、统计与成本**",
            "",
            "每个variant×seed用同一DEV规则独立选择checkpoint（checkpoint在前期加密）。selected="
            + json.dumps(selected, ensure_ascii=False)
            + "。先对每个query/role等权平均所有seed，再平均query/role得到scene值；10,000次paired scene bootstrap，seed 20260928，95% percentile CI。成本见cost_analysis.json；深度旁路只增加构建状态时的计算，renderer与推理参数（除旁路25个）相同。",
            "",
            f"基础设施故障重跑次数={len(incidents)}（失败尝试归档在audit/failed_attempts且从不使用，见audit/incidents.json）；SECONDARY_C2_STATUS={c2_status}；CPU亲和性固定为8–23号核。协议见PROTOCOL.md。",
            "",
            "TEST_TIME_INPUT=RGB+DEPTH+CAMERA；TEST_TIME_DEPTH_USED=true（仅上下文深度）；FINAL_HOLDOUT_TOUCHED=false；DYNAMIC_TTT_RUN=false。",
        ]
    )
    summary = (
        "RGBD EVIDENCE CARRIER V5 FINAL\n"
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
