#!/usr/bin/env python3
"""Report saved V3 sparse/dense evidence DEV evidence without opening fresh data or models."""
# ruff: noqa: E501 -- Chinese scientific report prose

import argparse
import json
from pathlib import Path

LABELS = {
    "C0": "SPARSE128_SURFACE",
    "C1": "DENSE4096_SURFACE",
    "C2": "DENSE4096_NO_SURFACE",
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
        seeds = sorted({r["seed"] for r in rows if r["variant"] == variant})
        per_seed = []
        for seed in seeds:
            tail = [
                (r["terms"]["A"]["loss_depth"] + r["terms"]["B"]["loss_depth"]) / 2
                for r in rows
                if r["variant"] == variant and r["seed"] == seed and r["step"] > steps - 100
            ]
            if tail:
                per_seed.append(sum(tail) / len(tail))
        fit[variant] = sum(per_seed) / len(per_seed) if per_seed else None
    return fit


def run(root, docs=None):
    root, docs = Path(root), Path(docs) if docs else Path(root)
    results = read(root, "static_results")
    qualification = read(root, "qualification_results")
    state = read(root, "state_use_results")
    controls = read(root, "wrong_scene_results")["contrasts"]
    obs = read(root, "observability_analysis")
    split = read(root, "scene_split")
    contract = read(root, "training_contract", {"steps": 3000})
    curves = read(root, "training_curves", {"training": [], "dev": []})
    selected = read(root, "selected_checkpoints", {"status": "NOT_RECORDED"})
    roster = read(root, "evaluated_variants", {"secondary_status": {}})
    correlation = read(root, "state_correlation", {"per_variant_mean_over_seeds": {}})
    incidents = read(root, "audit/incidents", [])
    tests, integrity = (read(root, n, {"status": "PENDING"}) for n in ("tests", "integrity"))
    if results["primary"] != "C1_SURFACE16":
        raise PermissionError("Frozen V2 estimator primary label expected (C1 = V3 primary)")
    if (
        qualification["FRESH_QUALIFICATION_OPENED"]
        or qualification["DYNAMIC_TTT_NEXT_STAGE_ALLOWED"]
    ):
        raise PermissionError("V3 is a DEV mechanism round; fresh and Dynamic TTT stay closed")
    groups = results["groups"]
    density = results["surface_gain"]
    secondary = "C2|direct" in groups
    surface_at_dense = negated(results.get("free_space_extra_gain"))
    c2_status = roster["secondary_status"].get("C2", "NOT_RUN_SECONDARY_OPTIONAL")
    wrong, shuffle = controls["C1|wrong_scene"], controls["C1|spatial_shuffle"]
    specificity = (
        "SUPPORTED" if wrong["ci95"][0] > 0 and shuffle["ci95"][0] > 0 else "NOT_ESTABLISHED"
    )
    density_status = qualification["SURFACE_TRAINING_STATUS"]
    if density_status == "SUPPORTED" and specificity == "SUPPORTED":
        branch = "A"
    elif density["ci95"][0] > 0 or specificity == "SUPPORTED":
        branch = "B"
    else:
        branch = "C"
    next_step = {
        "A": "稀疏证据通路是V2失败的主要原因之一。下一轮在稠密carrier上重做表面监督的matched对照，并建立经过独立性审计的fresh队列；仍不开Dynamic TTT。",
        "B": "证据密度有部分作用。下一轮先定位剩余瓶颈（融合、可见性、训练规模），再决定是否引入新模块；仍不开Dynamic TTT。",
        "C": "排除“候选稀疏”作为主因。下一轮按V2 CASE A引入可见性感知融合或cost-volume几何推理，另写协议；仍不开Dynamic TTT。",
    }[branch]
    fit = train_fit(curves, contract["steps"])
    corr = correlation["per_variant_mean_over_seeds"]
    variants = ("C0", "C1", "C2") if secondary else ("C0", "C1")
    fields = {
        "N_TRAIN_SCENES": sum(r.get("split") == "TRAIN" for r in split["scenes"]),
        "N_DEV_SCENES": len(results["scene_ids"]),
        "DEV_REUSED_FROM_V2": True,
        "SPARSE128_SURFACE_CHECKPOINT": selected.get("C0", "NOT_RECORDED"),
        "DENSE4096_SURFACE_CHECKPOINT": selected.get("C1", "NOT_RECORDED"),
        "DENSE4096_NO_SURFACE_CHECKPOINT": selected.get("C2", c2_status),
        "SECONDARY_C2_STATUS": c2_status,
        "SPARSE128_QUERY_ABSREL": number(
            groups["C0|direct"]["query_metrics"]["depth_absrel"]["mean"]
        ),
        "DENSE4096_QUERY_ABSREL": number(
            groups["C1|direct"]["query_metrics"]["depth_absrel"]["mean"]
        ),
        "DENSE4096_NO_SURFACE_QUERY_ABSREL": number(
            groups["C2|direct"]["query_metrics"]["depth_absrel"]["mean"]
        )
        if secondary
        else "NOT_RUN_SECONDARY_OPTIONAL",
        "DENSITY_GAIN": number(density["mean"]),
        "DENSITY_GAIN_CI": density["ci95"],
        "SURFACE_AT_DENSE_GAIN": number(surface_at_dense["mean"])
        if surface_at_dense
        else "NOT_RUN_SECONDARY_OPTIONAL",
        "SURFACE_AT_DENSE_GAIN_CI": surface_at_dense["ci95"]
        if surface_at_dense
        else "NOT_RUN_SECONDARY_OPTIONAL",
        "SPARSE128_FULL_CONTEXT_GAIN": number(results["full_context_gain"]["C0"]["mean"]),
        "DENSE4096_FULL_CONTEXT_GAIN": number(results["full_context_gain"]["C1"]["mean"]),
        "DENSE4096_FULL_CONTEXT_GAIN_CI": results["full_context_gain"]["C1"]["ci95"],
        "SPARSE128_WRONG_SCENE_DAMAGE": number(controls["C0|wrong_scene"]["mean"]),
        "DENSE4096_WRONG_SCENE_DAMAGE": number(wrong["mean"]),
        "DENSE4096_WRONG_SCENE_DAMAGE_CI": wrong["ci95"],
        "SPARSE128_SHUFFLE_DAMAGE": number(controls["C0|spatial_shuffle"]["mean"]),
        "DENSE4096_SHUFFLE_DAMAGE": number(shuffle["mean"]),
        "DENSE4096_SHUFFLE_DAMAGE_CI": shuffle["ci95"],
        "TRAIN_DEPTH_ABSREL_LAST100": {v: number(fit.get(v)) for v in variants},
        "WITHIN_VS_ACROSS_SCENE_STATE_CORR": {
            v: [
                number(corr.get(v, {}).get("within_scene_A_B")),
                number(corr.get(v, {}).get("across_scene_A")),
            ]
            for v in variants
        },
        "OBS2PLUS_SPARSE128": number(obs["groups"]["C0"]["OBS2PLUS"]["conditional_absrel"]["mean"]),
        "OBS2PLUS_DENSE4096": number(obs["groups"]["C1"]["OBS2PLUS"]["conditional_absrel"]["mean"]),
        "OBS2PLUS_DENSITY_GAIN": number(obs["gains"]["OBS2PLUS"]["mean"]),
        "DENSITY_STATUS": density_status,
        "SCENE_SPECIFICITY_STATUS": specificity,
        "STATIC_DEV_STATUS": qualification["STATIC_DEV_STATUS"],
        "SEED_ROBUSTNESS": qualification["SEED_ROBUSTNESS"],
        "INTERPRETATION_BRANCH": branch,
        "FRESH_QUALIFICATION_INCLUDED": False,
        "FINAL_STATIC_STATUS": "NOT_ESTABLISHED",
        "DYNAMIC_TTT_NEXT_STAGE_ALLOWED": False,
        "TEST_TIME_INPUT": "RGB+CAMERA",
        "TEST_TIME_DEPTH_USED": False,
        "FINAL_HOLDOUT_TOUCHED": False,
        "DYNAMIC_TTT_RUN": False,
        "INFRASTRUCTURE_RETRIES": len(incidents),
        "TESTS": tests["status"],
        "FINAL_INTEGRITY": integrity["status"],
        "REPORT_DIR": str(docs),
    }
    lines = [
        "# EXP-3D-DENSE-EVIDENCE-CARRIER-V3",
        "",
        "单因素机制诊断：同一个16³ learned carrier，唯一差别是多视角证据候选点从128个（V2配方，只覆盖3.1%体素）加密到全部4096个体素。"
        + "DENSITY_GAIN=AbsRel(C0 SPARSE128_SURFACE)−AbsRel(C1 DENSE4096_SURFACE)="
        + stat(density)
        + f"；DENSITY_STATUS={density_status}；SCENE_SPECIFICITY_STATUS={specificity}；STATIC_DEV_STATUS={qualification['STATIC_DEV_STATUS']}；解读分支={branch}。",
        "",
        "Test-time input = RGB + calibrated cameras。训练是RGB+D utility track；depth只用于TRAIN loss和seal后的DEV评价。所有场景都是历史曝光场景，8个DEV场景已在V2中被评价：本轮是DEV上的机制诊断，不是资格轮，没有fresh队列。",
        "",
        "参数量（5125）、同seed初始参数、lifting规则（≥2视角支持）、融合、refinement、heads、renderer、bounds、数据、预算与checkpoint选择规则全部相同。C2只去掉表面监督，是次要变体，不进入任何gate。统计复用V2冻结的估计器，JSON中的SURFACE_*字段在本轮表示主对比C0−C1（密度）。",
        "",
        "| Variant | Query AbsRel (95% CI) | Context AbsRel | Query−context gap | 训练末100步深度AbsRel | Opacity |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for v in variants:
        g = groups[v + "|direct"]
        lines.append(
            f"| {v} {LABELS[v]} | {stat(g['query_metrics']['depth_absrel'])} | {number(g['context_metrics']['depth_absrel']['mean'])} | {number(g['generalization_gap']['mean'])} | {number(fit.get(v))} | {number(g['query_metrics']['opacity']['mean'])} |"
        )
    answers = [
        (
            "稠密证据是否改善DEV query几何？",
            f"DENSITY_GAIN={stat(density)}；DENSITY_STATUS={density_status}。gate沿用V2主对比：均值、CI、75%场景不差、所有LOSO>0、正向top1≤0.5、C1 wrong-scene与shuffle损伤CI>0、opacity与状态审计。",
        ),
        (
            "是否跨多个场景和seed？",
            f"improved/tied/worse={density['positive']}/{density['tied']}/{density['negative']}；LOSO范围={density['loso_range']}；positive top1/top3={density['positive_top1_contribution']}/{density['positive_top3_contribution']}；seeds={results['seeds']}；SEED_ROBUSTNESS={qualification['SEED_ROBUSTNESS']}。每个seed的结果见static_results.json。",
        ),
        (
            "稠密carrier的状态是否变得场景专属？",
            f"C1 wrong-scene损伤={stat(wrong)}；C1 spatial shuffle损伤={stat(shuffle)}；SCENE_SPECIFICITY_STATUS={specificity}。对照C0：wrong-scene={stat(controls['C0|wrong_scene'])}，shuffle={stat(controls['C0|spatial_shuffle'])}。wrong-scene同时替换donor bounds，shuffle保留recipient bounds与相机。",
        ),
        (
            "full context是否稳定优于anchor？",
            "C0="
            + stat(results["full_context_gain"]["C0"])
            + "；C1="
            + stat(results["full_context_gain"]["C1"])
            + "。anchor只有一个视角，达不到≥2视角支持，因此anchor状态只含学到的先验。",
        ),
        (
            "训练集拟合是否改善？",
            "最后100步训练深度AbsRel（A/B平均、seed平均）="
            + json.dumps({v: number(fit.get(v)) for v in variants}, ensure_ascii=False)
            + "。只是描述，不是gate。",
        ),
        (
            "context-query gap是否缩小？",
            "；".join(f"{v}={stat(groups[v + '|direct']['generalization_gap'])}" for v in variants)
            + "。未预注册gap差值的显著性，不从CI重叠推断。",
        ),
        (
            "稠密证据下表面监督是否有收益？",
            (
                "SURFACE_AT_DENSE_GAIN=AbsRel(C2)−AbsRel(C1)="
                + stat(surface_at_dense)
                + f"（正值表示表面监督有益；improved/tied/worse={surface_at_dense['positive']}/{surface_at_dense['tied']}/{surface_at_dense['negative']}）。仅描述，不参与gate，不改变PRIMARY_METHOD=C1。"
            )
            if surface_at_dense
            else f"C2未纳入评价（SECONDARY_C2_STATUS={c2_status}）。",
        ),
        (
            "改善发生在哪些可观测区域？",
            "C0−C1条件AbsRel："
            + "；".join(f"{r}={stat(obs['gains'][r])}" for r in ("OBS0", "OBS1", "OBS2PLUS"))
            + "。这是seal后的区域诊断，不替代全图主指标。",
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
            "每个variant×seed用同一DEV规则独立选择checkpoint。selected="
            + json.dumps(selected, ensure_ascii=False)
            + "。先对每个query/role等权平均所有seed，再平均query/role得到scene值；10,000次paired scene bootstrap，seed 20260928，95% percentile CI。成本见cost_analysis.json；稠密与稀疏的推理参数相同，只是候选点更多。",
            "",
            f"基础设施故障重跑次数={len(incidents)}（失败尝试归档在audit/failed_attempts且从不使用，见audit/incidents.json）；SECONDARY_C2_STATUS={c2_status}；CPU亲和性固定为8–23号核。协议见PROTOCOL.md。",
            "",
            "TEST_TIME_INPUT=RGB+CAMERA；TEST_TIME_DEPTH_USED=false；FINAL_HOLDOUT_TOUCHED=false；DYNAMIC_TTT_RUN=false。",
        ]
    )
    summary = (
        "DENSE EVIDENCE CARRIER V3 FINAL\n"
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
