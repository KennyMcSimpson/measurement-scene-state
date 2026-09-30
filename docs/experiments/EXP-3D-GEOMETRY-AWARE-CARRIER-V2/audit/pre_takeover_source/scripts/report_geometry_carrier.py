#!/usr/bin/env python3
"""Report saved matched carrier DEV evidence without opening fresh data or models."""
# ruff: noqa: E501 -- Chinese scientific report prose

import argparse
import json
from pathlib import Path


def read(root, name, default=None):
    path = root / (name + ".json")
    if not path.exists() and default is not None:
        return default
    return json.loads(path.read_text())


def number(value):
    return "UNDEFINED" if value is None else f"{value:.6f}"


def stat(value):
    if value["mean"] is None:
        return "UNDEFINED（空区域未填零）"
    ci = value["ci95"]
    return f"{value['mean']:.6f}，95% CI [{ci[0]:.6f}, {ci[1]:.6f}]"


def run(root, docs=None):
    root, docs = Path(root), Path(docs) if docs else Path(root)
    results = read(root, "static_results")
    qualification = read(root, "qualification_results")
    state = read(root, "state_use_results")
    controls = read(root, "wrong_scene_results")["contrasts"]
    obs = read(root, "observability_analysis")
    split = read(root, "scene_split")
    cost = read(root, "cost_analysis")
    selected = read(root, "selected_checkpoints", {"status": "NOT_RECORDED"})
    tests, integrity = (read(root, n, {"status": "PENDING"}) for n in ("tests", "integrity"))
    if results["primary"] != "C1_SURFACE16":
        raise PermissionError("SURFACE16 is the predeclared primary")
    if (
        qualification["FRESH_QUALIFICATION_OPENED"]
        or qualification["DYNAMIC_TTT_NEXT_STAGE_ALLOWED"]
    ):
        raise PermissionError("This report currently supports blocked-fresh DEV-only execution")
    groups = results["groups"]
    gain = results["surface_gain"]
    regions = obs["groups"]
    fields = {
        "N_TRAIN_SCENES": sum(r.get("split") == "TRAIN" for r in split["scenes"]),
        "N_DEV_SCENES": len(results["scene_ids"]),
        "N_FRESH_QUALIFICATION_SCENES": sum(
            r.get("split") == "FRESH_QUALIFICATION" for r in split["scenes"]
        ),
        "BASELINE16_CHECKPOINT": selected.get("C0", "NOT_RECORDED"),
        "SURFACE16_CHECKPOINT": selected.get("C1", "NOT_RECORDED"),
        "FREE_SURFACE16_CHECKPOINT": "NOT_RUN_SECONDARY_OPTIONAL",
        "BASELINE16_QUERY_ABSREL": number(
            groups["C0|direct"]["query_metrics"]["depth_absrel"]["mean"]
        ),
        "SURFACE16_QUERY_ABSREL": number(
            groups["C1|direct"]["query_metrics"]["depth_absrel"]["mean"]
        ),
        "SURFACE_GAIN": number(gain["mean"]),
        "SURFACE_GAIN_CI": gain["ci95"],
        "FREE_SURFACE_QUERY_ABSREL": "NOT_RUN_SECONDARY_OPTIONAL",
        "FREE_SPACE_EXTRA_GAIN": "NOT_ESTABLISHED",
        "BASELINE16_FULL_CONTEXT_GAIN": number(results["full_context_gain"]["C0"]["mean"]),
        "SURFACE16_FULL_CONTEXT_GAIN": number(results["full_context_gain"]["C1"]["mean"]),
        "BASELINE16_WRONG_SCENE_DAMAGE": number(controls["C0|wrong_scene"]["mean"]),
        "SURFACE16_WRONG_SCENE_DAMAGE": number(controls["C1|wrong_scene"]["mean"]),
        "BASELINE16_SHUFFLE_DAMAGE": number(controls["C0|spatial_shuffle"]["mean"]),
        "SURFACE16_SHUFFLE_DAMAGE": number(controls["C1|spatial_shuffle"]["mean"]),
        "OBS2PLUS_BASELINE16": number(regions["C0"]["OBS2PLUS"]["conditional_absrel"]["mean"]),
        "OBS2PLUS_SURFACE16": number(regions["C1"]["OBS2PLUS"]["conditional_absrel"]["mean"]),
        "OBS2PLUS_SURFACE_GAIN": number(obs["gains"]["OBS2PLUS"]["mean"]),
        "SURFACE_TRAINING_STATUS": qualification["SURFACE_TRAINING_STATUS"],
        "STATIC_DEV_STATUS": qualification["STATIC_DEV_STATUS"],
        "SEED_ROBUSTNESS": qualification["SEED_ROBUSTNESS"],
        "FRESH_QUALIFICATION_INDEPENDENCE": qualification["FRESH_QUALIFICATION_INDEPENDENCE"],
        "FRESH_QUALIFICATION_OPENED": False,
        "FRESH_BASELINE16_ABSREL": "NOT_RUN",
        "FRESH_SURFACE16_ABSREL": "NOT_RUN",
        "FRESH_SURFACE_GAIN": "NOT_RUN",
        "FRESH_SURFACE_GAIN_CI": "NOT_RUN",
        "FINAL_STATIC_STATUS": "NOT_ESTABLISHED",
        "DYNAMIC_TTT_NEXT_STAGE_ALLOWED": False,
        "TEST_TIME_INPUT": "RGB+CAMERA",
        "TEST_TIME_DEPTH_USED": False,
        "FINAL_HOLDOUT_TOUCHED": False,
        "DYNAMIC_TTT_RUN": False,
        "TESTS": tests["status"],
        "FINAL_INTEGRITY": integrity["status"],
        "REPORT_DIR": str(docs),
    }
    next_step = (
        "DEV对surface及static的证据均支持，但fresh独立性尚未解决。先完成新的可信fresh cohort和冻结资格协议，不启动Dynamic TTT。"
        if qualification["SURFACE_TRAINING_STATUS"]
        == qualification["STATIC_DEV_STATUS"]
        == "SUPPORTED"
        else "未满足进入下一阶段的全部条件。保留matched对照：surface无收益时研究carrier lifting/fusion/geometry reasoning；full context不胜anchor时改善shared-state构造；wrong-scene不显著时检查global-prior依赖。本轮不引入这些新架构，也不开Dynamic TTT。"
    )
    lines = [
        "# EXP-3D-GEOMETRY-AWARE-CARRIER-V2",
        "",
        "主比较是matched16³ BASELINE16(C0) vs SURFACE16(C1)，唯一方法差别是TRAIN-side显式surface loss。"
        + "SURFACE_GAIN="
        + stat(gain)
        + f"；SURFACE_TRAINING_STATUS={qualification['SURFACE_TRAINING_STATUS']}；STATIC_DEV_STATUS={qualification['STATIC_DEV_STATUS']}。",
        "",
        "Test-time input = RGB + calibrated cameras. 训练是RGB+D utility track，不是RGB-only training；depth仅供TRAIN loss及seal后的DEV评价/诊断，未作为test-time carrier输入。",
        "",
        "C1固定为PRIMARY_METHOD=SURFACE16；C2=NOT_RUN_SECONDARY_OPTIONAL。未新算旧8³ B-final，只保留历史上下文；不能把C0改进单独归因为resolution、重训或训练数据增多。C0/C1推理architecture完全一致，surface loss只存在训练阶段，不能宣称它使推理架构更贵。",
        "",
        "| Method | Query AbsRel (95% CI) | Context AbsRel | Query−context gap | Opacity | Coverage |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for v in ("C0", "C1"):
        g = groups[v + "|direct"]
        lines.append(
            f"| {v} | {stat(g['query_metrics']['depth_absrel'])} | {number(g['context_metrics']['depth_absrel']['mean'])} | {number(g['generalization_gap']['mean'])} | {number(g['query_metrics']['opacity']['mean'])} | {number(g['query_metrics']['coverage']['mean'])} |"
        )
    answers = [
        (
            "Surface supervision在learned carrier中是否有效？",
            f"{qualification['SURFACE_TRAINING_STATUS']}；C0−C1={stat(gain)}。本轮主gate是>0而非沿用旧.02门槛，同时检查CI、75%场景不差、所有LOSO>0、positive top1 share≤.5、controls与opacity。",
        ),
        (
            "是否跨多个scenes和seeds？",
            f"improved/tied/worse={gain['positive']}/{gain['tied']}/{gain['negative']}；LOSO范围={gain['loso_range']}；positive top1/top3={gain['positive_top1_contribution']}/{gain['positive_top3_contribution']}。seed列表={results['seeds']}；SEED_ROBUSTNESS={qualification['SEED_ROBUSTNESS']}。所有seed保留，不选best seed；每seed结果见static_results.json。top1 share≤.5参与gate，top3仅报告不作为gate。",
        ),
        (
            "是否复现OBS2PLUS改善signature？",
            "OBS2PLUS C0−C1="
            + stat(obs["gains"]["OBS2PLUS"])
            + f"；scene coverage={obs['gains']['OBS2PLUS']['scene_coverage']}。这是post-seal条件区域诊断；空scene保留null，不把conditional收益替代全图净收益。",
        ),
        (
            "16³ baseline retraining本身提升多少？",
            "本轮未新增旧8³模型matched评价，因此不能独立分离grid、更多数据、训练预算和重训收益。正式归因只比较相同architecture/grid/data/seed/budget/selection规则的C0和C1。",
        ),
        (
            "surface相对matched baseline额外提升多少？",
            stat(gain) + "。这才是主方法增量，方向为C0 AbsRel−C1 AbsRel，正值更好。",
        ),
        (
            "full context是否稳定优于anchor？",
            "C0="
            + stat(results["full_context_gain"]["C0"])
            + "；C1="
            + stat(results["full_context_gain"]["C1"])
            + f"；STATIC_DEV_STATUS={qualification['STATIC_DEV_STATUS']}。anchor_A只编码共同anchor RGB，沿用A的合法context-camera bounds；anchor_B同理，不是只允许单相机几何信息的baseline。",
        ),
        (
            "wrong-scene是否稳定更差？",
            "C0="
            + stat(controls["C0|wrong_scene"])
            + "；C1="
            + stat(controls["C1|wrong_scene"])
            + "。recipient query camera不变；wrong-scene同时替换donor bounds，不能将全部damage独归density内容。",
        ),
        (
            "state是否真正scene-specific并有空间结构？",
            "C0 shuffle="
            + stat(controls["C0|spatial_shuffle"])
            + "；C1 shuffle="
            + stat(controls["C1|spatial_shuffle"])
            + "。shuffle保留recipient bounds。shared-state/seal审计="
            + json.dumps(state["input_audit"], ensure_ascii=False)
            + "。多个query必须使用同一sealed state，controls CI不支持时不能声称已证明。",
        ),
        (
            "context-query gap是否缩小？",
            "C0="
            + stat(groups["C0|direct"]["generalization_gap"])
            + "；C1="
            + stat(groups["C1|direct"]["generalization_gap"])
            + "。这里分别报告两者，未预注册额外gap-change显著性则不从两个CI重叠与否推出差值显著性；query净收益是主目标。",
        ),
        (
            "free-space有没有额外价值？",
            "C2未运行，FREE_SPACE_EXTRA_GAIN=NOT_ESTABLISHED。不可把上一轮direct-state结果或本轮C1结果当作C2训练证据。",
        ),
        (
            "DEV是否达到static qualification？",
            qualification["STATIC_DEV_STATUS"]
            + "；检查="
            + json.dumps(qualification["static_checks"], ensure_ascii=False)
            + "。surface检查="
            + json.dumps(qualification["surface_checks"], ensure_ascii=False)
            + "。DEV资格不是fresh最终资格。",
        ),
        (
            "fresh qualification是否允许打开？",
            "FRESH_QUALIFICATION_OPENED=false；独立性="
            + qualification["FRESH_QUALIFICATION_INDEPENDENCE"]
            + "。即使DEV良好也不替换fresh cohort或将已曝光scene重新包装。",
        ),
        (
            "fresh scenes是否复现主要signature？",
            "NOT_RUN；没有新fresh结果，FINAL_STATIC_STATUS=NOT_ESTABLISHED。",
        ),
        (
            "现在是否可以重新打开Dynamic TTT？",
            "DYNAMIC_TTT_NEXT_STAGE_ALLOWED=false；DYNAMIC_TTT_RUN=false。" + next_step,
        ),
    ]
    for i, (question, answer) in enumerate(answers, 1):
        lines.extend(["", f"**{i}. {question}**", "", answer])
    lines.extend(
        [
            "",
            "**预锁opacity与固定bounds限制**",
            "",
            "在V2正式训练前，主agent查阅已曝光历史ai_009_001发现固定bounds query rays全miss，因此未改scene/roles/bounds，改用每scene coverage≥.99×ray_hitfraction、C1−C0coverage≥−.01。NO_HIT不是新model collapse。所有8个DEV场景及全部GT-valid pixel仍计入主指标，固定bounds盲区保留。",
            "",
            "共mask要求C0和C1 opacity>1e-6；空scene显式null，共mask统计覆盖至少75%scene。还要求rendered_depth/opacity.clamp_min(1e-6)这一纯诊断上的C0−C1收益CI下界>0。主rendered depth语义和主AbsRel不改变。诊断降低透明度缩放解释的风险，不构成排除所有opacity相关机制的证明。",
            "",
            "opacity checks="
            + json.dumps(state["opacity_checks"], ensure_ascii=False)
            + "；NO_HIT scenes="
            + str(state["no_hit_scenes"])
            + "。",
            "",
            "**选择、统计与成本**",
            "",
            "每variant×seed用同一预锁DEV选择规则独立选择checkpoint；不是强行同步步数。selected step/path="
            + json.dumps(selected, ensure_ascii=False)
            + "。lambda_surface=.1与tau=.5×physical voxel diagonal预先冻结，不能由DEV/fresh重新调节。surface-band mass会通过transmittance向前方density传播梯度。",
            "",
            "先对每个query/role等权平均所有seed，再平均query/role得到scene值；10,000 paired scene bootstrap，seed20260928，95% percentile CI。seeds、A/B和pixels不能作为独立scene扩充样本量。区域conditional均值、pixel fraction、可加误差贡献与空区域scene coverage见observability_analysis.json；完整mean/median/LOSO/top1/top3在bootstrap_results.json。",
            "",
            "training curves和8类图见training_curves.json及figures/。各variant成本原始记录="
            + json.dumps(cost, ensure_ascii=False)
            + "。training_seconds是worker累计时间，与phase_wall_times分开；inference与renderer latency分开，checkpoint bytes/VRAM亦分别记录，缺失值不伪造。",
            "",
            "本报告仅读取保存的JSON，不进行模型forward或新评价。复现见commands.sh；raw、配置、checkpoint选择锁、数据provenance、state/GT访问审计与完整性清单保存在本实验目录。",
            "",
            "TEST_TIME_INPUT=RGB+CAMERA；TEST_TIME_DEPTH_USED=false；FINAL_HOLDOUT_TOUCHED=false；DYNAMIC_TTT_RUN=false。",
        ]
    )
    summary = (
        "16G GEOMETRY-AWARE CARRIER V2 FINAL\n"
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
