#!/usr/bin/env python3
"""Chinese scientific report from saved statistics only; never load query media or states."""
# ruff: noqa: E501 -- preserve Chinese scientific report sentences for editorial review

import argparse
import json
from pathlib import Path

VARIANTS = ("S0", "S1", "S2", "S3")
REGIONS = ("OBS0", "OBS1", "OBS2PLUS")


def read(root, name):
    return json.loads((root / (name + ".json")).read_text())


def number(value):
    return "UNDEFINED" if value is None else f"{value:.6f}"


def stat(value):
    if value["mean"] is None:
        return "未定义（空区域；不填零）"
    ci = value["ci95"]
    return (
        f"{value['mean']:.6f}，95% CI [{ci[0]:.6f}, {ci[1]:.6f}]"
        if ci is not None
        else number(value["mean"])
    )


def direction(observability, supervision):
    if supervision["GEOMETRY_AWARE_CARRIER_RECOMMENDED"]:
        return "下一轮单独研究16³ geometry-aware carrier，并保留显式 ray free-space/surface 监督与独立验证；本轮不训练。"
    if observability["OBSERVABILITY_STATUS"] == "UNOBSERVED_REGION_DOMINANT":
        return "优先研究 geometry completion / learned prior；有限可见约束不能直接确定未观察表面，仍需独立实验区分先验与真实恢复。"
    if observability["OBSERVABILITY_STATUS"] in ("OBSERVED_REGION_FAILURE", "MIXED"):
        return "优先研究 visibility-aware multi-view lifting / cost volume / MVS-style geometry reasoning；若联合监督未稳定缩小 gap，不继续堆叠 loss。"
    return "证据不足以选择单一机制。先保留已冻结定义，研究多视图几何推断、可见性与证据充分性；不直接启动 carrier 训练。"


def run(root, docs=None):
    root = Path(root)
    docs = Path(docs) if docs else root
    obs = read(root, "observability_analysis")
    sup = read(root, "supervision_analysis")
    gap = read(root, "gap_closure_analysis")
    control = read(root, "wrong_scene_analysis")
    bootstrap = read(root, "bootstrap_results")
    cost = read(root, "cost_analysis")
    decisions = read(root, "decision_summary")
    if sup["primary_variant"] != "S3" or decisions["BEST_SUPERVISION"] != "S3_PREDECLARED_PRIMARY":
        raise PermissionError("Report cannot select a hindsight supervision winner")
    if any(
        decisions[k] is not False
        for k in ("FINAL_HOLDOUT_TOUCHED", "NEW_CARRIER_TRAINED", "DYNAMIC_TTT_RUN")
    ):
        raise PermissionError("Forbidden experiment activity recorded")
    groups, references, regions = sup["groups"], sup["reference_groups"], obs["regions"]
    contrasts = sup["overall_absrel_comparisons"]
    main_gain = contrasts["S0_minus_S3"]
    main_closure = gap["variants"]["S3"]
    recommendations = direction(obs, sup)
    tests = read(root, "tests") if (root / "tests.json").exists() else {"status": "PENDING"}
    integrity = (
        read(root, "integrity") if (root / "integrity.json").exists() else {"status": "PENDING"}
    )
    baseline = (
        read(root, "baseline_reproduction")
        if (root / "baseline_reproduction.json").exists()
        else {"status": "PENDING"}
    )
    terminal = {
        "N_SCENES": len(obs["scene_ids"]),
        "BASELINE_REPRO_STATUS": baseline["status"],
        "BASELINE_QUERY_ABSREL": number(
            groups["S0|direct"]["query_metrics"]["depth_absrel"]["mean"]
        ),
        "QUERY_ORACLE_ABSREL": number(
            references["ORACLE|direct"]["query_metrics"]["depth_absrel"]["mean"]
        ),
        "BASE_CONTEXT_ORACLE_GAP": number(gap["base_gap"]["mean"]),
    }
    for region in REGIONS:
        terminal[f"{region}_FRACTION"] = number(regions["S0"][region]["pixel_fraction"]["mean"])
    for region in REGIONS:
        terminal[f"{region}_BASE_ABSREL"] = number(
            regions["S0"][region]["conditional_absrel"]["mean"]
        )
        terminal[f"{region}_ORACLE_ABSREL"] = number(
            regions["ORACLE"][region]["conditional_absrel"]["mean"]
        )
    terminal["OBSERVABILITY_STATUS"] = obs["OBSERVABILITY_STATUS"]
    for variant, label in zip(
        VARIANTS, ("RENDER_ONLY", "FREE_SPACE", "SURFACE", "FREE_SURFACE"), strict=True
    ):
        terminal[f"{variant}_{label}_ABSREL"] = number(
            groups[f"{variant}|direct"]["query_metrics"]["depth_absrel"]["mean"]
        )
    terminal.update(
        BEST_SUPERVISION="S3_PREDECLARED_PRIMARY_NOT_HINDSIGHT_WINNER",
        BEST_SUPERVISION_GAIN=number(main_gain["mean"]),
        BEST_SUPERVISION_CI=main_gain["ci95"],
        GAP_CLOSED=number(main_closure["gap_closed"]["mean"]),
        GAP_CLOSED_FRACTION=number(main_closure["gap_closed_fraction"]["value"]),
        WRONG_SCENE_DAMAGE_BEST=number(control["contrasts"]["S3|wrong_scene"]["mean"]),
        GEOMETRY_SUPERVISION_STATUS=sup["GEOMETRY_SUPERVISION_STATUS"],
        VISIBILITY_AWARE_FUSION_RECOMMENDED=sup["VISIBILITY_AWARE_FUSION_RECOMMENDED"],
        GEOMETRY_AWARE_CARRIER_RECOMMENDED=sup["GEOMETRY_AWARE_CARRIER_RECOMMENDED"],
        NEXT_CARRIER_DIRECTION=recommendations,
        FINAL_HOLDOUT_TOUCHED=False,
        NEW_CARRIER_TRAINED=False,
        DYNAMIC_TTT_RUN=False,
        TESTS=tests["status"],
        FINAL_INTEGRITY=integrity["status"],
        REPORT_DIR=str(docs),
    )
    lines = [
        "# EXP-3D-CONTEXT-OBSERVABILITY-SUPERVISION-V1",
        "",
        f"预先指定的主比较 S0−S3 query AbsRel gain={stat(main_gain)}。"
        f"OBSERVABILITY_STATUS={obs['OBSERVABILITY_STATUS']}；GEOMETRY_SUPERVISION_STATUS={sup['GEOMETRY_SUPERVISION_STATUS']}。",
        "",
        "本轮是17个 CAPACITY_DEV_EXPOSED 场景上的可观测性与几何监督归因，不是独立 validation、final test 或资格确认。BEST_SUPERVISION 固定指 S3（PREDECLARED_PRIMARY），没有根据 query 分数更换主方法。",
        "",
        "保持16³、renderer64、旧 frozen GT-free bounds、10000 steps、Adam/LR/clip/RGB loss/rendered-depth loss/regularization、context/query IDs 不变。lambda_free=lambda_surface=0.1，tau_surface=0.5×||bounds extent/16||，epsilon=1e-8，均预先冻结，未按 query 调参。",
        "",
        "| Variant | Context AbsRel | Overall query AbsRel (95% CI) | RGB MSE | RMSE | δ1 | Opacity | Coverage |",
        "|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for variant in VARIANTS:
        cell = groups[f"{variant}|direct"]
        q = cell["query_metrics"]
        values = " | ".join(
            number(q[m]["mean"])
            for m in ("rgb_mse", "depth_rmse", "depth_delta1", "opacity", "coverage")
        )
        lines.append(
            f"| {variant} | {number(cell['context_metrics']['depth_absrel']['mean'])} | {stat(q['depth_absrel'])} | {values} |"
        )
    lines.extend(
        [
            "",
            "Coverage 是预测 opacity>1e-6 的比例，不等于真实表面可见性。下面区域标签是 query GT surface 与 context GT nearest-depth 一致性诊断，只有全部136 states 完成并 seal 后才计算。",
            "",
            "| Region | Pixel fraction (scene equal) | S0 conditional AbsRel | Oracle conditional AbsRel | S0−oracle gap | Covered scenes / all | Empty scenes |",
            "|---|---:|---|---|---|---:|---:|",
        ]
    )
    for region in REGIONS:
        conditional = regions["S0"][region]["conditional_absrel"]
        lines.append(
            f"| {region} | {number(regions['S0'][region]['pixel_fraction']['mean'])} | {stat(conditional)} | {stat(regions['ORACLE'][region]['conditional_absrel'])} | {stat(obs['base_conditional_context_oracle_gaps'][region])} | {conditional['n_covered_scenes']}/{conditional['n_scenes']} | {conditional['empty_mask_scene_count']} |"
        )
    angle_regions = ("ANGLE_0_5", "ANGLE_5_15", "ANGLE_15_30", "ANGLE_GT30")
    angle_text = "；".join(
        f"{r}: {stat(regions['S0'][r]['conditional_absrel'])}，scene coverage={number(regions['S0'][r]['conditional_absrel']['scene_coverage'])}"
        for r in angle_regions
    )
    contribution_text = "；".join(
        f"{r}={stat(regions['S0'][r]['additive_error_contribution'])}" for r in REGIONS
    )
    regional_gain_text = "；".join(
        f"{r}: conditional gain={stat(sup['regional_gains'][r]['S3']['conditional_gain'])}，overall additive gain contribution={stat(sup['regional_gains'][r]['S3']['additive_gain_contribution'])}"
        for r in REGIONS
    )
    context_changes = "; ".join(
        f"{v} context={number(groups[v + '|direct']['context_metrics']['depth_absrel']['mean'])}, query={number(groups[v + '|direct']['query_metrics']['depth_absrel']['mean'])}"
        for v in VARIANTS
    )
    answers = [
        (
            "query geometry 中多少属于 OBS0 / OBS1 / OBS2PLUS？",
            "见上表场景等权 pixel fractions；三类保留全部 GT-valid pixels，未因低可观测性删掉 query。OBS0 不等同于物理上绝对不可见，只表示该容差/采样近似下没有 visible support。",
        ),
        (
            "当前误差主要集中在哪种可观测性区域？",
            "总体 AbsRel 的可加区域贡献分别为 "
            + contribution_text
            + "。贡献相加重建 overall，不能只凭某个小区域 conditional error 很高就说它主导总体误差。正式状态="
            + obs["OBSERVABILITY_STATUS"]
            + "。",
        ),
        (
            "OBS2PLUS 上 context-direct 与 query oracle 还差多少？",
            stat(obs["base_conditional_context_oracle_gaps"]["OBS2PLUS"])
            + "。这是覆盖场景上的条件差值，空区域/低覆盖会限制分类；不把 A/B 或像素当独立 scene。",
        ),
        (
            "triangulation angle 与 error 有什么关系？",
            angle_text
            + "。只对>=2 visible views计算；主 bins基于最大夹角[0,5],(5,15],(15,30],>30，median angle及bins另存。这里只描述关联，不推断视差因果。",
        ),
        (
            "occluded regions 的 error 是否更高？",
            "OCCLUDED_ANY S0 conditional="
            + stat(regions["S0"]["OCCLUDED_ANY"]["conditional_absrel"])
            + "；CONFLICT_ANY="
            + stat(regions["S0"]["CONFLICT_ANY"]["conditional_absrel"])
            + "；overall="
            + stat(groups["S0|direct"]["query_metrics"]["depth_absrel"])
            + "。这些是重叠区域的描述性比较，没有额外预注册的遮挡因果检验；不能把条件均值差自动解释为显著因果伤害。",
        ),
        (
            "Free-space supervision 是否改善 query？",
            "S0−S1="
            + stat(contrasts["S0_minus_S1"])
            + "；正值代表净总体收益，稳定 practical gain 还需均值≥.02、CI下界>0、≥75%场景不变差。",
        ),
        (
            "Surface supervision 是否改善 query？",
            "S0−S2="
            + stat(contrasts["S0_minus_S2"])
            + "；使用同一冻结 practical gate，不能仅凭方向为正宣布有效。",
        ),
        (
            "两者联合是否互补？",
            "S0−S3="
            + stat(main_gain)
            + "；S1−S3="
            + stat(contrasts["S1_minus_S3"])
            + "；S2−S3="
            + stat(contrasts["S2_minus_S3"])
            + f"。预注册分类={sup['GEOMETRY_SUPERVISION_STATUS']}；不把联合均值最低自动称为互补。",
        ),
        (
            "主 geometry supervision 缩小多少 context→oracle gap？",
            "S3 GAP_CLOSED="
            + stat(main_closure["gap_closed"])
            + "；GAP_CLOSED_FRACTION="
            + number(main_closure["gap_closed_fraction"]["value"])
            + "，比例CI="
            + str(main_closure["gap_closed_fraction"]["ci95"])
            + "。比例是描述性 gap closure，不是恢复理论最优的百分比；bootstrap 分母非正的未定义抽样数="
            + str(main_closure["gap_closed_fraction"]["undefined_nonpositive_base_draws"])
            + "。",
        ),
        (
            "改善来自 observed 还是 unobserved regions？",
            regional_gain_text
            + "。用有符号可加贡献判断总体来源；OBS0改善可能由regularization/prior/间接一致性引起，没有证据时不能称为从观察恢复了未见真值。",
        ),
        (
            "是否牺牲 context fit 但改善 query？",
            context_changes
            + "。同时检查上方paired query gain，不因context误差单独升高就拒绝方法，也不因context更好就接受。",
        ),
        (
            "wrong-scene damage 是否仍然成立？",
            "S0="
            + stat(control["contrasts"]["S0|wrong_scene"])
            + "；S3="
            + stat(control["contrasts"]["S3|wrong_scene"])
            + "；S3 spatial shuffle damage="
            + stat(control["contrasts"]["S3|spatial_shuffle"])
            + "；zero damage="
            + stat(control["contrasts"]["S3|zero"])
            + "。所有差值为control−direct，recipient camera保持不变。Wrong-scene沿旧定义连同donor原bounds一起替换，因此敏感性不能完全归为density内容的独立贡献；S3 spatial shuffle保持recipient同bounds，补充空间结构证据。不能把CI跨0当scene-specific或空间结构成立。",
        ),
        (
            "当前问题更像 observability、supervision、completion、lifting 或混合？",
            f"冻结证据分类：{obs['OBSERVABILITY_STATUS']} / {sup['GEOMETRY_SUPERVISION_STATUS']}。"
            + recommendations
            + " 这些是本次归因条件下的研究方向，不是单一原因的排他证明。",
        ),
        (
            "下一版 carrier 应具体增加什么？",
            recommendations
            + f" GEOMETRY_AWARE_CARRIER_RECOMMENDED={sup['GEOMETRY_AWARE_CARRIER_RECOMMENDED']}；VISIBILITY_AWARE_FUSION_RECOMMENDED={sup['VISIBILITY_AWARE_FUSION_RECOMMENDED']}。没有在本轮启动carrier训练。",
        ),
    ]
    for i, (question, answer) in enumerate(answers, 1):
        lines.extend(["", f"**{i}. {question}**", "", answer])
    lines.extend(
        [
            "",
            "区域总体贡献与条件均值是不同估计量：每scene先平均query和A/B的区域fraction、error contribution，再以贡献/fraction得到该scene条件均值。空scene的条件值为null，完整scene名单、coverage和empty counts保留；bootstrap每次重采样所有scene并重算有效覆盖分母。未对空区域填零误差，亦未静默删除场景。",
            "",
            "low-support positive gap share="
            + str(obs["positive_low_support_gap_share"])
            + "。该量截取正gap后计算比例，不等于有符号净gap贡献。对应有符号区域gap见observability_analysis.json/signed_additive_gap_by_region。",
            "",
            "visible support是投影在前/图内、有效context depth且ray-distance差≤0.05m+1%context depth的nearest-depth近似，不是表面可见性证明。occlusion与front-conflict分别保存，可在不同context views同时发生，不是可加分区。",
            "",
            "free loss只作用于t<d−tau的sample，surface后方未知不标为空。surface loss为−log(band内rendering mass+eps)，有效hit但空band/miss仅从这个新增term排除并计数；旧任务loss和总体评价仍保留所有合法ray/pixel。surface目标band局部，但梯度通过transmittance还会影响band之前density。",
            "",
            "S0–S3使用相同初始化、seed、ray minibatch stream和固定10k终态。full context objective仅用于诊断，query GT只在seal后标注区域；未用于optimizer、lambda、tau、bounds、checkpoint或scene选择。旧oracle是使用query GT的特权诊断，不是可部署方法，也不是数学能力上界。",
            "",
            f"BASELINE_REPRO_STATUS={baseline['status']}；新S0先完成再通过复现门控，S1–S3才允许启动。正常CUDA浮点复算不承诺bitwise；正式预锁AbsRel容差为scene mean .001、每row .01，实际差值见baseline_reproduction.json。CPU精确回归与CUDA容差检查、state文件/tensor SHA完整性各有不同含义，不能混称。",
            "",
            f"统计单位={bootstrap['unit']}；draws={bootstrap['draws']}、seed={bootstrap['seed']}、95% percentile CI。mean/median/CI/improved-tied-worse/LOSO/top1/top3贡献见bootstrap_results.json。像素不是独立统计样本。",
            "",
            "| Variant | Worker optimization seconds | Seconds/state | Peak allocated bytes | Peak reserved bytes | Direct render seconds |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for variant in VARIANTS:
        value = cost["variants"][variant]
        lines.append(
            f"| {variant} | {value['optimization_seconds_total']:.3f} | {value['optimization_seconds_per_state']:.3f} | {value['peak_cuda_allocated_bytes']} | {value['peak_cuda_reserved_bytes']} | {value['render_seconds_total']:.3f} |"
        )
    lines.extend(
        [
            "",
            f"136个state的worker optimization累计={cost['actual_optimization_seconds']:.3f}秒；这是累计worker时间，不是墙钟总耗时。phase wall-time单独记录："
            + json.dumps(
                cost.get("phase_wall_times", {"status": "NOT_RECORDED"}), ensure_ascii=False
            )
            + "。四进程共享GPU，不能当作独占GPU benchmark；旧参考预测复用，未来carrier成本未实测。",
            "",
            "七类曲线见figures/。完整raw、mask NPZ、配置/锁定、访问日志、测试、state/prediction hashes和复现命令保存在本实验目录；发布归档checkpoint索引指向本地outputs。",
            "",
            "FINAL_HOLDOUT_TOUCHED=false；NEW_CARRIER_TRAINED=false；DYNAMIC_TTT_RUN=false。未打开保护集，未新训练learned carrier。",
        ]
    )
    summary = (
        "CONTEXT GEOMETRY OBSERVABILITY + SUPERVISION FINAL\n"
        + "\n".join(
            f"{k}={str(v).lower() if isinstance(v, bool) else v}" for k, v in terminal.items()
        )
        + "\n"
    )
    (root / "README.md").write_text("\n".join(lines) + "\n")
    (root / "STATUS.md").write_text("```text\n" + summary + "```\n")
    (root / "terminal_summary.txt").write_text(summary)
    print(summary)
    return terminal


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--docs", type=Path)
    args = parser.parse_args()
    run(args.root, args.docs)
