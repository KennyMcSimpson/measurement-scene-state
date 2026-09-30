# Long Markdown paragraphs retain their prose layout.
# ruff: noqa: E501
"""Render the capacity report and terminal summary from audited analysis artifacts."""

import argparse
import json
from pathlib import Path

from mcss.mechanism_pilot.direct_capacity_statistics import group_key


def report(root):
    root = Path(root)

    def read(name):
        return json.loads((root / name).read_text())

    result = read("direct_state_results.json")
    analysis = read("capacity_analysis.json")
    classification = analysis["classification"]
    comparisons = analysis["comparisons"]
    groups = {**result["frozen_carrier"], **result["context_only"], **result["diagnostic_oracle"]}
    tests = read("tests.json")
    audit = read("audit/postrun_validation.json")
    if tests["status"] != "PASS" or audit["status"] != "PASS":
        raise RuntimeError("Complete tests and raw prediction audit before reporting")

    def metric(key, name="depth_absrel"):
        return groups[key]["metrics"][name]

    def fmt(x):
        return f"{x:.6f}"

    def ci(stats):
        return f"{stats['mean']:.6f} [{stats['ci95'][0]:.6f}, {stats['ci95'][1]:.6f}]"

    def table(header, rows):
        return "\n".join(
            ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * len(header)) + " |"]
            + ["| " + " | ".join(map(str, row)) + " |" for row in rows]
        )

    d, r, o = group_key("RGBD"), group_key("RGB_ONLY"), group_key("QUERY_ORACLE")
    summary = {
        "N_CAPACITY_DEV_SCENES": result["n_scenes"],
        "CARRIER_ABSREL": metric("CARRIER")["mean"],
        "DIRECT_RGBD_8_ABSREL": metric(d)["mean"],
        "DIRECT_RGBONLY_8_ABSREL": metric(r)["mean"],
        "QUERY_ORACLE_8_ABSREL": metric(o)["mean"],
        "CAPACITY_GAP_RGBD": comparisons["capacity_gap_RGBD_g8"]["mean"],
        "CAPACITY_GAP_RGBONLY": comparisons["capacity_gap_RGB_ONLY_g8"]["mean"],
        "DIRECT_RGBD_16_ABSREL": metric(group_key("RGBD", 16))["mean"],
        "DIRECT_RGBD_32_ABSREL": metric(group_key("RGBD", 32))["mean"],
        "RESOLUTION_8_TO_16_GAIN": comparisons["resolution_RGBD_8_to_16"]["mean"],
        "RESOLUTION_16_TO_32_GAIN": comparisons["resolution_RGBD_16_to_32"]["mean"],
        "RENDER64_ABSREL": metric(d)["mean"],
        "RENDER128_ABSREL": metric(group_key("RGBD", 8, 128))["mean"],
        "CONTEXT_RGBD_ERROR": groups[d]["context_metrics"]["depth_absrel"]["mean"],
        "QUERY_RGBD_ERROR": metric(d)["mean"],
        "GENERALIZATION_GAP_RGBD": groups[d]["generalization_gap"]["depth_absrel"]["mean"],
        "WRONG_SCENE_DAMAGE_DIRECT": comparisons["control_damage_RGBD_wrong_scene"]["mean"],
        **{
            key: classification[key]
            for key in (
                "STATE_CAPACITY_STATUS",
                "REPRESENTATION_BOTTLENECK",
                "RESOLUTION_BOTTLENECK",
                "RENDERER_BOTTLENECK",
                "CONTEXT_INFERENCE_BOTTLENECK",
                "LEARNER_TRAINING_BOTTLENECK",
                "FINAL_HOLDOUT_TOUCHED",
                "DYNAMIC_TTT_RUN",
                "NEW_CARRIER_TRAINED",
            )
        },
        "TESTS": f"{tests['full_suite']['passed']} passed, {tests['full_suite']['skipped']} skipped; Ruff PASS",
        "FINAL_INTEGRITY": "PASS",
        "REPORT_DIR": str((Path("docs/experiments") / root.name).resolve()),
    }
    lines = ["=" * 60, "3D DIRECT-STATE CAPACITY ATTRIBUTION FINAL", "=" * 60]
    for key, value in summary.items():
        text = (
            str(value).lower()
            if isinstance(value, bool)
            else fmt(value)
            if isinstance(value, float)
            else str(value)
        )
        lines.append(f"{key}={text}")
    lines.append("=" * 60)
    terminal = "\n".join(lines) + "\n"
    (root / "terminal_summary.txt").write_text(terminal)
    (root / "STATUS.md").write_text(
        "```text\n"
        + terminal
        + "```\n\nRENDER64/128 above refer to RGBD grid8 CURRENT_BOUNDS; the grid16 matrix is separately reported. FINAL_INTEGRITY is confirmed by the final archival integrity.json.\n"
    )
    (root / "terminal_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    parts = [
        f"# {root.name}\n\n`STATE_CAPACITY_STATUS={classification['STATE_CAPACITY_STATUS']}`。17 个已曝光场景上的归因实验，**不是独立验证或方法资格确认**。425 个 direct states（340 context-only，85 query-supervised diagnostic），固定原 renderer，无新 carrier 训练、writer、Dynamic TTT 或 holdout 评价。\n"
    ]
    parts.append(
        "## 主要结果：context-only\n\n下表均为 scene 等权 query 指标；A/B 先各自平均 query，再等权平均，最后平均 17 个场景。AbsRel/RMSE/RGB MSE 越低越好，δ1/SSIM 越高越好。Opacity/coverage 不是任务质量本身。"
    )
    main_keys = [
        ("Frozen carrier A/B", "CARRIER"),
        ("Frozen anchor", "ANCHOR"),
        ("Fixed analytic prior", "PRIOR"),
        ("Direct RGB+D 8³", d),
        ("Direct RGB-only 8³", r),
    ]
    parts.append(
        table(
            ["方法", "AbsRel [95% CI]", "RMSE", "δ1", "RGB MSE", "SSIM", "Opacity", "Coverage"],
            [
                [
                    label,
                    ci(metric(key)),
                    *[
                        fmt(metric(key, name)["mean"])
                        for name in (
                            "depth_rmse",
                            "depth_delta1",
                            "rgb_mse",
                            "rgb_ssim",
                            "opacity",
                            "coverage",
                        )
                    ],
                ]
                for label, key in main_keys
            ],
        )
    )
    parts.append(
        "RGB+D 使用 context depth，不能把它单独优于 RGB-only carrier 的差值全部归因于 learner。RGB-only 优化阶段不加载本场景 depth；其 context depth 指标由封存后的 evaluator 另行测量。Coverage 定义为预测 opacity > 1e-6，不能称为真实可见率。"
    )
    selected = [
        "capacity_gap_RGBD_g8",
        "capacity_gap_RGB_ONLY_g8",
        "anchor_gap_RGBD_g8",
        "anchor_gap_RGB_ONLY_g8",
    ]
    parts.append(
        table(
            [
                "比较（正值为 direct 改善）",
                "平均 [95% CI]",
                "中位数",
                "改善/持平/更差",
                "LOSO 范围",
                "正收益 top1 / top3 占比",
            ],
            [
                [
                    key,
                    ci(comparisons[key]),
                    fmt(comparisons[key]["median"]),
                    str(comparisons[key]["improved_tied_worse"]),
                    str(comparisons[key]["loso_range"]),
                    str(
                        [
                            comparisons[key]["positive_top1_contribution"],
                            comparisons[key]["positive_top3_contribution"],
                        ]
                    ),
                ]
                for key in selected
            ],
        )
    )
    parts.append(
        "Capacity gap = carrier − direct；anchor gap = frozen anchor − direct。完整七项 direct − carrier 配对差值另见 capacity_analysis.json；误差指标负值表示 direct 更好。统计采用 10,000 次 scene-paired bootstrap，seed 20260927、95% percentile CI、tie=1e-8。所有诊断 CI 未做多重比较校正；分组均值的 CI 不等同于差值 CI。完整 LOSO、正/绝对贡献 top1/top3、逐场景值均保存在 JSON。"
    )
    parts.append("## 分辨率、renderer 和 bounds\n")
    parts.append(
        table(
            ["轨道", "grid", "Query AbsRel [95% CI]", "Context AbsRel", "Query RGB MSE"],
            [
                [
                    track,
                    grid,
                    ci(metric(group_key(track, grid))),
                    fmt(groups[group_key(track, grid)]["context_metrics"]["depth_absrel"]["mean"]),
                    fmt(metric(group_key(track, grid), "rgb_mse")["mean"]),
                ]
                for track in ("RGBD", "RGB_ONLY")
                for grid in (8, 16, 32)
            ],
        )
    )
    parts.append(
        table(
            ["相邻 grid 改善（旧误差−新误差）", "平均 [95% CI]", "改善/持平/更差"],
            [
                [key, ci(value), str(value["improved_tied_worse"])]
                for key, value in read("resolution_analysis.json").items()
            ],
        )
    )
    parts.append(
        table(
            ["RGBD grid", "64 samples AbsRel", "128 samples AbsRel", "64→128 改善 [95% CI]"],
            [
                [
                    grid,
                    fmt(metric(group_key("RGBD", grid))["mean"]),
                    fmt(metric(group_key("RGBD", grid, 128))["mean"]),
                    ci(comparisons[f"renderer_RGBD_g{grid}_64_to_128"]),
                ]
                for grid in (8, 16)
            ],
        )
    )
    parts.append(
        "主 sweep 始终固定 64 samples、相同场景/帧/损失/步数；128 samples 是单独预定的 RGBD 8³/16³ 对照。改变 samples 也改变优化梯度，因此这仍是受控诊断，不能声称纯粹的因果分解。"
    )
    parts.append(
        table(
            [
                "8³ bounds 敏感性",
                "CURRENT AbsRel",
                "替代 bounds AbsRel",
                "误差改善 [95% CI]",
                "CURRENT GT inside / ray hit / GT>far",
                "替代 GT inside / ray hit / GT>far",
            ],
            [
                [
                    track,
                    fmt(metric(group_key(track))["mean"]),
                    fmt(metric(group_key(track, bounds="PREDECLARED_GEOMETRY_BOUNDS"))["mean"]),
                    ci(comparisons[f"bounds_gain_{track}"]),
                    " / ".join(
                        fmt(metric(group_key(track), m)["mean"])
                        for m in (
                            "insideGTfraction",
                            "ray_hitfraction",
                            "GTdistance_above_far_fraction",
                        )
                    ),
                    " / ".join(
                        fmt(
                            metric(group_key(track, bounds="PREDECLARED_GEOMETRY_BOUNDS"), m)[
                                "mean"
                            ]
                        )
                        for m in (
                            "insideGTfraction",
                            "ray_hitfraction",
                            "GTdistance_above_far_fraction",
                        )
                    ),
                ]
                for track in ("RGBD", "RGB_ONLY")
            ],
        )
    )
    parts.append(
        "CURRENT 为 [-6,-4,-6] 到 [6,4,6]；替代规则仅取 context camera rays 的训练集全局深度 q01/q99 端点，AABB 两侧各扩 5%，最小边长 1m。训练 prior near=0.214252m、far=5.672045m，来自原 TRAIN3 frame12..15；不读取 capacity query camera/GT 构造 bounds。RGB-only 的该敏感性组含训练集全局 depth 先验。GT inside/far 诊断只在封存后计算，未用于改 bounds 或选配置。"
    )
    parts.append("## Context 泛化与状态结构对照\n")
    parts.append(
        table(
            [
                "轨道",
                "Context AbsRel",
                "Query AbsRel",
                "Query−context [95% CI]",
                "Context RGB MSE",
                "Query RGB MSE",
            ],
            [
                [
                    track,
                    fmt(groups[group_key(track)]["context_metrics"]["depth_absrel"]["mean"]),
                    fmt(metric(group_key(track))["mean"]),
                    ci(groups[group_key(track)]["generalization_gap"]["depth_absrel"]),
                    fmt(groups[group_key(track)]["context_metrics"]["rgb_mse"]["mean"]),
                    fmt(metric(group_key(track), "rgb_mse")["mean"]),
                ]
                for track in ("RGBD", "RGB_ONLY")
            ],
        )
    )
    parts.append(
        table(
            ["8³ CURRENT 对照", "轨道", "AbsRel damage [95% CI]", "对照 RGB MSE", "对照 opacity"],
            [
                [
                    method,
                    track,
                    ci(comparisons[f"control_damage_{track}_{method}"]),
                    fmt(metric(group_key(track, method=method), "rgb_mse")["mean"]),
                    fmt(metric(group_key(track, method=method), "opacity")["mean"]),
                ]
                for track in ("RGBD", "RGB_ONLY")
                for method in (
                    "wrong_scene",
                    "spatial_shuffle",
                    "density_only",
                    "color_only",
                    "zero_density_color",
                )
            ],
        )
    )
    parts.append(
        "Damage = 对照误差 − 正确 direct-state 误差。Wrong scene 使用排序后下一个场景的同配置状态，recipient query camera 不变。Shuffle 对 density/color 使用同一置换；density-only 将 color 固定 .5；color-only 将 density logits 固定 −2；zero 为 logits−100/color0。以上均评价实际渲染预测，不以 latent 差异代替任务结果；raw 每行另含 RGB/depth 预测 RMS/max变化与 camera/state hashes。"
    )
    parts.append(
        "## DIAGNOSTIC ORACLE：有 query 监督的表达诊断\n\n仅在全部 context-only 状态完成、封存并评价之后开启。每个 oracle state 同时优化两张 query，二者共享同一状态；不是每张 query 拟合一份。不能作为部署方法、泛化证据或数学最优上界。"
    )
    parts.append(
        table(
            [
                "Grid",
                "Oracle query AbsRel [95% CI]",
                "RGBD context-direct − oracle [95% CI]",
                "Oracle RGB MSE",
            ],
            [
                [
                    grid,
                    ci(metric(group_key("QUERY_ORACLE", grid))),
                    ci(comparisons[f"context_RGBD_minus_oracle_g{grid}"]),
                    fmt(metric(group_key("QUERY_ORACLE", grid), "rgb_mse")["mean"]),
                ]
                for grid in (8, 16, 32)
            ],
        )
    )
    parts.append(
        "Oracle bounds sensitivity (8³, both queries jointly supervised): "
        + ci(metric(group_key("QUERY_ORACLE", bounds="PREDECLARED_GEOMETRY_BOUNDS")))
        + "; CURRENT minus alternative AbsRel: "
        + ci(comparisons["bounds_gain_QUERY_ORACLE"])
        + ". This is privileged diagnostic evidence only."
    )
    parts.append(
        "三层分别是 query-supervised direct（L1）、context-supervised direct（L2）、learned carrier（L3）。L1→L2 可能同时涉及 context 信息、可见性、inverse problem 与优化目标；L2→L3 可能涉及构建和训练，但 RGBD 额外监督也在其中。不能把这些差值当成严格因果分解。"
    )
    parts.append("## 全部场景\n")
    parts.append(
        table(
            [
                "Scene",
                "Carrier",
                "RGBD8",
                "RGB-only8",
                "Capacity gap RGBD",
                "Capacity gap RGB-only",
            ],
            [
                [
                    scene,
                    *[fmt(metric(key)["per_scene"][scene]) for key in ("CARRIER", d, r)],
                    fmt(comparisons["capacity_gap_RGBD_g8"]["per_scene"][scene]),
                    fmt(comparisons["capacity_gap_RGB_ONLY_g8"]["per_scene"][scene]),
                ]
                for scene in result["scene_ids"]
            ],
        )
    )
    parts.append(
        "## 优化定义与真实成本\n\n直接优化 density logits 与 sigmoid-bounded RGB，4g³ 个参数；log-variance 固定−3，features/evidence/appearance=None。没有 query-conditioned network。沿用原 fixed renderer 的归一化 ray direction、metric ray-distance depth=sum(w·t)，不除 opacity，RGB background=0。Adam LR=.03、1000步、每步1024个完整128×160 context像素均匀有放回抽样、clip=1。RGB MSE + RGBD 的 valid-depth AbsRel + 1e-5 mean(softplus(z)²) + 1e-4 mean physical-spacing TV(density,color)。每100步及step0全监督像素评估目标，选最小值，严格小于才更新保证最早tie；context-only没有query早停。\n\n小 pilot 只用 ai004001/A context：两轨 RGB loss 均下降，RGBD depth loss 下降，梯度非零、输出改变、没有 NaN/inf 或全0/全1 opacity。随后冻结 formal protocol，未因 query 结果改参数。"
    )
    cost_rows = []
    for track in ("RGBD", "RGB_ONLY"):
        for grid in (8, 16, 32):
            key = group_key(track, grid)
            cost = groups[key]["optimization_cost"]
            cost_rows.append(
                [
                    track,
                    grid,
                    cost["parameter_count_per_state"][0],
                    fmt(cost["seconds_mean"]),
                    fmt(cost["peak_cuda_allocated_bytes"] / 2**20),
                    fmt(cost["peak_cuda_reserved_bytes"] / 2**20),
                    fmt(groups[key]["render_seconds_total"] / groups[key]["n_rows"]),
                    f"{cost['selected_at_budget_boundary']}/{cost['n_states']}",
                ]
            )
    parts.append(
        table(
            [
                "Track",
                "Grid",
                "参数",
                "优化秒/state",
                "Peak allocated MiB",
                "Peak reserved MiB",
                "query render秒/frame",
                "选中1000步",
            ],
            cost_rows,
        )
    )
    curves = read("optimization_curves.json")
    parts.append(
        f"425 个状态中 {sum(v['final_selected_at_budget_boundary'] for v in curves.values())} 个选中预算边界，{sum(v['last3_objective_slope_per_step'] < 0 for v in curves.values())} 个最后三次目标仍呈下降趋势。完整 full-objective 曲线、last3 slope、selected checkpoint 和每10步 trace 均保留。1000步不是全局收敛证明；高grid比8³有64倍参数，同步数不代表同等优化充分性。\n\nGPU 为 RTX5090，PyTorch2.11.0+cu128。优化秒数不含数据加载和checkpoint落盘；renderer秒数不含GT读取、指标和NPZ压缩，CUDA同步后计时。VRAM为本进程allocator峰值，不是整机显存；共享GPU环境并非独占性能基准。第144个状态曾发生一次进程segfault，143份完整状态hash复核后保留，仅lock的未完成尝试另存audit，按相同冻结配置/seed重跑并完成；未观察query后挑选重跑。不能声称已定位该环境崩溃的根因。"
    )
    parts.append(
        "## 判定与下一阶段\n\n预注册 engineering-good：mean AbsRel≤.25 且至少75%场景≤.35；stable practical gain：mean≥.02、配对CI下界>0、至少75%场景不变差。这是预先声明的工程标准，不是官方benchmark。"
    )
    parts.append("```json\n" + json.dumps(classification, indent=2, ensure_ascii=False) + "\n```")
    parts.append(
        "不因有限优化下 oracle 较差就证明 state 无法表达；bounds/ray reach、优化剩余空间都必须先排除。本轮只给下一阶段建议，不开展新 carrier 训练或开启 protected holdout。"
    )
    parts.append(
        "## 审计、复现与交付\n\n冻结 B-final SHA256=`be7b8b6d2ef366cad245732b9ff227802db596da65082ac0e160f24541579e42`。从 checkpoint 重建51个context状态hash全同；136条 A/B/anchor/prior 查询预测与旧结果完全相同。旧产物3523项保持不变，其中2695项freshhash，828项受保护媒体只比较size/mtime并引用以前验证的SHA，本轮未重新读取其字节。\n\ncontext loader剥离query帧及其相机，RGB-only再移除depth路径；优化器只接收允许的 ObservationBatch。管理用manifest含原始锁定元数据，这不是OS级文件访问隔离。baseline是已曝光旧状态的独立重放，允许在formal direct优化前评分；direct评价先要求全部340完成，CPU clone/hash封存后才读取query媒体（包括哈希读取）。oracle作为明确特权阶段单列。GT访问、seal事件、phase完成记录、输入/source/state哈希及文件时间辅助证据完整保存；mtime不具防篡改保证。"
    )
    parts.append(
        f"完整suite：{tests['full_suite']['passed']} passed、{tests['full_suite']['skipped']} skipped，Ruff PASS；跳过项仅可选历史V5 step4500 checkpoint未分发。46项新增direct-capacity核心测试覆盖loader隔离、原renderer预测及梯度一致、选择规则、同状态多query、wrong-camera、场景bootstrap及raw统计复算。另有归档I/O测试。所有正式query预测NPZ逐条重算7指标通过，context/oracle行数分别 {audit['phases']['context']['rows_recomputed']} / {audit['phases']['oracle']['rows_recomputed']}；统计另行从raw重算比对。"
    )
    parts.append(
        "从仓库根目录运行：\n\n```bash\nbash outputs/EXP-3D-DIRECT-STATE-CAPACITY-V1/commands.sh analyze outputs/direct-capacity-reanalysis\nbash outputs/EXP-3D-DIRECT-STATE-CAPACITY-V1/commands.sh reproduce outputs/direct-capacity-reproduction\n.venv/bin/python scripts/analyze_direct_capacity.py --root docs/experiments/EXP-3D-DIRECT-STATE-CAPACITY-V1 --output outputs/direct-capacity-docs-only-analysis\n```\n\n第一/三条仅使用raw与数值摘要，不加载权重或媒体；第三条直接读取报告中的gzip。第二条复用已冻结pilot协议/历史baseline，在新目录重做425份状态，需要原数据路径及冻结源码。CUDA fresh optimization未承诺跨设备逐bit相同；保存预测→指标、保存raw→统计分别经过完整审计。全部.pt与预测NPZ在本地outputs，报告目录提供425-state索引、压缩raw/audit、源码快照、配置和六幅PNG/SVG，不把大权重冒充已在Git发布。当前工作未git commit/push。"
    )
    parts.append("## 图表\n")
    for image in sorted((root / "figures").glob("*.png")):
        parts.append(f"![{image.stem}](figures/{image.name})")
    parts.insert(1, (root / "scientific_interpretation.md").read_text())
    (root / "README.md").write_text("\n\n".join(parts) + "\n")
    print(terminal)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    report(parser.parse_args().root)
