#!/usr/bin/env python3
"""Generate the Chinese final report from frozen statistics, never media or states."""

# Chinese report prose is kept intact for editorial review.
# ruff: noqa: E501

import argparse
import hashlib
import json
from pathlib import Path

from mcss.mechanism_pilot.optimization_bounds_statistics import CURRENT, GTFREE, group_key


def read(root, name):
    return json.loads((root / f"{name}.json").read_text())


def number(value):
    return f"{value:.6f}"


def stat(value):
    return f"{value['mean']:.6f}，95% CI [{value['ci95'][0]:.6f}, {value['ci95'][1]:.6f}]"


def run(root, docs=None, secondary=None):
    root = Path(root)
    docs = Path(docs) if docs else root
    results = read(root, "optimization_results")
    suff = read(root, "optimization_sufficiency")
    bounds = read(root, "bounds_analysis")
    oracle = read(root, "query_oracle_diagnostic")
    bootstrap = read(root, "bootstrap_results")
    costs = read(root, "cost_analysis")
    rules = read(root, "decision_rules")
    groups, oracle_groups = results["context_only"], oracle["groups"]
    primary_key = group_key("RGBD", GTFREE, 10000)
    if results["primary_key"] != primary_key or results["budgets"] != [1000, 3000, 10000]:
        raise ValueError("Report only accepts the frozen primary and budget list")
    primary = groups[primary_key]
    decision = suff["primary"]
    label = f"RGBD|{GTFREE}|FIXED_BUDGET"
    comparisons = bootstrap["comparisons"]
    gain = comparisons[f"query_gain|{label}|first_to_max"]
    context_gain = comparisons[f"context_gain|{label}|first_to_max"]
    gap = oracle["context_minus_oracle"][f"{GTFREE}|10000"]
    bounds_main = bounds["RGBD|FIXED_BUDGET"]
    gate = rules["engineering_good"]
    oracle_primary = oracle_groups[group_key("QUERY_ORACLE", GTFREE, 10000)]["metrics"][
        "depth_absrel"
    ]
    oracle_good_count = sum(
        v <= gate["scene_absrel_max"] for v in oracle_primary["per_scene"].values()
    )
    oracle_good = (
        oracle_primary["mean"] <= gate["mean_absrel_max"]
        and oracle_good_count / len(oracle_primary["per_scene"]) >= gate["scene_fraction_min"]
    )
    context_inference = not decision["ENGINEERING_GOOD"] and oracle_good
    representation_bounds = not decision["ENGINEERING_GOOD"] and not oracle_good
    diagnostic_interpretation = (
        "主 context-direct 尚未达到工程好结果门槛，而同一预定 GT-free/10k 的 query-supervised oracle 达到门槛，"
        "支持 context inference / visibility / context evidence 是剩余问题之一。应优先研究如何从合法 context 推断几何。"
        "这不是排他的因果定位，也未证明 representation/bounds 没有任何限制。"
        if context_inference
        else "context-direct 与有限预算 oracle 都未达到工程好结果门槛，representation/bounds 是待查候选；"
        "必须同时保留优化不足解释，不能据此证明表示不可能。"
        if representation_bounds
        else "主 context-direct 已通过工程门槛；这两个用户第24节的失败诊断条件均未触发，是否训练仍由完整 readiness 决定。"
    )
    secondary_cost_note = "Secondary 成本尚未纳入此报告，不能把主阶段累计称为全实验总成本。"
    primary_cost_note = (
        f"主阶段（context + query oracle，119 条轨迹）累计 worker optimization "
        f"{costs['actual_trajectory_optimization_seconds']:.3f} 秒；"
        "主阶段 phase wall-time 独立记录为 "
        + json.dumps(costs.get("phase_wall_times", {"status": "NOT_RECORDED"}), ensure_ascii=False)
        + "。"
    )
    secondary_value = "NOT_RUN"
    secondary_note = "RGB-only secondary 尚未运行；主结论不能外推为无深度的几何适配已解决。"
    if secondary:
        secondary_root = Path(secondary)
        secondary_audit = read(secondary_root, "secondary_decision_audit")
        if secondary_audit.get("status") != "PASS" or not secondary_audit.get(
            "primary_bits_exactly_unchanged"
        ):
            raise PermissionError("Secondary may not alter primary decisions")
        secondary_costs = read(secondary_root, "cost_analysis")
        secondary_seconds = (
            secondary_costs["actual_trajectory_optimization_seconds"]
            - costs["actual_trajectory_optimization_seconds"]
        )
        if secondary_seconds < -1e-6:
            raise ValueError("Combined trajectory cost cannot be below primary cost")
        secondary_cost_note = (
            f"Secondary（34 条轨迹）累计 worker optimization {secondary_seconds:.3f} 秒；"
            f"全阶段（153 条轨迹）累计 worker optimization "
            f"{secondary_costs['actual_trajectory_optimization_seconds']:.3f} 秒；"
            "secondary phase wall-time 独立记录为 "
            + json.dumps(
                secondary_costs.get("secondary_phase_wall_times", {"status": "NOT_RECORDED"}),
                ensure_ascii=False,
            )
            + "。"
        )
        secondary_results = read(secondary_root, "optimization_results")
        secondary_cell = secondary_results["context_only"][group_key("RGB_ONLY", GTFREE, 10000)]
        secondary_value = number(secondary_cell["metrics"]["depth_absrel"]["mean"])
        secondary_note = (
            "RGB-only secondary query AbsRel="
            + stat(secondary_cell["metrics"]["depth_absrel"])
            + "；RGB_ONLY−RGBD="
            + stat(secondary_results["secondary"]["RGB_ONLY_minus_RGBD"])
            + "。这是主分析之后的固定配置诊断，不改变主决策。"
        )
    tests = read(root, "tests") if (root / "tests.json").exists() else {"status": "PENDING"}
    integrity = (
        read(root, "integrity") if (root / "integrity.json").exists() else {"status": "PENDING"}
    )
    terminal = {"N_SCENES": len(results["scene_ids"]), "GRID": 16, "RENDER_SAMPLES": 64}
    for prefix, mode in [("CURRENT", CURRENT), ("GTFREE", GTFREE)]:
        for budget in (1000, 3000, 10000, 30000):
            terminal[f"{prefix}_{budget // 1000}K_ABSREL"] = (
                number(groups[group_key("RGBD", mode, budget)]["metrics"]["depth_absrel"]["mean"])
                if budget != 30000
                else "NOT_RUN_PREDECLARED_COST_LIMIT"
            )
    terminal.update(
        BEST_CONTEXT_ONLY_CONFIG=f"{primary_key}; PREDECLARED_PRIMARY_NOT_QUERY_MINIMUM",
        BEST_CONTEXT_ONLY_ABSREL=number(primary["metrics"]["depth_absrel"]["mean"]),
        QUERY_GAIN_1K_TO_MAX=number(gain["mean"]),
        QUERY_GAIN_CI=gain["ci95"],
        CONTEXT_GAIN_1K_TO_MAX=number(context_gain["mean"]),
        GENERALIZATION_GAP_MAX=number(primary["generalization_gap"]["mean"]),
        QUERY_ORACLE_CURRENT_MAX=number(
            oracle_groups[group_key("QUERY_ORACLE", CURRENT, 10000)]["metrics"]["depth_absrel"][
                "mean"
            ]
        ),
        QUERY_ORACLE_GTFREE_MAX=number(
            oracle_groups[group_key("QUERY_ORACLE", GTFREE, 10000)]["metrics"]["depth_absrel"][
                "mean"
            ]
        ),
        CONTEXT_TO_ORACLE_GAP=number(gap["mean"]),
        OPTIMIZATION_STATUS=decision["OPTIMIZATION_STATUS"],
        BOUNDS_STATUS=bounds_main["BOUNDS_STATUS"],
        ENGINEERING_GOOD=decision["ENGINEERING_GOOD"],
        CARRIER_TRAINING_READINESS=decision["CARRIER_TRAINING_READINESS"],
        MATCHED_CARRIER_TRAINING_RECOMMENDED=decision["MATCHED_CARRIER_TRAINING_RECOMMENDED"],
        RGB_ONLY_SECONDARY_ABSREL=secondary_value,
        CONTEXT_INFERENCE_BOTTLENECK=context_inference,
        REPRESENTATION_OR_BOUNDS_BOTTLENECK=representation_bounds,
        FINAL_HOLDOUT_TOUCHED=False,
        NEW_CARRIER_TRAINED=False,
        DYNAMIC_TTT_RUN=False,
        TESTS=tests.get("status", "UNKNOWN"),
        FINAL_INTEGRITY=integrity.get("status", "PENDING"),
        REPORT_DIR=str(docs),
    )
    recommendation = (
        "下一轮单独建立 EXP-3D-16G-MATCHED-CARRIER-TRAINING-V1；本轮仅给出建议，未训练新 carrier。"
        if decision["MATCHED_CARRIER_TRAINING_RECOMMENDED"]
        else "本轮不建议立即训练新 carrier。先按优化状态及 context−oracle gap 制定独立协议，检查优化充分性、context inference、visibility 与监督；不能用有限预算 oracle 的失败证明表示不可能。"
    )
    lines = [
        "# EXP-3D-16G-OPTIMIZATION-BOUNDS-V1",
        "",
        f"主配置为提前指定的 RGBD / 16³ / renderer64 / GT-free bounds / 10000 steps / FIXED_BUDGET。"
        f"query AbsRel={stat(primary['metrics']['depth_absrel'])}。"
        f"OPTIMIZATION_STATUS={decision['OPTIMIZATION_STATUS']}；"
        f"BOUNDS_STATUS={bounds_main['BOUNDS_STATUS']}；"
        f"CARRIER_TRAINING_READINESS={decision['CARRIER_TRAINING_READINESS']}。",
        "",
        "这是 17 个已暴露场景上的归因实验，不是新独立确认。30k 在正式运行前因成本排除，"
        "NOT_RUN；不根据 query 增加预算。BEST 字段只是 PREDECLARED_PRIMARY，不是事后最低 query。",
        "",
        "| Bounds | Steps | Context AbsRel | Query AbsRel (95% CI) | Query−context gap |",
        "|---|---:|---:|---|---:|",
    ]
    for mode in (CURRENT, GTFREE):
        for budget in results["budgets"]:
            cell = groups[group_key("RGBD", mode, budget)]
            lines.append(
                f"| {mode} | {budget} | {cell['context_metrics']['depth_absrel']['mean']:.6f} | "
                f"{stat(cell['metrics']['depth_absrel'])} | {cell['generalization_gap']['mean']:.6f} |"
            )
    answers = [
        (
            "1000 steps 是否明显 underoptimized？",
            "不能仅凭 objective 下降判断。GT-free 1k→10k query gain="
            + stat(gain)
            + f"；按完整预注册规则得到 {decision['OPTIMIZATION_STATUS']}。",
        ),
        (
            "3k / 10k / 30k 带来多少 query gain？",
            "；".join(
                f"{t['from_budget']}→{t['to_budget']}=" + stat(t["query_gain"])
                for t in suff["transitions"][label]
            )
            + "。30k=NOT_RUN。两 bounds、两 selection 的全部相邻和首末差值见 bootstrap_results.json。",
        ),
        (
            "context objective 下降时 query 是否继续改善？",
            "各相邻预算的 scene 方向计数："
            + json.dumps(
                [
                    {"from": t["from_budget"], "to": t["to_budget"], **t["scene_joint_directions"]}
                    for t in suff["transitions"][label]
                ],
                ensure_ascii=False,
            )
            + "。这些三类不是全部可能方向；完整每 scene 差值保存在统计 JSON。",
        ),
        (
            "是否出现 context overfit？",
            f"预注册状态为 {decision['OPTIMIZATION_STATUS']}。只有 QUERY_OVERFIT 分类建立该项系统性证据；其他状态不排除个别场景 query 变差。",
        ),
        (
            "16³ 最大充分 budget 的 query AbsRel 是多少？",
            stat(primary["metrics"]["depth_absrel"])
            + f"；最大已运行预算为10k，plateau scene fraction={decision['plateau_scene_fraction_at_max']:.6f}。不能把最大已运行预算自动称为充分优化或全局最优。",
        ),
        (
            "GT-free 是否稳定优于 current？",
            "10k CURRENT−GT-free="
            + stat(bounds_main["gains_by_budget"]["10000"])
            + f"；正式分类 {bounds_main['BOUNDS_STATUS']}，稳定收益同时要求均值≥.02、CI下界>0、≥75%场景不变差。",
        ),
        (
            "bounds 收益是否依赖 optimization budget？",
            "收益差的差 (10k bounds gain − 1k bounds gain)="
            + stat(bounds_main["interaction_max_minus_first"])
            + "。判断基于 RGBD 同 role 同 seed 的比较。",
        ),
        (
            "Query oracle 更多优化后能到多少？",
            "10k CURRENT="
            + stat(
                oracle_groups[group_key("QUERY_ORACLE", CURRENT, 10000)]["metrics"]["depth_absrel"]
            )
            + "；GT-free="
            + stat(
                oracle_groups[group_key("QUERY_ORACLE", GTFREE, 10000)]["metrics"]["depth_absrel"]
            )
            + "。两 query 共享 state；监督使用 query GT，只有诊断意义。",
        ),
        (
            "Context-direct 与 oracle gap 是否缩小？",
            "GT-free 10k context−oracle="
            + stat(gap)
            + "；1k gap−10k gap="
            + stat(oracle["oracle_gap_shrinkage_first_to_max"][GTFREE])
            + "。正数表示 gap 缩小，不单独证明因果机制。",
        ),
        (
            "主要剩余瓶颈是什么？",
            f"证据标签：optimization={decision['OPTIMIZATION_STATUS']}，bounds={bounds_main['BOUNDS_STATUS']}。"
            f"CONTEXT_INFERENCE_BOTTLENECK={str(context_inference).lower()}；"
            f"REPRESENTATION_OR_BOUNDS_BOTTLENECK={str(representation_bounds).lower()}。"
            f"Oracle 工程门槛：均值={oracle_primary['mean']:.6f}，"
            f"{oracle_good_count}/{len(oracle_primary['per_scene'])} 个场景 AbsRel≤{gate['scene_absrel_max']}。"
            + diagnostic_interpretation
            + " 这两个 flag 仅按用户第24节和既有工程门槛派生，不改变冻结的三项主分类。",
        ),
        (
            "是否值得训练新的16³ carrier？",
            f"{decision['CARRIER_TRAINING_READINESS']}；engineering-good={decision['ENGINEERING_GOOD']}。逐项资格检查："
            + json.dumps(decision["checks"], ensure_ascii=False)
            + "。READY_WITH_LIMITATION 明确表示尚未通过工程好结果门槛。",
        ),
        ("下一轮训练 carrier 还是先修 context inference？", recommendation),
    ]
    for index, (question, answer) in enumerate(answers, 1):
        lines.extend(["", f"**{index}. {question}**", "", answer])
    lines.extend(
        [
            "",
            "辅助检查：frozen carrier−主配置="
            + stat(suff["primary_capacity_gap"])
            + "；wrong-scene−正确 scene="
            + stat(suff["primary_wrong_scene_damage"])
            + "；query−context="
            + stat(primary["generalization_gap"])
            + "。",
            "",
            secondary_note,
            "",
            "统计单位是 scene：先平均 query，再平均 A/B，再场景等权；10,000 paired scene bootstrap，"
            "seed=20260927，95% percentile CI。per_scene、median、improved/tied/worse、LOSO、top1/top3 贡献保存在 bootstrap_results.json。"
            "两条 selection 均完整报告，CONTEXT_SELECTED 只使用预算内 full context objective，不能替换主配置。",
            "",
            "Oracle CURRENT 的 joint_query seed 与 GT-free A/B seed 不配对；同 bounds 的 RGBD 与 QUERY_ORACLE 也因 track 不同而 seed 不同。"
            "Oracle 跨 bounds 差异不是纯 bounds 因果效果，也不参与 BOUNDS_STATUS 主判定。",
            "",
            "固定 kernel 与 CPU regression 可以精确检查；CUDA 浮点复算不承诺 bitwise 相等，需报告实际容差与误差。"
            "状态 hash 完整性和数值复算容差是不同检查，不能混称。",
            "",
            "成本按真实共享轨迹计一次，不能把不同 budget/selection 的累计时间再相加。四子进程共享 GPU，"
            "累计 worker seconds 不等于 phase wall time，也不可直接当独占 GPU 延迟。未来 carrier training 成本未实测。",
            "",
            primary_cost_note
            + secondary_cost_note
            + "每配置 seconds/state、total seconds、allocated/reserved VRAM 和 render latency 见 cost_analysis.json；所有七组曲线见 figures/。",
            "",
            "FINAL_HOLDOUT_TOUCHED=false；NEW_CARRIER_TRAINED=false；DYNAMIC_TTT_RUN=false。"
            "没有新增模型训练、没有开启 holdout；direct-state 优化属于已授权归因实验。",
            "",
            "复现命令见 commands.sh；raw、配置、锁定、访问日志、测试和完整性证据均在本实验目录，"
            "checkpoint 文件以本地 outputs 与发布索引为准。报告生成不加载任何媒体、模型或 checkpoint。",
        ]
    )
    summary = (
        "16G OPTIMIZATION SUFFICIENCY + GT-FREE BOUNDS FINAL\n"
        + "\n".join(
            f"{key}={str(value).lower() if isinstance(value, bool) else value}"
            for key, value in terminal.items()
        )
        + "\n"
    )
    audit = {
        "status": "PASS",
        "request_source": "User protocol section 24: poor context-direct with good query oracle => context inference diagnostic; both poor => representation/bounds candidate diagnostic",
        "decision_rules_sha256": hashlib.sha256(
            (root / "decision_rules.json").read_bytes()
        ).hexdigest(),
        "report_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "diagnostic_scope": "Report-only derived flags; not exclusive causal attribution; no frozen statistics, thresholds or primary decision changed",
        "primary_key": primary_key,
        "engineering_thresholds": gate,
        "context_engineering_good": decision["ENGINEERING_GOOD"],
        "oracle_engineering_good": oracle_good,
        "oracle_scene_good_count": oracle_good_count,
        "oracle_scene_count": len(oracle_primary["per_scene"]),
        "CONTEXT_INFERENCE_BOTTLENECK": context_inference,
        "REPRESENTATION_OR_BOUNDS_BOTTLENECK": representation_bounds,
        "cost_scope": "Primary119 cumulative worker seconds separate from secondary34 and combined153; wall times reported separately; no prefix double counting",
        "secondary_cost_included": secondary is not None,
    }
    (root / "audit").mkdir(exist_ok=True)
    (root / "audit/report_contract_completion.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n"
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
    parser.add_argument("--secondary", type=Path)
    args = parser.parse_args()
    run(args.root, args.docs, args.secondary)
