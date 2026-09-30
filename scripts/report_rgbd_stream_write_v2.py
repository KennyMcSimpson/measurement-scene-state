#!/usr/bin/env python3
"""Report the saved Core B V2 DEV results (no model, media or fresh data is opened)."""
# ruff: noqa: E501 -- Chinese scientific report prose

import argparse
import json
from pathlib import Path

LABELS = {
    "NO_STREAM": "只看上下文（静态 Core A 状态）",
    "OFF": "缓存 stream，不写入",
    "OFF_CLAMP3": "缓存 stream，视角计数截断为 3",
    "FUSE": "写入 fuse 矩阵",
    "COMPLETE": "写入 complete 矩阵",
    "ALL": "写入两个矩阵（主方法）",
    "ALL_UNTRAINED": "写入两个矩阵，写入规则未训练",
    "ALL_WRONG_SCENE": "别的场景的快速权重（对照）",
}
NEXT = {
    "A": "在对视角数稳健的 carrier 上，学到的写入稳定超过预注册的非学习基线，并且是场景专属的：Core B 的核心主张首次在 DEV 上得到支持。下一步需要新的独立队列，对 carrier 与写入一起做资格验证。",
    "B": "写入对非学习基线有正的点估计，但主 gate 未全部通过。下一步分析写入在哪些区域、哪些场景起作用，再决定是否重新设计写入。",
    "C": "在对视角数稳健的 carrier 上，冻结的写入设计没有超过非学习基线：Core B 的核心主张未得到支持。下一步重新设计写入（例如利用 stream 帧测得深度的写入统计，或基于梯度的测试时训练）。",
}


def read(root, name, default=None):
    path = Path(root) / (name + ".json")
    if not path.exists() and default is not None:
        return default
    return json.loads(path.read_text())


def stat(value):
    if value is None or value.get("mean") is None:
        return "UNDEFINED"
    lo, hi = value["ci95"]
    return f"{value['mean']:+.4f}，95% CI [{lo:+.4f}, {hi:+.4f}]"


def run(root, docs=None):
    root = Path(root)
    result = read(root, "stream_results")
    contract = read(root, "training_contract")
    tests, integrity = (read(root, n, {"status": "PENDING"}) for n in ("tests", "integrity"))
    absolute, contrasts, statuses = (
        result["absolute_depth_absrel"],
        result["contrasts"],
        result["statuses"],
    )
    branch = result["INTERPRETATION_BRANCH"]
    fields = {
        "N_DEV_SCENES": len(result["scene_ids"]),
        "SEEDS": contract["seeds"],
        "STREAM_LENGTH": contract["stream_length"],
        "WRITE_RULE_STEPS": contract["steps"],
        "CARRIER_VARIANT": contract["carrier_variant"],
        "BASELINE_POLICY": contract["baseline_policy"],
        "DEVICE": "cpu",
        **{f"{p}_QUERY_ABSREL": round(v["mean"], 6) for p, v in absolute.items()},
        **{name: round(v["mean"], 6) for name, v in contrasts.items()},
        **{f"{name}_CI": v["ci95"] for name, v in contrasts.items()},
        **statuses,
        "SEED_ROBUSTNESS": result["SEED_ROBUSTNESS"],
        "INTERPRETATION_BRANCH": branch,
        "SLOW_CARRIER_UPDATED": False,
        "FRESH_QUALIFICATION_INCLUDED": False,
        "FINAL_HOLDOUT_TOUCHED": False,
        "TESTS": tests["status"],
        "FINAL_INTEGRITY": integrity["status"],
    }
    lines = [
        "# EXP-3D-RGBD-DYNAMIC-WRITE-V2",
        "",
        f"Core B 第二轮：在 V9 的 32³ carrier（{contract['carrier_variant']}，按 V9 出结果前写定的计划选择；冻结、从不更新）上，重新检验项目原设计的学习写入规则（DirectWriteRule）。"
        f"每个 episode 为一个上下文角色（3 个 RGB-D 视角）加上 {contract['stream_length']} 个按帧号规则选出的 stream RGB-D 帧；写入规则在 TRAIN72 上训练 {contract['steps']} 步（每个 stream 到达都执行 ALL），不用 DEV 选择 checkpoint；全程 CPU。"
        f"BEYOND_BASELINE=AbsRel({contract['baseline_policy']})−AbsRel(ALL)={stat(contrasts['BEYOND_BASELINE'])}；BEYOND_BASELINE_STATUS={statuses['BEYOND_BASELINE_STATUS']}；SPECIFICITY_STATUS={statuses['SPECIFICITY_STATUS']}；解读分支={branch}。",
        "",
        "| 策略 | 说明 | DEV query AbsRel (95% CI) |",
        "|---|---|---|",
    ]
    for policy, label in LABELS.items():
        if policy in absolute:
            v = absolute[policy]
            lines.append(
                f"| {policy} | {label} | {v['mean']:.4f} [{v['ci95'][0]:.4f}, {v['ci95'][1]:.4f}] |"
            )
    lines += [
        "",
        "| 对比 | 定义 | 结果 | 场景 improved/tied/worse |",
        "|---|---|---|---|",
    ]
    for name, definition in result["contrast_definitions"].items():
        v = contrasts[name]
        lines.append(
            f"| {name} | {definition} | {stat(v)} | {v['positive']}/{v['tied']}/{v['negative']} |"
        )
    lines += [
        "",
        "## 判定",
        "",
        f"- BEYOND_BASELINE_STATUS（主 gate，写入对预注册的非学习基线 {contract['baseline_policy']}）={statuses['BEYOND_BASELINE_STATUS']}",
        f"- SPECIFICITY_STATUS（主 gate，换成别的场景的快速权重是否变差，V2 gate）={statuses['SPECIFICITY_STATUS']}",
        f"- WRITE_STATUS（写入对只缓存）={statuses['WRITE_STATUS']}",
        f"- STREAM_STATUS（写入对静态状态）={statuses['STREAM_STATUS']}",
        f"- BEYOND_CLAMP_STATUS（写入对非学习的计数修正）={statuses['BEYOND_CLAMP_STATUS']}",
        f"- WRITE_SPECIFICITY_STATUS（换成别的场景的快速权重是否变差）={statuses['WRITE_SPECIFICITY_STATUS']}",
        f"- SEED_ROBUSTNESS={result['SEED_ROBUSTNESS']}；各 seed 的 BEYOND_BASELINE="
        + json.dumps(
            {s: round(v["BEYOND_BASELINE"], 4) for s, v in result["per_seed_mean_gains"].items()}
        ),
        "",
        f"分支{branch}：{NEXT[branch]}",
        "",
        "说明：DEV 的 8 个场景在 V2–V10 中反复使用，本轮是 DEV 机制诊断，不是资格验证；V9 的 carrier 是合格 V7 C1 配方在 DEV 上的后继，本身未经独立资格验证；FRESH-V1/V2 未使用；受保护的 final holdout 未打开。",
    ]
    summary = (
        "RGBD DYNAMIC WRITE V2 FINAL\n"
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--docs", type=Path)
    args = parser.parse_args()
    run(args.root, args.docs)
