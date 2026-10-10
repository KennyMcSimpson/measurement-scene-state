#!/usr/bin/env python3
"""Write the V20 README from eval_results.json, width_contrast_results.json and the control."""

from __future__ import annotations

import argparse
from pathlib import Path

from mcss.mechanism_pilot.geometry_carrier_experiment import read

KEY_METHODS = (
    "CONSTANT",
    "REPROJ_NN",
    "REPROJ_HARMONIC",
    "T6000",
    "T24000",
    "SEL",
    "HFILL8_T6000",
    "HFILL8_T24000",
    "AHFILL8_T6000",
    "AHFILL8_T24000",
)
COHORT_NAMES = {
    "EVAL_V3": "EVAL-V3",
    "EVAL_V4": "EVAL-V4",
    "EVAL_FAR": "EVAL-FAR",
    "FRESH_V2": "FRESH-V2",
}


def stat(value):
    if not value:
        return "无数据"
    lo, hi = value["ci95"]
    return (
        f"{value['mean']:+.4f} [{lo:+.4f}, {hi:+.4f}]，场景 "
        f"{value['positive']}/{value['tied']}/{value['negative']}"
    )


def mean(summary):
    return f"{summary['mean']:.4f}" if summary else "—"


def contrast_rows(contrasts):
    lines = [
        "| 对比 | 定义 | 差值 [95% CI]，场景 好/平/差 | 标签 | gate |",
        "|---|---|---|---|---|",
    ]
    for name, item in contrasts.items():
        lines.append(
            f"| {name} | {item['definition']} | {stat(item['gain'])} | {item.get('label', '—')} | "
            f"{item.get('gate', '—')} |"
        )
    return lines


def two_by_two(block):
    far, gain = block["absolute_far"], block["length_gain_far"]
    return [
        "| 宽度 | 第 6000 步 | 第 24000 步 | 6000 → 24000 的改善 |",
        "|---|---:|---:|---:|",
        f"| 32（V18） | {mean(far['W32_T6000'])} | {mean(far['W32_T24000'])} | "
        f"{mean(gain['W32'])} |",
        f"| 64（V20） | {mean(far['W64_T6000'])} | {mean(far['W64_T24000'])} | "
        f"{mean(gain['W64'])} |",
    ]


def run(root):
    root = Path(root).resolve()
    result = read(root / "eval_results.json")
    width = read(root / "width_contrast_results.json")
    control = read(root / "audit/v15_prefix_control.json")
    config = read(root / "training_contract.json")
    primary = result["primary"]
    p1 = primary["contrasts"].get("LENGTH_GAIN", {})
    blocks = width["width_contrast"]
    p2 = blocks["primary"]["contrasts"]["WIDTH_GAIN_24K"] if blocks else {}
    hybrid = primary["contrasts"].get("HARMONIC_MINUS_AHFILL8_T24000", {})
    selected = {
        seed: read(root / "checkpoints" / f"L_{seed}" / "selection.json")["selected_step"]
        for seed in config["seeds"]
    }
    lines = [
        "# EXP-3D-RGBD-LONG-TRAIN-WIDTH-DATA-V20",
        "",
        "V15 C1 配方（宽度 64，TRAIN72 + TRAIN-EXT，共 127 个场景）训练到 24000 步，"
        "检验长训练下更宽的载体能否用好更多数据。"
        "计划见 [PLAN_BEFORE_V20_TRAINING.md](PLAN_BEFORE_V20_TRAINING.md)，"
        "协议见 [PROTOCOL.md](PROTOCOL.md)。"
        "在 hit 上用 CPU（AVX2）运行。",
        "",
        f"- **内置对照：** {control['status']}。每个 seed 的前 6000 步训练记录"
        "与第 6000 步参数都与 V15 C1 逐位相同。"
        if control["status"] == "PASS"
        else f"- **内置对照：** {control['status']}",
        f"- **P1（LENGTH_GAIN = FAR(第 6000 步) − FAR(第 24000 步)，宽度 64）：** "
        f"{stat(p1.get('gain'))}，{p1.get('gate', '—')}",
        "- **P2（WIDTH_GAIN_24K = FAR(V18 第 24000 步) − FAR(V20 第 24000 步)）：** "
        + (
            f"{stat(p2.get('gain'))}，{p2.get('gate', '—')}"
            if p2
            else "无数据（V18 没有第 24000 步）"
        ),
        f"- **分支 {width['INTERPRETATION_BRANCH']}**",
        "- **DEV 选中的步数：** " + "，".join(f"seed {s}：{step}" for s, step in selected.items()),
        f"- **描述性：** 调和插值 − AHFILL8(第 24000 步) {stat(hybrid.get('gain'))}，"
        f"{hybrid.get('label', '—')}",
        "",
        f"主队列为 EVAL-V3 ∪ EVAL-V4，共 {primary['n_scenes']} 个场景，FAR 像素占 "
        f"{primary['far_pixel_fraction']:.1%}。",
    ]
    if blocks:
        lines += [
            "",
            "## 宽度 × 步数（主队列 FAR AbsRel，三个 seed 平均）",
            "",
            *two_by_two(blocks["primary"]),
            "",
            *contrast_rows(blocks["primary"]["contrasts"]),
        ]
    lines += [
        "",
        "## 补全随训练步数的变化（V20，主队列，AbsRel，三个 seed 平均）",
        "",
        "| 步数 | FAR | NEAR | ALL |",
        "|---:|---:|---:|---:|",
    ]
    for step, regions in primary["curve"].items():
        lines.append(
            f"| {step} | {mean(regions['FAR'])} | {mean(regions['NEAR'])} | "
            f"{mean(regions['ALL'])} |"
        )
    lines += [
        "",
        "## 主队列各方法的 AbsRel（V20）",
        "",
        "| 方法 | ALL | NEAR | FAR |",
        "|---|---:|---:|---:|",
    ]
    absolute = primary["absolute_absrel"]
    for method in KEY_METHODS:
        if method in absolute:
            a = absolute[method]
            lines.append(f"| {method} | {mean(a['ALL'])} | {mean(a['NEAR'])} | {mean(a['FAR'])} |")
    lines += ["", "## 主队列的全部对比（V20）", ""] + contrast_rows(primary["contrasts"])
    for name, block in result["cohorts"].items():
        if name in ("EVAL_V3", "EVAL_V4"):
            continue
        lines += [
            "",
            f"## {COHORT_NAMES.get(name, name)}（描述性，{block['n_scenes']} 个场景）",
            "",
        ]
        lines += contrast_rows(block["contrasts"])
        if name in blocks:
            lines += ["", *two_by_two(blocks[name]), "", *contrast_rows(blocks[name]["contrasts"])]
    lines += [
        "",
        "## 限制",
        "",
        "- EVAL-V3/V4 已被多个实验使用，FRESH-V2 是第七次使用，本轮结论只作机制对比；"
        "受保护的 final holdout 未打开。",
        "- 只改变训练步数，配方与 V15 C1 相同；P2 与 V18 配对比较，两者只有宽度不同。"
        "结论不推广到更宽的模型或更多、更多样的数据。",
    ]
    (root / "README.md").write_text("\n".join(lines) + "\n")
    print("REPORT_WRITTEN", root / "README.md", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    run(p.parse_args().root)
