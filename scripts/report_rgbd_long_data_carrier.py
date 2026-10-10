#!/usr/bin/env python3
"""Write the V18 README from eval_results.json, the built-in control and the training summaries."""

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


def contrast_rows(block):
    lines = [
        "| 对比 | 定义 | 差值 [95% CI]，场景 好/平/差 | 标签 | gate |",
        "|---|---|---|---|---|",
    ]
    for name, item in block["contrasts"].items():
        lines.append(
            f"| {name} | {item['definition']} | {stat(item['gain'])} | {item.get('label', '—')} | "
            f"{item.get('gate', '—')} |"
        )
    return lines


def run(root):
    root = Path(root).resolve()
    result = read(root / "eval_results.json")
    control = read(root / "audit/v14_prefix_control.json")
    config = read(root / "training_contract.json")
    primary = result["primary"]
    p1 = primary["contrasts"].get("LENGTH_GAIN", {})
    p2 = primary["contrasts"].get("HARMONIC_MINUS_AHFILL8_T24000", {})
    selected = {
        seed: read(root / "checkpoints" / f"L_{seed}" / "selection.json")["selected_step"]
        for seed in config["seeds"]
    }
    lines = [
        "# EXP-3D-RGBD-LONG-TRAIN-DATA-V18",
        "",
        "V14 C1 配方（宽度 32，TRAIN72 + TRAIN-EXT，共 127 个场景）训练到 24000 步，"
        "检验更长的训练能否扭转数据的作用。"
        "计划见 [PLAN_BEFORE_V18_TRAINING.md](PLAN_BEFORE_V18_TRAINING.md)，"
        "协议见 [PROTOCOL.md](PROTOCOL.md)。"
        "在 hit 上用 CPU（AVX2）运行。",
        "",
        f"- **内置对照：** {control['status']}。每个 seed 的前 6000 步训练记录"
        "与第 6000 步参数都与 V14 C1 逐位相同。"
        if control["status"] == "PASS"
        else f"- **内置对照：** {control['status']}",
        f"- **P1（LENGTH_GAIN = FAR(第 6000 步) − FAR(第 24000 步)）：** "
        f"{stat(p1.get('gain'))}，{p1.get('gate', '—')}",
        f"- **P2（调和插值 − AHFILL8(第 24000 步)）：** "
        f"{stat(p2.get('gain'))}，{p2.get('label', '—')}",
        f"- **分支 {result['INTERPRETATION_BRANCH']}**",
        "- **DEV 选中的步数：** " + "，".join(f"seed {s}：{step}" for s, step in selected.items()),
        "",
        f"主队列为 EVAL-V3 ∪ EVAL-V4，共 {primary['n_scenes']} 个场景，FAR 像素占 "
        f"{primary['far_pixel_fraction']:.1%}。",
        "",
        "## 补全随训练步数的变化（主队列，AbsRel，三个 seed 平均）",
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
        "## 主队列各方法的 AbsRel",
        "",
        "| 方法 | ALL | NEAR | FAR |",
        "|---|---:|---:|---:|",
    ]
    absolute = primary["absolute_absrel"]
    for method in KEY_METHODS:
        if method in absolute:
            a = absolute[method]
            lines.append(f"| {method} | {mean(a['ALL'])} | {mean(a['NEAR'])} | {mean(a['FAR'])} |")
    lines += ["", "## 主队列的全部对比", ""] + contrast_rows(primary)
    for name, block in result["cohorts"].items():
        if name in ("EVAL_V3", "EVAL_V4"):
            continue
        lines += [
            "",
            f"## {COHORT_NAMES.get(name, name)}（描述性，{block['n_scenes']} 个场景）",
            "",
        ]
        lines += contrast_rows(block)
    lines += [
        "",
        "## 限制",
        "",
        "- EVAL-V3/V4 已被多个实验使用，FRESH-V2 是第四次使用，本轮结论只作机制对比；"
        "受保护的 final holdout 未打开。",
        "- 只改变训练步数，配方与 V14 C1 相同；结论不推广到更大的模型或更多的数据。",
    ]
    (root / "README.md").write_text("\n".join(lines) + "\n")
    print("REPORT_WRITTEN", root / "README.md", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    run(p.parse_args().root)
