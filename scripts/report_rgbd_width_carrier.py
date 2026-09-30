#!/usr/bin/env python3
"""Report the saved V12 results: EVAL-V4 primary gates, DEV secondary (no model or media)."""
# ruff: noqa: E501 -- Chinese scientific report prose

import argparse
import json
from pathlib import Path

V11 = Path("outputs/EXP-3D-RGBD-COMPLETION-V11")
NEXT = {
    "A": "宽度从 32 加到 64 继续改善了未观测区域的补全，最宽的 carrier 在可部署组合中也超过了经典几何补洞：补全随容量增长的趋势在 CPU 可及的范围内仍未饱和，这支持在 GPU 上继续扩大容量。",
    "B": "两个条件只成立一个。需要分开解读：是补全本身随宽度改善但还不足以胜过经典补洞，还是组合优势并非来自更宽的 carrier。",
    "C": "宽度 64 没有稳定改善补全，也没有让组合胜过经典几何补洞：在 TRAIN72 的数据量下，宽度 32 之后的容量收益已经看不到。",
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


def c0_reproduces_v11(root, contract):
    """Largest |DEV query AbsRel| difference between V12 C0 and V11 C1, per step."""
    diffs = []
    for seed in contract["seeds"]:
        mine = read(root / "checkpoints" / f"C0_{seed}", "dev_curve", [])
        base = read(V11 / "checkpoints" / f"C1_{seed}", "dev_curve", [])
        if not mine or len(mine) != len(base):
            return None
        diffs += [
            abs(a["query_absrel"] - b["query_absrel"]) for a, b in zip(mine, base, strict=True)
        ]
    return max(diffs) if diffs else None


def run(root, docs=None):
    root, docs = Path(root), Path(docs) if docs else Path(root)
    contract = read(root, "training_contract")
    eval_v4 = read(root, "eval_v4_results")
    static = read(root, "static_results")
    qualification = read(root, "qualification_results")
    tests, integrity = (read(root, n, {"status": "PENDING"}) for n in ("tests", "integrity"))
    absolute = eval_v4["absolute_absrel"]
    statuses = eval_v4["statuses"]
    branch = eval_v4["INTERPRETATION_BRANCH"]
    groups = static["groups"]
    reproduction = c0_reproduces_v11(root, contract)
    methods = ("REPROJ_NN", "REPROJ_HARMONIC", "V11C1", "C0", "C1", "HFILL8_V11C1", "HFILL8_C1")
    fields = {
        "EVAL_V4_SCENES": eval_v4["n_scenes"],
        "FAR_PIXEL_FRACTION": round(eval_v4["far_pixel_fraction"], 4),
        **{
            f"EVAL_V4_{method}_{region}": round(absolute[method][region]["mean"], 6)
            for method in methods
            for region in ("ALL", "NEAR", "FAR")
            if method in absolute and (absolute[method][region] or {}).get("mean") is not None
        },
        **{
            f"{name}{suffix}": (
                (round(gain["mean"], 6) if suffix == "" else gain["ci95"]) if gain else "UNDEFINED"
            )
            for name, gain in (
                ("WIDTH64_COMPLETION_GAIN", eval_v4["WIDTH64_COMPLETION_GAIN"]),
                ("HARMONIC_HYBRID_GAIN", eval_v4["HARMONIC_HYBRID_GAIN"]),
                ("HFILL8_WIDTH_GAIN", eval_v4["HFILL8_WIDTH_GAIN"]),
            )
            for suffix in ("", "_CI")
        },
        **statuses,
        "V12_C0_MINUS_V11_C1_EVAL_V4_MAX_ABS": eval_v4["consistency"][
            "v12_c0_minus_v11_c1_max_abs"
        ],
        "CONTROL_MINUS_REPLICATION_MAX_ABS": eval_v4["consistency"][
            "control_minus_replication_max_abs"
        ],
        "DEV_C0_QUERY_ABSREL": round(
            groups["C0|direct"]["query_metrics"]["depth_absrel"]["mean"], 6
        ),
        "DEV_C1_QUERY_ABSREL": round(
            groups["C1|direct"]["query_metrics"]["depth_absrel"]["mean"], 6
        ),
        "WIDTH_GAIN_DEV": round(static["surface_gain"]["mean"], 6),
        "WIDTH_GAIN_DEV_CI": static["surface_gain"]["ci95"],
        "WIDTH_DEV_STATUS": qualification["SURFACE_TRAINING_STATUS"],
        "STATIC_DEV_STATUS": qualification["STATIC_DEV_STATUS"],
        "C0_REPRODUCES_V11_C1_MAX_DEV_DIFF": reproduction,
        "INTERPRETATION_BRANCH": branch,
        "FRESH_QUALIFICATION_INCLUDED": False,
        "FINAL_HOLDOUT_TOUCHED": False,
        "DYNAMIC_TTT_NEXT_STAGE_ALLOWED": False,
        "TEST_TIME_INPUT": "RGB+DEPTH+CAMERA",
        "TESTS": tests["status"],
        "FINAL_INTEGRITY": integrity["status"],
        "REPORT_DIR": str(docs),
    }
    lines = [
        "# EXP-3D-RGBD-WIDTH-V12",
        "",
        f"单因素机制检验：在 V11 C1 的配方上把 carrier 宽度从 32 加到 64（250,542 个参数）。按 V11 出结果前写定的计划，只有 V11 的 COMPLETION_GAIN 点估计 > 0 时才运行；主对比的对照是 V11 C1 已选中的 checkpoint（不重新训练）。C0 按同一配方重新训练宽度 32，只用来检查逐位复现。主评价在从未读取过的 EVAL-V4（{eval_v4['n_scenes']} 个场景）上进行；所有预测在读取 query 深度之前封存。全程 CPU。",
        "",
        f"FAR 为离上下文深度重投影命中点超过 8 像素的像素（占 {eval_v4['far_pixel_fraction']:.1%}）；HFILL8 在其余像素用调和插值补洞（INPAINT-V1 的主基线），在 FAR 用 carrier。",
        "",
        "| EVAL-V4 AbsRel | ALL | NEAR | FAR |",
        "|---|---:|---:|---:|",
    ]
    for method in methods:
        if method in absolute:
            cells = [(absolute[method][r] or {}).get("mean") for r in ("ALL", "NEAR", "FAR")]
            lines.append(
                f"| {method} | "
                + " | ".join("—" if c is None else f"{c:.4f}" for c in cells)
                + " |"
            )
    lines += [
        "",
        "## 判定",
        "",
        f"- WIDTH64_COMPLETION_GAIN = FAR AbsRel(V11 C1) − FAR AbsRel(V12 C1) = {stat(eval_v4['WIDTH64_COMPLETION_GAIN'])}；{statuses['WIDTH64_COMPLETION_STATUS']}",
        f"- HARMONIC_HYBRID_GAIN = AbsRel(REPROJ_HARMONIC) − AbsRel(HFILL8 V12 C1) = {stat(eval_v4['HARMONIC_HYBRID_GAIN'])}；{statuses['HARMONIC_HYBRID_LABEL']}",
        f"- HFILL8_WIDTH_GAIN = AbsRel(HFILL8 V11 C1) − AbsRel(HFILL8 V12 C1) = {stat(eval_v4['HFILL8_WIDTH_GAIN'])}；{statuses['HFILL8_WIDTH_STATUS']}",
        f"- DEV（次要）：C0 {fields['DEV_C0_QUERY_ABSREL']:.4f}，C1 {fields['DEV_C1_QUERY_ABSREL']:.4f}，WIDTH_GAIN_DEV={stat(static['surface_gain'])}，{fields['WIDTH_DEV_STATUS']}",
        f"- 完整性：V12 C0 与 V11 C1 的 DEV 曲线最大差 = {reproduction}；EVAL-V4 预测最大差 = {fields['V12_C0_MINUS_V11_C1_EVAL_V4_MAX_ABS']}；对照与复现实验封存预测的最大差 = {fields['CONTROL_MINUS_REPLICATION_MAX_ABS']}",
        "",
        f"分支{branch}：{NEXT[branch]}",
        "",
        "说明：DEV 的 8 个场景在 V2–V11 中反复使用，本轮只用它选择 checkpoint 并给出描述性结果；EVAL-V4 是机制队列，不是资格验证；FRESH-V1/V2 与 EVAL-V3 未使用；受保护的 final holdout 未打开。",
    ]
    summary = (
        "RGBD WIDTH V12 FINAL\n"
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
