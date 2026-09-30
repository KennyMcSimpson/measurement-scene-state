#!/usr/bin/env python3
"""Report the saved V11 results: EVAL-V3 primary gates, DEV secondary (no model or media)."""
# ruff: noqa: E501 -- Chinese scientific report prose

import argparse
import json
from pathlib import Path

V9 = Path("outputs/EXP-3D-RGBD-VIEWCOUNT-V9")
NEXT = {
    "A": "加宽 carrier 改善了补全，测量保留组合（FILL8，宽 carrier）也超过了直接重投影：学到的补全第一次在足够大的机制队列上带来超过几何的收益。下一步需要一个按 volume 独立的队列做确认；仍不开 Dynamic TTT 的资格验证。",
    "B": "两个主 gate 只成立一个。下一步分析补全收益集中在哪些场景与区域，再决定继续扩容量还是扩训练数据。",
    "C": "加宽没有稳定改善补全，测量保留组合也没有超过直接重投影：在这个规模下，学到的补全还不足以让状态超过几何。",
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


def c0_reproduces_v9(root, contract):
    """Largest |DEV query AbsRel| difference between V11 C0 and the V9 base variant, per step."""
    diffs = []
    for seed in contract["seeds"]:
        mine = read(root / "checkpoints" / f"C0_{seed}", "dev_curve", [])
        base = read(V9 / "checkpoints" / f"{contract['base_variant']}_{seed}", "dev_curve", [])
        if not mine or len(mine) != len(base):
            return None
        diffs += [
            abs(a["query_absrel"] - b["query_absrel"]) for a, b in zip(mine, base, strict=True)
        ]
    return max(diffs) if diffs else None


def run(root, docs=None):
    root, docs = Path(root), Path(docs) if docs else Path(root)
    contract = read(root, "training_contract")
    eval_v3 = read(root, "eval_v3_results")
    static = read(root, "static_results")
    qualification = read(root, "qualification_results")
    tests, integrity = (read(root, n, {"status": "PENDING"}) for n in ("tests", "integrity"))
    absolute = eval_v3["absolute_absrel"]
    statuses = eval_v3["statuses"]
    branch = eval_v3["INTERPRETATION_BRANCH"]
    groups = static["groups"]
    reproduction = c0_reproduces_v9(root, contract)
    fields = {
        "BASE_VARIANT": contract["base_variant"],
        "BASE_EXTRA": contract["base_extra"],
        "EVAL_V3_SCENES": eval_v3["n_scenes"],
        "FAR_PIXEL_FRACTION": round(eval_v3["far_pixel_fraction"], 4),
        **{
            f"EVAL_V3_{method}_{region}": round(absolute[method][region]["mean"], 6)
            for method in ("C0", "C1", "REPROJ_NN", "FILL8_C0", "FILL8_C1")
            for region in ("ALL", "NEAR", "FAR")
            if method in absolute and (absolute[method][region] or {}).get("mean") is not None
        },
        **{
            f"{name}{suffix}": (
                (round(gain["mean"], 6) if suffix == "" else gain["ci95"]) if gain else "UNDEFINED"
            )
            for name, gain in (
                ("COMPLETION_GAIN", eval_v3["COMPLETION_GAIN"]),
                ("HYBRID_GAIN", eval_v3["HYBRID_GAIN"]),
            )
            for suffix in ("", "_CI")
        },
        **statuses,
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
        "C0_REPRODUCES_V9_BASE_MAX_DEV_DIFF": reproduction,
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
        "# EXP-3D-RGBD-COMPLETION-V11",
        "",
        f"单因素机制检验：carrier 宽度对补全的作用。两个变体都是 V9 {contract['base_variant']} 的配方（V9 出结果前写定的规则选出），只有宽度不同：C0 为 hidden 8（5150 个参数，重新训练并应与 V9 逐位相同），C1 为 hidden 32（64238 个参数）。主评价在 EVAL-V3（{eval_v3['n_scenes']} 个场景，按场景与源资产与全部已用场景不相交，与 DEV、FRESH 的 volume 不相交，可能与 TRAIN72 共享 volume）上进行；所有预测在读取 query 深度之前封存。全程 CPU。",
        "",
        f"FAR 为离上下文深度重投影命中点超过 8 像素的像素（占 {eval_v3['far_pixel_fraction']:.1%}）；FILL8 在其余像素用 REPROJ_NN，在 FAR 用 carrier。",
        "",
        "| EVAL-V3 AbsRel | ALL | NEAR | FAR |",
        "|---|---:|---:|---:|",
    ]
    for method in ("REPROJ_NN", "C0", "C1", "FILL8_C0", "FILL8_C1"):
        if method in absolute:
            cells = [absolute[method][r]["mean"] for r in ("ALL", "NEAR", "FAR")]
            lines.append(
                f"| {method} | "
                + " | ".join("—" if c is None else f"{c:.4f}" for c in cells)
                + " |"
            )
    lines += [
        "",
        "## 判定",
        "",
        f"- COMPLETION_GAIN = FAR AbsRel(C0) − FAR AbsRel(C1) = {stat(eval_v3['COMPLETION_GAIN'])}；COMPLETION_STATUS={statuses['COMPLETION_STATUS']}",
        f"- HYBRID_GAIN = AbsRel(REPROJ_NN) − AbsRel(FILL8 C1) = {stat(eval_v3['HYBRID_GAIN'])}；HYBRID_STATUS={statuses['HYBRID_STATUS']}",
        f"- DEV（次要）：C0 {fields['DEV_C0_QUERY_ABSREL']:.4f}，C1 {fields['DEV_C1_QUERY_ABSREL']:.4f}，WIDTH_GAIN_DEV={stat(static['surface_gain'])}，{fields['WIDTH_DEV_STATUS']}",
        f"- 完整性：V11 C0 与 V9 {contract['base_variant']} 的 DEV 曲线最大差 = {reproduction}",
        "",
        f"分支{branch}：{NEXT[branch]}",
        "",
        "说明：DEV 的 8 个场景在 V2–V10 中反复使用，本轮只用它选择 checkpoint 并给出描述性结果；EVAL-V3 是机制队列，不是资格验证；FRESH-V1/V2 未使用；受保护的 final holdout 未打开。",
    ]
    summary = (
        "RGBD COMPLETION V11 FINAL\n"
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
