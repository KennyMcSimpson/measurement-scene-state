#!/usr/bin/env python3
"""Report the saved V14 results: pooled EVAL-V3/V4 primary gates, DEV secondary (no model/media)."""
# ruff: noqa: E501 -- Chinese scientific report prose

import argparse
import json
from pathlib import Path

V11 = Path("outputs/EXP-3D-RGBD-COMPLETION-V11")
METHODS = (
    "CONSTANT",
    "REPROJ_NN",
    "REPROJ_HARMONIC",
    "C0",
    "C1",
    "HFILL8_C0",
    "HFILL8_C1",
    "AHFILL8_C0",
    "AHFILL8_C1",
)
NEXT = {
    "A": "训练场景从 72 个增加到 {n} 个，改善了未观测区域的补全；用新数据训练的 carrier 在可部署组合中也超过了经典几何补洞。数据规模是有效的杠杆，这支持在 GPU 上继续扩大数据与模型。",
    "B": "两个条件只成立一个，需要分开解读：是补全本身随数据改善、但还不足以胜过经典补洞，还是组合相对经典补洞的优势并非来自更多的数据。",
    "C": "在 CPU 可及的规模内（训练场景 72→{n}），增加数据没有稳定改善补全，也没有让组合胜过经典几何补洞。",
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
    return f"{value['mean']:+.4f}，95% CI [{lo:+.4f}, {hi:+.4f}]，场景 {value['positive']}/{value['tied']}/{value['negative']}"


def c0_reproduces_v11(root, contract):
    """Largest |DEV query AbsRel| difference between V14 C0 and V11 C1, per step."""
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
    result = read(root, "eval_results")
    static = read(root, "static_results")
    qualification = read(root, "qualification_results")
    tests, integrity = (read(root, n, {"status": "PENDING"}) for n in ("tests", "integrity"))
    primary = result["primary"]
    absolute = primary["absolute_absrel"]
    contrasts = primary["contrasts"]
    statuses = result["statuses"]
    branch = result["INTERPRETATION_BRANCH"]
    groups = static["groups"]
    reproduction = c0_reproduces_v11(root, contract)
    n_train = len(contract["train_sets"]["TRAIN_EXT"])
    fields = {
        "TRAIN72_SCENES": len(contract["train_sets"]["TRAIN72"]),
        "TRAIN_EXT_SCENES": n_train,
        "EVAL_SCENES": primary["n_scenes"],
        "FAR_PIXEL_FRACTION": round(primary["far_pixel_fraction"], 4),
        **{
            f"EVAL_{method}_{region}": round(absolute[method][region]["mean"], 6)
            for method in METHODS
            for region in ("ALL", "NEAR", "FAR")
            if method in absolute and (absolute[method][region] or {}).get("mean") is not None
        },
        **{
            f"{name}{suffix}": (
                (round(gain["mean"], 6) if suffix == "" else gain["ci95"]) if gain else "UNDEFINED"
            )
            for name, gain in (
                (name, contrasts[name]["gain"])
                for name in ("DATA_GAIN", "HARMONIC_HYBRID_GAIN", "HFILL8_DATA_GAIN")
            )
            for suffix in ("", "_CI")
        },
        **statuses,
        **{
            f"{cohort}_{name}": (
                round(block["contrasts"][name]["gain"]["mean"], 6)
                if block["contrasts"][name]["gain"]
                else "UNDEFINED"
            )
            for cohort, block in result["cohorts"].items()
            for name in ("DATA_GAIN", "HARMONIC_HYBRID_GAIN")
        },
        "C0_MINUS_V11_C1_EVAL_MAX_ABS": result["consistency"]["c0_minus_v11_c1_max_abs"],
        "C0_V11_QUERIES_COMPARED": result["consistency"]["queries_compared"],
        "DEV_C0_QUERY_ABSREL": round(
            groups["C0|direct"]["query_metrics"]["depth_absrel"]["mean"], 6
        ),
        "DEV_C1_QUERY_ABSREL": round(
            groups["C1|direct"]["query_metrics"]["depth_absrel"]["mean"], 6
        ),
        "DATA_GAIN_DEV": round(static["surface_gain"]["mean"], 6),
        "DATA_GAIN_DEV_CI": static["surface_gain"]["ci95"],
        "DATA_DEV_STATUS": qualification["SURFACE_TRAINING_STATUS"],
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
        "# EXP-3D-RGBD-DATA-SCALE-V14",
        "",
        f"单因素机制检验：V11 C1 的配方不变，只把训练场景从 TRAIN72 增加到 TRAIN72 + TRAIN-EXT（共 {n_train} 个）。计划在读取任何新训练数据之前写定；C0 按同一配方在 TRAIN72 上重新训练，必须与 V11 C1 逐位相同。主评价在 EVAL-V3 与 EVAL-V4 的合并队列（{primary['n_scenes']} 个场景）上进行；EVAL-FAR 的远处 query 只作描述。所有预测在读取 query 深度之前封存。全程 CPU。",
        "",
        f"FAR 为离上下文深度重投影命中点超过 8 像素的像素（占 {primary['far_pixel_fraction']:.1%}）；HFILL8 在其余像素用调和插值补洞，在 FAR 用 carrier；AHFILL8 另把不透明度 < 0.5 的 FAR 像素也交给调和插值（V13）。",
        "",
        "| EVAL-V3 + EVAL-V4 AbsRel | ALL | NEAR | FAR |",
        "|---|---:|---:|---:|",
    ]
    for method in METHODS:
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
        f"- DATA_GAIN = FAR AbsRel(C0 TRAIN72) − FAR AbsRel(C1 TRAIN72+EXT) = {stat(contrasts['DATA_GAIN']['gain'])}；{statuses['DATA_STATUS']}",
        f"- HARMONIC_HYBRID_GAIN = AbsRel(REPROJ_HARMONIC) − AbsRel(HFILL8 C1) = {stat(contrasts['HARMONIC_HYBRID_GAIN']['gain'])}；{statuses['HARMONIC_HYBRID_LABEL']}",
        f"- HFILL8_DATA_GAIN = AbsRel(HFILL8 C0) − AbsRel(HFILL8 C1) = {stat(contrasts['HFILL8_DATA_GAIN']['gain'])}；{statuses['HFILL8_DATA_STATUS']}",
        f"- 描述性：AHFILL8_DATA_GAIN = {stat(contrasts['AHFILL8_DATA_GAIN']['gain'])}；调和插值 − AHFILL8 C1 = {stat(contrasts['HARMONIC_MINUS_AHFILL8_C1']['gain'])}",
        f"- DEV（次要）：C0 {fields['DEV_C0_QUERY_ABSREL']:.4f}，C1 {fields['DEV_C1_QUERY_ABSREL']:.4f}，DATA_GAIN_DEV={stat(static['surface_gain'])}，{fields['DATA_DEV_STATUS']}",
        f"- 完整性：C0 与 V11 C1 的 DEV 曲线最大差 = {reproduction}；C0 与 V11 C1 封存预测的最大差 = {fields['C0_MINUS_V11_C1_EVAL_MAX_ABS']}（{fields['C0_V11_QUERIES_COMPARED']} 个 query）",
        "",
        "## 各队列",
        "",
        "| 队列 | 场景 | DATA_GAIN | 调和插值 − HFILL8 C1 | 调和插值 − AHFILL8 C1 |",
        "|---|---:|---|---|---|",
    ]
    for cohort, block in result["cohorts"].items():
        cells = [
            stat(block["contrasts"][name]["gain"])
            + f"，{block['contrasts'][name].get('label', 'UNDEFINED')}"
            for name in ("DATA_GAIN", "HARMONIC_HYBRID_GAIN", "HARMONIC_MINUS_AHFILL8_C1")
        ]
        lines.append(f"| {cohort} | {block['n_scenes']} | " + " | ".join(cells) + " |")
    lines += [
        "",
        f"分支 {branch}：{NEXT[branch].format(n=n_train)}",
        "",
        "说明：DEV 的 8 个场景在 V2–V12 中反复使用，本轮只用它选择 checkpoint 并给出描述性结果；EVAL-V3/V4/FAR 是机制队列，不是资格验证；FRESH-V1/V2 未使用；受保护的 final holdout 未打开。",
    ]
    summary = (
        "RGBD DATA SCALE V14 FINAL\n"
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
