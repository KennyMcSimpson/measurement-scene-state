#!/usr/bin/env python3
"""EXP-3D-RGBD-FAR-FOCUS-V26 analysis: P1, P2, the branch and the descriptive contrasts.

P1 FAR_FOCUS_GAIN = FAR AbsRel(G1 at T24000) - FAR AbsRel(GF at T24000) over the pooled EVAL-V3 +
EVAL-V4 scenes, paired by scene, with V11's aggregation, scene bootstrap and the frozen V2 gate.
P2 = the INPAINT-V1 label of AbsRel(REPROJ_HARMONIC) - AbsRel(AHFILL8 of GF at T24000), from the
frozen V17 analysis of the GF rows. Branch A: P1 SUPPORTED and P2 ABOVE; B: exactly one; C: neither.
Descriptive: the same at T6000; the NEAR cost and ALL pixels; EVAL-FAR and FRESH-V2; each arm's
harmonic-minus-carrier FAR contrast and harmonic-minus-hybrid contrasts; each arm's frozen V17
analysis. Writes results.json and README.md into the V26 directory. Run from the repository root.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

V26 = Path("outputs/EXP-3D-RGBD-FAR-FOCUS-V26")
ARMS = ("G1", "GF")
PRIMARY = ("EVAL_V3", "EVAL_V4")
DESCRIPTIVE = ("EVAL_FAR", "FRESH_V2")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def frozen_v17_evaluation():
    spec = importlib.util.spec_from_file_location(
        "v26_frozen_v17_eval", Path("scripts/evaluate_rgbd_long_train_eval.py")
    )
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def branch(far_focus_status, ahfill8_label):
    first, second = far_focus_status == "SUPPORTED", ahfill8_label == "ABOVE"
    return "A" if first and second else "B" if first or second else "C"


def analyze(rows, evaluation):
    v11 = evaluation.host()

    def values(table, cohorts, label, region):
        subset = [r for r in table if r["cohort"] in cohorts]
        return v11.scene_values(subset, label, f"absrel_{region}")

    def contrast(arm_a, arm_b, cohorts, label, region, gate=False):
        stats = v11.contrast(
            values(rows[arm_a], cohorts, label, region), values(rows[arm_b], cohorts, label, region)
        )
        out = {"gain": stats}
        if gate and stats is not None:
            out["gate"], out["gate_checks"] = v11.gate(stats)
        return out

    p1 = contrast("G1", "GF", PRIMARY, "T24000", "FAR", True)
    blocks = {
        arm: evaluation.block([r for r in table if r["cohort"] in PRIMARY])
        for arm, table in rows.items()
    }
    p2 = blocks["GF"]["contrasts"].get("HARMONIC_MINUS_AHFILL8_T24000", {})
    statuses = {
        "FAR_FOCUS_STATUS": p1.get("gate", "UNDEFINED"),
        "AHFILL8_LABEL": p2.get("label", "UNDEFINED"),
    }

    def arm_contrast(name):
        return {arm: block["contrasts"].get(name, {}) for arm, block in blocks.items()}

    descriptive = {
        "FAR_FOCUS_GAIN_T6000": contrast("G1", "GF", PRIMARY, "T6000", "FAR"),
        "NEAR_COST_T24000": contrast("G1", "GF", PRIMARY, "T24000", "NEAR"),
        "ALL_T24000": contrast("G1", "GF", PRIMARY, "T24000", "ALL"),
        "cohorts": {
            name: {step: contrast("G1", "GF", (name,), step, "FAR") for step in ("T6000", "T24000")}
            for name in DESCRIPTIVE
        },
        "harmonic_minus_carrier_far_T24000": arm_contrast("HARMONIC_MINUS_T24000_FAR"),
        "harmonic_minus_ahfill8_T24000": arm_contrast("HARMONIC_MINUS_AHFILL8_T24000"),
        "harmonic_minus_hfill8_T24000": arm_contrast("HARMONIC_MINUS_HFILL8_T24000"),
        "eval_far_harmonic_minus_ahfill8_T24000": {
            arm: evaluation.block([r for r in table if r["cohort"] == "EVAL_FAR"])["contrasts"].get(
                "HARMONIC_MINUS_AHFILL8_T24000", {}
            )
            for arm, table in rows.items()
        },
    }
    return {
        "statuses": statuses,
        "INTERPRETATION_BRANCH": branch(statuses["FAR_FOCUS_STATUS"], statuses["AHFILL8_LABEL"]),
        "P1_FAR_FOCUS_GAIN_T24000": p1,
        "P2_HARMONIC_MINUS_AHFILL8_GF_T24000": p2,
        "descriptive": descriptive,
        "arms": blocks,
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


def labelled(item):
    return f"{stat(item.get('gain'))}，{item.get('label', '—')}"


def readme(result, far_fraction):
    p1, p2 = result["P1_FAR_FOCUS_GAIN_T24000"], result["P2_HARMONIC_MINUS_AHFILL8_GF_T24000"]
    d = result["descriptive"]
    lines = [
        "# EXP-3D-RGBD-FAR-FOCUS-V26",
        "",
        "训练时把光线集中在测量覆盖不到的像素上，carrier 在这些像素上的补全能否稳定变好。"
        "计划见 [PLAN_BEFORE_V26.md](PLAN_BEFORE_V26.md)。"
        "两组都在 GPU（4 号卡，关闭 TF32）上用 V17 L 配方训练 24000 步："
        "G1（V21 已训练好的组）均匀采样；"
        f"GF 每个角色 {round(100 * far_fraction)}% 的光线取自 FAR 像素，其余均匀。"
        "GPU 不能逐位复现，所有比较都在 GPU 训练的组之间进行。",
        "",
        f"- **P1（FAR_FOCUS_GAIN = FAR(G1) − FAR(GF)，第 24000 步）：** {stat(p1.get('gain'))}，"
        f"{p1.get('gate', '—')}",
        f"- **P2（调和插值 − AHFILL8(GF，第 24000 步)）：** {stat(p2.get('gain'))}，"
        f"{p2.get('label', '—')}",
        f"- **分支 {result['INTERPRETATION_BRANCH']}**",
        f"- **第 6000 步的同样对比（描述性）：** {stat(d['FAR_FOCUS_GAIN_T6000']['gain'])}",
        f"- **NEAR 的代价（NEAR(G1) − NEAR(GF)）：** {stat(d['NEAR_COST_T24000']['gain'])}；"
        f"全部像素 {stat(d['ALL_T24000']['gain'])}",
        "",
        "## 各组的补全曲线（主队列 FAR AbsRel，三个 seed 平均）",
        "",
        "| 步数 | G1 | GF |",
        "|---:|---:|---:|",
    ]
    curves = {arm: block["curve"] for arm, block in result["arms"].items()}
    for step in sorted(set(curves["G1"]) | set(curves["GF"]), key=int):
        g1 = curves["G1"].get(step, {}).get("FAR")
        gf = curves["GF"].get(step, {}).get("FAR")
        lines.append(f"| {step} | {mean(g1)} | {mean(gf)} |")
    lines += [
        "",
        "## 与调和插值的比较（主队列，第 24000 步）",
        "",
        "| 组 | 调和插值 − carrier（FAR） | 调和插值 − HFILL8 | 调和插值 − AHFILL8 "
        "| EVAL-FAR：调和插值 − AHFILL8 |",
        "|---|---|---|---|---|",
    ]
    for arm in ARMS:
        lines.append(
            f"| {arm} | {stat(d['harmonic_minus_carrier_far_T24000'][arm].get('gain'))} | "
            f"{labelled(d['harmonic_minus_hfill8_T24000'][arm])} | "
            f"{labelled(d['harmonic_minus_ahfill8_T24000'][arm])} | "
            f"{labelled(d['eval_far_harmonic_minus_ahfill8_T24000'][arm])} |"
        )
    lines += [
        "",
        "| 队列（描述性） | 第 6000 步 G1 − GF | 第 24000 步 G1 − GF |",
        "|---|---|---|",
    ]
    for name, steps in d["cohorts"].items():
        lines.append(
            f"| {name} | {stat(steps['T6000']['gain'])} | {stat(steps['T24000']['gain'])} |"
        )
    lines += [
        "",
        "## 限制",
        "",
        "- EVAL-V3/V4 与 EVAL-FAR 已被多个实验使用，FRESH-V2 是第九次使用，本轮结论只作机制对比；"
        "受保护的 final holdout 未打开。",
        "- G1 与 GF 是不同时间的两次 GPU 训练；GPU 训练本身就不能逐位复现。",
    ]
    return "\n".join(lines) + "\n"


def main():
    freeze = read(V26 / "freeze.json")
    rows, inputs = {}, {}
    for arm in ARMS:
        root = Path(freeze["arms"][arm]["root"])
        if sha(root / "preregistration.json") != freeze["arms"][arm]["preregistration_sha256"]:
            raise PermissionError(f"{arm} lock changed after the V26 freeze")
        path = root / "raw/eval/query_rows.json"
        rows[arm], inputs[str(path)] = read(path), sha(path)
    result = analyze(rows, frozen_v17_evaluation())
    result["input_sha256"] = inputs
    result["source_sha256"] = {str(Path(__file__).resolve()): sha(Path(__file__))}
    (V26 / "results.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    (V26 / "README.md").write_text(readme(result, freeze["far_fraction"]))
    print("V26_BRANCH", result["INTERPRETATION_BRANCH"], json.dumps(result["statuses"]), flush=True)


if __name__ == "__main__":
    main()
