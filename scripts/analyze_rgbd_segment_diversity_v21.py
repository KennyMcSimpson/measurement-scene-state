#!/usr/bin/env python3
"""EXP-3D-RGBD-SEGMENT-DIVERSITY-V21 analysis: P1, P2, the branch and the descriptive contrasts.

P1 DIVERSITY_GAIN = FAR AbsRel(G1 at T24000) - FAR AbsRel(G5 at T24000) over the pooled EVAL-V3 +
EVAL-V4 scenes, paired by scene, with V11's aggregation, scene bootstrap and the frozen V2 gate.
P2 = the INPAINT-V1 label of AbsRel(REPROJ_HARMONIC) - AbsRel(AHFILL8 of G5 at T24000), from the
frozen V17 analysis of the G5 rows. Branch A: P1 SUPPORTED and P2 ABOVE; B: exactly one; C: neither.
Descriptive: the same at T6000; each descriptive cohort; every arm's frozen V17 analysis; the share
of scenes where each arm beats harmonic inpainting on FAR pixels; G1 (GPU) against V17 L (CPU) at
T24000. Writes results.json and README.md into the V21 directory. Run from the repository root.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

V21 = Path("outputs/EXP-3D-RGBD-SEGMENT-DIVERSITY-V21")
ROOTS = {arm: V21 / arm / "EXP-3D-RGBD-LONG-TRAIN-V17" for arm in ("G1", "G5")}
V17 = Path("outputs/EXP-3D-RGBD-LONG-TRAIN-V17")
PRIMARY = ("EVAL_V3", "EVAL_V4")
DESCRIPTIVE = ("EVAL_FAR", "FRESH_V2")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def frozen_v17_evaluation():
    spec = importlib.util.spec_from_file_location(
        "v21_frozen_v17_eval", Path("scripts/evaluate_rgbd_long_train_eval.py")
    )
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def branch(diversity_status, ahfill8_label):
    first, second = diversity_status == "SUPPORTED", ahfill8_label == "ABOVE"
    return "A" if first and second else "B" if first or second else "C"


def analyze(rows, cpu_rows, evaluation):
    v11 = evaluation.host()

    def far(table, cohorts, label):
        return v11.scene_values([r for r in table if r["cohort"] in cohorts], label, "absrel_FAR")

    def contrast(a, b, gate=False):
        stats = v11.contrast(a, b)
        out = {"gain": stats}
        if gate and stats is not None:
            out["gate"], out["gate_checks"] = v11.gate(stats)
        return out

    p1 = contrast(far(rows["G1"], PRIMARY, "T24000"), far(rows["G5"], PRIMARY, "T24000"), True)
    blocks = {
        arm: evaluation.block([r for r in table if r["cohort"] in PRIMARY])
        for arm, table in rows.items()
    }
    p2 = blocks["G5"]["contrasts"].get("HARMONIC_MINUS_AHFILL8_T24000", {})
    statuses = {
        "DIVERSITY_STATUS": p1.get("gate", "UNDEFINED"),
        "AHFILL8_LABEL": p2.get("label", "UNDEFINED"),
    }
    descriptive = {
        "DIVERSITY_GAIN_T6000": contrast(
            far(rows["G1"], PRIMARY, "T6000"), far(rows["G5"], PRIMARY, "T6000")
        ),
        "cohorts": {
            name: {
                step: contrast(far(rows["G1"], (name,), step), far(rows["G5"], (name,), step))
                for step in ("T6000", "T24000")
            }
            for name in DESCRIPTIVE
        },
        "harmonic_minus_carrier_far_T24000": {
            arm: block["contrasts"].get("HARMONIC_MINUS_T24000_FAR", {}).get("gain")
            for arm, block in blocks.items()
        },
        "DEVICE_CHECK_V17_CPU_MINUS_G1_GPU_T24000": contrast(
            far(cpu_rows, PRIMARY, "T24000"), far(rows["G1"], PRIMARY, "T24000")
        ),
    }
    return {
        "statuses": statuses,
        "INTERPRETATION_BRANCH": branch(statuses["DIVERSITY_STATUS"], statuses["AHFILL8_LABEL"]),
        "P1_DIVERSITY_GAIN_T24000": p1,
        "P2_HARMONIC_MINUS_AHFILL8_G5_T24000": p2,
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


def readme(result, segments):
    p1, p2 = result["P1_DIVERSITY_GAIN_T24000"], result["P2_HARMONIC_MINUS_AHFILL8_G5_T24000"]
    d = result["descriptive"]
    lines = [
        "# EXP-3D-RGBD-SEGMENT-DIVERSITY-V21",
        "",
        "同一批 TRAIN72 场景里更多样的训练内容（相机轨迹后段的片段）能否让补全更好、更一致。"
        "计划见 [PLAN_BEFORE_V21_DATA.md](PLAN_BEFORE_V21_DATA.md)。"
        "两组都在 GPU（4 号卡，关闭 TF32）"
        f"上用冻结的 V17 代码训练 24000 步；G5 多了 {segments} 个 SEG-V1 片段。GPU 不能逐位复现，"
        "所有比较都在本轮内部进行。",
        "",
        f"- **P1（DIVERSITY_GAIN = FAR(G1) − FAR(G5)，第 24000 步）：** {stat(p1.get('gain'))}，"
        f"{p1.get('gate', '—')}",
        f"- **P2（调和插值 − AHFILL8(G5，第 24000 步)）：** {stat(p2.get('gain'))}，"
        f"{p2.get('label', '—')}",
        f"- **分支 {result['INTERPRETATION_BRANCH']}**",
        f"- **第 6000 步的同样对比（描述性）：** {stat(d['DIVERSITY_GAIN_T6000']['gain'])}",
        "",
        "## 各组的补全曲线（主队列 FAR AbsRel，三个 seed 平均）",
        "",
        "| 步数 | G1 | G5 |",
        "|---:|---:|---:|",
    ]
    curves = {arm: block["curve"] for arm, block in result["arms"].items()}
    for step in sorted(set(curves["G1"]) | set(curves["G5"]), key=int):
        g1 = curves["G1"].get(step, {}).get("FAR")
        g5 = curves["G5"].get(step, {}).get("FAR")
        lines.append(f"| {step} | {mean(g1)} | {mean(g5)} |")
    lines += [
        "",
        "## 描述性",
        "",
        "| 队列 | 第 6000 步 G1 − G5 | 第 24000 步 G1 − G5 |",
        "|---|---|---|",
    ]
    for name, steps in d["cohorts"].items():
        lines.append(
            f"| {name} | {stat(steps['T6000']['gain'])} | {stat(steps['T24000']['gain'])} |"
        )
    lines += [
        "",
        "- 调和插值 − carrier（FAR，第 24000 步）："
        + "；".join(
            f"{arm} {stat(v)}" for arm, v in d["harmonic_minus_carrier_far_T24000"].items()
        ),
        f"- 设备检查，V17 L（CPU）− G1（GPU），第 24000 步 FAR："
        f"{stat(d['DEVICE_CHECK_V17_CPU_MINUS_G1_GPU_T24000']['gain'])}",
        "",
        "## 限制",
        "",
        "- EVAL-V3/V4 已被多个实验使用，FRESH-V2 是第八次使用，本轮结论只作机制对比；"
        "受保护的 final holdout 未打开。",
        "- 片段来自同一批训练场景；结论不推广到新的场景或新的数据集。",
    ]
    return "\n".join(lines) + "\n"


def main():
    freeze = read(V21 / "freeze.json")
    rows, inputs = {}, {}
    for arm, root in ROOTS.items():
        if sha(root / "preregistration.json") != freeze["arms"][arm]["preregistration_sha256"]:
            raise PermissionError(f"{arm} lock changed after the V21 freeze")
        path = root / "raw/eval/query_rows.json"
        rows[arm], inputs[str(path)] = read(path), sha(path)
    cpu_path = V17 / "raw/eval/query_rows.json"
    inputs[str(cpu_path)] = sha(cpu_path)
    result = analyze(rows, read(cpu_path), frozen_v17_evaluation())
    result["input_sha256"] = inputs
    result["source_sha256"] = {str(Path(__file__).resolve()): sha(Path(__file__))}
    (V21 / "results.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    (V21 / "README.md").write_text(readme(result, freeze["seg_v1"]["segments"]))
    print("V21_BRANCH", result["INTERPRETATION_BRANCH"], json.dumps(result["statuses"]), flush=True)


if __name__ == "__main__":
    main()
