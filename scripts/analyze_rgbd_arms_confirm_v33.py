#!/usr/bin/env python3
"""EXP-3D-RGBD-ARMS-CONFIRM-V33 analysis: P1, P2, the branch and the descriptive contrasts.

P1 CONTEXT_GAIN_CONFIRM = FAR AbsRel(G1) - FAR AbsRel(U) over the pooled EVAL-V5 + TRAIN-EXT scenes
(step 24000, three seeds), paired by scene, with V11's aggregation, scene bootstrap and the frozen
V2 gate; P2 = the INPAINT-V1 label of AbsRel(REPROJ_HARMONIC) - AbsRel(AHFILL8 of U) on all pixels
of the same scenes. Branch A: P1 SUPPORTED and P2 ABOVE; B: exactly one; C: neither. Descriptive:
each cohort, Replica, arm P. Writes results.json and README.md. Run from the repository root.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

V33 = Path("outputs/EXP-3D-RGBD-ARMS-CONFIRM-V33")
V32_RESULTS = Path("outputs/EXP-3D-RGBD-UNET-CONTEXT-V32/results.json")
HYPERSIM = ("EVAL_V5", "TRAIN_EXT")
ARMS = ("G1", "P", "U")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def evaluator():
    spec = importlib.util.spec_from_file_location(
        "v33_arms_eval", Path("scripts/evaluate_rgbd_carrier_arms.py")
    )
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def run_condition(v32_results):
    """The preregistered condition: V32's P1 mean (FAR(G1) - FAR(U), T24000) is above zero."""
    gain = v32_results["P1_CONTEXT_GAIN_T24000"]["gain"]
    return gain is not None and gain["mean"] > 0


def branch(context_status, ahfill8_label):
    first, second = context_status == "SUPPORTED", ahfill8_label == "ABOVE"
    return "A" if first and second else "B" if first or second else "C"


def analyze(rows, block):
    def subset(names):
        return [r for r in rows if r["cohort"] in names]

    pooled = block(subset(HYPERSIM), list(ARMS), "G1")
    p1 = pooled["contrasts"]["G1_MINUS_U_FAR"]
    p2 = pooled["contrasts"]["HARMONIC_MINUS_AHFILL8_U"]
    statuses = {
        "CONTEXT_CONFIRM_STATUS": p1.get("gate", "UNDEFINED"),
        "AHFILL8_U_LABEL": p2.get("label", "UNDEFINED"),
    }
    names = sorted({r["cohort"] for r in rows})
    return {
        "statuses": statuses,
        "INTERPRETATION_BRANCH": branch(*statuses.values()),
        "P1_CONTEXT_GAIN_CONFIRM": p1,
        "P2_HARMONIC_MINUS_AHFILL8_U": p2,
        "hypersim_pooled": pooled,
        "cohorts": {name: block(subset((name,)), list(ARMS), "G1") for name in names},
    }


def stat(value):
    if not value:
        return "无数据"
    lo, hi = value["ci95"]
    return (
        f"{value['mean']:+.4f} [{lo:+.4f}, {hi:+.4f}]，场景 "
        f"{value['positive']}/{value['tied']}/{value['negative']}"
    )


def item(block, name):
    entry = block["contrasts"].get(name, {})
    tail = entry.get("gate") or entry.get("label") or "—"
    return f"{stat(entry.get('gain'))}，{tail}" if entry else "—"


def absolute(block, method, region):
    summary = block["absolute_absrel"].get(method, {}).get(region)
    return f"{summary['mean']:.4f}" if summary else "—"


def readme(result):
    p1, p2 = result["P1_CONTEXT_GAIN_CONFIRM"], result["P2_HARMONIC_MINUS_AHFILL8_U"]
    lines = [
        "# EXP-3D-RGBD-ARMS-CONFIRM-V33",
        "",
        "在三个新队列上确认三维上下文分支（U，V32）相对 G1（V21）的效果，"
        "预训练特征（P，V31）只作描述。"
        "计划见 [PLAN_BEFORE_V33.md](PLAN_BEFORE_V33.md)。各组都取第 24000 步、3 个 seed，"
        "评价在 CPU 上（冻结的 V17 EVAL 流程推广到多组："
        "`scripts/evaluate_rgbd_carrier_arms.py`）。",
        "",
        f"- **P1（FAR(G1) − FAR(U)，EVAL-V5 ∪ TRAIN-EXT）：** {stat(p1.get('gain'))}，"
        f"{p1.get('gate', '—')}",
        f"- **P2（调和插值 − AHFILL8(U)，全部像素）：** {stat(p2.get('gain'))}，"
        f"{p2.get('label', '—')}",
        f"- **分支 {result['INTERPRETATION_BRANCH']}**",
        "",
        "## 各队列（第 24000 步，三个 seed 平均）",
        "",
        "| 队列 | 场景 | FAR：调和插值 / G1 / U / P | G1 − U（FAR） | G1 − P（FAR） "
        "| 调和插值 − AHFILL8(U) | 调和插值 − U（FAR） |",
        "|---|---:|---|---|---|---|---|",
    ]
    blocks = {"EVAL-V5 ∪ TRAIN-EXT": result["hypersim_pooled"], **result["cohorts"]}
    for name, block in blocks.items():
        far = " / ".join(absolute(block, m, "FAR") for m in ("REPROJ_HARMONIC", "G1", "U", "P"))
        lines.append(
            f"| {name} | {block['n_scenes']} | {far} | {item(block, 'G1_MINUS_U_FAR')} | "
            f"{item(block, 'G1_MINUS_P_FAR')} | {item(block, 'HARMONIC_MINUS_AHFILL8_U')} | "
            f"{item(block, 'HARMONIC_MINUS_U_FAR')} |"
        )
    lines += [
        "",
        "## 限制",
        "",
        "- 这三个队列在 Core B（V6、V7、V9、V27、V28）中使用过，"
        "静态补全的 GPU 试点也用过 TRAIN-EXT；"
        "G1、U、P 都是第一次在这些队列上评价。受保护的 final holdout 未打开。",
        "- Replica 只有 8 个场景，只作描述。",
    ]
    return "\n".join(lines) + "\n"


def main():
    freeze = read(V33 / "preregistration.json")
    if freeze["status"] != "FROZEN_BEFORE_V33_EVALUATION":
        raise PermissionError("V33 is not frozen")
    for name, digest in freeze["source_sha256"].items():
        if sha(name) != digest:
            raise PermissionError(f"Frozen source changed: {name}")
    if sha(V33 / "evaluation_contract.json") != freeze["evaluation_contract_sha256"]:
        raise PermissionError("The V33 contract changed")
    if not run_condition(read(V32_RESULTS)):
        raise PermissionError("The preregistered run condition does not hold")
    path = V33 / "raw/eval/query_rows.json"
    result = analyze(read(path), evaluator().block)
    result["input_sha256"] = {str(path): sha(path), str(V32_RESULTS): sha(V32_RESULTS)}
    result["source_sha256"] = {str(Path(__file__).resolve()): sha(Path(__file__))}
    (V33 / "results.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    (V33 / "README.md").write_text(readme(result))
    print("V33_BRANCH", result["INTERPRETATION_BRANCH"], json.dumps(result["statuses"]), flush=True)


if __name__ == "__main__":
    main()
