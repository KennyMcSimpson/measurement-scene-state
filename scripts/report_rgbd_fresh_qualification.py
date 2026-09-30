#!/usr/bin/env python3
"""Report the frozen FRESH-V1 qualification of the V5 RGB-D carriers (saved results only)."""
# ruff: noqa: E501 -- Chinese scientific report prose

import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np


def read(root, name, default=None):
    path = root / (name + ".json")
    if not path.exists() and default is not None:
        return default
    return json.loads(path.read_text())


def stat(value):
    if value is None or value.get("mean") is None:
        return "UNDEFINED"
    lo, hi = value["ci95"]
    return f"{value['mean']:+.4f}，95% CI [{lo:+.4f}, {hi:+.4f}]"


def per_scene(root):
    rows = read(root, "raw/fresh_readout_rows", [])
    refs = read(root, "raw/fresh_reference_rows", [])
    table = defaultdict(dict)
    for variant in ("C0", "C1", "C2"):
        values = defaultdict(list)
        for r in rows:
            if r["variant"] == variant and r["method"] == "direct" and r["readout"] == "RAW":
                values[r["scene_id"]].append(r["depth_absrel"])
        for scene, v in values.items():
            table[scene][variant] = float(np.mean(v))
    constant = defaultdict(list)
    for r in refs:
        if r["reference"] == "REF_TRAIN_ABSREL_OPTIMAL" and r["readout"] == "RAW":
            constant[r["scene_id"]].append(r["depth_absrel"])
    for scene, v in constant.items():
        table[scene]["constant"] = float(np.mean(v))
    return dict(table)


def run(root, docs=None):
    root = Path(root)
    results = read(root, "static_results")
    qualification = read(root, "qualification_results")
    controls = read(root, "wrong_scene_results")["contrasts"]
    obs = read(root, "observability_analysis")
    reference = read(root, "reference_results", {}) or None
    contract = read(root, "qualification_contract")
    v5 = Path(contract["v5_root"])
    v5_summary = (v5 / "terminal_summary.txt").read_text().splitlines()
    v5_fields = dict(line.split("=", 1) for line in v5_summary if "=" in line)
    if results["primary"] != "C1_SURFACE16":
        raise PermissionError("Frozen V2 estimator primary label expected (C1 = V5 primary)")
    groups = results["groups"]
    depth = results["surface_gain"]
    wrong, shuffle = controls["C1|wrong_scene"], controls["C1|spatial_shuffle"]
    depth_status = qualification["SURFACE_TRAINING_STATUS"]
    specificity = (
        "SUPPORTED" if wrong["ci95"][0] > 0 and shuffle["ci95"][0] > 0 else "NOT_ESTABLISHED"
    )
    static = qualification["STATIC_DEV_STATUS"]
    verdict = (
        "QUALIFIED"
        if depth_status == "SUPPORTED" and specificity == "SUPPORTED" and static == "SUPPORTED"
        else "NOT_QUALIFIED"
    )
    reference_status = (
        reference["summary"]["CARRIER_REFERENCE_STATUS"] if reference else "NOT_RECORDED"
    )
    c1_reference = reference["carriers"]["V5_C1"]["RAW"]["depth_absrel"] if reference else None
    sensitivity = reference.get("sensitivity_excluding_no_hit_scenes", {}) if reference else {}
    table = per_scene(root)
    fields = {
        "FRESH_SCENES": len(results["scene_ids"]),
        "FRESH_SCENE_IDS": results["scene_ids"],
        "DENSE_QUERY_ABSREL_FRESH": round(
            groups["C0|direct"]["query_metrics"]["depth_absrel"]["mean"], 6
        ),
        "DEPTH_QUERY_ABSREL_FRESH": round(
            groups["C1|direct"]["query_metrics"]["depth_absrel"]["mean"], 6
        ),
        "DEPTH_GAIN_FRESH": round(depth["mean"], 6),
        "DEPTH_GAIN_FRESH_CI": depth["ci95"],
        "DEPTH_GAIN_DEV_V5": v5_fields.get("DEPTH_GAIN"),
        "C1_WRONG_SCENE_DAMAGE_FRESH": round(wrong["mean"], 6),
        "C1_WRONG_SCENE_DAMAGE_FRESH_CI": wrong["ci95"],
        "C1_SHUFFLE_DAMAGE_FRESH": round(shuffle["mean"], 6),
        "C1_SHUFFLE_DAMAGE_FRESH_CI": shuffle["ci95"],
        "C1_FULL_CONTEXT_GAIN_FRESH": round(results["full_context_gain"]["C1"]["mean"], 6),
        "C1_FULL_CONTEXT_GAIN_FRESH_CI": results["full_context_gain"]["C1"]["ci95"],
        "DEPTH_STATUS_FRESH": depth_status,
        "SCENE_SPECIFICITY_STATUS_FRESH": specificity,
        "STATIC_STATUS_FRESH": static,
        "SEED_ROBUSTNESS_FRESH": qualification["SEED_ROBUSTNESS"],
        "CARRIER_REFERENCE_STATUS_FRESH": reference_status,
        "C1_REFERENCE_LABEL_FRESH": c1_reference["reference_label"] if c1_reference else None,
        "C1_REFERENCE_LABEL_FRESH_HIT_SCENES": sensitivity.get("carriers", {})
        .get("V5_C1", {})
        .get("RAW", {})
        .get("depth_absrel", {})
        .get("reference_label", "NO_NO_HIT_SCENE"),
        "FRESH_QUALIFICATION_STATUS": verdict,
        "CORE_A_STATIC_CARRIER_RGBD_TRACK": "QUALIFIED"
        if verdict == "QUALIFIED"
        else "NOT_QUALIFIED",
        "DYNAMIC_TTT_NEXT_STAGE_ALLOWED": verdict == "QUALIFIED",
        "FINAL_HOLDOUT_OPENED": False,
        "TRAINING_RUN": False,
        "DEVICE": "cpu",
    }
    lines = [
        "# EXP-3D-RGBD-FRESH-QUALIFICATION-V1",
        "",
        f"`FRESH_QUALIFICATION_STATUS={verdict}`。",
        "",
        "对 V5 已封存的 RGB-D carrier（每个 seed 各自的 DEV 选择，不重新训练）在独立队列 FRESH-V1 上做预注册资格验证。"
        f"FRESH-V1 共 {fields['FRESH_SCENES']} 个场景，均来自官方 val split、从未被观察、与 V2–V5 的 TRAIN/DEV volume 不重叠，每个 volume 一个场景。"
        "所有 fresh 状态封存之后才读取 query GT；统计沿用冻结的 V2 估计器与 V5 的 gate；受保护的官方 final holdout 未打开。",
        "",
        f"样本量说明：{fields['FRESH_SCENES']} 个场景低于数据锁的最低 6 个（修订 1 没有找到替代场景）。QUALIFIED 仍要求每一个冻结 gate；NOT_QUALIFIED 可能反映统计功效不足。",
        "",
        "| 项目 | FRESH-V1 | V5 DEV（参照） |",
        "|---|---|---|",
        f"| C0 只用 RGB，query AbsRel | {fields['DENSE_QUERY_ABSREL_FRESH']:.4f} | {v5_fields.get('DENSE_QUERY_ABSREL')} |",
        f"| C1 测得深度，query AbsRel | {fields['DEPTH_QUERY_ABSREL_FRESH']:.4f} | {v5_fields.get('DEPTH_QUERY_ABSREL')} |",
        f"| DEPTH_GAIN | {stat(depth)} | {v5_fields.get('DEPTH_GAIN')} |",
        f"| C1 wrong-scene 损伤 | {stat(wrong)} | {v5_fields.get('DEPTH_WRONG_SCENE_DAMAGE')} |",
        f"| C1 shuffle 损伤 | {stat(shuffle)} | {v5_fields.get('DEPTH_SHUFFLE_DAMAGE')} |",
        f"| 3 视角相对 anchor | {stat(results['full_context_gain']['C1'])} | {v5_fields.get('DEPTH_FULL_CONTEXT_GAIN')} |",
        f"| DEPTH_STATUS | {depth_status} | {v5_fields.get('DEPTH_STATUS')} |",
        f"| SCENE_SPECIFICITY_STATUS | {specificity} | {v5_fields.get('SCENE_SPECIFICITY_STATUS')} |",
        f"| STATIC_STATUS | {static} | {v5_fields.get('STATIC_DEV_STATUS')} |",
        "",
        "## 逐场景（RAW AbsRel，seed 与角色平均）",
        "",
        "| 场景 | 常数（TRAIN 拟合） | C0 | C1 | C2 |",
        "|---|---:|---:|---:|---:|",
    ]
    for scene in sorted(table):
        t = table[scene]
        lines.append(
            f"| {scene} | {t.get('constant', float('nan')):.3f} | {t.get('C0', float('nan')):.3f} | "
            f"{t.get('C1', float('nan')):.3f} | {t.get('C2', float('nan')):.3f} |"
        )
    lines += [
        "",
        "## 与无几何常数相比（描述性）",
        "",
        f"CARRIER_REFERENCE_STATUS={reference_status}；C1 RAW AbsRel 相对 AbsRel 最优常数："
        + (
            stat(c1_reference["vs_REF_TRAIN_ABSREL_OPTIMAL"])
            + f"（{c1_reference['reference_label']}）"
            if c1_reference
            else "未记录"
        )
        + f"；排除 NO_HIT 场景后标签：{fields['C1_REFERENCE_LABEL_FRESH_HIT_SCENES']}。",
        "",
        "## 区域诊断（描述性）",
        "",
        "C0−C1 条件 AbsRel："
        + "；".join(f"{r}={stat(obs['gains'][r])}" for r in ("OBS0", "OBS1", "OBS2PLUS"))
        + "。",
        "",
        "## 结论",
        "",
        (
            "三个冻结 gate 在独立队列上全部复现：Core A 的静态 carrier 在 RGB-D 轨道上通过资格验证。按项目规则，Core B（动态 TTT 写入）的下一阶段可以开启；官方 final holdout 仍保持关闭。"
            if verdict == "QUALIFIED"
            else "冻结 gate 未在独立队列上全部复现：Core A 在 RGB-D 轨道上尚未通过资格验证，Core B 保持关闭。未通过的 gate 见上表；由于只有 "
            f"{fields['FRESH_SCENES']} 个场景，需区分“效应不存在”与“功效不足”。"
        ),
    ]
    summary = (
        "RGBD FRESH QUALIFICATION V1\n"
        + "\n".join(
            f"{k}={str(v).lower() if isinstance(v, bool) else v}" for k, v in fields.items()
        )
        + "\n"
    )
    (root / "README.md").write_text("\n".join(lines) + "\n")
    (root / "STATUS.md").write_text("```text\n" + summary + "```\n")
    (root / "terminal_summary.txt").write_text(summary)
    (root / "fresh_qualification_results.json").write_text(
        json.dumps({**fields, "per_scene": table}, indent=2, ensure_ascii=False) + "\n"
    )
    if docs:
        docs = Path(docs)
        docs.mkdir(parents=True, exist_ok=True)
        for p in root.iterdir():
            if p.is_file() and p.suffix in (".json", ".md", ".txt"):
                shutil.copyfile(p, docs / p.name)
    print(summary)
    return fields


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--docs", type=Path)
    args = parser.parse_args()
    run(args.root, args.docs)
