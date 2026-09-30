#!/usr/bin/env python3
"""Execute the frozen readout/reference re-analysis: hashes, DEV GT, rows, statistics."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from mcss.mechanism_pilot.readout_reanalysis import (
    EXPERIMENT,
    METRICS,
    READOUTS,
    SOURCES,
    VARIANTS,
    analyze_with_sensitivity,
    score_predictions,
    score_references,
    validate_matrix,
)
from mcss.mechanism_pilot.small_training import sha, write_json


def verify(lock):
    if lock["status"] != "FROZEN_BEFORE_DEV_GT_ANALYSIS" or lock["experiment"] != EXPERIMENT:
        raise PermissionError("Frozen re-analysis lock required")
    for group in ("source_sha256", "input_sha256", "train_reference_files_sha256"):
        for name, digest in lock[group].items():
            if sha(Path(name)) != digest:
                raise PermissionError(f"Frozen file changed: {name}")
    for p in lock["predictions"]:
        if sha(Path(p["path"])) != p["sha256"]:
            raise PermissionError(f"Sealed prediction changed: {p['path']}")


def dev_gt(manifest, access):
    """DEV primary-query GT depth, read once per frame after every hash is verified."""
    gt = {}
    for record in manifest["scenes"]:
        if record["split"] != "DEV":
            continue
        frames = {f["frame_id"]: f for f in record["frames"]}
        for fid in record["roles"]["primary_query"]:
            path = Path(frames[fid]["depth"])
            access.append(
                {
                    "scene_id": record["scene_id"],
                    "frame_id": fid,
                    "channel": "depth",
                    "purpose": "POST_SEAL_REANALYSIS_DEV_PRIMARY_QUERY",
                    "sha256": sha(path),
                }
            )
            data = np.load(path).squeeze().astype(np.float32)
            valid = np.isfinite(data) & (data > 0)
            gt[record["scene_id"], fid] = np.where(valid, data, 0).astype(np.float32)
    return gt


def table(result, readout, metric):
    lines = [
        "| carrier | mean | vs primary reference | label | vs anchor |",
        "|---|---:|---:|---|---|",
    ]
    for name, carrier in result["carriers"].items():
        entry = carrier[readout][metric]
        ref = entry[f"vs_{result['primary_reference'][metric]}"]
        static = result["static"][name][readout][metric]["label"]
        lines.append(
            f"| {name} | {entry['absolute_mean']:.4f} | {ref['mean']:+.4f} "
            f"[{ref['ci95'][0]:+.4f}, {ref['ci95'][1]:+.4f}] | {entry['reference_label']} | "
            f"{static} |"
        )
    return "\n".join(lines)


def readme(result, lock):
    refs = lock["references"]
    out = [
        f"# {EXPERIMENT}",
        "",
        f"`CARRIER_REFERENCE_STATUS={result['summary']['CARRIER_REFERENCE_STATUS']}`。",
        "",
        "这是对 V2-V4 已封存 DEV 预测的事后二次分析：不训练、不加载模型、不改变任何冻结状态。"
        "参考常数只由 TRAIN primary-query GT 拟合："
        f"REF_TRAIN_ABSREL_OPTIMAL={refs['REF_TRAIN_ABSREL_OPTIMAL']:.4f} m（AbsRel 最优常数），"
        f"REF_TRAIN_MEDIAN={refs['REF_TRAIN_MEDIAN']:.4f} m。"
        "差值为正表示 carrier 更好（AbsRel 更低 / delta1 更高）；"
        "CI 为 10000 次 scene-paired bootstrap。",
        "",
    ]
    sensitivity = result["sensitivity_excluding_no_hit_scenes"]
    out += [
        "Amendment 1（见 AMENDMENTS.md，在任何指标计算之前作出）：预测中位数不为正的视角"
        "（全部射线落在 bounds 之外、渲染全零的 NO_HIT 视角）在 MEDIAN_SCALED 下保持不缩放；"
        f"受影响视角数={result['integrity']['median_scale_undefined_views']}。"
        f"排除全 NO_HIT 场景 {sensitivity['excluded_no_hit_scenes']} 的次要敏感性分析："
        + (
            f"`CARRIER_REFERENCE_STATUS={sensitivity['summary']['CARRIER_REFERENCE_STATUS']}`，"
            f"ABOVE 计数={sensitivity['summary']['carriers_above_reference']}。"
            if "summary" in sensitivity
            else "无此类场景。"
        ),
        "",
    ]
    for readout in READOUTS:
        for metric in METRICS:
            out += [f"## {readout} · {metric}", "", table(result, readout, metric), ""]
    out += [
        "## 冻结主对比在不同 readout 下（仅描述）",
        "",
        "| source | contrast | readout | metric | gain | 95% CI | label |",
        "|---|---|---|---|---:|---:|---|",
    ]
    for source, entry in result["primary_contrasts"].items():
        for readout in READOUTS:
            for metric in METRICS:
                c = entry["by_readout"][readout][metric]
                out.append(
                    f"| {source} | {entry['name']} | {readout} | {metric} | "
                    f"{c['stats']['mean']:+.4f} | [{c['stats']['ci95'][0]:+.4f}, "
                    f"{c['stats']['ci95'][1]:+.4f}] | {c['label']} |"
                )
    out += [
        "",
        "## Opacity normalization 的作用（正数 = 归一化后更好）",
        "",
        "| carrier | AbsRel effect | label | delta1 effect | label |",
        "|---|---:|---|---:|---|",
    ]
    for name, effect in result["readout_effect"].items():
        a, d = effect["depth_absrel"], effect["depth_delta1"]
        out.append(
            f"| {name} | {a['stats']['mean']:+.4f} | {a['label']} | "
            f"{d['stats']['mean']:+.4f} | {d['label']} |"
        )
    return "\n".join(out) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=f"outputs/{EXPERIMENT}")
    root = Path(parser.parse_args().root)
    lock = json.loads((root / "preregistration.json").read_text())
    verify(lock)
    manifest = json.loads(Path(SOURCES["V4"]["root"], "scene_split.json").read_text())
    access = []
    gt = dev_gt(manifest, access)
    write_json(root / "audit" / "gt_access.json", access)
    rows, worst = score_predictions(lock["predictions"], gt)
    queries = {
        r["scene_id"]: list(r["roles"]["primary_query"])
        for r in manifest["scenes"]
        if r["split"] == "DEV"
    }
    reference_rows = score_references(lock["references"], gt, queries)
    seeds = lock["sources"]["V2"]["seeds"]
    validate_matrix(rows, reference_rows, seeds, sorted(queries), queries)
    write_json(root / "raw" / "rows.json", rows)
    write_json(root / "raw" / "reference_rows.json", reference_rows)
    result = analyze_with_sensitivity(rows, reference_rows)
    result["primary_reference"] = lock["analysis"]["primary_reference"]
    result["integrity"] = {
        "max_raw_absrel_deviation": worst["absrel"],
        "max_raw_delta1_pixel_deviation": worst["delta1_pixels"],
        "median_scale_undefined_views": worst["median_scale_undefined_views"],
        "rows": len(rows),
        "reference_rows": len(reference_rows),
        "variants": list(VARIANTS),
    }
    write_json(root / "results.json", result)
    (root / "README.md").write_text(readme(result, lock))
    docs = Path("docs/experiments") / EXPERIMENT
    docs.mkdir(parents=True, exist_ok=False)
    for name in ("preregistration.json", "results.json", "README.md", "AMENDMENTS.md"):
        if (root / name).exists():
            shutil.copyfile(root / name, docs / name)
    shutil.copyfile(root / "audit" / "gt_access.json", docs / "gt_access.json")
    print(
        json.dumps(
            {
                "status": result["summary"]["CARRIER_REFERENCE_STATUS"],
                "above": result["summary"]["carriers_above_reference"],
                "integrity": result["integrity"],
            },
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
