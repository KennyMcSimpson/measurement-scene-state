#!/usr/bin/env python3
"""Addendum 1 to EXP-3D-RGBD-GEOMETRIC-BASELINES-V1: more sealed carriers vs the sealed baselines.

Descriptive only. Reuses the sealed REPROJ_NN baseline files (hash-checked) and the V7 masks;
scores V7/V8 sealed carrier predictions (RAW and the GT-free opacity-normalized readout) with
the frozen depth metric. No model is run, no baseline is recomputed.
"""
# ruff: noqa: E501 -- Chinese scientific report prose

import argparse
import hashlib
import importlib.util
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from mcss.mechanism_pilot.direct_capacity_statistics import describe, paired
from mcss.mechanism_pilot.readout_reanalysis import depth_metrics

ROOT = Path("outputs/EXP-3D-RGBD-GEOMETRIC-BASELINES-V1")
V7 = Path("outputs/EXP-3D-RGBD-DEPTH-BOUNDS-V7")
V8 = Path("outputs/EXP-3D-RGBD-RESOLUTION-V8")
QUALIFICATION = Path("outputs/EXP-3D-RGBD-FRESH-QUALIFICATION-V2")
CARRIERS = {
    "DEV": {
        "V7_C1": (V7 / "raw/selected_dev", "C1", "RAW"),
        "V7_C1_NORM": (V7 / "raw/selected_dev", "C1", "NORM"),
        "V8_C0": (V8 / "raw/selected_dev", "C0", "RAW"),
        "V8_C1": (V8 / "raw/selected_dev", "C1", "RAW"),
        "V8_C1_NORM": (V8 / "raw/selected_dev", "C1", "NORM"),
        "V8_C2": (V8 / "raw/selected_dev", "C2", "RAW"),
    },
    "FRESH_V2": {
        "V7_C1": (QUALIFICATION / "raw/selected_fresh", "C1", "RAW"),
        "V7_C1_NORM": (QUALIFICATION / "raw/selected_fresh", "C1", "NORM"),
    },
}
MANIFESTS = {
    "DEV": (V7 / "scene_split.json", "DEV"),
    "FRESH_V2": (QUALIFICATION / "scene_split.json", "FRESH_QUALIFICATION"),
}
MASKS = {"DEV": V7 / "raw/observability", "FRESH_V2": QUALIFICATION / "raw/observability"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def main_module():
    spec = importlib.util.spec_from_file_location("geo", "scripts/run_rgbd_geometric_baselines.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def readout(path, kind):
    arrays = np.load(path)
    depth = arrays["depth"].astype(np.float64)
    if kind == "NORM":
        return depth / np.maximum(arrays["opacity"].astype(np.float64), 1e-6)
    return depth


def run(root=ROOT):
    root = Path(root)
    if (root / "addendum1_results.json").exists():
        raise FileExistsError("Addendum 1 runs once")
    if not (root / "ADDENDUM1.md").is_file():
        raise PermissionError("Addendum text must be written first")
    geo = main_module()
    geo.verify(root.resolve())
    seal = read(root / "raw/baseline_seal.json")
    seeds = read(V7 / "training_contract.json")["seeds"]
    rows = []
    for cohort, carriers in CARRIERS.items():
        manifest_path, split = MANIFESTS[cohort]
        manifest = read(manifest_path)
        records = {r["scene_id"]: r for r in manifest["scenes"] if r["split"] == split}
        sealed_rows = {}
        for name, (directory, variant, _) in carriers.items():
            for seed in seeds:
                run_dir = directory / f"{variant}_{seed}"
                for r in read(run_dir / "query_results.json"):
                    if r["method"] == "direct":
                        path = run_dir / r["prediction_path"]
                        if sha(path) != r["prediction_file_sha256"]:
                            raise PermissionError(f"Sealed prediction changed: {path}")
                        sealed_rows[name, seed, r["scene_id"], r["role"], r["query_id"]] = path
        for sid, record in records.items():
            frames = {f["frame_id"]: f for f in record["frames"]}
            for role in ("A", "B"):
                for qid in record["roles"]["primary_query"]:
                    entry = seal[f"{cohort}/{sid}/{role}/{qid}"]
                    if sha(entry["path"]) != entry["sha256"]:
                        raise AssertionError("Sealed baseline changed")
                    baseline = dict(np.load(entry["path"]))
                    gt = np.load(frames[qid]["depth"]).squeeze().astype(np.float64)
                    obs = np.load(MASKS[cohort] / f"{sid}_{role}_{qid}.npz")["obs_class"]
                    common = {"cohort": cohort, "scene_id": sid, "role": role, "query_id": qid}
                    rows.append(
                        {
                            **common,
                            "seed": 0,
                            "method": "REPROJ_NN",
                            **depth_metrics(baseline["REPROJ_NN"], gt),
                            "regions": geo.region_rows(baseline["REPROJ_NN"], gt, obs),
                        }
                    )
                    for name, (_, _, kind) in carriers.items():
                        for seed in seeds:
                            prediction = readout(sealed_rows[name, seed, sid, role, qid], kind)
                            hybrid = np.where(baseline["hit"], baseline["reproj"], prediction)
                            for method, value in ((name, prediction), (f"HYBRID_{name}", hybrid)):
                                rows.append(
                                    {
                                        **common,
                                        "seed": seed,
                                        "method": method,
                                        **depth_metrics(value, gt),
                                        "regions": geo.region_rows(value, gt, obs),
                                    }
                                )
    result = {"cohorts": {}, "descriptive_only": True}
    for cohort, carriers in CARRIERS.items():
        methods = ("REPROJ_NN", *carriers, *(f"HYBRID_{n}" for n in carriers))
        values = {m: geo.aggregate(rows, m, cohort) for m in methods}
        block = {"absolute_depth_absrel": {}, "vs_reproj_nn": {}, "regions": {}, "delta1": {}}
        for method in methods:
            block["absolute_depth_absrel"][method] = describe(
                values[method], draws=geo.DRAWS, seed=geo.SEED
            )
            block["delta1"][method] = float(
                np.mean(list(geo.aggregate(rows, method, cohort, "depth_delta1").values()))
            )
            regions = {}
            for region in ("OBS0", "OBS1", "OBS2PLUS"):
                grouped = defaultdict(list)
                for r in rows:
                    if (
                        r["method"] == method
                        and r["cohort"] == cohort
                        and r["regions"][region] is not None
                    ):
                        grouped[r["scene_id"]].append(r["regions"][region])
                regions[region] = float(np.mean([np.mean(v) for v in grouped.values()]))
            block["regions"][method] = regions
            if method != "REPROJ_NN":
                stats = paired(values["REPROJ_NN"], values[method], draws=geo.DRAWS, seed=geo.SEED)
                block["vs_reproj_nn"][method] = {"gain": stats, "label": geo.label(stats)}
        result["cohorts"][cohort] = block
    dev = result["cohorts"]["DEV"]["absolute_depth_absrel"]
    result["integrity"] = {
        "v7_c1_reproduces_main_dev": abs(
            dev["V7_C1"]["mean"]
            - read(root / "baseline_results.json")["cohorts"]["DEV"]["absolute_depth_absrel"][
                "V7_C1"
            ]["mean"]
        ),
        "v8_c0_equals_v7_c1": abs(dev["V8_C0"]["mean"] - dev["V7_C1"]["mean"]),
    }
    (root / "addendum1_results.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    lines = ["# 补充分析 1：分辨率与读出（描述性）", ""]
    for cohort, block in result["cohorts"].items():
        lines += [
            f"## {cohort}",
            "",
            "| 方法 | AbsRel | delta1 | OBS0 / OBS1 / OBS2PLUS | 相对 REPROJ_NN（基线−方法） | 标签 |",
            "|---|---|---|---|---|---|",
        ]
        for method, v in block["absolute_depth_absrel"].items():
            reg = block["regions"][method]
            comparison = block["vs_reproj_nn"].get(method)
            gain = (
                f"{comparison['gain']['mean']:+.4f} [{comparison['gain']['ci95'][0]:+.4f}, {comparison['gain']['ci95'][1]:+.4f}]"
                if comparison
                else "—"
            )
            lines.append(
                f"| {method} | {v['mean']:.4f} | {block['delta1'][method]:.3f} | {reg['OBS0']:.3f} / {reg['OBS1']:.3f} / {reg['OBS2PLUS']:.3f} | {gain} | {comparison['label'] if comparison else '—'} |"
            )
        lines.append("")
    lines.append(
        f"完整性：V7_C1 与主分析的差 {result['integrity']['v7_c1_reproduces_main_dev']:.2e}；V8_C0 与 V7_C1 的差 {result['integrity']['v8_c0_equals_v7_c1']:.2e}。"
    )
    (root / "ADDENDUM1_RESULTS.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    run(parser.parse_args().root)
