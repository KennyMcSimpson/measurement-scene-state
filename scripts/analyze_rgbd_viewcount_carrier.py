#!/usr/bin/env python3
"""V9 raw-only statistics with the frozen V2 estimator and eight figures."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mcss.mechanism_pilot.geometry_carrier_statistics import analyze, analyze_regions


def load(root, name, provenance):
    path = root / name
    if not path.is_file():
        path = Path(str(path) + ".gz")
    data = path.read_bytes()
    provenance[str(path.resolve())] = hashlib.sha256(data).hexdigest()
    return json.loads(gzip.decompress(data) if path.suffix == ".gz" else data)


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def figures(artifacts, regions, curves, cost, directory):
    directory.mkdir(parents=True, exist_ok=True)
    groups = artifacts["static_results"]["groups"]
    colors = ("#0072B2", "#D55E00")
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "font.size": 10})

    def finish(fig, name):
        for ext in ("png", "svg"):
            fig.savefig(directory / f"{name}.{ext}", bbox_inches="tight", dpi=180)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    shown = [v for v in ("C0", "C1", "C2") if f"{v}|direct" in groups]
    values = [groups[f"{v}|direct"]["query_metrics"]["depth_absrel"] for v in shown]
    palette = (*colors, "#009E73")
    ax.bar(range(len(shown)), [v["mean"] for v in values], color=palette[: len(shown)])
    for i, v in enumerate(values):
        ax.plot([i, i], v["ci95"], color="black")
    labels = {
        "C0": "C0 fixed 3 views",
        "C1": "C1 variable 3-7 views (primary)",
        "C2": "C2 fixed 7 views (secondary)",
    }
    ax.set(
        xticks=list(range(len(shown))),
        xticklabels=[labels[v] for v in shown],
        ylabel="DEV overall query AbsRel",
    )
    finish(fig, "01_c0_c1_query_absrel")

    gain = artifacts["static_results"]["surface_gain"]
    ids = artifacts["static_results"]["scene_ids"]
    fig, ax = plt.subplots(figsize=(8, max(4, len(ids) * 0.3)))
    ax.barh(ids, [gain["per_scene"][s] for s in ids], color=colors[1])
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("C0 − C1 overall AbsRel; all DEV scenes retained")
    finish(fig, "02_per_scene_surface_gain")

    fig, ax = plt.subplots(figsize=(7, 4))
    for j, method in enumerate(("direct", "anchor")):
        ax.bar(
            np.arange(2) + (j - 0.5) * 0.3,
            [
                groups[f"{v}|{method}"]["query_metrics"]["depth_absrel"]["mean"]
                for v in ("C0", "C1")
            ],
            0.3,
            label=method,
        )
    ax.set(xticks=[0, 1], xticklabels=["C0", "C1"], ylabel="Query AbsRel")
    ax.legend()
    ax.set_title("Full context versus anchor, same role-derived bounds")
    finish(fig, "03_full_context_vs_anchor")

    fig, ax = plt.subplots(figsize=(8, 4))
    for j, variant in enumerate(("C0", "C1")):
        vals = [
            regions["groups"][variant][r]["conditional_absrel"]["mean"]
            for r in ("OBS0", "OBS1", "OBS2PLUS")
        ]
        ax.bar(
            np.arange(3) + (j - 0.5) * 0.3,
            [np.nan if v is None else v for v in vals],
            0.3,
            color=colors[j],
            label=variant,
        )
    ax.set(
        xticks=range(3), xticklabels=["OBS0", "OBS1", "OBS2PLUS"], ylabel="Conditional scene AbsRel"
    )
    ax.legend()
    ax.set_title("Diagnostic only; empty scene counts reported in JSON")
    finish(fig, "04_observability_signature")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, phase, metric in zip(axes, ("training", "dev"), ("loss", "depth_absrel"), strict=True):
        rows = curves.get(phase, [])
        for j, variant in enumerate(("C0", "C1")):
            for seed in sorted({r["seed"] for r in rows if r["variant"] == variant}):
                subset = sorted(
                    (r for r in rows if r["variant"] == variant and r["seed"] == seed),
                    key=lambda r: r["step"],
                )
                ax.plot(
                    [r["step"] for r in subset],
                    [r[metric] for r in subset],
                    color=colors[j],
                    label=f"{variant} seed{seed}",
                )
        ax.set(xlabel="Frozen training steps", ylabel=metric, title=phase)
        if rows:
            ax.legend(fontsize=7)
        else:
            ax.text(0.5, 0.5, "NOT_RECORDED", ha="center", transform=ax.transAxes)
    finish(fig, "05_training_and_dev_curves")

    fig, ax = plt.subplots(figsize=(6, 4))
    for j, variant in enumerate(("C0", "C1")):
        cell = groups[f"{variant}|direct"]
        x, y = (
            cell["context_metrics"]["depth_absrel"]["mean"],
            cell["query_metrics"]["depth_absrel"]["mean"],
        )
        ax.scatter([x], [y], color=colors[j], label=variant, s=70)
        ax.annotate(variant, (x, y))
    ax.set(
        xlabel="Context AbsRel",
        ylabel="Query AbsRel",
        title="Same selected checkpoint; depth for evaluation only",
    )
    ax.legend()
    finish(fig, "06_context_query_gap")

    fig, ax = plt.subplots(figsize=(7, 4))
    for j, variant in enumerate(("C0", "C1")):
        stats = [
            artifacts["wrong_scene_results"]["contrasts"][f"{variant}|{m}"]
            for m in ("wrong_scene", "spatial_shuffle")
        ]
        xx = np.arange(2) + (j - 0.5) * 0.3
        ax.bar(xx, [s["mean"] for s in stats], 0.3, color=colors[j], label=variant)
        for x, value in zip(xx, stats, strict=True):
            ax.plot([x, x], value["ci95"], color="black")
    ax.set(
        xticks=[0, 1],
        xticklabels=["Wrong scene", "Spatial shuffle"],
        ylabel="Control − direct AbsRel",
    )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.legend()
    finish(fig, "07_wrong_scene_shuffle_damage")

    fig, ax = plt.subplots(figsize=(7, 4))
    any_cost = False
    for j, variant in enumerate(("C0", "C1")):
        seconds = cost.get("variants", {}).get(variant, {}).get("training_seconds")
        if seconds is not None:
            any_cost = True
            error = groups[f"{variant}|direct"]["query_metrics"]["depth_absrel"]["mean"]
            ax.scatter([seconds], [error], color=colors[j], label=variant, s=70)
            ax.annotate(variant, (seconds, error))
    if any_cost:
        ax.legend()
    else:
        ax.text(0.5, 0.5, "COST NOT_RECORDED", ha="center", transform=ax.transAxes)
    ax.set(xlabel="Training worker seconds (not phase wall time)", ylabel="Query AbsRel")
    finish(fig, "08_quality_vs_compute")


def run(root, output=None):
    root, output = Path(root), Path(output) if output else Path(root)
    provenance = {}
    manifest = load(root, "scene_split.json", provenance)
    contract = load(root, "training_contract.json", provenance)
    args = [
        load(root, f"raw/dev_{name}_results.json", provenance)
        for name in ("query", "context", "matched")
    ]
    audit = load(root, "audit/dev_state_use.json", provenance)
    static = load(root, "raw/dev_static_matched_results.json", provenance)
    artifacts = analyze(
        *args,
        manifest,
        seeds=contract["seeds"],
        state_use_audit=audit,
        static_matched_rows=static,
    )
    regions = analyze_regions(
        load(root, "raw/dev_region_results.json", provenance), manifest, seeds=contract["seeds"]
    )
    artifacts["observability_analysis"] = regions
    curves = load(root, "training_curves.json", provenance)
    cost = load(root, "cost_analysis.json", provenance)
    output.mkdir(parents=True, exist_ok=True)
    for name, value in artifacts.items():
        save(output / f"{name}.json", value)
    figures(artifacts, regions, curves, cost, output / "figures")
    save(
        output / "statistics_reproduction.json",
        {
            "status": "PASS",
            "raw_only": True,
            "media_or_checkpoint_loaded": False,
            "input_sha256": provenance,
            "source_sha256": {
                str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in (
                    Path(__file__),
                    Path(__file__).resolve().parents[1]
                    / "src/mcss/mechanism_pilot/geometry_carrier_statistics.py",
                )
            },
        },
    )
    return artifacts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    run(args.root, args.output)
