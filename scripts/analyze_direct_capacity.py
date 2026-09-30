#!/usr/bin/env python3
"""Rebuild direct-state attribution JSON and six scientific figures from saved raw only."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mcss.mechanism_pilot.direct_capacity_statistics import analyze, group_key


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def figures(artifacts, output):
    output.mkdir(parents=True, exist_ok=True)
    result = artifacts["direct_state_results"]
    groups = {**result["frozen_carrier"], **result["context_only"], **result["diagnostic_oracle"]}
    comparisons = artifacts["capacity_analysis"]["comparisons"]
    scenes = result["scene_ids"]
    colors = {"RGBD": "#0072B2", "RGB_ONLY": "#D55E00", "QUERY_ORACLE": "#8A568D"}
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

    def save(fig, name):
        fig.savefig(output / f"{name}.png", dpi=180, bbox_inches="tight")
        fig.savefig(output / f"{name}.svg", bbox_inches="tight")
        plt.close(fig)

    def errorbar(ax, stats, x, **kwargs):
        # A percentile CI need not bracket the estimate on tiny fixtures.
        ax.plot(x, stats["mean"], **kwargs)
        ax.vlines(x, stats["ci95"][0], stats["ci95"][1], color=kwargs.get("color", "black"))

    fig, ax = plt.subplots(figsize=(7, 4))
    names = ["CARRIER", group_key("RGBD"), group_key("RGB_ONLY")]
    for x, key in enumerate(names):
        color = "#333333" if x == 0 else colors[["RGBD", "RGB_ONLY"][x - 1]]
        errorbar(ax, groups[key]["metrics"]["depth_absrel"], x, marker="o", color=color)
    ax.set_xticks(range(3), ["Frozen carrier", "RGB+D direct", "RGB-only direct"])
    ax.set_ylabel("Query AbsRel (lower is better)")
    ax.set_title("8³ / 64 samples / current bounds: equal scene means and 95% CI")
    save(fig, "01_carrier_vs_direct_query_absrel")

    fig, ax = plt.subplots(figsize=(6, 5))
    for track in ("RGBD", "RGB_ONLY"):
        group = groups[group_key(track)]
        x = group["context_metrics"]["depth_absrel"]["per_scene"]
        y = group["metrics"]["depth_absrel"]["per_scene"]
        ax.scatter(
            [x[s] for s in scenes],
            [y[s] for s in scenes],
            label=track,
            color=colors[track],
            alpha=0.8,
        )
    low, high = min(ax.get_xlim()[0], ax.get_ylim()[0]), max(ax.get_xlim()[1], ax.get_ylim()[1])
    ax.plot([low, high], [low, high], "k--", linewidth=0.8)
    ax.set(
        xlabel="Context AbsRel (post-seal evaluation)",
        ylabel="Sealed query AbsRel",
        title="Context vs query: every scene; 8³ / current bounds",
    )
    ax.legend()
    save(fig, "02_context_vs_query_error")

    fig, ax = plt.subplots(figsize=(7, 4))
    for offset, track in [(-0.12, "RGBD"), (0.12, "RGB_ONLY")]:
        means = []
        for i, grid in enumerate((8, 16, 32)):
            stats = groups[group_key(track, grid)]["metrics"]["depth_absrel"]
            errorbar(ax, stats, i + offset, marker="o", color=colors[track])
            means.append(stats["mean"])
        ax.plot(np.arange(3) + offset, means, color=colors[track], label=track)
    ax.axhline(
        groups["CARRIER"]["metrics"]["depth_absrel"]["mean"],
        color="gray",
        linestyle="--",
        label="Carrier 8³",
    )
    ax.set_xticks(range(3), ["8³", "16³", "32³"])
    ax.set(
        ylabel="Query AbsRel",
        xlabel="State resolution",
        title="Fixed 1,000-step budget; 64 samples / current bounds",
    )
    ax.legend()
    save(fig, "03_resolution_vs_query_absrel")

    fig, ax = plt.subplots(figsize=(9, max(5, len(scenes) * 0.32)))
    y = np.arange(len(scenes))
    for offset, track in [(-0.15, "RGBD"), (0.15, "RGB_ONLY")]:
        values = comparisons[f"capacity_gap_{track}_g8"]["per_scene"]
        ax.barh(
            y + offset, [values[s] for s in scenes], height=0.29, label=track, color=colors[track]
        )
    ax.set_yticks(y, scenes)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set(
        xlabel="Carrier − direct query AbsRel (positive = direct improvement)",
        title="8³ capacity gap: every exposed attribution scene",
    )
    ax.legend()
    save(fig, "04_per_scene_capacity_gap")

    fig, (left, right) = plt.subplots(
        1, 2, figsize=(10, 4), sharey=True, gridspec_kw={"width_ratios": [3, 1]}
    )
    for i, key in enumerate(names):
        errorbar(
            left,
            groups[key]["metrics"]["depth_absrel"],
            i,
            marker="o",
            color="black" if i == 0 else colors[["RGBD", "RGB_ONLY"][i - 1]],
        )
    left.set_xticks(range(3), ["Carrier", "Context RGB+D", "Context RGB-only"], rotation=15)
    left.set_title("Context-only evidence")
    for i, grid in enumerate((8, 16, 32)):
        errorbar(
            right,
            groups[group_key("QUERY_ORACLE", grid)]["metrics"]["depth_absrel"],
            i,
            marker="D",
            markerfacecolor="none",
            color=colors["QUERY_ORACLE"],
        )
    right.set_xticks(range(3), ["8³", "16³", "32³"])
    right.set_title("Privileged oracle")
    left.set_ylabel("Query AbsRel")
    fig.suptitle("Finite optimization diagnostic; oracle fits both scored queries jointly")
    save(fig, "05_three_capacity_levels_oracle_separate")

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for track in ("RGBD", "RGB_ONLY"):
        costs = [groups[group_key(track, g)].get("optimization_cost", {}) for g in (8, 16, 32)]
        axes[0].plot(
            [8, 16, 32], [4 * g**3 for g in (8, 16, 32)], "o-", color=colors[track], label=track
        )
        axes[1].plot(
            [8, 16, 32], [v.get("seconds_mean", np.nan) for v in costs], "o-", color=colors[track]
        )
        axes[2].plot(
            [8, 16, 32],
            [v.get("peak_cuda_allocated_bytes", np.nan) / 2**20 for v in costs],
            "o-",
            color=colors[track],
        )
    for ax, label in zip(
        axes, ("Optimized parameters", "Seconds / state", "Peak CUDA allocated MiB"), strict=True
    ):
        ax.set(xlabel="Grid edge", ylabel=label, xticks=[8, 16, 32])
    axes[0].set_yscale("log")
    axes[0].legend()
    fig.suptitle("State cost: fixed optimizer budget; measured runtime and allocator peak")
    save(fig, "06_resolution_vs_compute_memory")


def resolve_json(path):
    """Prefer plain JSON and otherwise use its sibling compressed archive."""
    path = Path(path)
    if path.is_file():
        return path
    compressed = Path(str(path) + ".gz")
    if compressed.is_file():
        return compressed
    raise FileNotFoundError(f"Missing raw input: {path} (or .gz)")


def load_json(path):
    actual = resolve_json(path)
    if actual.suffix == ".gz":
        with gzip.open(actual, "rt", encoding="utf-8") as stream:
            value = json.load(stream)
    else:
        value = json.loads(actual.read_text())
    return value, actual


def load_summaries(root, plan):
    """Use all individual summaries or one complete aggregate; never mix sources."""
    root = Path(root)
    states = plan["states"]
    expected = {spec["key"] for spec in states}
    if len(expected) != len(states):
        raise ValueError("Duplicate planned state keys")
    paths = {}
    for spec in states:
        p = Path(spec["optimization_path"])
        p = p if p.is_absolute() else root / p
        # Archived checkpoint directories need not exist to identify their summary path.
        if p.name not in ("summary.json", "summary.json.gz"):
            p = p / "summary.json"
        try:
            paths[spec["key"]] = resolve_json(p)
        except FileNotFoundError:
            break
    if len(paths) == len(states):
        summaries = {key: load_json(path)[0] for key, path in paths.items()}
        actual_paths = list(paths.values())
    else:
        summaries, actual = load_json(root / "raw/optimization_summaries.json")
        actual_paths = [actual]
    if not isinstance(summaries, dict) or set(summaries) != expected:
        raise ValueError("Optimization summaries must exactly match every planned state key")
    return summaries, actual_paths


def run(root, output=None):
    root = Path(root)
    output = Path(output) if output else root
    required = [
        "baseline_results.json",
        "raw/context_results.json",
        "raw/context_context_results.json",
        "raw/oracle_results.json",
        "state_plan.json",
        "scene_manifest.json",
        "preregistration.json",
    ]
    loaded = [load_json(root / name) for name in required]
    values, paths = [v[0] for v in loaded], [v[1] for v in loaded]
    baseline, query, context, oracle, plan, manifest, prereg = values
    summaries, summary_paths = load_summaries(root, plan)
    scene_ids = [s["scene_id"] for s in manifest["scenes"]]
    if set(scene_ids) & set(manifest["FINAL_HOLDOUT_PROHIBITED"]):
        raise ValueError("Protected holdout cannot enter analysis")
    artifacts = analyze(
        baseline,
        query,
        context,
        oracle,
        summaries,
        scene_ids,
        plan=plan,
        manifest=manifest,
        draws=prereg["statistics"]["draws"],
        seed=prereg["statistics"]["seed"],
    )
    output.mkdir(parents=True, exist_ok=True)
    for name, value in artifacts.items():
        save_json(output / f"{name}.json", value)
    figures(artifacts, output / "figures")
    hashes = {
        str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths + summary_paths
    }
    save_json(
        output / "statistics_reproduction.json",
        {
            "inputs_sha256": hashes,
            "raw_only": True,
            "media_or_checkpoint_loaded": False,
            "source_sha256": {
                str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in [
                    Path(__file__),
                    Path(__file__).resolve().parents[1]
                    / "src/mcss/mechanism_pilot/direct_capacity_statistics.py",
                ]
            },
        },
    )
    return artifacts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    artifacts = run(args.root, args.output)
    print(json.dumps(artifacts["capacity_analysis"]["classification"], indent=2))
