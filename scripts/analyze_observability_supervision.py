#!/usr/bin/env python3
"""Rebuild observability/supervision attribution from saved raw JSON only."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mcss.mechanism_pilot.observability_supervision_statistics import PARTITION, VARIANTS, analyze


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def load_json(path, provenance):
    path = Path(path)
    if not path.is_file():
        path = Path(str(path) + ".gz")
    if not path.is_file():
        raise FileNotFoundError(f"Required complete raw artifact missing: {path}")
    raw = path.read_bytes()
    provenance[str(path.resolve())] = hashlib.sha256(raw).hexdigest()
    return json.loads(gzip.decompress(raw) if path.suffix == ".gz" else raw)


def figures(artifacts, directory):
    directory.mkdir(parents=True, exist_ok=True)
    obs = artifacts["observability_analysis"]
    supervision = artifacts["supervision_analysis"]
    regions, scenes = obs["regions"], obs["scene_ids"]
    groups = supervision["groups"]
    colors = {"S0": "#444444", "S1": "#E69F00", "S2": "#009E73", "S3": "#0072B2"}
    region_colors = {"OBS0": "#D55E00", "OBS1": "#E69F00", "OBS2PLUS": "#009E73"}
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

    def save(fig, name):
        for suffix in ("png", "svg"):
            fig.savefig(directory / f"{name}.{suffix}", dpi=180, bbox_inches="tight")
        plt.close(fig)

    def point(ax, x, stats, color, marker="o", label=None):
        if stats["mean"] is None:
            ax.annotate("empty", (x, 0), rotation=90, fontsize=8)
            return
        ax.plot(x, stats["mean"], marker=marker, color=color, label=label)
        if stats["ci95"] is not None:
            ax.vlines(x, *stats["ci95"], color=color)

    fig, ax = plt.subplots(figsize=(max(8, len(scenes) * 0.5), 4))
    bottom = np.zeros(len(scenes) + 1)
    for region in PARTITION:
        stats = regions["S0"][region]["pixel_fraction"]
        values = np.array([stats["per_scene"][s] for s in scenes] + [stats["mean"]])
        ax.bar(
            np.arange(len(values)), values, bottom=bottom, label=region, color=region_colors[region]
        )
        bottom += values
    ax.set_xticks(
        range(len(scenes) + 1), scenes + ["Scene mean"], rotation=60, ha="right", fontsize=8
    )
    ax.set(
        ylabel="Fraction of all GT-valid query pixels",
        ylim=(0, 1),
        title="Context observability: every exposed scene; query → A/B → scene weighting",
    )
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
    save(fig, "01_query_observability_fraction")

    fig, ax = plt.subplots(figsize=(8, 4))
    for offset, variant, color in [(-0.12, "S0", colors["S0"]), (0.12, "ORACLE", "#8A568D")]:
        for i, region in enumerate(PARTITION):
            stats = regions[variant][region]["conditional_absrel"]
            point(
                ax,
                i + offset,
                stats,
                color,
                marker="D" if variant == "ORACLE" else "o",
                label=("Privileged query oracle" if variant == "ORACLE" else "Context S0")
                if i == 0
                else None,
            )
    labels = [
        f"{r}\n{regions['S0'][r]['conditional_absrel']['n_covered_scenes']}/{len(scenes)} scenes"
        for r in PARTITION
    ]
    ax.set_xticks(range(3), labels)
    overall = [regions[v]["OVERALL"]["conditional_absrel"]["mean"] for v in ("S0", "ORACLE")]
    ax.set(
        ylabel="Conditional region AbsRel",
        title=f"Overall S0={overall[0]:.4f}, oracle={overall[1]:.4f}; overall remains primary",
    )
    ax.legend()
    save(fig, "02_context_vs_oracle_by_region")

    fig, ax = plt.subplots(figsize=(7, 4))
    for i, variant in enumerate(VARIANTS):
        point(
            ax,
            i,
            groups[f"{variant}|direct"]["query_metrics"]["depth_absrel"],
            colors[variant],
            marker="*" if variant == "S3" else "o",
        )
    ax.set_xticks(
        range(4),
        ["S0 render only", "S1 free-space", "S2 surface", "S3 combined\npredeclared primary"],
    )
    ax.set(
        ylabel="Overall query AbsRel",
        title="All scenes and all GT-valid query pixels; 95% scene bootstrap CI",
    )
    save(fig, "03_overall_query_absrel_supervision")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for offset, variant in [(-0.2, "S1"), (0, "S2"), (0.2, "S3")]:
        for i, region in enumerate(PARTITION):
            stats = supervision["regional_gains"][region][variant]
            point(
                axes[0],
                i + offset,
                stats["conditional_gain"],
                colors[variant],
                label=variant if i == 0 else None,
            )
            point(
                axes[1],
                i + offset,
                stats["additive_gain_contribution"],
                colors[variant],
                label=variant if i == 0 else None,
            )
    for ax in axes:
        ax.set_xticks(range(3), PARTITION)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.legend()
    axes[0].set(
        ylabel="Conditional S0 − variant AbsRel",
        title="Covered scenes only; empty scenes retained as null",
    )
    axes[1].set(
        ylabel="Additive overall-error gain contribution",
        title="All scenes, including zero for empty regions",
    )
    save(fig, "04_supervision_gain_by_observability")

    fig, ax = plt.subplots(figsize=(8, max(5, 0.32 * len(scenes))))
    values = supervision["overall_absrel_comparisons"]["S0_minus_S3"]["per_scene"]
    ax.barh(scenes, [values[s] for s in scenes], color=colors["S3"])
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set(
        xlabel="S0 − S3 query AbsRel (positive: combined supervision improves)",
        title="Every scene; S3 was specified before evaluation",
    )
    save(fig, "05_per_scene_primary_gain")

    fig, ax = plt.subplots(figsize=(6, 5))
    for variant in VARIANTS:
        group = groups[f"{variant}|direct"]
        x = group["context_metrics"]["depth_absrel"]["per_scene"]
        y = group["query_metrics"]["depth_absrel"]["per_scene"]
        ax.scatter(
            [x[s] for s in scenes],
            [y[s] for s in scenes],
            color=colors[variant],
            label=variant,
            alpha=0.8,
        )
    lo, hi = min(ax.get_xlim()[0], ax.get_ylim()[0]), max(ax.get_xlim()[1], ax.get_ylim()[1])
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=0.8)
    ax.set(
        xlabel="Context AbsRel",
        ylabel="Sealed query AbsRel",
        title="Context fit and query performance for all four variants",
    )
    ax.legend()
    save(fig, "06_context_vs_query_all_variants")

    fig, ax = plt.subplots(figsize=(7, 4))
    closure = artifacts["gap_closure_analysis"]
    for i, variant in enumerate(VARIANTS):
        stats = closure["base_gap"] if variant == "S0" else closure["variants"][variant]["new_gap"]
        point(ax, i, stats, colors[variant], marker="*" if variant == "S3" else "o")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(range(4), ["S0", "S1", "S2", "S3 (primary)"])
    ax.set(
        ylabel="Context-state − query-oracle AbsRel",
        title="Remaining diagnostic gap; oracle is not a theoretical optimum",
    )
    save(fig, "07_oracle_gap_before_after_supervision")


def run(root, output=None):
    root = Path(root)
    output = Path(output) if output else root
    provenance = {}
    integrity = load_json(root / "audit/observability_integrity.json", provenance)
    if integrity.get("status") != "PASS":
        raise PermissionError("Post-seal observability audit must pass before formal analysis")
    query = load_json(root / "raw/query_results.json", provenance)
    query += load_json(root / "raw/reference_results.json", provenance)
    context = load_json(root / "raw/context_results.json", provenance)
    region = load_json(root / "raw/region_results.json", provenance)
    geometry = load_json(root / "raw/observability_summary.json", provenance)
    aggregate = root / "raw/optimization_summaries.json"
    if aggregate.is_file() or Path(str(aggregate) + ".gz").is_file():
        trajectories = load_json(aggregate, provenance)
    else:
        plan = load_json(root / "state_plan.json", provenance)
        trajectories = {}
        for spec in plan["states"]:
            path = Path(spec["optimization_path"])
            path = path if path.is_absolute() else root / path
            summary = load_json(path / "summary.json", provenance)
            if spec["key"] in trajectories:
                raise ValueError("Duplicate planned state key")
            trajectories[spec["key"]] = {"spec": spec, "summary": summary}
    manifest = load_json(root / "scene_manifest.json", provenance)
    rules = load_json(root / "decision_rules.json", provenance)
    artifacts = analyze(query, context, region, geometry, trajectories, manifest, rules)
    walls = root / "raw/phase_wall_times.json"
    artifacts["cost_analysis"]["phase_wall_times"] = (
        load_json(walls, provenance)
        if walls.is_file() or Path(str(walls) + ".gz").is_file()
        else {"status": "NOT_RECORDED"}
    )
    output.mkdir(parents=True, exist_ok=True)
    for name, value in artifacts.items():
        save_json(output / f"{name}.json", value)
    figures(artifacts, output / "figures")
    save_json(
        output / "statistics_reproduction.json",
        {
            "raw_only": True,
            "media_or_state_checkpoint_loaded": False,
            "input_sha256": provenance,
            "source_sha256": {
                str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in [
                    Path(__file__),
                    Path(__file__).resolve().parents[1]
                    / "src/mcss/mechanism_pilot/observability_supervision_statistics.py",
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
    result = run(args.root, args.output)
    print(json.dumps(result["decision_summary"], indent=2))
