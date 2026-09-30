#!/usr/bin/env python3
"""Pure saved-raw reproduction for 16-grid optimization and GT-free bounds attribution."""

import argparse
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mcss.mechanism_pilot.optimization_bounds_statistics import CURRENT, GTFREE, analyze, group_key


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def resolve(path):
    path = Path(path)
    if path.is_file():
        return path
    compressed = Path(str(path) + ".gz")
    if compressed.is_file():
        return compressed
    raise FileNotFoundError(f"Required saved raw missing: {path} (or .gz)")


def read_text(path, provenance):
    actual = resolve(path)
    provenance[str(actual.resolve())] = hashlib.sha256(actual.read_bytes()).hexdigest()
    return (
        gzip.decompress(actual.read_bytes()).decode()
        if actual.suffix == ".gz"
        else actual.read_text()
    )


def load_json(path, provenance):
    return json.loads(read_text(path, provenance))


def load_trajectories(root, plan, provenance, include_secondary=False):
    chains = [s for s in plan["chains"] if include_secondary or s["phase"] != "secondary"]
    expected = {s["key"] for s in chains}
    if len(expected) != len(chains):
        raise ValueError("Duplicate trajectory key")
    aggregate = root / "raw/trajectory_summaries.json"
    try:
        aggregate_path = resolve(aggregate)
    except FileNotFoundError:
        aggregate_path = None
    if aggregate_path is not None:
        values = load_json(aggregate, provenance)
        if not isinstance(values, dict):
            raise ValueError("Trajectory archive must be keyed by chain identity")
        # A completed archive may additionally contain the separately labelled secondary phase.
        permitted = {s["key"] for s in plan["chains"]}
        if not expected <= set(values) or not set(values) <= permitted:
            raise ValueError("Trajectory archive keys disagree with state plan")
        return {key: values[key] for key in sorted(expected)}
    values = {}
    for spec in chains:
        path = Path(spec["optimization_path"])
        path = path if path.is_absolute() else root / path
        trace_text = read_text(path / "optimization_trace.csv", provenance)
        trace = [
            {k: (float(v) if k != "kind" and v not in ("", "None") else v) for k, v in row.items()}
            for row in csv.DictReader(io.StringIO(trace_text))
        ]
        summary = load_json(path / "trajectory_summary.json", provenance)
        checks = load_json(path / "full_objective_checkpoints.json", provenance)
        if isinstance(checks, dict):
            checks = checks["full_objective_checkpoints"]
        budget_summaries = {}
        for budget in spec["budgets"]:
            budget_summaries[str(budget)] = {
                selection: load_json(
                    path / f"budget_{budget}" / selection / "summary.json", provenance
                )
                for selection in ("FIXED_BUDGET", "CONTEXT_SELECTED")
            }
        values[spec["key"]] = {
            "spec": spec,
            "summary": summary,
            "checkpoints": checks,
            "trace": trace,
            "budget_summaries": budget_summaries,
        }
    return values


def figures(artifacts, directory):
    directory.mkdir(parents=True, exist_ok=True)
    results = artifacts["optimization_results"]
    groups = results["context_only"]
    oracle = artifacts["query_oracle_diagnostic"]["groups"]
    budgets, scenes = results["budgets"], results["scene_ids"]
    colors = {CURRENT: "#D55E00", GTFREE: "#0072B2"}
    labels = {CURRENT: "Current bounds", GTFREE: "Frozen GT-free bounds"}
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

    def save(fig, name):
        for suffix in ("png", "svg"):
            fig.savefig(directory / f"{name}.{suffix}", dpi=180, bbox_inches="tight")
        plt.close(fig)

    def series(ax, bounds, stats, *, label=None, style="-", marker="o"):
        values = [s["mean"] for s in stats]
        ax.plot(
            budgets,
            values,
            style,
            marker=marker,
            color=colors[bounds],
            label=label or labels[bounds],
        )
        ax.fill_between(
            budgets,
            [s["ci95"][0] for s in stats],
            [s["ci95"][1] for s in stats],
            color=colors[bounds],
            alpha=0.13,
        )
        ax.set_xscale("log")
        ax.set_xticks(budgets, [f"{b // 1000}k" for b in budgets])
        ax.set_xlabel("Frozen optimization budget")

    for field, title, filename in [
        ("context", "Context AbsRel", "01_steps_vs_context_absrel"),
        ("query", "Sealed query AbsRel", "02_steps_vs_query_absrel"),
        ("gap", "Query − context AbsRel", "03_steps_vs_generalization_gap"),
    ]:
        fig, ax = plt.subplots(figsize=(7, 4))
        for bounds in (CURRENT, GTFREE):
            cells = [groups[group_key("RGBD", bounds, b)] for b in budgets]
            stats = [
                c["generalization_gap"]
                if field == "gap"
                else c["context_metrics" if field == "context" else "metrics"]["depth_absrel"]
                for c in cells
            ]
            series(ax, bounds, stats)
        ax.set_ylabel(title)
        ax.set_title("16³ RGB+D / fixed-budget; scene mean and 95% paired bootstrap CI")
        ax.legend()
        save(fig, filename)

    fig, ax = plt.subplots(figsize=(7, 4))
    for selection, style in [("FIXED_BUDGET", "-"), ("CONTEXT_SELECTED", "--")]:
        stats = [
            artifacts["bounds_analysis"][f"RGBD|{selection}"]["gains_by_budget"][str(b)]
            for b in budgets
        ]
        series(ax, GTFREE, stats, label=selection, style=style)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Current − GT-free query AbsRel (positive: GT-free improves)")
    ax.set_title("Bounds comparison across all frozen budgets")
    ax.legend()
    save(fig, "04_bounds_gain_across_budgets")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, bounds in zip(axes, (CURRENT, GTFREE), strict=True):
        series(
            ax,
            bounds,
            [groups[group_key("RGBD", bounds, b)]["metrics"]["depth_absrel"] for b in budgets],
            label="Context RGB+D",
        )
        series(
            ax,
            bounds,
            [
                oracle[group_key("QUERY_ORACLE", bounds, b)]["metrics"]["depth_absrel"]
                for b in budgets
            ],
            label="Query-supervised oracle (privileged)",
            style="--",
            marker="D",
        )
        ax.set_title(labels[bounds])
        ax.set_ylabel("Scored query AbsRel")
        ax.legend(fontsize=8)
    fig.suptitle("Oracle fits both scored queries jointly; not a deployable ranking")
    save(fig, "05_context_vs_privileged_oracle")

    fig, ax = plt.subplots(figsize=(9, max(5, len(scenes) * 0.32)))
    y = np.arange(len(scenes))
    for offset, bounds in [(-0.15, CURRENT), (0.15, GTFREE)]:
        name = f"query_gain|RGBD|{bounds}|FIXED_BUDGET|first_to_max"
        values = artifacts["bootstrap_results"]["comparisons"][name]["per_scene"]
        ax.barh(
            y + offset,
            [values[s] for s in scenes],
            0.29,
            color=colors[bounds],
            label=labels[bounds],
        )
    ax.set_yticks(y, scenes)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("1k − max-budget query AbsRel (positive: more optimization improves)")
    ax.set_title("Every exposed attribution scene; fixed budget selection")
    ax.legend()
    save(fig, "06_per_scene_1k_to_max_gain")

    fig, ax = plt.subplots(figsize=(7, 4))
    for bounds in (CURRENT, GTFREE):
        cells = [groups[group_key("RGBD", bounds, b)] for b in budgets]
        x = [c["cost"]["seconds_per_state"] for c in cells]
        y = [c["metrics"]["depth_absrel"]["mean"] for c in cells]
        ax.plot(x, y, "o-", color=colors[bounds], label=labels[bounds])
        for seconds, error, budget in zip(x, y, budgets, strict=True):
            ax.annotate(
                f"{budget // 1000}k", (seconds, error), xytext=(4, 4), textcoords="offset points"
            )
    ax.set(
        xlabel="Measured cumulative optimization seconds / state",
        ylabel="Query AbsRel",
        title="Measured direct-state cost; shared checkpoints are not independent runs",
    )
    ax.legend()
    save(fig, "07_compute_vs_query_quality")


def run(root, output=None, include_secondary=False):
    root = Path(root)
    output = Path(output) if output else root
    if include_secondary and output.resolve() == root.resolve():
        raise ValueError("Secondary report must use a separate output directory")
    provenance = {}
    manifest = load_json(root / "scene_manifest.json", provenance)
    rules = load_json(root / "decision_rules.json", provenance)
    plan = load_json(root / "state_plan.json", provenance)
    baseline = load_json(root / "baseline_results.json", provenance)
    query = load_json(root / "raw/context_results.json", provenance)
    context = load_json(root / "raw/context_context_results.json", provenance)
    oracle = load_json(root / "raw/oracle_results.json", provenance)
    if include_secondary:
        if not (root / "primary_analysis_complete.json").is_file():
            raise PermissionError("Primary analysis must precede secondary diagnosis")
        query += load_json(root / "raw/secondary_results.json", provenance)
        context += load_json(root / "raw/secondary_context_results.json", provenance)
    trajectories = load_trajectories(root, plan, provenance, include_secondary)
    artifacts = analyze(query, context, oracle, baseline, trajectories, manifest, rules)
    wall_path = root / "raw/primary_phase_wall_times.json"
    try:
        resolve(wall_path)
    except FileNotFoundError:
        artifacts["cost_analysis"]["phase_wall_times"] = {"status": "NOT_RECORDED"}
    else:
        artifacts["cost_analysis"]["phase_wall_times"] = load_json(wall_path, provenance)
    if include_secondary:
        secondary_wall_path = root / "raw/secondary_phase_wall_times.json"
        try:
            resolve(secondary_wall_path)
        except FileNotFoundError:
            artifacts["cost_analysis"]["secondary_phase_wall_times"] = {"status": "NOT_RECORDED"}
        else:
            artifacts["cost_analysis"]["secondary_phase_wall_times"] = load_json(
                secondary_wall_path, provenance
            )
    artifacts["cost_analysis"]["timing_limitations"] = (
        "Sum of cumulative worker seconds is not phase wall time. Shared-GPU concurrent "
        "per-state latency is not directly comparable to previous single-process timing."
    )
    primary_decision = artifacts["optimization_sufficiency"]["primary"]
    decision_hash = hashlib.sha256(
        json.dumps(primary_decision, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if include_secondary:
        locked = load_json(root / "primary_analysis_complete.json", provenance)
        if locked.get("status") != "PASS" or locked.get("primary_decision_sha256") != decision_hash:
            raise PermissionError("Secondary analysis changed frozen primary decision bits")
        if locked.get("primary_decision") != primary_decision:
            raise PermissionError("Primary decision payload differs after secondary")
        artifacts["secondary_decision_audit"] = {
            "status": "PASS",
            "primary_decision_sha256": decision_hash,
            "primary_bits_exactly_unchanged": True,
        }
    output.mkdir(parents=True, exist_ok=True)
    for name, value in artifacts.items():
        save_json(output / f"{name}.json", value)
    figures(artifacts, output / "figures")
    save_json(
        output / "statistics_reproduction.json",
        {
            "raw_only": True,
            "media_or_checkpoint_loaded": False,
            "inputs_sha256": provenance,
            "secondary_included": include_secondary,
            "source_sha256": {
                str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in [
                    Path(__file__),
                    Path(__file__).resolve().parents[1]
                    / "src/mcss/mechanism_pilot/optimization_bounds_statistics.py",
                ]
            },
        },
    )
    if not include_secondary:
        save_json(
            output / "primary_analysis_complete.json",
            {
                "status": "PASS",
                "primary_key": artifacts["optimization_results"]["primary_key"],
                "meaning": "PREDECLARED_PRIMARY_NOT_QUERY_MINIMUM",
                "primary_decision": primary_decision,
                "primary_decision_sha256": decision_hash,
                "inputs_sha256": provenance,
                "secondary_not_used": True,
            },
        )
    return artifacts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--include-secondary", action="store_true")
    args = parser.parse_args()
    result = run(args.root, args.output, args.include_secondary)
    print(json.dumps(result["optimization_sufficiency"]["primary"], indent=2))
