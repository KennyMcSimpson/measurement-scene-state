"""Summarize a completed exploratory probe using sequences as sampling units."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def mean(values):
    return float(np.mean(values)) if values else None


def paired_interval(values, seed=20260921):
    if not values:
        return {"mean": None, "lower95": None, "upper95": None, "n_sequences": 0}
    x = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    boot = x[rng.integers(0, len(x), size=(10000, len(x)))].mean(axis=1)
    return {
        "mean": float(x.mean()),
        "lower95": float(np.quantile(boot, 0.025)),
        "upper95": float(np.quantile(boot, 0.975)),
        "n_sequences": len(x),
    }


def sequence_mean(rows, field):
    grouped = defaultdict(list)
    for row in rows:
        value = row.get(field)
        if value is not None:
            grouped[row["sequence"]].append(float(value))
    return {key: mean(values) for key, values in grouped.items()}


def paired_delta(rows, reference):
    grouped = defaultdict(list)
    deltas = []
    for row in rows:
        key = (row["sequence"], row["target_slot"])
        base = reference[key]
        if row.get("mean_iou") is None or base.get("mean_iou") is None:
            continue
        delta = row["mean_iou"] - base["mean_iou"]
        grouped[row["sequence"]].append(delta)
        deltas.append(delta)
    out = paired_interval([mean(values) for values in grouped.values()])
    out.update(
        {
            "pair_wins": sum(d > 1e-8 for d in deltas),
            "pair_losses": sum(d < -1e-8 for d in deltas),
            "pair_ties": sum(abs(d) <= 1e-8 for d in deltas),
            "per_sequence": {key: mean(values) for key, values in grouped.items()},
        }
    )
    return out


def analyze(directory: Path):
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    if summary["status"] != "complete":
        raise ValueError("Only completed runs may be analyzed")
    rows = [json.loads(line) for line in (directory / "raw.jsonl").read_text().splitlines()]
    selected_eta = summary["discovery_selection"]["eta"]
    output = {
        "schema": "mcss.vision_probe.analysis.v1",
        "status": "complete",
        "official_benchmark": False,
        "scope": (
            "Frozen DINO features, observed RGB feedback, per-image private writes; "
            "not full inverse JEPA or streaming TTT"
        ),
        "interval_note": (
            "Descriptive paired percentile bootstrap, 10000 draws over sequences; "
            "no multiplicity correction"
        ),
        "selected_eta": selected_eta,
        "splits": {},
    }
    for split in ("discovery", "validation"):
        selected = [r for r in rows if r["split"] == split and r["eta"] == selected_eta]
        by_method = defaultdict(list)
        for row in selected:
            by_method[row["method"]].append(row)
        baseline = {(r["sequence"], r["target_slot"]): r for r in by_method["OFFOFF"]}
        methods = {}
        for method, records in by_method.items():
            if method == "trajectory":
                continue
            methods[method] = {
                "n_pairs": len(records),
                "mean_object_iou": mean(list(sequence_mean(records, "mean_iou").values())),
                "foreground_iou": mean(list(sequence_mean(records, "foreground_iou").values())),
                "rgb_mse": mean(list(sequence_mean(records, "mse").values())),
                "delta_iou_vs_off": paired_delta(records, baseline),
                "action_pairs": dict(Counter("/".join(r["action_pair"]) for r in records)),
                "mean_final_a_norm": mean([r["state_a_norm"] for r in records]),
                "mean_final_b_norm": mean([r["state_b_norm"] for r in records]),
                "mean_committed_write_events": mean(
                    [
                        sum(2 if a == "ALL" else 0 if a == "OFF" else 1 for a in r["action_pair"])
                        for r in records
                    ]
                ),
                "mean_candidate_readouts": mean(
                    [
                        sum(step.get("candidate_evaluations", 0) for step in r["candidate_ledger"])
                        for r in records
                    ]
                ),
            }
        constants = {
            (r["sequence"], r["target_slot"]): r for r in by_method["oracle_bestconstantperpair"]
        }
        opportunity = paired_delta(by_method["oracle_bestsequenceperpair"], constants)
        delta_rgb, delta_iou = [], []
        for row in by_method["trajectory"]:
            if row["action_pair"] == ["OFF", "OFF"] or row.get("mean_iou") is None:
                continue
            base = baseline[(row["sequence"], row["target_slot"])]
            if base.get("mean_iou") is None:
                continue
            delta_rgb.append(base["mse"] - row["mse"])
            delta_iou.append(row["mean_iou"] - base["mean_iou"])
        correlation = None
        if len(delta_rgb) > 2 and np.std(delta_rgb) > 1e-12 and np.std(delta_iou) > 1e-12:
            correlation = float(np.corrcoef(delta_rgb, delta_iou)[0, 1])
        output["splits"][split] = {
            "methods": methods,
            "oracle_trajectory_minus_oracle_constant": opportunity,
            "pooled_rgb_improvement_iou_delta_correlation_descriptive_only": correlation,
            "correlation_note": (
                "Actions and target frames are dependent; "
                "this correlation has no inferential p-value"
            ),
        }
    (directory / "analysis.json").write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = analyze(args.output)
    for split, values in result["splits"].items():
        print(split)
        for method, metrics in values["methods"].items():
            delta = metrics["delta_iou_vs_off"]
            print(
                method,
                "IoU",
                metrics["mean_object_iou"],
                "RGB",
                metrics["rgb_mse"],
                "delta",
                delta["mean"],
                "CI",
                delta["lower95"],
                delta["upper95"],
            )
