#!/usr/bin/env python3
"""Core B V1 raw-only statistics: policy contrasts with the frozen V2 bootstrap and gate checks."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from mcss.mechanism_pilot.direct_capacity_statistics import describe, paired
from mcss.mechanism_pilot.geometry_carrier_statistics import _evidence, surface_status

DRAWS, SEED = 10000, 20260928
CONTRASTS = {
    "WRITE_GAIN": ("OFF", "ALL"),
    "STREAM_WRITE_GAIN": ("NO_STREAM", "ALL"),
    "BEYOND_CLAMP_GAIN": ("OFF_CLAMP3", "ALL"),
    "WRITE_SPECIFICITY": ("ALL_WRONG_SCENE", "ALL"),
    "STREAM_CACHE_EFFECT": ("NO_STREAM", "OFF"),
    "CLAMP_EFFECT": ("OFF", "OFF_CLAMP3"),
    "CLAMP_VS_STATIC": ("NO_STREAM", "OFF_CLAMP3"),
    "FUSE_GAIN": ("OFF", "FUSE"),
    "COMPLETE_GAIN": ("OFF", "COMPLETE"),
    "TRAINING_GAIN": ("ALL_UNTRAINED", "ALL"),
}
GATED = {
    "WRITE_STATUS": "WRITE_GAIN",
    "STREAM_STATUS": "STREAM_WRITE_GAIN",
    "BEYOND_CLAMP_STATUS": "BEYOND_CLAMP_GAIN",
}


def load(root, name, provenance):
    path = Path(root) / name
    data = path.read_bytes()
    provenance[str(path.resolve())] = hashlib.sha256(data).hexdigest()
    return json.loads(data)


def scene_values(rows, policy, metric="depth_absrel", seed=None):
    """Per scene: seeds equally weighted; within a seed, the mean of role means of queries."""
    grouped = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r["policy"] == policy and (seed is None or r["seed"] == seed):
            grouped[r["seed"], r["scene_id"]][r["role"]].append(r[metric])
    per_scene = defaultdict(list)
    for (_, sid), roles in grouped.items():
        per_scene[sid].append(float(np.mean([np.mean(v) for v in roles.values()])))
    return {sid: float(np.mean(v)) for sid, v in sorted(per_scene.items())}


def gate(gain, state_ok):
    checks = {
        "mean_ci_and_scene_consistency": _evidence(gain),
        "every_leave_one_scene_out_gain_positive": bool(gain["loso"])
        and min(gain["loso"].values()) > 0,
        "positive_top1_share_at_most_0_5": gain["positive_top1_contribution"] is not None
        and gain["positive_top1_contribution"] <= 0.5,
        "state_integrity": state_ok,
    }
    return surface_status(gain, checks), checks


def analyze(rows, seeds, state_ok):
    policies = sorted({r["policy"] for r in rows})
    absolute = {p: describe(scene_values(rows, p), draws=DRAWS, seed=SEED) for p in policies}
    delta1 = {
        p: describe(scene_values(rows, p, "depth_delta1"), draws=DRAWS, seed=SEED) for p in policies
    }
    contrasts = {
        name: paired(scene_values(rows, a), scene_values(rows, b), draws=DRAWS, seed=SEED)
        for name, (a, b) in CONTRASTS.items()
    }
    statuses, checks = {}, {}
    for status, name in GATED.items():
        statuses[status], checks[status] = gate(contrasts[name], state_ok)
    specificity = contrasts["WRITE_SPECIFICITY"]
    statuses["WRITE_SPECIFICITY_STATUS"] = (
        "SUPPORTED" if specificity["ci95"][0] > 0 else "NOT_ESTABLISHED"
    )
    if all(statuses[k] == "SUPPORTED" for k in (*GATED, "WRITE_SPECIFICITY_STATUS")):
        branch = "A"
    elif statuses["WRITE_STATUS"] == "SUPPORTED":
        branch = "B"
    else:
        branch = "C"
    per_seed = {
        str(seed): {
            name: paired(
                scene_values(rows, a, seed=seed),
                scene_values(rows, b, seed=seed),
                draws=DRAWS,
                seed=SEED,
            )["mean"]
            for name, (a, b) in CONTRASTS.items()
        }
        for seed in seeds
    }
    direction = all(per_seed[str(s)]["WRITE_GAIN"] > 0 for s in seeds)
    return {
        "absolute_depth_absrel": absolute,
        "absolute_depth_delta1": delta1,
        "contrasts": contrasts,
        "contrast_definitions": {
            k: f"AbsRel({a}) - AbsRel({b})" for k, (a, b) in CONTRASTS.items()
        },
        "statuses": statuses,
        "gate_checks": checks,
        "per_seed_mean_gains": per_seed,
        "SEED_ROBUSTNESS": "DIRECTION_REPLICATED" if direction else "NOT_REPLICATED",
        "INTERPRETATION_BRANCH": branch,
        "scene_ids": sorted({r["scene_id"] for r in rows}),
    }


def figures(result, curves, directory):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    directory.mkdir(parents=True, exist_ok=True)
    order = [
        p
        for p in (
            "NO_STREAM",
            "OFF",
            "OFF_CLAMP3",
            "FUSE",
            "COMPLETE",
            "ALL",
            "ALL_UNTRAINED",
            "ALL_WRONG_SCENE",
        )
        if p in result["absolute_depth_absrel"]
    ]
    values = [result["absolute_depth_absrel"][p] for p in order]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(range(len(order)), [v["mean"] for v in values], color="#0072B2")
    for i, v in enumerate(values):
        ax.plot([i, i], v["ci95"], color="black")
    ax.set(xticks=list(range(len(order))), xticklabels=order, ylabel="DEV query AbsRel")
    ax.tick_params(axis="x", rotation=30)
    fig.savefig(directory / "01_policy_absrel.png", bbox_inches="tight", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4))
    for seed, rows in curves.items():
        steps = [r["step"] for r in rows]
        window = max(1, min(50, len(rows)))
        loss = np.convolve([r["loss_depth"] for r in rows], np.ones(window) / window, mode="same")
        ax.plot(steps, loss, label=f"seed {seed}")
    ax.set(xlabel="write-rule step", ylabel="TRAIN query depth AbsRel (50-step mean)")
    ax.legend()
    fig.savefig(directory / "02_write_rule_training.png", bbox_inches="tight", dpi=160)
    plt.close(fig)


def run(root, output=None):
    root = Path(root)
    output = Path(output) if output else root
    provenance = {}
    rows = load(root, "raw/dev_stream_query_results.json", provenance)
    contract = load(root, "training_contract.json", provenance)
    audit = load(root, "audit/dev_state_use.json", provenance)
    curves = load(root, "training_curves.json", provenance)
    result = analyze(rows, contract["seeds"], audit["status"] == "PASS")
    output.mkdir(parents=True, exist_ok=True)
    (output / "stream_results.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    figures(result, curves, output / "figures")
    (output / "statistics_reproduction.json").write_text(
        json.dumps(
            {
                "status": "PASS",
                "raw_only": True,
                "input_sha256": provenance,
                "source_sha256": {
                    str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in (
                        Path(__file__).resolve(),
                        Path(__file__).resolve().parents[1]
                        / "src/mcss/mechanism_pilot/direct_capacity_statistics.py",
                        Path(__file__).resolve().parents[1]
                        / "src/mcss/mechanism_pilot/geometry_carrier_statistics.py",
                    )
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    run(args.root, args.output)
