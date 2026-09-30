#!/usr/bin/env python3
"""Preregistered descriptive state-correlation diagnostic on sealed V12 DEV states.

Reads only the saved sealed states (no model forward, no media, no labels). For every evaluated
variant x seed: mean Pearson correlation of density logits between the A and B states of the
same scene, between A states of different scenes, and between direct A and anchor-only states.
"""

import argparse
import itertools
from pathlib import Path

import numpy as np
import torch

from mcss.mechanism_pilot.geometry_carrier_experiment import read
from mcss.mechanism_pilot.small_training import sha, write_json


def correlations(states):
    scenes = sorted({key.split("/")[0] for key in states})
    density = {k: s.density_logits.flatten().double().numpy() for k, s in states.items()}

    def corr(a, b):
        x, y = density[a], density[b]
        if x.std() == 0 or y.std() == 0:
            return None
        return float(np.corrcoef(x, y)[0, 1])

    def mean(values):
        values = [v for v in values if v is not None]
        return float(np.mean(values)) if values else None

    return {
        "within_scene_A_B": mean(corr(f"{s}/A", f"{s}/B") for s in scenes),
        "across_scene_A": mean(
            corr(f"{a}/A", f"{b}/A") for a, b in itertools.combinations(scenes, 2)
        ),
        "direct_A_vs_anchor_A": mean(corr(f"{s}/A", f"{s}/anchor_A") for s in scenes),
        "n_scenes": len(scenes),
    }


def diagnose(root):
    root = Path(root).resolve()
    evaluated = read(root / "evaluated_variants.json")["variants"]
    seeds = read(root / "training_contract.json")["seeds"]
    rows, inputs = {}, {}
    for variant in evaluated:
        for seed in seeds:
            path = root / "raw/selected_dev" / f"{variant}_{seed}" / "states.pt"
            inputs[str(path)] = sha(path)
            states = torch.load(path, map_location="cpu", weights_only=False)
            rows[f"{variant}_{seed}"] = correlations(states)
    summary = {}
    for variant in evaluated:
        per = [rows[f"{variant}_{seed}"] for seed in seeds]
        summary[variant] = {
            key: float(np.mean([p[key] for p in per if p[key] is not None]))
            if any(p[key] is not None for p in per)
            else None
            for key in ("within_scene_A_B", "across_scene_A", "direct_A_vs_anchor_A")
        }
    write_json(
        root / "state_correlation.json",
        {
            "per_run": rows,
            "per_variant_mean_over_seeds": summary,
            "input_sha256": inputs,
            "descriptive_only": True,
            "definition": "Pearson correlation of flattened density logits; no gate uses it",
        },
    )
    print("STATE_CORRELATION_DONE", summary)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    a = p.parse_args()
    diagnose(a.root)
