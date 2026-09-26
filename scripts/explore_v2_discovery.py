"""Post-hoc discovery diagnostics; never evaluate media or modify frozen V2."""

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from mcss.vision_probe.v2_statistics import TRAJECTORIES, selector_summary

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs/experiments/EXP-2D-opportunity-selector-v2-20260926T015728+0800"
OUT = ROOT / "docs/experiments/EXP-2D-followup-20260927"


def crossfit_global(rows):
    """Choose a shared path using other sequences only, then apply to held pairs."""
    seqs = sorted({r["sequence"] for r in rows})
    if len(seqs) < 2:
        raise ValueError("Need at least two sequences")
    groups = defaultdict(dict)
    for r in rows:
        key = (r["sequence"], r["target_slot"])
        path = tuple(r["trajectory"])
        if path in groups[key]:
            raise ValueError("Duplicate candidate")
        groups[key][path] = r["J"]
    if any(set(g) != set(TRAJECTORIES) for g in groups.values()):
        raise ValueError("Every pair must include all16 candidates")
    means = {
        s: np.mean(
            [[g[t] for t in TRAJECTORIES] for key, g in groups.items() if key[0] == s], axis=0
        )
        for s in seqs
    }
    decisions = []
    for held in seqs:
        train = [s for s in seqs if s != held]
        scores = np.mean([means[s] for s in train], axis=0)
        winner = next(i for i, v in enumerate(scores) if scores.max() - v <= 1e-12)
        for sequence, slot in sorted(groups):
            if sequence == held:
                decisions.append(
                    {
                        "sequence": held,
                        "target_slot": slot,
                        "trajectory": list(TRAJECTORIES[winner]),
                        "training_sequences": train,
                        "training_score": float(scores[winner]),
                    }
                )
    return decisions


def main():
    OUT.mkdir(exist_ok=True)
    inputs = ["discovery_rows.jsonl", "nested_cv_results.json", "selectors.json"]
    hashes = {name: hashlib.sha256((SOURCE / name).read_bytes()).hexdigest() for name in inputs}
    rows = [json.loads(x) for x in (SOURCE / inputs[0]).read_text().splitlines()]
    if {r["split"] for r in rows} != {"discovery"}:
        raise ValueError("Only discovery is permitted")
    nested = json.loads((SOURCE / inputs[1]).read_text())
    methods = {
        "LOSO_global16": crossfit_global(rows),
        **{name: data["outer_oof_decisions"] for name, data in nested.items()},
    }
    summaries = {
        name: selector_summary(rows, decisions, seed=20260927, draws=10000)
        for name, decisions in methods.items()
    }
    seqs = sorted({r["sequence"] for r in rows})
    sequence_j = {
        name: {r["sequence"]: r["J"] for r in summary["per_sequence"]}
        for name, summary in summaries.items()
    }
    rng = np.random.default_rng(20260927)
    indices = rng.integers(0, len(seqs), (10000, len(seqs)))
    comparisons = {}
    for left, right in [
        ("CycleGate", "LOSO_global16"),
        ("GateOnly", "LOSO_global16"),
        ("CycleGate", "GateOnly"),
    ]:
        values = np.array([sequence_j[left][s] - sequence_j[right][s] for s in seqs])
        comparisons[left + "-" + right] = {
            "mean": float(values.mean()),
            "ci95": np.quantile(values[indices].mean(1), [0.025, 0.975]).tolist(),
            "per_sequence": dict(zip(seqs, values.tolist(), strict=True)),
            "scope": (
                "post-hoc descriptive interval conditional on saved OOF policies; "
                "no retraining in bootstrap"
            ),
        }
    pairs = defaultdict(list)
    for r in rows:
        pairs[(r["sequence"], r["target_slot"])].append(r)
    signals = []
    for (s, slot), group in sorted(pairs.items()):
        off = next(r["J"] for r in group if tuple(r["trajectory"]) == ("OFF", "OFF"))
        nonoff = [r for r in group if tuple(r["trajectory"]) != ("OFF", "OFF")]
        feature = np.array([r["visible"]["f_cycle"] for r in nonoff])
        reward = np.array([r["J"] - off for r in nonoff])
        correlation = (
            float(spearmanr(feature, reward).statistic)
            if np.ptp(feature) > 1e-12 and np.ptp(reward) > 1e-12
            else None
        )
        signals.append(
            {
                "sequence": s,
                "target_slot": slot,
                "candidate_spearman": correlation,
                "f_cycle_range": float(np.ptp(feature)),
                "reward_range": float(np.ptp(reward)),
                "positive_cycle_but_harmful_candidates": int(
                    ((feature > 1e-12) & (reward < -1e-12)).sum()
                ),
                "positive_cycle_candidates": int((feature > 1e-12).sum()),
            }
        )
    folds = []
    for fold in nested["CycleGate"]["outer_folds"]:
        model = fold["model"]
        folds.append(
            {
                "held_out": fold["held_out_sequences"],
                "cycle_standardized_coefficient": None
                if model is None
                else model["coefficients"][model["feature_names"].index("f_cycle")],
            }
        )
    result = {
        "scope": "POST_HOC_DISCOVERY_ONLY_NOT_INDEPENDENT_CONFIRMATION",
        "seed": 20260927,
        "source_hashes": hashes,
        "methods": summaries,
        "comparisons": comparisons,
        "LOSO_global_decisions": methods["LOSO_global16"],
        "LOSO_global_winners": dict(
            Counter("/".join(r["trajectory"]) for r in methods["LOSO_global16"][::2])
        ),
        "cycle_within_pair_diagnostics": signals,
        "cycle_outer_coefficients": folds,
        "confirmation_status": "BLOCKED_NO_INDEPENDENT_DATA",
        "media_reads": 0,
        "target_gt_file_reads": 0,
        "v2_modified": False,
    }
    for name, digest in hashes.items():
        assert hashlib.sha256((SOURCE / name).read_bytes()).hexdigest() == digest
    (OUT / "discovery_diagnostics.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    print(
        json.dumps(
            {
                "comparisons": {
                    k: {x: v[x] for x in ["mean", "ci95"]} for k, v in comparisons.items()
                },
                "LOSO_global_winners": result["LOSO_global_winners"],
                "cycle_positive_but_harmful": sum(
                    x["positive_cycle_but_harmful_candidates"] for x in signals
                ),
                "positive_cycle": sum(x["positive_cycle_candidates"] for x in signals),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
