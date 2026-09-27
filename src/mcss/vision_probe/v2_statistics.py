"""Label-side V2 statistics, with sequence macro weighting and global reoptimization.

These functions consume saved candidate scores only. They never load media,
fit a selector, or choose a deployable policy. Hindsight-global16 and both
oracles use answers and are diagnostic only. Their nonnegative gaps are partly
constructive, not evidence of deployment benefit or an order-causal mechanism.
"""

from __future__ import annotations

import itertools
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

ACTIONS = ("OFF", "A", "B", "ALL")
TRAJECTORIES = tuple(itertools.product(ACTIONS, repeat=2))
OFF = ("OFF", "OFF")
CONSTANT = tuple(i for i, t in enumerate(TRAJECTORIES) if t[0] == t[1])


def _trajectory(value: Any) -> tuple[str, str]:
    value = tuple(value.split("/")) if isinstance(value, str) else tuple(value)
    if value not in TRAJECTORIES:
        raise ValueError(f"Unknown trajectory: {value}")
    return value


def _label(trajectory: tuple[str, str]) -> str:
    return "/".join(trajectory)


def _prepare(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("Candidate rows cannot be empty")
    for field in ("resolution", "split"):
        values = {row.get(field) for row in rows}
        if len(values) > 1:
            raise ValueError(f"Rows must contain exactly one {field}")
    pairs: dict[tuple[str, str], dict[tuple[str, str], float]] = {}
    for row in rows:
        key = (str(row["sequence"]), str(row["target_slot"]))
        trajectory = _trajectory(row["trajectory"])
        score = float(row["J"])
        if not np.isfinite(score) or not 0.0 <= score <= 1.0:
            raise ValueError("J must be finite and within [0,1]")
        candidates = pairs.setdefault(key, {})
        if trajectory in candidates:
            raise ValueError(f"Duplicate candidate: {key}/{trajectory}")
        candidates[trajectory] = score
    missing = [
        {
            "sequence": key[0],
            "target_slot": key[1],
            "missing_trajectories": [_label(t) for t in TRAJECTORIES if t not in values],
        }
        for key, values in sorted(pairs.items())
        if len(values) != 16
    ]
    if missing:
        return {
            "status": "RAW_INCOMPLETE",
            "missing": missing,
            "scope": "Coverage of observed pairs only; absent whole pairs need manifest checks",
        }
    keys = sorted(pairs)
    sequences = sorted({key[0] for key in keys})
    indices = [np.array([i for i, key in enumerate(keys) if key[0] == s]) for s in sequences]
    scores = np.array([[pairs[key][t] for t in TRAJECTORIES] for key in keys], dtype=float)
    return {
        "status": "COMPLETE",
        "keys": keys,
        "sequences": sequences,
        "indices": indices,
        "scores": scores,
    }


def _parameters(seed: int, draws: int, tol: float) -> None:
    if isinstance(draws, bool) or not isinstance(draws, int) or draws < 1:
        raise ValueError("draws must be a positive integer")
    if not np.isfinite(tol) or tol < 0:
        raise ValueError("tol must be finite and nonnegative")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")


def _sequence_values(values: np.ndarray, indices: list[np.ndarray]) -> np.ndarray:
    return np.array([values[idx].mean(axis=0) for idx in indices])


def _intervals(point: float, samples: np.ndarray) -> dict[str, Any]:
    return {
        "mean": float(point),
        "ci95": np.quantile(samples, [0.025, 0.975]).tolist(),
        "ci97_5": np.quantile(samples, [0.0125, 0.9875]).tolist(),
    }


def _counts(values: np.ndarray, tol: float) -> dict[str, int]:
    return {
        "improved": int(np.sum(values > tol)),
        "tied": int(np.sum(np.abs(values) <= tol)),
        "worse": int(np.sum(values < -tol)),
    }


def _contributions(sequences: list[str], values: np.ndarray) -> dict[str, Any]:
    positive = np.maximum(values, 0)
    total = float(positive.sum())
    order = sorted(range(len(sequences)), key=lambda i: (-positive[i], sequences[i]))
    return {
        "per_sequence": [
            {
                "sequence": s,
                "mean_gap": float(value),
                "weighted_contribution": float(value / len(sequences)),
                "positive_contribution_share": float(max(value, 0) / total) if total > 0 else None,
            }
            for s, value in zip(sequences, values, strict=True)
        ],
        "top1_positive_share": float(positive[order[:1]].sum() / total) if total > 0 else None,
        "top3_positive_share": float(positive[order[:3]].sum() / total) if total > 0 else None,
        "concentration_denominator": "sum of positive sequence mean gaps; equal sequence weights",
    }


def _global_subset(seq_scores: np.ndarray, seq_dynamic: np.ndarray, subset: np.ndarray) -> dict:
    """Repeated indices retain resampled sequence multiplicity; winner is recomputed."""
    averages = seq_scores[subset].mean(axis=0)
    best = int(averages.argmax())
    return {
        "mean": float(seq_dynamic[subset].mean() - averages[best]),
        "hindsight_global16": _label(TRAJECTORIES[best]),
        "hindsight_global16_J": float(averages[best]),
    }


def global_diagnostic(
    rows: Sequence[Mapping[str, Any]],
    fixed_trajectory: Any = None,
    seed: int = 20260926,
    draws: int = 10000,
    tol: float = 1e-12,
) -> dict[str, Any]:
    """Analyze one resolution/split's complete 16-candidate saved scores.

    Discovery-fixed16 must be supplied, never chosen from these answers. The
    bootstrap samples sequences uniformly with replacement and preserves every
    pair within each draw, with equal pair weights inside each sequence. The
    hindsight winner is reoptimized within EVERY bootstrap and LOSO sample.
    ci97_5 is available but oracle intervals remain descriptive, not deployment
    tests. Missing candidate rows return RAW_INCOMPLETE with explicit omissions.
    """
    _parameters(seed, draws, tol)
    data = _prepare(rows)
    if data["status"] != "COMPLETE":
        return data
    scores, indices = data["scores"], data["indices"]
    sequences, keys = data["sequences"], data["keys"]
    fixed = None if fixed_trajectory is None else TRAJECTORIES.index(_trajectory(fixed_trajectory))
    seq_scores = _sequence_values(scores, indices)
    pair_dynamic = scores.max(axis=1)
    pair_constant = scores[:, CONSTANT].max(axis=1)
    seq_dynamic = _sequence_values(pair_dynamic, indices)
    seq_constant = _sequence_values(pair_constant, indices)
    global_means = seq_scores.mean(axis=0)
    winner = int(global_means.argmax())
    global_max = float(global_means[winner])
    tied_global = np.flatnonzero(global_max - global_means <= tol)
    tied_pairs = [
        np.flatnonzero(best - row <= tol) for best, row in zip(pair_dynamic, scores, strict=True)
    ]
    intersection = set(range(16)).intersection(*(set(v.tolist()) for v in tied_pairs))
    sample = np.random.default_rng(seed).integers(0, len(sequences), (draws, len(sequences)))
    bootstrap_dynamic = seq_dynamic[sample].mean(axis=1)
    bootstrap_global = seq_scores[sample].mean(axis=1).max(axis=1)
    gap_inputs = {
        "write_opportunity": (
            pair_dynamic - scores[:, 0],
            seq_dynamic - seq_scores[:, 0],
            bootstrap_dynamic - seq_scores[sample, 0].mean(axis=1),
        ),
        "nonconstant_extra": (
            pair_dynamic - pair_constant,
            seq_dynamic - seq_constant,
            bootstrap_dynamic - seq_constant[sample].mean(axis=1),
        ),
        "sample_dependence_gap": (
            pair_dynamic - scores[:, winner],
            seq_dynamic - seq_scores[:, winner],
            bootstrap_dynamic - bootstrap_global,
        ),
    }
    gaps = {}
    for name, (pair_gap, seq_gap, samples) in gap_inputs.items():
        loso = []
        if len(sequences) > 1:
            for i, sequence in enumerate(sequences):
                subset = np.delete(np.arange(len(sequences)), i)
                result = (
                    _global_subset(seq_scores, seq_dynamic, subset)
                    if name == "sample_dependence_gap"
                    else {"mean": float(seq_gap[subset].mean())}
                )
                loso.append({"omitted_sequence": sequence, **result})
        gaps[name] = {
            **_intervals(float(seq_gap.mean()), samples),
            **_contributions(sequences, seq_gap),
            "pair_counts": _counts(pair_gap, tol),
            "sequence_counts": _counts(seq_gap, tol),
            "pair_gain_threshold_counts": {
                str(t): int(np.sum(pair_gap >= t)) for t in (0.005, 0.01, 0.02)
            },
            "sequence_gain_threshold_counts": {
                str(t): int(np.sum(seq_gap >= t)) for t in (0.005, 0.01, 0.02)
            },
            "leave_one_sequence_out": loso,
            "loso_min": min((v["mean"] for v in loso), default=None),
            "loso_max": max((v["mean"] for v in loso), default=None),
            "positive_after_every_omission": all(v["mean"] > tol for v in loso) if loso else None,
            "hindsight_reoptimized_in_resamples": name == "sample_dependence_gap",
        }
    return {
        "status": "COMPLETE",
        "n_sequences": len(sequences),
        "n_pairs": len(keys),
        "n_candidates_per_pair": 16,
        "aggregation": "object-mean J input; pair mean within sequence; equal sequence mean",
        "coverage_scope": (
            "All 16 trajectories for observed pairs; validate identities/slots against manifest"
        ),
        "seed": seed,
        "draws": draws,
        "tie_tolerance": tol,
        "means": {
            "OFF": float(global_means[0]),
            "discovery_fixed16": None if fixed is None else float(global_means[fixed]),
            "constant_oracle": float(seq_constant.mean()),
            "dynamic_oracle": float(seq_dynamic.mean()),
            "hindsight_global16": global_max,
        },
        "discovery_fixed16_trajectory": None if fixed is None else _label(TRAJECTORIES[fixed]),
        "hindsight_global16_trajectory": _label(TRAJECTORIES[winner]),
        "all_global_optimal_trajectories": [_label(TRAJECTORIES[i]) for i in tied_global],
        "global_trajectories": [
            {
                "trajectory": _label(t),
                "J": float(global_means[i]),
                "regret_to_pair_oracle": float(seq_dynamic.mean() - global_means[i]),
            }
            for i, t in enumerate(TRAJECTORIES)
        ],
        "pair_optimal_set_intersection": [_label(TRAJECTORIES[i]) for i in sorted(intersection)],
        "one_trajectory_optimal_for_all_pairs": bool(intersection),
        "strict_nonconstant_pair_count": int(np.sum(pair_dynamic - pair_constant > tol)),
        "strict_nonconstant_sequence_count": int(
            sum(np.any((pair_dynamic - pair_constant)[idx] > tol) for idx in indices)
        ),
        "per_pair": [
            {
                "sequence": key[0],
                "target_slot": key[1],
                "optimal_trajectories": [_label(TRAJECTORIES[i]) for i in tied],
                "optimal_constant_trajectories": [
                    _label(TRAJECTORIES[i]) for i in CONSTANT if constant - row[i] <= tol
                ],
                "dynamic_oracle_J": float(best),
                "constant_oracle_J": float(constant),
                "OFF_J": float(row[0]),
                "strict_nonconstant_extra": float(best - constant),
            }
            for key, row, best, constant, tied in zip(
                keys, scores, pair_dynamic, pair_constant, tied_pairs, strict=True
            )
        ],
        "gaps": gaps,
        "interpretation": (
            "Answer-visible finite-family opportunity; nonconstant is not causal order evidence"
        ),
    }


def _decisions(decisions: Any, keys: list[tuple[str, str]]) -> np.ndarray:
    if isinstance(decisions, Mapping):
        selected = {(str(k[0]), str(k[1])): _trajectory(v) for k, v in decisions.items()}
        if len(selected) != len(decisions):
            raise ValueError("Duplicate normalized decision keys")
    else:
        selected = {}
        for row in decisions:
            key = (str(row["sequence"]), str(row["target_slot"]))
            if key in selected:
                raise ValueError("Duplicate decision")
            selected[key] = _trajectory(row["trajectory"])
    if set(selected) != set(keys):
        raise ValueError("Decisions must cover every pair exactly, without extra identities")
    return np.array([TRAJECTORIES.index(selected[key]) for key in keys])


def selector_summary(
    rows: Sequence[Mapping[str, Any]],
    decisions: Any,
    fixed_trajectory: Any = None,
    seed: int = 20260926,
    draws: int = 10000,
    tol: float = 1e-12,
) -> dict[str, Any]:
    """Score one frozen selector's externally supplied decisions, never fit it.

    decisions is {(sequence,target_slot): trajectory} or a list of records with
    sequence, target_slot, trajectory. Gains/captures/CI use sequence macro
    weights. Decision rates explicitly count pairs (weighted counterparts are
    also supplied). Beneficial recall labels a pair by dynamic-oracle gain>tol;
    precision and harmful-write rates divide by all writing pairs. Zero ratio
    denominators return None/JSON null (NA), never a fabricated zero or one.
    """
    _parameters(seed, draws, tol)
    data = _prepare(rows)
    if data["status"] != "COMPLETE":
        return data
    scores, indices = data["scores"], data["indices"]
    sequences, keys = data["sequences"], data["keys"]
    chosen = _decisions(decisions, keys)
    fixed = None if fixed_trajectory is None else TRAJECTORIES.index(_trajectory(fixed_trajectory))
    values = scores[np.arange(len(keys)), chosen]
    off, oracle = scores[:, 0], scores.max(axis=1)
    gain = values - off
    seq_j = _sequence_values(values, indices)
    seq_gain = _sequence_values(gain, indices)
    seq_oracle = _sequence_values(oracle - off, indices)
    positive = float(_sequence_values(np.maximum(gain, 0), indices).mean())
    harmful = float(_sequence_values(np.maximum(-gain, 0), indices).mean())
    net = positive - harmful
    oracle_gain = float(seq_oracle.mean())
    sample = np.random.default_rng(seed).integers(0, len(sequences), (draws, len(sequences)))
    comparisons = {"vs_OFF": _intervals(float(seq_gain.mean()), seq_gain[sample].mean(axis=1))}
    seq_fixed = None
    if fixed is not None:
        seq_fixed = _sequence_values(values - scores[:, fixed], indices)
        comparisons["vs_discovery_fixed16"] = _intervals(
            float(seq_fixed.mean()), seq_fixed[sample].mean(axis=1)
        )
    writes = chosen != 0
    benefit = gain > tol
    harm = gain < -tol
    opportunity = oracle - off > tol
    n_writes = int(writes.sum())
    per_sequence = []
    for i, (s, idx) in enumerate(zip(sequences, indices, strict=True)):
        per_sequence.append(
            {
                "sequence": s,
                "n_pairs": len(idx),
                "J": float(seq_j[i]),
                "delta_OFF": float(seq_gain[i]),
                "delta_fixed": None if seq_fixed is None else float(seq_fixed[i]),
                "weighted_net_contribution": float(seq_gain[i] / len(sequences)),
                "positive_gain": float(np.maximum(gain[idx], 0).mean()),
                "harmful_loss": float(np.maximum(-gain[idx], 0).mean()),
                "write_rate": float(writes[idx].mean()),
            }
        )
    loso = []
    if len(sequences) > 1:
        for i, s in enumerate(sequences):
            keep = np.delete(np.arange(len(sequences)), i)
            loso.append(
                {
                    "omitted_sequence": s,
                    "delta_OFF": float(seq_gain[keep].mean()),
                    "delta_fixed": None if seq_fixed is None else float(seq_fixed[keep].mean()),
                }
            )
    return {
        "status": "COMPLETE",
        "n_sequences": len(sequences),
        "n_pairs": len(keys),
        "mean_J": float(seq_j.mean()),
        "mean_OFF_J": float(_sequence_values(off, indices).mean()),
        "mean_fixed_J": None
        if fixed is None
        else float(_sequence_values(scores[:, fixed], indices).mean()),
        "comparisons": comparisons,
        "seed": seed,
        "draws": draws,
        "tie_tolerance": tol,
        "ci_scope": (
            "95% descriptive; 97.5% two-sided Bonferroni for the two "
            "prespecified primary H2 comparisons only"
        ),
        "positive_gain": positive,
        "harmful_loss": harmful,
        "net_gain": net,
        "oracle_gain": oracle_gain,
        "positive_capture": positive / oracle_gain if oracle_gain > 0 else None,
        "net_capture": net / oracle_gain if oracle_gain > 0 else None,
        "regret_to_dynamic_oracle": float(_sequence_values(oracle - values, indices).mean()),
        "write_pairs": n_writes,
        "write_rate": n_writes / len(keys),
        "OFF_rate": 1 - n_writes / len(keys),
        "sequence_weighted_write_rate": float(_sequence_values(writes, indices).mean()),
        "harmful_write_pairs": int(np.sum(harm & writes)),
        "harmful_write_rate": int(np.sum(harm & writes)) / n_writes if n_writes else None,
        "beneficial_write_precision": int(np.sum(benefit & writes)) / n_writes
        if n_writes
        else None,
        "beneficial_write_recall": (
            int(np.sum(benefit & writes)) / int(opportunity.sum()) if opportunity.any() else None
        ),
        "decision_rate_denominators": {
            "harmful_write_rate": "all writing pairs",
            "beneficial_write_precision": "all writing pairs",
            "beneficial_write_recall": "pairs with dynamic-oracle gain>tol",
        },
        "pair_counts": _counts(gain, tol),
        "sequence_counts": _counts(seq_gain, tol),
        "pair_gain_threshold_counts": {str(t): int(np.sum(gain >= t)) for t in (0.005, 0.01, 0.02)},
        "sequence_gain_threshold_counts": {
            str(t): int(np.sum(seq_gain >= t)) for t in (0.005, 0.01, 0.02)
        },
        "per_sequence": per_sequence,
        "leave_one_sequence_out": loso,
        "net_gain_contributions": _contributions(sequences, seq_gain),
    }
