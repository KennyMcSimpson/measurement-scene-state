"""Hand-computable V2 statistics; no media or previously sealed labels."""

import json

import numpy as np
import pytest

from mcss.vision_probe.v2_statistics import (
    TRAJECTORIES,
    global_diagnostic,
    selector_summary,
)


def _pair(sequence, slot, overrides=None, baseline=0.2):
    overrides = overrides or {}
    return [
        {
            "sequence": sequence,
            "target_slot": slot,
            "trajectory": list(t),
            "J": overrides.get("/".join(t), baseline),
        }
        for t in TRAJECTORIES
    ]


def _opposed():
    return _pair("s1", 1, {"A/ALL": 0.8}) + _pair("s2", 1, {"B/A": 0.8})


def test_global_oracles_are_distinct_and_loso_reoptimizes_global_winner():
    result = global_diagnostic(_opposed(), fixed_trajectory=("OFF", "OFF"), draws=1000)
    assert result["means"] == pytest.approx(
        {
            "OFF": 0.2,
            "discovery_fixed16": 0.2,
            "constant_oracle": 0.2,
            "dynamic_oracle": 0.8,
            "hindsight_global16": 0.5,
        }
    )
    assert result["all_global_optimal_trajectories"] == ["A/ALL", "B/A"]
    assert result["pair_optimal_set_intersection"] == []
    assert result["strict_nonconstant_pair_count"] == 2
    assert result["strict_nonconstant_sequence_count"] == 2
    gap = result["gaps"]["sample_dependence_gap"]
    assert gap["mean"] == pytest.approx(0.3)
    assert gap["ci95"] == pytest.approx([0, 0.3])
    assert gap["loso_min"] == gap["loso_max"] == 0
    assert [x["hindsight_global16"] for x in gap["leave_one_sequence_out"]] == ["B/A", "A/ALL"]
    assert gap["hindsight_reoptimized_in_resamples"]
    assert not gap["positive_after_every_omission"]
    assert result["gaps"]["nonconstant_extra"]["mean"] == pytest.approx(0.6)


def test_bootstrap_repeated_sequence_draws_retain_multiplicity_and_reoptimize():
    # Three unlike sequences make 2x s1 + 1x s2 different from unique {s1,s2}.
    rows = _opposed() + _pair("s3", 1, {"ALL/B": 0.65}, baseline=0.1)
    result = global_diagnostic(rows, draws=37, seed=17)
    groups = [[row for row in rows if row["sequence"] == s] for s in ("s1", "s2", "s3")]
    samples = np.random.default_rng(17).integers(0, 3, size=(37, 3))
    reference = []
    for draw in samples:
        dynamic = sum(max(row["J"] for row in groups[i]) for i in draw) / 3
        global_j = max(sum(groups[i][t]["J"] for i in draw) / 3 for t in range(16))
        reference.append(dynamic - global_j)
    gap = result["gaps"]["sample_dependence_gap"]
    assert gap["ci95"] == pytest.approx(np.quantile(reference, [0.025, 0.975]))
    assert gap["ci97_5"] == pytest.approx(np.quantile(reference, [0.0125, 0.9875]))


def _unequal_pairs():
    rows = _pair("s1", 1, {"OFF/OFF": 0.1, "A/ALL": 0.9}, baseline=0.1)
    for slot in (1, 2, 3):
        rows += _pair("s2", slot, {"A/ALL": 0.1, "B/B": 0.8}, baseline=0.2)
    return rows


def test_all_target_slots_count_but_sequences_have_equal_weight():
    result = global_diagnostic(_unequal_pairs(), draws=50)
    assert result["n_pairs"] == 4
    assert result["means"]["OFF"] == pytest.approx(0.15)
    assert result["means"]["dynamic_oracle"] == pytest.approx(0.85)
    assert result["means"]["hindsight_global16"] == pytest.approx(0.5)
    assert result["hindsight_global16_trajectory"] == "A/ALL"
    assert result["gaps"]["write_opportunity"]["mean"] == pytest.approx(0.7)
    contribution = result["gaps"]["write_opportunity"]
    assert contribution["top1_positive_share"] == pytest.approx(0.8 / 1.4)
    assert sum(x["weighted_contribution"] for x in contribution["per_sequence"]) == pytest.approx(
        0.7
    )


def test_selector_positive_harmful_and_net_capture_use_sequence_macro_weights():
    rows = _unequal_pairs()
    decisions = {(row["sequence"], row["target_slot"]): "A/ALL" for row in rows}
    result = selector_summary(rows, decisions, fixed_trajectory="OFF/OFF", draws=100)
    assert result["mean_J"] == pytest.approx(0.5)
    assert result["positive_gain"] == pytest.approx(0.4)
    assert result["harmful_loss"] == pytest.approx(0.05)
    assert result["net_gain"] == pytest.approx(0.35)
    assert result["oracle_gain"] == pytest.approx(0.7)
    assert result["positive_capture"] == pytest.approx(4 / 7)
    assert result["net_capture"] == pytest.approx(0.5)
    assert result["harmful_write_rate"] == 0.75
    assert result["beneficial_write_precision"] == 0.25
    assert result["beneficial_write_recall"] == 0.25
    assert result["pair_counts"] == {"improved": 1, "tied": 0, "worse": 3}
    assert result["sequence_counts"] == {"improved": 1, "tied": 0, "worse": 1}
    comparisons = result["comparisons"]
    assert comparisons["vs_OFF"] == comparisons["vs_discovery_fixed16"]
    assert comparisons["vs_OFF"]["ci97_5"][0] <= comparisons["vs_OFF"]["ci95"][0]
    assert comparisons["vs_OFF"]["ci97_5"][1] >= comparisons["vs_OFF"]["ci95"][1]
    json.dumps(result, allow_nan=False)


def test_ties_preserve_all_optimal_sets_and_zero_denominator_is_na():
    rows = _pair("s1", 1, baseline=0.4) + _pair("s2", 1, baseline=0.4)
    result = global_diagnostic(rows, draws=10)
    assert len(result["all_global_optimal_trajectories"]) == 16
    assert len(result["pair_optimal_set_intersection"]) == 16
    assert all(len(x["optimal_trajectories"]) == 16 for x in result["per_pair"])
    assert result["strict_nonconstant_pair_count"] == 0
    assert result["gaps"]["write_opportunity"]["top1_positive_share"] is None
    decisions = [
        {"sequence": s, "target_slot": 1, "trajectory": ["OFF", "OFF"]} for s in ("s1", "s2")
    ]
    selector = selector_summary(rows, decisions, draws=10)
    assert selector["positive_capture"] is selector["net_capture"] is None
    assert selector["harmful_write_rate"] is selector["beneficial_write_precision"] is None
    assert selector["beneficial_write_recall"] is None
    assert selector["write_rate"] == 0
    assert selector["pair_counts"]["tied"] == 2
    assert selector["comparisons"]["vs_OFF"]["ci95"] == [0.0, 0.0]
    json.dumps(result, allow_nan=False)


def test_tolerance_keeps_near_optima_without_claiming_strict_nonconstant():
    rows = _pair("s1", 1, {"A/A": 0.7, "B/ALL": 0.7 + 5e-13})
    result = global_diagnostic(rows, draws=10)
    assert result["per_pair"][0]["optimal_trajectories"] == ["A/A", "B/ALL"]
    assert result["strict_nonconstant_pair_count"] == 0
    assert result["gaps"]["write_opportunity"]["leave_one_sequence_out"] == []


def test_incomplete_raw_reports_missing_candidates_without_filling_scores():
    rows = _opposed()[:-1]
    result = global_diagnostic(rows)
    assert result["status"] == "RAW_INCOMPLETE"
    assert result["missing"] == [
        {"sequence": "s2", "target_slot": "1", "missing_trajectories": ["ALL/ALL"]}
    ]
    assert "means" not in result


def test_duplicates_mixed_splits_and_incomplete_decisions_fail():
    rows = _opposed()
    with pytest.raises(ValueError, match="Duplicate candidate"):
        global_diagnostic(rows + [rows[0]])
    with pytest.raises(ValueError, match="one split"):
        global_diagnostic(
            [{**r, "split": "discovery" if i else "validation"} for i, r in enumerate(rows)]
        )
    with pytest.raises(ValueError, match="cover every pair"):
        selector_summary(rows, {("s1", 1): "OFF/OFF"})


def test_reproducible_and_input_not_mutated():
    rows = _opposed()
    before = json.dumps(rows)
    assert global_diagnostic(rows, draws=100) == global_diagnostic(rows, draws=100)
    assert json.dumps(rows) == before


def test_shared_nonconstant_winner_has_no_sample_dependence_gap():
    # Nonconstant opportunity alone cannot establish a benefit of per-pair choice.
    rows = _pair("s1", 1, {"A/ALL": 0.8}) + _pair("s2", 1, {"A/ALL": 0.7})
    result = global_diagnostic(rows, draws=100)
    assert result["strict_nonconstant_pair_count"] == 2
    assert result["gaps"]["nonconstant_extra"]["mean"] == pytest.approx(0.55)
    assert result["pair_optimal_set_intersection"] == ["A/ALL"]
    assert result["one_trajectory_optimal_for_all_pairs"]
    gap = result["gaps"]["sample_dependence_gap"]
    assert gap["mean"] == 0
    assert gap["ci95"] == [0, 0]
    assert gap["loso_min"] == gap["loso_max"] == 0
