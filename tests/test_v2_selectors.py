"""Nested protocol, fold isolation, reward baseline and conservative gate tests."""

import copy
import inspect

import numpy as np
import pytest

from mcss.vision_probe.v2_selectors import (
    TRAJECTORIES,
    _choice,
    _fit,
    _winner,
    fit_v2,
    predict_v2,
    select_v2,
)


def rows(zero=False):
    result = []
    for seq in range(3):
        for slot in range(1 + (seq == 2)):
            for i, t in enumerate(TRAJECTORIES):
                reward = 0.0 if zero else (0.02 if i == 5 else -0.01 if i else 0.0)
                result.append(
                    {
                        "split": "discovery",
                        "sequence": str(seq),
                        "target_slot": slot,
                        "trajectory": list(t),
                        "J": 0.4 + reward,
                        "visible": {
                            "quality": reward,
                            "sequence_signal": float(seq),
                            "f_cycle": reward / 2,
                        },
                    }
                )
    return result


def test_nested_isolation_weighted_scaler_and_final_model():
    data = rows()
    fitted = fit_v2(data, ["quality", "sequence_signal"], {})
    for name, cv in fitted["nested_cv_results"].items():
        assert cv["outer_oof_score_before_fallback"]["net_delta_J"] > 0
        assert cv["final_policy"] == "RIDGE_GATE"
        for outer in cv["outer_folds"]:
            held = set(outer["held_out_sequences"])
            assert held.isdisjoint(outer["training_sequences"])
            for fold in outer["inner_cv"]["folds"]:
                model = fold["model"]
                train = set(model["training_sequences"])
                assert train.isdisjoint(held | set(fold["held_out_sequences"]))
                index = model["feature_names"].index("sequence_signal")
                assert model["mean"][index] == pytest.approx(np.mean([float(s) for s in train]))
                totals = list(model["scaler_provenance"]["sequence_total_weights"].values())
                assert max(totals) == pytest.approx(min(totals))
                assert model["scaler_provenance"]["weight_sum"] == pytest.approx(
                    model["scaler_provenance"]["row_count"]
                )
            for decision in outer["oof_decisions"]:
                assert held.isdisjoint(decision["fitted_sequences"])
        artifact = fitted["selectors"][name]
        group = data[:16]
        assert (
            select_v2([r["visible"] for r in group], [r["trajectory"] for r in group], artifact)
            == 5
        )


def test_zero_reward_forces_always_off_and_no_final_tuning():
    data = rows(zero=True)
    fitted = fit_v2(data, ["quality", "sequence_signal"], {})
    for name, artifact in fitted["selectors"].items():
        assert artifact["policy"] == "ALWAYS_OFF"
        assert artifact["outer_nonpositive_forced_off"] is True
        assert fitted["nested_cv_results"][name]["final_tuning"] is None
        group = data[:16][::-1]
        features = [r["visible"] for r in group]
        assert select_v2(features, [r["trajectory"] for r in group], artifact) == 15
        assert predict_v2(features, artifact) == [0.0] * 16


def test_no_validation_rows_and_no_inference_label_injection():
    data = rows(zero=True)
    bad = copy.deepcopy(data)
    bad[0]["split"] = "validation"
    with pytest.raises(ValueError, match="discovery"):
        fit_v2(bad, ["quality", "sequence_signal"], {})
    artifact = fit_v2(data, ["quality", "sequence_signal"], {})["selectors"]["CycleGate"]
    features = [{**r["visible"], "target_J": r["J"]} for r in data[:16]]
    with pytest.raises(ValueError, match="visible"):
        predict_v2(features, artifact)
    assert set(inspect.signature(select_v2).parameters) == {"features", "trajectories", "artifact"}


def test_off_never_competes_and_strict_threshold_and_ties():
    prediction = [-1.0] * 16
    prediction[0] = 100.0
    prediction[2] = 0.001
    prediction[1] = 0.001
    assert _choice(prediction, TRAJECTORIES, 0.001) == 0
    assert _choice(prediction, TRAJECTORIES, 0.0005) == 1
    prediction[1] = 0.0009999999999
    assert _choice(prediction, TRAJECTORIES, 0.0005) == 1


def test_weighted_reward_fit_pair_baseline_invariance():
    data = rows()
    changed = copy.deepcopy(data)
    for row in changed:
        row["J"] += 7 * float(row["sequence"]) + float(row["target_slot"])
    first = _fit(data, ["quality", "sequence_signal"], 1.0)
    second = _fit(changed, ["quality", "sequence_signal"], 1.0)
    assert first["coefficients"] == pytest.approx(second["coefficients"], abs=1e-12)
    assert first["mean"][1] == pytest.approx(1.0)


def test_tuning_tie_break_low_writes_then_threshold_then_regularization():
    def c(reward, write, threshold, alpha):
        return {
            "score": {"net_delta_J": reward, "write_rate": write},
            "threshold": threshold,
            "alpha": alpha,
        }

    candidates = [
        c(0.001, 0.5, 0.005, 100),
        c(0.001 + 1e-13, 0.1, 0.0, 0.1),
        c(0.001, 0.1, 0.002, 1),
        c(0.001, 0.1, 0.002, 100),
    ]
    assert _winner(candidates) is candidates[-1]


def test_nested_results_are_deterministic_and_json_serializable():
    import json

    data = rows()
    first = fit_v2(data, ["quality", "sequence_signal"], {"seed": 20260926})
    second = fit_v2(data, ["quality", "sequence_signal"], {"seed": 20260926})
    assert json.dumps(first, sort_keys=True, allow_nan=False) == json.dumps(
        second, sort_keys=True, allow_nan=False
    )
    assert first["selectors"]["GateOnly"]["feature_names"] == ["quality", "sequence_signal"]
    assert first["selectors"]["CycleGate"]["feature_names"] == [
        "quality",
        "sequence_signal",
        "f_cycle",
    ]
