"""Behavioral tests for mask-free, locked opportunity selection."""

import copy
import inspect
import json

import numpy as np
import pytest
import torch

from mcss.vision_probe.adaptation import FastState, Readout
from mcss.vision_probe.opportunity_selectors import (
    enrich_candidates,
    fit_selectors,
    predict_candidates,
    select_candidates,
    validate_artifact,
    visible_features,
)


def rows():
    output = []
    for sequence in range(4):
        for slot in range(2):
            for trajectory, loss, reward in (
                (["OFF", "OFF"], 1.0, 0.0),
                (["A", "A"], 0.5, 0.1),
                (["B", "B"], 0.1, -0.2),
            ):
                output.append(
                    {
                        "split": "discovery",
                        "sequence": str(sequence),
                        "target_slot": slot,
                        "resolution": 224,
                        "trajectory": trajectory,
                        "J": 0.5 + reward,
                        "visible": {
                            "probe_mse_after": loss,
                            "quality": reward,
                            "sequence_signal": float(sequence),
                        },
                    }
                )
    return output


def test_visible_features_identity_has_perfect_cycles_and_zero_drift():
    target = torch.eye(4).reshape(2, 2, 4)
    readout = Readout(
        torch.zeros(4), torch.eye(4), torch.ones(4), torch.zeros(4, 3), torch.zeros(3)
    )
    features = visible_features(
        target, target, target, torch.ones(2, 2, 3), readout, FastState.zero(4), target
    )
    assert features["mutual_nn_fraction"] == 1
    assert features["cycle_error"] == 0
    assert features["feature_drift"] == 0
    assert features["feature_large_change_fraction"] == 0
    assert features["probe_mse_after"] == 1
    assert features["probe_residual_abs_before_0"] == 1
    assert features["rgb_residual_abs_after_2"] == 1
    assert features["displacement_smoothness"] == 0
    assert all(np.isfinite(list(features.values())))
    assert "mask" not in inspect.signature(visible_features).parameters


def test_candidates_enrichment_is_label_free_and_permutation_equivariant():
    features = [
        {"probe_mse_after": 0.3, "feature_drift": 0.0},
        {"probe_mse_after": 0.1, "feature_drift": 0.2},
    ]
    result = enrich_candidates(features)
    assert result[1]["rgb_rank"] == 0
    assert result[1]["rgb_confidence_margin"] == pytest.approx(0.2)
    assert result[0]["candidate_disagreement"] == pytest.approx(0.1)
    assert enrich_candidates(features[::-1]) == result[::-1]
    assert "rgb_rank" not in features[0]


def test_fit_rejects_validation_and_requires_sequence_cv():
    data = rows()
    data[-1]["split"] = "validation"
    with pytest.raises(ValueError, match="discovery"):
        fit_selectors(data, {})
    with pytest.raises(ValueError, match="two discovery"):
        fit_selectors([r for r in rows() if r["sequence"] == "0"], {})


def test_model_learns_task_reward_not_rgb_and_abstains_rgb():
    data = rows()
    artifact = json.loads(json.dumps(fit_selectors(data, {})))
    group = data[:3]
    selection = select_candidates(
        [r["visible"] for r in group], [r["trajectory"] for r in group], artifact
    )
    assert selection == {"visible_selector": 1, "rgb_selector": 0, "best_fixed": 1}
    model = validate_artifact(artifact)
    assert model["training_sequences"] == ["0", "1", "2", "3"]
    assert model["metadata"]["cv"] == "leave_one_sequence_out"
    assert model["rgb_threshold"] == 1
    assert len(predict_candidates([r["visible"] for r in group], artifact)) == 3


def test_resolution_models_are_separate_and_gate_ambiguous_inference():
    data = rows()
    second = copy.deepcopy(data)
    for r in second:
        r["resolution"] = 448
        r["visible"]["sequence_signal"] += 100
    artifact = fit_selectors(data + second, {})
    models = artifact["by_resolution"]
    index = models["224"]["feature_names"].index("sequence_signal")
    assert models["448"]["mean"][index] - models["224"]["mean"][index] == 100
    with pytest.raises(ValueError, match="resolution"):
        validate_artifact(artifact)


def test_ties_prefer_off_even_when_reordered_and_reject_metric_injection():
    data = rows()
    for r in data:
        r["J"] = 0.5
        r["visible"] = {"probe_mse_after": 1.0, "quality": 0.0, "sequence_signal": 0.0}
    artifact = fit_selectors(data, {})
    group = data[:3][::-1]
    assert select_candidates(
        [r["visible"] for r in group], [r["trajectory"] for r in group], artifact
    ) == {"rgb_selector": 2, "visible_selector": 2, "best_fixed": 2}
    injected = [{**r["visible"], "J": r["J"]} for r in group]
    with pytest.raises(ValueError, match="schema"):
        predict_candidates(injected, artifact)
    invalid = copy.deepcopy(artifact)
    invalid["fit_split"] = "validation"
    with pytest.raises(ValueError, match="discovery"):
        validate_artifact(invalid)


def test_cv_standardization_excludes_held_out_sequences(monkeypatch):
    import mcss.vision_probe.opportunity_selectors as module

    original = module._fit
    calls = []

    def record_fit(data, names, alpha):
        model = original(data, names, alpha)
        calls.append((sorted({r["sequence"] for r in data}), model))
        return model

    monkeypatch.setattr(module, "_fit", record_fit)
    fit_selectors(rows(), {})
    folds = [item for item in calls if len(item[0]) == 3]
    assert len(folds) == 16
    for sequences, model in folds:
        index = model["feature_names"].index("sequence_signal")
        assert model["mean"][index] == pytest.approx(np.mean([float(s) for s in sequences]))


def test_reward_regression_is_invariant_to_per_pair_baseline_shifts():
    import mcss.vision_probe.opportunity_selectors as module

    data = rows()
    shifted = copy.deepcopy(data)
    for r in shifted:
        r["J"] += 10.0 * int(r["sequence"]) + 3.0 * r["target_slot"]
    names = sorted(data[0]["visible"])
    first = module._fit(data, names, 1.0)
    second = module._fit(shifted, names, 1.0)
    assert second["coefficients"] == pytest.approx(first["coefficients"], abs=1e-12)
    assert second["intercept"] == pytest.approx(first["intercept"], abs=1e-12)
    assert first["intercept"] == pytest.approx(-0.1 / 3)


def test_visible_selection_pins_off_reward_to_zero():
    from mcss.vision_probe.opportunity_selectors import _choose_visible

    trajectories = [["OFF", "OFF"], ["A", "A"]]
    assert _choose_visible([100.0, 0.03], trajectories, 0.02) == 1
    assert _choose_visible([-100.0, 0.01], trajectories, 0.02) == 0
