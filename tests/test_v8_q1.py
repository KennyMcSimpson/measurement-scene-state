import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mcss.geometry_supplier import GeometrySupplierRequest
from mcss.types import Cameras


def _load_runner():
    path = Path(__file__).parents[1] / "scripts" / "run_v8_q1.py"
    spec = importlib.util.spec_from_file_location("run_v8_q1", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_q1_rejects_non_train_and_non_32_window_manifests(tmp_path: Path) -> None:
    runner = _load_runner()
    path = tmp_path / "windows.json"
    path.write_text(json.dumps({"partition": "val", "windows": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="32 frozen development-train"):
        runner._load_frozen_windows(path)


def test_q1_requires_successful_q0_gate(tmp_path: Path) -> None:
    runner = _load_runner()
    path = tmp_path / "q0.json"
    path.write_text(json.dumps({"status": "STOP_EMVSNET_Q0"}), encoding="utf-8")
    with pytest.raises(ValueError, match="Q0 must pass"):
        runner._load_q0_gate(path)


def test_q1_gate_stops_when_full_risk_does_not_beat_comparators() -> None:
    runner = _load_runner()
    summary = {
        "window_count": 32,
        "comparison_coverage_macro": 1.0,
        "supported_coverage_macro": 0.2,
        "supported_nonempty_windows": 32,
        "drop_instability_median_macro": 0.3,
        "shuffle_response_median_macro": 0.5,
        "metrics": {
            "full": {"aurc_macro": 0.4, "ause_macro": 0.2, "spearman_macro": 0.3},
            "best_native": {"aurc_macro": 0.3, "ause_macro": 0.1},
            "same_family": {"aurc_macro": 0.35, "ause_macro": 0.15},
            "random": {"aurc_macro": 0.5},
            "full_no_drop": {"aurc_macro": 0.39},
            "full_no_disagreement": {"aurc_macro": 0.39},
        },
        "paired_ci": {
            "shuffle_minus_drop": {"ci95": [0.1, 0.3]},
            "full_minus_best_native_aurc": {"ci95": [0.05, 0.2]},
            "full_minus_same_family_aurc": {"ci95": [0.01, 0.1]},
            "full_minus_best_native_ause": {"ci95": [0.02, 0.1]},
        },
        "qjf": {"defined_windows": 32, "beats_native_at_both_thresholds": True},
    }

    decision = runner._decide(summary, _decision_config())

    assert decision["status"] == "STOP_V8_Q1_RISK_STATE"
    assert decision["gates"]["beats_best_native_aurc"] is False


def test_q1_decision_reads_ten_percent_gain_and_does_not_gate_supported_coverage() -> None:
    runner = _load_runner()
    summary = _passing_summary()
    summary["supported_coverage_macro"] = 0.0
    summary["supported_nonempty_windows"] = 0
    summary["metrics"]["full"]["aurc_macro"] = 0.91
    summary["metrics"]["full"]["ause_macro"] = 0.455

    decision = runner._decide(summary, _decision_config())

    assert decision["gates"]["beats_best_native_aurc"] is False
    assert decision["gates"]["beats_best_native_ause"] is False
    assert "supported_coverage" not in decision["gates"]


def test_q1_component_necessity_accepts_exact_two_percent_ablation_degradation() -> None:
    runner = _load_runner()
    summary = _passing_summary()
    summary["metrics"]["full"]["aurc_macro"] = 1.0
    summary["metrics"]["full_no_drop"]["aurc_macro"] = 1.02
    summary["metrics"]["full_no_disagreement"]["aurc_macro"] = 1.02

    decision = runner._decide(summary, _decision_config())

    assert decision["gates"]["component_necessity"] is True


def test_selective_severe_comparison_requires_configured_relative_gain() -> None:
    runner = _load_runner()
    negative_ci = {"ci95": [-0.2, -0.01]}

    assert not runner._selective_severe_comparison_passes(
        full=0.91,
        best_native=1.0,
        random=1.0,
        best_ci=negative_ci,
        random_ci=negative_ci,
        relative_gain=0.10,
    )
    assert runner._selective_severe_comparison_passes(
        full=0.90,
        best_native=1.0,
        random=1.0,
        best_ci=negative_ci,
        random_ci=negative_ci,
        relative_gain=0.10,
    )


def test_quiet_joint_failure_excludes_high_risk_shared_errors() -> None:
    runner = _load_runner()
    label_valid = np.asarray([True, True, True, False])
    supported = np.asarray([True, False, True, False])
    emv_error = np.asarray([0.6, 0.8, 0.1, 0.9])
    uni_error = np.asarray([0.7, 0.9, 0.8, 0.9])

    quiet, shared = runner._quiet_joint_failure_masks(
        emv_error, uni_error, label_valid, supported, threshold=0.5
    )

    np.testing.assert_array_equal(shared, [True, True, False, False])
    np.testing.assert_array_equal(quiet, [True, False, False, False])


def test_drop_anchor_contract_rejects_camera_drift() -> None:
    runner = _load_runner()
    intrinsics = torch.eye(3).repeat(4, 1, 1)
    c2w = torch.eye(4).repeat(4, 1, 1)
    request = GeometrySupplierRequest(
        (1, 2, 3, 4),
        torch.zeros((4, 3, 128, 160)),
        Cameras(intrinsics, c2w, (128, 160)),
    )
    drop_request = runner._select_request(request, (0, 2, 3))
    proposal_c2w = drop_request.cameras.c2w.numpy().copy()
    proposal_c2w[0, 0, 3] = 0.01
    proposal = SimpleNamespace(
        frame_ids=drop_request.frame_ids,
        intrinsics=drop_request.cameras.intrinsics.numpy().copy(),
        c2w=proposal_c2w,
    )

    with pytest.raises(ValueError, match="drop proposal anchor camera"):
        runner._check_drop_anchor_contract(request, drop_request, proposal)


def _decision_config() -> dict[str, float | int]:
    return {
        "minimum_windows": 32,
        "minimum_comparison_coverage_macro": 0.90,
        "minimum_relative_risk_gain": 0.10,
        "minimum_ablation_degradation": 0.02,
        "minimum_spearman_macro": 0.20,
        "minimum_nonnegative_spearman_windows": 26,
        "minimum_qjf_defined_windows": 24,
        "minimum_type_ordering_defined_windows": 24,
    }


def _passing_summary() -> dict:
    return {
        "window_count": 32,
        "comparison_coverage_macro": 1.0,
        "supported_coverage_macro": 0.2,
        "supported_nonempty_windows": 32,
        "drop_instability_median_macro": 0.3,
        "shuffle_response_median_macro": 0.5,
        "metrics": {
            "full": {
                "aurc_macro": 0.8,
                "ause_macro": 0.4,
                "spearman_macro": 0.3,
                "nonnegative_spearman_windows": 32,
            },
            "best_native": {"aurc_macro": 1.0, "ause_macro": 0.5},
            "same_family": {"aurc_macro": 1.0, "ause_macro": 0.5},
            "random": {"aurc_macro": 1.0},
            "full_no_drop": {"aurc_macro": 1.0},
            "full_no_disagreement": {"aurc_macro": 1.0},
        },
        "paired_ci": {
            "shuffle_minus_drop": {"ci95": [0.1, 0.3]},
            "full_minus_best_native_aurc": {"ci95": [-0.3, -0.1]},
            "full_minus_best_native_ause": {"ci95": [-0.2, -0.05]},
            "full_minus_same_family_aurc": {"ci95": [-0.3, -0.1]},
            "full_minus_same_family_ause": {"ci95": [-0.2, -0.05]},
            "full_minus_random_aurc": {"ci95": [-0.3, -0.1]},
            "full_minus_no_drop_aurc": {"ci95": [-0.3, -0.1]},
            "full_minus_no_disagreement_aurc": {"ci95": [-0.3, -0.1]},
        },
        "qjf": {"defined_windows": 32, "beats_native_at_both_thresholds": True},
        "selective_severe_pass": True,
        "type_ordering": {"defined_windows": 32, "passed": True},
    }
