from pathlib import Path

from mcss.certificate_audit import (
    AuditDecisionThresholds,
    decide_audit,
    render_markdown_report,
    select_frozen_windows,
)


def test_hash_window_selection_is_deterministic_and_scene_balanced() -> None:
    entries = [(f"scene_{scene}", start) for scene in range(8) for start in range(6)]

    first = select_frozen_windows(entries, count=16, seed=17)
    second = select_frozen_windows(list(reversed(entries)), count=16, seed=17)

    assert first == second
    assert len(first) == 16
    assert len({scene for scene, _ in first}) >= 4


def test_decision_requires_every_frozen_primary_gate() -> None:
    thresholds = AuditDecisionThresholds()
    passing = {
        "window_count": 32,
        "certified_coverage_macro": 0.20,
        "nonzero_window_fraction": 0.90,
        "geometry_error_median_m": 0.10,
        "geometry_error_p90_m": 0.30,
        "matched_confidence_median_error_m": 0.20,
        "permutation_max_point_delta_m": 0.0,
        "shuffled_median_error_m": 1.0,
        "risk_error_spearman": 0.3,
    }

    assert decide_audit(passing, thresholds)["status"] == "GO_TO_STATE_INTEGRATION"
    failures = {
        "certified_coverage_macro": 0.10,
        "nonzero_window_fraction": 0.50,
        "geometry_error_median_m": 0.30,
        "geometry_error_p90_m": 0.60,
        "matched_confidence_median_error_m": 0.05,
        "permutation_max_point_delta_m": 0.01,
        "shuffled_median_error_m": 0.11,
        "risk_error_spearman": -0.1,
    }
    for key, value in failures.items():
        failing = dict(passing)
        failing[key] = value
        assert decide_audit(failing, thresholds)["status"] == "STOP_CERTIFICATE_FRONTEND"


def test_markdown_report_discloses_nonformal_research_and_council_status(tmp_path: Path) -> None:
    report = render_markdown_report(
        {"status": "INCONCLUSIVE", "gates": {}},
        {"window_count": 1},
        config_path=tmp_path / "audit.yaml",
        artifact_paths=[],
        started_at="2026-08-12T00:00:00+08:00",
        finished_at="2026-08-12T00:01:00+08:00",
    )

    assert "Council not convened" in report
    assert "not invoked" in report
    assert "INCONCLUSIVE" in report
    assert "未写入项目或全局记忆" in report
