"""Saved-row reporting never retrains or invents confirmation results."""

import json

import pytest

from mcss.vision_probe.v2_reporting import analyze_saved_rows, build_report
from mcss.vision_probe.v2_statistics import TRAJECTORIES


def _raw():
    rows = []
    for sequence, gain_path in (("s1", ("A", "ALL")), ("s2", ("B", "A"))):
        for trajectory in TRAJECTORIES:
            j = 0.8 if trajectory == gain_path else 0.2
            rows.append(
                {
                    "sequence": sequence,
                    "target_slot": 1,
                    "trajectory": list(trajectory),
                    "J": j,
                    "F": j / 2,
                    "JF": 0.75 * j,
                    "objects": [
                        {
                            "object_id": 1,
                            "J": j,
                            "size_bucket": "tiny",
                            "pixel_count": 1,
                            "pixel_fraction": 0.001,
                        }
                    ],
                }
            )
    return rows


def test_report_uses_saved_choices_and_keeps_secondary_tiny_data():
    rows = _raw()
    saved = [{"sequence": s, "target_slot": 1, "trajectory": ["OFF", "OFF"]} for s in ("s1", "s2")]
    result = analyze_saved_rows(rows, {"CycleGate": saved}, ["OFF", "OFF"], draws=50)
    cycle = result["selector_analysis"]["CycleGate"]
    assert cycle["mean_J"] == 0.2  # never replace saved OFF with the answer oracle
    assert cycle["secondary"]["F"] == 0.1
    assert cycle["secondary"]["JF"] == pytest.approx(0.15)
    assert cycle["net_gain"] == 0
    assert result["global_diagnostic"]["gaps"]["sample_dependence_gap"]["mean"] == pytest.approx(
        0.3
    )
    assert result["selector_analysis"]["hindsight_global16"]["comparisons"] == {}
    tiny = result["object_size_sensitivity"]["CycleGate"]["buckets"]["tiny"]
    assert tiny["n_objects"] == 2
    assert tiny["conditional_sequence_macro_gain"] == 0
    assert len(result["per_pair_results"]) == 12
    json.dumps(result, allow_nan=False)


def test_incomplete_saved_decisions_fail_closed():
    with pytest.raises(ValueError, match="every raw pair"):
        analyze_saved_rows(_raw(), {"CycleGate": []}, None, draws=10)


def test_blocked_regeneration_has_no_confirmation_numbers_or_media_reads(tmp_path, monkeypatch):
    import mcss.vision_probe.v2_reporting as reporting

    report, work = tmp_path / "report", tmp_path / "work"
    report.mkdir()
    work.mkdir()
    (report / "config.json").write_text(json.dumps({"schema": "test.v2", "bootstrap_samples": 10}))
    (report / "independent_data_audit.json").write_text(
        json.dumps(
            {
                "status": "BLOCKED_NO_INDEPENDENT_DATA",
                "confirmation_allowed": False,
            }
        )
    )
    old = {"fixed_trajectory_provenance": {"trajectory": ["OFF", "OFF"]}, "diagnostics": {}}
    monkeypatch.setattr(reporting, "old_raw_diagnostics", lambda _: (old, {"status": "PASS"}))
    result = build_report(report, work, tmp_path / "old")
    assert result["confirmation_status"] == "BLOCKED_NO_INDEPENDENT_DATA"
    assert result["raw_rows"] == 0
    assert result["independent_test_numbers"] is None
    assert result["no_media_reads"] and result["no_training_or_inference"]
    assert json.loads((report / "selector_analysis.json").read_text()) == {}
    assert (report / "per_pair_results.csv").read_text().strip() == "status"
    first = {p.name: p.read_bytes() for p in report.iterdir()}
    assert build_report(report, work, tmp_path / "old") == result
    assert {p.name: p.read_bytes() for p in report.iterdir()} == first
