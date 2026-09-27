import csv
import itertools
import json

import pytest

from mcss.vision_probe.opportunity_analysis import analyze, opportunity_summary


def synthetic():
    rows = []
    for split in ("discovery", "validation"):
        for sequence in ("one", "two"):
            for slot in (1, 2):
                for path in itertools.product(("OFF", "A", "B", "ALL"), repeat=2):
                    gain = (0.02 if slot == 1 else 0.08) if path == ("OFF", "A") else 0.0
                    selected = ["best_fixed", "visible_selector"] if path == ("OFF", "A") else []
                    if path == ("OFF", "OFF"):
                        selected.append("rgb_selector")
                    rows.append(
                        {
                            "resolution": 448,
                            "split": split,
                            "sequence": sequence,
                            "target_slot": slot,
                            "trajectory": list(path),
                            "J": 0.4 + gain,
                            "F": 0.3,
                            "JF": 0.35 + gain / 2,
                            "token_iou": 0.2,
                            "visible": {"rgb_decrease": gain},
                            "selected_methods": selected,
                            "objects": [
                                {
                                    "object_id": 1,
                                    "J": 0.4 + gain,
                                    "F": 0.3,
                                    "JF": 0.35 + gain / 2,
                                    "size_bucket": "small",
                                }
                            ],
                        }
                    )
    return rows


def test_every_target_and_primary_j():
    report = opportunity_summary(synthetic())["448/validation"]
    stat = report["oracle_minus_off"]
    assert stat["sequence_bootstrap"]["mean"] == pytest.approx(0.05)
    assert stat["improved_pairs"] == 4
    assert stat["all_loso_ci_lower_positive"] is True
    assert set(stat["leave_one_sequence_out_ci"]) == {"one", "two"}
    for result in stat["leave_one_sequence_out_ci"].values():
        assert result["ci95"] == pytest.approx([0.05, 0.05])
        assert result["positive_sequence_count"] == 1
        assert result["top1_positive_gain_share"] == 1.0
        assert result["directional_ci_positive"] is True
    assert report["trajectory_minus_constant"]["sequence_bootstrap"]["mean"] == pytest.approx(0.05)


def test_reject_missing_slot_and_duplicate_trajectory():
    rows = synthetic()
    with pytest.raises(ValueError, match="Both target"):
        opportunity_summary([r for r in rows if r["target_slot"] == 1])
    with pytest.raises(ValueError, match="Incomplete or duplicate"):
        opportunity_summary(rows + [rows[0]])


def test_reproducibility_capture_and_outputs(tmp_path):
    raw = tmp_path / "raw.jsonl"
    raw.write_text("\n".join(json.dumps(r) for r in synthetic()))
    a = analyze(raw, tmp_path / "a", {"primary_metric": "J"})
    b = analyze(raw, tmp_path / "b", {"primary_metric": "J"})
    assert a == b
    selected = a["selector_analysis"]["448/validation/visible_selector"]
    assert selected["opportunity_capture"] == pytest.approx(1.0)
    assert selected["net_gain"] == pytest.approx(0.05)
    assert selected["beneficial_write_precision"] == 1.0
    with (tmp_path / "a/per_sequence_results.csv").open() as f:
        fixed = [r for r in csv.DictReader(f) if r["method"] == "best_fixed"]
    assert all(float(r["J"]) == pytest.approx(0.45) for r in fixed)
    assert (tmp_path / "a/signal_scatter.csv").exists()
    with (tmp_path / "a/visible_features.csv").open() as f:
        visible = list(csv.DictReader(f))
    assert len(visible) == len(synthetic())
    assert set(visible[0]) == {
        "resolution",
        "split",
        "sequence",
        "target_slot",
        "trajectory",
        "rgb_decrease",
    }
    assert {r["target_slot"] for r in visible} == {"1", "2"}
    assert (tmp_path / "a/visible_features.csv").read_bytes() == (
        tmp_path / "b/visible_features.csv"
    ).read_bytes()


def test_capture_is_positive_gain_not_net_gain(tmp_path):
    rows = synthetic()
    for row in rows:
        if row["trajectory"] == ["B", "B"]:
            row["J"] = 0.3
            if row["target_slot"] == 2:
                row["selected_methods"].append("visible_selector")
        if row["target_slot"] == 2 and row["trajectory"] == ["OFF", "A"]:
            row["selected_methods"].remove("visible_selector")
    raw = tmp_path / "raw.jsonl"
    raw.write_text("\n".join(json.dumps(r) for r in rows))
    selected = analyze(raw, tmp_path / "out", {})["selector_analysis"][
        "448/validation/visible_selector"
    ]
    assert selected["opportunity_capture"] == pytest.approx(0.2)
    assert selected["net_gain"] < 0
    assert selected["H2"] == "NOT_ESTABLISHED"
