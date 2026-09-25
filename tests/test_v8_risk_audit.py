import json
from pathlib import Path

import numpy as np
import pytest

from mcss.v8_risk_audit import (
    build_anchor_risk_fields,
    deterministic_source_derangement,
    evaluate_selective_risk,
    midrank_percentile,
    paired_bootstrap_mean_ci,
    read_v8_risk_cache,
    write_v8_risk_cache,
)


def _points(depth: np.ndarray) -> np.ndarray:
    zeros = np.zeros_like(depth)
    return np.stack((zeros, zeros, depth), axis=-1).astype(np.float32)


def test_midrank_percentile_is_stable_and_tie_aware() -> None:
    values = np.asarray([4.0, 1.0, 1.0, 8.0], dtype=np.float32)
    valid = np.asarray([True, True, True, False])

    ranked = midrank_percentile(values, valid)

    assert ranked[0] == pytest.approx(5.0 / 6.0)
    assert ranked[1] == pytest.approx(1.0 / 3.0)
    assert ranked[2] == pytest.approx(1.0 / 3.0)
    assert np.isnan(ranked[3])


def test_anchor_risk_fields_keep_emvsnet_as_the_action_point() -> None:
    full_depth = np.asarray([[1.0, 1.0, 1.0, 1.0]], dtype=np.float32)
    uni_depth = np.asarray([[1.0, 1.10, 1.30, 1.0]], dtype=np.float32)
    drops = [
        _points(np.asarray([[1.0, 1.0, 1.0, 1.40]], dtype=np.float32)),
        _points(np.asarray([[1.0, 1.0, 1.0, 1.45]], dtype=np.float32)),
        _points(np.asarray([[1.0, 1.0, 1.0, 1.35]], dtype=np.float32)),
    ]
    valid = np.ones_like(full_depth, dtype=np.bool_)

    fields = build_anchor_risk_fields(
        emvsnet_points=_points(full_depth),
        unidepth_points=_points(uni_depth),
        emvsnet_native_risk=np.asarray([[0.1, 0.2, 0.3, 0.4]], dtype=np.float32),
        unidepth_native_risk=np.asarray([[0.1, 0.2, 0.3, 0.4]], dtype=np.float32),
        emvsnet_valid=valid,
        unidepth_valid=valid,
        drop_points=drops,
        drop_native_risks=[np.full_like(full_depth, value) for value in (0.1, 0.2, 0.3)],
        drop_valids=[valid, valid, valid],
        cell_width_m=0.25,
    )

    np.testing.assert_array_equal(fields["comparison_valid"], valid)
    np.testing.assert_allclose(fields["action_points_m"], _points(full_depth))
    assert fields["disagreement_raw"][0, 2] == pytest.approx(1.2)
    assert fields["drop_instability_raw"][0, 3] == pytest.approx(1.6)
    assert fields["type_code"][0, 0] == 1  # model-supported
    assert fields["type_code"][0, 2] == 3  # conflict
    assert fields["type_code"][0, 3] == 2  # uncertain
    expected = np.maximum.reduce(
        [
            fields["emvsnet_native_rank"],
            fields["unidepth_native_rank"],
            fields["disagreement_risk"],
            fields["drop_instability_risk"],
        ]
    )
    np.testing.assert_allclose(fields["full_risk"], expected)


def test_drop_invalidity_is_risk_evidence_not_a_reason_to_delete_the_pixel() -> None:
    depth = np.asarray([[1.0, 1.0, 1.0, 1.0]], dtype=np.float32)
    supplier_valid = np.ones_like(depth, dtype=np.bool_)
    first_drop_valid = supplier_valid.copy()
    first_drop_valid[0, 1] = False

    fields = build_anchor_risk_fields(
        emvsnet_points=_points(depth),
        unidepth_points=_points(depth),
        emvsnet_native_risk=np.asarray([[0.1, 0.2, 0.3, 0.4]], dtype=np.float32),
        unidepth_native_risk=np.asarray([[0.1, 0.2, 0.3, 0.4]], dtype=np.float32),
        emvsnet_valid=supplier_valid,
        unidepth_valid=supplier_valid,
        drop_points=[_points(depth), _points(depth), _points(depth)],
        drop_native_risks=[np.full_like(depth, value) for value in (0.1, 0.2, 0.3)],
        drop_valids=[first_drop_valid, supplier_valid, supplier_valid],
        cell_width_m=0.25,
    )

    assert fields["comparison_valid"][0, 1]
    assert not fields["all_drop_valid"][0, 1]
    assert fields["drop_instability_risk"][0, 1] == pytest.approx(1.0)
    assert fields["full_risk"][0, 1] == pytest.approx(1.0)
    assert fields["type_code"][0, 1] == 2  # uncertain


def test_selective_metrics_reward_correct_error_ordering() -> None:
    errors = np.asarray([0.05, 0.10, 0.40, 0.80], dtype=np.float64)
    valid = np.ones(4, dtype=np.bool_)
    good = evaluate_selective_risk(errors, errors, valid, coverages=(0.5, 1.0))
    bad = evaluate_selective_risk(errors[::-1], errors, valid, coverages=(0.5, 1.0))

    assert good["aurc"] < bad["aurc"]
    assert good["ause"] == pytest.approx(0.0)
    assert good["spearman"] == pytest.approx(1.0)
    assert bad["spearman"] == pytest.approx(-1.0)
    assert good["risk_curve"][0] == pytest.approx(0.075)
    assert good["aurc"] == pytest.approx(
        np.mean([0.05, 0.075, (0.05 + 0.10 + 0.40) / 3.0, 0.3375])
    )
    assert good["aurc_definition"] == "empirical_mean_over_all_retained_counts"


def test_source_derangement_never_keeps_a_source_in_place() -> None:
    first = deterministic_source_derangement("scene", (1, 2, 3, 4))
    second = deterministic_source_derangement("scene", (1, 2, 3, 4))

    assert first == second
    assert sorted(first) == [1, 2, 3]
    assert all(index != value for index, value in enumerate(first, start=1))


def test_paired_bootstrap_reports_window_level_difference() -> None:
    result = paired_bootstrap_mean_ci(
        np.asarray([-0.4, -0.2, -0.3]),
        replicates=1000,
        seed=7,
    )

    assert result["sample_count"] == 3
    assert result["mean"] == pytest.approx(-0.3)
    assert result["ci95"][1] < 0.0


def test_risk_cache_is_label_free_sealed_and_overwrite_refusing(tmp_path: Path) -> None:
    fields = {
        "comparison_valid": np.ones((2, 3), dtype=np.bool_),
        "full_risk": np.arange(6, dtype=np.float32).reshape(2, 3),
        "type_code": np.ones((2, 3), dtype=np.uint8),
    }
    destination = tmp_path / "risk"

    write_v8_risk_cache(
        destination,
        fields,
        metadata={"scene_id": "scene", "label_access": False},
    )
    loaded, metadata = read_v8_risk_cache(destination)

    np.testing.assert_array_equal(loaded["full_risk"], fields["full_risk"])
    assert metadata["sealed"] is True
    assert metadata["label_access"] is False
    assert "error" not in json.dumps(metadata).lower()
    with pytest.raises(FileExistsError, match="overwrite"):
        write_v8_risk_cache(destination, fields, metadata={"label_access": False})
