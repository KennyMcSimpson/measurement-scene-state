"""Paired scene-level attribution from tiny synthetic raw rows, never model output."""

from copy import deepcopy

import pytest

from mcss.mechanism_pilot.support_redesign_statistics import analyze


def rows_fixture():
    rows = []
    for sid, cohort, nqueries, base in [
        ("s0", "exposed", 4, 1.0),
        ("s1", "new_dev", 1, 2.0),
        ("s2", "new_dev", 2, 3.0),
    ]:
        for design in ["R0", "ORACLE_VOLUME_SUPPORT"]:
            for query in range(nqueries):
                for method in ["A", "B", "anchor", "prior", "wrong_scene"]:
                    error = (
                        base
                        + {"A": 0.0, "B": 0.0, "anchor": 0.5, "prior": 1.0, "wrong_scene": 1.0}[
                            method
                        ]
                    )
                    if design != "R0" and method in ("A", "B"):
                        error -= 0.25
                    rows.append(
                        {
                            "scene_id": sid,
                            "cohort": cohort,
                            "query_id": query,
                            "design": design,
                            "method": method,
                            "depth_absrel": error,
                            "rgb_mse": 0.1,
                            "rgb_ssim": 0.5,
                            "depth_rmse": 1.0,
                            "depth_delta1": 0.5,
                            "opacity": 0.8,
                            "coverage": 0.9,
                            "inside_fraction": 1.0,
                            "two_view_candidate_fraction": 1.0,
                            "supported_surface_fraction": 0.8,
                            "candidate_count": 128,
                            "volume_extent": [12.0, 8.0, 12.0],
                        }
                    )
    return rows


def test_queries_then_scenes_not_cohort_equal_or_query_equal():
    result = analyze(rows_fixture())
    pooled = result["pooled"]
    assert pooled["n_scenes"] == 3
    assert pooled["designs"]["R0"]["methods"]["A"]["depth_absrel"]["mean"] == 2.0
    assert (
        result["cohorts"]["new_dev"]["designs"]["R0"]["methods"]["A"]["depth_absrel"]["mean"] == 2.5
    )
    gain = pooled["designs"]["ORACLE_VOLUME_SUPPORT"]["full_context_gain"]
    assert gain["mean"] == 0.75 and gain["ci95"] == [0.75, 0.75]
    assert gain["draws"] == 10000 and gain["seed"] == 20260927 and gain["unit"] == "scene"
    assert set(gain["loso"].values()) == {0.75}
    assert (
        pooled["designs"]["ORACLE_VOLUME_SUPPORT"]["change_full_context_gain_vs_R0"]["mean"] == 0.25
    )
    assert pooled["oracle_attribution_status"] == "ORACLE_RECOVERY_GT_FREE_UNTESTED"


def test_concentration_and_constant_correlation():
    result = analyze(rows_fixture())["pooled"]["designs"]["R0"]
    gain = result["full_context_gain"]
    assert gain["improved_tied_worse"]["improved"] == 3
    assert gain["concentration"]["positive"]["top1_fraction"] == pytest.approx(1 / 3)
    assert gain["concentration"]["absolute"]["top3_fraction"] == 1.0
    corr = result["support_gain_correlation"]
    assert corr["interpretation"] == "DESCRIPTIVE_ONLY_NOT_CAUSAL"
    assert all(r["pearson"] is None for r in corr["by_support_metric"].values())


def test_adequacy_blocks_claim_that_insufficient_oracle_disproves_support():
    rows = rows_fixture()
    for row in rows:
        if row["design"] == "ORACLE_VOLUME_SUPPORT":
            row["inside_fraction"] = 0.8
    result = analyze(rows)["pooled"]
    assert result["oracle_attribution_status"] == "INCONCLUSIVE"
    assert not result["designs"]["ORACLE_VOLUME_SUPPORT"]["oracle_adequacy"]["adequate"]


def test_adequate_oracle_negative_gain_is_not_sufficient():
    rows = rows_fixture()
    for row in rows:
        if row["design"] == "ORACLE_VOLUME_SUPPORT" and row["method"] in ("A", "B"):
            row["depth_absrel"] += 2.0
    result = analyze(rows)["pooled"]
    assert result["oracle_attribution_status"] == "NOT_SUFFICIENT"
    gain = result["designs"]["ORACLE_VOLUME_SUPPORT"]["full_context_gain"]
    assert gain["improved_tied_worse"]["worse"] == 3
    assert gain["concentration"]["positive"]["top1_fraction"] is None


@pytest.mark.parametrize(
    "bad",
    ["duplicate", "missingmethod", "differentqueries", "holdout", "candidatebudget", "nonfinite"],
)
def test_invalid_or_unpaired_raw_fails_closed(bad):
    rows = rows_fixture()
    if bad == "duplicate":
        rows.append(deepcopy(rows[0]))
    elif bad == "missingmethod":
        rows.pop()
    elif bad == "differentqueries":
        rows[-1]["query_id"] = 100
    elif bad == "holdout":
        rows[0]["cohort"] = "holdout"
    elif bad == "candidatebudget":
        rows[0]["candidate_count"] = 129
    else:
        rows[0]["depth_absrel"] = float("nan")
    with pytest.raises((ValueError, PermissionError)):
        analyze(rows)


def test_preregistered_tie_tolerance_is_one_e_minus_eight():
    rows = rows_fixture()
    for row in rows:
        if row["method"] == "anchor":
            row["depth_absrel"] -= 0.5 - 5e-9
    counts = analyze(rows)["pooled"]["designs"]["R0"]["full_context_gain"]["improved_tied_worse"]
    assert counts["tie_absolute_tolerance"] == 1e-8
    assert counts["tied"] == 3 and counts["improved"] == 0
