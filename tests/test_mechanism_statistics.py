"""Analytic fixtures test weighting, counterfactual baselines, and GT masking."""
import numpy as np
import pytest

from mcss.mechanism_pilot.statistics import (
    ACTIONS,
    analyze_history,
    measurement_metrics,
    paired_scene_bootstrap,
    scene_macro,
)


def rows_for(scene, gains_fc, gains_cf=None, continuation="next"):
    gains_cf = gains_fc if gains_cf is None else gains_cf
    return [
        {"scene_id": scene, "continuation_id": continuation, "history": history,
         "action": action, "depth_absrel": 2 - gain}
        for history, gains in (("FC", gains_fc), ("CF", gains_cf))
        for action, gain in zip(ACTIONS, gains, strict=True)
    ]


def test_depth_metrics_never_filter_by_predicted_opacity():
    rgb = np.ones((4, 4, 3)) * .4
    truth = np.full((4, 4), 2.)
    pred = np.ones((4, 4))
    pred[0, 0] = 0
    truth[0, 1] = np.nan
    metrics = measurement_metrics(rgb, rgb, pred, truth, np.zeros((4, 4)))
    assert metrics["depth_absrel"] == pytest.approx(8 / 15)
    assert metrics["depth_rmse"] == pytest.approx(np.sqrt(18 / 15))
    assert metrics["depth_delta1"] == 0
    assert metrics["valid_depth_fraction"] == 15 / 16
    assert metrics["coverage"] == 0
    assert metrics["rgb_psnr_perfect"]
    assert metrics["rgb_ssim"] == pytest.approx(1)
    pred[1, 1] = np.nan
    with pytest.raises(ValueError, match="Nonfinite prediction"):
        measurement_metrics(rgb, rgb, pred, truth, np.ones((4, 4)))


def test_psnr_and_common_region_mask_are_explicit():
    target = np.zeros((4, 4, 3))
    pred = np.full_like(target, .1)
    depth = np.ones((4, 4))
    mask = np.zeros((4, 4), dtype=bool)
    mask[:2] = True
    metrics = measurement_metrics(pred, target, depth, depth, depth, region_mask=mask)
    assert metrics["rgb_psnr"] == pytest.approx(20)
    assert metrics["region_pixel_count"] == 8
    assert metrics["depth_delta1"] == 1


def test_macro_and_bootstrap_resample_scenes_not_pixels_queries_or_continuations():
    macro = scene_macro([{"scene_id": "one", "score": 1.}] * 1000
                        + [{"scene_id": "two", "score": 0.}], "score")
    assert macro["mean"] == .5
    bootstrap = paired_scene_bootstrap(macro["per_scene"], draws=1000)
    assert bootstrap["ci95"] == [0., 1.]
    many = sum((rows_for("one", [0, 1, 0, 0], continuation=str(i))
                for i in range(31)), [])
    analysis = analyze_history(many + rows_for("two", [0, 0, 0, 0]), draws=100)
    assert analysis["write_oracle_gain"] == .5
    assert analysis["bootstrap"]["unit"] == "scene"


def test_global_action_is_reoptimized_in_every_bootstrap_and_loso_subset():
    rows = rows_for("one", [0, 1, 0, 0]) + rows_for("two", [0, 0, 1, 0])
    result = analyze_history(rows, draws=1000)
    assert result["global_action_gap"] == .5
    assert result["bootstrap"]["metrics"]["global_action_gap"]["ci95"] == [0., .5]
    assert result["loso"]["one"]["best_fixed_action"] == "COMPLETE"
    assert result["loso"]["two"]["best_fixed_action"] == "FUSE"
    assert all(x["global_action_gap"] == 0 for x in result["loso"].values())


def test_real_history_flip_and_observation_only_oracle_gap():
    result = analyze_history(rows_for("one", [0, 1, 0, 0], [0, 0, 1, 0]), draws=50)
    assert result["beneficial_action_flip_count"] == 1
    assert result["scenes_with_beneficial_flip"] == 1
    assert result["state_dependent_action_gap"] == .5
    assert result["gamma"]["FUSE__COMPLETE"]["mean"] == 2
    assert result["per_continuation"][0]["optimal_action_sets"] == {
        "FC": ["FUSE"], "CF": ["COMPLETE"]}


def test_harmful_rank_swap_is_not_a_beneficial_flip_and_off_participates():
    result = analyze_history(rows_for("one", [0, -.1, -.2, -.3],
                                     [0, -.2, -.1, -.3]), draws=50)
    assert result["beneficial_action_flip_count"] == 0
    assert result["write_oracle_gain"] == 0
    assert result["best_fixed_action"] == "OFF"
    assert all(x["mean"] == 1 for x in result["harmful_write_prevalence"].values())


def test_near_ties_and_overlapping_winner_sets_do_not_manufacture_flips():
    result = analyze_history(rows_for("one", [0, 1, 1 - 1e-9, 0],
                                     [0, 1 - 1e-9, 1, 0]), draws=50)
    assert result["beneficial_action_flip_count"] == 0
    assert result["per_continuation"][0]["optimal_action_sets"]["FC"] == [
        "FUSE", "COMPLETE"]


def test_queries_are_averaged_with_identical_query_sets_per_candidate():
    rows = rows_for("one", [0, 1, 0, 0])
    raw = [{**row, "query_id": "a"} for row in rows]
    raw += [{**row, "query_id": "b", "depth_absrel": 2.} for row in rows]
    assert analyze_history(raw, draws=10)["write_oracle_gain"] == .5
    with pytest.raises(ValueError, match="Incomplete"):
        analyze_history(raw[:-1], draws=10)


@pytest.mark.parametrize("mutation", ["duplicate", "missing", "nan", "unknown"])
def test_rejects_invalid_or_unmatched_history_rows(mutation):
    rows = rows_for("one", [0, 1, 0, 0])
    if mutation == "duplicate":
        rows.append(rows[0].copy())
    elif mutation == "missing":
        rows.pop()
    elif mutation == "nan":
        rows[0]["depth_absrel"] = float("nan")
    else:
        rows[0]["history"] = "OTHER"
    with pytest.raises(ValueError):
        analyze_history(rows, draws=10)
