"""Post-hoc readout/reference re-analysis: readouts, metric parity, constants, labels."""

import numpy as np
import pytest

from mcss.mechanism_pilot.readout_reanalysis import (
    READOUTS,
    REFERENCES,
    SOURCES,
    VARIANTS,
    absrel_optimal_constant,
    analyze,
    analyze_with_sensitivity,
    apply_readout,
    depth_metrics,
    reference_values,
    score_predictions,
    score_references,
    validate_matrix,
)
from mcss.mechanism_pilot.statistics import measurement_metrics


def sample(seed=0, shape=(12, 15)):
    rng = np.random.default_rng(seed)
    gt = rng.uniform(0.5, 8.0, shape).astype(np.float32)
    gt[0, :4] = 0.0  # invalid GT pixels are excluded exactly as in the frozen metric
    depth = (gt * rng.uniform(0.4, 1.3, shape)).astype(np.float32)
    opacity = rng.uniform(0.2, 1.0, shape).astype(np.float32)
    return depth, opacity, gt


def test_raw_readout_reproduces_frozen_measurement_metrics():
    depth, opacity, gt = sample()
    rgb = np.full((*gt.shape, 3), 0.5, dtype=np.float32)
    frozen = measurement_metrics(rgb, rgb, depth, gt, opacity)
    ours = depth_metrics(apply_readout("RAW", depth, opacity, gt), gt)
    assert ours["depth_absrel"] == pytest.approx(frozen["depth_absrel"], abs=1e-9)
    assert ours["depth_delta1"] == pytest.approx(frozen["depth_delta1"], abs=1e-12)


def test_normalized_and_median_scaled_readouts():
    depth, opacity, gt = sample(1)
    opacity[1, 1] = 0.0
    normalized = apply_readout("OPACITY_NORMALIZED", depth, opacity, gt)
    assert normalized[2, 2] == pytest.approx(depth[2, 2] / opacity[2, 2])
    assert np.isfinite(normalized).all() and normalized[1, 1] == pytest.approx(depth[1, 1] / 1e-6)
    scaled = apply_readout("MEDIAN_SCALED", depth, opacity, gt)
    valid = gt > 0
    expected = np.median(gt[valid].astype(np.float64))
    assert np.median(scaled[valid]) == pytest.approx(expected, rel=1e-9)
    with pytest.raises(ValueError):
        apply_readout("UNKNOWN", depth, opacity, gt)


def test_absrel_optimal_constant_beats_every_other_constant():
    rng = np.random.default_rng(3)
    t = np.exp(rng.normal(1.2, 0.6, 5000))
    best = absrel_optimal_constant(t)
    grid = np.linspace(t.min(), t.max(), 4000)
    losses = np.abs(grid[:, None] - t[None]) / t[None]
    assert np.mean(np.abs(best - t) / t) <= losses.mean(1).min() + 1e-9
    assert best < np.median(t)  # the 1/t weighting favours nearer depths
    values = reference_values([t.reshape(50, 100), np.zeros(3)])
    assert values["train_pixel_count"] == 5000
    assert values["REF_TRAIN_MEDIAN"] == pytest.approx(np.median(t))


def matrix(carrier_error, c1_gain, anchor_penalty=0.1):
    scenes = [f"s{i}" for i in range(8)]
    queries = {s: [14, 15] for s in scenes}
    rows, refs = [], []
    for i, scene in enumerate(scenes):
        for role in ("A", "B"):
            for query in queries[scene]:
                for readout in READOUTS:
                    for ref in REFERENCES:
                        refs.append(
                            {
                                "reference": ref,
                                "scene_id": scene,
                                "role": role,
                                "query_id": query,
                                "readout": readout,
                                "depth_absrel": 0.5,
                                "depth_delta1": 0.4,
                            }
                        )
                    for source in SOURCES:
                        for variant in VARIANTS:
                            for method in ("direct", "anchor"):
                                for seed in (1, 2):
                                    e = 0.5 + carrier_error + 0.001 * i
                                    e -= c1_gain if variant == "C1" else 0.0
                                    e += anchor_penalty if method == "anchor" else 0.0
                                    rows.append(
                                        {
                                            "source": source,
                                            "variant": variant,
                                            "method": method,
                                            "seed": seed,
                                            "scene_id": scene,
                                            "role": role,
                                            "query_id": query,
                                            "readout": readout,
                                            "depth_absrel": e,
                                            "depth_delta1": 0.9 - e,
                                        }
                                    )
    return rows, refs, scenes, queries


def test_labels_follow_sign_conventions():
    rows, refs, scenes, queries = matrix(carrier_error=-0.2, c1_gain=0.05)
    validate_matrix(rows, refs, [1, 2], scenes, queries)
    result = analyze(rows, refs, draws=200)
    entry = result["carriers"]["V3_C1"]["RAW"]["depth_absrel"]
    assert entry["reference_label"] == "ABOVE" and entry["vs_REF_TRAIN_MEDIAN"]["mean"] > 0
    assert result["primary_contrasts"]["V4"]["by_readout"]["RAW"]["depth_absrel"]["label"] == (
        "SUPPORTED"
    )
    assert result["static"]["V2_C0"]["RAW"]["depth_delta1"]["label"] == "ABOVE"
    assert result["summary"]["CARRIER_REFERENCE_STATUS"] == (
        "SOME_CARRIER_ABOVE_GEOMETRY_FREE_REFERENCE"
    )
    worse, refs, _, _ = matrix(carrier_error=0.2, c1_gain=-0.05)
    result = analyze(worse, refs, draws=200)
    assert result["carriers"]["V2_C1"]["RAW"]["depth_absrel"]["reference_label"] == "BELOW"
    assert result["summary"]["CARRIER_REFERENCE_STATUS"] == (
        "NO_CARRIER_ABOVE_GEOMETRY_FREE_REFERENCE"
    )
    assert result["primary_contrasts"]["V2"]["by_readout"]["RAW"]["depth_absrel"]["label"] == (
        "NOT_ESTABLISHED"
    )


def test_scoring_reproduces_sealed_metrics_and_rejects_drift(tmp_path):
    depth, opacity, gt = sample(5)
    path = tmp_path / "prediction.npz"
    np.savez_compressed(path, depth=depth, opacity=opacity)
    sealed = depth_metrics(depth, gt)
    record = {
        "source": "V5",
        "variant": "C1",
        "seed": 1,
        "scene_id": "s",
        "role": "A",
        "query_id": 14,
        "method": "direct",
        "step": 5,
        "path": str(path),
        "sealed_depth_absrel": sealed["depth_absrel"],
        "sealed_depth_delta1": sealed["depth_delta1"],
        "sealed_depth_valid_count": int((gt > 0).sum()),
    }
    rows, worst = score_predictions([record], {("s", 14): gt})
    assert [r["readout"] for r in rows] == list(READOUTS) and worst["absrel"] < 1e-12
    drifted = {**record, "sealed_depth_absrel": sealed["depth_absrel"] + 1e-3}
    with pytest.raises(RuntimeError):
        score_predictions([drifted], {("s", 14): gt})
    values = {"REF_TRAIN_ABSREL_OPTIMAL": 2.0, "REF_TRAIN_MEDIAN": 4.0}
    refs = score_references(values, {("s", 14): gt}, {"s": [14]})
    assert len(refs) == len(REFERENCES) * len(READOUTS) * 2
    scaled = [r for r in refs if r["readout"] == "MEDIAN_SCALED"]
    assert len({round(r["depth_absrel"], 12) for r in scaled}) == 1


def test_no_hit_views_stay_unscaled_and_feed_the_sensitivity(tmp_path):
    _, _, gt = sample(6)
    zero = np.zeros_like(gt)
    assert np.array_equal(apply_readout("MEDIAN_SCALED", zero, zero, gt), zero)
    path = tmp_path / "nohit.npz"
    np.savez_compressed(path, depth=zero, opacity=zero)
    sealed = depth_metrics(zero, gt)
    record = {
        "source": "V4",
        "variant": "C0",
        "seed": 1,
        "scene_id": "s0",
        "role": "A",
        "query_id": 14,
        "method": "direct",
        "step": 5,
        "path": str(path),
        "sealed_depth_absrel": sealed["depth_absrel"],
        "sealed_depth_delta1": sealed["depth_delta1"],
        "sealed_depth_valid_count": int((gt > 0).sum()),
    }
    rows, worst = score_predictions([record], {("s0", 14): gt})
    assert worst["median_scale_undefined_views"] == 1
    assert all(r["all_zero_prediction"] for r in rows)
    assert {r["depth_absrel"] for r in rows} == {1.0}
    grid, refs, scenes, queries = matrix(carrier_error=-0.2, c1_gain=0.05)
    for row in grid:
        row["all_zero_prediction"] = row["scene_id"] == "s0"
    result = analyze_with_sensitivity(grid, refs, draws=100)
    sensitivity = result["sensitivity_excluding_no_hit_scenes"]
    assert sensitivity["excluded_no_hit_scenes"] == ["s0"]
    assert (
        result["carriers"]["V2_C1"]["RAW"]["depth_absrel"]["vs_REF_TRAIN_MEDIAN"]["n_scenes"] == 8
    )
    assert (
        sensitivity["carriers"]["V2_C1"]["RAW"]["depth_absrel"]["vs_REF_TRAIN_MEDIAN"]["n_scenes"]
        == 7
    )


def test_single_source_analysis():
    rows, refs, scenes, queries = matrix(carrier_error=-0.2, c1_gain=0.05)
    only = {"V4": SOURCES["V4"]}
    rows = [r for r in rows if r["source"] == "V4"]
    validate_matrix(rows, refs, [1, 2], scenes, queries, sources=only)
    result = analyze(rows, refs, sources=only, draws=100)
    assert set(result["carriers"]) == {"V4_C0", "V4_C1", "V4_C2"}
    assert result["summary"]["carrier_count"] == 3


def test_incomplete_matrix_is_rejected():
    rows, refs, scenes, queries = matrix(carrier_error=0.0, c1_gain=0.0)
    with pytest.raises(ValueError):
        validate_matrix(rows[:-1], refs, [1, 2], scenes, queries)
    with pytest.raises(ValueError):
        validate_matrix(rows, refs + refs[:1], [1, 2], scenes, queries)
