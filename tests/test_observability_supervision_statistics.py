"""Region-empty policy and frozen supervision attribution, using only synthetic raw."""

import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from mcss.mechanism_pilot.direct_capacity_statistics import METRICS, describe
from mcss.mechanism_pilot.observability_supervision_statistics import (
    PARTITION,
    REGIONS,
    VARIANTS,
    analyze,
    conditional_difference,
    conditional_summary,
    gap_fraction,
    geometry_status,
    observability_status,
    positive_low_support_share,
)


def rules():
    return {
        "primary_variant": "S3",
        "statistics": {"draws": 10000, "seed": 20260927},
        "region_rules": {"minimum_scene_coverage": 0.75},
        "observability_rules": {
            "observed_gap_min": 0.02,
            "equivalence_margin": 0.02,
            "mixed_low_support_positive_share_min": 0.5,
            "unobserved_positive_share_min": 0.75,
        },
    }


def fixture():
    ids = ["a", "b", "c", "d"]
    manifest = {
        "FINAL_HOLDOUT_PROHIBITED": ["protected"],
        "scenes": [
            {
                "scene_id": s,
                "roles": {"primary_query": [8, 9], "context_a": [0, 1], "context_b": [0, 2]},
            }
            for s in ids
        ],
    }
    query, context, regions, geometry, trajectories = [], [], [], [], {}
    errors = {
        "S0": 0.4,
        "S1": 0.35,
        "S2": 0.32,
        "S3": 0.25,
        "ORACLE": 0.1,
        "CARRIER": 0.6,
        "ANCHOR": 0.5,
    }
    for scene in ids:
        for role in ("A", "B"):
            for variant, error in errors.items():
                for q in (8, 9):
                    common = {
                        "scene_id": scene,
                        "role": role,
                        "variant": variant,
                        "method": "direct",
                        "query_id": q,
                    }
                    query.append({**common, **{m: error for m in METRICS}})
                    controls = ["wrong_scene"] if variant in VARIANTS else []
                    if variant == "S3":
                        controls += ["spatial_shuffle", "zero"]
                    for control in controls:
                        query.append(
                            {**common, "method": control, **{m: error + 0.1 for m in METRICS}}
                        )
                    counts = {
                        "OVERALL": 10,
                        "OBS0": 8 if scene == "d" else 4,
                        "OBS1": 2,
                        "OBS2PLUS": 0 if scene == "d" else 4,
                        "OBS01": 10 if scene == "d" else 6,
                        "ANGLE_0_5": 0 if scene == "d" else 4,
                        "ANGLE_5_15": 0,
                        "ANGLE_15_30": 0,
                        "ANGLE_GT30": 0,
                        "OCCLUDED_ANY": 1,
                        "CONFLICT_ANY": 0,
                    }
                    for region in REGIONS:
                        n = counts[region]
                        regions.append(
                            {
                                **common,
                                "region": region,
                                "pixel_count": n,
                                "total_valid": 10,
                                "absrel_sum": n * error,
                                "absrel_mean": error if n else None,
                            }
                        )
                if variant in VARIANTS:
                    for frame in [0, 1] if role == "A" else [0, 2]:
                        context.append(
                            {
                                "scene_id": scene,
                                "role": role,
                                "variant": variant,
                                "method": "direct",
                                "frame_id": frame,
                                **{m: error / 2 for m in METRICS},
                            }
                        )
                    key = f"{scene}|{role}|{variant}"
                    trajectories[key] = {
                        "spec": {"scene_id": scene, "role": role, "variant": variant},
                        "summary": {"seconds": 1.0},
                    }
            for q in (8, 9):
                geometry.append(
                    {
                        "scene_id": scene,
                        "role": role,
                        "query_id": q,
                        "spearman_visible_count_absrel": -0.4 if scene != "d" else None,
                    }
                )
    return query, context, regions, geometry, trajectories, manifest, rules()


def test_empty_scenes_retained_and_bootstrap_recomputes_coverage_denominator():
    values = {"a": 1.0, "b": None, "c": 3.0, "d": None}
    result = conditional_summary(values)
    assert result["per_scene"] == values
    assert result["n_scenes"] == 4 and result["n_covered_scenes"] == 2
    assert result["empty_mask_scene_count"] == 2 and result["mean"] == 2
    indices = np.random.default_rng(20260927).integers(0, 4, (10000, 4))
    numerator = np.array([1, 0, 3, 0])[indices].sum(1)
    denom = np.array([1, 0, 1, 0])[indices].sum(1)
    valid = denom > 0
    expected = np.percentile(numerator[valid] / denom[valid], [2.5, 97.5]).tolist()
    assert result["ci95"] == expected
    assert result["bootstrap"]["undefined_empty_draws"] == int((~valid).sum())
    assert result["loso"]["b"] == 2
    empty = conditional_summary({"a": None, "b": None})
    assert empty["mean"] is empty["ci95"] is None
    assert empty["bootstrap"]["undefined_empty_draws"] == 10000


def test_paired_regions_use_intersection_coverage_and_no_imputed_zero():
    result = conditional_difference({"a": 1, "b": None, "c": 3}, {"a": 0.5, "b": 1, "c": None})
    assert result["per_scene"] == {"a": 0.5, "b": None, "c": None}
    assert result["mean"] == 0.5 and result["scene_coverage"] == 1 / 3


def test_contribution_partition_all_scenes_and_raw_roundtrip():
    inputs = fixture()
    result = analyze(*inputs, draws=100)
    assert result == analyze(*json.loads(json.dumps(inputs)), draws=100)
    regional = result["observability_analysis"]["regions"]["S0"]
    assert regional["OBS2PLUS"]["conditional_absrel"]["per_scene"]["d"] is None
    assert regional["OBS2PLUS"]["additive_error_contribution"]["per_scene"]["d"] == 0
    for scene in ("a", "b", "c", "d"):
        reconstructed = sum(
            regional[r]["additive_error_contribution"]["per_scene"][scene] for r in PARTITION
        )
        assert reconstructed == pytest.approx(
            regional["OVERALL"]["additive_error_contribution"]["per_scene"][scene]
        )
    decision = result["decision_summary"]
    assert decision["BEST_SUPERVISION"] == "S3_PREDECLARED_PRIMARY"
    assert decision["GEOMETRY_SUPERVISION_STATUS"] == "COMPLEMENTARY"
    assert decision["GEOMETRY_AWARE_CARRIER_RECOMMENDED"]
    assert result["gap_closure_analysis"]["variants"]["S3"]["gap_closed"]["mean"] == pytest.approx(
        0.15
    )
    assert result["gap_closure_analysis"]["variants"]["S3"]["gap_closed_fraction"][
        "value"
    ] == pytest.approx(0.5)
    assert "ORACLE|direct" not in result["supervision_analysis"]["groups"]
    assert result["cost_analysis"]["actual_optimization_seconds"] == 32


def comparisons(scores):
    return {
        f"{a}_minus_{b}": describe({s: scores[a] - scores[b] for s in ("a", "b", "c", "d")})
        for a in VARIANTS
        for b in VARIANTS
        if a != b
    }


@pytest.mark.parametrize(
    "scores,status",
    [
        ({"S0": 0.4, "S1": 0.3, "S2": 0.35, "S3": 0.5}, "HARMFUL"),
        ({"S0": 0.4, "S1": 0.3, "S2": 0.3, "S3": 0.2}, "COMPLEMENTARY"),
        ({"S0": 0.4, "S1": 0.3, "S2": 0.4, "S3": 0.3}, "FREE_SPACE_DOMINANT"),
        ({"S0": 0.4, "S1": 0.4, "S2": 0.3, "S3": 0.3}, "SURFACE_DOMINANT"),
        ({"S0": 0.4, "S1": 0.4, "S2": 0.4, "S3": 0.4}, "NO_CLEAR_GAIN"),
        ({"S0": 0.4, "S1": 0.3, "S2": 0.3, "S3": 0.3}, "INCONCLUSIVE"),
    ],
)
def test_frozen_geometry_classification(scores, status):
    assert geometry_status(comparisons(scores)) == status


def test_low_coverage_cannot_classify_and_signed_share_not_net_gain():
    gap = conditional_summary({"a": 0.1, "b": None, "c": None, "d": None})
    assert observability_status(gap, 1.0, 0.99, rules()) == "INCONCLUSIVE"
    gap = conditional_summary({s: 0.1 for s in ("a", "b", "c", "d")})
    assert observability_status(gap, 1.0, 0.6, rules()) == "MIXED"
    assert observability_status(gap, 1.0, 0.2, rules()) == "OBSERVED_REGION_FAILURE"
    zero = conditional_summary({s: 0 for s in ("a", "b", "c", "d")})
    assert observability_status(zero, 1.0, 0.8, rules()) == "UNOBSERVED_REGION_DOMINANT"
    share = positive_low_support_share(
        {"OBS0": {"a": 1, "b": -10}, "OBS1": {"a": 1, "b": 0}, "OBS2PLUS": {"a": 2, "b": 0}}
    )
    assert share["share"] == 0.5


def test_gap_fraction_recomputes_each_paired_draw_and_undefined_denominators():
    result = gap_fraction({"a": 1, "b": -1}, {"a": 0.5, "b": -0.5})
    assert result["value"] is None
    assert result["ci95"] == [0.5, 0.5]
    assert result["undefined_nonpositive_base_draws"] > 0


@pytest.mark.parametrize(
    "fault",
    [
        "missing_query",
        "missing_empty_region",
        "bad_partition",
        "changed_mask",
        "missing_context",
        "protected",
        "chosen_primary",
        "missing_state",
    ],
)
def test_incomplete_or_adaptive_inputs_fail_closed(fault):
    inputs = list(copy.deepcopy(fixture()))
    if fault == "missing_query":
        inputs[0].pop(0)
    elif fault == "missing_empty_region":
        inputs[2] = [
            r
            for i, r in enumerate(inputs[2])
            if i != next(j for j, x in enumerate(inputs[2]) if x["pixel_count"] == 0)
        ]
    elif fault == "bad_partition":
        inputs[2][1]["absrel_sum"] += 1
    elif fault == "changed_mask":
        inputs[2][1]["pixel_count"] += 1
    elif fault == "missing_context":
        inputs[1].pop(0)
    elif fault == "protected":
        inputs[0][0]["scene_id"] = "protected"
    elif fault == "chosen_primary":
        inputs[6]["primary_variant"] = "S1"
    else:
        inputs[4].pop(next(iter(inputs[4])))
    with pytest.raises((ValueError, KeyError)):
        analyze(*inputs, draws=50)


def analysis_script():
    path = Path(__file__).resolve().parents[1] / "scripts/analyze_observability_supervision.py"
    spec = importlib.util.spec_from_file_location("observability_analysis", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_seven_figures_include_overall_and_all_scenes(tmp_path):
    module = analysis_script()
    module.figures(analyze(*fixture(), draws=50), tmp_path)
    assert len(list(tmp_path.glob("*.png"))) == 7
    assert len(list(tmp_path.glob("*.svg"))) == 7


def test_cli_references_marker_and_compressed_archives_exact(tmp_path):
    import gzip

    module = analysis_script()
    query, context, regions, geometry, trajectories, manifest, config = fixture()
    config["statistics"]["draws"] = 100
    files = {
        "audit/observability_integrity.json": {"status": "PASS"},
        "raw/query_results.json": [r for r in query if r["variant"] in VARIANTS],
        "raw/reference_results.json": [r for r in query if r["variant"] not in VARIANTS],
        "raw/context_results.json": context,
        "raw/region_results.json": regions,
        "raw/observability_summary.json": geometry,
        "raw/optimization_summaries.json": trajectories,
        "scene_manifest.json": manifest,
        "decision_rules.json": config,
    }
    for name, value in files.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
    plain = module.run(tmp_path, tmp_path / "plain")
    for name in files:
        path = tmp_path / name
        with gzip.open(Path(str(path) + ".gz"), "wt") as stream:
            stream.write(path.read_text())
        path.unlink()
    archive = module.run(tmp_path, tmp_path / "archive")
    assert plain == archive
    provenance = json.loads((tmp_path / "archive/statistics_reproduction.json").read_text())
    assert all(path.endswith(".gz") for path in provenance["input_sha256"])
    (tmp_path / "audit/observability_integrity.json").write_text('{"status":"FAIL"}')
    with pytest.raises(PermissionError):
        module.run(tmp_path, tmp_path / "denied")


def test_any_undefined_query_correlation_makes_entire_scene_null():
    inputs = fixture()
    inputs[3][0]["spearman_visible_count_absrel"] = None
    result = analyze(*inputs, draws=50)
    association = result["observability_analysis"]["S0_visible_count_error_spearman"]
    assert association["per_scene"]["a"] is None
    assert association["scene_coverage"] == 0.5
    assert result["decision_summary"]["VISIBILITY_AWARE_FUSION_RECOMMENDED"] == "inconclusive"
