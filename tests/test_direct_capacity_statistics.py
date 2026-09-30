"""Independent synthetic raw checks for scene-level capacity attribution."""

import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from mcss.mechanism_pilot.direct_capacity_statistics import (
    ALTERNATIVE,
    CURRENT,
    METRICS,
    analyze,
    describe,
    group_key,
    paired,
    scene_values,
    stable_gain,
)


def fixture():
    baseline, query, context, oracle = [], [], [], []
    scenes = ["scene_a", "scene_b", "scene_c", "scene_d"]
    for scene_index, scene in enumerate(scenes):
        extra = scene_index * 0.001
        for method in ("A", "B", "anchor", "prior"):
            for fid in (8, 9):
                baseline.append(
                    {
                        "scene_id": scene,
                        "method": method,
                        "query_id": fid,
                        **{m: 0.5 + extra for m in METRICS},
                    }
                )
        configurations = []
        for track in ("RGBD", "RGB_ONLY", "QUERY_ORACLE"):
            configurations += [(track, grid, 64, CURRENT) for grid in (8, 16, 32)]
            configurations.append((track, 8, 64, ALTERNATIVE))
        configurations += [("RGBD", grid, 128, CURRENT) for grid in (8, 16)]
        for track, grid, samples, bounds in configurations:
            roles = (
                ("joint_query",) if track == "QUERY_ORACLE" and bounds == CURRENT else ("A", "B")
            )
            for role in roles:
                spec = {
                    "scene_id": scene,
                    "track": track,
                    "grid": grid,
                    "samples": samples,
                    "bounds_mode": bounds,
                    "role": role,
                    "method": "direct",
                }
                value = (0.05 if track == "QUERY_ORACLE" else 0.15) + extra
                for fid in (8, 9):
                    row = {**spec, "query_id": fid, **{m: value for m in METRICS}}
                    (oracle if track == "QUERY_ORACLE" else query).append(row)
                    if (
                        track != "QUERY_ORACLE"
                        and grid == 8
                        and samples == 64
                        and bounds == CURRENT
                    ):
                        for method in (
                            "wrong_scene",
                            "spatial_shuffle",
                            "density_only",
                            "color_only",
                            "zero_density_color",
                        ):
                            query.append(
                                {**row, "method": method, **{m: value + 0.1 for m in METRICS}}
                            )
                if track != "QUERY_ORACLE":
                    for fid in (0, 3, 5):
                        context.append(
                            {**spec, "frame_id": fid, **{m: value - 0.05 for m in METRICS}}
                        )
    return baseline, query, context, oracle, {}, scenes


def test_scene_unit_not_query_or_role_count():
    rows = [{"scene_id": "a", "role": "A", "x": 0}] * 9
    rows += [{"scene_id": "a", "role": "B", "x": 2}, {"scene_id": "b", "role": "A", "x": 3}]
    values = scene_values(rows, "x")
    assert values == {"a": 1, "b": 3}
    stats = describe(values)
    index = np.random.default_rng(20260927).integers(0, 2, (10000, 2))
    assert stats["ci95"] == np.percentile(np.array([1, 3])[index].mean(1), [2.5, 97.5]).tolist()
    assert stats["mean"] == stats["median"] == 2
    assert stats["loso"] == {"a": 3, "b": 1}


def test_paired_scene_resampling_preserves_covariance():
    left = {"a": 100, "b": 200, "c": 300}
    right = {"a": 99, "b": 199, "c": 299}
    result = paired(left, right)
    assert result["ci95"] == [1, 1]
    assert result["improved_tied_worse"] == [3, 0, 0]
    assert result["positive_top1_contribution"] == pytest.approx(1 / 3)
    assert result["positive_top3_contribution"] == 1
    with pytest.raises(ValueError, match="identical"):
        paired(left, {"a": 99})


def test_builder_raw_exact_reproduction_and_oracle_separation():
    args = fixture()
    first = analyze(*args, draws=200)
    second = analyze(*json.loads(json.dumps(args)), draws=200)
    assert first == second
    results = first["direct_state_results"]
    assert all(k.startswith("QUERY_ORACLE|") for k in results["diagnostic_oracle"])
    assert all(not k.startswith("QUERY_ORACLE|") for k in results["context_only"])
    comparisons = first["capacity_analysis"]["comparisons"]
    assert comparisons["capacity_gap_RGBD_g8"]["mean"] == pytest.approx(0.35)
    assert comparisons["direct_minus_carrier_RGBD_g8"]["mean"] == pytest.approx(-0.35)
    assert comparisons["control_damage_RGBD_wrong_scene"]["mean"] == pytest.approx(0.1)
    assert comparisons["resolution_RGBD_8_to_16"]["mean"] == 0
    group = results["context_only"][group_key("RGBD")]
    assert set(METRICS) <= set(group["metrics"])
    assert group["generalization_gap"]["depth_absrel"]["mean"] == pytest.approx(0.05)
    classification = first["capacity_analysis"]["classification"]
    assert classification["STATE_CAPACITY_STATUS"] == "MIXED"
    assert classification["LEARNER_TRAINING_BOTTLENECK"]
    assert classification["CONTEXT_INFERENCE_BOTTLENECK"]
    assert not classification["RESOLUTION_BOTTLENECK"]
    assert classification["REPRESENTATION_BOTTLENECK"] == "NOT_ESTABLISHED_BY_FINITE_OPTIMIZATION"


@pytest.mark.parametrize(
    "fault", ["duplicate", "missing_query", "foreign_scene", "oracle_mix", "missing_metric"]
)
def test_incomplete_or_contaminated_raw_fails_closed(fault):
    args = list(copy.deepcopy(fixture()))
    if fault == "duplicate":
        args[1].append(args[1][0])
    elif fault == "missing_query":
        args[1].pop(0)
    elif fault == "foreign_scene":
        args[1][0]["scene_id"] = "protected_scene"
    elif fault == "oracle_mix":
        args[1][0]["track"] = "QUERY_ORACLE"
    else:
        args[1][0]["depth_absrel"] = None
    with pytest.raises((ValueError, KeyError)):
        analyze(*args, draws=100)


def test_preregistered_stability_requires_all_three_conditions():
    assert stable_gain(describe({"a": 0.1, "b": 0.1, "c": 0.1, "d": 0.1}))
    assert not stable_gain(describe({"a": 0.01, "b": 0.01, "c": 0.01, "d": 0.01}))
    assert not stable_gain(describe({"a": 0.4, "b": 0.1, "c": -0.2, "d": -0.2}))


def test_six_scientific_figures_use_complete_fixture(tmp_path):
    path = Path(__file__).resolve().parents[1] / "scripts/analyze_direct_capacity.py"
    spec = importlib.util.spec_from_file_location("analyze_direct_capacity", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.figures(analyze(*fixture(), draws=100), tmp_path)
    assert len(list(tmp_path.glob("*.png"))) == 6
    assert len(list(tmp_path.glob("*.svg"))) == 6


def locked_manifest(scenes):
    return {
        "FINAL_HOLDOUT_PROHIBITED": ["protected_scene"],
        "scenes": [
            {
                "scene_id": s,
                "roles": {"primary_query": [8, 9], "context_a": [0, 3, 5], "context_b": [0, 3, 5]},
            }
            for s in scenes
        ],
    }


def test_planned_state_cost_curves_and_role_completeness():
    args = list(fixture())
    specs = {}
    summaries = {}
    for rows, phase in ((args[1], "context"), (args[3], "oracle")):
        for row in rows:
            key = "|".join(
                str(row[k]) for k in ("scene_id", "role", "track", "grid", "samples", "bounds_mode")
            )
            row["key"] = key
            if row["method"] != "direct":
                continue
            specs[key] = {
                k: row[k]
                for k in ("key", "scene_id", "role", "track", "grid", "samples", "bounds_mode")
            }
            specs[key]["phase"] = phase
            summaries[key] = {
                "seconds": 2.0,
                "parameter_count": 4 * row["grid"] ** 3,
                "peak_cuda_allocated_bytes": 1024,
                "peak_cuda_reserved_bytes": 2048,
                "final_selected_at_budget_boundary": True,
                "full_objective_checkpoints": [
                    {"step": 0, "context_objective": 1.0},
                    {"step": 1, "context_objective": 0.5},
                    {"step": 2, "context_objective": 0.25},
                ],
            }
    for row in args[2]:
        row["key"] = "|".join(
            str(row[k]) for k in ("scene_id", "role", "track", "grid", "samples", "bounds_mode")
        )
    args[4] = summaries
    manifest = locked_manifest(args[5])
    plan = {"states": list(specs.values())}
    result = analyze(*args, plan=plan, manifest=manifest, draws=50)
    cost = result["direct_state_results"]["context_only"][group_key("RGBD")]["optimization_cost"]
    assert cost["n_states"] == 8 and cost["seconds_total"] == 16
    assert cost["parameter_count_per_state"] == [2048]
    assert cost["selected_at_budget_boundary"] == 8
    assert all(
        v["last3_objective_slope_per_step"] == pytest.approx(-0.375)
        for v in result["optimization_curves"].values()
    )
    args[4].pop(next(iter(args[4])))
    with pytest.raises(ValueError, match="summary"):
        analyze(*args, plan=plan, manifest=manifest, draws=50)


@pytest.mark.parametrize(
    "fault",
    [
        "baseline_query",
        "baseline_role",
        "baseline_unknown",
        "context_frame",
        "context_role",
        "context_state",
        "wrong_manifest_query",
        "wrong_context_frame",
    ],
)
def test_manifest_roles_and_frames_fail_closed(fault):
    args = list(fixture())
    manifest = locked_manifest(args[5])
    if fault == "baseline_query":
        args[0].pop(0)
    elif fault == "baseline_role":
        args[0] = [
            r for r in args[0] if not (r["scene_id"] == "scene_a" and r["method"] == "prior")
        ]
    elif fault == "baseline_unknown":
        args[0][0]["method"] = "unexpected"
    elif fault == "context_frame":
        args[2].pop(0)
    elif fault == "context_role":
        args[2] = [r for r in args[2] if r["role"] != "A"]
    elif fault == "context_state":
        args[2] = [r for r in args[2] if r["scene_id"] != "scene_a"]
    elif fault == "wrong_manifest_query":
        manifest["scenes"][0]["roles"]["primary_query"] = [12, 13]
    else:
        args[2][0]["frame_id"] = 999
    with pytest.raises(ValueError):
        analyze(*args, manifest=manifest, draws=50)


def test_manifest_complete_raw_roundtrip_exact():
    args = fixture()
    manifest = locked_manifest(args[-1])
    expected = analyze(*args, manifest=manifest, draws=50)
    assert expected == analyze(*json.loads(json.dumps(args)), manifest=manifest, draws=50)


def analysis_script():
    path = Path(__file__).resolve().parents[1] / "scripts/analyze_direct_capacity.py"
    spec = importlib.util.spec_from_file_location("analyze_direct_capacity_io", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_archived_json_loader_prefers_plain_and_records_actual_path(tmp_path):
    import gzip

    module = analysis_script()
    plain = tmp_path / "raw.json"
    compressed = tmp_path / "raw.json.gz"
    with gzip.open(compressed, "wt") as stream:
        json.dump({"source": "compressed"}, stream)
    assert module.load_json(plain) == ({"source": "compressed"}, compressed)
    plain.write_text(json.dumps({"source": "plain"}))
    assert module.load_json(plain) == ({"source": "plain"}, plain)
    plain.unlink()
    compressed.unlink()
    with pytest.raises(FileNotFoundError):
        module.load_json(plain)


def test_aggregate_summary_loader_exact_keys_fallback_and_individual_precedence(tmp_path):
    import gzip

    module = analysis_script()
    plan = {
        "states": [
            {"key": f"state_{i}", "optimization_path": f"checkpoints/state_{i}"} for i in range(425)
        ]
    }
    expected = {s["key"]: {"value": i} for i, s in enumerate(plan["states"])}
    archive = tmp_path / "raw/optimization_summaries.json.gz"
    archive.parent.mkdir()

    def write_archive(value):
        with gzip.open(archive, "wt") as stream:
            json.dump(value, stream)

    write_archive(expected)
    summaries, paths = module.load_summaries(tmp_path, plan)
    assert summaries == expected and paths == [archive]
    # Partial individual availability must not create a mixed-source result.
    individual = tmp_path / "checkpoints/state_0/summary.json"
    individual.parent.mkdir(parents=True)
    individual.write_text('{"value": "individual"}')
    assert module.load_summaries(tmp_path, plan) == (expected, [archive])
    for malformed in ({}, {**expected, "unexpected": {}}, {"summaries": expected}):
        write_archive(malformed)
        with pytest.raises(ValueError, match="exactly match"):
            module.load_summaries(tmp_path, plan)
    # All individual files exist: malformed unused archive is never read.
    for state in plan["states"]:
        path = tmp_path / state["optimization_path"] / "summary.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"individual": state["key"]}))
    summaries, paths = module.load_summaries(tmp_path, plan)
    assert set(summaries) == set(expected) and len(paths) == 425
    assert archive not in paths and all(path.is_file() for path in paths)
    assert summaries["state_0"] == {"individual": "state_0"}
