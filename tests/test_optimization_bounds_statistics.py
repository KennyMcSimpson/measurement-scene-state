"""Synthetic-only tests for frozen budget, bounds and readiness decisions."""

import copy
import importlib.util
import json
from pathlib import Path

import pytest

from mcss.mechanism_pilot.direct_capacity_statistics import METRICS, describe
from mcss.mechanism_pilot.optimization_bounds_statistics import (
    CURRENT,
    GTFREE,
    SELECTIONS,
    analyze,
    bounds_status,
    group_key,
    optimization_status,
    plateau,
)


def rules():
    return {
        "budgets": [1000, 3000, 10000],
        "primary": {
            "track": "RGBD",
            "bounds_mode": GTFREE,
            "budget": 10000,
            "selection": "FIXED_BUDGET",
        },
        "statistics": {"draws": 10000, "seed": 20260927},
        "plateau": {
            "window_fraction": 0.1,
            "objective_relative_tolerance": 0.001,
            "gradient_relative_tolerance": 0.1,
            "minimum_gradient_points": 4,
            "plateau_scene_fraction": 0.75,
            "saturation_equivalence_margin": 0.02,
        },
        "readiness": {
            "maximum_top1_positive_share": 0.25,
            "maximum_generalization_mean": 0.2,
            "maximum_generalization_scene": 0.35,
            "generalization_scene_fraction": 0.75,
        },
    }


def fixture():
    config = rules()
    ids = ["scene_a", "scene_b", "scene_c", "scene_d"]
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
    query, context, oracle, baseline, trajectories = [], [], [], [], {}
    for scene in ids:
        for method in ("A", "B", "anchor", "prior"):
            for fid in (8, 9):
                baseline.append(
                    {
                        "scene_id": scene,
                        "method": method,
                        "query_id": fid,
                        **{m: 0.55 for m in METRICS},
                    }
                )
        for track in ("RGBD", "QUERY_ORACLE"):
            for bounds in (CURRENT, GTFREE):
                roles = (
                    ["joint_query"] if track == "QUERY_ORACLE" and bounds == CURRENT else ["A", "B"]
                )
                for role in roles:
                    spec = {
                        "scene_id": scene,
                        "track": track,
                        "bounds_mode": bounds,
                        "role": role,
                        "grid": 16,
                        "samples": 64,
                    }
                    chain_key = f"{scene}|{track}|{bounds}|{role}"
                    trajectory = {
                        "spec": spec,
                        "checkpoints": [
                            {"step": s, "context_objective": 0.1} for s in range(0, 10001, 100)
                        ],
                        "trace": [{"step": s, "grad_norm": 1.0} for s in range(0, 10001, 10)],
                        "budget_summaries": {},
                    }
                    for budget in config["budgets"]:
                        trajectory["budget_summaries"][str(budget)] = {}
                        for selection in SELECTIONS:
                            error = {1000: 0.4, 3000: 0.3, 10000: 0.295}[budget] - (
                                0.06 if bounds == GTFREE else 0
                            )
                            if track == "QUERY_ORACLE":
                                error = 0.07
                            context_error = {1000: 0.2, 3000: 0.11, 10000: 0.1}[budget]
                            trajectory["budget_summaries"][str(budget)][selection] = {
                                "selected_full_objective": {"context_objective": context_error},
                                "cumulative_optimization_seconds": budget / 1000,
                                "peak_cuda_allocated_bytes": 1024,
                                "peak_cuda_reserved_bytes": 2048,
                            }
                            for fid in (8, 9):
                                row = {
                                    **spec,
                                    "budget": budget,
                                    "selection": selection,
                                    "method": "direct",
                                    "query_id": fid,
                                    **{m: error for m in METRICS},
                                }
                                (oracle if track == "QUERY_ORACLE" else query).append(row)
                                if track == "RGBD":
                                    query.append(
                                        {
                                            **row,
                                            "method": "wrong_scene",
                                            **{m: error + 0.1 for m in METRICS},
                                        }
                                    )
                            if track == "RGBD":
                                for fid in [0, 1] if role == "A" else [0, 2]:
                                    context.append(
                                        {
                                            **spec,
                                            "budget": budget,
                                            "selection": selection,
                                            "method": "direct",
                                            "frame_id": fid,
                                            **{m: context_error for m in METRICS},
                                        }
                                    )
                    trajectories[chain_key] = trajectory
    return query, context, oracle, baseline, trajectories, manifest, config


def point(value):
    return describe({s: value for s in ("a", "b", "c", "d")})


def transition(query_gain, context_gain=0.01, objective_gain=0.01):
    return {
        "query_gain": point(query_gain),
        "context_absrel_gain": point(context_gain),
        "context_objective_gain": point(objective_gain),
    }


def test_plateau_requires_flat_objective_and_gradient_and_enough_points():
    checks = [{"step": 900, "context_objective": 1}, {"step": 1000, "context_objective": 0.9995}]
    trace = [{"step": s, "grad_norm": 1} for s in (900, 920, 960, 1000)]
    assert plateau(checks, trace, 1000, rules()["plateau"])["status"] == "PLATEAU"
    trace[-1]["grad_norm"] = 0.1
    trace[-2]["grad_norm"] = 0.1
    assert plateau(checks, trace, 1000, rules()["plateau"])["status"] == "NOT_PLATEAU"
    assert plateau(checks, trace[:3], 1000, rules()["plateau"])["status"].startswith("UNKNOWN")
    checks[-1]["context_objective"] = 1.5
    assert plateau(checks, trace, 1000, rules()["plateau"])["status"] == "NOT_PLATEAU"


@pytest.mark.parametrize(
    "transitions,fraction,expected",
    [
        ([transition(0.1), transition(-0.03)], 1, "QUERY_OVERFIT"),
        ([transition(0.1), transition(0.03)], 0, "UNDEROPTIMIZED"),
        ([transition(0.1), transition(0.001)], 1, "SATURATED"),
        ([transition(0.1), transition(0.001)], 0, "PARTIALLY_SATURATED"),
        ([transition(0.001), transition(0.001)], 0, "INCONCLUSIVE"),
    ],
)
def test_status_priority_is_predeclared(transitions, fraction, expected):
    assert optimization_status(transitions, fraction, rules()["plateau"]) == expected


def test_bounds_interaction_precedes_final_winner():
    assert bounds_status([point(0.01), point(0.08)], point(0.07)) == "INTERACTION_WITH_OPTIMIZATION"
    assert bounds_status([point(0.06), point(0.06)], point(0)) == "GT_FREE_BETTER"
    assert bounds_status([point(-0.06), point(-0.06)], point(0)) == "CURRENT_BETTER"
    assert bounds_status([point(0.001), point(0.001)], point(0)) == "NO_CLEAR_DIFFERENCE"


def test_raw_reproduction_scene_unit_primary_and_readiness():
    args = fixture()
    result = analyze(*args, draws=100)
    assert result == analyze(*json.loads(json.dumps(args)), draws=100)
    primary = result["optimization_sufficiency"]["primary"]
    assert primary["OPTIMIZATION_STATUS"] == "SATURATED"
    assert primary["CARRIER_TRAINING_READINESS"] == "READY"
    assert result["optimization_results"]["primary_key"] == group_key("RGBD", GTFREE, 10000)
    assert all(k.startswith("QUERY_ORACLE|") for k in result["query_oracle_diagnostic"]["groups"])
    gain = result["bootstrap_results"]["comparisons"][
        f"query_gain|RGBD|{GTFREE}|FIXED_BUDGET|first_to_max"
    ]
    assert gain["n_scenes"] == 4 and gain["mean"] == pytest.approx(0.105)
    assert gain["ci95"] == pytest.approx([0.105, 0.105])
    assert gain["improved_tied_worse"] == [4, 0, 0]
    # 4 scenes * (4 RGBD chains + 3 oracle chains) * max 10 sec each.
    assert result["cost_analysis"]["actual_trajectory_optimization_seconds"] == 280


@pytest.mark.parametrize(
    "fault",
    ["query", "context", "baseline", "extra_budget", "query_primary", "trajectory", "protected"],
)
def test_incomplete_or_unregistered_raw_fails_closed(fault):
    args = list(copy.deepcopy(fixture()))
    if fault == "query":
        args[0].pop(0)
    elif fault == "context":
        args[1].pop(0)
    elif fault == "baseline":
        args[3].pop(0)
    elif fault == "extra_budget":
        args[0][0]["budget"] = 30000
    elif fault == "query_primary":
        args[6]["primary"]["budget"] = 3000
    elif fault == "trajectory":
        args[4].pop(next(iter(args[4])))
    else:
        args[0][0]["scene_id"] = "protected"
    with pytest.raises((ValueError, KeyError)):
        analyze(*args, draws=50)


def test_seven_scientific_figures(tmp_path):
    path = Path(__file__).resolve().parents[1] / "scripts/analyze_optimization_bounds.py"
    spec = importlib.util.spec_from_file_location("bounds_figures", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.figures(analyze(*fixture(), draws=50), tmp_path)
    assert len(list(tmp_path.glob("*.png"))) == 7
    assert len(list(tmp_path.glob("*.svg"))) == 7


def test_core_scalar_objective_and_preclip_gradient_schema():
    args = fixture()
    expected = analyze(*args, draws=50)
    for trajectory in args[4].values():
        for row in trajectory["trace"]:
            row["gradient_norm_before_clip"] = row.pop("grad_norm")
        for summaries in trajectory["budget_summaries"].values():
            for summary in summaries.values():
                summary["selected_context_metrics"] = summary["selected_full_objective"]
                summary["selected_full_objective"] = summary["selected_context_metrics"][
                    "context_objective"
                ]
                summary["context_at_budget"] = {"context_objective": 999}
    assert analyze(*args, draws=50) == expected


def test_rgbd_secondary_cannot_change_primary_decision():
    args = list(fixture())
    primary = analyze(*args, draws=50)["optimization_sufficiency"]["primary"]
    extra_query, extra_context, extra_trajectories = [], [], {}
    for source, output in ((args[0], extra_query), (args[1], extra_context)):
        for row in source:
            if row["bounds_mode"] == GTFREE and row["budget"] == 10000:
                output.append({**row, "track": "RGB_ONLY", **{m: 100 for m in METRICS}})
    for key, trajectory in args[4].items():
        if trajectory["spec"]["track"] == "RGBD" and trajectory["spec"]["bounds_mode"] == GTFREE:
            extra_trajectories[key + "secondary"] = {
                **trajectory,
                "spec": {**trajectory["spec"], "track": "RGB_ONLY"},
                "budget_summaries": {"10000": trajectory["budget_summaries"]["10000"]},
            }
    args[0] += extra_query
    args[1] += extra_context
    args[4].update(extra_trajectories)
    result = analyze(*args, draws=50)
    assert result["optimization_sufficiency"]["primary"] == primary
    assert result["optimization_results"]["secondary"]["RGB_ONLY_minus_RGBD"]["mean"] > 90


def test_cli_plain_and_compressed_archive_exact_reproduction(tmp_path):
    import gzip

    path = Path(__file__).resolve().parents[1] / "scripts/analyze_optimization_bounds.py"
    spec = importlib.util.spec_from_file_location("bounds_cli", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    query, context, oracle, baseline, trajectories, manifest, config = fixture()
    config["statistics"]["draws"] = 100
    chains = [
        {
            **t["spec"],
            "key": k,
            "phase": "oracle" if t["spec"]["track"] == "QUERY_ORACLE" else "context",
            "optimization_path": f"checkpoints/{k}",
        }
        for k, t in trajectories.items()
    ]
    files = {
        "scene_manifest.json": manifest,
        "decision_rules.json": config,
        "state_plan.json": {"chains": chains},
        "baseline_results.json": baseline,
        "raw/context_results.json": query,
        "raw/context_context_results.json": context,
        "raw/oracle_results.json": oracle,
        "raw/trajectory_summaries.json": trajectories,
    }
    for name, value in files.items():
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(value))
    first = module.run(tmp_path, tmp_path / "plain_reproduction")
    for name in files:
        target = tmp_path / name
        with gzip.open(Path(str(target) + ".gz"), "wt") as stream:
            stream.write(target.read_text())
        target.unlink()
    second = module.run(tmp_path, tmp_path / "archive_reproduction")
    assert first == second
    provenance = json.loads(
        (tmp_path / "archive_reproduction/statistics_reproduction.json").read_text()
    )
    assert all(p.endswith(".gz") for p in provenance["inputs_sha256"])
    marker = json.loads(
        (tmp_path / "archive_reproduction/primary_analysis_complete.json").read_text()
    )
    assert marker["status"] == "PASS" and len(marker["primary_decision_sha256"]) == 64
    assert marker["primary_decision"] == second["optimization_sufficiency"]["primary"]


def test_readiness_limited_status_and_missing_scene_specificity():
    from mcss.mechanism_pilot.optimization_bounds_statistics import readiness

    arguments = (
        point(0.30),
        point(0.10),
        point(0.10),
        point(0.10),
        "SATURATED",
        point(0.001),
        rules()["readiness"],
    )
    assert readiness(*arguments)["CARRIER_TRAINING_READINESS"] == "READY_WITH_LIMITATION"
    missing_wrongscene = list(arguments)
    missing_wrongscene[2] = point(0)
    result = readiness(*missing_wrongscene)
    assert result["CARRIER_TRAINING_READINESS"] == "PROMISING_BUT_NOT_READY"
    assert not result["MATCHED_CARRIER_TRAINING_RECOMMENDED"]
    no_gain = list(arguments)
    no_gain[1] = point(-0.1)
    assert readiness(*no_gain)["CARRIER_TRAINING_READINESS"] == "NOT_READY"
