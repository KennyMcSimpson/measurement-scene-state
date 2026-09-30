"""Freeze the 16-grid experiment before any formal optimization or new query evaluation."""

import argparse
import datetime
import json
from dataclasses import asdict
from pathlib import Path

from mcss.mechanism_pilot.direct_capacity_optimization import DirectOptimizationConfig
from mcss.mechanism_pilot.small_training import sha, write_json


def freeze(root):
    root = Path(root)
    if any((root / n).exists() for n in ["config.json", "preregistration.json", "state_plan.json"]):
        raise FileExistsError("Formal protocol already exists; no silent re-registration")
    manifest = json.loads((root / "scene_manifest.json").read_text())
    bounds = json.loads((root / "bounds_lock.json").read_text())["bounds"]
    chains, states = [], []
    for phase, track in [
        ("context", "RGBD"),
        ("oracle", "QUERY_ORACLE"),
        ("secondary", "RGB_ONLY"),
    ]:
        modes = (
            ["FROZEN_GT_FREE_BOUNDS"]
            if phase == "secondary"
            else ["CURRENT_BOUNDS", "FROZEN_GT_FREE_BOUNDS"]
        )
        for mode in modes:
            for scene in manifest["scenes"]:
                sid = scene["scene_id"]
                roles = (
                    ["joint_query"]
                    if phase == "oracle" and mode == "CURRENT_BOUNDS"
                    else ["A", "B"]
                )
                for role in roles:
                    geometry = bounds[f"{sid}/{'A' if role == 'joint_query' else role}"][mode]
                    key = f"{sid}__{role}__{track}__{mode}"
                    budgets = [10000] if phase == "secondary" else [1000, 3000, 10000]
                    chain = {
                        "key": key,
                        "scene_id": sid,
                        "phase": phase,
                        "role": role,
                        "track": track,
                        "grid": 16,
                        "samples": 64,
                        "bounds_mode": mode,
                        "bounds": geometry,
                        "budgets": budgets,
                        "optimization_path": f"checkpoints/{key}",
                    }
                    chains.append(chain)
                    for budget in budgets:
                        for selection in ["FIXED_BUDGET", "CONTEXT_SELECTED"]:
                            directory = f"checkpoints/{key}/budget_{budget}/{selection}"
                            states.append(
                                {
                                    k: v
                                    for k, v in {
                                        **chain,
                                        "key": f"{key}__b{budget}__{selection}",
                                        "chain_key": key,
                                        "budget": budget,
                                        "selection": selection,
                                        "optimization_path": directory,
                                        "state_path": directory + "/state.pt",
                                    }.items()
                                    if k != "budgets"
                                }
                            )
    write_json(root / "state_plan.json", {"chains": chains, "states": states})
    sources = [
        "src/mcss/mechanism_pilot/budget_state_optimization.py",
        "src/mcss/mechanism_pilot/optimization_bounds_contracts.py",
        "src/mcss/mechanism_pilot/optimization_bounds_evaluation.py",
        "src/mcss/mechanism_pilot/direct_capacity_contracts.py",
        "src/mcss/mechanism_pilot/direct_capacity_optimization.py",
        "src/mcss/mechanism_pilot/direct_capacity_evaluation.py",
        "src/mcss/dynamic/types.py",
        "src/mcss/types.py",
        "src/mcss/geometry.py",
        "src/mcss/measurements.py",
        "scripts/run_optimization_bounds.py",
    ]
    config = {
        "optimization": asdict(DirectOptimizationConfig(steps=10000)),
        "parallel_workers": 4,
        "source_sha256": {p: sha(Path(p)) for p in sources},
        "state_plan_sha256": sha(root / "state_plan.json"),
        "scene_manifest_sha256": sha(root / "scene_manifest.json"),
        "primary": {
            "track": "RGBD",
            "bounds_mode": "FROZEN_GT_FREE_BOUNDS",
            "budget": 10000,
            "selection": "FIXED_BUDGET",
        },
        "seed_rule": "20260927 + first8hex sha256(scene|role|track); same as prior experiment",
        "no_query_choice_of_budget_or_state": True,
    }
    write_json(root / "config.json", config)
    write_json(
        root / "optimization_budget_lock.json",
        {
            "budgets": [1000, 3000, 10000],
            "not_run": {"30000": "PREDECLARED_COST_LIMIT"},
            "cost_basis": (
                "Previous RGBD16 approximately3.9 seconds/1000steps; "
                "119 main+oracle trajectories x30 gives13923 seconds=3.87h serial, "
                "excluding I/O/evaluation/secondary"
            ),
            "shared_prefix_trajectory": True,
            "independent_states_per_bounds_scene_context": True,
            "checkpoints_saved_fixed": [0, 100, 300, 1000, 3000, 10000],
            "full_objective_every": 100,
            "trace_every": 10,
            "checkpoint_selection": (
                "earliest minimum full supervision objective among checkpoints <= budget"
            ),
            "primary_selection": "FIXED_BUDGET",
            "secondary_selection": "CONTEXT_SELECTED",
            "plateau": {
                "window_fraction": 0.10,
                "relative_objective_abs_change_max": 0.001,
                "gradient_first_last_half_median_relative_abs_change_max": 0.10,
                "minimum_gradient_trace_points": 4,
                "scene_requires_both_contexts": True,
                "aggregate_scene_fraction": 0.75,
                "never_early_stops": True,
            },
            "optimizer_loss_unchanged": config["optimization"],
            "resource_execution": (
                "Four independent subprocesses shareGPU; per-process time/memory and "
                "wholephase elapsed reported separately; not an exclusive-hardware benchmark"
            ),
            "secondary_RGB_ONLY": (
                "Only after main analysis, fixed GTfree10k config; "
                "single budget, cannot change maindecision"
            ),
        },
    )
    locked = [
        "config.json",
        "state_plan.json",
        "scene_manifest.json",
        "optimization_budget_lock.json",
        "bounds_lock.json",
        "bounds_contract.json",
        "decision_rules.json",
    ]
    write_json(
        root / "preregistration.json",
        {
            "experiment": root.name,
            "created_utc": datetime.datetime.now(datetime.UTC).isoformat(),
            "stage": "FROZEN_BEFORE_FORMAL_OPTIMIZATION_AND_NEW_QUERY_SCORES",
            "locked_file_sha256": {p: sha(root / p) for p in locked},
            "all_context_budgets_sealed_before_any_new_query_evaluation": True,
            "oracle_follows_complete_context_evaluation": True,
            "primary_is_predeclared_not_hindsight_query_best": True,
            "geometry_vs_optimization_only": True,
            "data_role": "CAPACITY_DEV_EXPOSED_ATTRIBUTION_ONLY",
            "bounds_two_only": True,
            "grid": 16,
            "samples": 64,
            "counts": {
                phase: {
                    "chains": sum(c["phase"] == phase for c in chains),
                    "states": sum(s["phase"] == phase for s in states),
                }
                for phase in ["context", "oracle", "secondary"]
            },
            "FINAL_HOLDOUT_TOUCHED": False,
            "NEW_CARRIER_TRAINED": False,
            "DYNAMIC_TTT_RUN": False,
        },
    )
    return len(chains), len(states)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    print(freeze(args.root))
