"""Raw-only scene statistics for the frozen 16-grid optimization/bounds study.

No media, state checkpoints, model, optimizer or query loader is imported.
Thresholds are supplied by the experiment's pre-formal decision lock.
"""

from collections import defaultdict

import numpy as np

from mcss.mechanism_pilot.direct_capacity_statistics import (
    METRICS,
    describe,
    engineering_good,
    paired,
    scene_values,
    stable_gain,
)

CURRENT = "CURRENT_BOUNDS"
GTFREE = "FROZEN_GT_FREE_BOUNDS"
SELECTIONS = ("FIXED_BUDGET", "CONTEXT_SELECTED")


def group_key(track, bounds, budget, selection="FIXED_BUDGET", method="direct"):
    return f"{track}|{bounds}|b{budget}|{selection}|{method}"


def row_key(row):
    return group_key(
        row["track"], row["bounds_mode"], row["budget"], row["selection"], row["method"]
    )


def reverse(stats, *, draws=10000, seed=20260927):
    return describe(
        {s: -v for s, v in stats["per_scene"].items()},
        draws=draws,
        seed=seed,
        positive_means_improvement=True,
    )


def plateau(checkpoints, trace, budget, thresholds):
    """Objective and stochastic pre-clip gradients provide limited plateau evidence."""
    lower = budget * (1 - thresholds["window_fraction"])
    checks = sorted(
        (c for c in checkpoints if lower <= c["step"] <= budget), key=lambda c: c["step"]
    )
    gradients = sorted((r for r in trace if lower <= r["step"] <= budget), key=lambda r: r["step"])
    if len(checks) < 2 or len(gradients) < thresholds["minimum_gradient_points"]:
        return {
            "status": "UNKNOWN_INSUFFICIENT_CHECKPOINTS",
            "budget": budget,
            "n_objective_points": len(checks),
            "n_gradient_points": len(gradients),
        }
    objectives = np.asarray([c["context_objective"] for c in checks], dtype=float)
    norms = np.asarray(
        [r.get("gradient_norm_before_clip", r.get("grad_norm")) for r in gradients], dtype=float
    )
    if not np.isfinite(objectives).all() or not np.isfinite(norms).all():
        raise ValueError("Plateau inputs must be finite")
    relative = float((objectives[0] - objectives[-1]) / max(abs(objectives[0]), 1e-8))
    half = len(norms) // 2
    early, late = float(np.median(norms[:half])), float(np.median(norms[half:]))
    gradient_change = (late - early) / max(abs(early), 1e-8)
    passed = (
        abs(relative) <= thresholds["objective_relative_tolerance"]
        and abs(gradient_change) <= thresholds["gradient_relative_tolerance"]
    )
    return {
        "status": "PLATEAU" if passed else "NOT_PLATEAU",
        "budget": budget,
        "objective_relative_reduction": relative,
        "gradient_relative_change": gradient_change,
        "early_gradient_median": early,
        "late_gradient_median": late,
        "n_objective_points": len(checks),
        "n_gradient_points": len(gradients),
        "not_global_optimality_proof": True,
    }


def optimization_status(transitions, plateau_fraction, thresholds):
    """Evaluate ordered adjacent transitions without selecting the best query step."""
    if not transitions:
        raise ValueError("At least one adjacent budget transition is required")
    last = transitions[-1]
    saturated = plateau_fraction >= thresholds["plateau_scene_fraction"]
    query_harm = stable_gain(reverse(last["query_gain"]))
    context_improves = (
        last["context_objective_gain"]["positive"] / last["context_objective_gain"]["n_scenes"]
        >= 0.75
    )
    if query_harm and context_improves:
        return "QUERY_OVERFIT"
    if (
        stable_gain(last["query_gain"])
        and all(t["query_gain"]["mean"] >= 0 for t in transitions)
        and not saturated
    ):
        return "UNDEROPTIMIZED"
    margin = thresholds["saturation_equivalence_margin"]
    equivalent = all(
        -margin <= last[k]["ci95"][0] and last[k]["ci95"][1] <= margin
        for k in ("query_gain", "context_absrel_gain")
    )
    if saturated and equivalent:
        return "SATURATED"
    if (
        any(stable_gain(t["query_gain"]) for t in transitions[:-1])
        and not stable_gain(last["query_gain"])
        and not query_harm
    ):
        return "PARTIALLY_SATURATED"
    return "INCONCLUSIVE"


def bounds_status(by_budget, interaction):
    first, last = by_budget[0], by_budget[-1]
    if (
        stable_gain(interaction)
        or stable_gain(reverse(interaction))
        or (stable_gain(first) and stable_gain(reverse(last)))
        or (stable_gain(reverse(first)) and stable_gain(last))
    ):
        return "INTERACTION_WITH_OPTIMIZATION"
    if stable_gain(last):
        return "GT_FREE_BETTER"
    if stable_gain(reverse(last)):
        return "CURRENT_BETTER"
    return "NO_CLEAR_DIFFERENCE"


def readiness(
    query, capacity_gap, wrong_scene_damage, generalization_gap, status, last_query_gain, thresholds
):
    values = list(generalization_gap["per_scene"].values())
    checks = {
        "optimization_saturated_or_partially": status in ("SATURATED", "PARTIALLY_SATURATED"),
        "stable_carrier_gain": stable_gain(capacity_gap),
        "scene_specific_state": wrong_scene_damage["mean"] > 0
        and wrong_scene_damage["ci95"][0] > 0,
        "not_single_scene": (
            capacity_gap["positive_top1_contribution"] is not None
            and capacity_gap["positive_top1_contribution"]
            <= thresholds["maximum_top1_positive_share"]
            and bool(capacity_gap["loso"])
            and min(capacity_gap["loso"].values()) > 0
        ),
        "bounded_generalization_gap": (
            generalization_gap["mean"] <= thresholds["maximum_generalization_mean"]
            and np.mean(np.asarray(values) <= thresholds["maximum_generalization_scene"])
            >= thresholds["generalization_scene_fraction"]
        ),
        "no_stable_late_query_harm": not stable_gain(reverse(last_query_gain)),
    }
    checks = {k: bool(v) for k, v in checks.items()}
    good = bool(engineering_good(query))
    if all(checks.values()):
        decision = "READY" if good else "READY_WITH_LIMITATION"
    elif checks["stable_carrier_gain"] or good:
        decision = "PROMISING_BUT_NOT_READY"
    else:
        decision = "NOT_READY"
    return {
        "CARRIER_TRAINING_READINESS": decision,
        "ENGINEERING_GOOD": good,
        "checks": checks,
        "MATCHED_CARRIER_TRAINING_RECOMMENDED": decision in ("READY", "READY_WITH_LIMITATION"),
        "training_executed": False,
        "independent_qualification": False,
    }


def _validate_raw(query, context, oracle, baseline, manifest, budgets):
    records = {r["scene_id"]: r for r in manifest["scenes"]}
    ids = sorted(records)
    if len(ids) != len(manifest["scenes"]) or set(ids) & set(manifest["FINAL_HOLDOUT_PROHIBITED"]):
        raise ValueError("Unique exposed roster disjoint from holdout required")
    for rows, kind in ((query, "query_id"), (oracle, "query_id"), (context, "frame_id")):
        seen, actual = set(), defaultdict(set)
        for row in rows:
            if row["scene_id"] not in records or row["grid"] != 16 or row["samples"] != 64:
                raise ValueError("Raw scene or renderer/grid differs from frozen experiment")
            if row["budget"] not in budgets or row["selection"] not in SELECTIONS:
                raise ValueError("Unregistered budget/selection")
            if row["bounds_mode"] not in (CURRENT, GTFREE):
                raise ValueError("Unregistered bounds")
            if rows is oracle and row["track"] != "QUERY_ORACLE":
                raise ValueError("Oracle block requires privileged track")
            if rows is not oracle and row["track"] not in ("RGBD", "RGB_ONLY"):
                raise ValueError("Oracle cannot enter context-only block")
            identity = (row_key(row), row["scene_id"], row["role"], row[kind])
            if identity in seen:
                raise ValueError("Duplicate raw observation")
            seen.add(identity)
            actual[identity[:3]].add(row[kind])
        for (_, scene, role), frames in actual.items():
            locked = records[scene]["roles"]
            expected = (
                locked["primary_query"]
                if kind == "query_id"
                else locked["context_a" if role == "A" else "context_b"]
            )
            if frames != set(expected):
                raise ValueError("Raw query/context frame IDs must match locked roles")
        grouped_roles = defaultdict(set)
        for key, scene, role in actual:
            grouped_roles[key, scene].add(role)
        for (key, _), roles in grouped_roles.items():
            expected = (
                {"joint_query"} if key.startswith("QUERY_ORACLE|CURRENT_BOUNDS|") else {"A", "B"}
            )
            if roles != expected:
                raise ValueError("Missing or unexpected A/B/joint-query role")
    expected = {
        (scene, method, fid)
        for scene, record in records.items()
        for method in ("A", "B", "anchor", "prior")
        for fid in record["roles"]["primary_query"]
    }
    actual = {(r["scene_id"], r["method"], r["query_id"]) for r in baseline}
    if actual != expected or len(actual) != len(baseline):
        raise ValueError("Baseline must cover every locked method/scene/query exactly")
    return ids


def analyze(
    query_rows,
    context_rows,
    oracle_rows,
    baseline_rows,
    trajectories,
    manifest,
    rules,
    *,
    draws=None,
    seed=None,
):
    """Produce raw-driven JSON artifacts without choosing a query-optimal budget."""
    budgets = rules["budgets"]
    draws = rules["statistics"]["draws"] if draws is None else draws
    seed = rules["statistics"]["seed"] if seed is None else seed
    if budgets != sorted(set(budgets)) or len(budgets) < 2:
        raise ValueError("Budgets must be frozen, ordered and unique")
    primary = rules["primary"]
    if primary != {
        "track": "RGBD",
        "bounds_mode": GTFREE,
        "budget": budgets[-1],
        "selection": "FIXED_BUDGET",
    }:
        raise ValueError("Primary must be predeclared RGBD GT-free maximum fixed budget")
    ids = _validate_raw(query_rows, context_rows, oracle_rows, baseline_rows, manifest, budgets)
    grouped, contextual = defaultdict(list), defaultdict(list)
    for row in query_rows + oracle_rows:
        grouped[row_key(row)].append(row)
    for row in context_rows:
        if row["method"] != "direct":
            raise ValueError("Context metric rows must be direct states")
        contextual[row_key(row)].append(row)
    required = {
        group_key(track, bounds, budget, selection, method)
        for track in ("RGBD", "QUERY_ORACLE")
        for bounds in (CURRENT, GTFREE)
        for budget in budgets
        for selection in SELECTIONS
        for method in (("direct", "wrong_scene") if track == "RGBD" else ("direct",))
    }
    optional = {
        group_key("RGB_ONLY", GTFREE, budgets[-1], selection, method)
        for selection in SELECTIONS
        for method in ("direct", "wrong_scene")
    }
    if not required <= set(grouped) or not set(grouped) <= required | optional:
        raise ValueError("Incomplete/unregistered main matrix or secondary configuration")
    expected_context = {
        k for k in grouped if not k.startswith("QUERY_ORACLE|") and k.endswith("|direct")
    }
    if set(contextual) != expected_context:
        raise ValueError("Every context-only direct group requires context evaluation")
    groups = {}
    for key, rows in grouped.items():
        group = {
            "spec": {
                k: rows[0][k] for k in ("track", "bounds_mode", "budget", "selection", "method")
            },
            "metrics": {},
            "n_rows": len(rows),
            "render_seconds_total": sum(r.get("seconds", 0) for r in rows),
            "render_seconds_per_query": float(np.mean([r.get("seconds", 0) for r in rows])),
        }
        available_metrics = METRICS + tuple(
            m
            for m in ("insideGTfraction", "ray_hitfraction", "GTdistance_above_far_fraction")
            if all(m in row for row in rows)
        )
        for metric in available_metrics:
            values = scene_values(rows, metric)
            if sorted(values) != ids:
                raise ValueError("Each method must cover all scenes")
            group["metrics"][metric] = describe(values, draws=draws, seed=seed)
        if key in contextual:
            group["context_metrics"] = {
                m: describe(scene_values(contextual[key], m), draws=draws, seed=seed)
                for m in METRICS
            }
            group["generalization_gap"] = paired(
                group["metrics"]["depth_absrel"]["per_scene"],
                group["context_metrics"]["depth_absrel"]["per_scene"],
                draws=draws,
                seed=seed,
                improvement=False,
            )
        groups[key] = group
    carrier_rows = [{**r, "role": r["method"]} for r in baseline_rows if r["method"] in ("A", "B")]
    carrier = {m: describe(scene_values(carrier_rows, m), draws=draws, seed=seed) for m in METRICS}
    objectives, plateaus, costs = defaultdict(list), {}, defaultdict(list)
    seen_specs = set()
    for chain_key, trajectory in sorted(trajectories.items()):
        spec = trajectory["spec"]
        identity = (spec["track"], spec["bounds_mode"], spec["scene_id"], spec["role"])
        if identity in seen_specs:
            raise ValueError("Duplicate trajectory spec")
        seen_specs.add(identity)
        checks, trace = trajectory["checkpoints"], trajectory["trace"]
        plateaus[chain_key] = {"spec": spec, "budgets": {}}
        available_budgets = sorted(int(b) for b in trajectory["budget_summaries"])
        expected_budgets = [budgets[-1]] if spec["track"] == "RGB_ONLY" else budgets
        if available_budgets != expected_budgets:
            raise ValueError("Trajectory budget summaries differ from frozen matrix")
        for budget in available_budgets:
            plateaus[chain_key]["budgets"][str(budget)] = plateau(
                checks, trace, budget, rules["plateau"]
            )
            summaries = trajectory["budget_summaries"][str(budget)]
            if set(summaries) != set(SELECTIONS):
                raise ValueError("Both checkpoint selections must be recorded")
            for selection, summary in summaries.items():
                key = group_key(spec["track"], spec["bounds_mode"], budget, selection)
                selected = summary.get(
                    "selected_context_metrics", summary["selected_full_objective"]
                )
                if not isinstance(selected, dict):
                    selected = {"context_objective": selected}
                objectives[key].append(
                    {
                        "scene_id": spec["scene_id"],
                        "role": spec["role"],
                        "objective": selected["context_objective"],
                    }
                )
                costs[key].append(summary)
    expected_specs = {
        (r["track"], r["bounds_mode"], r["scene_id"], r["role"])
        for r in query_rows + oracle_rows
        if r["method"] == "direct"
    }
    if seen_specs != expected_specs:
        raise ValueError("Trajectory summaries do not match all evaluated states")
    for key, rows in objectives.items():
        groups[key]["supervision_objective"] = describe(
            scene_values(rows, "objective"), draws=draws, seed=seed
        )
        group_cost = costs[key]
        groups[key]["cost"] = {
            "n_trajectories": len(group_cost),
            "seconds_per_state": float(
                np.mean([r["cumulative_optimization_seconds"] for r in group_cost])
            ),
            "total_cumulative_seconds": sum(
                r["cumulative_optimization_seconds"] for r in group_cost
            ),
            "peak_cuda_allocated_bytes": max(
                (r.get("peak_cuda_allocated_bytes") or 0) for r in group_cost
            ),
            "peak_cuda_reserved_bytes": max(
                (r.get("peak_cuda_reserved_bytes") or 0) for r in group_cost
            ),
            "note": "Shared trajectory cumulative cost; do not sum budgets or selection rules",
        }
    comparisons = {}

    def contrast(name, left, right, metric="depth_absrel", field="metrics"):
        value = paired(
            groups[left][field][metric]["per_scene"],
            groups[right][field][metric]["per_scene"],
            draws=draws,
            seed=seed,
        )
        value["definition"] = f"{left} minus {right}: {field}/{metric}"
        comparisons[name] = value
        return value

    transitions, statuses = {}, {}
    for track in ("RGBD", "QUERY_ORACLE"):
        for bounds in (CURRENT, GTFREE):
            for selection in SELECTIONS:
                label = f"{track}|{bounds}|{selection}"
                transitions[label] = []
                for a, b in zip(budgets[:-1], budgets[1:], strict=True):
                    left, right = (
                        group_key(track, bounds, a, selection),
                        group_key(track, bounds, b, selection),
                    )
                    query_gain = contrast(f"query_gain|{label}|{a}_to_{b}", left, right)
                    objective_gain = paired(
                        groups[left]["supervision_objective"]["per_scene"],
                        groups[right]["supervision_objective"]["per_scene"],
                        draws=draws,
                        seed=seed,
                    )
                    transition = {
                        "from_budget": a,
                        "to_budget": b,
                        "query_gain": query_gain,
                        "context_objective_gain": objective_gain,
                    }
                    if track == "RGBD":
                        transition["context_absrel_gain"] = contrast(
                            f"context_gain|{label}|{a}_to_{b}", left, right, field="context_metrics"
                        )
                        transition["scene_joint_directions"] = {
                            "context_improves_query_improves": sum(
                                objective_gain["per_scene"][s] > 1e-8
                                and query_gain["per_scene"][s] > 1e-8
                                for s in ids
                            ),
                            "context_improves_query_worsens": sum(
                                objective_gain["per_scene"][s] > 1e-8
                                and query_gain["per_scene"][s] < -1e-8
                                for s in ids
                            ),
                            "both_flat": sum(
                                abs(objective_gain["per_scene"][s]) <= 1e-8
                                and abs(query_gain["per_scene"][s]) <= 1e-8
                                for s in ids
                            ),
                        }
                    transitions[label].append(transition)
                contrast(
                    f"query_gain|{label}|first_to_max",
                    group_key(track, bounds, budgets[0], selection),
                    group_key(track, bounds, budgets[-1], selection),
                )
                if track == "RGBD":
                    contrast(
                        f"context_gain|{label}|first_to_max",
                        group_key(track, bounds, budgets[0], selection),
                        group_key(track, bounds, budgets[-1], selection),
                        field="context_metrics",
                    )
                    scene_pass = defaultdict(list)
                    for item in plateaus.values():
                        spec = item["spec"]
                        if spec["track"] == track and spec["bounds_mode"] == bounds:
                            scene_pass[spec["scene_id"]].append(
                                item["budgets"][str(budgets[-1])]["status"] == "PLATEAU"
                            )
                    fraction = sum(all(v) for v in scene_pass.values()) / len(ids)
                    statuses[label] = {
                        "OPTIMIZATION_STATUS": optimization_status(
                            transitions[label], fraction, rules["plateau"]
                        ),
                        "plateau_scene_fraction_at_max": fraction,
                        "plateau_per_scene_at_max": {s: all(v) for s, v in scene_pass.items()},
                    }
    bounds_analyses = {}
    for track in ("RGBD", "QUERY_ORACLE"):
        for selection in SELECTIONS:
            gains = [
                contrast(
                    f"bounds_gain|{track}|{selection}|{budget}",
                    group_key(track, CURRENT, budget, selection),
                    group_key(track, GTFREE, budget, selection),
                )
                for budget in budgets
            ]
            interaction = paired(
                gains[-1]["per_scene"], gains[0]["per_scene"], draws=draws, seed=seed
            )
            bounds_analyses[f"{track}|{selection}"] = {
                "BOUNDS_STATUS": bounds_status(gains, interaction),
                "gains_by_budget": dict(zip(map(str, budgets), gains, strict=True)),
                "interaction_max_minus_first": interaction,
            }
    wrong_scene_by_cell = {}
    for bounds in (CURRENT, GTFREE):
        for budget in budgets:
            for selection in SELECTIONS:
                key = group_key("RGBD", bounds, budget, selection)
                wrong_scene_by_cell[key] = contrast(
                    f"wrong_scene_damage|{key}",
                    group_key("RGBD", bounds, budget, selection, "wrong_scene"),
                    key,
                )
    primary_key = group_key("RGBD", GTFREE, budgets[-1])
    primary_status = statuses[f"RGBD|{GTFREE}|FIXED_BUDGET"]
    capacity = paired(
        carrier["depth_absrel"]["per_scene"],
        groups[primary_key]["metrics"]["depth_absrel"]["per_scene"],
        draws=draws,
        seed=seed,
    )
    wrong = contrast(
        "primary_wrong_scene_damage",
        group_key("RGBD", GTFREE, budgets[-1], method="wrong_scene"),
        primary_key,
    )
    decision = readiness(
        groups[primary_key]["metrics"]["depth_absrel"],
        capacity,
        wrong,
        groups[primary_key]["generalization_gap"],
        primary_status["OPTIMIZATION_STATUS"],
        transitions[f"RGBD|{GTFREE}|FIXED_BUDGET"][-1]["query_gain"],
        rules["readiness"],
    )
    oracle_gaps = {}
    for bounds in (CURRENT, GTFREE):
        for budget in budgets:
            oracle_gaps[f"{bounds}|{budget}"] = contrast(
                f"context_oracle_gap|{bounds}|{budget}",
                group_key("RGBD", bounds, budget),
                group_key("QUERY_ORACLE", bounds, budget),
            )
    oracle_gap_shrinkage = {
        bounds: paired(
            oracle_gaps[f"{bounds}|{budgets[0]}"]["per_scene"],
            oracle_gaps[f"{bounds}|{budgets[-1]}"]["per_scene"],
            draws=draws,
            seed=seed,
        )
        for bounds in (CURRENT, GTFREE)
    }
    secondary = {}
    if group_key("RGB_ONLY", GTFREE, budgets[-1]) in groups:
        secondary["RGB_ONLY_minus_RGBD"] = contrast(
            "secondary_RGB_ONLY_minus_RGBD", group_key("RGB_ONLY", GTFREE, budgets[-1]), primary_key
        )
    total_seconds = sum(
        t["budget_summaries"][str(max(map(int, t["budget_summaries"])))]["FIXED_BUDGET"][
            "cumulative_optimization_seconds"
        ]
        for t in trajectories.values()
    )
    return {
        "optimization_results": {
            "scene_ids": ids,
            "budgets": budgets,
            "primary_key": primary_key,
            "BEST_CONTEXT_ONLY_CONFIG_MEANING": "PREDECLARED_PRIMARY_NOT_QUERY_MINIMUM",
            "context_only": {k: v for k, v in groups.items() if not k.startswith("QUERY_ORACLE|")},
            "frozen_carrier": carrier,
            "secondary": secondary,
        },
        "optimization_sufficiency": {
            "primary": {**primary_status, **decision},
            "by_bounds_selection": statuses,
            "transitions": transitions,
            "plateau_by_trajectory": plateaus,
            "primary_capacity_gap": capacity,
            "primary_wrong_scene_damage": wrong,
            "wrong_scene_damage_all_budgets": wrong_scene_by_cell,
            "fixed_budget_not_global_optimality": True,
        },
        "bounds_analysis": bounds_analyses,
        "query_oracle_diagnostic": {
            "scientific_role": "PRIVILEGED_DIAGNOSTIC_NOT_DEPLOYABLE_OR_INDEPENDENT",
            "groups": {k: v for k, v in groups.items() if k.startswith("QUERY_ORACLE|")},
            "context_minus_oracle": oracle_gaps,
            "oracle_gap_shrinkage_first_to_max": oracle_gap_shrinkage,
            "poor_oracle_does_not_prove_impossible_representation": True,
        },
        "bootstrap_results": {
            "draws": draws,
            "seed": seed,
            "unit": "scene after query then role averaging",
            "comparisons": comparisons,
            "primary_capacity_gap": capacity,
        },
        "cost_analysis": {
            "actual_trajectory_optimization_seconds": total_seconds,
            "groups": {
                k: {
                    **v["cost"],
                    "render_seconds_per_query": v["render_seconds_per_query"],
                    "render_seconds_total": v["render_seconds_total"],
                }
                for k, v in groups.items()
                if "cost" in v
            },
            "future_carrier_cost": "NOT_MEASURED",
        },
    }
