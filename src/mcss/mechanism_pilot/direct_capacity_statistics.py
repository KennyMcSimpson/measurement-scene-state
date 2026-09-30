"""Raw-only, scene-paired attribution statistics; no media/model/optimizer access."""

from collections import defaultdict

import numpy as np

METRICS = (
    "depth_absrel",
    "depth_rmse",
    "depth_delta1",
    "rgb_mse",
    "rgb_ssim",
    "opacity",
    "coverage",
)
GEOMETRY = ("insideGTfraction", "ray_hitfraction", "GTdistance_above_far_fraction")
CURRENT = "CURRENT_BOUNDS"
ALTERNATIVE = "PREDECLARED_GEOMETRY_BOUNDS"
TIE = 1e-8


def group_key(track, grid=8, samples=64, bounds=CURRENT, method="direct"):
    return f"{track}|g{grid}|s{samples}|{bounds}|{method}"


def row_key(row):
    return group_key(row["track"], row["grid"], row["samples"], row["bounds_mode"], row["method"])


def describe(values, *, draws=10000, seed=20260927, positive_means_improvement=False):
    """Resample complete paired scene values, never queries or contexts."""
    ids = sorted(values)
    x = np.asarray([values[s] for s in ids], dtype=float)
    if not len(x) or not np.isfinite(x).all():
        raise ValueError("Nonempty finite scene values required")
    index = np.random.default_rng(seed).integers(0, len(x), (draws, len(x)))
    ci = np.percentile(x[index].mean(axis=1), [2.5, 97.5]).tolist()
    positive = np.maximum(x, 0)
    absolute = np.abs(x)

    def share(v, n):
        return float(np.sort(v)[-n:].sum() / v.sum()) if v.sum() else None

    loso = {s: float(np.delete(x, i).mean()) for i, s in enumerate(ids)} if len(x) > 1 else {}
    result = {
        "n_scenes": len(ids),
        "per_scene": dict(zip(ids, x.tolist(), strict=True)),
        "mean": float(x.mean()),
        "median": float(np.median(x)),
        "ci95": ci,
        "loso": loso,
        "loso_range": [min(loso.values()), max(loso.values())] if loso else None,
        "positive": int((x > TIE).sum()),
        "tied": int((np.abs(x) <= TIE).sum()),
        "negative": int((x < -TIE).sum()),
        "positive_top1_contribution": share(positive, 1),
        "positive_top3_contribution": share(positive, 3),
        "absolute_top1_contribution": share(absolute, 1),
        "absolute_top3_contribution": share(absolute, 3),
        "bootstrap": {"unit": "scene", "draws": draws, "seed": seed, "CI": "percentile95"},
    }
    if positive_means_improvement:
        result["improved_tied_worse"] = [result["positive"], result["tied"], result["negative"]]
    return result


def scene_values(rows, metric):
    """Each role has equal weight regardless of its number of scored frames."""
    role_values = defaultdict(list)
    for row in rows:
        value = row.get(metric)
        if value is None or not np.isfinite(value):
            raise ValueError(f"Missing or nonfinite {metric}; cannot silently discard a scene")
        role_values[row["scene_id"], row.get("role", "joint")].append(float(value))
    scenes = defaultdict(list)
    for (scene, _), values in role_values.items():
        scenes[scene].append(float(np.mean(values)))
    return {scene: float(np.mean(values)) for scene, values in sorted(scenes.items())}


def paired(left, right, *, draws=10000, seed=20260927, improvement=True):
    if not left or set(left) != set(right):
        raise ValueError("Paired comparisons require identical complete scene sets")
    return describe(
        {s: left[s] - right[s] for s in left},
        draws=draws,
        seed=seed,
        positive_means_improvement=improvement,
    )


def stable_gain(stats):
    return (
        stats["mean"] >= 0.02
        and stats["ci95"][0] > 0
        and (stats["positive"] + stats["tied"]) / stats["n_scenes"] >= 0.75
    )


def engineering_good(stats):
    return (
        stats["mean"] <= 0.25
        and np.mean(np.asarray(list(stats["per_scene"].values())) <= 0.35) >= 0.75
    )


def classify(groups, comparisons):
    """Frozen engineering thresholds; evidence is not a causal decomposition."""
    rgbd = groups[group_key("RGBD")]["metrics"]["depth_absrel"]
    rgbonly = groups[group_key("RGB_ONLY")]["metrics"]["depth_absrel"]
    oracle = groups[group_key("QUERY_ORACLE")]["metrics"]["depth_absrel"]
    resolution = [comparisons[f"resolution_RGBD_{a}_to_{b}"] for a, b in [(8, 16), (16, 32)]]
    render = [comparisons[f"renderer_RGBD_g{g}_64_to_128"] for g in (8, 16)]
    stable_render = [s["mean"] for s in render if stable_gain(s)]
    flags = {
        "LEARNER_TRAINING_BOTTLENECK": bool(
            engineering_good(rgbonly) and stable_gain(comparisons["capacity_gap_RGB_ONLY_g8"])
        ),
        "CONTEXT_INFERENCE_BOTTLENECK": bool(
            engineering_good(oracle) and stable_gain(comparisons["context_RGBD_minus_oracle_g8"])
        ),
        "RESOLUTION_BOTTLENECK": any(stable_gain(s) for s in resolution),
        "RENDERER_BOTTLENECK": bool(
            stable_render and max(stable_render) > max(s["mean"] for s in resolution)
        ),
    }
    sufficient = bool(
        engineering_good(rgbd)
        and stable_gain(comparisons["capacity_gap_RGBD_g8"])
        and stable_gain(comparisons["anchor_gap_RGBD_g8"])
    )
    names = {
        "LEARNER_TRAINING_BOTTLENECK": "LEARNER_LIMITED",
        "CONTEXT_INFERENCE_BOTTLENECK": "CONTEXT_INFERENCE_LIMITED",
        "RESOLUTION_BOTTLENECK": "RESOLUTION_LIMITED",
        "RENDERER_BOTTLENECK": "RENDERER_LIMITED",
    }
    active = [k for k, v in flags.items() if v]
    status = (
        "MIXED"
        if len(active) > 1
        else names[active[0]]
        if active
        else ("SUFFICIENT" if sufficient else "INCONCLUSIVE")
    )
    recommendations = []
    if flags["LEARNER_TRAINING_BOTTLENECK"]:
        recommendations.append(
            "Design a separate matched carrier retraining protocol; do not train now"
        )
    if flags["CONTEXT_INFERENCE_BOTTLENECK"]:
        recommendations.append(
            "Study context lifting, information and inverse-problem optimization"
        )
    if flags["RESOLUTION_BOTTLENECK"]:
        recommendations.append(
            "Design a separate higher-resolution carrier study with optimization controls"
        )
    if flags["RENDERER_BOTTLENECK"]:
        recommendations.append(
            "Study ray integration resolution separately from state construction"
        )
    if not recommendations:
        recommendations.append(
            "Resolve finite-optimization and bounds ambiguity before new carrier training"
        )
    return {
        "STATE_CAPACITY_STATUS": status,
        "next_stage_recommendations_only": recommendations,
        "NEW_CARRIER_TRAINED": False,
        "FINAL_HOLDOUT_TOUCHED": False,
        "DYNAMIC_TTT_RUN": False,
        **flags,
        "REPRESENTATION_BOTTLENECK": "NOT_ESTABLISHED_BY_FINITE_OPTIMIZATION",
        "sufficient_rgbd_evidence": sufficient,
        "engineering_good": {
            "RGBD8": bool(engineering_good(rgbd)),
            "RGB_ONLY8": bool(engineering_good(rgbonly)),
            "QUERY_ORACLE8": bool(engineering_good(oracle)),
        },
        "renderer_comparison_rule": (
            "largest stable renderer mean gain > largest RGBD grid mean gain"
        ),
        "limitations": [
            "Exposed attribution scenes, not independent qualification",
            "RGBD depth is privileged; this gap alone cannot identify RGB-only learner failure",
            "Finite optimization and bounds limitations preclude proving expressivity impossible",
            "Evidence flags and unadjusted diagnostic CIs are not strict causal attribution",
        ],
    }


def _validate(rows, scene_ids, *, plan=None, observation="query_id"):
    identities = set()
    for row in rows:
        identity = (row_key(row), row["scene_id"], row["role"], row[observation])
        if identity in identities:
            raise ValueError("Duplicate raw observation")
        identities.add(identity)
        if row["scene_id"] not in scene_ids:
            raise ValueError("Unknown or protected scene in raw")
    if plan is not None:
        expected = {(s["key"], s["scene_id"]) for s in plan}
        actual = {(r["key"], r["scene_id"]) for r in rows if r["method"] == "direct"}
        if actual != expected:
            raise ValueError("Raw does not cover exactly the planned states")


def analyze(
    baseline_rows,
    query_rows,
    context_rows,
    oracle_rows,
    summaries,
    expected_scene_ids,
    *,
    plan=None,
    manifest=None,
    draws=10000,
    seed=20260927,
):
    """Return seven JSON-ready artifacts; all inputs are saved raw observations."""
    scene_ids = sorted(expected_scene_ids)
    if len(set(scene_ids)) != len(scene_ids) or not scene_ids:
        raise ValueError("Unique nonempty scene roster required")
    states = None if plan is None else plan["states"]
    if states is not None and manifest is None:
        raise ValueError("A state plan requires the locked scene manifest")
    if states is not None and set(summaries) != {s["key"] for s in states}:
        raise ValueError("Every planned state must have a saved optimization summary")
    baseline_identities = {(r["scene_id"], r["method"], r["query_id"]) for r in baseline_rows}
    if len(baseline_identities) != len(baseline_rows):
        raise ValueError("Duplicate baseline observation")
    if any(r["method"] not in {"A", "B", "anchor", "prior"} for r in baseline_rows):
        raise ValueError("Unknown baseline method")
    query_ids = defaultdict(set)
    for row in baseline_rows:
        query_ids[row["scene_id"]].add(row["query_id"])
    if sorted(query_ids) != scene_ids:
        raise ValueError("Baseline must cover exactly the exposed scene roster")
    expected_baseline = {
        (scene, role, fid)
        for scene in scene_ids
        for role in ("A", "B", "anchor", "prior")
        for fid in query_ids[scene]
    }
    if baseline_identities != expected_baseline:
        raise ValueError("Every baseline role must contain all locked queries")
    if manifest is not None:
        records = {r["scene_id"]: r for r in manifest["scenes"]}
        if sorted(records) != scene_ids or set(scene_ids) & set(
            manifest["FINAL_HOLDOUT_PROHIBITED"]
        ):
            raise ValueError("Manifest roster must match exposed scenes without holdout")
        if any(
            query_ids[scene] != set(records[scene]["roles"]["primary_query"]) for scene in scene_ids
        ):
            raise ValueError("Queries disagree with locked manifest")
        expected_context = set()
        if states is None:
            permitted = {
                (row_key(r), r["scene_id"], r["role"])
                for r in query_rows
                if r["method"] == "direct"
            }
        else:
            permitted = {
                (row_key({**s, "method": "direct"}), s["scene_id"], s["role"])
                for s in states
                if s["phase"] == "context"
            }
        for key, scene, role in permitted:
            frames = records[scene]["roles"]["context_a" if role == "A" else "context_b"]
            expected_context.update((key, scene, role, fid) for fid in frames)
        actual_context = {
            (row_key(r), r["scene_id"], r["role"], r["frame_id"]) for r in context_rows
        }
        if actual_context != expected_context:
            raise ValueError(
                "Context results must cover every planned role and locked frame exactly"
            )
        if states is not None:
            expected_keys = {
                (s["key"], s["scene_id"], s["role"]) for s in states if s["phase"] == "context"
            }
            actual_keys = {(r["key"], r["scene_id"], r["role"]) for r in context_rows}
            if actual_keys != expected_keys:
                raise ValueError("Context evaluation state keys do not match the plan")
    for rows in (query_rows, oracle_rows):
        role_ids = defaultdict(set)
        for row in rows:
            role_ids[row_key(row), row["scene_id"], row["role"]].add(row["query_id"])
        if any(ids != query_ids[scene] for (_, scene, _), ids in role_ids.items()):
            raise ValueError("Every state/control must score exactly the same locked queries")
    if states is not None:
        for rows, phase in ((query_rows, "context"), (oracle_rows, "oracle")):
            expected_roles = defaultdict(set)
            actual_roles = defaultdict(set)
            for spec in states:
                if spec["phase"] == phase:
                    expected_roles[row_key({**spec, "method": "direct"}), spec["scene_id"]].add(
                        spec["role"]
                    )
            for row in rows:
                if row["method"] == "direct":
                    actual_roles[row_key(row), row["scene_id"]].add(row["role"])
            if actual_roles != expected_roles:
                raise ValueError("Incomplete A/B roles in planned states")
    _validate(
        query_rows,
        scene_ids,
        plan=None if states is None else [s for s in states if s["phase"] == "context"],
    )
    _validate(
        oracle_rows,
        scene_ids,
        plan=None if states is None else [s for s in states if s["phase"] == "oracle"],
    )
    _validate(context_rows, scene_ids, observation="frame_id")
    if any(r["track"] == "QUERY_ORACLE" for r in query_rows) or any(
        r["track"] != "QUERY_ORACLE" for r in oracle_rows
    ):
        raise ValueError("Context and privileged oracle blocks must be separate")
    grouped, contexts = defaultdict(list), defaultdict(list)
    for row in query_rows + oracle_rows:
        grouped[row_key(row)].append(row)
    for row in context_rows:
        contexts[row_key(row)].append(row)
    baseline_groups = defaultdict(list)
    for row in baseline_rows:
        role = row["method"]
        name = "CARRIER" if role in ("A", "B") else role.upper()
        baseline_groups[name].append({**row, "role": role})
    for name in ("CARRIER", "ANCHOR", "PRIOR"):
        if name not in baseline_groups:
            raise ValueError("Missing carrier/anchor/prior baseline")
    # Each control and baseline must retain the locked A/B roles as well as query IDs.
    for key, rows in grouped.items():
        roles = defaultdict(set)
        for row in rows:
            roles[row["scene_id"]].add(row["role"])
        expected = (
            {"joint_query"} if key.startswith("QUERY_ORACLE|") and CURRENT in key else {"A", "B"}
        )
        if any(value != expected for value in roles.values()):
            raise ValueError("Missing or unexpected context roles")
    for name, rows in baseline_groups.items():
        if name == "CARRIER":
            for scene in scene_ids:
                if {r["role"] for r in rows if r["scene_id"] == scene} != {"A", "B"}:
                    raise ValueError("Missing carrier A/B role")
    groups = {}
    for key, rows in {**grouped, **baseline_groups}.items():
        metrics = {}
        available = METRICS + tuple(k for k in GEOMETRY if all(k in r for r in rows))
        for metric in available:
            values = scene_values(rows, metric)
            if sorted(values) != scene_ids:
                raise ValueError("Every method must cover every scene")
            metrics[metric] = describe(values, draws=draws, seed=seed)
        groups[key] = {
            "metrics": metrics,
            "n_rows": len(rows),
            "scientific_role": "DIAGNOSTIC_ORACLE"
            if key.startswith("QUERY_ORACLE|")
            else "CONTEXT_ONLY"
            if "|" in key
            else "FROZEN_CARRIER_BASELINE",
        }
        if key in grouped:
            groups[key]["spec"] = {
                k: rows[0][k] for k in ("track", "grid", "samples", "bounds_mode", "method")
            }
            groups[key]["render_seconds_total"] = sum(r.get("seconds", 0) for r in rows)
        if key in contexts:
            groups[key]["context_metrics"] = {
                metric: describe(scene_values(contexts[key], metric), draws=draws, seed=seed)
                for metric in METRICS
            }
    comparisons = {}

    def contrast(name, left, right, metric="depth_absrel", improvement=True):
        comparisons[name] = paired(
            groups[left]["metrics"][metric]["per_scene"],
            groups[right]["metrics"][metric]["per_scene"],
            draws=draws,
            seed=seed,
            improvement=improvement,
        )
        comparisons[name]["definition"] = f"{left} - {right} ({metric})"
        comparisons[name]["positive_means"] = "improvement" if improvement else "larger_gap"

    for track in ("RGBD", "RGB_ONLY"):
        for grid in (8, 16, 32):
            key = group_key(track, grid)
            contrast(f"capacity_gap_{track}_g{grid}", "CARRIER", key)
            contrast(f"anchor_gap_{track}_g{grid}", "ANCHOR", key)
            contrast(f"direct_minus_carrier_{track}_g{grid}", key, "CARRIER", improvement=False)
            contrast(f"context_{track}_minus_oracle_g{grid}", key, group_key("QUERY_ORACLE", grid))
        for a, b in ((8, 16), (16, 32)):
            contrast(f"resolution_{track}_{a}_to_{b}", group_key(track, a), group_key(track, b))
        contrast(f"bounds_gain_{track}", group_key(track), group_key(track, bounds=ALTERNATIVE))
        for method in (
            "wrong_scene",
            "spatial_shuffle",
            "density_only",
            "color_only",
            "zero_density_color",
        ):
            contrast(
                f"control_damage_{track}_{method}",
                group_key(track, method=method),
                group_key(track),
            )
    contrast(
        "bounds_gain_QUERY_ORACLE",
        group_key("QUERY_ORACLE"),
        group_key("QUERY_ORACLE", bounds=ALTERNATIVE),
    )
    for grid in (8, 16):
        contrast(
            f"renderer_RGBD_g{grid}_64_to_128",
            group_key("RGBD", grid),
            group_key("RGBD", grid, 128),
        )
    for group in groups.values():
        if "context_metrics" not in group:
            continue
        group["generalization_gap"] = {
            metric: paired(
                group["metrics"][metric]["per_scene"],
                group["context_metrics"][metric]["per_scene"],
                draws=draws,
                seed=seed,
                improvement=False,
            )
            for metric in ("depth_absrel", "depth_rmse", "rgb_mse")
        }
    metric_comparisons = {}
    for track in ("RGBD", "RGB_ONLY"):
        for grid in (8, 16, 32):
            key = group_key(track, grid)
            metric_comparisons[key] = {}
            for metric in METRICS:
                stats = paired(
                    groups[key]["metrics"][metric]["per_scene"],
                    groups["CARRIER"]["metrics"][metric]["per_scene"],
                    draws=draws,
                    seed=seed,
                    improvement=False,
                )
                if metric in ("depth_absrel", "depth_rmse", "rgb_mse"):
                    stats["improved_tied_worse"] = [
                        stats["negative"],
                        stats["tied"],
                        stats["positive"],
                    ]
                    stats["better_direction"] = "negative"
                elif metric in ("depth_delta1", "rgb_ssim"):
                    stats["improved_tied_worse"] = [
                        stats["positive"],
                        stats["tied"],
                        stats["negative"],
                    ]
                    stats["better_direction"] = "positive"
                else:
                    stats["better_direction"] = "not_intrinsically_task_quality"
                stats["definition"] = f"{key} - CARRIER ({metric})"
                metric_comparisons[key][metric] = stats
    classification = classify(groups, comparisons)
    curves, costs = {}, defaultdict(list)
    specs = {s["key"]: s for s in states} if states is not None else {}
    for key, summary in sorted(summaries.items()):
        checks = summary["full_objective_checkpoints"]
        last = checks[-3:]
        trend = (
            float(
                np.polyfit([v["step"] for v in last], [v["context_objective"] for v in last], 1)[0]
            )
            if len(last) > 1
            else None
        )
        curves[key] = {
            **summary,
            "last3_objective_slope_per_step": trend,
            "convergence_claim": "NOT_ESTABLISHED_BY_FIXED_BUDGET",
        }
        if key in specs:
            spec = specs[key]
            group = group_key(spec["track"], spec["grid"], spec["samples"], spec["bounds_mode"])
            costs[group].append(summary)
    for key, rows in costs.items():
        groups[key]["optimization_cost"] = {
            "n_states": len(rows),
            "seconds_total": sum(r["seconds"] for r in rows),
            "seconds_mean": float(np.mean([r["seconds"] for r in rows])),
            "parameter_count_per_state": sorted({r["parameter_count"] for r in rows}),
            "peak_cuda_allocated_bytes": max(
                (r.get("peak_cuda_allocated_bytes") or 0) for r in rows
            ),
            "peak_cuda_reserved_bytes": max((r.get("peak_cuda_reserved_bytes") or 0) for r in rows),
            "selected_at_budget_boundary": sum(
                r["final_selected_at_budget_boundary"] for r in rows
            ),
        }
    blocks = {"context_only": {}, "diagnostic_oracle": {}, "frozen_carrier": {}}
    for key, group in groups.items():
        block = (
            "diagnostic_oracle"
            if key.startswith("QUERY_ORACLE|")
            else "context_only"
            if "|" in key
            else "frozen_carrier"
        )
        blocks[block][key] = group
    return {
        "direct_state_results": {
            "scene_ids": scene_ids,
            "n_scenes": len(scene_ids),
            **blocks,
            "query_metric_scope": "All GT-valid depth pixels; no prediction-derived masking",
            "coverage_definition": "opacity>1e-6, not true scene visibility",
        },
        "capacity_analysis": {
            "classification": classification,
            "comparisons": comparisons,
            "direct_minus_carrier_all_metrics": metric_comparisons,
            "generalization": {
                k: g["generalization_gap"] for k, g in groups.items() if "generalization_gap" in g
            },
        },
        "resolution_analysis": {
            k: v for k, v in comparisons.items() if k.startswith("resolution_")
        },
        "renderer_capacity_analysis": {
            k: v for k, v in comparisons.items() if k.startswith("renderer_")
        },
        "bootstrap_results": {
            "unit": "scene; first average queries within role then roles A/B",
            "draws": draws,
            "seed": seed,
            "comparisons": comparisons,
        },
        "optimization_curves": curves,
        "bounds_analysis": {k: v for k, v in comparisons.items() if k.startswith("bounds_")},
    }
