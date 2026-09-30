"""Scene-level, raw-only observability and geometry-supervision attribution.

Empty masks remain explicit. Conditional estimands and additive overall-error
contributions are kept separate; query pixels never become bootstrap units.
"""

from collections import defaultdict

import numpy as np

from mcss.mechanism_pilot.direct_capacity_statistics import METRICS, describe, paired, scene_values

REGIONS = (
    "OVERALL",
    "OBS0",
    "OBS1",
    "OBS2PLUS",
    "OBS01",
    "ANGLE_0_5",
    "ANGLE_5_15",
    "ANGLE_15_30",
    "ANGLE_GT30",
    "OCCLUDED_ANY",
    "CONFLICT_ANY",
)
PARTITION = ("OBS0", "OBS1", "OBS2PLUS")
VARIANTS = ("S0", "S1", "S2", "S3")


def conditional_summary(values, *, draws=10000, seed=20260927, improvement=False):
    """All scenes are sampled; every draw recomputes its covered-scene denominator."""
    ids = sorted(values)
    if not ids:
        raise ValueError("A nonempty complete scene roster is required")
    finite = {s: float(values[s]) for s in ids if values[s] is not None}
    if not all(np.isfinite(v) for v in finite.values()):
        raise ValueError("Nonfinite values must not be silently removed")
    x = np.asarray([0.0 if values[s] is None else values[s] for s in ids], dtype=float)
    valid = np.asarray([values[s] is not None for s in ids], dtype=bool)
    indices = np.random.default_rng(seed).integers(0, len(ids), size=(draws, len(ids)))
    denominators = valid[indices].sum(1)
    defined = denominators > 0
    means = x[indices].sum(1)[defined] / denominators[defined]
    positive = np.maximum(np.asarray(list(finite.values())), 0)
    absolute = np.abs(np.asarray(list(finite.values())))

    def share(array, count):
        return float(np.sort(array)[-count:].sum() / array.sum()) if array.sum() else None

    loso = {}
    for i, scene in enumerate(ids):
        n = int(valid.sum() - valid[i])
        loso[scene] = float((x.sum() - x[i]) / n) if n else None
    observed = np.asarray(list(finite.values()), dtype=float)
    result = {
        "n_scenes": len(ids),
        "n_covered_scenes": len(finite),
        "scene_coverage": len(finite) / len(ids),
        "empty_mask_scene_count": len(ids) - len(finite),
        "per_scene": {s: values[s] for s in ids},
        "mean": float(observed.mean()) if len(observed) else None,
        "median": float(np.median(observed)) if len(observed) else None,
        "ci95": np.percentile(means, [2.5, 97.5]).tolist() if len(means) else None,
        "positive": int((observed > 1e-8).sum()),
        "tied": int((np.abs(observed) <= 1e-8).sum()),
        "negative": int((observed < -1e-8).sum()),
        "loso": loso,
        "positive_top1_contribution": share(positive, 1),
        "positive_top3_contribution": share(positive, 3),
        "absolute_top1_contribution": share(absolute, 1),
        "absolute_top3_contribution": share(absolute, 3),
        "bootstrap": {
            "unit": "all scenes with coverage denominator recomputed each draw",
            "draws": draws,
            "seed": seed,
            "undefined_empty_draws": int((~defined).sum()),
            "defined_draws": int(defined.sum()),
        },
        "estimand": "equal mean of defined scene-conditional values; all scene keys retained",
    }
    if improvement:
        result["improved_tied_worse"] = [result["positive"], result["tied"], result["negative"]]
    return result


def conditional_difference(left, right, *, draws=10000, seed=20260927):
    if set(left) != set(right):
        raise ValueError("Paired regional contrasts require the same complete scene roster")
    return conditional_summary(
        {s: None if left[s] is None or right[s] is None else left[s] - right[s] for s in left},
        draws=draws,
        seed=seed,
        improvement=True,
    )


def stable_gain(stats):
    return (
        stats["mean"] is not None
        and stats["mean"] >= 0.02
        and stats["ci95"][0] > 0
        and (stats["positive"] + stats["tied"]) / stats["n_scenes"] >= 0.75
    )


def negative(stats, *, draws=10000, seed=20260927):
    return conditional_summary(
        {s: None if v is None else -v for s, v in stats["per_scene"].items()},
        draws=draws,
        seed=seed,
        improvement=True,
    )


def geometry_status(comparisons):
    if stable_gain(comparisons["S3_minus_S0"]):
        return "HARMFUL"
    if all(stable_gain(comparisons[k]) for k in ("S0_minus_S3", "S1_minus_S3", "S2_minus_S3")):
        return "COMPLEMENTARY"
    free = stable_gain(comparisons["S0_minus_S1"])
    surface = stable_gain(comparisons["S0_minus_S2"])
    if free and not surface and stable_gain(comparisons["S2_minus_S1"]):
        return "FREE_SPACE_DOMINANT"
    if surface and not free and stable_gain(comparisons["S1_minus_S2"]):
        return "SURFACE_DOMINANT"
    if not any(stable_gain(comparisons[f"S0_minus_{v}"]) for v in ("S1", "S2", "S3")):
        return "NO_CLEAR_GAIN"
    return "INCONCLUSIVE"


def positive_low_support_share(contribution_gaps):
    positive = {
        region: sum(max(v, 0) for v in contribution_gaps[region].values()) for region in PARTITION
    }
    total = sum(positive.values())
    return {
        "share": (positive["OBS0"] + positive["OBS1"]) / total if total else None,
        "positive_contribution_by_region": positive,
        "definition": "positive scene-region additive contributions; not signed net-gap share",
    }


def observability_status(observed_gap, low_support_coverage, positive_share, rules):
    coverage = rules["region_rules"]["minimum_scene_coverage"]
    if observed_gap["scene_coverage"] < coverage or observed_gap["ci95"] is None:
        return "INCONCLUSIVE"
    decision = rules["observability_rules"]
    significant = (
        observed_gap["mean"] >= decision["observed_gap_min"] and observed_gap["ci95"][0] > 0
    )
    if significant:
        mixed = (
            low_support_coverage >= coverage
            and positive_share is not None
            and positive_share >= decision["mixed_low_support_positive_share_min"]
        )
        return "MIXED" if mixed else "OBSERVED_REGION_FAILURE"
    margin = decision["equivalence_margin"]
    near = -margin <= observed_gap["ci95"][0] and observed_gap["ci95"][1] <= margin
    if (
        near
        and low_support_coverage >= coverage
        and positive_share is not None
        and positive_share >= decision["unobserved_positive_share_min"]
    ):
        return "UNOBSERVED_REGION_DOMINANT"
    return "INCONCLUSIVE"


def gap_fraction(base, closed, *, draws=10000, seed=20260927):
    if not base or set(base) != set(closed):
        raise ValueError("Paired complete scene gaps required")
    ids = sorted(base)
    denominator = np.asarray([base[s] for s in ids], dtype=float)
    numerator = np.asarray([closed[s] for s in ids], dtype=float)
    index = np.random.default_rng(seed).integers(0, len(ids), (draws, len(ids)))
    b, c = denominator[index].mean(1), numerator[index].mean(1)
    valid = b > 0
    ratios = c[valid] / b[valid]
    return {
        "value": float(numerator.mean() / denominator.mean()) if denominator.mean() > 0 else None,
        "ci95": np.percentile(ratios, [2.5, 97.5]).tolist() if len(ratios) else None,
        "undefined_nonpositive_base_draws": int((~valid).sum()),
        "draws": draws,
        "seed": seed,
        "meaning": "ratio of scene-macro gap closure to base gap; not theoretical optimum recovery",
    }


def _identity(row):
    return row["scene_id"], row["role"], row["query_id"]


def validate_rows(query, context, regions, observability, manifest):
    records = {s["scene_id"]: s for s in manifest["scenes"]}
    ids = sorted(records)
    if (
        not ids
        or len(ids) != len(manifest["scenes"])
        or set(ids) & set(manifest["FINAL_HOLDOUT_PROHIBITED"])
    ):
        raise ValueError("Unique exposed roster without protected scenes required")
    identities = {
        (scene, role, q)
        for scene, record in records.items()
        for role in ("A", "B")
        for q in record["roles"]["primary_query"]
    }
    variants = set(VARIANTS) | {"ORACLE", "CARRIER", "ANCHOR"}
    if any(r["variant"] == "OLD_BASELINE" for r in query):
        variants.add("OLD_BASELINE")
    expected = {(variant, "direct", *identity) for variant in variants for identity in identities}
    expected |= {
        (variant, "wrong_scene", *identity) for variant in VARIANTS for identity in identities
    }
    expected |= {
        ("S3", method, *identity)
        for method in ("spatial_shuffle", "zero")
        for identity in identities
    }
    actual = {(r["variant"], r["method"], *_identity(r)) for r in query}
    if len(actual) != len(query) or actual != expected:
        raise ValueError(
            "Every registered variant/control must cover each locked scene/role/query once"
        )
    expected_context = {
        (v, "direct", scene, role, fid)
        for v in VARIANTS
        for scene, record in records.items()
        for role in ("A", "B")
        for fid in record["roles"]["context_a" if role == "A" else "context_b"]
    }
    actual_context = {
        (r["variant"], r["method"], r["scene_id"], r["role"], r["frame_id"]) for r in context
    }
    if len(actual_context) != len(context) or actual_context != expected_context:
        raise ValueError("Context results must cover all locked frames, roles and variants")
    expected_regions = {
        (v, region, *identity) for v in variants for region in REGIONS for identity in identities
    }
    region_lookup = {(r["variant"], r["region"], *_identity(r)): r for r in regions}
    if len(region_lookup) != len(regions) or set(region_lookup) != expected_regions:
        raise ValueError("Complete region matrix required, including explicit empty masks")
    if any(r["method"] != "direct" for r in regions):
        raise ValueError("Region rows are direct states only")
    actual_obs = {_identity(r) for r in observability}
    if len(actual_obs) != len(observability) or actual_obs != identities:
        raise ValueError("Observability summaries must cover all locked observations")
    query_lookup = {(r["variant"], *_identity(r)): r for r in query if r["method"] == "direct"}
    maximum_residual = 0.0
    for identity in identities:
        for region in REGIONS:
            masks = {
                (
                    region_lookup[v, region, *identity]["pixel_count"],
                    region_lookup[v, region, *identity]["total_valid"],
                )
                for v in variants
            }
            if len(masks) != 1:
                raise ValueError("Masks must be identical across variants")
        for variant in variants:
            rows = {region: region_lookup[variant, region, *identity] for region in REGIONS}
            for row in rows.values():
                count, total, error = row["pixel_count"], row["total_valid"], row["absrel_sum"]
                if not (
                    isinstance(count, int)
                    and isinstance(total, int)
                    and 0 <= count <= total
                    and total > 0
                ):
                    raise ValueError("Valid integer region counts and nonempty query GT required")
                if not np.isfinite(error) or error < 0:
                    raise ValueError("Finite nonnegative region error sums required")
                if count == 0:
                    if error != 0 or row["absrel_mean"] is not None:
                        raise ValueError("Empty region must have zero error sum and null mean")
                elif row["absrel_mean"] is None or not np.isclose(
                    row["absrel_mean"], error / count, rtol=1e-6, atol=1e-8
                ):
                    raise ValueError("Region mean disagrees with sum and pixel count")
            if len({r["total_valid"] for r in rows.values()}) != 1:
                raise ValueError("Regional total-valid denominator must remain identical")
            overall = rows["OVERALL"]
            if overall["pixel_count"] != overall["total_valid"]:
                raise ValueError("Overall region must retain every GT-valid query pixel")
            for parent, children in (
                ("OVERALL", PARTITION),
                ("OBS01", ("OBS0", "OBS1")),
                ("OBS2PLUS", ("ANGLE_0_5", "ANGLE_5_15", "ANGLE_15_30", "ANGLE_GT30")),
            ):
                if sum(rows[r]["pixel_count"] for r in children) != rows[parent]["pixel_count"]:
                    raise ValueError("Regional pixel partitions do not reconstruct parent")
                residual = abs(
                    sum(rows[r]["absrel_sum"] for r in children) - rows[parent]["absrel_sum"]
                )
                maximum_residual = max(maximum_residual, residual / overall["total_valid"])
                if not np.isclose(
                    sum(rows[r]["absrel_sum"] for r in children),
                    rows[parent]["absrel_sum"],
                    rtol=1e-6,
                    atol=1e-8,
                ):
                    raise ValueError("Regional error contributions do not reconstruct parent")
            if not np.isclose(
                overall["absrel_sum"] / overall["total_valid"],
                query_lookup[variant, *identity]["depth_absrel"],
                rtol=1e-6,
                atol=1e-8,
            ):
                raise ValueError("Overall region disagrees with unfiltered query metric")
    return ids, sorted(variants), maximum_residual


def analyze(
    query_rows,
    context_rows,
    region_rows,
    observability_rows,
    trajectories,
    manifest,
    rules,
    *,
    draws=None,
    seed=None,
):
    draws = rules["statistics"]["draws"] if draws is None else draws
    seed = rules["statistics"]["seed"] if seed is None else seed
    if rules["primary_variant"] != "S3":
        raise ValueError("Primary cannot be selected after seeing query scores")
    ids, variants, residual = validate_rows(
        query_rows, context_rows, region_rows, observability_rows, manifest
    )
    query_groups, context_groups, region_groups = (
        defaultdict(list),
        defaultdict(list),
        defaultdict(list),
    )
    for row in query_rows:
        query_groups[row["variant"], row["method"]].append(row)
    for row in context_rows:
        context_groups[row["variant"]].append(row)
    for row in region_rows:
        region_groups[row["variant"], row["region"]].append(row)
    groups = {}
    for (variant, method), rows in query_groups.items():
        key = f"{variant}|{method}"
        groups[key] = {
            "variant": variant,
            "method": method,
            "query_metrics": {
                m: describe(scene_values(rows, m), draws=draws, seed=seed) for m in METRICS
            },
            "render_seconds_total": sum(r.get("seconds", 0) for r in rows),
        }
        if method == "direct" and variant in context_groups:
            groups[key]["context_metrics"] = {
                m: describe(scene_values(context_groups[variant], m), draws=draws, seed=seed)
                for m in METRICS
            }
            groups[key]["generalization_gap"] = paired(
                groups[key]["query_metrics"]["depth_absrel"]["per_scene"],
                groups[key]["context_metrics"]["depth_absrel"]["per_scene"],
                draws=draws,
                seed=seed,
                improvement=False,
            )
    regions = {}
    for variant in variants:
        regions[variant] = {}
        for region in REGIONS:
            rows = [
                {
                    **r,
                    "fraction": r["pixel_count"] / r["total_valid"],
                    "contribution": r["absrel_sum"] / r["total_valid"],
                }
                for r in region_groups[variant, region]
            ]
            fraction, contribution = (
                scene_values(rows, "fraction"),
                scene_values(rows, "contribution"),
            )
            conditional = {
                s: contribution[s] / fraction[s] if fraction[s] > 0 else None for s in ids
            }
            regions[variant][region] = {
                "conditional_absrel": conditional_summary(conditional, draws=draws, seed=seed),
                "pixel_fraction": describe(fraction, draws=draws, seed=seed),
                "additive_error_contribution": describe(contribution, draws=draws, seed=seed),
            }
    comparisons = {}
    for left in VARIANTS:
        for right in VARIANTS:
            if left == right:
                continue
            comparisons[f"{left}_minus_{right}"] = paired(
                groups[f"{left}|direct"]["query_metrics"]["depth_absrel"]["per_scene"],
                groups[f"{right}|direct"]["query_metrics"]["depth_absrel"]["per_scene"],
                draws=draws,
                seed=seed,
            )
    region_gaps, regional_gains = {}, {}
    for region in REGIONS:
        region_gaps[region] = conditional_difference(
            regions["S0"][region]["conditional_absrel"]["per_scene"],
            regions["ORACLE"][region]["conditional_absrel"]["per_scene"],
            draws=draws,
            seed=seed,
        )
        regional_gains[region] = {}
        for variant in ("S1", "S2", "S3"):
            regional_gains[region][variant] = {
                "conditional_gain": conditional_difference(
                    regions["S0"][region]["conditional_absrel"]["per_scene"],
                    regions[variant][region]["conditional_absrel"]["per_scene"],
                    draws=draws,
                    seed=seed,
                ),
                "additive_gain_contribution": paired(
                    regions["S0"][region]["additive_error_contribution"]["per_scene"],
                    regions[variant][region]["additive_error_contribution"]["per_scene"],
                    draws=draws,
                    seed=seed,
                ),
            }
    additive_gaps = {
        r: {
            s: regions["S0"][r]["additive_error_contribution"]["per_scene"][s]
            - regions["ORACLE"][r]["additive_error_contribution"]["per_scene"][s]
            for s in ids
        }
        for r in PARTITION
    }
    share = positive_low_support_share(additive_gaps)
    obs_status = observability_status(
        region_gaps["OBS2PLUS"],
        regions["S0"]["OBS01"]["conditional_absrel"]["scene_coverage"],
        share["share"],
        rules,
    )
    controls = {}
    for variant in VARIANTS:
        for method in (
            ("wrong_scene", "spatial_shuffle", "zero") if variant == "S3" else ("wrong_scene",)
        ):
            controls[f"{variant}|{method}"] = paired(
                groups[f"{variant}|{method}"]["query_metrics"]["depth_absrel"]["per_scene"],
                groups[f"{variant}|direct"]["query_metrics"]["depth_absrel"]["per_scene"],
                draws=draws,
                seed=seed,
            )
    oracle_values = groups["ORACLE|direct"]["query_metrics"]["depth_absrel"]["per_scene"]
    base_gap = paired(
        groups["S0|direct"]["query_metrics"]["depth_absrel"]["per_scene"],
        oracle_values,
        draws=draws,
        seed=seed,
    )
    closure = {}
    for variant in ("S1", "S2", "S3"):
        new = paired(
            groups[f"{variant}|direct"]["query_metrics"]["depth_absrel"]["per_scene"],
            oracle_values,
            draws=draws,
            seed=seed,
        )
        closed = comparisons[f"S0_minus_{variant}"]
        closure[variant] = {
            "new_gap": new,
            "gap_closed": closed,
            "gap_closed_fraction": gap_fraction(
                base_gap["per_scene"], closed["per_scene"], draws=draws, seed=seed
            ),
        }
    correlations = {}
    for scene in ids:
        rows = [r for r in observability_rows if r["scene_id"] == scene]
        if any(r["spearman_visible_count_absrel"] is None for r in rows):
            correlations[scene] = None
        else:
            correlations[scene] = scene_values(rows, "spearman_visible_count_absrel")[scene]
    correlation_stats = conditional_summary(correlations, draws=draws, seed=seed)
    low_minus_observed = conditional_difference(
        regions["S0"]["OBS01"]["conditional_absrel"]["per_scene"],
        regions["S0"]["OBS2PLUS"]["conditional_absrel"]["per_scene"],
        draws=draws,
        seed=seed,
    )
    minimum = rules["region_rules"]["minimum_scene_coverage"]
    observed = region_gaps["OBS2PLUS"]
    margin = rules["observability_rules"]["equivalence_margin"]
    observed_near = (
        observed["scene_coverage"] >= minimum
        and observed["ci95"] is not None
        and -margin <= observed["ci95"][0]
        and observed["ci95"][1] <= margin
    )
    visibility = (
        obs_status not in ("OBSERVED_REGION_FAILURE", "MIXED")
        and observed_near
        and low_minus_observed["scene_coverage"] >= minimum
        and low_minus_observed["mean"] >= 0.02
        and low_minus_observed["ci95"][0] > 0
        and correlation_stats["scene_coverage"] >= minimum
        and correlation_stats["ci95"][1] < 0
    )
    geometry_recommended = (
        stable_gain(comparisons["S0_minus_S3"])
        and controls["S3|wrong_scene"]["ci95"][0] > 0
        and controls["S3|spatial_shuffle"]["ci95"][0] > 0
    )
    costs, seen_trajectories = defaultdict(list), set()
    for key, item in trajectories.items():
        spec, summary = item["spec"], item["summary"]
        identity = spec["variant"], spec["scene_id"], spec["role"]
        if identity in seen_trajectories:
            raise ValueError("Duplicate optimized trajectory identity")
        seen_trajectories.add(identity)
        costs[spec["variant"]].append({"key": key, **summary})
    expected_trajectories = {(v, s, role) for v in VARIANTS for s in ids for role in ("A", "B")}
    if seen_trajectories != expected_trajectories:
        raise ValueError("All four matched optimization variants must have per-role summaries")
    costs_by_variant = {}
    for variant, summaries in costs.items():
        seconds = [
            r["seconds"] if "seconds" in r else r["cumulative_optimization_seconds"]
            for r in summaries
        ]
        costs_by_variant[variant] = {
            "n_states": len(summaries),
            "final_context_objective_components": {
                r["key"]: r.get("final_context_metrics") for r in summaries
            },
            "optimization_seconds_total": sum(seconds),
            "optimization_seconds_per_state": float(np.mean(seconds)),
            "peak_cuda_allocated_bytes": max(
                (r.get("peak_cuda_allocated_bytes") or 0) for r in summaries
            ),
            "peak_cuda_reserved_bytes": max(
                (r.get("peak_cuda_reserved_bytes") or 0) for r in summaries
            ),
            "render_seconds_total": groups[f"{variant}|direct"]["render_seconds_total"],
        }
    frozen_recommendations = {
        "GEOMETRY_AWARE_CARRIER_RECOMMENDED": bool(geometry_recommended),
        "VISIBILITY_AWARE_FUSION_RECOMMENDED": True if visibility else "inconclusive",
        "meaning": "future research only; no carrier training or independent qualification",
    }
    return {
        "observability_analysis": {
            "scene_ids": ids,
            "regions": regions,
            "OBSERVABILITY_STATUS": obs_status,
            "base_conditional_context_oracle_gaps": region_gaps,
            "positive_low_support_gap_share": share,
            "signed_additive_gap_by_region": {
                r: describe(v, draws=draws, seed=seed) for r, v in additive_gaps.items()
            },
            "partition_max_abs_error_residual": residual,
            "S0_visible_count_error_spearman": correlation_stats,
            "S0_low_support_minus_observed_error": low_minus_observed,
            "observability_semantics": "approximate nearest-depth consistency",
        },
        "supervision_analysis": {
            "primary_variant": "S3",
            "BEST_SUPERVISION_MEANING": "PREDECLARED_PRIMARY_NOT_QUERY_WINNER",
            "groups": {k: v for k, v in groups.items() if v["variant"] in VARIANTS},
            "reference_groups": {k: v for k, v in groups.items() if v["variant"] not in VARIANTS},
            "overall_absrel_comparisons": comparisons,
            "regional_gains": regional_gains,
            "GEOMETRY_SUPERVISION_STATUS": geometry_status(comparisons),
            **frozen_recommendations,
        },
        "gap_closure_analysis": {
            "base_gap": base_gap,
            "variants": closure,
            "oracle_role": "privileged diagnostic, not mathematical upper bound or deployment",
        },
        "wrong_scene_analysis": {
            "contrasts": controls,
            "positive_means": "direct better than control at identical recipient camera",
        },
        "bootstrap_results": {
            "draws": draws,
            "seed": seed,
            "unit": "scene, never pixels",
            "overall_comparisons": comparisons,
            "regional_gains": regional_gains,
            "base_regional_gaps": region_gaps,
            "region_empty_policy": "All scenes kept; bootstrap coverage denominator is recomputed",
        },
        "cost_analysis": {
            "variants": costs_by_variant,
            "actual_optimization_seconds": sum(
                v["optimization_seconds_total"] for v in costs_by_variant.values()
            ),
            "timing_note": "sum worker seconds; shared-GPU wall time separate; references reused",
        },
        "decision_summary": {
            "OBSERVABILITY_STATUS": obs_status,
            "GEOMETRY_SUPERVISION_STATUS": geometry_status(comparisons),
            "BEST_SUPERVISION": "S3_PREDECLARED_PRIMARY",
            **frozen_recommendations,
            "FINAL_HOLDOUT_TOUCHED": False,
            "NEW_CARRIER_TRAINED": False,
            "DYNAMIC_TTT_RUN": False,
        },
    }
