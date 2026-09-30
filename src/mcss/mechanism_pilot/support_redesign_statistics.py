"""Scene-equal support attribution from raw rows only; no model or geometry search.

Oracle rows are privileged diagnostics, not mathematical upper bounds. Correlations
are descriptive and do not identify a causal mechanism.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from mcss.mechanism_pilot.statistics import paired_scene_bootstrap

METRICS = (
    "rgb_mse",
    "rgb_ssim",
    "depth_absrel",
    "depth_rmse",
    "depth_delta1",
    "opacity",
    "coverage",
    "inside_fraction",
    "two_view_candidate_fraction",
    "supported_surface_fraction",
)
METHODS = ("A", "B", "anchor", "prior", "wrong_scene")
ORACLES = ("ORACLE_VOLUME", "ORACLE_SUPPORT", "ORACLE_VOLUME_SUPPORT")


def _diagnose(values):
    result = paired_scene_bootstrap(values, draws=10000, seed=20260927)
    a = np.array(list(values.values()), dtype=float)
    tolerance = 1e-8
    result["improved_tied_worse"] = {
        "improved": int((a > tolerance).sum()),
        "tied": int((np.abs(a) <= tolerance).sum()),
        "worse": int((a < -tolerance).sum()),
        "tie_absolute_tolerance": tolerance,
    }
    positive = np.maximum(a, 0)
    absolute = np.abs(a)
    result["concentration"] = {}
    for name, masses in [("positive", positive), ("absolute", absolute)]:
        ordered = np.sort(masses)[::-1]
        total = float(ordered.sum())
        result["concentration"][name] = {
            "total": total,
            "top1_fraction": float(ordered[:1].sum() / total) if total else None,
            "top3_fraction": float(ordered[:3].sum() / total) if total else None,
        }
    result["direction"] = (
        "positive favors the left-defined gain; not necessarily better for every statistic"
    )
    return result


def _pearson(x, y):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if len(x) < 2 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _validate(rows):
    if not rows:
        raise ValueError("No support attribution rows")
    seen, identities = set(), {}
    for row in rows:
        key = tuple(row[k] for k in ("cohort", "scene_id", "design", "method", "query_id"))
        if key in seen:
            raise ValueError("Duplicate raw scene/design/method/query row")
        seen.add(key)
        if row["method"] not in METHODS:
            raise ValueError("Unknown static method")
        if row["cohort"] not in ("exposed", "new_dev"):
            raise PermissionError("Attribution statistics reject holdout/unknown cohorts")
        previous = identities.setdefault(row["scene_id"], row["cohort"])
        if previous != row["cohort"]:
            raise PermissionError("Same scene repeated under multiple cohorts")
        for metric in METRICS:
            value = row[metric]
            if value is None or not np.isfinite(value):
                raise ValueError(f"Nonfinite or missing required metric {metric}")
        if row.get("candidate_count") != 128:
            raise ValueError("Candidate count differs from fixed budget")
        extent = np.asarray(row["volume_extent"], dtype=float)
        if extent.shape != (3,) or not np.isfinite(extent).all() or np.any(extent <= 0):
            raise ValueError("Invalid volume extent")
    queries = defaultdict(dict)
    for row in rows:
        key = row["cohort"], row["scene_id"], row["design"]
        queries[key].setdefault(row["method"], set()).add(row["query_id"])
    reference = {}
    for (cohort, sid, _design), by_method in queries.items():
        if (
            set(by_method) != set(METHODS)
            or len({tuple(sorted(v)) for v in by_method.values()}) != 1
        ):
            raise ValueError("Incomplete or unpaired methods/queries")
        ids = tuple(sorted(by_method["A"]))
        ref = reference.setdefault((cohort, sid), ids)
        if ids != ref:
            raise ValueError("Designs evaluated on different queries")
    designs = {r["design"] for r in rows}
    if "R0" not in designs:
        raise ValueError("R0 reference required")
    scenes = {r["scene_id"] for r in rows}
    if any({r["scene_id"] for r in rows if r["design"] == d} != scenes for d in designs):
        raise ValueError("Designs evaluated on different scenes")


def _analyze_group(rows):
    scenes = sorted({r["scene_id"] for r in rows})
    designs = {}
    for design in sorted({r["design"] for r in rows}):
        by_method = {}
        for method in METHODS:
            selected = [r for r in rows if r["design"] == design and r["method"] == method]
            metrics = {}
            for metric in METRICS:
                values = {
                    sid: float(np.mean([r[metric] for r in selected if r["scene_id"] == sid]))
                    for sid in scenes
                }
                metrics[metric] = {
                    "mean": float(np.mean(list(values.values()))),
                    "per_scene": values,
                }
            metrics["volume_extent"] = {
                "per_scene": {
                    sid: np.mean(
                        [r["volume_extent"] for r in selected if r["scene_id"] == sid], axis=0
                    ).tolist()
                    for sid in scenes
                },
                "note": "axis extents in meters; no candidate budget increase",
            }
            metrics["candidate_count"] = 128
            by_method[method] = metrics

        def metric_values(method, metric="depth_absrel", by_method=by_method):
            return by_method[method][metric]["per_scene"]

        a, b, anchor = (metric_values(method) for method in ("A", "B", "anchor"))
        gain = {sid: anchor[sid] - (a[sid] + b[sid]) / 2 for sid in scenes}
        wrong = {sid: metric_values("wrong_scene")[sid] - a[sid] for sid in scenes}
        ab_geometry = {}
        for metric in (
            "inside_fraction",
            "two_view_candidate_fraction",
            "supported_surface_fraction",
        ):
            values = {
                sid: (metric_values("A", metric)[sid] + metric_values("B", metric)[sid]) / 2
                for sid in scenes
            }
            ab_geometry[metric] = {
                "mean": float(np.mean(list(values.values()))),
                "per_scene": values,
            }
        adequacy = {
            "every_scene_AB_inside_at_least_0999": all(
                v >= 0.999 for v in ab_geometry["inside_fraction"]["per_scene"].values()
            ),
            "pooled_scene_equal_AB_two_view_at_least_09": ab_geometry[
                "two_view_candidate_fraction"
            ]["mean"]
            >= 0.9,
            "pooled_scene_equal_AB_surface_support_at_least_075": ab_geometry[
                "supported_surface_fraction"
            ]["mean"]
            >= 0.75,
        }
        adequate = all(adequacy.values())
        designs[design] = {
            "kind": "PRIVILEGED_DIAGNOSTIC_NOT_MATHEMATICAL_UPPER_BOUND"
            if design in ORACLES
            else "DEPLOYABLE_SUPPORT_DEFINITION",
            "methods": by_method,
            "full_context_gain": _diagnose(gain),
            "wrong_scene_damage": _diagnose(wrong),
            "AB_geometry": ab_geometry,
            "oracle_adequacy": {
                "applicable": design in ORACLES,
                "adequate": adequate,
                "checks": adequacy,
                "radius_m": 1.1726,
            },
        }
    reference = designs["R0"]
    for result in designs.values():
        gain = result["full_context_gain"]["per_scene"]
        baseline = reference["full_context_gain"]["per_scene"]
        differences = {sid: gain[sid] - baseline[sid] for sid in scenes}
        result["change_full_context_gain_vs_R0"] = _diagnose(differences)
        correlation = {}
        for metric in (
            "inside_fraction",
            "two_view_candidate_fraction",
            "supported_surface_fraction",
        ):
            improved_support = {
                sid: result["AB_geometry"][metric]["per_scene"][sid]
                - reference["AB_geometry"][metric]["per_scene"][sid]
                for sid in scenes
            }
            correlation[metric] = {
                "pearson": _pearson(list(improved_support.values()), list(differences.values())),
                "support_change_per_scene": improved_support,
            }
        result["support_gain_correlation"] = {
            "interpretation": "DESCRIPTIVE_ONLY_NOT_CAUSAL",
            "by_support_metric": correlation,
        }
    combined = designs.get("ORACLE_VOLUME_SUPPORT")
    if combined is None:
        status = "INCONCLUSIVE"
        reason = "Combined oracle diagnostic absent"
    elif not combined["oracle_adequacy"]["adequate"]:
        status = "INCONCLUSIVE"
        reason = "Combined oracle did not meet predefined support/coverage adequacy"
    elif combined["full_context_gain"]["ci95"][0] <= 0:
        status = "NOT_SUFFICIENT"
        reason = "Adequate combined oracle did not establish stable full-context gain"
    else:
        status = "ORACLE_RECOVERY_GT_FREE_UNTESTED"
        reason = (
            "Adequate combined oracle restores stable gain; deployment attribution still required"
        )
    return {
        "n_scenes": len(scenes),
        "scene_ids": scenes,
        "designs": designs,
        "oracle_attribution_status": status,
        "oracle_attribution_reason": reason,
        "adequacy_rule": (
            "every scene mean AB inside>=.999; scene-equal AB two-view>=.9; supported surface>=.75"
        ),
        "aggregation": (
            "query mean, A/B mean when specified, scene equal; pooled never weights cohorts equally"
        ),
    }


def analyze(rows):
    """Validate complete paired raw table and regenerate cohort/pooled attribution."""
    rows = list(rows)
    _validate(rows)
    cohorts = {
        cohort: _analyze_group([r for r in rows if r["cohort"] == cohort])
        for cohort in sorted({r["cohort"] for r in rows})
    }
    return {
        "schema": "mcss.support_redesign.statistics.v1",
        "cohorts": cohorts,
        "pooled": _analyze_group(rows),
        "dynamic_ttt_run": False,
        "oracle_note": "GT-privileged diagnostics are not guaranteed mathematical upper bounds",
    }
