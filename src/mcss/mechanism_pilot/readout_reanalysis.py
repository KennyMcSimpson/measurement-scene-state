"""Post-hoc DEV re-analysis of the sealed V2-V4 carrier predictions; never model code.

Only the sealed per-query prediction files (depth, opacity) of the frozen per-seed DEV
selections, the frozen DEV primary-query GT depth, and two geometry-free constants fitted on
TRAIN primary-query GT are read.  Nothing here can change a frozen V2-V4 status: every label
is a secondary, preregistered description of the same sealed predictions.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from mcss.mechanism_pilot.direct_capacity_statistics import TIE, paired
from mcss.mechanism_pilot.geometry_carrier_statistics import _evidence, _macro

EXPERIMENT = "EXP-3D-READOUT-REFERENCE-REANALYSIS-V1"
READOUTS = ("RAW", "OPACITY_NORMALIZED", "MEDIAN_SCALED")
METRICS = ("depth_absrel", "depth_delta1")
METHODS = ("direct", "anchor")
VARIANTS = ("C0", "C1", "C2")
OPACITY_FLOOR = 1e-6
SOURCES = {
    "V2": {
        "root": "outputs/EXP-3D-GEOMETRY-AWARE-CARRIER-V2",
        "labels": {"C0": "BASELINE16", "C1": "SURFACE16", "C2": "FREE_SURFACE16"},
        "primary": "SURFACE_GAIN",
    },
    "V3": {
        "root": "outputs/EXP-3D-DENSE-EVIDENCE-CARRIER-V3",
        "labels": {"C0": "SPARSE_SURFACE", "C1": "DENSE_SURFACE", "C2": "DENSE_NO_SURFACE"},
        "primary": "DENSITY_GAIN",
    },
    "V4": {
        "root": "outputs/EXP-3D-PLANE-SWEEP-CARRIER-V4",
        "labels": {"C0": "DENSE_SURFACE", "C1": "SWEEP_SURFACE", "C2": "SWEEP_NO_SURFACE"},
        "primary": "SWEEP_GAIN",
    },
}
REFERENCES = ("REF_TRAIN_ABSREL_OPTIMAL", "REF_TRAIN_MEDIAN")
# Each metric is compared with the constant that is optimal for it on TRAIN (conservative).
PRIMARY_REFERENCE = {"depth_absrel": "REF_TRAIN_ABSREL_OPTIMAL", "depth_delta1": "REF_TRAIN_MEDIAN"}
LOWER_IS_BETTER = {"depth_absrel": True, "depth_delta1": False}


def valid_mask(gt):
    gt = np.asarray(gt)
    return np.isfinite(gt) & (gt > 0)


def apply_readout(name, depth, opacity, gt):
    """Preregistered depth readouts of one sealed rendered query view."""
    depth = np.asarray(depth, dtype=np.float64)
    if name == "RAW":
        return depth
    if name == "OPACITY_NORMALIZED":
        opacity = np.asarray(opacity, dtype=np.float64)
        if opacity.shape != depth.shape or not np.isfinite(opacity).all():
            raise ValueError("Finite camera-aligned opacity required")
        return depth / np.maximum(opacity, OPACITY_FLOOR)
    if name == "MEDIAN_SCALED":
        valid = valid_mask(gt)
        if not valid.any():
            raise ValueError("GT-valid pixels required")
        if not median_scale_defined(depth, gt):
            # Amendment 1: no positive predicted median (e.g. an all-zero no-hit view): no
            # scale exists, so the view stays unscaled; identical wherever the rule is defined.
            return depth
        median = float(np.median(depth[valid]))
        return depth * (float(np.median(np.asarray(gt, dtype=np.float64)[valid])) / median)
    raise ValueError(f"Unknown readout {name}")


def median_scale_defined(depth, gt):
    valid = valid_mask(gt)
    median = float(np.median(np.asarray(depth, dtype=np.float64)[valid]))
    return bool(np.isfinite(median) and median > 0)


def depth_metrics(prediction, gt):
    """Exactly the depth fields of statistics.measurement_metrics on GT-valid pixels."""
    prediction = np.asarray(prediction, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    if prediction.shape != gt.shape:
        raise ValueError("Prediction and GT shapes disagree")
    valid = valid_mask(gt)
    if not valid.any():
        raise ValueError("No GT-valid pixel")
    if not np.isfinite(prediction[valid]).all():
        raise ValueError("Nonfinite prediction on GT-valid pixels")
    p, t = prediction[valid], gt[valid]
    ratio = np.full_like(p, np.inf)
    positive = p > 0
    ratio[positive] = np.maximum(p[positive] / t[positive], t[positive] / p[positive])
    return {
        "depth_absrel": float(np.mean(np.abs(p - t) / t)),
        "depth_delta1": float(np.mean(ratio < 1.25)),
    }


def absrel_optimal_constant(values):
    """argmin_c mean(|c - t| / t): the 1/t-weighted median of the pooled TRAIN depths."""
    t = np.sort(np.asarray(values, dtype=np.float64))
    if not len(t) or not np.isfinite(t).all() or (t <= 0).any():
        raise ValueError("Positive finite TRAIN depths required")
    cumulative = np.cumsum(1.0 / t)
    return float(t[np.searchsorted(cumulative, 0.5 * cumulative[-1])])


def reference_values(train_depths):
    pooled = np.concatenate([np.asarray(d, dtype=np.float64).ravel() for d in train_depths])
    pooled = pooled[valid_mask(pooled)]
    return {
        "REF_TRAIN_ABSREL_OPTIMAL": absrel_optimal_constant(pooled),
        "REF_TRAIN_MEDIAN": float(np.median(pooled)),
        "train_pixel_count": int(pooled.size),
    }


def score_predictions(predictions, gt):
    """Rows for every sealed prediction and readout; RAW must reproduce the sealed metric."""
    rows, worst = [], {"absrel": 0.0, "delta1_pixels": 0.0, "median_scale_undefined_views": 0}
    for p in predictions:
        with np.load(p["path"]) as data:
            depth, opacity = data["depth"], data["opacity"]
        truth = gt[p["scene_id"], p["query_id"]]
        base = {
            k: p[k]
            for k in ("source", "variant", "seed", "scene_id", "role", "query_id", "method", "step")
        }
        base["all_zero_prediction"] = bool(not np.any(depth) and not np.any(opacity))
        worst["median_scale_undefined_views"] += int(not median_scale_defined(depth, truth))
        for readout in READOUTS:
            metrics = depth_metrics(apply_readout(readout, depth, opacity, truth), truth)
            if readout == "RAW":
                absrel = abs(metrics["depth_absrel"] - p["sealed_depth_absrel"])
                pixels = (
                    abs(metrics["depth_delta1"] - p["sealed_depth_delta1"])
                    * p["sealed_depth_valid_count"]
                )
                worst["absrel"] = max(worst["absrel"], absrel)
                worst["delta1_pixels"] = max(worst["delta1_pixels"], pixels)
                if absrel > 1e-5 or pixels > 1.0 + 1e-6:
                    raise RuntimeError(f"RAW does not reproduce the sealed metric: {p['path']}")
            rows.append({**base, "readout": readout, **metrics})
    return rows, worst


def score_references(values, gt, queries):
    """Constant-depth rows per DEV primary query, duplicated for the A/B role pairing."""
    rows = []
    for scene, frames in sorted(queries.items()):
        for fid in frames:
            truth = gt[scene, fid]
            for reference in REFERENCES:
                constant = np.full(truth.shape, values[reference], dtype=np.float64)
                for readout in READOUTS:
                    prediction = apply_readout(readout, constant, np.ones_like(constant), truth)
                    metrics = depth_metrics(prediction, truth)
                    for role in ("A", "B"):
                        rows.append(
                            {
                                "reference": reference,
                                "scene_id": scene,
                                "role": role,
                                "query_id": fid,
                                "readout": readout,
                                **metrics,
                            }
                        )
    return rows


def _values(rows, metric, **where):
    selected = [r for r in rows if all(r[k] == v for k, v in where.items())]
    if not selected:
        raise ValueError(f"No rows for {where}")
    return _macro(selected, metric)


def _improvement(better, worse, metric, *, draws, seed):
    """Positive means `better` has the better metric (paired over identical scenes)."""
    if LOWER_IS_BETTER[metric]:
        return paired(worse, better, draws=draws, seed=seed)
    return paired(better, worse, draws=draws, seed=seed)


def _label(stats):
    x = np.asarray(list(stats["per_scene"].values()))
    reverse = stats["mean"] < 0 and stats["ci95"][1] < 0 and np.mean(x <= TIE) >= 0.75
    if _evidence(stats):
        return "ABOVE"
    return "BELOW" if reverse else "NOT_DISTINGUISHABLE"


def validate_matrix(
    rows, reference_rows, seeds, scenes, queries, sources=SOURCES, variants=VARIANTS
):
    expected = {
        (source, variant, method, seed, scene, role, query, readout)
        for source in sources
        for variant in variants
        for method in METHODS
        for seed in seeds
        for scene in scenes
        for role in ("A", "B")
        for query in queries[scene]
        for readout in READOUTS
    }
    got = [
        (
            r["source"],
            r["variant"],
            r["method"],
            r["seed"],
            r["scene_id"],
            r["role"],
            r["query_id"],
            r["readout"],
        )
        for r in rows
    ]
    if len(got) != len(set(got)) or set(got) != expected:
        raise ValueError("Complete unique source/variant/method/seed/scene/role/query matrix")
    expected_ref = {
        (ref, scene, role, query, readout)
        for ref in REFERENCES
        for scene in scenes
        for role in ("A", "B")
        for query in queries[scene]
        for readout in READOUTS
    }
    got_ref = [
        (r["reference"], r["scene_id"], r["role"], r["query_id"], r["readout"])
        for r in reference_rows
    ]
    if len(got_ref) != len(set(got_ref)) or set(got_ref) != expected_ref:
        raise ValueError("Complete unique reference matrix required")


def analyze(
    rows, reference_rows, *, sources=SOURCES, variants=VARIANTS, draws=10000, seed=20260929
):
    """Scene-paired bootstrap with the frozen V2 macro and paired estimators."""
    if not {"C0", "C1"} <= set(variants):
        raise ValueError("The primary C0/C1 pair is always analyzed")
    args = {"draws": draws, "seed": seed}
    result = {"carriers": {}, "primary_contrasts": {}, "readout_effect": {}, "static": {}}
    for source in sources:
        for variant in variants:
            name = f"{source}_{variant}"
            carrier, effect, static = {}, {}, {}
            for readout in READOUTS:
                carrier[readout] = {}
                static[readout] = {}
                for metric in METRICS:
                    direct = _values(
                        rows,
                        metric,
                        source=source,
                        variant=variant,
                        method="direct",
                        readout=readout,
                    )
                    anchor = _values(
                        rows,
                        metric,
                        source=source,
                        variant=variant,
                        method="anchor",
                        readout=readout,
                    )
                    entry = {"absolute_mean": float(np.mean(list(direct.values())))}
                    for ref in REFERENCES:
                        ref_values = _values(reference_rows, metric, reference=ref, readout=readout)
                        stats = _improvement(direct, ref_values, metric, **args)
                        entry[f"vs_{ref}"] = stats
                        entry[f"vs_{ref}_label"] = _label(stats)
                    entry["reference_label"] = entry[f"vs_{PRIMARY_REFERENCE[metric]}_label"]
                    carrier[readout][metric] = entry
                    stats = _improvement(direct, anchor, metric, **args)
                    static[readout][metric] = {"stats": stats, "label": _label(stats)}
            for metric in METRICS:
                raw = _values(
                    rows, metric, source=source, variant=variant, method="direct", readout="RAW"
                )
                normalized = _values(
                    rows,
                    metric,
                    source=source,
                    variant=variant,
                    method="direct",
                    readout="OPACITY_NORMALIZED",
                )
                stats = _improvement(normalized, raw, metric, **args)
                effect[metric] = {"stats": stats, "label": _label(stats)}
            result["carriers"][name] = carrier
            result["readout_effect"][name] = effect
            result["static"][name] = static
        contrasts = {}
        for readout in READOUTS:
            contrasts[readout] = {}
            for metric in METRICS:
                c0 = _values(
                    rows, metric, source=source, variant="C0", method="direct", readout=readout
                )
                c1 = _values(
                    rows, metric, source=source, variant="C1", method="direct", readout=readout
                )
                stats = _improvement(c1, c0, metric, **args)
                contrasts[readout][metric] = {
                    "stats": stats,
                    "label": "SUPPORTED" if _evidence(stats) else "NOT_ESTABLISHED",
                }
        result["primary_contrasts"][source] = {
            "name": sources[source]["primary"],
            "definition": "C0 minus C1 (AbsRel) / C1 minus C0 (delta1); positive favors C1",
            "by_readout": contrasts,
            "note": "secondary description; the frozen V2-V4 statuses are unchanged",
        }
    result["summary"] = summarize(result)
    return result


def no_hit_scenes(rows):
    """Scenes in which every sealed direct prediction renders nothing (all rays miss bounds)."""
    scenes = defaultdict(list)
    for row in rows:
        if row["method"] == "direct" and row["readout"] == "RAW":
            scenes[row["scene_id"]].append(row["all_zero_prediction"])
    return sorted(scene for scene, flags in scenes.items() if flags and all(flags))


def analyze_with_sensitivity(rows, reference_rows, **kwargs):
    """Frozen primary analysis on all scenes, plus the Amendment 1 no-hit sensitivity."""
    result = analyze(rows, reference_rows, **kwargs)
    excluded = no_hit_scenes(rows)
    sensitivity = {"excluded_no_hit_scenes": excluded}
    if excluded:
        kept = [r for r in rows if r["scene_id"] not in excluded]
        kept_ref = [r for r in reference_rows if r["scene_id"] not in excluded]
        other = analyze(kept, kept_ref, **kwargs)
        sensitivity["summary"] = other["summary"]
        sensitivity["primary_contrasts"] = other["primary_contrasts"]
        sensitivity["carriers"] = other["carriers"]
    result["sensitivity_excluding_no_hit_scenes"] = sensitivity
    return result


def summarize(result):
    labels = defaultdict(dict)
    for name, carrier in result["carriers"].items():
        for readout in READOUTS:
            for metric in METRICS:
                labels[name][f"{readout}:{metric}"] = carrier[readout][metric]["reference_label"]
    above = {
        (readout, metric): sum(
            labels[name][f"{readout}:{metric}"] == "ABOVE" for name in result["carriers"]
        )
        for readout in READOUTS
        for metric in METRICS
    }
    raw_above = above["RAW", "depth_absrel"] + above["RAW", "depth_delta1"]
    normalized_above = (
        above["OPACITY_NORMALIZED", "depth_absrel"] + above["OPACITY_NORMALIZED", "depth_delta1"]
    )
    if raw_above == 0 and normalized_above == 0:
        status = "NO_CARRIER_ABOVE_GEOMETRY_FREE_REFERENCE"
    elif raw_above == 0:
        status = "ABOVE_REFERENCE_ONLY_AFTER_OPACITY_NORMALIZATION"
    else:
        status = "SOME_CARRIER_ABOVE_GEOMETRY_FREE_REFERENCE"
    return {
        "CARRIER_REFERENCE_STATUS": status,
        "carriers_above_reference": {f"{r}:{m}": n for (r, m), n in above.items()},
        "labels": dict(labels),
        "carrier_count": len(result["carriers"]),
    }
