"""Scene-paired DEV analysis for matched RGB-input learned carriers, never model code."""

from collections import defaultdict

import numpy as np

from mcss.mechanism_pilot.direct_capacity_statistics import describe, paired, scene_values
from mcss.mechanism_pilot.observability_supervision_statistics import conditional_difference

METRICS = (
    "depth_absrel",
    "rgb_mse",
    "rgb_ssim",
    "depth_rmse",
    "depth_delta1",
    "opacity",
    "coverage",
)
METHODS = ("direct", "anchor", "wrong_scene", "spatial_shuffle", "zero")
STATE_CHECKS = (
    "shared_state_multiple_queries",
    "query_after_state_seal",
    "recipient_camera_preserved",
)


def dev_records(manifest):
    rows = [r for r in manifest["scenes"] if r.get("split", "DEV") == "DEV"]
    records = {r["scene_id"]: r for r in rows}
    if not records or len(records) != len(rows):
        raise ValueError("Unique nonempty DEV roster required")
    forbidden = set(manifest.get("FINAL_HOLDOUT_PROHIBITED", []))
    forbidden |= {
        r["scene_id"]
        for r in manifest["scenes"]
        if r.get("split") != "DEV" and r.get("split") is not None
    }
    if set(records) & forbidden:
        raise PermissionError("TRAIN/fresh/holdout cannot enter DEV analysis")
    return records


def _macro(rows, metric, *, observation="query_id", allow_null=False):
    """First equal seeds per identical query/role, then equal queries, roles, scenes."""
    groups = defaultdict(list)
    for row in rows:
        groups[row["scene_id"], row["role"], row[observation]].append(row[metric])
    averaged, invalid = [], set()
    for (scene, role, query), values in groups.items():
        if any(v is None for v in values):
            if not allow_null:
                raise ValueError("No missing primary metric allowed")
            invalid.add(scene)
            continue
        if not np.isfinite(values).all():
            raise ValueError("Nonfinite metrics cannot be silently discarded")
        averaged.append(
            {"scene_id": scene, "role": role, "query_id": query, "value": float(np.mean(values))}
        )
    values = scene_values(averaged, "value") if averaged else {}
    for scene in invalid:
        values[scene] = None
    return values


def _evidence(stats):
    return (
        stats["mean"] > 0
        and stats["ci95"][0] > 0
        and (stats["positive"] + stats["tied"]) / stats["n_scenes"] >= 0.75
    )


def surface_status(gain, checks):
    if gain["mean"] < 0 and gain["ci95"][1] < 0:
        return "HARMFUL"
    if all(checks.values()):
        return "SUPPORTED"
    if gain["mean"] > 0 and gain["ci95"][0] > 0:
        return "PARTIAL"
    return "NOT_ESTABLISHED"


def static_status(gain, checks):
    if all(checks.values()):
        return "SUPPORTED"
    if gain["mean"] > 0 and gain["ci95"][0] > 0:
        return "PARTIAL"
    return "NOT_ESTABLISHED"


def validate(query, context, matched, manifest, seeds):
    records = dev_records(manifest)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Nonempty unique frozen seeds required")
    variants = {r["variant"] for r in query}
    if not {"C0", "C1"} <= variants or not variants <= {"C0", "C1", "C2"}:
        raise ValueError("Primary must include matched C0/C1; optional C2 remains secondary")
    identities = {
        (seed, sid, role, fid)
        for seed in seeds
        for sid, record in records.items()
        for role in ("A", "B")
        for fid in record["roles"]["primary_query"]
    }
    if any(len(set(r["roles"]["primary_query"])) < 2 for r in records.values()):
        raise ValueError("At least two locked queries per scene required")
    for rows, expected, id_field in (
        (
            query,
            {(v, m, *key) for v in variants for m in METHODS for key in identities},
            "query_id",
        ),
        (
            context,
            {
                (v, "direct", seed, sid, role, fid)
                for v in variants
                for seed in seeds
                for sid, rec in records.items()
                for role in ("A", "B")
                for fid in rec["roles"]["context_a" if role == "A" else "context_b"]
            },
            "frame_id",
        ),
        (matched, {(v, "direct", *key) for v in ("C0", "C1") for key in identities}, "query_id"),
    ):
        actual = {
            (r["variant"], r["method"], r["seed"], r["scene_id"], r["role"], r[id_field])
            for r in rows
        }
        if len(actual) != len(rows) or actual != expected:
            raise ValueError("Complete exact DEV/seed/variant/role/frame matrix required")
    pairs = defaultdict(dict)
    for row in matched:
        pairs[row["seed"], row["scene_id"], row["role"], row["query_id"]][row["variant"]] = row
        count, total = row["common_valid_count"], row["total_gt_valid"]
        if (
            type(count) is not int
            or type(total) is not int
            or not 0 <= count <= total
            or total <= 0
        ):
            raise ValueError("Valid common-mask counts required")
        for name in ("depth_absrel", "normalized_depth_absrel"):
            value = row[name]
            if (count == 0 and value is not None) or (
                count > 0 and (value is None or not np.isfinite(value) or value < 0)
            ):
                raise ValueError("Empty common mask must be null, never imputed zero")
    for values in pairs.values():
        if any(
            values["C0"][k] != values["C1"][k]
            for k in ("common_valid_count", "total_gt_valid", "common_mask_hash")
        ):
            raise ValueError("C0/C1 must use exactly the same opacity-intersection mask")
    return sorted(records), sorted(variants)


def analyze(
    query_rows,
    context_rows,
    matched_rows,
    manifest,
    *,
    seeds,
    state_use_audit,
    draws=10000,
    bootstrap_seed=20260928,
):
    ids, variants = validate(query_rows, context_rows, matched_rows, manifest, seeds)
    describe_args = dict(draws=draws, seed=bootstrap_seed)
    groups = {}
    for variant in variants:
        for method in METHODS:
            subset = [r for r in query_rows if r["variant"] == variant and r["method"] == method]
            groups[f"{variant}|{method}"] = {
                "variant": variant,
                "method": method,
                "query_metrics": {m: describe(_macro(subset, m), **describe_args) for m in METRICS},
            }
        cell = groups[f"{variant}|direct"]
        context = [r for r in context_rows if r["variant"] == variant]
        cell["context_metrics"] = {
            m: describe(_macro(context, m, observation="frame_id"), **describe_args)
            for m in METRICS
        }
        cell["generalization_gap"] = paired(
            cell["query_metrics"]["depth_absrel"]["per_scene"],
            cell["context_metrics"]["depth_absrel"]["per_scene"],
            **describe_args,
        )

    def contrast(left, right):
        return paired(
            groups[left]["query_metrics"]["depth_absrel"]["per_scene"],
            groups[right]["query_metrics"]["depth_absrel"]["per_scene"],
            **describe_args,
        )

    gain = contrast("C0|direct", "C1|direct")
    controls = {
        f"{v}|{m}": contrast(f"{v}|{m}", f"{v}|direct")
        for v in variants
        for m in ("wrong_scene", "spatial_shuffle", "zero")
    }
    full = {v: contrast(f"{v}|anchor", f"{v}|direct") for v in variants}
    common = {}
    for metric in ("depth_absrel", "normalized_depth_absrel"):
        left, right = [
            _macro([r for r in matched_rows if r["variant"] == v], metric, allow_null=True)
            for v in ("C0", "C1")
        ]
        common[metric] = conditional_difference(left, right, **describe_args)
    coverage = {
        v: groups[f"{v}|direct"]["query_metrics"]["coverage"]["per_scene"] for v in ("C0", "C1")
    }
    ray_hits = {
        v: _macro(
            [r for r in query_rows if r["variant"] == v and r["method"] == "direct"],
            "ray_hitfraction",
        )
        for v in ("C0", "C1")
    }
    if any(abs(ray_hits["C0"][s] - ray_hits["C1"][s]) > 1e-8 for s in ids):
        raise ValueError("Matched variants must share frozen geometry ray-hit fractions")
    opacity_checks = {
        "all_scenes_coverage_at_least_0_99_ray_hitfraction": all(
            coverage[v][s] >= 0.99 * ray_hits[v][s] for v in coverage for s in ids
        ),
        "all_scenes_coverage_drop_at_most_0_01": all(
            coverage["C1"][s] - coverage["C0"][s] >= -0.01 for s in ids
        ),
        "common_mask_gain_ci_positive_with_75pct_scene_coverage": common["depth_absrel"][
            "scene_coverage"
        ]
        >= 0.75
        and common["depth_absrel"]["ci95"] is not None
        and common["depth_absrel"]["ci95"][0] > 0,
        "opacity_normalized_gain_ci_positive_with_75pct_scene_coverage": common[
            "normalized_depth_absrel"
        ]["scene_coverage"]
        >= 0.75
        and common["normalized_depth_absrel"]["ci95"] is not None
        and common["normalized_depth_absrel"]["ci95"][0] > 0,
    }
    opacity_ok = all(opacity_checks.values())
    state_ok = state_use_audit.get("status") == "PASS" and all(
        state_use_audit.get(k) is True for k in STATE_CHECKS
    )
    surface_checks = {
        "mean_ci_and_scene_consistency": _evidence(gain),
        "every_leave_one_scene_out_gain_positive": bool(gain["loso"])
        and min(gain["loso"].values()) > 0,
        "positive_top1_share_at_most_0_5": gain["positive_top1_contribution"] is not None
        and gain["positive_top1_contribution"] <= 0.5,
        "wrong_scene_ci_positive": controls["C1|wrong_scene"]["ci95"][0] > 0,
        "spatial_shuffle_ci_positive": controls["C1|spatial_shuffle"]["ci95"][0] > 0,
        "opacity_artifact_gate": opacity_ok,
        "state_integrity": state_ok,
    }
    static_checks = {
        "full_context_mean_ci_scene_consistency": _evidence(full["C1"]),
        "wrong_scene_ci_positive": controls["C1|wrong_scene"]["ci95"][0] > 0,
        "shared_sealed_state_integrity": state_ok,
        "opacity_artifact_gate": opacity_ok,
    }
    per_seed = {}
    for seed in seeds:

        def seed_scene(v, method="direct", seed=seed):
            return _macro(
                [
                    r
                    for r in query_rows
                    if r["seed"] == seed and r["variant"] == v and r["method"] == method
                ],
                "depth_absrel",
            )

        per_seed[str(seed)] = {
            "surface_gain": paired(seed_scene("C0"), seed_scene("C1"), **describe_args),
            "full_context_gain": {
                v: paired(seed_scene(v, "anchor"), seed_scene(v), **describe_args) for v in variants
            },
            "query_absrel": {v: describe(seed_scene(v), **describe_args) for v in variants},
        }
    seed_robustness = (
        "DIRECTION_REPLICATED"
        if len(seeds) >= 2 and all(v["surface_gain"]["mean"] > 0 for v in per_seed.values())
        else "NOT_ESTABLISHED"
    )
    qualification = {
        "SURFACE_TRAINING_STATUS": surface_status(gain, surface_checks),
        "STATIC_DEV_STATUS": static_status(full["C1"], static_checks),
        "surface_checks": surface_checks,
        "static_checks": static_checks,
        "SEED_ROBUSTNESS": seed_robustness,
        "FRESH_QUALIFICATION_INDEPENDENCE": manifest.get(
            "fresh_status", "BLOCKED_INDEPENDENCE_UNRESOLVED"
        ),
        "FRESH_QUALIFICATION_OPENED": False,
        "FINAL_STATIC_STATUS": "NOT_ESTABLISHED",
        "DYNAMIC_TTT_NEXT_STAGE_ALLOWED": False,
        "DEV_does_not_establish_final_static_qualification": True,
    }
    return {
        "static_results": {
            "scene_ids": ids,
            "seeds": list(seeds),
            "primary": "C1_SURFACE16",
            "groups": groups,
            "surface_gain": gain,
            "full_context_gain": full,
            "per_seed": per_seed,
            "free_space_extra_gain": contrast("C1|direct", "C2|direct")
            if "C2" in variants
            else None,
            "aggregation": (
                "equal seeds per query, then queries per role, roles per scene; scene bootstrap"
            ),
        },
        "wrong_scene_results": {
            "contrasts": controls,
            "bounds_caveat": (
                "wrong-scene donor includes bounds; spatial shuffle preserves recipient bounds"
            ),
        },
        "state_use_results": {
            "input_audit": state_use_audit,
            "opacity_checks": opacity_checks,
            "opacity_artifact_gate": opacity_ok,
            "common_mask_gains": common,
            "normalized_depth_definition": (
                "rendered_depth / opacity.clamp_min(1e-6), "
                "common GT-valid C0/C1 opacity>1e-6 mask; diagnostic only"
            ),
            "primary_metric_unchanged": True,
            "ray_hitfraction_per_scene": ray_hits,
            "no_hit_scenes": [s for s in ids if ray_hits["C0"][s] == 0],
            "coverage_gate_design": (
                "Pre-training revision based on exposed historical ai_009_001 "
                "all-miss bounds; no V2 outcomes. Overall retains all scenes/pixels; "
                "no scene/role/bounds replacement."
            ),
        },
        "bootstrap_results": {
            "draws": draws,
            "seed": bootstrap_seed,
            "unit": "scene, not seeds or pixels",
            "surface_gain": gain,
            "full_context_gain": full,
            "controls": controls,
        },
        "qualification_results": qualification,
    }


def analyze_regions(rows, manifest, *, seeds, draws=10000, bootstrap_seed=20260928):
    """Optional OBS signature; no regional result can change primary DEV gates."""
    from mcss.mechanism_pilot.observability_supervision_statistics import conditional_summary

    records = dev_records(manifest)
    names = ("OBS0", "OBS1", "OBS2PLUS")
    expected = {
        (v, seed, sid, role, q, region)
        for v in ("C0", "C1")
        for seed in seeds
        for sid, rec in records.items()
        for role in ("A", "B")
        for q in rec["roles"]["primary_query"]
        for region in names
    }
    actual = {
        (r["variant"], r["seed"], r["scene_id"], r["role"], r["query_id"], r["region"])
        for r in rows
    }
    if actual != expected or len(actual) != len(rows):
        raise ValueError("Complete paired OBS diagnostic rows required, including empty masks")
    lookup = {
        (r["variant"], r["seed"], r["scene_id"], r["role"], r["query_id"], r["region"]): r
        for r in rows
    }
    for row in rows:
        count, total, error = row["pixel_count"], row["total_valid"], row["absrel_sum"]
        if (
            type(count) is not int
            or type(total) is not int
            or not 0 <= count <= total
            or total <= 0
            or not np.isfinite(error)
            or error < 0
        ):
            raise ValueError(
                "Finite sufficient statistics and positive overall denominator required"
            )
        other = lookup[
            (
                "C1" if row["variant"] == "C0" else "C0",
                row["seed"],
                row["scene_id"],
                row["role"],
                row["query_id"],
                row["region"],
            )
        ]
        if any(row[k] != other[k] for k in ("pixel_count", "total_valid")):
            raise ValueError("OBS masks cannot depend on carrier variant")
        if count == 0 and (error != 0 or row.get("absrel_mean") is not None):
            raise ValueError("Empty region must remain null with zero contribution")
        if row["region"] == "OBS0":
            parts = [
                lookup[
                    tuple(row[k] for k in ("variant", "seed", "scene_id", "role", "query_id"))
                    + (name,)
                ]
                for name in names
            ]
            if sum(p["pixel_count"] for p in parts) != total:
                raise ValueError("OBS regions must reconstruct unfiltered overall")
    options = dict(draws=draws, seed=bootstrap_seed)
    output = {v: {} for v in ("C0", "C1")}
    for variant in output:
        for name in names:
            subset = [
                {
                    **r,
                    "fraction": r["pixel_count"] / r["total_valid"],
                    "contribution": r["absrel_sum"] / r["total_valid"],
                }
                for r in rows
                if r["variant"] == variant and r["region"] == name
            ]
            fraction, contribution = (_macro(subset, k) for k in ("fraction", "contribution"))
            conditional = {
                s: contribution[s] / fraction[s] if fraction[s] > 0 else None for s in records
            }
            output[variant][name] = {
                "conditional_absrel": conditional_summary(conditional, **options),
                "pixel_fraction": describe(fraction, **options),
                "additive_error_contribution": describe(contribution, **options),
            }
    return {
        "groups": output,
        "gains": {
            r: conditional_difference(
                output["C0"][r]["conditional_absrel"]["per_scene"],
                output["C1"][r]["conditional_absrel"]["per_scene"],
                **options,
            )
            for r in names
        },
        "diagnostic_only": True,
        "does_not_change_qualification": True,
    }
