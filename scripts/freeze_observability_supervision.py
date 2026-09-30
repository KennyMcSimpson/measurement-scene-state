"""Freeze observability rules and matched supervision before any new optimization/evaluation."""
# ruff: noqa: E501 -- keep frozen scientific contract prose intact

import argparse
import datetime
import json
from dataclasses import asdict
from pathlib import Path

from mcss.mechanism_pilot.direct_capacity_optimization import DirectOptimizationConfig
from mcss.mechanism_pilot.small_training import sha, write_json


def freeze(root):
    root = Path(root)
    if (root / "preregistration.json").exists():
        raise FileExistsError("Already frozen")
    manifest = json.loads((root / "scene_manifest.json").read_text())
    bounds = json.loads((root / "bounds_lock.json").read_text())["bounds"]
    states = []
    for variant in ("S0", "S1", "S2", "S3"):
        for record in manifest["scenes"]:
            for role in ("A", "B"):
                sid = record["scene_id"]
                key = f"{sid}__{role}__{variant}"
                directory = "checkpoints/" + key
                states.append(
                    {
                        "key": key,
                        "scene_id": sid,
                        "role": role,
                        "variant": variant,
                        "grid": 16,
                        "samples": 64,
                        "bounds": bounds[f"{sid}/{role}"]["FROZEN_GT_FREE_BOUNDS"],
                        "optimization_path": directory,
                        "state_path": directory + "/state.pt",
                    }
                )
    write_json(root / "state_plan.json", {"states": states})
    write_json(
        root / "supervision_contract.json",
        {
            "grid": 16,
            "samples": 64,
            "budget": 10000,
            "selection": "FIXED_BUDGET",
            "bounds": "EXACT_PREVIOUS_FROZEN_GT_FREE_BOUNDS",
            "variants": {
                "S0": "RENDER_ONLY",
                "S1": "FREE_SPACE",
                "S2": "SURFACE",
                "S3": "FREE_SPACE_PLUS_SURFACE",
            },
            "primary": "S3_PREDECLARED_NOT_HINDSIGHT_BEST",
            "weights": {"lambda_free": 0.1, "lambda_surface": 0.1, "epsilon": 1e-8},
            "tau_surface": ".5 * L2norm((bounds_max-bounds_min)/16); physical voxel spacing matches existing renderer/regularizer",
            "sampling": "same old seed per scene|role|RGBD and local torch.Generator; same 1024 randint indices per step across all variants; SHA256 full little-endian int64 index stream",
            "free_loss": "Per valid context-depth ray: mean alpha over t < depth-tau; empty sample subset gives0. Average over ALL valid context-depth rays including rays missing bounds. No post-surface free-space labels.",
            "surface_loss": "Mean -log(sum weights where abs(t-depth)<=tau +1e-8) only over valid-depth hit rays with >=1 band sample. Out-of-bounds/empty-band rays excluded from this additional term only and counted; all retained in RGB/rendered-depth task loss and overall evaluation.",
            "old_loss_retained": True,
            "weights_selected_from_query": False,
            "weight_search": False,
            "smoke": "Synthetic engineering stability/gradient/equivalence only; fixed .1/.1 defaults, no real query tuning",
            "context_only": True,
            "behind_surface": "UNKNOWN_NOT_EMPTY",
            "query_GT_allowed_in_loss": False,
            "fixed_final_not_objective_selected": True,
            "init": "old DirectState density_logits=-2 color_logits=0 logvariance=-3",
            "controls": {
                "wrong_scene": "all S0-S3, cyclic sorted scene donor matching role/variant; recipient camera unchanged",
                "spatial_shuffle": "S3 joint spatial permutation all state fields, fixed seed20260927",
                "zero": "S3 density_logits=-1e6,color=0,geometry/bounds/camera unchanged",
            },
        },
    )
    write_json(
        root / "observability_contract.json",
        {
            "purpose": "POST_ALL_STATE_SEAL_DIAGNOSTIC_REGION_LABELING_ONLY",
            "GT_used_in_optimizer": False,
            "depth": "metric normalized ray-distance",
            "context_count": "exact locked three frames for each A/B role, no selection",
            "query_surface": "world camera ray origin+normalized direction*query GT depth",
            "visibility": {
                "abs_tol_m": 0.05,
                "relative_tol": 0.01,
                "sampling": "nearest pixel floor(projected_uv+.5) after image/front checks; same as old visibility.py",
                "criterion": "in_front & inside_image & valid contextdepth & abs(projected Euclidean distance-context GT)<=.05+.01*context GT",
            },
            "occluded": "in_front & inside & valid depth & projected distance > context GT + tolerance",
            "conflict_front": "in_front & inside & valid depth & projected distance < context GT - tolerance",
            "classes": {"OBS0": "0visible", "OBS1": "1visible", "OBS2PLUS": ">=2visible"},
            "triangulation": "visible context-camera pairs subtended at true query point; only>=2visible",
            "angle_bins_degrees": ["[0,5]", "(5,15]", "(15,30]", "(30,180]"],
            "primary_angle_binning": "maximum triangulation angle; median bins also saved",
            "nearest_context_angle": "minimum query-point versus context-point viewing angle over all context cameras",
            "nearest_context_distance": "minimum query surface to context camera Euclidean distance",
            "camera_baseline": "maximum pair distance among visible context cameras; null/NaN if fewer than2",
            "bounds_inside": "world query point transformed to frozen context-anchor coordinates",
            "invalid_pixels": "arrays retained with validity mask; only GT-invalid pixels excluded from error per prior metric convention",
            "empty_regions": "all17 scenes retained; conditional metric null, coverage and empty count explicit; additive error contributions include zero empty-region contribution",
            "association": "Within each role/query Spearman(visible view count,S0 per-pixel AbsRel) on all GT-valid pixels; constant inputs undefined. Per scene query then role average only if ALL role/query coefficients defined, otherwise scene null; >=75% scene coverage required for correlation classification.",
            "no_region_selection_for_overall": True,
            "no_model_mutation": True,
            "final_holdout_forbidden": True,
        },
    )
    sources = [
        "src/mcss/mechanism_pilot/geometric_supervision_optimization.py",
        "src/mcss/mechanism_pilot/context_observability.py",
        "src/mcss/mechanism_pilot/observability_supervision_evaluation.py",
        "src/mcss/mechanism_pilot/observability_supervision_statistics.py",
        "src/mcss/mechanism_pilot/direct_capacity_contracts.py",
        "src/mcss/mechanism_pilot/direct_capacity_optimization.py",
        "src/mcss/mechanism_pilot/direct_capacity_evaluation.py",
        "src/mcss/mechanism_pilot/direct_capacity_statistics.py",
        "src/mcss/mechanism_pilot/statistics.py",
        "src/mcss/mechanism_pilot/visibility.py",
        "src/mcss/dynamic/types.py",
        "src/mcss/types.py",
        "src/mcss/geometry.py",
        "src/mcss/measurements.py",
        "scripts/run_observability_supervision.py",
        "scripts/evaluate_observability_supervision.py",
        "scripts/audit_context_observability.py",
        "scripts/analyze_observability_supervision.py",
    ]
    config = {
        "optimization": asdict(DirectOptimizationConfig(steps=10000)),
        "parallel_workers": 4,
        "primary_variant": "S3",
        "baseline_reproduction": {
            "mean_atol": 0.001,
            "per_row_atol": 0.01,
            "metrics": "AbsRel context/query freshS0 vs savedoldGTfree10kFIXED; other metric deltas report only",
            "on_failure": "STOP_NEW_SUPERVISION_NO_THRESHOLD_RELAXATION",
        },
        "source_sha256": {p: sha(Path(p)) for p in sources},
        "state_plan_sha256": sha(root / "state_plan.json"),
        "scene_manifest_sha256": sha(root / "scene_manifest.json"),
        "shuffle_seed": 20260927,
    }
    write_json(root / "config.json", config)
    locked = [
        "config.json",
        "scene_manifest.json",
        "state_plan.json",
        "bounds_lock.json",
        "bounds_contract.json",
        "train_depth_prior.json",
        "decision_rules.json",
        "observability_contract.json",
        "supervision_contract.json",
        "raw/historical_baseline_query.json",
        "raw/historical_baseline_context.json",
        "raw/historical_oracle_query.json",
    ]
    write_json(
        root / "preregistration.json",
        {
            "experiment": root.name,
            "created_utc": datetime.datetime.now(datetime.UTC).isoformat(),
            "stage": "BEFORE_FRESH_S0_AND_ALL_NEW_QUERY_REGIONS",
            "locked_file_sha256": {p: sha(root / p) for p in locked},
            "new_states": 136,
            "data_role": "CAPACITY_DEV_EXPOSED_ATTRIBUTION_ONLY",
            "phase_order": [
                "freshS0_34states",
                "fresh_baseline_reproduction_gate",
                "S1_S2_S3_102states",
                "all136seal",
                "evaluation_and_observability",
                "frozen_statistics",
            ],
            "no_new_query_feedback_changes": True,
            "FINAL_HOLDOUT_TOUCHED": False,
            "NEW_CARRIER_TRAINED": False,
            "DYNAMIC_TTT_RUN": False,
        },
    )
    print("FROZEN", len(states), "states")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True)
    a = p.parse_args()
    freeze(a.root)
