"""Recompute TRAIN-query/frame-heldout diagnostics from audited data and saved static raw.

No model initialization, checkpoint loading, inference, or Dynamic TTT is performed.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

COHORTS = {"heldout_query": (8, 9), "training_query": (12, 13, 14, 15)}


def read(path):
    return json.loads(Path(path).read_text())


def angle(left, right):
    left, right = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    cosine = np.dot(left, right) / (np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.degrees(np.arccos(np.clip(cosine, -1, 1))))


def analyze(manifest_path, experiment, previous_raw):
    manifest_path, experiment, previous_raw = (
        Path(manifest_path),
        Path(experiment),
        Path(previous_raw),
    )
    inputs = [
        manifest_path,
        experiment / "geometry_audit/volume_coverage.json",
        experiment / "geometry_audit/support_audit.json",
        experiment / "rgb_audit/frame_audit.json",
        previous_raw,
    ]
    manifest, volume, support, rgb, raw = map(read, inputs)
    assert all(s["split"] == "train" for s in manifest["scenes"])
    vg = {(r["scene_id"], r["frame_id"]): r for r in volume["frames"]}
    rg = {(r["scene_id"], r["frame_id"]): r for r in rgb}
    sr = {(r["scene_id"], r["query_id"]): r for r in raw if r["method"] == "A"}
    results = {}
    for scene in manifest["scenes"]:
        sid = scene["scene_id"]
        frames = {f["frame_id"]: f for f in scene["frames"]}
        groups = {label: scene["roles"][label] for label in ("context_a", "context_b")}
        result = {"support_counts": support[sid], "cohorts": {}}
        for label, ids in COHORTS.items():
            records = []
            for fid in ids:
                v, r, score = vg[sid, fid], rg[sid, fid], sr[sid, fid]
                pose = np.asarray(frames[fid]["c2w"])
                camera = {}
                for group, context_ids in groups.items():
                    poses = [np.asarray(frames[i]["c2w"]) for i in context_ids]
                    distances = [float(np.linalg.norm(pose[:3, 3] - p[:3, 3])) for p in poses]
                    angles = [angle(pose[:3, 2], p[:3, 2]) for p in poses]
                    camera[group] = {
                        "minimum_camera_baseline_m": min(distances),
                        "mean_camera_baseline_m": float(np.mean(distances)),
                        "minimum_view_angle_degrees": min(angles),
                        "mean_view_angle_degrees": float(np.mean(angles)),
                    }
                records.append(
                    {
                        "frame_id": fid,
                        "saved_context_A_depth_absrel": score["depth_absrel"],
                        "anchor_camera_baseline_m": v["anchor_camera_baseline_m"],
                        "anchor_view_angle_degrees": angle(v["anchor_relative_forward"], [0, 0, 1]),
                        "context_camera_comparison": camera,
                        "depth_distribution_m": v["depth_m"],
                        "inside_volume_fraction": v["inside_volume_fraction"],
                        "outside_volume_distance_m": v["outside_distance_m"],
                        "valid_GT_fraction": v["valid_GT_fraction"],
                        "candidate_neighborhood_fraction": v["any_candidate_neighborhood_fraction"],
                        "A_supported_surface_fraction": v["context_supported_neighborhood"][
                            "context_a"
                        ]["surface_support_fraction"],
                        "B_supported_surface_fraction": v["context_supported_neighborhood"][
                            "context_b"
                        ]["surface_support_fraction"],
                        "common_frustum_fraction": score["region_counts"]["common_frustum"]
                        / score["region_pixel_count"],
                        "approx_depth_consistent_common_fraction": score["region_counts"][
                            "depth_consistent_common"
                        ]
                        / score["region_pixel_count"],
                        "source_RGB_mean_0_255": r["source_rgb"]["mean"],
                        "source_RGB_zero_pixel_fraction": r["source_rgb"][
                            "exact_zero_pixels_percent"
                        ]
                        / 100,
                        "source_RGB_near_black_pixel_fraction": r["source_rgb"][
                            "near_black_pixels_percent"
                        ]
                        / 100,
                    }
                )
            numeric = [
                k for k, v in records[0].items() if isinstance(v, (int, float)) and k != "frame_id"
            ]
            means = {k: float(np.mean([r[k] for r in records])) for k in numeric}
            means.update(
                {
                    "depth_mean_m": float(
                        np.mean([r["depth_distribution_m"]["mean"] for r in records])
                    ),
                    "mean_frame_depth_median_m": float(
                        np.mean([r["depth_distribution_m"]["p50"] for r in records])
                    ),
                    "mean_frame_depth_p95_m": float(
                        np.mean([r["depth_distribution_m"]["p95"] for r in records])
                    ),
                }
            )
            for group in groups:
                for key in camera[group]:
                    means[f"{group}_{key}"] = float(
                        np.mean([r["context_camera_comparison"][group][key] for r in records])
                    )
            result["cohorts"][label] = {
                "frame_ids": ids,
                "frame_equal_means": means,
                "per_frame": records,
            }
        a = result["cohorts"]["heldout_query"]["frame_equal_means"]
        b = result["cohorts"]["training_query"]["frame_equal_means"]
        result["heldout_minus_training_absrel"] = (
            a["saved_context_A_depth_absrel"] - b["saved_context_A_depth_absrel"]
        )
        results[sid] = result
    # Descriptive attribution of this locked cohort, explicitly not a fitted classifier.
    interpretations = {
        "ai_001_001": {
            "supported_contributors": ["SUPPORT FAILURE"],
            "causal_gap_diagnosis": "UNKNOWN",
            "reason": (
                "Only 4/128 A candidates have >=2-view support. Neither cohort has a GT "
                "surface point within the fixed neighborhood of an A-supported candidate. "
                "Both cohorts lie inside the volume; heldout views are not farther in "
                "orientation from anchor. Sparse support is shared by both cohorts, so it "
                "does not uniquely explain the gap. Depth/view distributions differ; "
                "memorization remains untested."
            ),
        },
        "ai_002_001": {
            "supported_contributors": ["VOLUME COVERAGE", "SUPPORT FAILURE"],
            "causal_gap_diagnosis": "VOLUME COVERAGE",
            "reason": (
                "Heldout surfaces lie outside the modeled volume more often and have less "
                "A-supported neighborhood coverage, despite heldout cameras being closer to "
                "anchor and less rotated. This supports a coverage-distribution "
                "contribution, not a proven sole cause or simple farther-view extrapolation."
            ),
        },
        "ai_003_001": {
            "supported_contributors": ["DATA QUALITY", "SUPPORT FAILURE"],
            "causal_gap_diagnosis": "UNKNOWN",
            "reason": (
                "All source RGB frames are black and A has zero >=2-view candidate support "
                "in both cohorts. A equals anchor under this architecture. Both cohorts' GT "
                "surfaces lie inside the volume; invalid GT and outside-volume geometry do "
                "not explain the gap. Query orientations, context overlap and per-frame "
                "depth distributions differ. No controlled experiment separates learned "
                "prior/view dependence from training memorization; black source alone does "
                "not identify why only heldout depth collapses."
            ),
        },
    }
    for sid, result in results.items():
        result["interpretation"] = interpretations.get(sid, {"causal_gap_diagnosis": "UNKNOWN"})
    macro = {
        label: float(
            np.mean(
                [
                    r["cohorts"][label]["frame_equal_means"]["saved_context_A_depth_absrel"]
                    for r in results.values()
                ]
            )
        )
        for label in COHORTS
    }
    total_gap = macro["heldout_query"] - macro["training_query"]
    for result in results.values():
        result["fraction_of_macro_gap"] = (
            (result["heldout_minus_training_absrel"] / len(results) / total_gap)
            if total_gap
            else None
        )
    report = {
        "scope": "POSTHOC_STATIC_ONLY_DATA_GEOMETRY_ATTRIBUTION_SAME_TRAIN_SCENES",
        "model_inference_calls": 0,
        "DYNAMIC_TTT_RUN": False,
        "macro_context_A_absrel": macro,
        "macro_heldout_minus_training_absrel": total_gap,
        "scenes": results,
        "neighborhood_rule": volume["neighborhood_rule"],
        "neighborhood_radius_m": volume["radius_m"],
        "input_sha256": {
            str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs
        },
        "limits": [
            "No causal overfitting conclusion from train/heldout score gap alone.",
            (
                "Both query cohorts belong to carrier training scenes; frame-heldout is not "
                "unseen-scene validation."
            ),
            (
                "Candidate-neighborhood support is a fixed geometric proxy, not a test of "
                "all learned receptive fields."
            ),
            "Context overlap uses approximate nearest-depth consistency, not exact visibility.",
            (
                "Depth medians/p95 shown in means are means of frame quantiles, not "
                "pooled-pixel quantiles."
            ),
            (
                "Saved static scores are previous locked B-final outputs; no rerender or "
                "checkpoint selection here."
            ),
        ],
    }
    output = experiment / "train_heldout_gap_diagnosis.json"
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    lines = [
        "# Train-query versus frame-heldout gap",
        "",
        (
            "Static-only attribution using existing audited data and saved predictions; "
            "no model rerun."
        ),
        "",
        (
            f"Context A scene-macro AbsRel: training queries {macro['training_query']:.6f}, "
            f"heldout queries {macro['heldout_query']:.6f}; gap {total_gap:.6f}. "
            "Both use the same three TRAIN scenes."
        ),
        "",
    ]
    keys = [
        "saved_context_A_depth_absrel",
        "anchor_camera_baseline_m",
        "anchor_view_angle_degrees",
        "context_a_minimum_camera_baseline_m",
        "context_a_minimum_view_angle_degrees",
        "depth_mean_m",
        "mean_frame_depth_median_m",
        "mean_frame_depth_p95_m",
        "inside_volume_fraction",
        "valid_GT_fraction",
        "A_supported_surface_fraction",
        "B_supported_surface_fraction",
        "approx_depth_consistent_common_fraction",
        "source_RGB_zero_pixel_fraction",
    ]
    for sid, result in results.items():
        lines += [f"## {sid}", "", "| Quantity | Heldout 8,9 | Training 12–15 |", "|---|---:|---:|"]
        for key in keys:
            values = [result["cohorts"][label]["frame_equal_means"][key] for label in COHORTS]
            lines.append(f"| {key} | {values[0]:.6f} | {values[1]:.6f} |")
        lines += [
            "",
            (
                f"A/B >=2-view supported candidates: {support[sid]['context_a']['at_least_two']}/"
                f"{support[sid]['context_b']['at_least_two']} of 128. "
                f"Gap contribution: {result['fraction_of_macro_gap']:.2%}."
            ),
            "",
            (
                "Observed contributors: "
                f"{', '.join(result['interpretation']['supported_contributors'])}. "
                f"Gap attribution: {result['interpretation']['causal_gap_diagnosis']}."
            ),
            "",
            result["interpretation"]["reason"],
            "",
        ]
    lines += ["## Limits", ""] + [f"- {s}" for s in report["limits"]]
    output.with_suffix(".md").write_text("\n".join(lines) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--previous-static-raw", required=True)
    args = parser.parse_args()
    result = analyze(args.manifest, args.experiment, args.previous_static_raw)
    print(
        json.dumps(
            {
                "macro": result["macro_context_A_absrel"],
                "gap": result["macro_heldout_minus_training_absrel"],
            },
            indent=2,
        )
    )
