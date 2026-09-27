"""Synthetic engineering run only; never establishes any scientific hypothesis."""

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import torch

from mcss.dynamic.carrier import DynamicSceneCarrier
from mcss.dynamic.config import CarrierConfig, WriteConfig
from mcss.dynamic.feedback import batched_camera
from mcss.dynamic.types import hash_scene_state, hash_value
from mcss.dynamic.write_rule import DirectWriteRule
from mcss.geometry import project_world
from mcss.measurements import FixedMeasurementRenderer
from mcss.mechanism_pilot.contracts import EvaluationLedger
from mcss.mechanism_pilot.runtime import FrameRoles, MechanismRuntime
from mcss.mechanism_pilot.statistics import analyze_history, measurement_metrics, scene_macro
from mcss.mechanism_pilot.synthetic import arrived_observation, fixture, fixture_camera

ROOT = Path(__file__).resolve().parents[1]
NAME = "EXP-3D-20260927-state-history-pilot-v1"


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


@torch.no_grad()
def run(output):
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Refusing to overwrite an existing smoke run")
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.manual_seed(20260927)
    started = time.perf_counter()
    config = CarrierConfig()
    carrier = DynamicSceneCarrier(config)
    writer = DirectWriteRule(config, WriteConfig())
    renderer = FixedMeasurementRenderer()
    roles = FrameRoles((0, 1, 2), (0, 3, 4), (5, 6), (7,), (8, 9))
    runtime = MechanismRuntime(carrier, writer, renderer, roles)
    # Freeze implementation and actual random-weight content before inference.
    write(
        output / "run_lock.json",
        {
            "scope": "SYNTHETIC_ENGINEERING_ONLY_RANDOM_UNTRAINED_WEIGHTS",
            "seed": 20260927,
            "carrier_config": asdict(config),
            "write_config": asdict(WriteConfig()),
            "frame_roles": asdict(roles),
            "image_size": [16, 16],
            "renderer_samples": 64,
            "checkpoint_hash": runtime.checkpoint_hash,
            "config_hash": runtime.config_hash,
            "bootstrap_draws": 10000,
            "tie_tolerance": 1e-8,
            "primary_metric": "depth_absrel",
            "scientific_scenes": 0,
            "synthetic_fixtures": 3,
        },
    )
    ids = [
        f"{s}/{method}"
        for s in range(3)
        for method in ["A", "B", "anchor"]
        + [f"{h}/{a}" for h in ["FC", "CF"] for a in ["OFF", "FUSE", "COMPLETE", "ALL"]]
    ]
    ledger = EvaluationLedger(ids)
    sealed, branches, histories, predictions_pre = {}, {}, {}, {}
    history_raw, numerical = [], []
    construction_started = time.perf_counter()
    for s in range(3):
        observed = {i: arrived_observation(s, i) for i in range(8)}
        for name, frames in [("A", roles.context_a), ("B", roles.context_b), ("anchor", (0,))]:
            branch = runtime.build([observed[i] for i in frames], episode_id=f"s{s}-{name}")
            branches[f"{s}/{name}"] = branch
        matched = runtime.matched_histories(
            [observed[i] for i in roles.context_a],
            [observed[i] for i in roles.stream],
            episode_id=f"s{s}-matched",
        )
        histories[s] = matched
        for h, branch in matched.items():
            pre = renderer(branch.state, batched_camera(observed[7].camera), ("rgb", "depth"))
            predictions_pre[s, h] = pre
        numerical.append(
            {
                "scene_id": f"synthetic-{s}",
                "continuation_id": 7,
                "fast_distance": float(
                    torch.sqrt(
                        sum(
                            (getattr(matched["FC"].fast, n) - getattr(matched["CF"].fast, n))
                            .square()
                            .sum()
                            for n in ["delta_fuse", "delta_complete"]
                        )
                    )
                ),
                "pre_rgb_rms_difference": float(
                    (predictions_pre[s, "FC"]["rgb"] - predictions_pre[s, "CF"]["rgb"])
                    .square()
                    .mean()
                    .sqrt()
                ),
                "pre_depth_rms_difference": float(
                    (predictions_pre[s, "FC"]["depth"] - predictions_pre[s, "CF"]["depth"])
                    .square()
                    .mean()
                    .sqrt()
                ),
            }
        )
        candidates = runtime.continuation_branches(matched, observed[7])
        for h, actions in candidates.items():
            for action, branch in actions.items():
                branches[f"{s}/{h}/{action}"] = branch
                history_raw.append(
                    {
                        "scene_id": f"synthetic-{s}",
                        "history": h,
                        "action": action,
                        "summary": branch.summary,
                        "history_records": branch.history,
                    }
                )
    construction_seconds = time.perf_counter() - construction_started
    # The evaluator receives only sealed states, never a live runtime or writer.
    for key, branch in branches.items():
        sealed[key] = runtime.seal(branch)
        ledger.seal(key, sealed[key])
    write(
        output / "state_manifest.json",
        {
            key: {
                "state_hash": v.state_hash,
                "observed_ids": v.observed_ids,
                "fast_state_hash": v.fast_state_hash,
            }
            for key, v in sealed.items()
        },
    )
    static_rows, dynamic_rows, prediction_rows = [], [], []
    post_effects = []
    evaluate_started = time.perf_counter()
    for s in range(3):
        for q in roles.sealed_query:
            camera = ledger.read_query_camera(f"{s}/{q}", lambda q=q: fixture_camera(q))
            truth = ledger.read_query_ground_truth(f"{s}/{q}", lambda s=s, q=q: fixture(s, q)[1:])
            rgb, depth, points = truth
            common = torch.ones_like(depth, dtype=torch.bool)
            for ctx in [roles.context_a, roles.context_b]:
                visible = torch.zeros_like(common)
                for frame in ctx:
                    _, _, valid = project_world(
                        points.reshape(-1, 3), arrived_observation(s, frame).camera
                    )
                    visible |= valid.reshape_as(common)
                common &= visible
            query_camera = batched_camera(camera)
            camera_hash = hash_value(query_camera)
            for method in ["A", "B", "anchor", "prior", "wrong_scene"]:
                if method == "prior":
                    pred = {
                        "rgb": torch.full((1, 1, 3, *depth.shape), 0.5),
                        "depth": torch.full((1, 1, 1, *depth.shape), 5.0),
                        "visibility": torch.ones(1, 1, 1, *depth.shape),
                    }
                else:
                    key = f"{(s + 1) % 3}/A" if method == "wrong_scene" else f"{s}/{method}"
                    state = sealed[key]
                    assert hash_scene_state(state.scene_state) == state.state_hash
                    pred = renderer(state.scene_state, query_camera, ("rgb", "depth", "visibility"))
                    assert hash_scene_state(state.scene_state) == state.state_hash
                assert camera_hash == hash_value(query_camera)
                metrics = measurement_metrics(
                    pred["rgb"][0, 0],
                    rgb,
                    pred["depth"][0, 0, 0],
                    depth,
                    pred["visibility"][0, 0, 0],
                )
                region_metrics = (
                    measurement_metrics(
                        pred["rgb"][0, 0],
                        rgb,
                        pred["depth"][0, 0, 0],
                        depth,
                        pred["visibility"][0, 0, 0],
                        region_mask=common.numpy(),
                    )
                    if common.any()
                    else None
                )
                static_rows.append(
                    dict(
                        scene_id=f"synthetic-{s}",
                        query_id=q,
                        method=method,
                        **metrics,
                        common_visible=region_metrics,
                        query_camera_hash=camera_hash,
                    )
                )
            post = {}
            for h in ["FC", "CF"]:
                for action in ["OFF", "FUSE", "COMPLETE", "ALL"]:
                    state = sealed[f"{s}/{h}/{action}"]
                    assert hash_scene_state(state.scene_state) == state.state_hash
                    pred = renderer(state.scene_state, query_camera, ("rgb", "depth", "visibility"))
                    post[h, action] = pred
                    metrics = measurement_metrics(
                        pred["rgb"][0, 0],
                        rgb,
                        pred["depth"][0, 0, 0],
                        depth,
                        pred["visibility"][0, 0, 0],
                    )
                    dynamic_rows.append(
                        dict(
                            scene_id=f"synthetic-{s}",
                            continuation_id=7,
                            query_id=q,
                            history=h,
                            action=action,
                            **metrics,
                        )
                    )
                    assert hash_scene_state(state.scene_state) == state.state_hash
                    prediction_rows.append(
                        {
                            "scene_id": s,
                            "query": q,
                            "history": h,
                            "action": action,
                            "rgb_hash": hash_value(pred["rgb"]),
                            "depth_hash": hash_value(pred["depth"]),
                        }
                    )
            for action in ["OFF", "FUSE", "COMPLETE", "ALL"]:
                effect = {"scene_id": f"synthetic-{s}", "query_id": q, "action": action}
                for channel in ["rgb", "depth"]:
                    diff = post["FC", action][channel] - post["CF", action][channel]
                    effect[channel + "_rms"] = float(diff.square().mean().sqrt())
                    effect[channel + "_maxabs"] = float(diff.abs().max())
                post_effects.append(effect)
    write(output / "post_prediction_history_effect.json", post_effects)
    analysis = analyze_history(dynamic_rows, draws=10000, seed=20260927, tie_tolerance=1e-8)
    write(output / "controlled_history_results.json", dynamic_rows)
    write(output / "static_state_results.json", static_rows)
    write(output / "history_traces.json", history_raw)
    write(output / "prediction_hashes.json", prediction_rows)
    write(output / "numerical_history_effect.json", numerical)
    write(output / "history_analysis.json", analysis)
    metrics = [
        "rgb_psnr",
        "rgb_ssim",
        "depth_absrel",
        "depth_rmse",
        "depth_delta1",
        "opacity",
        "coverage",
        "valid_depth_fraction",
    ]
    static_analysis = {
        method: {
            m: scene_macro([r for r in static_rows if r["method"] == method], m) for m in metrics
        }
        for method in ["A", "B", "anchor", "prior", "wrong_scene"]
    }
    static_analysis["R-Residual"] = {"status": "NOT_RUN_UNTRAINED_HEAD_NOT_FAIR_COMPARISON"}
    static_analysis["depth_absrel_deltas"] = {
        label: static_analysis[left]["depth_absrel"]["mean"]
        - static_analysis[right]["depth_absrel"]["mean"]
        for label, left, right in [
            ("context_B_minus_A", "B", "A"),
            ("anchor_minus_A", "anchor", "A"),
            ("wrong_scene_minus_A", "wrong_scene", "A"),
        ]
    }
    write(output / "static_analysis.json", static_analysis)
    write(
        output / "access_log.json",
        {
            "scope": "SYNTHETIC_ONLY",
            "events": ledger.events,
            "formal_reads": 0,
            "note": "Synthetic labels only; no depth enters runtime.",
        },
    )
    evaluation_seconds = time.perf_counter() - evaluate_started
    # Pairwise tensor storage independence, including candidate caches/fast/state.
    all_branches = list(branches.values())
    ptrs = [b.fast.delta_fuse.data_ptr() for b in all_branches]
    isolation = len(set(ptrs)) == len(ptrs)
    assert isolation
    write(
        output / "summary.json",
        {
            "scope": "SYNTHETIC_ENGINEERING_ONLY_NOT_SCIENTIFIC_EVIDENCE",
            "synthetic_scenes": 3,
            "scientific_scenes": 0,
            "controlled_pairs": 3,
            "dynamic_query_rows": len(dynamic_rows),
            "static_query_rows": len(static_rows),
            "candidate_isolation_verified": isolation,
            "all_states_sealed_before_query_reads": True,
            "carrier_parameters": sum(p.numel() for p in carrier.parameters()),
            "writer_parameters": sum(p.numel() for p in writer.parameters()),
            "fixed_renderer_parameters": sum(p.numel() for p in renderer.parameters()),
            "construction_seconds": construction_seconds,
            "evaluation_seconds": evaluation_seconds,
            "total_seconds": time.perf_counter() - started,
            "device": "cpu",
            "threads": 1,
            "candidate_branch_steps": 24,
            "matched_history_steps": 12,
            "bidirectional_matching_calls": 0,
            "policy_training_runs": 0,
        },
    )
    print(json.dumps(json.loads((output / "summary.json").read_text()), indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / NAME / "smoke")
    run(parser.parse_args().output)
