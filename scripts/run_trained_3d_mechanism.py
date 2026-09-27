"""Frozen-checkpoint static/controlled-history diagnostic on TRAIN scenes only."""

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from mcss.dynamic.checkpoint import load_dynamic_checkpoint
from mcss.dynamic.feedback import anchored_observation, batched_camera
from mcss.dynamic.types import OnlineObservation, hash_scene_state, hash_value
from mcss.geometry import transform_cameras
from mcss.measurements import FixedMeasurementRenderer
from mcss.mechanism_pilot.contracts import EvaluationLedger
from mcss.mechanism_pilot.runtime import FrameRoles, MechanismRuntime
from mcss.mechanism_pilot.small_training import (
    AccessLog,
    TrainScene,
    sha,
    validate_manifest,
    write_json,
)
from mcss.mechanism_pilot.statistics import (
    analyze_history,
    measurement_metrics,
    paired_scene_bootstrap,
)
from mcss.mechanism_pilot.visibility import common_visibility_masks

COHORTS = {"frame_holdout": (8, 9), "training_query": (12, 13, 14, 15)}
METRICS = (
    "rgb_mse",
    "rgb_psnr",
    "rgb_ssim",
    "depth_absrel",
    "depth_rmse",
    "depth_delta1",
    "opacity",
    "coverage",
    "valid_depth_fraction",
)
METHODS = ("A", "B", "anchor", "prior", "wrong_scene")
ACTIONS = ("OFF", "FUSE", "COMPLETE", "ALL")


class PilotData:
    """Arrived provider rejects query IDs; evaluator loaders require all-state seal."""

    def __init__(self, manifest_path, device, access):
        self.path = Path(manifest_path).resolve()
        self.manifest = json.loads(self.path.read_text())
        self.scenes = {
            r["scene_id"]: TrainScene(r, self.manifest["image_size"], self.path.parent, device, [])
            for r in self.manifest["scenes"]
        }
        self.access = access

    def log(self, scene, frame, kind, stage):
        self.access.append(
            {"scene_id": scene, "frame_id": frame, "kind": kind, "stage": stage, "split": "train"}
        )

    def observation(self, scene_id, frame_id):
        if frame_id not in range(8):
            raise PermissionError("Only locked arrived frames 0..7 may construct states")
        scene = self.scenes[scene_id]
        frame = scene.frames[frame_id]
        self.log(scene_id, frame_id, "rgb_camera", "state_construction")
        return OnlineObservation(scene_id, frame_id, scene.rgb(frame), scene.camera(frame))

    def query_camera(self, ledger, scene_id, frame_id):
        if frame_id not in sum(COHORTS.values(), ()):
            raise ValueError("Undeclared query")

        def load():
            self.log(scene_id, frame_id, "camera", "sealed_evaluator")
            return self.scenes[scene_id].camera(self.scenes[scene_id].frames[frame_id])

        return ledger.read_query_camera(f"{scene_id}/{frame_id}", load)

    def query_truth(self, ledger, scene_id, frame_id):
        if frame_id not in sum(COHORTS.values(), ()):
            raise ValueError("Undeclared query")

        def load():
            self.log(scene_id, frame_id, "rgb_depth", "sealed_evaluator")
            scene = self.scenes[scene_id]
            frame = scene.frames[frame_id]
            return scene.rgb(frame), torch.from_numpy(
                np.load(scene.path(frame, "depth"))
            ).float().to(scene.device).squeeze()

        return ledger.read_query_ground_truth(f"{scene_id}/{frame_id}", load)

    def context_depth(self, ledger, scene_id, frame_id):
        if frame_id not in range(5):
            raise PermissionError("Only declared contexts for evaluator visibility")

        def load():
            self.log(scene_id, frame_id, "depth", "sealed_evaluator_visibility_only")
            scene = self.scenes[scene_id]
            return (
                torch.from_numpy(np.load(scene.path(scene.frames[frame_id], "depth")))
                .float()
                .to(scene.device)
                .squeeze()
            )

        return ledger.read_query_ground_truth(f"context-visibility/{scene_id}/{frame_id}", load)


def summarize_static(rows):
    result = {}
    for method in METHODS:
        selected = [r for r in rows if r["method"] == method]
        scene_ids = sorted({r["scene_id"] for r in selected})
        result[method] = {}
        for region in ("full_image", "common_frustum", "depth_consistent_common"):
            regional = (
                selected
                if region == "full_image"
                else [
                    {"scene_id": r["scene_id"], **r[region]}
                    for r in selected
                    if r[region] is not None
                ]
            )
            statistics = {}
            for metric in METRICS:
                per_scene = {}
                for scene_id in scene_ids:
                    values = [r[metric] for r in regional if r["scene_id"] == scene_id]
                    per_scene[scene_id] = (
                        None
                        if not values or any(v is None for v in values)
                        else float(np.mean(values))
                    )
                statistics[metric] = {
                    "per_scene": per_scene,
                    "mean": None
                    if any(v is None for v in per_scene.values())
                    else float(np.mean(list(per_scene.values()))),
                }
            result[method][region] = {
                "metrics": statistics,
                "eligible_queries": len(regional),
                "total_queries": len(selected),
            }
    deltas = {}
    for label, left, right in [
        ("B_minus_A", "B", "A"),
        ("anchor_minus_A", "anchor", "A"),
        ("prior_minus_A", "prior", "A"),
        ("wrong_scene_minus_A", "wrong_scene", "A"),
    ]:
        a = result[left]["full_image"]["metrics"]["depth_absrel"]["per_scene"]
        b = result[right]["full_image"]["metrics"]["depth_absrel"]["per_scene"]
        deltas[label] = paired_scene_bootstrap({s: a[s] - b[s] for s in a})
    return {
        "methods": result,
        "depth_absrel_deltas": deltas,
        "residual_control": "NOT_RUN_TRAINED_RESIDUAL_HEAD_MISSING",
        "common_visibility_note": (
            "Nearest depth tolerance approximation; empty masks "
            "retained as null, eligible counts reported"
        ),
    }


def tensor_difference(left, right):
    d = left - right
    return {"rms": float(d.square().mean().sqrt()), "maxabs": float(d.abs().max())}


@torch.no_grad()
def evaluate(sealed, ledger, data, renderer, observed, output):
    """Receives sealed states, no carrier/writer/runtime; GT cannot drive any write."""
    ledger.assert_ready()
    static_rows, history_rows, prediction_hashes, post_effects = [], [], [], []
    scene_ids = list(data.scenes)
    for index, scene_id in enumerate(scene_ids):
        context_depths = {i: data.context_depth(ledger, scene_id, i) for i in range(5)}
        contexts = [
            [(observed[scene_id][i].camera, context_depths[i]) for i in ids]
            for ids in [(0, 1, 2), (0, 3, 4)]
        ]
        anchor = sealed[f"{scene_id}/A"].anchor_c2w
        for cohort, query_ids in COHORTS.items():
            for q in query_ids:
                world_camera = data.query_camera(ledger, scene_id, q)
                rgb, depth = data.query_truth(ledger, scene_id, q)
                masks = common_visibility_masks(world_camera, depth, *contexts)
                camera = batched_camera(transform_cameras(world_camera, torch.linalg.inv(anchor)))
                camera_hash = hash_value(camera)
                common = {
                    "scene_id": scene_id,
                    "query_id": q,
                    "cohort": cohort,
                    "split": "train",
                    "query_camera_hash": camera_hash,
                }
                for method in METHODS:
                    if method == "prior":
                        pred = {
                            "rgb": torch.full_like(rgb[None, None], 0.5),
                            "depth": torch.full_like(depth[None, None, None], 5.0),
                            "visibility": torch.ones_like(depth[None, None, None]),
                        }
                    else:
                        donor = (
                            scene_ids[(index + 1) % len(scene_ids)]
                            if method == "wrong_scene"
                            else scene_id
                        )
                        state = sealed[f"{donor}/{'A' if method == 'wrong_scene' else method}"]
                        assert hash_scene_state(state.scene_state) == state.state_hash
                        pred = renderer(state.scene_state, camera, ("rgb", "depth", "visibility"))
                        assert hash_scene_state(state.scene_state) == state.state_hash
                    args = (
                        pred["rgb"][0, 0],
                        rgb,
                        pred["depth"][0, 0, 0],
                        depth,
                        pred["visibility"][0, 0, 0],
                    )
                    regions = {
                        name: measurement_metrics(*args, region_mask=mask.cpu().numpy())
                        if mask.any()
                        else None
                        for name, mask in masks.items()
                    }
                    static_rows.append(
                        {
                            **common,
                            "method": method,
                            **measurement_metrics(*args),
                            **regions,
                            "region_counts": {
                                name: int(mask.sum()) for name, mask in masks.items()
                            },
                        }
                    )
                    prediction_hashes.append(
                        {
                            **common,
                            "method": method,
                            "rgb": hash_value(pred["rgb"]),
                            "depth": hash_value(pred["depth"]),
                        }
                    )
                    assert camera_hash == hash_value(camera)
                predictions = {}
                for history in ("FC", "CF"):
                    for action in ACTIONS:
                        state = sealed[f"{scene_id}/{history}/{action}"]
                        assert hash_scene_state(state.scene_state) == state.state_hash
                        pred = renderer(state.scene_state, camera, ("rgb", "depth", "visibility"))
                        predictions[history, action] = pred
                        metrics = measurement_metrics(
                            pred["rgb"][0, 0],
                            rgb,
                            pred["depth"][0, 0, 0],
                            depth,
                            pred["visibility"][0, 0, 0],
                        )
                        history_rows.append(
                            {
                                **common,
                                "continuation_id": 7,
                                "history": history,
                                "action": action,
                                **metrics,
                            }
                        )
                        prediction_hashes.append(
                            {
                                **common,
                                "history": history,
                                "action": action,
                                "rgb": hash_value(pred["rgb"]),
                                "depth": hash_value(pred["depth"]),
                            }
                        )
                        assert hash_scene_state(state.scene_state) == state.state_hash
                        assert camera_hash == hash_value(camera)
                for action in ACTIONS:
                    post_effects.append(
                        {
                            **common,
                            "action": action,
                            **{
                                channel: tensor_difference(
                                    predictions["FC", action][channel],
                                    predictions["CF", action][channel],
                                )
                                for channel in ("rgb", "depth")
                            },
                        }
                    )
    ledger.assert_ready()
    write_json(output / "static_state_results.json", static_rows)
    write_json(output / "controlled_history_results.json", history_rows)
    write_json(output / "prediction_hashes.json", prediction_hashes)
    write_json(output / "post_prediction_history_effect.json", post_effects)
    return static_rows, history_rows


@torch.no_grad()
def run(manifest, checkpoint, output, *, device="cuda", engineering_fixture=False):
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Refusing to overwrite mechanism run")
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    torch.set_num_threads(1)
    access = AccessLog(output / "label_access.jsonl")
    data = PilotData(manifest, device, access)
    validate_manifest(data.manifest, engineering_fixture=engineering_fixture)
    input_hashes = {
        str(data.path): sha(data.path),
        str(Path(checkpoint).resolve()): sha(Path(checkpoint)),
    }
    if not engineering_fixture:
        assert sha(data.path.parent / "data_lock.json") == data.manifest["data_lock_sha256"]
        for path, expected in json.loads((data.path.parent / "hashes.json").read_text()).items():
            full = data.path.parent / path
            assert sha(full) == expected
            input_hashes[str(full)] = expected
    source_root = Path(__file__).resolve().parents[1]
    source_paths = list((source_root / "src/mcss").rglob("*.py")) + [Path(__file__).resolve()]
    source_hashes = {str(p): sha(p) for p in source_paths}
    carrier, writer, metadata = load_dynamic_checkpoint(checkpoint, device=device)
    renderer = FixedMeasurementRenderer(n_samples=64, ray_chunk_size=2048).to(device)
    roles = FrameRoles((0, 1, 2), (0, 3, 4), (5, 6), (7,), sum(COHORTS.values(), ()))
    runtime = MechanismRuntime(carrier, writer, renderer, roles)
    scene_ids = list(data.scenes)
    write_json(
        output / "run_lock.json",
        {
            "scope": "POSTHOC_TRAIN_SCENE_DIAGNOSTIC",
            "cohorts": COHORTS,
            "checkpoint": metadata,
            "carrier": asdict(carrier.config),
            "writer": asdict(writer.write_config),
            "roles": asdict(roles),
            "scene_ids": scene_ids,
            "input_sha256": input_hashes,
            "source_sha256": source_hashes,
            "seed": 20260927,
            "bootstrap_draws": 10000,
            "tie_tolerance": 1e-8,
            "renderer_samples": 64,
            "primary_metric": "depth_absrel",
            "n_dev_scenes": 0,
            "n_test_scenes": 0,
        },
    )
    ids = [
        f"{s}/{method}"
        for s in scene_ids
        for method in ["A", "B", "anchor"] + [f"{h}/{a}" for h in ("FC", "CF") for a in ACTIONS]
    ]
    ledger = EvaluationLedger(ids)
    observed, branches, numerical, traces, parents = {}, {}, [], [], []
    for scene_id in scene_ids:
        obs = {i: data.observation(scene_id, i) for i in range(8)}
        observed[scene_id] = obs
        for method, ids in [("A", roles.context_a), ("B", roles.context_b), ("anchor", (0,))]:
            branches[f"{scene_id}/{method}"] = runtime.build(
                [obs[i] for i in ids], episode_id=f"{scene_id}-{method}"
            )
        histories = runtime.matched_histories(
            [obs[i] for i in roles.context_a],
            [obs[i] for i in roles.stream],
            episode_id=f"{scene_id}-history",
        )
        parents.extend(histories.values())
        before = {
            h: hash_value(
                {"state": b.state, "fast": b.fast, "cache": b.cache, "history": b.history}
            )
            for h, b in histories.items()
        }
        arrived = anchored_observation(obs[7], histories["FC"].anchor_c2w)
        pre = {
            h: renderer(b.state, batched_camera(arrived.camera), ("rgb", "depth"))
            for h, b in histories.items()
        }
        numerical.append(
            {
                "scene_id": scene_id,
                "continuation_id": 7,
                "fast_l2_distance": float(
                    torch.sqrt(
                        sum(
                            (getattr(histories["FC"].fast, k) - getattr(histories["CF"].fast, k))
                            .square()
                            .sum()
                            for k in ("delta_fuse", "delta_complete")
                        )
                    )
                ),
                "pre_prediction": {
                    c: tensor_difference(pre["FC"][c], pre["CF"][c]) for c in ("rgb", "depth")
                },
            }
        )
        children = runtime.continuation_branches(histories, obs[7])
        for h, actions in children.items():
            assert before[h] == hash_value(
                {
                    "state": histories[h].state,
                    "fast": histories[h].fast,
                    "cache": histories[h].cache,
                    "history": histories[h].history,
                }
            )
            for action, branch in actions.items():
                branches[f"{scene_id}/{h}/{action}"] = branch
                traces.append(
                    {
                        "scene_id": scene_id,
                        "history": h,
                        "action": action,
                        "summary": branch.summary,
                        "records": branch.history,
                    }
                )
                assert branch.summary["observed_ids"] == [0, 1, 2, 5, 6, 7]
                if action == "OFF":
                    assert all(
                        torch.equal(getattr(branch.fast, k), getattr(histories[h].fast, k))
                        for k in ("delta_fuse", "delta_complete")
                    )
    ptrs = [b.fast.delta_fuse.data_ptr() for b in branches.values()]
    assert len(ptrs) == len(set(ptrs))
    sealed = {
        key: runtime.seal(branch, split_id="train", query_vault_id="train-diagnostic")
        for key, branch in branches.items()
    }
    for key, state in sealed.items():
        ledger.seal(key, state)
    write_json(
        output / "state_manifest.json",
        {
            k: {
                "state_hash": s.state_hash,
                "observed_ids": s.observed_ids,
                "fast_hash": s.fast_state_hash,
                "anchor_hash": hash_value(s.anchor_c2w),
            }
            for k, s in sealed.items()
        },
    )
    torch.save(
        {
            k: {
                "state": {
                    n: getattr(s.scene_state, n)
                    for n in ("density_logits", "color", "log_variance", "features", "bounds")
                },
                "anchor_c2w": s.anchor_c2w,
            }
            for k, s in sealed.items()
        },
        output / "sealed_states.pt",
    )
    write_json(output / "history_traces.json", traces)
    write_json(output / "numerical_history_effect.json", numerical)
    construction_seconds = time.perf_counter() - started
    static, dynamic = evaluate(sealed, ledger, data, renderer, observed, output)
    for cohort in COHORTS:
        write_json(
            output / f"static_analysis_{cohort}.json",
            summarize_static([r for r in static if r["cohort"] == cohort]),
        )
        write_json(
            output / f"history_analysis_{cohort}.json",
            analyze_history([r for r in dynamic if r["cohort"] == cohort]),
        )
    assert runtime.checkpoint_hash == hash_value(
        {"carrier": carrier.state_dict(), "writer": writer.state_dict()}
    )
    assert all(sha(Path(p)) == h for p, h in input_hashes.items())
    assert all(sha(Path(p)) == h for p, h in source_hashes.items())
    write_json(output / "sealed_query_access_log.json", ledger.events)
    write_json(
        output / "summary.json",
        {
            "scope": "POSTHOC_TRAIN_SCENE_DIAGNOSTIC",
            "n_train_scenes": 3,
            "n_dev_scenes": 0,
            "n_test_scenes": 0,
            "controlled_pairs": 3,
            "static_rows": len(static),
            "dynamic_rows": len(dynamic),
            "sealed_states": len(sealed),
            "candidate_isolation": True,
            "all_states_sealed_before_query_access": True,
            "checkpoint_unchanged": True,
            "source_data_unchanged": True,
            "construction_seconds": construction_seconds,
            "elapsed_seconds": time.perf_counter() - started,
            "device": device,
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated()
            if device == "cuda"
            else None,
            "carrier_parameters": sum(p.numel() for p in carrier.parameters()),
            "writer_parameters": sum(p.numel() for p in writer.parameters()),
            "fixed_renderer_parameters": 0,
            "history_write_steps": 12,
            "candidate_steps": 24,
            "candidate_writes": 18,
            "candidate_feedback_renders": 24,
            "history_feedback_renders": 12,
            "pre_prediction_comparison_renders": 6,
            "total_renderer_calls": 258,
            "materialize_calls": 48,
            "trace_proposal_calls": 30,
            "image_encode_calls": 66,
            "query_renders": 216,
            "prior_predictions": 18,
            "matching_calls": 0,
            "policy_training_runs": 0,
            "residual": "NOT_RUN_TRAINED_HEAD_MISSING",
            "natural_history": "SKIPPED_STATIC_QUALIFICATION",
            "policy": "SKIPPED_STATIC_QUALIFICATION",
        },
    )
    print(json.dumps(json.loads((output / "summary.json").read_text()), indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    run(args.manifest, args.checkpoint, args.output, device=args.device)
