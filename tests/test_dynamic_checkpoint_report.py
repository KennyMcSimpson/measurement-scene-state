"""Checkpoint evaluator regression tests for dynamic scene reporting."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

import mcss.evaluation.checkpoint_report as checkpoint_report
from mcss.dynamic.carrier import DynamicSceneCarrier
from mcss.dynamic.checkpoint import save_dynamic_checkpoint
from mcss.dynamic.config import CarrierConfig, WriteConfig
from mcss.dynamic.policy import FEATURE_SCHEMA_VERSION, POLICY_FEATURE_DIM, LearnedActionPolicy
from mcss.dynamic.policy_checkpoint import save_policy_checkpoint
from mcss.dynamic.types import hash_value
from mcss.dynamic.write_rule import DirectWriteRule
from mcss.training.action_teacher import make_binding
from mcss.training.policy_fit import ACTION_TEACHER_SCHEMA, fit_action_policy


def test_fit_checkpoint_evaluator_cpu_round_trip(tmp_path: Path) -> None:
    checkpoint, expected_state_hash = _write_checkpoint(tmp_path)
    entry, prepared_root = _write_episode(tmp_path, split_id="dev", scene_id="scene_a")
    binding = make_binding(
        model_content_hash=expected_state_hash,
        renderer_samples=2,
        ray_chunk_size=2048,
        max_units=1e12,
        horizon=3,
    )
    binding["teacher_dataset_hash"] = "teacher-dataset"
    binding["teacher_source_hashes"] = {"teacher.jsonl": "teacher-source"}
    rows = []
    for index, value in enumerate((-0.2, 0.0, 0.2)):
        features = [0.0] * POLICY_FEATURE_DIM
        features[0] = value
        rows.append(
            {
                "schema_version": ACTION_TEACHER_SCHEMA,
                "row_id": f"row-{index}",
                "episode_id": "train-episode",
                "scene_id": "train-scene",
                "split_id": "train",
                "prefix_step": index,
                "prefix_frame_id": index,
                "control_input": {},
                "features": features,
                "actions": ["OFF", "FUSE", "COMPLETE", "ALL"],
                "feasible_mask": [True, True, True, True],
                "losses": [1.0, 0.8, 0.6, 0.4],
                "advantages": [0.0, 0.2, 0.4, 0.6],
                "costs": [0.0, 1.0, 2.0, 3.0],
                "horizon_used": 3,
                "provenance": dict(binding),
            }
        )

    result = fit_action_policy(
        rows,
        hidden_dim=32,
        epochs=2,
        holdout_fraction=0.0,
        seed=5,
    )
    policy_binding = dict(result.binding)
    policy_binding["teacher_dataset_hash"] = binding["teacher_dataset_hash"]
    policy_binding["teacher_source_hashes"] = binding["teacher_source_hashes"]
    policy_checkpoint = tmp_path / "fitted_policy.pt"
    save_policy_checkpoint(
        policy_checkpoint,
        result.policy,
        binding=policy_binding,
        provenance=result.checkpoint_provenance(),
    )

    report = checkpoint_report.evaluate_dynamic_checkpoint(
        [entry],
        checkpoint,
        tmp_path / "fit_evaluation",
        device="cpu",
        policies=("OFF", "LEARNED"),
        render_samples=2,
        ray_chunk_size=2048,
        policy_checkpoint=policy_checkpoint,
        prepared_root=prepared_root,
    )

    assert report["status"] == "complete"
    learned = next(record for record in report["records"] if record["policy"] == "LEARNED")
    assert learned["trained_policy"] is True
    assert learned["policy_binding"]["model_content_hash"] == expected_state_hash


def test_evaluator_loads_checkpoint_and_writes_scene_policy_records(
    tmp_path: Path, monkeypatch
) -> None:
    checkpoint, expected_state_hash = _write_checkpoint(tmp_path)
    entry, prepared_root = _write_episode(tmp_path, split_id="dev", scene_id="scene_a")
    calls: list[Path] = []
    original_load = checkpoint_report.load_dynamic_checkpoint

    def tracked_load(path: str | Path, device: str | torch.device = "cpu"):
        calls.append(Path(path).resolve())
        return original_load(path, device)

    monkeypatch.setattr(checkpoint_report, "load_dynamic_checkpoint", tracked_load)
    output = tmp_path / "evaluation"

    report = checkpoint_report.evaluate_dynamic_checkpoint(
        [entry],
        checkpoint,
        output,
        device="cpu",
        policies=("OFF", "ALL"),
        render_samples=2,
        prepared_root=prepared_root,
    )

    assert calls == [checkpoint.resolve()]
    assert report["status"] == "complete"
    assert report["checkpoint"]["sha256"] == _sha256(checkpoint)
    assert len(report["records"]) == 2
    assert {record["policy"] for record in report["records"]} == {"OFF", "ALL"}
    assert {record["checkpoint_hash"] for record in report["records"]} == {expected_state_hash}
    assert {record["checkpoint_file_sha256"] for record in report["records"]} == {
        _sha256(checkpoint)
    }
    assert all(record["sealed_state_hash"] for record in report["records"])
    assert all("averages" in record["metrics"] for record in report["records"])
    assert all(
        record["actions"] == [item["action"] for item in record["history"]]
        for record in report["records"]
    )
    assert all(
        record["budget"]["units_kind"].endswith("not_measured_flops")
        for record in report["records"]
    )

    persisted = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert persisted["records"] == report["records"]
    with (output / "summary.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert {(row["split_id"], row["policy"]) for row in rows} == {
        ("dev", "OFF"),
        ("dev", "ALL"),
    }


def test_max_episodes_does_not_read_unselected_episode_data(tmp_path: Path) -> None:
    checkpoint, _ = _write_checkpoint(tmp_path)
    first, prepared_root = _write_episode(tmp_path, split_id="dev", scene_id="scene_a")
    second, _ = _write_episode(tmp_path, split_id="dev", scene_id="scene_b")
    second_online = json.loads(Path(second["online_manifest"]).read_text(encoding="utf-8"))
    Path(second_online["warmup"][0]["rgb"]).write_bytes(b"unselected image must not be decoded")

    report = checkpoint_report.evaluate_dynamic_checkpoint(
        [first, second],
        checkpoint,
        tmp_path / "subset",
        device="cpu",
        policies=("OFF",),
        render_samples=2,
        max_episodes=1,
        prepared_root=prepared_root,
    )

    assert report["actual_subset"] == {
        "requested_max_episodes": 1,
        "selected_episodes": 1,
        "selected_episode_ids": ["dev--scene_a--cam_00"],
    }
    assert [record["episode_id"] for record in report["records"]] == [
        "dev--scene_a--cam_00"
    ]


def test_evaluator_rejects_train_entry_before_decoding_data(tmp_path: Path) -> None:
    checkpoint, _ = _write_checkpoint(tmp_path)
    entry, prepared_root = _write_episode(tmp_path, split_id="train", scene_id="scene_a")
    online = json.loads(Path(entry["online_manifest"]).read_text(encoding="utf-8"))
    Path(online["warmup"][0]["rgb"]).write_bytes(b"must not be decoded")

    with pytest.raises(ValueError, match="only dev entries"):
        checkpoint_report.evaluate_dynamic_checkpoint(
            [entry],
            checkpoint,
            tmp_path / "train_rejected",
            device="cpu",
            policies=("OFF",),
            render_samples=2,
            prepared_root=prepared_root,
        )


def test_evaluator_requires_the_exact_dev_episode_protocol(tmp_path: Path) -> None:
    checkpoint, _ = _write_checkpoint(tmp_path)
    entry, prepared_root = _write_episode(tmp_path, split_id="dev", scene_id="scene_a")
    online_path = Path(entry["online_manifest"])
    online = json.loads(online_path.read_text(encoding="utf-8"))
    online["stream"] = online["stream"][:-1]
    online["declared_length"] = 11
    online_path.write_text(json.dumps(online), encoding="utf-8")
    entry["stream_steps"] = 7

    with pytest.raises(ValueError, match="exactly warmup=4, stream=8, and query=4"):
        checkpoint_report.evaluate_dynamic_checkpoint(
            [entry],
            checkpoint,
            tmp_path / "bad_protocol",
            device="cpu",
            policies=("OFF",),
            render_samples=2,
            prepared_root=prepared_root,
        )


def test_fixed_policy_fallback_to_off_fails_comparison(tmp_path: Path) -> None:
    checkpoint, _ = _write_checkpoint(tmp_path)
    entry, prepared_root = _write_episode(tmp_path, split_id="dev", scene_id="scene_a")

    with pytest.raises(RuntimeError, match="fixed policy FUSE fell back"):
        checkpoint_report.evaluate_dynamic_checkpoint(
            [entry],
            checkpoint,
            tmp_path / "fallback",
            device="cpu",
            policies=("OFF", "FUSE"),
            render_samples=2,
            max_units=9704.0,
            prepared_root=prepared_root,
        )


def test_learned_policy_requires_checkpoint_and_logs_bound_trajectory(tmp_path: Path) -> None:
    checkpoint, expected_state_hash = _write_checkpoint(tmp_path)
    entry, prepared_root = _write_episode(tmp_path, split_id="dev", scene_id="scene_a")
    policy_checkpoint = _write_policy_checkpoint(
        tmp_path,
        model_content_hash=expected_state_hash,
        render_samples=2,
        ray_chunk_size=2048,
        max_units=1e12,
    )

    with pytest.raises(ValueError, match="requires an explicit policy_checkpoint"):
        checkpoint_report.evaluate_dynamic_checkpoint(
            [entry],
            checkpoint,
            tmp_path / "missing_policy",
            device="cpu",
            policies=("OFF", "LEARNED"),
            render_samples=2,
            prepared_root=prepared_root,
        )

    report = checkpoint_report.evaluate_dynamic_checkpoint(
        [entry],
        checkpoint,
        tmp_path / "learned",
        device="cpu",
        policies=("OFF", "LEARNED"),
        render_samples=2,
        ray_chunk_size=2048,
        policy_checkpoint=policy_checkpoint,
        prepared_root=prepared_root,
    )

    learned = next(record for record in report["records"] if record["policy"] == "LEARNED")
    assert learned["trained_policy"] is True
    assert learned["policy_file_sha256"] == _sha256(policy_checkpoint)
    assert learned["policy_checkpoint_file_sha256"] == _sha256(policy_checkpoint)
    assert learned["policy_content_hash"]
    assert learned["policy_binding"]["model_content_hash"] == expected_state_hash
    assert len(learned["actions"]) == 8
    assert all("control_input" in item for item in learned["history"])


@pytest.mark.parametrize(
    ("binding_field", "replacement", "message"),
    [
        ("model_content_hash", "wrong-carrier", "binding mismatch"),
        (
            "feature_schema_version",
            "wrong-schema",
            "binding mismatch",
        ),
        (
            "render_protocol",
            {"renderer": "FixedMeasurementRenderer", "renderer_samples": 4, "ray_chunk_size": 2048},
            "binding mismatch",
        ),
        (
            "budget_protocol",
            {"max_units": 11.0, "policy_work_units": 1},
            "binding mismatch",
        ),
    ],
)
def test_learned_policy_rejects_mismatched_binding_before_episode_io(
    tmp_path: Path,
    binding_field: str,
    replacement: object,
    message: str,
) -> None:
    checkpoint, expected_state_hash = _write_checkpoint(tmp_path)
    entry, prepared_root = _write_episode(tmp_path, split_id="dev", scene_id="scene_a")
    policy_checkpoint = _write_policy_checkpoint(
        tmp_path,
        model_content_hash=expected_state_hash,
        render_samples=2,
        ray_chunk_size=2048,
        max_units=1e12,
        binding_field=binding_field,
        replacement=replacement,
    )

    with pytest.raises(ValueError, match=message):
        checkpoint_report.evaluate_dynamic_checkpoint(
            [entry],
            checkpoint,
            tmp_path / f"bad_{binding_field}",
            device="cpu",
            policies=("OFF", "LEARNED"),
            render_samples=2,
            ray_chunk_size=2048,
            policy_checkpoint=policy_checkpoint,
            prepared_root=prepared_root,
        )


def test_learned_evaluation_opens_query_only_after_seal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint, expected_state_hash = _write_checkpoint(tmp_path)
    entry, prepared_root = _write_episode(tmp_path, split_id="dev", scene_id="scene_a")
    policy_checkpoint = _write_policy_checkpoint(
        tmp_path,
        model_content_hash=expected_state_hash,
        render_samples=2,
        ray_chunk_size=2048,
        max_units=1e12,
    )
    query_path = Path(entry["query_manifest"])
    query = json.loads(query_path.read_text(encoding="utf-8"))
    for frame in query["query"]:
        Path(frame["rgb"]).write_bytes(b"query RGB must not be opened before seal")
        Path(frame["depth"]).write_bytes(b"query depth must not be opened before seal")

    calls: list[tuple[str, tuple[int, ...]]] = []

    def fake_evaluate(sealed, *_args, **_kwargs):
        calls.append(("evaluate", sealed.observed_ids))
        return {
            "per_query": [
                {
                    "frame_id": frame_id,
                    "rgb_mse": 0.0,
                    "rgb_psnr": float("inf"),
                    "depth_abs_rel": 0.0,
                    "opacity_mean": 0.0,
                    "coverage": 0.0,
                }
                for frame_id in range(12, 16)
            ],
            "averages": {
                "rgb_mse": 0.0,
                "rgb_psnr": float("inf"),
                "depth_abs_rel": 0.0,
                "opacity_mean": 0.0,
                "coverage": 0.0,
            },
        }

    monkeypatch.setattr(checkpoint_report, "evaluate_sealed", fake_evaluate)
    checkpoint_report.evaluate_dynamic_checkpoint(
        [entry],
        checkpoint,
        tmp_path / "sealed_boundary",
        device="cpu",
        policies=("OFF", "LEARNED"),
        render_samples=2,
        policy_checkpoint=policy_checkpoint,
        prepared_root=prepared_root,
    )
    assert len(calls) == 2
    assert all(observed_ids == tuple(range(12)) for _, observed_ids in calls)


def test_scene_aggregation_uses_scene_means_and_matched_off_deltas() -> None:
    records = [
        _record("scene_a", "OFF", 10.0, 20.0, 1.0),
        _record("scene_a", "FUSE", 7.0, 22.0, 0.8),
        _record("scene_b", "OFF", 4.0, 24.0, 0.5),
        _record("scene_b", "FUSE", 1.0, 26.0, 0.3),
    ]

    summaries, scene_results, metric_names = checkpoint_report.aggregate_scene_records(records)

    fuse = next(summary for summary in summaries if summary["policy"] == "FUSE")
    assert metric_names == ("coverage", "rgb_mse", "rgb_psnr")
    assert fuse["scene_count"] == 2
    assert fuse["metrics_mean"]["rgb_mse"] == 4.0
    assert fuse["metric_valid_scene_counts"]["rgb_mse"] == 2
    assert fuse["paired_delta_vs_off_mean"]["rgb_mse"] == -3.0
    assert {(row["scene_id"], row["policy"]) for row in scene_results} == {
        ("scene_a", "OFF"),
        ("scene_a", "FUSE"),
        ("scene_b", "OFF"),
        ("scene_b", "FUSE"),
    }


def _write_checkpoint(directory: Path) -> tuple[Path, str]:
    config = CarrierConfig(
        feature_dim=2,
        hidden_dim=2,
        expansion_dim=2,
        grid_size=(2, 2, 2),
        local_bounds_m=((-1.0, -1.0, 0.1), (1.0, 1.0, 2.0)),
        token_count=4,
    )
    carrier = DynamicSceneCarrier(config)
    write_rule = DirectWriteRule(config, WriteConfig(learning_rate=0.02, max_update_norm=0.1))
    checkpoint = directory / "dynamic.pt"
    save_dynamic_checkpoint(
        checkpoint,
        carrier,
        write_rule,
        phase="test",
        provenance={"split": "train"},
    )
    state_hash = hash_value(
        {"carrier": carrier.state_dict(), "write_rule": write_rule.state_dict()}
    )
    return checkpoint, state_hash


def _write_policy_checkpoint(
    directory: Path,
    *,
    model_content_hash: str,
    render_samples: int,
    ray_chunk_size: int,
    max_units: float,
    binding_field: str | None = None,
    replacement: object = None,
) -> Path:
    policy = LearnedActionPolicy(hidden_dim=4, seed=11)
    binding: dict[str, object] = {
        "model_content_hash": model_content_hash,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "render_protocol": {
            "renderer": "FixedMeasurementRenderer",
            "renderer_samples": render_samples,
            "ray_chunk_size": ray_chunk_size,
        },
        "budget_protocol": {
            "max_units": float(max_units),
            "policy_work_units": policy.declared_work_units,
        },
        "utility_protocol": {
            "horizon": 1,
            "utility": "mean_masked_depth_abs_rel",
        },
        "teacher_dataset_hash": "dataset-test",
        "teacher_source_hashes": {"rows": "rows-test"},
    }
    serialized_binding_replacement = None
    if binding_field == "feature_schema_version":
        serialized_binding_replacement = replacement
    elif binding_field is not None:
        binding[binding_field] = replacement
    path = directory / "policy.pt"
    save_policy_checkpoint(path, policy, binding=binding, provenance={"split": "train"})
    if serialized_binding_replacement is not None:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        payload["binding"][binding_field] = serialized_binding_replacement
        torch.save(payload, path)
    return path


def _write_episode(
    directory: Path, *, split_id: str, scene_id: str
) -> tuple[dict[str, object], Path]:
    prepared_root = directory / "prepared"
    prepared_split = {"train": "train", "dev": "val"}[split_id]
    scene_root = prepared_root / prepared_split / scene_id
    rgb_root = scene_root / "rgb"
    depth_root = scene_root / "depth"
    rgb_root.mkdir(parents=True)
    depth_root.mkdir(parents=True)
    for frame_id in range(16):
        value = 16 + frame_id * 8
        Image.fromarray(np.full((2, 2, 3), value, dtype=np.uint8)).save(
            rgb_root / f"{frame_id:06d}.png"
        )
    for frame_id in range(12, 16):
        np.save(depth_root / f"{frame_id:06d}.npy", np.ones((2, 2), dtype=np.float32))
    identity = {
        "episode_id": f"{split_id}--{scene_id}--cam_00",
        "scene_id": scene_id,
        "split_id": split_id,
        "query_vault_id": f"qv-{scene_id}",
    }
    online = {
        "schema_version": "mcss.dynamic.online_episode.v1",
        **identity,
        "image_size": [2, 2],
        "declared_length": 12,
        "warmup": [
            _online_frame(frame_id, rgb_root / f"{frame_id:06d}.png")
            for frame_id in range(4)
        ],
        "stream": [
            _online_frame(frame_id, rgb_root / f"{frame_id:06d}.png")
            for frame_id in range(4, 12)
        ],
    }
    query = {
        "schema_version": "mcss.dynamic.query_vault.v1",
        **identity,
        "image_size": [2, 2],
        "query": [
            _query_frame(
                frame_id,
                rgb_root / f"{frame_id:06d}.png",
                depth_root / f"{frame_id:06d}.npy",
            )
            for frame_id in range(12, 16)
        ],
    }
    artifact_root = directory / "artifacts"
    artifact_root.mkdir(exist_ok=True)
    online_path = artifact_root / f"{scene_id}.online.json"
    query_path = artifact_root / f"{scene_id}.query.json"
    online_path.write_text(json.dumps(online), encoding="utf-8")
    query_path.write_text(json.dumps(query), encoding="utf-8")
    return (
        {
            **identity,
            "online_manifest": str(online_path),
            "query_manifest": str(query_path),
            "stream_steps": 8,
            "query_count": 4,
        },
        prepared_root,
    )


def _online_frame(frame_id: int, rgb: Path) -> dict[str, object]:
    return {
        "frame_id": frame_id,
        "intrinsics": [[2.0, 0.0, 0.5], [0.0, 2.0, 0.5], [0.0, 0.0, 1.0]],
        "c2w": [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        "rgb": str(rgb),
    }


def _query_frame(frame_id: int, rgb: Path, depth: Path) -> dict[str, object]:
    return {**_online_frame(frame_id, rgb), "depth": str(depth)}


def _record(scene_id: str, policy: str, rgb_mse: float, rgb_psnr: float, coverage: float) -> dict:
    return {
        "episode_id": f"train--{scene_id}--cam_00",
        "scene_id": scene_id,
        "split_id": "train",
        "policy": policy,
        "metrics": {
            "averages": {
                "rgb_mse": rgb_mse,
                "rgb_psnr": rgb_psnr,
                "coverage": coverage,
            }
        },
    }


def _sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()
