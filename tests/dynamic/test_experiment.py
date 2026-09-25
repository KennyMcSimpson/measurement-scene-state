import json

import numpy as np
import pytest
import torch
from PIL import Image

from mcss.dynamic.checkpoint import load_dynamic_checkpoint
from mcss.dynamic.config import CarrierConfig, WriteConfig
from mcss.training.experiment import (
    CpuEpisodeCache,
    DynamicExperimentConfig,
    load_train_episode_index,
    run_dynamic_experiment,
)


def _write_fixture(
    tmp_path,
    *,
    observed_ids: tuple[int, ...] = tuple(range(12)),
    query_ids: tuple[int, ...] = tuple(range(12, 16)),
    all_invalid_query: int | None = None,
):
    prepared = tmp_path / "prepared"
    rgb_root = prepared / "train" / "scene-a" / "rgb"
    depth_root = prepared / "train" / "scene-a" / "depth"
    rgb_root.mkdir(parents=True)
    depth_root.mkdir(parents=True)
    image_size = [4, 6]
    for frame_id in (*observed_ids, *query_ids):
        Image.fromarray(np.full((4, 6, 3), 30 + frame_id, dtype=np.uint8)).save(
            rgb_root / f"{frame_id:06d}.png"
        )
    for frame_id in query_ids:
        depth = np.full((4, 6), 2.0, dtype=np.float32)
        if frame_id == query_ids[0]:
            depth[0, 0] = np.nan
        if frame_id == all_invalid_query:
            depth.fill(np.nan)
        np.save(depth_root / f"{frame_id:06d}.npy", depth)

    def record(frame_id, *, query=False):
        result = {
            "frame_id": frame_id,
            "intrinsics": [[4.0, 0.0, 2.5], [0.0, 4.0, 1.5], [0.0, 0.0, 1.0]],
            "c2w": np.eye(4).tolist(),
            "rgb": str(rgb_root / f"{frame_id:06d}.png"),
        }
        if query:
            result["depth"] = str(depth_root / f"{frame_id:06d}.npy")
        return result

    identity = {
        "episode_id": "train--scene-a--cam-00",
        "scene_id": "scene-a",
        "split_id": "train",
        "query_vault_id": "query-a",
        "image_size": image_size,
    }
    online_path = tmp_path / "online.json"
    online_path.write_text(
        json.dumps(
            {
                "schema_version": "mcss.dynamic.online_episode.v1",
                **identity,
                "declared_length": 12,
                "warmup": [record(frame_id) for frame_id in observed_ids[:4]],
                "stream": [record(frame_id) for frame_id in observed_ids[4:]],
            }
        ),
        encoding="utf-8",
    )
    query_path = tmp_path / "query.json"
    query_path.write_text(
        json.dumps(
            {
                "schema_version": "mcss.dynamic.query_vault.v1",
                **identity,
                "query": [record(frame_id, query=True) for frame_id in query_ids],
            }
        ),
        encoding="utf-8",
    )
    entry = {
        "episode_id": identity["episode_id"],
        "scene_id": identity["scene_id"],
        "split_id": "train",
        "query_vault_id": identity["query_vault_id"],
        "online_manifest": str(online_path),
        "query_manifest": str(query_path),
        "stream_steps": 8,
        "query_count": 4,
    }
    index_path = tmp_path / "episode_index_train.json"
    index_path.write_text(
        json.dumps({"schema_version": "mcss.dynamic.episode_index.v1", "episodes": [entry]}),
        encoding="utf-8",
    )
    return index_path, prepared, entry


def _config() -> DynamicExperimentConfig:
    return DynamicExperimentConfig(
        carrier=CarrierConfig(
            feature_dim=4,
            hidden_dim=4,
            expansion_dim=6,
            grid_size=(4, 4, 4),
            local_bounds_m=((-1.0, -1.0, 0.1), (1.0, 1.0, 3.0)),
            token_count=12,
        ),
        write=WriteConfig(learning_rate=0.05, max_update_norm=0.2),
        stage_a_steps=2,
        stage_b_steps=2,
        renderer_samples=2,
        ray_chunk_size=64,
        torch_num_threads=1,
    )


def test_two_stage_experiment_writes_checkpoints_history_and_resume(tmp_path) -> None:
    index_path, prepared, _ = _write_fixture(tmp_path)
    output = tmp_path / "training"

    result = run_dynamic_experiment(
        index_path,
        output,
        device="cpu",
        config=_config(),
        prepared_root=prepared,
    )

    assert result["status"] == "complete"
    assert (output / "initial.pt").is_file()
    assert (output / "phase_a_final.pt").is_file()
    assert (output / "phase_b_final.pt").is_file()
    _, _, initial = load_dynamic_checkpoint(output / "initial.pt")
    _, _, phase_a = load_dynamic_checkpoint(output / "phase_a_final.pt")
    _, _, phase_b = load_dynamic_checkpoint(output / "phase_b_final.pt")
    assert initial["phase"] == "initial"
    assert phase_a["phase"] == "phase_a_final"
    assert phase_b["phase"] == "phase_b_final"
    assert initial["provenance"]["training_mode"] == "offline-train-only"

    records = [json.loads(line) for line in (output / "history.jsonl").read_text().splitlines()]
    assert [record["global_step"] for record in records] == [1, 2, 3, 4]
    assert [record["phase"] for record in records] == ["A", "A", "B", "B"]
    assert all(record["action"] == "OFF" for record in records[:2])
    assert all(record["action"] in {"FUSE", "COMPLETE", "ALL"} for record in records[2:])
    assert all(record["valid_depth_fraction"] > 0.0 for record in records)
    assert all(record["write_gradient_norm"] == 0.0 for record in records[:2])
    assert set(records[0]) == {
        "global_step",
        "phase",
        "phase_step",
        "scene_id",
        "query_frame_id",
        "action",
        "loss",
        "terms",
        "carrier_gradient_norm",
        "write_gradient_norm",
        "seconds",
        "valid_depth_fraction",
    }

    resumed = run_dynamic_experiment(
        index_path,
        output,
        device="cpu",
        config=_config(),
        prepared_root=prepared,
        resume=True,
    )
    assert resumed["status"] == "complete"
    assert len((output / "history.jsonl").read_text().splitlines()) == 4


def test_train_index_rejects_dev_before_manifest_loading(tmp_path) -> None:
    index_path = tmp_path / "index.json"
    index_path.write_text(
        json.dumps(
            {
                "schema_version": "mcss.dynamic.episode_index.v1",
                "episodes": [{"episode_id": "dev--unreadable", "split_id": "dev"}],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="non-train"):
        load_train_episode_index(index_path)


def test_experiment_accepts_gapped_but_ordered_frame_ids(tmp_path) -> None:
    observed_ids = (10, 13, 30, 32, 34, 37, 41, 42, 43, 44, 46, 47)
    query_ids = (50, 52, 56, 60)
    index_path, prepared, _ = _write_fixture(
        tmp_path,
        observed_ids=observed_ids,
        query_ids=query_ids,
    )

    result = run_dynamic_experiment(
        index_path,
        tmp_path / "training",
        device="cpu",
        config=DynamicExperimentConfig(
            **{**_config().__dict__, "stage_a_steps": 1, "stage_b_steps": 1}
        ),
        prepared_root=prepared,
    )

    assert result["status"] == "complete"


def test_cpu_cache_sanitizes_invalid_depth_but_preserves_per_query_validity(tmp_path) -> None:
    index_path, _, entry = _write_fixture(tmp_path, all_invalid_query=15)
    entries = load_train_episode_index(index_path)

    cached = CpuEpisodeCache(entries, capacity=1).get(entry["episode_id"])

    assert 0.0 < cached.valid_depth_fractions[0] < 1.0
    assert cached.valid_depth_fractions[3] == 0.0
    assert cached.supervision.query_depth[:, 0].isfinite().all()
    assert (cached.supervision.query_depth[:, 3] == 0).all()


def test_experiment_fails_when_a_selected_query_has_no_valid_depth(tmp_path) -> None:
    index_path, prepared, _ = _write_fixture(tmp_path, all_invalid_query=15)
    output = tmp_path / "training"
    config = DynamicExperimentConfig(
        **{
            **_config().__dict__,
            "stage_a_steps": 4,
            "stage_b_steps": 0,
        }
    )

    with pytest.raises(ValueError, match="selected query frame"):
        run_dynamic_experiment(
            index_path,
            output,
            device="cpu",
            config=config,
            prepared_root=prepared,
        )

    status = json.loads((output / "status.json").read_text())
    assert status["status"] == "failed"


def test_resume_rejects_inconsistent_phase_optimizer_and_counters(tmp_path) -> None:
    index_path, prepared, _ = _write_fixture(tmp_path)
    output = tmp_path / "training"
    config = DynamicExperimentConfig(
        **{**_config().__dict__, "stage_a_steps": 1, "stage_b_steps": 1}
    )
    run_dynamic_experiment(
        index_path,
        output,
        device="cpu",
        config=config,
        prepared_root=prepared,
    )
    payload = torch.load(output / "resume.pt", map_location="cpu", weights_only=True)
    payload["phase"] = "A"
    torch.save(payload, output / "resume.pt")

    with pytest.raises(ValueError, match="phase, optimizer phase, and stage counters"):
        run_dynamic_experiment(
            index_path,
            output,
            device="cpu",
            config=config,
            prepared_root=prepared,
            resume=True,
        )


def test_mixed_schedule_logs_actual_actions_and_resume_is_exact(tmp_path) -> None:
    index_path, prepared, _ = _write_fixture(tmp_path)
    output = tmp_path / "mixed-training"
    config = DynamicExperimentConfig(
        **{
            **_config().__dict__,
            "stage_a_steps": 0,
            "stage_b_steps": 4,
            "action_schedule_mode": "mixed",
            "stage_b_actions": ("OFF", "FUSE", "COMPLETE", "ALL"),
            "stage_b_trajectories": (
                ("OFF", "FUSE", "OFF", "ALL", "OFF", "COMPLETE", "OFF", "ALL"),
                ("ALL", "OFF", "COMPLETE", "OFF", "FUSE", "OFF", "ALL", "OFF"),
            ),
        }
    )

    result = run_dynamic_experiment(
        index_path, output, device="cpu", config=config, prepared_root=prepared
    )
    assert result["status"] == "complete"
    records = [json.loads(line) for line in (output / "history.jsonl").read_text().splitlines()]
    assert len(records) == 4
    assert all(record["phase"] == "B" for record in records)
    assert all(len(record["actions"]) == 8 for record in records)
    assert all(set(record["actions"]) <= {"OFF", "FUSE", "COMPLETE", "ALL"} for record in records)
    assert any(record["action"] == "MIXED" for record in records)
    before = (output / "history.jsonl").read_text()

    resumed = run_dynamic_experiment(
        index_path,
        output,
        device="cpu",
        config=config,
        prepared_root=prepared,
        resume=True,
    )
    assert resumed["status"] == "complete"
    assert (output / "history.jsonl").read_text() == before


def test_init_checkpoint_is_strict_and_recorded_for_a_fresh_run(tmp_path) -> None:
    index_path, prepared, _ = _write_fixture(tmp_path)
    source_output = tmp_path / "source-training"
    source_config = DynamicExperimentConfig(
        **{**_config().__dict__, "stage_a_steps": 0, "stage_b_steps": 0}
    )
    run_dynamic_experiment(
        index_path, source_output, device="cpu", config=source_config, prepared_root=prepared
    )

    output = tmp_path / "warm-start-training"
    result = run_dynamic_experiment(
        index_path,
        output,
        device="cpu",
        config=source_config,
        prepared_root=prepared,
        init_checkpoint=source_output / "phase_b_final.pt",
    )
    assert result["status"] == "complete"
    metadata = json.loads((output / "run_metadata.json").read_text())
    assert metadata["initial_checkpoint_sha256"]
    resume = torch.load(output / "resume.pt", map_location="cpu", weights_only=True)
    assert resume["initial_checkpoint_sha256"] == metadata["initial_checkpoint_sha256"]

    mismatched = DynamicExperimentConfig(
        **{
            **source_config.__dict__,
            "carrier": CarrierConfig(
                feature_dim=5,
                hidden_dim=4,
                expansion_dim=6,
                grid_size=(4, 4, 4),
                local_bounds_m=((-1.0, -1.0, 0.1), (1.0, 1.0, 3.0)),
                token_count=12,
            ),
        }
    )
    with pytest.raises(ValueError, match="CarrierConfig/WriteConfig"):
        run_dynamic_experiment(
            index_path,
            tmp_path / "mismatch-training",
            device="cpu",
            config=mismatched,
            prepared_root=prepared,
            init_checkpoint=source_output / "phase_b_final.pt",
        )
