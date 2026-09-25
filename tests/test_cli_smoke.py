import hashlib
import json
import math
from pathlib import Path

import pytest
import torch

import mcss.engine as engine_module
from mcss.cli import main
from mcss.config import load_config
from mcss.engine import (
    _random_state,
    _restore_random_state,
    _ScalarTracker,
    _scheduled_learning_rate,
)


def _write_config(
    path: Path,
    output_dir: Path,
    *,
    device: str = "cpu",
    amp: bool = False,
    amp_dtype: str = "float16",
) -> None:
    path.write_text(
        f"""
seed: 3
device: {device}
dataset:
  name: synthetic
  root: null
  image_size: [8, 8]
  context_views: 2
  target_views: 1
  length: 2
model:
  mode: fixed
  voxel_resolution: 8
  image_feature_dim: 8
  state_feature_dim: 8
  refinement_blocks: 1
  n_samples: 8
  ray_chunk_size: 128
training:
  output_dir: {output_dir.as_posix()}
  batch_size: 1
  learning_rate: 0.001
  max_steps: 1
  num_workers: 0
  amp: {str(amp).lower()}
  amp_dtype: {amp_dtype}
  log_every: 1
  checkpoint_every: 1
  measurements: [rgb, depth, normal, point, visibility]
  lr_schedule: warmup_cosine
  warmup_steps: 1
  min_lr_ratio: 0.1
  ema_decay: 0.9
""".strip(),
        encoding="utf-8",
    )


def _write_context_subset_training_config(
    path: Path,
    output_dir: Path,
    *,
    device: str = "cpu",
    amp: bool = False,
    amp_dtype: str = "float16",
) -> None:
    path.write_text(
        f"""
seed: 17
device: {device}
dataset:
  name: synthetic
  root: null
  image_size: [8, 8]
  context_views: 4
  target_views: 1
  length: 2
model:
  mode: fixed
  state_architecture: evidence_residual
  voxel_resolution: [8, 8, 8]
  image_feature_dim: 8
  state_feature_dim: 8
  refinement_blocks: 1
  evidence_temperature: 0.1
  observed_residual_floor: 0.1
  completion_residual_scale: 4.0
  appearance_resolution_scale: 1
  n_samples: 8
  ray_chunk_size: 128
training:
  output_dir: {output_dir.as_posix()}
  batch_size: 1
  learning_rate: 0.001
  max_steps: 2
  num_workers: 0
  amp: {str(amp).lower()}
  amp_dtype: {amp_dtype}
  log_every: 1
  checkpoint_every: 2
  measurements: [rgb, depth, visibility]
  lr_schedule: warmup_cosine
  warmup_steps: 2
  context_subset_geometry_weight: 0.1
""".strip(),
        encoding="utf-8",
    )


def test_cli_trains_and_evaluates_synthetic_smoke(tmp_path: Path) -> None:
    config_path = tmp_path / "smoke.yaml"
    output_dir = tmp_path / "outputs"
    _write_config(config_path, output_dir)

    assert main(["train", "--config", str(config_path)]) == 0
    checkpoint = output_dir / "checkpoints" / "step_000001.pt"
    assert checkpoint.is_file()
    assert (output_dir / "train.jsonl").is_file()
    record = json.loads((output_dir / "train.jsonl").read_text(encoding="utf-8").splitlines()[-1])
    expected_fields = {
        "step",
        "epoch",
        "epoch_progress",
        "batch_in_epoch",
        "batches_per_epoch",
        "lr",
        "grad/norm",
        "amp/scale",
        "optimizer/step",
        "optimizer/updated",
        "optimizer/skipped",
        "loss",
        "window/loss",
        "ema/loss",
        "time/step_seconds",
        "scene_ids",
        "valid/support",
        "valid/depth",
    }
    assert expected_fields <= set(record)
    assert record["optimizer/updated"] is True
    assert record["optimizer/skipped"] is False
    assert record["scene_ids"][0].startswith("synthetic_")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert payload["optimizer_step"] == 1
    assert "telemetry" in payload

    assert main(["evaluate", "--config", str(config_path), "--checkpoint", str(checkpoint)]) == 0
    assert (output_dir / "evaluation.json").is_file()
    report_path = output_dir / "evaluation_report.json"
    assert report_path.is_file()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["schema_version"] == "mcss.evaluation.v1"
    assert report["checkpoint"]["absolute_path"] == str(checkpoint.resolve())
    assert report["checkpoint"]["sha256"] == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert report["checkpoint"]["step"] == 1
    assert report["resolved_config"]["dataset"]["name"] == "synthetic"
    assert report["evaluation"]["sample_count"] == 2
    assert report["evaluation"]["scene_count"] == 2
    assert report["legacy_window_macro_metrics"]
    assert report["valid_count_weighted_metrics"]
    assert len(report["per_scene_metrics"]) == 2
    assert report["finite_window_counts"]
    assert report["aggregation_definitions"]


def test_disabled_context_subset_path_is_not_called_or_checkpointed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "disabled.yaml"
    output_dir = tmp_path / "disabled_outputs"
    _write_config(config_path, output_dir)

    def fail_if_called(*args: object, **kwargs: object) -> None:
        raise AssertionError("disabled context-subset path was executed")

    monkeypatch.setattr(engine_module, "drop_context_view", fail_if_called)

    assert main(["train", "--config", str(config_path)]) == 0
    record = json.loads((output_dir / "train.jsonl").read_text(encoding="utf-8"))
    payload = torch.load(
        output_dir / "checkpoints" / "step_000001.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert "context_subset" not in record
    assert "loss/context_subset_geometry" not in record
    assert "context_subset_generator_state" not in payload


def test_enabled_context_subset_path_logs_auditable_metadata(tmp_path: Path) -> None:
    config_path = tmp_path / "enabled.yaml"
    output_dir = tmp_path / "enabled_outputs"
    _write_context_subset_training_config(config_path, output_dir)

    assert main(["train", "--config", str(config_path)]) == 0

    records = [
        json.loads(line)
        for line in (output_dir / "train.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [record["context_subset/effective_weight"] for record in records] == [0.05, 0.1]
    for record in records:
        assert math.isfinite(record["loss/context_subset_geometry"])
        assert math.isfinite(record["loss/context_subset_depth"])
        assert math.isfinite(record["loss/context_subset_visibility"])
        assert math.isfinite(record["context_subset/residual_abs_diff"])
        assert math.isfinite(record["context_subset/full_residual_abs_p50"])
        metadata = record["context_subset"]
        assert metadata["dropped_index"] in {0, 1, 2, 3}
        assert len(metadata["retained_indices"]) == 3
        assert metadata["dropped_index"] not in metadata["retained_indices"]
        assert len(metadata["per_scene"]) == 1
        assert metadata["per_scene"][0]["scene_id"].startswith("synthetic_")

    checkpoint = output_dir / "checkpoints" / "step_000002.pt"
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert "context_subset_generator_state" in payload
    assert payload["context_subset_generator_state"].dtype == torch.uint8


def test_context_subset_checkpoint_restores_next_dropped_view(tmp_path: Path) -> None:
    config_path = tmp_path / "resume_enabled.yaml"
    output_dir = tmp_path / "resume_enabled_outputs"
    _write_context_subset_training_config(config_path, output_dir)
    config = load_config(config_path)
    trainer = engine_module.Trainer(config)
    assert trainer.context_subset_generator is not None

    torch.randint(4, (1,), generator=trainer.context_subset_generator)
    trainer.step = 1
    checkpoint = trainer.save_checkpoint()
    expected_next = int(
        torch.randint(4, (1,), generator=trainer.context_subset_generator).item()
    )

    resumed = engine_module.Trainer(config)
    resumed.load_checkpoint(checkpoint, load_optimizer=True)
    assert resumed.context_subset_generator is not None
    actual_next = int(
        torch.randint(4, (1,), generator=resumed.context_subset_generator).item()
    )

    assert actual_next == expected_next


def test_cli_evaluate_overrides_are_explicit_and_immutable(tmp_path: Path) -> None:
    config_path = tmp_path / "smoke.yaml"
    training_output = tmp_path / "training"
    evaluation_output = tmp_path / "evaluation"
    _write_config(config_path, training_output)
    assert main(["train", "--config", str(config_path)]) == 0
    checkpoint = training_output / "checkpoints" / "step_000001.pt"

    command = [
        "evaluate",
        "--config",
        str(config_path),
        "--checkpoint",
        str(checkpoint),
        "--output-dir",
        str(evaluation_output),
        "--n-samples",
        "12",
        "--ray-chunk-size",
        "16",
    ]
    assert main(command) == 0
    report_path = evaluation_output / "evaluation_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["resolved_config"]["training"]["output_dir"] == str(
        evaluation_output.resolve()
    )
    assert report["resolved_config"]["model"]["n_samples"] == 12
    assert report["resolved_config"]["model"]["ray_chunk_size"] == 16
    original = report_path.read_bytes()

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        main(command)
    assert report_path.read_bytes() == original


def test_cli_download_dry_run_does_not_create_data_root(tmp_path: Path) -> None:
    root = tmp_path / "hypersim"

    assert (
        main(
            [
                "download",
                "hypersim",
                "--root",
                str(root),
                "--scenes",
                "ai_001_001",
                "--dry-run",
            ]
        )
        == 0
    )
    assert not root.exists()


def test_restore_random_state_accepts_cuda_mapped_cpu_rng_state() -> None:
    if not torch.cuda.is_available():
        return
    state = _random_state()
    state["torch"] = state["torch"].to("cuda")
    state["cuda"] = [cuda_state.to("cuda") for cuda_state in state["cuda"]]

    _restore_random_state(state)


def test_warmup_cosine_learning_rate_has_explicit_endpoints() -> None:
    base = 1e-3

    assert _scheduled_learning_rate(base, 1, 6, 2, 0.1) == pytest.approx(0.5e-3)
    assert _scheduled_learning_rate(base, 2, 6, 2, 0.1) == pytest.approx(1e-3)
    assert _scheduled_learning_rate(base, 4, 6, 2, 0.1) == pytest.approx(0.55e-3)
    assert _scheduled_learning_rate(base, 6, 6, 2, 0.1) == pytest.approx(0.1e-3)


def test_scalar_tracker_reports_window_means_and_resume_safe_ema() -> None:
    tracker = _ScalarTracker(ema_decay=0.5)
    tracker.update({"loss": 2.0, "loss/depth": 1.0})
    tracker.update({"loss": 4.0, "loss/depth": 3.0})

    first = tracker.snapshot(reset_window=True)
    state = tracker.state_dict()
    resumed = _ScalarTracker(ema_decay=0.5)
    resumed.load_state_dict(state)
    resumed.update({"loss": 6.0, "loss/depth": 5.0})
    second = resumed.snapshot(reset_window=True)

    assert first["window/loss"] == 3.0
    assert first["ema/loss"] == 3.0
    assert first["window/loss/depth"] == 2.0
    assert second["window/loss"] == 6.0
    assert second["ema/loss"] == 4.5


def test_bfloat16_cuda_training_updates_without_scaler_skip(tmp_path: Path) -> None:
    if not torch.cuda.is_available():
        return
    config_path = tmp_path / "cuda_bfloat16.yaml"
    output_dir = tmp_path / "cuda_outputs"
    _write_config(
        config_path,
        output_dir,
        device="cuda",
        amp=True,
        amp_dtype="bfloat16",
    )

    assert main(["train", "--config", str(config_path)]) == 0

    record = json.loads((output_dir / "train.jsonl").read_text(encoding="utf-8"))
    assert record["optimizer/updated"] is True
    assert record["optimizer/skipped"] is False
    assert math.isfinite(record["grad/norm"])


def test_context_subset_bfloat16_cuda_training_is_finite(tmp_path: Path) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    if not torch.cuda.is_bf16_supported():
        pytest.skip("CUDA device does not support bfloat16")
    config_path = tmp_path / "context_subset_cuda_bfloat16.yaml"
    output_dir = tmp_path / "context_subset_cuda_outputs"
    _write_context_subset_training_config(
        config_path,
        output_dir,
        device="cuda",
        amp=True,
        amp_dtype="bfloat16",
    )

    assert main(["train", "--config", str(config_path)]) == 0

    records = [
        json.loads(line)
        for line in (output_dir / "train.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert records[-1]["optimizer/updated"] is True
    assert records[-1]["optimizer/skipped"] is False
    for record in records:
        assert math.isfinite(record["loss/context_subset_geometry"])
        assert math.isfinite(record["loss/context_subset_depth"])
        assert math.isfinite(record["loss/context_subset_visibility"])
        assert math.isfinite(record["grad/norm"])
