import hashlib
import json
from pathlib import Path

import pytest
import torch

from mcss.cli import main as mcss_main
from mcss.oracle_runner import main as oracle_main


def _write_synthetic_config(
    path: Path,
    training_output: Path,
    *,
    appearance_resolution_scale: int = 1,
) -> None:
    path.write_text(
        f"""
seed: 7
device: cpu
dataset:
  name: synthetic
  root: null
  image_size: [8, 8]
  context_views: 2
  target_views: 2
  length: 1
model:
  mode: fixed
  state_architecture: evidence_residual
  voxel_resolution: 8
  image_feature_dim: 8
  state_feature_dim: 8
  refinement_blocks: 1
  appearance_resolution_scale: {appearance_resolution_scale}
  evidence_temperature: 0.1
  observed_residual_floor: 0.1
  completion_residual_scale: 4.0
  n_samples: 8
  ray_chunk_size: 128
training:
  output_dir: {training_output.as_posix()}
  batch_size: 1
  learning_rate: 0.001
  max_steps: 1
  num_workers: 0
  amp: false
  amp_dtype: float16
  log_every: 1
  checkpoint_every: 1
  measurements: [rgb, depth, normal, point, visibility]
""".strip(),
        encoding="utf-8",
    )


def test_oracle_runner_writes_provenance_complete_finite_report(tmp_path: Path) -> None:
    config = tmp_path / "synthetic.yaml"
    training_output = tmp_path / "training"
    oracle_output = tmp_path / "oracle"
    _write_synthetic_config(config, training_output)
    assert mcss_main(["train", "--config", str(config)]) == 0
    checkpoint = training_output / "checkpoints" / "step_000001.pt"

    assert (
        oracle_main(
            [
                "--config",
                str(config),
                "--checkpoint",
                str(checkpoint),
                "--output-dir",
                str(oracle_output),
                "--steps",
                "1",
                "--learning-rate",
                "0.01",
                "--seed",
                "19",
            ]
        )
        == 0
    )

    report = json.loads((oracle_output / "oracle.json").read_text(encoding="utf-8"))
    assert report["schema_version"] == "mcss.appearance_oracle.v1"
    assert report["diagnostic_only"] is True
    assert report["source"]["config_sha256"] == hashlib.sha256(config.read_bytes()).hexdigest()
    assert report["source"]["checkpoint_sha256"] == hashlib.sha256(
        checkpoint.read_bytes()
    ).hexdigest()
    assert report["source"]["scene_ids"] == ["synthetic_000000"]
    assert set(report["variants"]) == {
        "native_shared",
        "highres_shared",
        "native_per_view",
    }
    counts = {result["parameter_count"] for result in report["variants"].values()}
    assert len(counts) == 3
    for result in report["variants"].values():
        assert result["steps"] == 1
        assert result["parameter_count"] > 0
        for stage in ("initial", "best", "final"):
            assert torch.isfinite(torch.tensor(result[stage]["rgb/mse"]))
            assert torch.isfinite(torch.tensor(result[stage]["rgb/psnr"]))
            assert torch.isfinite(torch.tensor(result[stage]["rgb/ssim"]))

    records = [
        json.loads(line)
        for line in (oracle_output / "oracle.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(records) == 3
    assert {record["variant"] for record in records} == set(report["variants"])


def test_typed_oracle_runner_uses_existing_appearance_grid_and_refuses_overwrite(
    tmp_path: Path,
) -> None:
    config = tmp_path / "synthetic_typed.yaml"
    training_output = tmp_path / "training_typed"
    oracle_output = tmp_path / "oracle_typed"
    _write_synthetic_config(config, training_output, appearance_resolution_scale=2)
    assert mcss_main(["train", "--config", str(config)]) == 0
    checkpoint = training_output / "checkpoints" / "step_000001.pt"
    arguments = [
        "--config",
        str(config),
        "--checkpoint",
        str(checkpoint),
        "--output-dir",
        str(oracle_output),
        "--field",
        "typed_appearance",
        "--steps",
        "1",
    ]

    assert oracle_main(arguments) == 0
    report = json.loads((oracle_output / "oracle.json").read_text(encoding="utf-8"))
    assert report["field"] == "typed_appearance"
    assert report["native_state_shape"] == [8, 8, 8]
    assert report["optimized_field_shape"] == [16, 16, 16]
    assert set(report["variants"]) == {"typed_shared", "typed_per_view"}
    assert all(result["spatial_shape"] == [16, 16, 16] for result in report["variants"].values())

    log_before = (oracle_output / "oracle.jsonl").read_bytes()
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        oracle_main(arguments)
    assert (oracle_output / "oracle.jsonl").read_bytes() == log_before


def test_typed_oracle_runner_rejects_model_without_state_appearance(tmp_path: Path) -> None:
    config = tmp_path / "synthetic_native.yaml"
    training_output = tmp_path / "training_native"
    _write_synthetic_config(config, training_output)
    assert mcss_main(["train", "--config", str(config)]) == 0

    with pytest.raises(ValueError, match="StateAppearance"):
        oracle_main(
            [
                "--config",
                str(config),
                "--checkpoint",
                str(training_output / "checkpoints" / "step_000001.pt"),
                "--output-dir",
                str(tmp_path / "invalid_typed_oracle"),
                "--field",
                "typed_appearance",
                "--steps",
                "1",
            ]
        )
