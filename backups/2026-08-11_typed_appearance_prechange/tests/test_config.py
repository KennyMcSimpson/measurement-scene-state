from pathlib import Path

import pytest

from mcss.config import load_config


def test_config_loads_strict_nested_sections(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        """
seed: 7
device: cpu
dataset:
  name: synthetic
  root: null
  image_size: [8, 8]
  context_views: 2
  target_views: 1
  length: 4
model:
  mode: fixed
  voxel_resolution: 8
  image_feature_dim: 8
  state_feature_dim: 8
  refinement_blocks: 1
  n_samples: 8
  ray_chunk_size: 128
training:
  output_dir: outputs/test
  batch_size: 1
  learning_rate: 0.001
  max_steps: 1
  num_workers: 0
  amp: false
  amp_dtype: bfloat16
  log_every: 1
  checkpoint_every: 1
  measurements: [rgb, depth, visibility]
""".strip(),
        encoding="utf-8",
    )

    config = load_config(path)

    assert config.dataset.image_size == (8, 8)
    assert config.model.mode == "fixed"
    assert config.model.state_architecture == "legacy"
    assert config.training.evidence_residual_weight == 0.0
    assert config.training.max_steps == 1


def test_config_rejects_unknown_keys(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("seed: 1\ndevice: cpu\nunknown: true\n", encoding="utf-8")

    with pytest.raises(ValueError, match="unknown"):
        load_config(path)


def test_config_parses_context_local_metric_protocol_and_anisotropic_grid(
    tmp_path: Path,
) -> None:
    path = tmp_path / "a_plus.yaml"
    path.write_text(
        """
seed: 17
device: cpu
dataset:
  name: manifest
  root: data/prepared/train
  image_size: [8, 8]
  context_views: 2
  target_views: 1
  length: 4
  spatial_protocol: context_local_metric
  local_bounds_m: [[-4.0, -3.0, 0.1], [4.0, 3.0, 8.1]]
model:
  mode: fixed
  state_architecture: evidence_residual
  voxel_resolution: [12, 8, 10]
  image_feature_dim: 8
  state_feature_dim: 8
  refinement_blocks: 1
  evidence_temperature: 0.07
  observed_residual_floor: 0.15
  completion_residual_scale: 1.5
  n_samples: 8
  ray_chunk_size: 128
training:
  output_dir: outputs/a_plus
  batch_size: 1
  learning_rate: 0.001
  max_steps: 10
  num_workers: 0
  amp: false
  amp_dtype: bfloat16
  log_every: 1
  checkpoint_every: 5
  measurements: [rgb, depth, visibility]
  lr_schedule: warmup_cosine
  warmup_steps: 2
  min_lr_ratio: 0.1
  ema_decay: 0.9
  evidence_residual_weight: 0.002
""".strip(),
        encoding="utf-8",
    )

    config = load_config(path)

    assert config.dataset.spatial_protocol == "context_local_metric"
    assert config.dataset.local_bounds_m == ((-4.0, -3.0, 0.1), (4.0, 3.0, 8.1))
    assert config.model.voxel_resolution == (12, 8, 10)
    assert config.model.state_architecture == "evidence_residual"
    assert config.model.evidence_temperature == 0.07
    assert config.model.observed_residual_floor == 0.15
    assert config.model.completion_residual_scale == 1.5
    assert config.training.lr_schedule == "warmup_cosine"
    assert config.training.warmup_steps == 2
    assert config.training.min_lr_ratio == 0.1
    assert config.training.ema_decay == 0.9
    assert config.training.amp_dtype == "bfloat16"
    assert config.training.evidence_residual_weight == 0.002
