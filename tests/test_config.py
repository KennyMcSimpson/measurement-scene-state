from pathlib import Path

import pytest

from mcss.config import load_config
from mcss.model.system import ModelConfig


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
  scene_ids: [scene_b, scene_a]
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
    assert config.dataset.scene_ids == ("scene_b", "scene_a")
    assert config.model.mode == "fixed"
    assert config.model.state_architecture == "legacy"
    assert config.model.appearance_resolution_scale == 1
    assert config.training.evidence_residual_weight == 0.0
    assert config.training.context_subset_geometry_weight == 0.0
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
  surface_peak_temperature: 0.06
  appearance_resolution_scale: 2
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
    assert config.model.surface_peak_temperature == 0.06
    assert config.model.appearance_resolution_scale == 2
    assert config.training.lr_schedule == "warmup_cosine"
    assert config.training.warmup_steps == 2
    assert config.training.min_lr_ratio == 0.1
    assert config.training.ema_decay == 0.9
    assert config.training.amp_dtype == "bfloat16"
    assert config.training.evidence_residual_weight == 0.002


def test_model_config_rejects_highres_appearance_for_legacy_and_unknown_scale() -> None:
    with pytest.raises(ValueError, match="evidence_residual"):
        ModelConfig(state_architecture="legacy", appearance_resolution_scale=2)
    with pytest.raises(ValueError, match="appearance_resolution_scale"):
        ModelConfig(state_architecture="evidence_residual", appearance_resolution_scale=3)


def test_model_config_accepts_dual_evidence_and_rejects_nonpositive_peak_temperature() -> None:
    config = ModelConfig(
        state_architecture="dual_evidence",
        appearance_resolution_scale=2,
        surface_peak_temperature=0.05,
    )

    assert config.state_architecture == "dual_evidence"
    assert config.surface_peak_temperature == 0.05
    with pytest.raises(ValueError, match="surface_peak_temperature"):
        ModelConfig(state_architecture="dual_evidence", surface_peak_temperature=0.0)


def test_model_config_accepts_versioned_transport_architecture() -> None:
    config = ModelConfig(
        state_architecture="dual_evidence_transport",
        appearance_resolution_scale=2,
        surface_peak_temperature=0.05,
    )

    assert config.state_architecture == "dual_evidence_transport"


@pytest.mark.parametrize(
    ("scene_ids", "message"),
    [
        ("[]", "non-empty"),
        ("[scene_a, scene_a]", "unique"),
        ("[scene_a, 3]", "strings"),
    ],
)
def test_config_rejects_invalid_scene_id_allowlists(
    tmp_path: Path,
    scene_ids: str,
    message: str,
) -> None:
    path = tmp_path / "invalid_scene_ids.yaml"
    path.write_text(
        f"""
seed: 7
device: cpu
dataset:
  name: manifest
  root: data/prepared/train
  image_size: [8, 8]
  context_views: 2
  target_views: 1
  length: 4
  scene_ids: {scene_ids}
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
  log_every: 1
  checkpoint_every: 1
  measurements: [rgb]
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=message):
        load_config(path)


def _write_context_subset_config(
    path: Path,
    *,
    weight: float | str,
    context_views: int = 4,
    state_architecture: str = "evidence_residual",
    mode: str = "fixed",
    gradient_accumulation: int = 1,
    checkpoint_every: int = 4,
) -> None:
    path.write_text(
        f"""
seed: 17
device: cpu
dataset:
  name: synthetic
  root: null
  image_size: [8, 8]
  context_views: {context_views}
  target_views: 1
  length: 4
model:
  mode: {mode}
  state_architecture: {state_architecture}
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
  output_dir: outputs/context_subset
  batch_size: 1
  learning_rate: 0.001
  max_steps: 4
  num_workers: 0
  amp: false
  log_every: 1
  checkpoint_every: {checkpoint_every}
  measurements: [rgb, depth, visibility]
  lr_schedule: warmup_cosine
  warmup_steps: 2
  gradient_accumulation: {gradient_accumulation}
  context_subset_geometry_weight: {weight}
""".strip(),
        encoding="utf-8",
    )


def test_config_parses_opt_in_context_subset_geometry_weight(tmp_path: Path) -> None:
    path = tmp_path / "context_subset.yaml"
    _write_context_subset_config(path, weight=0.1)

    config = load_config(path)

    assert config.training.context_subset_geometry_weight == 0.1


def test_config_rejects_negative_context_subset_geometry_weight(tmp_path: Path) -> None:
    path = tmp_path / "context_subset_negative.yaml"
    _write_context_subset_config(path, weight=-0.1)

    with pytest.raises(ValueError, match="context_subset_geometry_weight must be non-negative"):
        load_config(path)


@pytest.mark.parametrize("weight", [".nan", ".inf", "true"])
def test_config_rejects_non_finite_or_boolean_context_subset_geometry_weight(
    tmp_path: Path,
    weight: str,
) -> None:
    path = tmp_path / "context_subset_non_finite.yaml"
    _write_context_subset_config(path, weight=weight)

    with pytest.raises(ValueError, match="finite non-negative number"):
        load_config(path)


def test_config_requires_context_subset_checkpoints_on_accumulation_boundaries(
    tmp_path: Path,
) -> None:
    path = tmp_path / "context_subset_misaligned_checkpoint.yaml"
    _write_context_subset_config(
        path,
        weight=0.1,
        gradient_accumulation=2,
        checkpoint_every=3,
    )

    with pytest.raises(ValueError, match="checkpoint_every.*gradient_accumulation"):
        load_config(path)


@pytest.mark.parametrize(
    ("context_views", "state_architecture", "mode", "message"),
    [
        (3, "evidence_residual", "fixed", "exactly four context views"),
        (4, "legacy", "fixed", "state_architecture: evidence_residual"),
        (4, "evidence_residual", "learned_heads", "model.mode: fixed"),
    ],
)
def test_config_restricts_context_subset_geometry_to_unchanged_v5(
    tmp_path: Path,
    context_views: int,
    state_architecture: str,
    mode: str,
    message: str,
) -> None:
    path = tmp_path / "context_subset_invalid_scope.yaml"
    _write_context_subset_config(
        path,
        weight=0.1,
        context_views=context_views,
        state_architecture=state_architecture,
        mode=mode,
    )

    with pytest.raises(ValueError, match=message):
        load_config(path)
