import json
from pathlib import Path

from mcss.config import load_config
from mcss.engine import Trainer
from mcss.evidence_telemetry import run_evidence_telemetry


def _write_dual_evidence_config(path: Path, output_dir: Path) -> None:
    path.write_text(
        f"""
seed: 17
device: cpu
dataset:
  name: synthetic
  root: null
  image_size: [8, 8]
  context_views: 2
  target_views: 1
  length: 1
model:
  mode: fixed
  state_architecture: dual_evidence
  voxel_resolution: 4
  image_feature_dim: 8
  state_feature_dim: 8
  refinement_blocks: 1
  evidence_temperature: 0.1
  surface_peak_temperature: 0.05
  observed_residual_floor: 0.1
  completion_residual_scale: 1.5
  n_samples: 4
  ray_chunk_size: 64
training:
  output_dir: {output_dir.as_posix()}
  batch_size: 1
  learning_rate: 0.001
  max_steps: 1
  num_workers: 0
  amp: false
  log_every: 1
  checkpoint_every: 1
  measurements: [rgb, depth, normal, point, visibility]
""".strip(),
        encoding="utf-8",
    )


def _write_transport_evidence_config(path: Path, output_dir: Path) -> None:
    path.write_text(
        f"""
seed: 17
device: cpu
dataset:
  name: synthetic
  root: null
  image_size: [8, 8]
  context_views: 3
  target_views: 1
  length: 1
model:
  mode: fixed
  state_architecture: dual_evidence_transport
  voxel_resolution: 4
  image_feature_dim: 8
  state_feature_dim: 8
  refinement_blocks: 1
  evidence_temperature: 0.1
  surface_peak_temperature: 0.05
  observed_residual_floor: 0.1
  completion_residual_scale: 1.5
  n_samples: 4
  ray_chunk_size: 64
training:
  output_dir: {output_dir.as_posix()}
  batch_size: 1
  learning_rate: 0.001
  max_steps: 1
  num_workers: 0
  amp: false
  log_every: 1
  checkpoint_every: 1
  measurements: [rgb, depth, normal, point, visibility]
""".strip(),
        encoding="utf-8",
    )


def test_evidence_telemetry_is_context_only_and_reconstructs_density(tmp_path: Path) -> None:
    config_path = tmp_path / "dual.yaml"
    training_dir = tmp_path / "training"
    report_dir = tmp_path / "telemetry"
    _write_dual_evidence_config(config_path, training_dir)
    trainer = Trainer(load_config(config_path))
    checkpoint = trainer.save_checkpoint()

    report = run_evidence_telemetry(config_path, checkpoint, report_dir)

    persisted = json.loads((report_dir / "evidence_telemetry.json").read_text(encoding="utf-8"))
    assert persisted == report
    assert report["schema_version"] == "mcss.v6_evidence_telemetry.v1"
    assert report["diagnostic_only"] is True
    assert report["target_labels_used"] is False
    assert report["target_camera_role"] == "query_only"
    assert report["candidate_profiles"]["profiles"]["total"] == 2 * 4**3
    assert "appearance_top_decile" in report["candidate_profiles"]["strata"]
    assert report["state"]["density_decomposition"]["reconstruction_max_abs_error"] < 1e-6
    assert report["state"]["surface_peakness_replay_max_abs_error"] < 1e-6


def test_transport_telemetry_replays_fixed_transport_fields(tmp_path: Path) -> None:
    config_path = tmp_path / "transport.yaml"
    training_dir = tmp_path / "training"
    report_dir = tmp_path / "telemetry"
    _write_transport_evidence_config(config_path, training_dir)
    trainer = Trainer(load_config(config_path))
    checkpoint = trainer.save_checkpoint()

    report = run_evidence_telemetry(config_path, checkpoint, report_dir)

    assert report["target_labels_used"] is False
    assert report["architecture"] == "dual_evidence_transport"
    assert report["candidate_profiles"]["profiles"]["total"] == 3 * 4**3
    assert report["state"]["transport_replay_max_abs_error"] < 1e-6
    assert report["state"]["density_decomposition"]["reconstruction_max_abs_error"] < 1e-6
    assert "transport_coverage" in report["state"]
    assert "transported_appearance_confidence" in report["state"]
