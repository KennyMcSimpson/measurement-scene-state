import hashlib
import json
from pathlib import Path

import pytest
import yaml

from mcss.cli import main
from mcss.phase0_runner import main as phase0_main
from mcss.phase0_runner import (
    pilot_main,
    run_renderer_sweep,
    run_surface_audit,
    select_audit_windows,
    select_pilot_scenes,
    write_pilot_artifacts,
)


def _trained_evidence_case(tmp_path: Path) -> tuple[Path, Path]:
    output_dir = tmp_path / "training"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"""
seed: 3
device: cpu
dataset:
  name: synthetic
  root: null
  image_size: [8, 8]
  context_views: 2
  target_views: 1
  length: 2
model:
  mode: fixed
  state_architecture: evidence_residual
  voxel_resolution: 8
  image_feature_dim: 8
  state_feature_dim: 8
  refinement_blocks: 1
  evidence_temperature: 0.1
  observed_residual_floor: 0.1
  completion_residual_scale: 1.5
  n_samples: 8
  ray_chunk_size: 128
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
    assert main(["train", "--config", str(config_path)]) == 0
    return config_path, output_dir / "checkpoints" / "step_000001.pt"


def test_renderer_sweep_writes_unique_provenance_and_refuses_overwrite(
    tmp_path: Path,
) -> None:
    config_path, checkpoint = _trained_evidence_case(tmp_path)
    output_root = tmp_path / "renderer_sweep"

    report = run_renderer_sweep(
        config_path,
        checkpoint,
        output_root,
        sample_counts=(8, 12),
        ray_chunk_size=16,
    )

    assert [item["n_samples"] for item in report["runs"]] == [8, 12]
    for count in (8, 12):
        child = output_root / f"samples_{count:04d}" / "evaluation_report.json"
        assert child.is_file()
        resolved = json.loads(child.read_text(encoding="utf-8"))["resolved_config"]
        assert resolved["model"]["n_samples"] == count
        assert resolved["model"]["ray_chunk_size"] == 16
    assert report["source"]["checkpoint_sha256"] == hashlib.sha256(
        checkpoint.read_bytes()
    ).hexdigest()

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        run_renderer_sweep(
            config_path,
            checkpoint,
            output_root,
            sample_counts=(8, 12),
            ray_chunk_size=16,
        )


def test_surface_audit_records_post_forward_scene_statistics(tmp_path: Path) -> None:
    config_path, checkpoint = _trained_evidence_case(tmp_path)
    output_dir = tmp_path / "surface_audit"

    report = run_surface_audit(
        config_path,
        checkpoint,
        output_dir,
        n_samples=32,
    )

    assert report["schema_version"] == "mcss.phase0_surface_audit.v1"
    assert report["source"]["config_sha256"] == hashlib.sha256(
        config_path.read_bytes()
    ).hexdigest()
    assert report["source"]["checkpoint_sha256"] == hashlib.sha256(
        checkpoint.read_bytes()
    ).hexdigest()
    assert report["evaluation"]["sample_count"] == 2
    assert report["evaluation"]["scene_count"] == 2
    assert len(report["per_scene"]) == 2
    assert "surface_vs_free" in report["scene_macro"]
    first_scene = next(iter(report["per_scene"].values()))
    assert first_scene["evidence_field"] == "evidence.confidence"
    assert first_scene["label_semantics"].startswith("target-visible surface-near")
    assert first_scene["surface_band_m"] > 0.0
    assert set(first_scene["bins"]) == {"free", "surface", "behind"}
    assert first_scene["bins"]["surface"]["count"] > 0
    assert len(first_scene["windows"]) == 1
    assert first_scene["error_by_risk"]["count"] > 0
    assert len(first_scene["error_by_risk"]["deciles"]) == 10
    assert "error_by_risk" in report["scene_macro"]
    for comparison in ("surface_vs_free", "surface_vs_behind"):
        assert first_scene["comparisons"][comparison]["positive_count"] > 0
        assert len(first_scene["comparisons"][comparison]["reliability"]) == 10
    assert (output_dir / "evidence_audit.json").is_file()

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        run_surface_audit(config_path, checkpoint, output_dir, n_samples=32)


def test_surface_audit_window_selection_is_label_blind_stable_and_balanced() -> None:
    candidates = [
        ("scene_a", start) for start in range(8)
    ] + [
        ("scene_b", start) for start in range(6)
    ]

    selected = select_audit_windows(candidates, windows_per_scene=3)
    repeated = select_audit_windows(candidates, windows_per_scene=3)

    assert selected == repeated
    assert len(selected) == 6
    assert sum(candidates[index][0] == "scene_a" for index in selected) == 3
    assert sum(candidates[index][0] == "scene_b" for index in selected) == 3


def test_phase0_cli_writes_top_level_action_report(tmp_path: Path) -> None:
    config_path, checkpoint = _trained_evidence_case(tmp_path)
    output_root = tmp_path / "phase0"

    assert (
        phase0_main(
            [
                "--config",
                str(config_path),
                "--checkpoint",
                str(checkpoint),
                "--output-root",
                str(output_root),
                "--sample-counts",
                "8",
                "12",
                "--ray-chunk-size",
                "16",
            ]
        )
        == 0
    )
    report = json.loads((output_root / "phase0_report.json").read_text(encoding="utf-8"))
    assert report["schema_version"] == "mcss.phase0.v1"
    assert report["actions"] == ["renderer_sweep"]
    assert (output_root / "renderer" / "renderer_sweep.json").is_file()


def test_pilot_selection_is_label_blind_and_order_independent(tmp_path: Path) -> None:
    for scene_id in ("scene_c", "scene_a", "scene_b"):
        scene = tmp_path / scene_id
        scene.mkdir()
        (scene / "manifest.json").write_text(
            json.dumps({"scene_id": scene_id, "labels_deliberately_omitted": True}),
            encoding="utf-8",
        )

    first = select_pilot_scenes(tmp_path, split="train", count=2)
    second = select_pilot_scenes(tmp_path, split="train", count=2)
    expected = tuple(
        sorted(
            ("scene_a", "scene_b", "scene_c"),
            key=lambda value: hashlib.sha256(
                f"v6-pilot:train:{value}".encode()
            ).hexdigest(),
        )[:2]
    )

    assert first == second == expected
    with pytest.raises(ValueError, match="available"):
        select_pilot_scenes(tmp_path, split="train", count=4)


def _write_pilot_base_config(path: Path, root: Path, output_dir: Path, max_steps: int) -> None:
    path.write_text(
        f"""
seed: 17
device: cpu
dataset:
  name: manifest
  root: {root.as_posix()}
  image_size: [8, 8]
  context_views: 2
  target_views: 1
  length: 99
  sample_stride: 2
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
  max_steps: {max_steps}
  num_workers: 0
  amp: false
  log_every: 1
  checkpoint_every: 1
  measurements: [rgb]
""".strip(),
        encoding="utf-8",
    )


def test_pilot_artifacts_are_immutable_and_only_replace_protocol_fields(tmp_path: Path) -> None:
    train_root = tmp_path / "train"
    validation_root = tmp_path / "validation"
    for root, scene_ids in (
        (train_root, ("train_c", "train_a", "train_b")),
        (validation_root, ("val_b", "val_a", "val_c")),
    ):
        for scene_id in scene_ids:
            scene = root / scene_id
            scene.mkdir(parents=True)
            (scene / "manifest.json").write_text(
                json.dumps({"scene_id": scene_id}), encoding="utf-8"
            )
    train_base = tmp_path / "train_base.yaml"
    validation_base = tmp_path / "validation_base.yaml"
    _write_pilot_base_config(train_base, train_root, tmp_path / "old_train", 20)
    _write_pilot_base_config(validation_base, validation_root, tmp_path / "old_val", 1)
    partition_registry = tmp_path / "partitions.csv"
    partition_registry.write_text(
        "scene_name,protocol_partition\n"
        "train_a,train\ntrain_b,train\ntrain_c,train\n"
        "val_a,val\nval_b,val\nval_c,val\n",
        encoding="utf-8",
    )
    selection_path = tmp_path / "pilot" / "pilot_selection.json"
    train_output_config = tmp_path / "configs" / "pilot_train.yaml"
    validation_output_config = tmp_path / "configs" / "pilot_validation.yaml"

    report = write_pilot_artifacts(
        train_base,
        validation_base,
        selection_path=selection_path,
        train_config_output=train_output_config,
        validation_config_output=validation_output_config,
        train_output_dir=tmp_path / "new_train",
        validation_output_dir=tmp_path / "new_validation",
        train_count=2,
        validation_count=2,
        train_max_steps=10_000,
        partition_registry=partition_registry,
    )

    assert report["splits"]["train"]["candidate_count"] == 3
    assert report["splits"]["validation"]["candidate_count"] == 3
    assert report["splits"]["train"]["split_token"] == "train"
    assert report["splits"]["validation"]["split_token"] == "val"
    assert report["partition_audit"]["passed"] is True
    assert report["partition_audit"]["registry_sha256"] == hashlib.sha256(
        partition_registry.read_bytes()
    ).hexdigest()
    expected_validation = sorted(
        ("val_a", "val_b", "val_c"),
        key=lambda scene_id: hashlib.sha256(
            f"v6-pilot:val:{scene_id}".encode()
        ).hexdigest(),
    )[:2]
    assert report["splits"]["validation"]["selected_scene_ids"] == expected_validation
    assert selection_path.is_file()
    train_generated = yaml.safe_load(train_output_config.read_text(encoding="utf-8"))
    validation_generated = yaml.safe_load(
        validation_output_config.read_text(encoding="utf-8")
    )
    train_original = yaml.safe_load(train_base.read_text(encoding="utf-8"))
    validation_original = yaml.safe_load(validation_base.read_text(encoding="utf-8"))
    assert train_generated["model"] == train_original["model"]
    assert validation_generated["model"] == validation_original["model"]
    assert train_generated["dataset"] | {"root": train_original["dataset"]["root"]} == (
        train_original["dataset"]
        | {
            "scene_ids": train_generated["dataset"]["scene_ids"],
            "length": 2,
        }
    )
    assert train_generated["training"] | {
        "output_dir": train_original["training"]["output_dir"],
        "max_steps": train_original["training"]["max_steps"],
    } == train_original["training"]
    assert validation_generated["dataset"]["scene_ids"] == report["splits"][
        "validation"
    ]["selected_scene_ids"]
    assert validation_generated["dataset"]["length"] == 2
    assert validation_generated["training"]["max_steps"] == 1

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_pilot_artifacts(
            train_base,
            validation_base,
            selection_path=selection_path,
            train_config_output=train_output_config,
            validation_config_output=validation_output_config,
            train_output_dir=tmp_path / "new_train",
            validation_output_dir=tmp_path / "new_validation",
            train_count=2,
            validation_count=2,
            train_max_steps=10_000,
            partition_registry=partition_registry,
        )


def test_pilot_artifacts_reject_partition_mismatch_before_writing(tmp_path: Path) -> None:
    train_root = tmp_path / "guard_train"
    validation_root = tmp_path / "guard_val"
    for root, scene_id in ((train_root, "train_a"), (validation_root, "val_a")):
        scene = root / scene_id
        scene.mkdir(parents=True)
        (scene / "manifest.json").write_text(
            json.dumps({"scene_id": scene_id}), encoding="utf-8"
        )
    train_base = tmp_path / "guard_train.yaml"
    validation_base = tmp_path / "guard_val.yaml"
    _write_pilot_base_config(train_base, train_root, tmp_path / "old_guard_train", 2)
    _write_pilot_base_config(validation_base, validation_root, tmp_path / "old_guard_val", 1)
    registry = tmp_path / "guard_partitions.csv"
    registry.write_text(
        "scene_name,protocol_partition\ntrain_a,final_holdout\nval_a,val\n",
        encoding="utf-8",
    )
    selection = tmp_path / "guard_selection.json"

    with pytest.raises(ValueError, match="partition mismatch"):
        write_pilot_artifacts(
            train_base,
            validation_base,
            selection_path=selection,
            train_config_output=tmp_path / "guard_train_output.yaml",
            validation_config_output=tmp_path / "guard_val_output.yaml",
            train_output_dir=tmp_path / "new_guard_train",
            validation_output_dir=tmp_path / "new_guard_val",
            train_count=1,
            validation_count=1,
            train_max_steps=2,
            partition_registry=registry,
        )
    assert not selection.exists()


def test_pilot_cli_writes_selection_and_both_configs(tmp_path: Path) -> None:
    train_root = tmp_path / "cli_train"
    validation_root = tmp_path / "cli_validation"
    for root, scene_id in ((train_root, "train_a"), (validation_root, "val_a")):
        scene = root / scene_id
        scene.mkdir(parents=True)
        (scene / "manifest.json").write_text(
            json.dumps({"scene_id": scene_id}), encoding="utf-8"
        )
    train_base = tmp_path / "cli_train_base.yaml"
    validation_base = tmp_path / "cli_validation_base.yaml"
    _write_pilot_base_config(train_base, train_root, tmp_path / "old_cli_train", 2)
    _write_pilot_base_config(
        validation_base,
        validation_root,
        tmp_path / "old_cli_validation",
        1,
    )
    selection = tmp_path / "cli_pilot_selection.json"
    train_config = tmp_path / "cli_pilot_train.yaml"
    validation_config = tmp_path / "cli_pilot_validation.yaml"

    assert (
        pilot_main(
            [
                "--train-config",
                str(train_base),
                "--validation-config",
                str(validation_base),
                "--selection-path",
                str(selection),
                "--train-config-output",
                str(train_config),
                "--validation-config-output",
                str(validation_config),
                "--train-output-dir",
                str(tmp_path / "new_cli_train"),
                "--validation-output-dir",
                str(tmp_path / "new_cli_validation"),
                "--train-count",
                "1",
                "--validation-count",
                "1",
                "--train-max-steps",
                "2",
            ]
        )
        == 0
    )
    assert selection.is_file()
    assert train_config.is_file()
    assert validation_config.is_file()
