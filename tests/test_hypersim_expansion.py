import csv
from pathlib import Path

import pytest

from mcss.config import load_config
from mcss.data.hypersim_expansion import build_expansion_plan, select_materialization_scenes


def _write_csv(
    path: Path,
    fieldnames: list[str],
    rows: list[dict[str, object]],
    *,
    encoding: str = "utf-8",
) -> None:
    with path.open("w", newline="", encoding=encoding) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_expansion_plan_preserves_official_splits_and_seals_unseen_test_scenes(
    tmp_path: Path,
) -> None:
    metadata = tmp_path / "metadata.csv"
    subset = tmp_path / "observed.csv"
    metadata_rows: list[dict[str, object]] = []
    for scene, split, cameras in (
        ("ai_001_001", "train", {"cam_00": 3}),
        ("ai_001_002", "train", {"cam_00": 2}),
        ("ai_002_001", "val", {"cam_00": 2}),
        ("ai_003_001", "test", {"cam_00": 2}),
        ("ai_003_002", "test", {"cam_00": 2}),
        ("ai_003_003", "test", {"cam_02": 3, "cam_01": 2}),
    ):
        for camera, frame_count in cameras.items():
            for frame_id in range(frame_count):
                metadata_rows.append(
                    {
                        "scene_name": scene,
                        "camera_name": camera,
                        "frame_id": frame_id,
                        "included_in_public_release": "True",
                        "exclude_reason": "",
                        "split_partition_name": split,
                    }
                )
    metadata_rows.append(
        {
            "scene_name": "ai_999_999",
            "camera_name": "cam_00",
            "frame_id": 0,
            "included_in_public_release": "False",
            "exclude_reason": "not public",
            "split_partition_name": "train",
        }
    )
    _write_csv(
        metadata,
        [
            "scene_name",
            "camera_name",
            "frame_id",
            "included_in_public_release",
            "exclude_reason",
            "split_partition_name",
        ],
        metadata_rows,
    )
    _write_csv(
        subset,
        ["scene_name", "split_partition_name", "frame_count"],
        [
            {
                "scene_name": "ai_001_001",
                "split_partition_name": "train",
                "frame_count": 3,
            },
            {
                "scene_name": "ai_003_001",
                "split_partition_name": "test",
                "frame_count": 2,
            },
        ],
        encoding="utf-8-sig",
    )

    plan = build_expansion_plan(metadata, subset)

    assert plan.counts == {"train": 2, "val": 1, "diagnostic_test": 1, "final_holdout": 2}
    by_scene = {row.scene_name: row for row in plan.scenes}
    assert by_scene["ai_001_001"].protocol_partition == "train"
    assert by_scene["ai_002_001"].protocol_partition == "val"
    assert by_scene["ai_003_001"].protocol_partition == "diagnostic_test"
    assert by_scene["ai_003_002"].protocol_partition == "final_holdout"
    assert by_scene["ai_003_003"].selected_camera == "cam_02"
    assert by_scene["ai_003_003"].frame_count == 3
    assert {row.scene_name for row in plan.exclusions} == {"ai_003_001"}
    assert not ({row.scene_name for row in plan.exclusions} & plan.final_holdout_scenes)


def test_expansion_plan_rejects_subset_split_drift(tmp_path: Path) -> None:
    metadata = tmp_path / "metadata.csv"
    subset = tmp_path / "observed.csv"
    _write_csv(
        metadata,
        [
            "scene_name",
            "camera_name",
            "frame_id",
            "included_in_public_release",
            "exclude_reason",
            "split_partition_name",
        ],
        [
            {
                "scene_name": "ai_001_001",
                "camera_name": "cam_00",
                "frame_id": 0,
                "included_in_public_release": "True",
                "exclude_reason": "",
                "split_partition_name": "train",
            }
        ],
    )
    _write_csv(
        subset,
        ["scene_name", "split_partition_name", "frame_count"],
        [
            {
                "scene_name": "ai_001_001",
                "split_partition_name": "test",
                "frame_count": 1,
            }
        ],
    )

    with pytest.raises(ValueError, match="split mismatch"):
        build_expansion_plan(metadata, subset)


def test_evidence_residual_configs_are_protocol_separated() -> None:
    root = Path(__file__).parents[1]
    train = load_config(root / "configs/hypersim_er_train.yaml")
    validation = load_config(root / "configs/hypersim_er_val.yaml")
    diagnostic = load_config(root / "configs/hypersim_er_diagnostic_test.yaml")
    final_holdout = load_config(root / "configs/hypersim_er_final_holdout.yaml")

    assert train.dataset.root == "data/hypersim_er_prepared/train"
    assert validation.dataset.root == "data/hypersim_er_prepared/val"
    assert diagnostic.dataset.root == "data/hypersim_er_prepared/diagnostic_test"
    assert final_holdout.dataset.root == "data/hypersim_er_prepared/final_holdout"
    assert train.model.state_architecture == "evidence_residual"
    assert train.dataset.context_views == 4
    assert train.dataset.target_views == 2
    assert train.dataset.sample_stride == 2
    assert train.training.max_steps == 150_000
    assert train.training.gradient_accumulation == 2
    assert validation.model == train.model
    assert diagnostic.model == train.model
    assert final_holdout.model == train.model


def test_v5_appearance_configs_preserve_data_protocol_and_isolate_outputs() -> None:
    root = Path(__file__).parents[1]
    train = load_config(root / "configs/hypersim_er_v5_appearance2x_train.yaml")
    validation = load_config(root / "configs/hypersim_er_v5_appearance2x_val.yaml")
    overfit = load_config(root / "configs/hypersim_er_v5_appearance2x_overfit.yaml")
    probe = load_config(root / "configs/hypersim_er_v5_appearance2x_probe.yaml")

    assert train.dataset.root == "data/hypersim_er_prepared/train"
    assert validation.dataset.root == "data/hypersim_er_prepared/val"
    assert train.model == validation.model
    assert train.model.appearance_resolution_scale == 2
    assert train.model.voxel_resolution == (48, 32, 48)
    assert train.training.max_steps == 150_000
    assert overfit.dataset.length == 1
    assert overfit.training.max_steps == 5_000
    assert probe.training.max_steps == 200
    assert all(
        "hypersim_er_v5_appearance2x" in config.training.output_dir
        for config in (train, validation, overfit, probe)
    )


def test_materialization_requires_explicit_final_holdout_release(tmp_path: Path) -> None:
    metadata = tmp_path / "metadata.csv"
    subset = tmp_path / "observed.csv"
    _write_csv(
        metadata,
        [
            "scene_name",
            "camera_name",
            "frame_id",
            "included_in_public_release",
            "exclude_reason",
            "split_partition_name",
        ],
        [
            {
                "scene_name": "ai_001_001",
                "camera_name": "cam_00",
                "frame_id": 0,
                "included_in_public_release": "True",
                "exclude_reason": "",
                "split_partition_name": "train",
            },
            {
                "scene_name": "ai_002_001",
                "camera_name": "cam_00",
                "frame_id": 0,
                "included_in_public_release": "True",
                "exclude_reason": "",
                "split_partition_name": "test",
            },
        ],
    )
    _write_csv(subset, ["scene_name", "split_partition_name"], [])
    plan = build_expansion_plan(metadata, subset)

    selected = select_materialization_scenes(plan, ("train",))
    assert [scene.scene_name for scene in selected] == ["ai_001_001"]
    with pytest.raises(ValueError, match="sealed"):
        select_materialization_scenes(plan, ("final_holdout",))
    released = select_materialization_scenes(
        plan, ("final_holdout",), allow_final_holdout=True
    )
    assert [scene.scene_name for scene in released] == ["ai_002_001"]
