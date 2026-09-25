"""Tests for metadata-only full train/dev episode compilation."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location(
    "compile_dynamic_training_episodes",
    Path(__file__).resolve().parents[1] / "scripts/compile_dynamic_training_episodes.py",
)
assert SPEC is not None and SPEC.loader is not None
compiler = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = compiler
SPEC.loader.exec_module(compiler)


def _write_scene(
    root: Path,
    split: str,
    scene: str,
    camera: str,
    count: int = 16,
    invalid_camera_frames: tuple[int, ...] = (),
) -> None:
    scene_root = root / split / scene
    (scene_root / "rgb").mkdir(parents=True)
    (scene_root / "depth").mkdir()
    frames = []
    for frame_id in range(count):
        (scene_root / "rgb" / f"{frame_id:06d}.png").write_bytes(b"not an image")
        (scene_root / "depth" / f"{frame_id:06d}.npy").write_bytes(b"not an array")
        c2w = np.eye(4).tolist()
        if frame_id in invalid_camera_frames:
            if frame_id % 2 == 0:
                c2w[0][0] = 1.01
            else:
                c2w[0][0] = -1.0
        frames.append(
            {
                "frame_id": frame_id,
                "rgb": f"rgb/{frame_id:06d}.png",
                "depth": f"depth/{frame_id:06d}.npy",
                "normal": f"normal/{frame_id:06d}.npy",
                "intrinsics": [[2.0, 0.0, 1.5], [0.0, 2.0, 1.5], [0.0, 0.0, 1.0]],
                "c2w": c2w,
            }
        )
    payload = {
        "schema_version": 1,
        "scene_id": scene,
        "coordinate_convention": "opencv_c2w",
        "image_size": [4, 6],
        "bounds": [[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]],
        "frames": frames[::-1],
    }
    (scene_root / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")


def _write_partitions(path: Path, rows: list[tuple[str, str, str, int]]) -> None:
    lines = [
        "scene_name,official_split,protocol_partition,selected_camera,frame_count",
        *(f"{scene},{split},{split},{camera},{count}" for scene, split, camera, count in rows),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_compiles_fixed_segments_and_does_not_decode_rgb_or_depth(tmp_path: Path) -> None:
    prepared = tmp_path / "prepared"
    _write_scene(prepared, "train", "scene_train", "cam_01")
    _write_scene(prepared, "val", "scene_val", "cam_00")
    partitions = tmp_path / "partitions.csv"
    _write_partitions(
        partitions,
        [("scene_train", "train", "cam_01", 16), ("scene_val", "val", "cam_00", 16)],
    )
    inventory = compiler.compile_training_episodes(
        prepared_root=prepared,
        partitions_path=partitions,
        output_root=tmp_path / "output",
        expected_counts=None,
        train_depth_health_scan=False,
    )
    assert inventory["splits"]["train"]["eligible_episode_count"] == 1
    assert inventory["splits"]["dev"]["eligible_episode_count"] == 1
    train_index = json.loads(
        (tmp_path / "output/episodes/episode_index_train.json").read_text(encoding="utf-8")
    )
    train_online = json.loads(
        Path(train_index["episodes"][0]["online_manifest"]).read_text(encoding="utf-8")
    )
    train_query = json.loads(
        Path(train_index["episodes"][0]["query_manifest"]).read_text(encoding="utf-8")
    )
    assert train_online["episode_id"] == "train--scene_train--cam_01"
    assert [frame["frame_id"] for frame in train_online["warmup"]] == [0, 1, 2, 3]
    assert [frame["frame_id"] for frame in train_online["stream"]] == list(range(4, 12))
    assert [frame["frame_id"] for frame in train_query["query"]] == [12, 13, 14, 15]
    assert all("depth" not in frame for frame in train_online["warmup"] + train_online["stream"])
    assert all("normal" not in frame for frame in train_query["query"])


def test_short_scene_fails_by_default_and_is_explicit_when_allowed(tmp_path: Path) -> None:
    prepared = tmp_path / "prepared"
    _write_scene(prepared, "train", "scene_short", "cam_00", count=9)
    (prepared / "val").mkdir(parents=True)
    partitions = tmp_path / "partitions.csv"
    _write_partitions(partitions, [("scene_short", "train", "cam_00", 9)])
    with pytest.raises(ValueError, match="ineligible scenes"):
        compiler.compile_training_episodes(
            prepared_root=prepared,
            partitions_path=partitions,
            output_root=tmp_path / "strict",
            expected_counts=None,
            train_depth_health_scan=False,
        )
    inventory = compiler.compile_training_episodes(
        prepared_root=prepared,
        partitions_path=partitions,
        output_root=tmp_path / "allowed",
        expected_counts=None,
        allow_ineligible=True,
        train_depth_health_scan=False,
    )
    assert inventory["ineligible_scene_ids"] == ["scene_short"]
    assert inventory["splits"]["train"]["eligible_episode_count"] == 0


def test_missing_selected_reference_fails(tmp_path: Path) -> None:
    prepared = tmp_path / "prepared"
    _write_scene(prepared, "train", "scene_train", "cam_00")
    (prepared / "val").mkdir(parents=True)
    (prepared / "train/scene_train/depth/000015.npy").unlink()
    partitions = tmp_path / "partitions.csv"
    _write_partitions(partitions, [("scene_train", "train", "cam_00", 16)])
    with pytest.raises(FileNotFoundError, match="missing .*depth reference"):
        compiler.compile_training_episodes(
            prepared_root=prepared,
            partitions_path=partitions,
            output_root=tmp_path / "output",
            expected_counts=None,
            train_depth_health_scan=False,
        )


def test_train_val_scene_intersection_fails(tmp_path: Path) -> None:
    prepared = tmp_path / "prepared"
    _write_scene(prepared, "train", "same_scene", "cam_00")
    _write_scene(prepared, "val", "same_scene", "cam_00")
    partitions = tmp_path / "partitions.csv"
    _write_partitions(
        partitions,
        [("same_scene", "train", "cam_00", 16), ("same_scene", "val", "cam_00", 16)],
    )
    with pytest.raises(ValueError, match="duplicate train/val partition row"):
        compiler.compile_training_episodes(
            prepared_root=prepared,
            partitions_path=partitions,
            output_root=tmp_path / "output",
            expected_counts=None,
            train_depth_health_scan=False,
        )


def test_compiles_multiple_chronological_windows_for_one_train_scene(tmp_path: Path) -> None:
    prepared = tmp_path / "prepared"
    _write_scene(prepared, "train", "scene_train", "cam_00", count=32)
    _write_scene(prepared, "val", "scene_val", "cam_00", count=16)
    partitions = tmp_path / "partitions.csv"
    _write_partitions(
        partitions,
        [
            ("scene_train", "train", "cam_00", 32),
            ("scene_val", "val", "cam_00", 16),
        ],
    )
    inventory = compiler.compile_training_episodes(
        prepared_root=prepared,
        partitions_path=partitions,
        output_root=tmp_path / "output",
        expected_counts=None,
        train_depth_health_scan=False,
        train_window_stride=16,
        max_train_windows_per_scene=None,
    )
    assert inventory["splits"]["train"]["eligible_episode_count"] == 2
    train_index = json.loads(
        (tmp_path / "output/episodes/episode_index_train.json").read_text(encoding="utf-8")
    )
    assert len(train_index["episodes"]) == 2
    assert len({entry["episode_id"] for entry in train_index["episodes"]}) == 2
    windows = []
    for entry in train_index["episodes"]:
        online = json.loads(Path(entry["online_manifest"]).read_text(encoding="utf-8"))
        query = json.loads(Path(entry["query_manifest"]).read_text(encoding="utf-8"))
        observed = [frame["frame_id"] for frame in online["warmup"] + online["stream"]]
        queries = [frame["frame_id"] for frame in query["query"]]
        assert max(observed) < min(queries)
        assert set(observed).isdisjoint(queries)
        windows.append(observed + queries)
    assert windows == [list(range(16)), list(range(16, 32))]
    dev_index = json.loads(
        (tmp_path / "output/episodes/episode_index_dev.json").read_text(encoding="utf-8")
    )
    assert len(dev_index["episodes"]) == 1


def test_rigid_camera_filter_excludes_metadata_invalid_frames_and_records_hashes(
    tmp_path: Path,
) -> None:
    prepared = tmp_path / "prepared"
    _write_scene(
        prepared,
        "train",
        "scene_train",
        "cam_00",
        count=18,
        invalid_camera_frames=(0, 1),
    )
    _write_scene(prepared, "val", "scene_val", "cam_00", count=16)
    partitions = tmp_path / "partitions.csv"
    _write_partitions(
        partitions,
        [
            ("scene_train", "train", "cam_00", 18),
            ("scene_val", "val", "cam_00", 16),
        ],
    )

    inventory = compiler.compile_training_episodes(
        prepared_root=prepared,
        partitions_path=partitions,
        output_root=tmp_path / "output",
        expected_counts=None,
        train_depth_health_scan=False,
        rigid_cameras_only=True,
    )

    manifest_path = prepared / "train/scene_train/manifest.json"
    assert inventory["rigid_cameras_only"] is True
    assert inventory["rigid_camera_tolerance"] == pytest.approx(1e-3)
    assert inventory["rigid_camera_exclusions"] == [
        {
            "scene_id": "scene_train",
            "source_split": "train",
            "split_id": "train",
            "manifest": manifest_path.resolve().as_posix(),
            "source_manifest_sha256": compiler.sha256(manifest_path),
            "original_frame_count": 18,
            "excluded_frame_ids": [0, 1],
            "excluded_frames": [
                {"frame_id": 0, "reason": "rotation_orthogonality"},
                {"frame_id": 1, "reason": "rotation_determinant"},
            ],
        }
    ]
    train_index = json.loads(
        (tmp_path / "output/episodes/episode_index_train.json").read_text(encoding="utf-8")
    )
    online = json.loads(Path(train_index["episodes"][0]["online_manifest"]).read_text())
    query = json.loads(Path(train_index["episodes"][0]["query_manifest"]).read_text())
    assert [frame["frame_id"] for frame in online["warmup"] + online["stream"]] == list(
        range(2, 14)
    )
    assert [frame["frame_id"] for frame in query["query"]] == list(range(14, 18))
    assert all(frame["c2w"] == np.eye(4).tolist() for frame in online["warmup"])

    scene_inventory = json.loads(
        (tmp_path / "output/inventory/scene_inventory_train.json").read_text(encoding="utf-8")
    )["scenes"][0]
    assert scene_inventory["original_frame_count"] == 18
    assert scene_inventory["frame_count"] == 16
    assert scene_inventory["excluded_camera_frames"] == [
        {"frame_id": 0, "reason": "rotation_orthogonality"},
        {"frame_id": 1, "reason": "rotation_determinant"},
    ]


def test_rigid_camera_filter_does_not_silently_drop_short_scene(tmp_path: Path) -> None:
    prepared = tmp_path / "prepared"
    _write_scene(
        prepared,
        "train",
        "scene_train",
        "cam_00",
        count=16,
        invalid_camera_frames=(0,),
    )
    _write_scene(prepared, "val", "scene_val", "cam_00", count=16)
    partitions = tmp_path / "partitions.csv"
    _write_partitions(
        partitions,
        [
            ("scene_train", "train", "cam_00", 16),
            ("scene_val", "val", "cam_00", 16),
        ],
    )

    with pytest.raises(ValueError, match="ineligible scenes.*scene_train"):
        compiler.compile_training_episodes(
            prepared_root=prepared,
            partitions_path=partitions,
            output_root=tmp_path / "output",
            expected_counts=None,
            train_depth_health_scan=False,
            rigid_cameras_only=True,
        )


def test_rigid_camera_filter_checks_partition_count_before_filtering(tmp_path: Path) -> None:
    prepared = tmp_path / "prepared"
    _write_scene(
        prepared,
        "train",
        "scene_train",
        "cam_00",
        count=16,
        invalid_camera_frames=(0,),
    )
    (prepared / "val").mkdir(parents=True)
    partitions = tmp_path / "partitions.csv"
    _write_partitions(partitions, [("scene_train", "train", "cam_00", 15)])

    with pytest.raises(ValueError, match="partition frame_count mismatch"):
        compiler.compile_training_episodes(
            prepared_root=prepared,
            partitions_path=partitions,
            output_root=tmp_path / "output",
            expected_counts=None,
            train_depth_health_scan=False,
            rigid_cameras_only=True,
        )
