import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

SPEC = importlib.util.spec_from_file_location(
    "pilot_preparation",
    Path(__file__).resolve().parents[1] / "scripts/prepare_dynamic_ttt_pilot.py",
)
pilot = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pilot)


def fixture_scene(root, split, scene, count=16):
    directory = root / split / scene
    directory.mkdir(parents=True)
    frames = []
    for index in range(count):
        # Deliberate gaps and reversed serialization exercise real frame ordering.
        frame_id = index * 2
        image = f"{frame_id}.png"
        depth = f"{frame_id}.depth.npy"
        normal = f"{frame_id}.normal.npy"
        Image.fromarray(np.zeros((4, 6, 3), dtype=np.uint8)).save(directory / image)
        np.save(directory / depth, np.ones((4, 6), dtype=np.float32))
        np.save(directory / normal, np.zeros((4, 6, 3), dtype=np.float32))
        frames.append(
            {
                "frame_id": frame_id,
                "rgb": image,
                "depth": depth,
                "normal": normal,
                "c2w": np.eye(4).tolist(),
                "intrinsics": np.eye(3).tolist(),
            }
        )
    payload = {
        "schema_version": 1,
        "scene_id": scene,
        "coordinate_convention": "opencv_c2w",
        "image_size": [4, 6],
        "bounds": [[-1, -1, -1], [1, 1, 1]],
        "frames": frames[::-1],
    }
    path = directory / "manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_real_order_disjoint_views_and_no_query_content_read(tmp_path):
    path = fixture_scene(tmp_path, "train", "scene_a")
    # Query content is intentionally unreadable; existence checks alone should pass.
    (path.parent / "24.depth.npy").write_bytes(b"query must not be opened")
    (path.parent / "24.normal.npy").write_bytes(b"query must not be opened")
    (path.parent / "24.png").write_bytes(b"query must not be opened")
    episode = pilot.make_episode(path, "cam_00", "train")
    assert [x["frame_id"] for x in episode["warmup"]] == [0, 2, 4, 6]
    assert [x["frame_id"] for x in episode["stream"]] == list(range(8, 24, 2))
    assert [x["frame_id"] for x in episode["query"]] == [24, 26, 28, 30]
    assert episode == pilot.make_episode(path, "cam_00", "train")


def test_short_scene_is_not_padded(tmp_path):
    path = fixture_scene(tmp_path, "train", "scene_a", count=15)
    with pytest.raises(ValueError, match="Too few"):
        pilot.make_episode(path, "cam_00", "train")


def test_cross_split_scene_is_rejected(tmp_path):
    fixture_scene(tmp_path, "train", "same_scene")
    fixture_scene(tmp_path, "val", "same_scene")
    partitions = tmp_path / "partitions.csv"
    partitions.write_text(
        "scene_name,protocol_partition,selected_camera\nsame_scene,train,cam_00\n"
    )
    with pytest.raises(ValueError, match="across splits"):
        pilot.build_plan(tmp_path, partitions, scenes_per_split=1)


def test_missing_data_fails_instead_of_selecting_another_scene(tmp_path):
    path = fixture_scene(tmp_path, "train", "scene_a")
    (path.parent / "0.depth.npy").unlink()
    with pytest.raises(FileNotFoundError):
        pilot.make_episode(path, "cam_00", "train")
