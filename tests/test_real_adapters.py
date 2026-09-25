import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from mcss.data.hypersim import convert_hypersim_pose
from mcss.data.manifest import ManifestFrame, SceneManifest, load_manifest, save_manifest
from mcss.data.replica import _resize_scalar, prepare_replica


def test_manifest_rejects_paths_outside_scene_root(tmp_path: Path) -> None:
    data = {
        "schema_version": 1,
        "scene_id": "scene",
        "coordinate_convention": "opencv_c2w",
        "image_size": [4, 4],
        "bounds": [[-1, -1, -1], [1, 1, 1]],
        "frames": [
            {
                "frame_id": 0,
                "rgb": "../escape.png",
                "intrinsics": np.eye(3).tolist(),
                "c2w": np.eye(4).tolist(),
            }
        ],
    }
    with pytest.raises(ValueError, match="relative"):
        SceneManifest.from_dict(data, tmp_path)


def test_manifest_roundtrip_preserves_frame_contract(tmp_path: Path) -> None:
    frame = ManifestFrame(
        frame_id=0,
        rgb="rgb/000000.png",
        intrinsics=np.eye(3).tolist(),
        c2w=np.eye(4).tolist(),
        depth="depth/000000.npy",
    )
    manifest = SceneManifest(
        scene_id="scene",
        image_size=(4, 4),
        bounds=((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0)),
        frames=(frame,),
        root=tmp_path,
    )
    path = save_manifest(manifest, tmp_path / "manifest.json")

    restored = load_manifest(path)

    assert restored.scene_id == "scene"
    assert restored.frames[0].depth == "depth/000000.npy"


def test_replica_preparation_converts_a_small_rendered_fixture(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    (raw / "rgb").mkdir(parents=True)
    (raw / "depth").mkdir()
    Image.new("RGB", (4, 4), (120, 80, 40)).save(raw / "rgb" / "frame000000.jpg")
    Image.fromarray(np.full((4, 4), 2000, dtype=np.uint16)).save(raw / "depth" / "depth000000.png")
    np.savetxt(raw / "traj.txt", np.eye(4).reshape(1, 16))
    (raw / "cam_params.json").write_text(
        json.dumps({"fx": 4.0, "fy": 4.0, "cx": 1.5, "cy": 1.5, "depth_scale": 1000.0}),
        encoding="utf-8",
    )

    manifest_path = prepare_replica(raw, tmp_path / "prepared", frame_limit=1)
    manifest = load_manifest(manifest_path)

    assert manifest.frames[0].depth is not None
    assert (manifest_path.parent / manifest.frames[0].depth).is_file()
    depth = np.load(manifest_path.parent / manifest.frames[0].depth)
    assert depth[0, 0] == 2.0


def test_hypersim_pose_converts_opengl_camera_axes_to_opencv() -> None:
    orientation = np.eye(3, dtype=np.float32)
    position = np.array([1.0, 2.0, 3.0], dtype=np.float32)

    pose = convert_hypersim_pose(orientation, position, meters_per_asset_unit=2.0)

    np.testing.assert_allclose(pose[:3, :3], np.diag([1.0, -1.0, -1.0]))
    np.testing.assert_allclose(pose[:3, 3], position * 2.0)


def test_resized_metric_array_remains_writable() -> None:
    value = _resize_scalar(np.ones((4, 4), dtype=np.float32), (2, 2))
    value[0, 0] = 0.0

    assert value[0, 0] == 0.0
