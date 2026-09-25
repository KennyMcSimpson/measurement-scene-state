import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from mcss.config import DatasetConfig
from mcss.data.manifest import ManifestFrame, SceneManifest, save_manifest
from mcss.data.manifest_dataset import ManifestSceneDataset, _find_manifests
from mcss.engine import build_dataset
from mcss.geometry import generate_rays


def _write_minimal_manifest(root: Path, scene_id: str) -> None:
    root.mkdir(parents=True)
    frames = tuple(
        ManifestFrame(
            frame_id=index,
            rgb=f"rgb/{index:06d}.png",
            intrinsics=[[2.0, 0.0, 0.5], [0.0, 2.0, 0.5], [0.0, 0.0, 1.0]],
            c2w=np.eye(4).tolist(),
        )
        for index in range(3)
    )
    save_manifest(
        SceneManifest(
            scene_id,
            (2, 2),
            ((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0)),
            frames,
            root,
        ),
        root / "manifest.json",
    )


def test_manifest_dataset_loads_and_resizes_modalities(tmp_path: Path) -> None:
    scene = tmp_path / "scene"
    (scene / "rgb").mkdir(parents=True)
    (scene / "depth").mkdir()
    (scene / "normal").mkdir()
    for index in range(3):
        Image.new("RGB", (4, 4), (20 + index, 30, 40)).save(scene / "rgb" / f"{index:06d}.png")
        np.save(scene / "depth" / f"{index:06d}.npy", np.full((4, 4), 2.0, dtype=np.float32))
        normal = np.zeros((4, 4, 3), dtype=np.float32)
        normal[..., 2] = 1.0
        np.save(scene / "normal" / f"{index:06d}.npy", normal)
    frames = tuple(
        ManifestFrame(
            frame_id=index,
            rgb=f"rgb/{index:06d}.png",
            intrinsics=[[4.0, 0.0, 1.5], [0.0, 4.0, 1.5], [0.0, 0.0, 1.0]],
            c2w=np.eye(4).tolist(),
            depth=f"depth/{index:06d}.npy",
            normal=f"normal/{index:06d}.npy",
        )
        for index in range(3)
    )
    save_manifest(
        SceneManifest("scene", (4, 4), ((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0)), frames, scene),
        scene / "manifest.json",
    )

    dataset = ManifestSceneDataset(scene, context_views=2, target_views=1, image_size=(2, 2))
    example = dataset[0]

    assert example.context_rgb.shape == (2, 3, 2, 2)
    assert example.target_depth.shape == (1, 1, 2, 2)
    assert example.target_normal.shape == (1, 3, 2, 2)
    assert example.target_point.shape == (1, 3, 2, 2)
    assert example.target_visibility.all()
    assert example.target_cameras.intrinsics[0, 0, 0] == 2.0
    assert example.target_cameras.intrinsics[0, 0, 2] == 0.75


def test_manifest_discovery_uses_direct_scene_fast_path(tmp_path: Path, monkeypatch) -> None:
    expected = []
    for name in ("scene_b", "scene_a"):
        scene = tmp_path / name
        scene.mkdir()
        manifest = scene / "manifest.json"
        manifest.write_text("{}", encoding="utf-8")
        expected.append(manifest)

    def fail_recursive_scan(_self: Path, _pattern: str):
        raise AssertionError("direct scene manifests should not trigger a recursive scan")

    monkeypatch.setattr(Path, "rglob", fail_recursive_scan)

    assert _find_manifests(tmp_path) == sorted(expected)


def test_manifest_dataset_filters_exact_scene_id_allowlist(tmp_path: Path) -> None:
    for scene_id in ("scene_c", "scene_a", "scene_b"):
        _write_minimal_manifest(tmp_path / scene_id, scene_id)

    dataset = ManifestSceneDataset(
        tmp_path,
        context_views=2,
        target_views=1,
        scene_ids=("scene_b", "scene_a"),
    )

    assert tuple(manifest.scene_id for manifest in dataset.manifests) == ("scene_b", "scene_a")
    assert len(dataset.entries) == 2
    with pytest.raises(ValueError, match="scene_missing"):
        ManifestSceneDataset(
            tmp_path,
            context_views=2,
            target_views=1,
            scene_ids=("scene_a", "scene_missing"),
        )


def test_build_dataset_passes_scene_id_allowlist(tmp_path: Path) -> None:
    for scene_id in ("scene_a", "scene_b"):
        _write_minimal_manifest(tmp_path / scene_id, scene_id)
    config = DatasetConfig(
        name="manifest",
        root=str(tmp_path),
        image_size=(2, 2),
        context_views=2,
        target_views=1,
        scene_ids=("scene_b",),
    )

    dataset = build_dataset(config, seed=3)

    assert isinstance(dataset, ManifestSceneDataset)
    assert tuple(manifest.scene_id for manifest in dataset.manifests) == ("scene_b",)


def _write_local_protocol_fixture(root: Path) -> Path:
    scene = root / "local_scene"
    (scene / "rgb").mkdir(parents=True)
    (scene / "depth").mkdir()
    (scene / "normal").mkdir()
    rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    poses = []
    for translation in ([10.0, 0.0, 0.0], [10.0, 0.5, 0.0], [10.0, 1.0, 0.0]):
        pose = np.eye(4, dtype=np.float32)
        pose[:3, :3] = rotation
        pose[:3, 3] = translation
        poses.append(pose)
    frames = []
    for index, pose in enumerate(poses):
        Image.new("RGB", (2, 2), (20 + index, 30, 40)).save(scene / "rgb" / f"{index:06d}.png")
        depth_path = None
        normal_path = None
        if index == 2:
            np.save(scene / "depth" / f"{index:06d}.npy", np.full((2, 2), 2.0, np.float32))
            world_normal = np.zeros((2, 2, 3), dtype=np.float32)
            world_normal[..., 1] = 1.0
            np.save(scene / "normal" / f"{index:06d}.npy", world_normal)
            depth_path = f"depth/{index:06d}.npy"
            normal_path = f"normal/{index:06d}.npy"
        frames.append(
            ManifestFrame(
                frame_id=index,
                rgb=f"rgb/{index:06d}.png",
                intrinsics=[[2.0, 0.0, 0.5], [0.0, 2.0, 0.5], [0.0, 0.0, 1.0]],
                c2w=pose.tolist(),
                depth=depth_path,
                normal=normal_path,
            )
        )
    save_manifest(
        SceneManifest(
            "local_scene",
            (2, 2),
            ((100.0, 100.0, 100.0), (200.0, 200.0, 200.0)),
            tuple(frames),
            scene,
        ),
        scene / "manifest.json",
    )
    return scene


def test_context_local_protocol_ignores_manifest_and_target_geometry_for_support(
    tmp_path: Path,
) -> None:
    scene = _write_local_protocol_fixture(tmp_path)
    local_bounds = ((-2.0, -2.0, 0.1), (2.0, 2.0, 4.1))
    dataset = ManifestSceneDataset(
        scene,
        context_views=2,
        target_views=1,
        spatial_protocol="context_local_metric",
        local_bounds_m=local_bounds,
    )

    first = dataset[0]
    np.save(scene / "depth" / "000002.npy", np.full((2, 2), 200.0, np.float32))
    changed_target = dataset[0]

    torch.testing.assert_close(first.bounds, torch.tensor(local_bounds))
    torch.testing.assert_close(first.bounds, changed_target.bounds)
    torch.testing.assert_close(first.target_support, changed_target.target_support)
    assert first.target_visibility.all()
    assert not changed_target.target_visibility.any()
    assert not changed_target.target_depth.any()
    assert not changed_target.target_normal.any()
    assert not changed_target.target_point.any()
    assert not changed_target.target_rgb.any()
    assert first.scene_id == "local_scene"


def test_context_local_protocol_rebases_cameras_points_and_world_normals(tmp_path: Path) -> None:
    scene = _write_local_protocol_fixture(tmp_path)
    dataset = ManifestSceneDataset(
        scene,
        context_views=2,
        target_views=1,
        spatial_protocol="context_local_metric",
        local_bounds_m=((-2.0, -2.0, 0.1), (2.0, 2.0, 4.1)),
    )

    example = dataset[0]
    origins, directions = generate_rays(example.target_cameras)
    expected_points = origins + directions * example.target_depth[:, 0].unsqueeze(-1)

    torch.testing.assert_close(example.context_cameras.c2w[0], torch.eye(4), atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(example.target_cameras.c2w[0, :3, 3], torch.tensor([1.0, 0.0, 0.0]))
    torch.testing.assert_close(
        example.target_point, expected_points.permute(0, 3, 1, 2), atol=1e-6, rtol=1e-6
    )
    expected_normal = (
        torch.tensor([1.0, 0.0, 0.0]).view(1, 3, 1, 1).expand_as(example.target_normal)
    )
    torch.testing.assert_close(example.target_normal, expected_normal, atol=1e-6, rtol=1e-6)
    assert example.target_support.shape == (1, 1, 2, 2)


def test_context_local_protocol_preserves_explicit_metric_point_magnitude(tmp_path: Path) -> None:
    scene = _write_local_protocol_fixture(tmp_path)
    (scene / "point").mkdir()
    world_points = np.empty((2, 2, 3), dtype=np.float32)
    world_points[...] = [10.0, 3.0, 2.0]
    np.save(scene / "point" / "000002.npy", world_points)
    manifest_path = scene / "manifest.json"
    manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_data["frames"][2]["point"] = "point/000002.npy"
    manifest_path.write_text(json.dumps(manifest_data), encoding="utf-8")
    dataset = ManifestSceneDataset(
        scene,
        context_views=2,
        target_views=1,
        spatial_protocol="context_local_metric",
        local_bounds_m=((-4.0, -2.0, 0.1), (4.0, 2.0, 4.1)),
    )

    example = dataset[0]

    expected = torch.tensor([3.0, 0.0, 2.0]).view(1, 3, 1, 1).expand_as(example.target_point)
    torch.testing.assert_close(example.target_point, expected, atol=1e-6, rtol=1e-6)
