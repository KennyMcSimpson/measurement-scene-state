"""Replica cohort preparation: camera model, depth semantics, geometry rule and a synthetic run."""

import importlib.util
import io
import json
import types
import zipfile
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from mcss.geometry import generate_rays
from mcss.types import Cameras

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "prepare_rgbd_replica_data.py"
spec = importlib.util.spec_from_file_location("prepare_rgbd_replica_data", SCRIPT)
replica = importlib.util.module_from_spec(spec)
spec.loader.exec_module(replica)
SOURCE = np.array([[600.0, 0.0, 599.5], [0.0, 600.0, 339.5], [0.0, 0.0, 1.0]])
ROOM = (np.array([-4.0, -3.0, -4.0]), np.array([4.0, 1.5, 4.0]))


def test_target_camera_matches_the_hypersim_field_of_view():
    target = replica.target_intrinsics(SOURCE)
    assert target[0, 2] == pytest.approx(79.5) and target[1, 2] == pytest.approx(63.5)
    assert target[0, 0] == pytest.approx(138.56, rel=5e-3) and target[1, 1] == pytest.approx(
        147.80, rel=5e-3
    )
    assert replica.CROP["y0"] * 2 + replica.CROP["height"] == 680
    assert replica.CROP["x0"] * 2 + replica.CROP["width"] == 1200


def look_at(position, target):
    forward = (target - position) / np.linalg.norm(target - position)
    right = np.cross(forward, np.array([0.0, -1.0, 0.0]))
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    pose = np.eye(4)
    pose[:3, :3] = np.stack((right, down, forward), -1)  # OpenCV: x right, y down, z forward
    pose[:3, 3] = position
    return pose


def render(pose, height=680, width=1200, intrinsics=SOURCE):
    """Planar z-depth and a textured RGB of the inside of an axis-aligned box room."""
    y, x = np.indices((height, width))
    rays = np.stack(
        (
            (x - intrinsics[0, 2]) / intrinsics[0, 0],
            (y - intrinsics[1, 2]) / intrinsics[1, 1],
            np.ones_like(x, float),
        ),
        -1,
    )
    world = rays @ pose[:3, :3].T
    origin = pose[:3, 3]
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.stack(
            [(ROOM[j][i] - origin[i]) / world[..., i] for j in range(2) for i in range(3)], -1
        )
    t = np.where(t > 0, t, np.inf).min(-1)
    points = origin + world * t[..., None]
    z = t  # rays have unit z, so the ray parameter is the planar depth
    checker = (
        np.floor(points[..., 0] * 2) + np.floor(points[..., 1] * 2) + np.floor(points[..., 2] * 2)
    ) % 2
    rgb = np.stack((60 + 150 * checker, 90 + 100 * (1 - checker), 140 + 0 * checker), -1).astype(
        np.uint8
    )
    return rgb, z


def test_ray_distance_matches_the_project_ray_convention():
    target = replica.target_intrinsics(SOURCE)
    camera = Cameras(torch.tensor(target), torch.eye(4, dtype=torch.float64), (128, 160))
    _, directions = generate_rays(camera)
    z = np.full((128, 160), 2.0)
    expected = 2.0 / directions[..., 2].numpy()
    assert np.allclose(replica.ray_distance(z, target), expected, rtol=1e-9)


def test_geometry_rule_accepts_opencv_poses_and_rejects_flipped_ones():
    a = look_at(np.array([0.0, -1.0, 0.0]), np.array([2.0, -1.0, 3.0]))
    b = look_at(np.array([0.3, -1.0, 0.2]), np.array([2.5, -1.0, 3.0]))
    _, za = render(a)
    _, zb = render(b)
    error, landing = replica.reprojection_error(za, a, zb, b, SOURCE)
    assert error < 1e-3 and landing > 0.3
    flip = np.diag([1.0, -1.0, -1.0, 1.0])
    bad, _ = replica.reprojection_error(za, a @ flip, zb, b @ flip, SOURCE)
    assert bad is None or bad > 0.05


def trajectory_text(n=2000):
    poses = []
    for k in range(n):
        angle = np.radians(0.3 * k)  # 15 degrees every 50 frames
        position = np.array([1.5 * np.cos(angle), -1.0, 1.5 * np.sin(angle)])
        target = np.array([-1.5 * np.cos(angle), -1.0, -1.5 * np.sin(angle)])  # opposite side
        poses.append(look_at(position, target))
    return "\n".join(" ".join(f"{v:.9f}" for v in pose.ravel()) for pose in poses) + "\n", poses


def synthetic_archive(path, scenes):
    params = {
        "camera": {
            "w": 1200,
            "h": 680,
            "fx": 600.0,
            "fy": 600.0,
            "cx": 599.5,
            "cy": 339.5,
            "scale": 6553.5,
        }
    }
    text, poses = trajectory_text()
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr("Replica/cam_params.json", json.dumps(params))
        for scene in scenes:
            archive.writestr(f"Replica/{scene}/traj.txt", text)
            for fid in range(0, 751, 50):
                rgb, z = render(poses[fid])
                buffer = io.BytesIO()
                Image.fromarray(rgb).save(buffer, format="JPEG", quality=95)
                archive.writestr(f"Replica/{scene}/results/frame{fid:06d}.jpg", buffer.getvalue())
                buffer = io.BytesIO()
                Image.fromarray(np.round(z * 6553.5).clip(0, 65535).astype(np.uint16)).save(
                    buffer, format="PNG"
                )
                archive.writestr(f"Replica/{scene}/results/depth{fid:06d}.png", buffer.getvalue())
    return poses


def test_lock_then_prepare_on_a_synthetic_archive(tmp_path, monkeypatch):
    scenes = ("office0",)
    monkeypatch.setattr(replica, "SCENES", scenes)
    monkeypatch.setattr(replica, "MIN_VALID_SCENES", 1)
    bounds = tmp_path / "bounds_contract.json"
    bounds.write_text(json.dumps({"near_m": 0.214, "far_m": 5.672, "source_sha256": "0" * 64}))
    monkeypatch.setattr(replica, "BOUNDS", bounds)
    poses = synthetic_archive(tmp_path / "Replica.zip", scenes)
    reader = types.SimpleNamespace(
        etag='"synthetic"', size=(tmp_path / "Replica.zip").stat().st_size, transferred=0
    )

    def opener(budget=None):
        return reader, zipfile.ZipFile(tmp_path / "Replica.zip")

    root = tmp_path / replica.EXPERIMENT
    root.mkdir()
    _, archive = opener()
    with archive:
        lock = replica.build_lock(archive, reader)
    assert lock["frame_rule"]["spacing"] == 50 and lock["image_or_depth_member_read"] is False
    (root / "candidate_lock.json").write_text(json.dumps(lock))
    replica.prepare(root, opener)
    integrity = json.loads((root / "preparation_integrity.json").read_text())
    assert integrity["status"] == "PASS", json.loads((root / "scene_failures.json").read_text())
    manifest = json.loads((root / "manifest_replica.json").read_text())
    record = manifest["scenes"][0]
    assert record["split"] == "REPLICA" and set(record["roles"]) >= {
        "context_a",
        "context_b",
        "primary_query",
    }
    frame = record["frames"][0]
    depth = np.load(frame["depth"])
    assert depth.shape == (128, 160) and manifest["depth_semantics"] == "ray_distance_meters"
    target = np.array(frame["intrinsics"])
    camera = Cameras(torch.tensor(target), torch.tensor(np.array(frame["c2w"])), (128, 160))
    _, z = render(poses[0], 128, 160, target)
    _, directions = generate_rays(camera)
    local = directions.numpy() @ np.array(frame["c2w"])[:3, :3]
    expected = z / local[..., 2]
    inner = (slice(8, -8), slice(8, -8))
    assert np.median(np.abs(depth[inner] - expected[inner]) / expected[inner]) < 0.02
    assert json.loads((root / "geometry_checks.json").read_text())["office0"]["pass"]
