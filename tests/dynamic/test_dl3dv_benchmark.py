"""Contract tests for the metadata-only and causal DL3DV-140 adapter."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
from PIL import Image

from mcss.data.dl3dv_benchmark import (
    DL3DV_SCENE_COUNT,
    DL3DV_SPLIT_SHA256,
    Dl3dvBenchmarkMetadata,
    Dl3dvFrame,
    Dl3dvOnlineObservationSource,
    Dl3dvProtocol,
    Dl3dvProtocolError,
    Dl3dvSceneMetadata,
    LongLRMSplitRecord,
    compute_warmup_gauge,
    load_benchmark_metadata,
    load_long_lrm_split,
    official_pose_transform,
    prepare_scene_metadata,
    preprocess_dl3dv_rgb,
    target_frame_indices,
    validate_complete_coverage,
)
from mcss.dynamic.types import SealedScene, hash_scene_state
from mcss.types import SceneState

PROJECT = Path(__file__).resolve().parents[2]
SPLIT_PATH = (
    PROJECT
    / "outputs"
    / "benchmark_protocol_20260920"
    / "sources"
    / "Long-LRM"
    / "data"
    / "dl3dv_fold_8_kmeans_input_idx.json"
)


def _frame(
    scene_id: str,
    frame_id: int,
    *,
    source_size: tuple[int, int] = (4, 6),
    x: float | None = None,
    rgb_name: str | None = None,
) -> Dl3dvFrame:
    pose = np.eye(4, dtype=np.float64)
    pose[0, 3] = float(frame_id if x is None else x)
    return Dl3dvFrame(
        scene_id=scene_id,
        frame_id=frame_id,
        rgb_relpath=rgb_name or f"images_4/frame_{frame_id:04d}.png",
        c2w=tuple(tuple(float(value) for value in row) for row in pose),
        raw_fxfycxcy=(5.0, 5.0, 2.5, 1.5),
        distortion=(0.0, 0.0, 0.0, 0.0),
        source_image_size=source_size,
    )


def _scene(
    root: Path,
    *,
    scene_id: str = "scene",
    input_indices: tuple[int, ...] = (
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        17,
        18,
    ),
    query_indices: tuple[int, ...] = (0, 8, 16),
    source_size: tuple[int, int] = (4, 6),
    target_size: tuple[int, int] = (2, 3),
    mode: str = "full",
    input_count: int = 16,
    frame_count: int = 20,
    pose_values: tuple[float, ...] | None = None,
) -> Dl3dvSceneMetadata:
    values = pose_values or tuple(float(index) for index in range(frame_count))
    frames = tuple(
        _frame(scene_id, index, source_size=source_size, x=values[index])
        for index in range(frame_count)
    )
    return Dl3dvSceneMetadata(
        scene_id=scene_id,
        scene_root=root,
        frames=frames,
        input_indices=input_indices,
        query_indices=query_indices,
        protocol=Dl3dvProtocol(mode, input_count),
        target_size=target_size,
    )


def _write_image(path: Path, value: int, size: tuple[int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (size[1], size[0]), (value, value, value)).save(path)


def _write_transforms(
    root: Path,
    *,
    scene_id: str,
    frame_count: int,
    input_indices: tuple[int, ...],
    source_size: tuple[int, int] = (4, 6),
    file_prefix: str = "images_4",
) -> Path:
    scene_root = root / scene_id
    transforms_dir = scene_root / "nerfstudio"
    transforms_dir.mkdir(parents=True)
    height, width = source_size
    transforms = {
        "w": width,
        "h": height,
        "fl_x": 5.0,
        "fl_y": 5.0,
        "cx": 2.5,
        "cy": 1.5,
        "k1": 0.0,
        "k2": 0.0,
        "p1": 0.0,
        "p2": 0.0,
        "frames": [],
    }
    for index in range(frame_count):
        pose = np.eye(4, dtype=np.float64)
        pose[0, 3] = float(index)
        transforms["frames"].append(
            {
                "file_path": f"{file_prefix}/frame_{index:04d}.png",
                "transform_matrix": pose.tolist(),
            }
        )
    transforms_path = transforms_dir / "transforms.json"
    transforms_path.write_text(json.dumps(transforms), encoding="utf-8")
    return transforms_path


def _dummy_sealed(scene: Dl3dvSceneMetadata) -> SealedScene:
    state = SceneState(
        torch.zeros((1, 1, 2, 2, 2)),
        torch.full((1, 3, 2, 2, 2), 0.5),
        torch.zeros((1, 1, 2, 2, 2)),
        torch.tensor([[[-6.0, -4.0, 0.05], [6.0, 4.0, 12.05]]]),
    )
    return SealedScene(
        scene.episode_id,
        scene.scene_id,
        scene.split_id,
        scene.query_vault_id,
        state,
        hash_scene_state(state),
        scene.arrival_indices,
        "checkpoint",
        "config",
        "fast",
        torch.eye(4),
    )


def test_pinned_long_lrm_split_has_complete_140_scene_coverage() -> None:
    records = load_long_lrm_split(SPLIT_PATH)
    assert len(records) == DL3DV_SCENE_COUNT
    assert records[0].input_indices(16)[:4] == (13, 36, 57, 76)
    assert all(
        len(record.input_indices(count)) == count
        for record in records
        for count in (16, 32, 64, 128)
    )


def test_pinned_long_lrm_split_rejects_modified_bytes(tmp_path: Path) -> None:
    modified = tmp_path / "split.json"
    original = SPLIT_PATH.read_bytes()
    modified.write_bytes(original.replace(b"scene_name", b"scene_namx", 1))

    with pytest.raises(Dl3dvProtocolError, match="SHA256"):
        load_long_lrm_split(modified)


def test_frame_rejects_non_rigid_camera_pose() -> None:
    pose = np.eye(4, dtype=np.float64)
    pose[0, 1] = 0.01

    with pytest.raises(Dl3dvProtocolError, match="orthonormal"):
        Dl3dvFrame(
            scene_id="scene",
            frame_id=0,
            rgb_relpath="images_4/frame_0000.png",
            c2w=tuple(tuple(float(value) for value in row) for row in pose),
            raw_fxfycxcy=(5.0, 5.0, 2.5, 1.5),
            distortion=(0.0, 0.0, 0.0, 0.0),
            source_image_size=(4, 6),
        )


def test_metadata_round_trip_rebinds_split_and_scene_protocol(tmp_path: Path) -> None:
    records = load_long_lrm_split(SPLIT_PATH)
    protocol = Dl3dvProtocol("full", 16)
    raw_root = tmp_path / "raw"
    scenes = []
    for record in records:
        inputs = record.input_indices(16)
        frame_count = max(inputs) + 1
        frames = tuple(
            _frame(record.scene_name, index, source_size=(540, 960))
            for index in range(frame_count)
        )
        scenes.append(
            Dl3dvSceneMetadata(
                record.scene_name,
                raw_root / record.scene_name,
                frames,
                inputs,
                target_frame_indices(frame_count),
                protocol,
            )
        )
    metadata = Dl3dvBenchmarkMetadata(
        tuple(scenes), raw_root, SPLIT_PATH, DL3DV_SPLIT_SHA256, protocol
    )
    path = tmp_path / "metadata.json"
    path.write_text(json.dumps(metadata.to_dict(), indent=2), encoding="utf-8")

    loaded = load_benchmark_metadata(path)
    assert loaded.protocol == protocol
    assert loaded.coverage()["scene_count"] == DL3DV_SCENE_COUNT

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["scenes"][0]["input_indices"][0] += 1
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(Dl3dvProtocolError, match="input_indices"):
        load_benchmark_metadata(path)


def test_supported_modes_match_paper_conditions() -> None:
    assert Dl3dvProtocol("full", 16).split_key == "fold_8_kmeans_16_input"
    assert Dl3dvProtocol("ar", 32).name == "ar32"
    with pytest.raises(Dl3dvProtocolError, match="supports input counts"):
        Dl3dvProtocol("ar", 16)


def test_prepare_scene_reads_metadata_without_opening_any_image(tmp_path: Path) -> None:
    scene_id = "scene"
    input_indices = tuple(index for index in range(32) if index % 8 != 0)[:16]
    _write_transforms(tmp_path, scene_id=scene_id, frame_count=32, input_indices=input_indices)
    split = LongLRMSplitRecord(scene_id, {16: input_indices})

    metadata = prepare_scene_metadata(
        tmp_path / scene_id,
        split,
        Dl3dvProtocol("full", 16),
        require_source_size=False,
    )

    assert metadata.raw_source_size == (4, 6)
    assert metadata.query_indices == target_frame_indices(32)
    assert not list((tmp_path / scene_id / "nerfstudio").glob("images_4/*"))


def test_prepare_scene_maps_original_resolution_transforms_to_images4(tmp_path: Path) -> None:
    scene_id = "scene"
    input_indices = tuple(index for index in range(32) if index % 8 != 0)[:16]
    _write_transforms(
        tmp_path,
        scene_id=scene_id,
        frame_count=32,
        input_indices=input_indices,
        source_size=(2160, 3840),
        file_prefix="images",
    )
    split = LongLRMSplitRecord(scene_id, {16: input_indices})

    metadata = prepare_scene_metadata(
        tmp_path / scene_id,
        split,
        Dl3dvProtocol("full", 16),
        require_source_size=True,
    )

    assert metadata.raw_source_size == (540, 960)
    assert metadata.frames[0].rgb_relpath == "images_4/frame_0000.png"
    assert metadata.frames[0].raw_fxfycxcy == pytest.approx((1.25, 1.25, 0.625, 0.375))


def test_prepare_scene_rejects_context_query_overlap() -> None:
    frames = tuple(_frame("scene", index) for index in range(16))
    with pytest.raises(Dl3dvProtocolError, match="context/query overlap"):
        # A direct protocol check is kept independent of image preparation.
        from mcss.data.dl3dv_benchmark import _validate_scene_indices

        _validate_scene_indices(16, (0, 1, 2, 8), target_stride=8)
    assert frames[0].frame_id == 0


def test_complete_coverage_rejects_duplicate_scene_ids(tmp_path: Path) -> None:
    scenes = [
        _scene(
            tmp_path,
            scene_id=f"scene-{index}",
            frame_count=20,
            input_indices=(
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                9,
                10,
                11,
                12,
                13,
                14,
                15,
                17,
                18,
            ),
            query_indices=(0, 8, 16),
        )
        for index in range(DL3DV_SCENE_COUNT)
    ]
    scenes[-1] = scenes[-2]
    with pytest.raises(Dl3dvProtocolError, match="duplicate scene IDs"):
        validate_complete_coverage(scenes)


def test_official_pose_transform_matches_pinned_axis_operations() -> None:
    raw = np.eye(4, dtype=np.float64)
    raw[0, 1] = 2.0
    raw[0, 3] = 3.0
    expected = raw.copy()
    expected[0:3, 1:3] *= -1.0
    expected = expected[[1, 0, 2, 3], :]
    expected[2, :] *= -1.0
    np.testing.assert_allclose(official_pose_transform(raw), expected)


def test_official_rgb_preprocessing_and_intrinsics_transform(tmp_path: Path) -> None:
    image_path = tmp_path / "frame.png"
    source_size = (4, 6)
    image = np.zeros((source_size[0], source_size[1], 3), dtype=np.uint8)
    image[:, :, 0] = np.arange(source_size[1], dtype=np.uint8)[None, :]
    image[:, :, 1] = 100
    cv2.imwrite(str(image_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    frame = _frame("scene", 0, source_size=source_size, rgb_name="images_4/frame.png")

    rgb, intrinsic = preprocess_dl3dv_rgb(image_path, frame, target_size=(4, 6))

    assert rgb.shape == (3, 4, 6)
    assert rgb.dtype == torch.float32
    assert float(rgb.min()) >= 0.0 and float(rgb.max()) <= 1.0
    expected_raw = np.asarray([[5.0, 0.0, 2.5], [0.0, 5.0, 1.5], [0.0, 0.0, 1.0]])
    expected_new, _ = cv2.getOptimalNewCameraMatrix(
        expected_raw, np.zeros(4), (6, 4), 0, (6, 4)
    )
    np.testing.assert_allclose(intrinsic, expected_new, rtol=1e-6, atol=1e-6)


def test_lazy_source_keeps_query_poison_unread_until_after_seal(tmp_path: Path) -> None:
    input_indices = tuple(index for index in range(24) if index % 8 != 0)[:16]
    query_indices = target_frame_indices(24)
    scene = _scene(
        tmp_path,
        input_indices=input_indices,
        query_indices=query_indices,
        frame_count=24,
        source_size=(4, 6),
        target_size=(2, 3),
    )
    for frame_id in input_indices:
        _write_image(
            tmp_path / "nerfstudio" / scene.frame(frame_id).rgb_relpath,
            17,
            (4, 6),
        )
    source = Dl3dvOnlineObservationSource(scene)
    warmup = source.warmup()
    while source.remaining_steps:
        source.next_observation()
    assert [observation.frame_id for observation in warmup] == list(scene.warmup_indices)
    assert source.sealed is False

    sealed = _dummy_sealed(scene)
    queries = source.queries_for_sealed(sealed)
    assert source.sealed is True
    # Query files did not need to exist while context was consumed.  They are
    # created only after callback construction to prove the supplier is lazy.
    query_frame = scene.frame(scene.query_indices[0])
    query_path = tmp_path / "nerfstudio" / query_frame.rgb_relpath
    _write_image(query_path, 239, (4, 6))
    assert float(queries[0].rgb_supplier().mean()) > 0.8
    with pytest.raises(ValueError, match="sealed"):
        source.next_observation()


def test_warmup_gauge_is_causal_and_rejects_zero_baseline() -> None:
    warmup = []
    for value in (0.0, 1.0, 2.0, 3.0):
        pose = np.eye(4)
        pose[0, 3] = value
        warmup.append(pose)
    first = compute_warmup_gauge(warmup)
    changed_future = np.eye(4)
    changed_future[0, 3] = 1000.0
    assert first.scale == pytest.approx(1.9428480059945201 / 3.0)
    expected_second = np.eye(4)
    expected_second[0, 3] = first.scale
    np.testing.assert_allclose(first.transform(warmup[1]), expected_second)
    assert first.transform(changed_future)[0, 3] == pytest.approx(1000.0 * first.scale)
    with pytest.raises(Dl3dvProtocolError, match="too small"):
        compute_warmup_gauge([np.eye(4)] * 4)


def test_source_gauge_and_prefix_cameras_ignore_later_and_query_poses(tmp_path: Path) -> None:
    input_indices = tuple(index for index in range(24) if index % 8 != 0)[:16]
    first_values = tuple(float(index) for index in range(24))
    second_values = list(first_values)
    second_values[0] = 500.0
    second_values[5] = 100.0
    second_values[6] = 200.0
    second_values[8] = 800.0
    scene_a = _scene(
        tmp_path / "a",
        input_indices=input_indices,
        query_indices=target_frame_indices(24),
        frame_count=24,
        pose_values=first_values,
        source_size=(4, 6),
        target_size=(2, 3),
    )
    scene_b = _scene(
        tmp_path / "b",
        input_indices=input_indices,
        query_indices=target_frame_indices(24),
        frame_count=24,
        pose_values=tuple(second_values),
        source_size=(4, 6),
        target_size=(2, 3),
    )
    source_a = Dl3dvOnlineObservationSource(scene_a)
    source_b = Dl3dvOnlineObservationSource(scene_b)
    assert source_a.gauge.scale == pytest.approx(source_b.gauge.scale)
    for index in scene_a.warmup_indices:
        np.testing.assert_allclose(
            source_a.gauge.transform(scene_a.frame(index).c2w_array),
            source_b.gauge.transform(scene_b.frame(index).c2w_array),
        )
