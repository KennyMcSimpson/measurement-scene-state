"""Tests for pilot manifest provenance before any raw image reads."""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path

import pytest

from mcss.data.pilot_provenance import validate_pilot_entry

PROJECT = Path(__file__).resolve().parents[1]


@pytest.fixture
def artifact_dir() -> Path:
    output_root = PROJECT / "outputs"
    output_root.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="test_pilot_provenance_", dir=output_root) as directory:
        yield Path(directory)


def test_valid_entry_returns_manifest_hashes_without_image_decoding(artifact_dir: Path) -> None:
    entry, prepared_root, online_path, query_path = _write_entry(artifact_dir)

    provenance = validate_pilot_entry(entry, prepared_root=prepared_root)

    assert provenance.episode_id == "train--scene_a--cam_00"
    assert provenance.online_manifest_sha256 == _sha256(online_path)
    assert provenance.query_manifest_sha256 == _sha256(query_path)
    assert provenance.online_rgb_references == 2
    assert provenance.query_rgb_references == 1
    assert provenance.query_depth_references == 1


def test_valid_dev_entry_binds_to_val_prepared_split(artifact_dir: Path) -> None:
    entry, prepared_root, _, _ = _write_entry(artifact_dir, split_id="dev")

    provenance = validate_pilot_entry(entry, prepared_root=prepared_root)

    assert provenance.split_id == "dev"
    assert (prepared_root / "val" / "scene_a" / "rgb" / "000000.png").is_file()


def test_rejects_cross_split_or_frame_alias_reference(artifact_dir: Path) -> None:
    entry, prepared_root, online_path, _ = _write_entry(artifact_dir)
    online = json.loads(online_path.read_text(encoding="utf-8"))
    online["stream"][0]["rgb"] = online["warmup"][0]["rgb"]
    online_path.write_text(json.dumps(online), encoding="utf-8")

    with pytest.raises(ValueError, match="online rgb reference"):
        validate_pilot_entry(entry, prepared_root=prepared_root)


def test_rejects_online_query_metadata_mismatch_before_reference_use(artifact_dir: Path) -> None:
    entry, prepared_root, _, query_path = _write_entry(artifact_dir)
    query = json.loads(query_path.read_text(encoding="utf-8"))
    query["scene_id"] = "other_scene"
    query_path.write_text(json.dumps(query), encoding="utf-8")

    with pytest.raises(ValueError, match="identity mismatch"):
        validate_pilot_entry(entry, prepared_root=prepared_root)


def test_rejects_query_rgb_alias_even_with_a_distinct_frame_id(artifact_dir: Path) -> None:
    entry, prepared_root, online_path, query_path = _write_entry(artifact_dir)
    online = json.loads(online_path.read_text(encoding="utf-8"))
    query = json.loads(query_path.read_text(encoding="utf-8"))
    query["query"][0]["rgb"] = online["warmup"][0]["rgb"]
    query_path.write_text(json.dumps(query), encoding="utf-8")

    with pytest.raises(ValueError, match="query rgb reference"):
        validate_pilot_entry(entry, prepared_root=prepared_root)


def _write_entry(
    directory: Path, *, split_id: str = "train"
) -> tuple[dict[str, object], Path, Path, Path]:
    prepared_root = directory / "prepared"
    prepared_split = {"train": "train", "dev": "val"}[split_id]
    scene_root = prepared_root / prepared_split / "scene_a"
    rgb_root = scene_root / "rgb"
    depth_root = scene_root / "depth"
    rgb_root.mkdir(parents=True)
    depth_root.mkdir(parents=True)
    for frame_id in range(3):
        (rgb_root / f"{frame_id:06d}.png").write_bytes(b"not decoded in provenance tests")
    (depth_root / "000002.npy").write_bytes(b"not decoded in provenance tests")
    identity = {
        "episode_id": f"{split_id}--scene_a--cam_00",
        "scene_id": "scene_a",
        "split_id": split_id,
        "query_vault_id": "qv-test",
    }
    online = {
        "schema_version": "mcss.dynamic.online_episode.v1",
        **identity,
        "image_size": [2, 2],
        "declared_length": 2,
        "warmup": [_online_frame(0, rgb_root / "000000.png")],
        "stream": [_online_frame(1, rgb_root / "000001.png")],
    }
    query = {
        "schema_version": "mcss.dynamic.query_vault.v1",
        **identity,
        "image_size": [2, 2],
        "query": [_query_frame(2, rgb_root / "000002.png", depth_root / "000002.npy")],
    }
    online_path = directory / "episode.online.json"
    query_path = directory / "episode.query.json"
    online_path.write_text(json.dumps(online), encoding="utf-8")
    query_path.write_text(json.dumps(query), encoding="utf-8")
    entry = {
        **identity,
        "online_manifest": str(online_path),
        "query_manifest": str(query_path),
        "stream_steps": 1,
        "query_count": 1,
    }
    return entry, prepared_root, online_path, query_path


def _online_frame(frame_id: int, rgb: Path) -> dict[str, object]:
    return {
        "frame_id": frame_id,
        "intrinsics": [[2.0, 0.0, 0.5], [0.0, 2.0, 0.5], [0.0, 0.0, 1.0]],
        "c2w": [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        "rgb": str(rgb),
    }


def _query_frame(frame_id: int, rgb: Path, depth: Path) -> dict[str, object]:
    return {**_online_frame(frame_id, rgb), "depth": str(depth)}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
