"""Regression tests for the online-only episode reader and compiler."""

from __future__ import annotations

import importlib.util
import json
import tempfile
from pathlib import Path

import pytest
import torch
from PIL import Image

from mcss.data.episodes import OnlineEpisodeSource

PROJECT = Path(__file__).resolve().parents[1]
_COMPILER_SPEC = importlib.util.spec_from_file_location(
    "compile_dynamic_episodes", PROJECT / "scripts/compile_dynamic_episodes.py"
)
assert _COMPILER_SPEC is not None and _COMPILER_SPEC.loader is not None
compiler = importlib.util.module_from_spec(_COMPILER_SPEC)
_COMPILER_SPEC.loader.exec_module(compiler)


@pytest.fixture
def artifact_dir() -> Path:
    output_root = PROJECT / "outputs"
    output_root.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="test_episode_access_", dir=output_root) as directory:
        yield Path(directory)


def test_compiler_strips_labels_and_source_reads_only_arrived_rgb(artifact_dir: Path) -> None:
    _write_rgb(artifact_dir / "warmup.png", 16)
    _write_rgb(artifact_dir / "stream_1.png", 32)
    pilot_path = _write_pilot(artifact_dir)

    compiled_dir = artifact_dir / "compiled"
    index = compiler.compile_pilot_manifest(pilot_path, compiled_dir)
    online_path = Path(index[0]["online_manifest"])
    online = json.loads(online_path.read_text(encoding="utf-8"))
    vault = json.loads(Path(index[0]["query_manifest"]).read_text(encoding="utf-8"))

    assert not _contains_key(online, "depth")
    assert not _contains_key(online, "normal")
    assert not _contains_key(online, "query")
    assert vault["query"][0]["depth"].endswith("3.depth.npy")

    source = OnlineEpisodeSource.from_file(online_path)
    warmup = source.warmup()
    assert [observation.frame_id for observation in warmup] == [0]
    assert torch.equal(warmup[0].camera.c2w, torch.eye(4))
    assert source.remaining_steps == 2

    first_stream = source.next_observation()
    assert first_stream.frame_id == 1
    assert source.remaining_steps == 1

    # The final stream image is deliberately absent: no future RGB was touched earlier.
    with pytest.raises(FileNotFoundError):
        source.next_observation()


def test_online_reader_rejects_query_or_label_fields(artifact_dir: Path) -> None:
    _write_rgb(artifact_dir / "warmup.png", 16)
    _write_rgb(artifact_dir / "stream_1.png", 32)
    pilot_path = _write_pilot(artifact_dir)
    compiled_dir = artifact_dir / "compiled"
    index = compiler.compile_pilot_manifest(pilot_path, compiled_dir)
    online_path = Path(index[0]["online_manifest"])
    online = json.loads(online_path.read_text(encoding="utf-8"))
    online["warmup"][0]["depth"] = "poison.depth.npy"
    online_path = artifact_dir / "poisoned_online.json"
    online_path.write_text(json.dumps(online), encoding="utf-8")

    with pytest.raises(ValueError, match="fields mismatch"):
        OnlineEpisodeSource.from_file(online_path)


def test_compiler_rejects_test_split_without_opening_references(artifact_dir: Path) -> None:
    pilot_path = _write_pilot(artifact_dir, split="test")

    with pytest.raises(ValueError, match="refuses split"):
        compiler.compile_pilot_manifest(pilot_path, artifact_dir / "compiled")


def _write_pilot(directory: Path, *, split: str = "dev") -> Path:
    records = {
        "warmup": [_record(directory, 0, "warmup.png")],
        "stream": [
            _record(directory, 1, "stream_1.png"),
            _record(directory, 2, "future_stream.png"),
        ],
        "query": [_record(directory, 3, "query.png")],
    }
    pilot = {
        "schema": "mcss.dynamic_ttt.pilot_plan.v1",
        "final_holdout_used": False,
        "diagnostic_test_used": False,
        "episodes": [
            {
                "scene_id": "scene_a",
                "split": split,
                "camera": "cam_00",
                "image_size": [2, 2],
                **records,
            }
        ],
    }
    path = directory / "pilot.json"
    path.write_text(json.dumps(pilot), encoding="utf-8")
    return path


def _record(directory: Path, frame_id: int, rgb_name: str) -> dict[str, object]:
    return {
        "frame_id": frame_id,
        "intrinsics": [[2.0, 0.0, 0.5], [0.0, 2.0, 0.5], [0.0, 0.0, 1.0]],
        "c2w": torch.eye(4).tolist(),
        "rgb": str(directory / rgb_name),
        "depth": str(directory / f"{frame_id}.depth.npy"),
        "normal": str(directory / f"{frame_id}.normal.npy"),
    }


def _write_rgb(path: Path, value: int) -> None:
    Image.new("RGB", (2, 2), (value, value, value)).save(path)


def _contains_key(value: object, expected: str) -> bool:
    if isinstance(value, dict):
        return expected in value or any(_contains_key(item, expected) for item in value.values())
    if isinstance(value, list):
        return any(_contains_key(item, expected) for item in value)
    return False
