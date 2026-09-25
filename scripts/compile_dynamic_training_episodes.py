"""Compile fixed-length train/dev episodes from prepared Hypersim manifests.

The compiler reads only train and val scene manifests.  It resolves and stats the RGB and
query-depth references used by each episode, but it never decodes RGB files.  A small optional
train-only depth health scan is kept separate from episode compilation because it reads four
query depth arrays per eligible train scene to detect invalid supervision.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_PREPARED_ROOT = PROJECT / "data/hypersim_er_prepared"
DEFAULT_PARTITIONS = PROJECT / "configs/hypersim_er_partitions.csv"
DEFAULT_OUTPUT_ROOT = PROJECT / "outputs/dynamic_training_run_20260919"
ONLINE_SCHEMA = "mcss.dynamic.online_episode.v1"
QUERY_SCHEMA = "mcss.dynamic.query_vault.v1"
INDEX_SCHEMA = "mcss.dynamic.episode_index.v1"
INVENTORY_SCHEMA = "mcss.dynamic.training_inventory.v1"
SPLIT_MAPPING = {"train": "train", "val": "dev"}
REQUIRED_COUNTS = {"train": 365, "val": 46}
EPISODE_LENGTH = 16
WARMUP_LENGTH = 4
STREAM_LENGTH = 8
QUERY_LENGTH = 4
RIGID_CAMERA_TOLERANCE = 1e-3


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid manifest JSON: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"manifest must contain an object: {path}")
    return value


def _identifier(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value or any(char in value for char in "\\/"):
        raise ValueError(f"{name} must be a simple non-empty identifier")
    return value


def _image_size(value: Any, path: Path) -> list[int]:
    if (
        not isinstance(value, list)
        or len(value) != 2
        or any(type(item) is not int or item < 2 for item in value)
    ):
        raise ValueError(f"invalid image_size in {path}")
    return list(value)


def _matrix(value: Any, shape: tuple[int, int], name: str, path: Path) -> list[list[float]]:
    if not isinstance(value, list) or len(value) != shape[0]:
        raise ValueError(f"invalid {name} shape in {path}")
    result: list[list[float]] = []
    for row in value:
        if not isinstance(row, list) or len(row) != shape[1]:
            raise ValueError(f"invalid {name} shape in {path}")
        converted = []
        for item in row:
            if type(item) not in {int, float} or not math.isfinite(float(item)):
                raise ValueError(f"invalid {name} values in {path}")
            converted.append(float(item))
        result.append(converted)
    if shape == (4, 4) and any(
        not math.isclose(actual, expected, abs_tol=1e-5)
        for actual, expected in zip(result[3], [0.0, 0.0, 0.0, 1.0], strict=True)
    ):
        raise ValueError(f"c2w must have a homogeneous final row in {path}")
    return result


def _safe_manifest_reference(scene_root: Path, value: Any, expected: Path, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty relative path")
    normalized = value.replace("\\", "/")
    pure = PurePosixPath(normalized)
    if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
        raise ValueError(f"unsafe {label} path: {value!r}")
    if len(pure.parts[0]) >= 2 and pure.parts[0][1] == ":":
        raise ValueError(f"drive-qualified {label} path: {value!r}")
    resolved = (scene_root / Path(*pure.parts)).resolve()
    expected_resolved = expected.resolve()
    try:
        resolved.relative_to(scene_root.resolve())
    except ValueError as error:
        raise ValueError(f"{label} escapes scene root: {value!r}") from error
    if resolved != expected_resolved:
        raise ValueError(f"{label} does not match scene/frame identity: {value!r}")
    try:
        resolved.stat()
    except OSError as error:
        raise FileNotFoundError(f"missing {label} reference: {resolved}") from error
    if not resolved.is_file():
        raise ValueError(f"{label} reference is not a regular file: {resolved}")
    return resolved


def _frame_records(manifest: dict[str, Any], path: Path) -> list[dict[str, Any]]:
    raw_frames = manifest.get("frames")
    if not isinstance(raw_frames, list) or not raw_frames:
        raise ValueError(f"manifest frames must be a non-empty list: {path}")
    frames: list[dict[str, Any]] = []
    for raw in raw_frames:
        if not isinstance(raw, dict):
            raise ValueError(f"manifest frame must be an object: {path}")
        frame_id = raw.get("frame_id")
        if type(frame_id) is not int or frame_id < 0:
            raise ValueError(f"invalid frame_id in {path}")
        _matrix(raw.get("intrinsics"), (3, 3), "intrinsics", path)
        _matrix(raw.get("c2w"), (4, 4), "c2w", path)
        frames.append(raw)
    frames.sort(key=lambda frame: frame["frame_id"])
    ids = [frame["frame_id"] for frame in frames]
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate frame_id values in {path}")
    return frames


def _camera_rigid_status(
    frame: dict[str, Any], *, tolerance: float = RIGID_CAMERA_TOLERANCE
) -> tuple[bool, str | None]:
    rotation = np.asarray(frame["c2w"], dtype=np.float64)[:3, :3]
    gram = rotation.T @ rotation
    determinant = float(np.linalg.det(rotation))
    if not np.allclose(gram, np.eye(3), atol=tolerance, rtol=0.0):
        return False, "rotation_orthogonality"
    if not math.isclose(determinant, 1.0, abs_tol=tolerance, rel_tol=0.0):
        return False, "rotation_determinant"
    return True, None


def _filter_camera_frames(
    frames: list[dict[str, Any]], *, rigid_cameras_only: bool
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not rigid_cameras_only:
        return frames, []
    retained: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for frame in frames:
        valid, reason = _camera_rigid_status(frame)
        if valid:
            retained.append(frame)
        else:
            excluded.append({"frame_id": frame["frame_id"], "reason": reason})
    return retained, excluded


def _load_partitions(path: Path) -> tuple[dict[str, dict[str, str]], str]:
    try:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
    except OSError as error:
        raise ValueError(f"cannot read partitions CSV: {path}") from error
    required = {
        "scene_name",
        "official_split",
        "protocol_partition",
        "selected_camera",
        "frame_count",
    }
    if not rows or any(not required.issubset(row) for row in rows):
        raise ValueError(f"partitions CSV is missing required columns: {path}")
    selected: dict[str, dict[str, str]] = {}
    for row in rows:
        partition = row["protocol_partition"]
        if partition not in SPLIT_MAPPING:
            continue
        scene = _identifier(row["scene_name"], "scene_name")
        if scene in selected:
            raise ValueError(f"duplicate train/val partition row: {scene}")
        _identifier(row["selected_camera"], "selected_camera")
        try:
            int(row["frame_count"])
        except ValueError as error:
            raise ValueError(f"invalid partition frame_count: {scene}") from error
        selected[scene] = row
    return selected, sha256(path)


def _parse_scene(
    manifest_path: Path,
    source_split: str,
    partitions: dict[str, dict[str, str]],
    *,
    rigid_cameras_only: bool = False,
) -> dict[str, Any]:
    manifest = _read_json(manifest_path)
    if manifest.get("schema_version") != 1:
        raise ValueError(f"unsupported manifest schema: {manifest_path}")
    if manifest.get("coordinate_convention") != "opencv_c2w":
        raise ValueError(f"unsupported coordinate convention: {manifest_path}")
    scene_id = _identifier(manifest.get("scene_id"), "scene_id")
    if scene_id != manifest_path.parent.name:
        raise ValueError(f"manifest scene_id does not match directory: {manifest_path}")
    row = partitions.get(scene_id)
    if row is None or row["protocol_partition"] != source_split:
        raise ValueError(f"partition mismatch for {scene_id}")
    if row["official_split"] != source_split:
        raise ValueError(f"official split mismatch for {scene_id}")
    frames = _frame_records(manifest, manifest_path)
    original_frame_count = len(frames)
    if int(row["frame_count"]) != original_frame_count:
        raise ValueError(f"partition frame_count mismatch for {scene_id}")
    frames, excluded_frames = _filter_camera_frames(
        frames, rigid_cameras_only=rigid_cameras_only
    )
    return {
        "scene_id": scene_id,
        "source_split": source_split,
        "split_id": SPLIT_MAPPING[source_split],
        "camera": _identifier(row["selected_camera"], "selected_camera"),
        "manifest_path": manifest_path.resolve(),
        "manifest_sha256": sha256(manifest_path),
        "image_size": _image_size(manifest.get("image_size"), manifest_path),
        "frame_count": len(frames),
        "original_frame_count": original_frame_count,
        "rigid_cameras_only": rigid_cameras_only,
        "excluded_camera_frames": excluded_frames,
        "frames": frames,
    }


def _window_starts(
    frame_count: int,
    *,
    total_length: int,
    stride: int,
    max_windows: int | None,
) -> list[int]:
    if total_length < 1 or stride < 1:
        raise ValueError("window total length and stride must be positive")
    if max_windows is not None and (type(max_windows) is not int or max_windows < 1):
        raise ValueError("max_windows must be positive or None")
    if frame_count < total_length:
        return []
    starts = list(range(0, frame_count - total_length + 1, stride))
    return starts if max_windows is None else starts[:max_windows]


def _make_episode(
    scene: dict[str, Any],
    *,
    start: int = 0,
    warmup_steps: int = WARMUP_LENGTH,
    stream_steps: int = STREAM_LENGTH,
    query_count: int = QUERY_LENGTH,
    window_index: int = 0,
    include_window_suffix: bool = False,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    total_length = warmup_steps + stream_steps + query_count
    if min(warmup_steps, stream_steps, query_count) < 1:
        raise ValueError("warmup_steps, stream_steps, and query_count must be positive")
    if start < 0 or scene["frame_count"] < start + total_length:
        raise ValueError(
            f"too few frames for {scene['scene_id']} window {start}:{start + total_length}"
        )
    selected = scene["frames"][start : start + total_length]
    if len(selected) != total_length:
        raise ValueError("selected window does not contain the requested number of frames")
    records: list[dict[str, Any]] = []
    scene_root = scene["manifest_path"].parent
    image_size = scene["image_size"]
    for position, frame in enumerate(selected):
        frame_id = frame["frame_id"]
        rgb_path = _safe_manifest_reference(
            scene_root,
            frame.get("rgb"),
            scene_root / "rgb" / f"{frame_id:06d}.png",
            f"{scene['scene_id']} frame {frame_id} rgb",
        )
        record = {
            "frame_id": frame_id,
            "intrinsics": _matrix(
                frame.get("intrinsics"), (3, 3), "intrinsics", scene["manifest_path"]
            ),
            "c2w": _matrix(frame.get("c2w"), (4, 4), "c2w", scene["manifest_path"]),
            "rgb": rgb_path.as_posix(),
        }
        if position >= warmup_steps + stream_steps:
            depth_path = _safe_manifest_reference(
                scene_root,
                frame.get("depth"),
                scene_root / "depth" / f"{frame_id:06d}.npy",
                f"{scene['scene_id']} frame {frame_id} depth",
            )
            record["depth"] = depth_path.as_posix()
        records.append(record)
    episode_id = f"{scene['split_id']}--{scene['scene_id']}--{scene['camera']}"
    if include_window_suffix:
        episode_id = f"{episode_id}--window-{window_index:04d}"
    query_records = records[-query_count:]
    query_vault_id = "qv-" + _digest({"episode_id": episode_id, "query": query_records})[:24]
    online = {
        "episode_id": episode_id,
        "scene_id": scene["scene_id"],
        "split_id": scene["split_id"],
        "query_vault_id": query_vault_id,
        "image_size": image_size,
        "declared_length": warmup_steps + stream_steps,
        "warmup": records[:warmup_steps],
        "stream": records[warmup_steps : warmup_steps + stream_steps],
    }
    query = {
        "episode_id": episode_id,
        "scene_id": scene["scene_id"],
        "split_id": scene["split_id"],
        "query_vault_id": query_vault_id,
        "image_size": image_size,
        "query": query_records,
    }
    scene_inventory = {
        "scene_id": scene["scene_id"],
        "source_split": scene["source_split"],
        "split_id": scene["split_id"],
        "camera": scene["camera"],
        "manifest": scene["manifest_path"].as_posix(),
        "source_manifest_sha256": scene["manifest_sha256"],
        "frame_count": scene["frame_count"],
        "original_frame_count": scene["original_frame_count"],
        "rigid_cameras_only": scene["rigid_cameras_only"],
        "excluded_camera_frames": scene["excluded_camera_frames"],
        "image_size": image_size,
        "selected_frame_ids": [frame["frame_id"] for frame in selected],
        "episode_id": episode_id,
        "window_index": window_index,
        "window_start": start,
        "window_length": total_length,
        "eligible": True,
        "issue": None,
    }
    return online, query, scene_inventory


def _digest(value: Any) -> str:
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _scan_scenes(
    prepared_root: Path,
    source_split: str,
    partitions: dict[str, dict[str, str]],
    expected_count: int | None,
    *,
    rigid_cameras_only: bool = False,
) -> list[dict[str, Any]]:
    split_root = (prepared_root / source_split).resolve()
    if not split_root.is_dir():
        raise FileNotFoundError(f"prepared split directory is missing: {split_root}")
    scene_dirs = sorted(
        (path for path in split_root.iterdir() if path.is_dir()), key=lambda path: path.name
    )
    manifest_paths: list[Path] = []
    for scene_dir in scene_dirs:
        manifest_path = scene_dir / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"missing scene manifest: {manifest_path}")
        manifest_paths.append(manifest_path)
    if expected_count is not None and len(manifest_paths) != expected_count:
        raise ValueError(
            f"expected {expected_count} {source_split} manifests, "
            f"found {len(manifest_paths)}"
        )
    partition_scenes = {
        scene for scene, row in partitions.items() if row["protocol_partition"] == source_split
    }
    scene_names = {path.parent.name for path in manifest_paths}
    if scene_names != partition_scenes:
        missing = sorted(partition_scenes - scene_names)
        extra = sorted(scene_names - partition_scenes)
        raise ValueError(
            f"{source_split} manifest/partition mismatch: "
            f"missing={missing}, extra={extra}"
        )
    return [
        _parse_scene(
            path,
            source_split,
            partitions,
            rigid_cameras_only=rigid_cameras_only,
        )
        for path in manifest_paths
    ]


def _train_depth_health(
    scenes: list[dict[str, Any]],
    *,
    warmup_steps: int,
    stream_steps: int,
    query_count: int,
    train_window_stride: int,
    max_train_windows_per_scene: int | None,
) -> dict[str, Any]:
    total_length = warmup_steps + stream_steps + query_count
    eligible = [scene for scene in scenes if scene["frame_count"] >= total_length]
    total_values = 0
    finite_positive_values = 0
    shape_mismatch_frames = 0
    shape_mismatch_scenes: set[str] = set()
    nan_scenes: set[str] = set()
    all_invalid_scenes: set[str] = set()
    scanned_frames = 0
    episode_count = 0
    for scene in eligible:
        scene_positive = 0
        scene_had_nonfinite = False
        starts = _window_starts(
            scene["frame_count"],
            total_length=total_length,
            stride=train_window_stride,
            max_windows=max_train_windows_per_scene,
        )
        episode_count += len(starts)
        for start in starts:
            for frame in scene["frames"][
                start + warmup_steps + stream_steps : start + total_length
            ]:
                frame_id = frame["frame_id"]
                depth_path = _safe_manifest_reference(
                    scene["manifest_path"].parent,
                    frame.get("depth"),
                    scene["manifest_path"].parent / "depth" / f"{frame_id:06d}.npy",
                    f"{scene['scene_id']} frame {frame_id} depth",
                )
                try:
                    values = np.load(depth_path, allow_pickle=False)
                except Exception as error:
                    raise ValueError(f"cannot load train query depth: {depth_path}") from error
                scanned_frames += 1
                array = np.asarray(values)
                if array.shape != tuple(scene["image_size"]):
                    shape_mismatch_frames += 1
                    shape_mismatch_scenes.add(scene["scene_id"])
                    continue
                finite = np.isfinite(array)
                positive = finite & (array > 0)
                total_values += int(array.size)
                finite_positive_values += int(positive.sum())
                scene_positive += int(positive.sum())
                scene_had_nonfinite |= bool((~finite).any())
        if scene_had_nonfinite:
            nan_scenes.add(scene["scene_id"])
        if scene_positive == 0:
            all_invalid_scenes.add(scene["scene_id"])
    return {
        "scope": "train only, query depth arrays per eligible compiled window",
        "scenes_available": len(scenes),
        "scenes_scanned": len(eligible),
        "episodes_scanned": episode_count,
        "scenes_skipped_for_insufficient_frames": len(scenes) - len(eligible),
        "frames_scanned": scanned_frames,
        "shape_mismatch_frames": shape_mismatch_frames,
        "shape_mismatch_scenes": sorted(shape_mismatch_scenes),
        "nan_scenes": sorted(nan_scenes),
        "all_invalid_scenes": sorted(all_invalid_scenes),
        "total_values": total_values,
        "finite_positive_values": finite_positive_values,
        "finite_positive_ratio": (
            finite_positive_values / total_values if total_values else None
        ),
        "dev_labels_read": False,
        "diagnostic_test_labels_read": False,
        "final_holdout_labels_read": False,
    }


def _scene_inventory(
    scenes: list[dict[str, Any]], compiled: dict[str, list[dict[str, Any]]], *, total_length: int
) -> list[dict[str, Any]]:
    result = []
    for scene in scenes:
        scene_id = scene["scene_id"]
        if scene_id in compiled:
            entries = compiled[scene_id]
            if len(entries) == 1:
                result.append(entries[0])
            else:
                primary = dict(entries[0])
                primary["episodes"] = entries
                primary["episode_count"] = len(entries)
                primary["selected_frame_ids"] = []
                result.append(primary)
            continue
        result.append(
            {
                "scene_id": scene_id,
                "source_split": scene["source_split"],
                "split_id": scene["split_id"],
                "camera": scene["camera"],
                "manifest": scene["manifest_path"].as_posix(),
                "source_manifest_sha256": scene["manifest_sha256"],
                "frame_count": scene["frame_count"],
                "original_frame_count": scene["original_frame_count"],
                "rigid_cameras_only": scene["rigid_cameras_only"],
                "excluded_camera_frames": scene["excluded_camera_frames"],
                "image_size": scene["image_size"],
                "selected_frame_ids": [],
                "episode_id": None,
                "eligible": False,
                "issue": f"insufficient_frames:{scene['frame_count']}<{total_length}",
                "episode_count": 0,
            }
        )
    return result


def _split_summary(entries: list[dict[str, Any]], expected_count: int | None) -> dict[str, Any]:
    eligible_episode_count = sum(
        int(entry.get("episode_count", 1)) for entry in entries if entry["eligible"]
    )
    return {
        "manifest_count": len(entries),
        "expected_manifest_count": expected_count,
        "eligible_episode_count": eligible_episode_count,
        "ineligible_scene_count": sum(not entry["eligible"] for entry in entries),
        "frame_count_distribution": dict(
            sorted(Counter(str(entry["frame_count"]) for entry in entries).items())
        ),
        "image_size_counts": dict(
            sorted(
                Counter(
                    f"{entry['image_size'][0]}x{entry['image_size'][1]}" for entry in entries
                ).items()
            )
        ),
        "camera_counts": dict(sorted(Counter(entry["camera"] for entry in entries).items())),
    }


def _write_json_exclusive(path: Path, value: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def compile_training_episodes(
    *,
    prepared_root: str | Path = DEFAULT_PREPARED_ROOT,
    partitions_path: str | Path = DEFAULT_PARTITIONS,
    output_root: str | Path,
    allow_ineligible: bool = False,
    train_depth_health_scan: bool = True,
    expected_counts: dict[str, int] | None = REQUIRED_COUNTS,
    warmup_steps: int = WARMUP_LENGTH,
    stream_steps: int = STREAM_LENGTH,
    query_count: int = QUERY_LENGTH,
    train_window_stride: int | None = None,
    max_train_windows_per_scene: int | None = 1,
    train_max_windows_per_scene: int | None = None,
    rigid_cameras_only: bool = False,
) -> dict[str, Any]:
    if min(warmup_steps, stream_steps, query_count) < 1:
        raise ValueError("warmup_steps, stream_steps, and query_count must be positive")
    total_length = warmup_steps + stream_steps + query_count
    if train_window_stride is None:
        train_window_stride = total_length
    if train_window_stride < 1:
        raise ValueError("train_window_stride must be positive")
    if train_max_windows_per_scene is not None:
        if max_train_windows_per_scene != 1:
            raise ValueError(
                "pass only one of max_train_windows_per_scene and train_max_windows_per_scene"
            )
        max_train_windows_per_scene = train_max_windows_per_scene
    if max_train_windows_per_scene is not None and max_train_windows_per_scene < 1:
        raise ValueError("max_train_windows_per_scene must be positive or None")
    prepared = Path(prepared_root).resolve()
    partitions = Path(partitions_path).resolve()
    output = Path(output_root).resolve()
    partition_rows, partitions_sha256 = _load_partitions(partitions)
    train_scenes = _scan_scenes(
        prepared,
        "train",
        partition_rows,
        None if expected_counts is None else expected_counts["train"],
        rigid_cameras_only=rigid_cameras_only,
    )
    dev_scenes = _scan_scenes(
        prepared,
        "val",
        partition_rows,
        None if expected_counts is None else expected_counts["val"],
        rigid_cameras_only=rigid_cameras_only,
    )
    train_ids = {scene["scene_id"] for scene in train_scenes}
    dev_ids = {scene["scene_id"] for scene in dev_scenes}
    intersection = sorted(train_ids & dev_ids)
    if intersection:
        raise ValueError(f"train/dev scene intersection is not empty: {intersection}")

    compiled_by_split: dict[str, list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]]] = {
        "train": [],
        "val": [],
    }
    compiled_inventory: dict[str, list[dict[str, Any]]] = {}
    ineligible: list[str] = []
    for source_split, scenes in (("train", train_scenes), ("val", dev_scenes)):
        for scene in scenes:
            if scene["frame_count"] < total_length:
                ineligible.append(scene["scene_id"])
                continue
            if source_split == "train":
                starts = _window_starts(
                    scene["frame_count"],
                    total_length=total_length,
                    stride=train_window_stride,
                    max_windows=max_train_windows_per_scene,
                )
            else:
                # The development selection remains one frozen first window regardless
                # of train sampling settings.
                starts = [0]
            include_suffix = len(starts) > 1 or starts[0] != 0
            scene_entries: list[dict[str, Any]] = []
            for window_index, start in enumerate(starts):
                online, query, inventory = _make_episode(
                    scene,
                    start=start,
                    warmup_steps=warmup_steps,
                    stream_steps=stream_steps,
                    query_count=query_count,
                    window_index=window_index,
                    include_window_suffix=include_suffix,
                )
                compiled_by_split[source_split].append((online, query, inventory))
                scene_entries.append(inventory)
            compiled_inventory[scene["scene_id"]] = scene_entries
    if ineligible and not allow_ineligible:
        raise ValueError(
            f"{total_length}-frame compilation has ineligible scenes; rerun with "
            "--allow-ineligible only "
            f"after accepting the explicit exclusion: {sorted(ineligible)}"
        )

    health = (
        _train_depth_health(
            train_scenes,
            warmup_steps=warmup_steps,
            stream_steps=stream_steps,
            query_count=query_count,
            train_window_stride=train_window_stride,
            max_train_windows_per_scene=max_train_windows_per_scene,
        )
        if train_depth_health_scan
        else {"enabled": False, "dev_labels_read": False, "final_holdout_labels_read": False}
    )
    episodes_dir = output / "episodes"
    inventory_dir = output / "inventory"
    planned_paths = [inventory_dir / "run_inventory.json"]
    for source_split, episodes in compiled_by_split.items():
        split_id = SPLIT_MAPPING[source_split]
        planned_paths.append(episodes_dir / f"episode_index_{split_id}.json")
        for index in range(len(episodes)):
            planned_paths.extend(
                (
                    episodes_dir / f"{split_id}_episode_{index:03d}.online.json",
                    episodes_dir / f"{split_id}_episode_{index:03d}.query.json",
                )
            )
        planned_paths.append(inventory_dir / f"scene_inventory_{split_id}.json")
    if any(path.exists() for path in planned_paths):
        raise FileExistsError("refusing to overwrite dynamic training artifacts")
    episodes_dir.mkdir(parents=True, exist_ok=True)
    inventory_dir.mkdir(parents=True, exist_ok=True)

    index_by_split: dict[str, list[dict[str, Any]]] = {}
    for source_split, episodes in compiled_by_split.items():
        split_id = SPLIT_MAPPING[source_split]
        index: list[dict[str, Any]] = []
        for position, (online, query, _) in enumerate(episodes):
            online_path = episodes_dir / f"{split_id}_episode_{position:03d}.online.json"
            query_path = episodes_dir / f"{split_id}_episode_{position:03d}.query.json"
            _write_json_exclusive(online_path, {"schema_version": ONLINE_SCHEMA, **online})
            _write_json_exclusive(query_path, {"schema_version": QUERY_SCHEMA, **query})
            index.append(
                {
                    "episode_id": online["episode_id"],
                    "scene_id": online["scene_id"],
                    "split_id": online["split_id"],
                    "query_vault_id": online["query_vault_id"],
                    "online_manifest": str(online_path.resolve()),
                    "query_manifest": str(query_path.resolve()),
                    "stream_steps": len(online["stream"]),
                    "query_count": len(query["query"]),
                }
            )
        index_by_split[split_id] = index
        _write_json_exclusive(
            episodes_dir / f"episode_index_{split_id}.json",
            {"schema_version": INDEX_SCHEMA, "episodes": index},
        )

    train_inventory = _scene_inventory(
        train_scenes, compiled_inventory, total_length=total_length
    )
    dev_inventory = _scene_inventory(dev_scenes, compiled_inventory, total_length=total_length)
    _write_json_exclusive(
        inventory_dir / "scene_inventory_train.json",
        {"schema_version": INVENTORY_SCHEMA, "split_id": "train", "scenes": train_inventory},
    )
    _write_json_exclusive(
        inventory_dir / "scene_inventory_dev.json",
        {"schema_version": INVENTORY_SCHEMA, "split_id": "dev", "scenes": dev_inventory},
    )
    inventory = {
        "schema_version": INVENTORY_SCHEMA,
        "status": "complete",
        "prepared_root": prepared.as_posix(),
        "partitions_csv": partitions.as_posix(),
        "partitions_sha256": partitions_sha256,
        "output_root": output.as_posix(),
        "source_manifest_sha256_recorded": True,
        "metadata_only_episode_preflight": True,
        "rgb_files_decoded": False,
        "normal_references_used": False,
        "bounds_used": False,
        "allow_ineligible": allow_ineligible,
        "rigid_cameras_only": rigid_cameras_only,
        "rigid_camera_tolerance": RIGID_CAMERA_TOLERANCE,
        "rigid_camera_exclusions": [
            {
                "scene_id": scene["scene_id"],
                "source_split": scene["source_split"],
                "split_id": scene["split_id"],
                "manifest": scene["manifest_path"].as_posix(),
                "source_manifest_sha256": scene["manifest_sha256"],
                "original_frame_count": scene["original_frame_count"],
                "excluded_frame_ids": [
                    item["frame_id"] for item in scene["excluded_camera_frames"]
                ],
                "excluded_frames": scene["excluded_camera_frames"],
            }
            for scene in [*train_scenes, *dev_scenes]
            if scene["excluded_camera_frames"]
        ],
        "window_spec": {
            "warmup_steps": warmup_steps,
            "stream_steps": stream_steps,
            "query_count": query_count,
            "total_length": total_length,
            "train_window_stride": train_window_stride,
            "max_train_windows_per_scene": max_train_windows_per_scene,
            "dev_window_start": 0,
            "dev_max_windows_per_scene": 1,
        },
        "ineligible_scene_ids": sorted(ineligible),
        "scene_intersection": intersection,
        "splits": {
            "train": _split_summary(
                train_inventory,
                None if expected_counts is None else expected_counts["train"],
            ),
            "dev": _split_summary(
                dev_inventory,
                None if expected_counts is None else expected_counts["val"],
            ),
        },
        "episode_index_train": (episodes_dir / "episode_index_train.json").resolve().as_posix(),
        "episode_index_dev": (episodes_dir / "episode_index_dev.json").resolve().as_posix(),
        "health_scan": health,
    }
    _write_json_exclusive(inventory_dir / "run_inventory.json", inventory)
    return inventory


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-root", type=Path, default=DEFAULT_PREPARED_ROOT)
    parser.add_argument("--partitions", type=Path, default=DEFAULT_PARTITIONS)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--allow-ineligible",
        action="store_true",
        help="write eligible episodes while recording scenes that cannot satisfy 16 frames",
    )
    parser.add_argument(
        "--skip-train-depth-health-scan",
        action="store_true",
        help="skip the train-only query depth health scan",
    )
    parser.add_argument("--warmup-steps", type=int, default=WARMUP_LENGTH)
    parser.add_argument("--stream-steps", type=int, default=STREAM_LENGTH)
    parser.add_argument("--query-count", type=int, default=QUERY_LENGTH)
    parser.add_argument(
        "--train-window-stride",
        type=int,
        help="chronological frame stride for train windows (default: total window length)",
    )
    parser.add_argument(
        "--max-train-windows-per-scene",
        "--train-max-windows-per-scene",
        dest="max_train_windows_per_scene",
        type=int,
        default=1,
        help="maximum train windows per scene; omit for the legacy one-window default",
    )
    parser.add_argument(
        "--all-train-windows",
        action="store_true",
        help="compile every eligible train window at the configured stride",
    )
    parser.add_argument(
        "--rigid-cameras-only",
        action="store_true",
        help=(
            "exclude frames whose metadata c2w rotation is not rigid "
            "(R.T @ R ~= I and det(R) ~= +1)"
        ),
    )
    args = parser.parse_args()
    try:
        max_windows = None if args.all_train_windows else args.max_train_windows_per_scene
        inventory = compile_training_episodes(
            prepared_root=args.prepared_root,
            partitions_path=args.partitions,
            output_root=args.output_root,
            allow_ineligible=args.allow_ineligible,
            train_depth_health_scan=not args.skip_train_depth_health_scan,
            warmup_steps=args.warmup_steps,
            stream_steps=args.stream_steps,
            query_count=args.query_count,
            train_window_stride=args.train_window_stride,
            max_train_windows_per_scene=max_windows,
            rigid_cameras_only=args.rigid_cameras_only,
        )
    except Exception as error:
        print(json.dumps({"status": "failed", "error": str(error)}), flush=True)
        return 1
    print(
        json.dumps(
            {
                "status": inventory["status"],
                "train_episodes": inventory["splits"]["train"]["eligible_episode_count"],
                "dev_episodes": inventory["splits"]["dev"]["eligible_episode_count"],
                "ineligible_scene_ids": inventory["ineligible_scene_ids"],
                "output_root": inventory["output_root"],
            }
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
