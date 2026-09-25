"""Compile a development pilot plan into separate online and sealed-query artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

PROJECT = Path(__file__).resolve().parents[1]
ONLINE_EPISODE_SCHEMA = "mcss.dynamic.online_episode.v1"
QUERY_VAULT_SCHEMA = "mcss.dynamic.query_vault.v1"
EPISODE_INDEX_SCHEMA = "mcss.dynamic.episode_index.v1"
_FORBIDDEN_SPLIT_TOKENS = ("test", "holdout")


def compile_pilot_manifest(
    pilot_manifest_path: str | Path, output_dir: str | Path
) -> list[dict[str, Any]]:
    """Compile development episodes and return the entrypoint index.

    The compiler reads only the pilot JSON and writes reference manifests. It never opens any
    RGB, depth, normal, or query file. The returned index is for the entrypoint only; the online
    source receives an individual ``*.online.json`` file and never receives a query path.
    """

    online_episodes, query_episodes = _compile_episodes(pilot_manifest_path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    index_path = output / "episode_index.json"
    planned_paths = [index_path]
    for index in range(len(online_episodes)):
        planned_paths.extend(
            (
                output / f"episode_{index:03d}.online.json",
                output / f"episode_{index:03d}.query.json",
            )
        )
    if any(path.exists() for path in planned_paths):
        raise FileExistsError("refusing to overwrite compiled episode artifacts")

    index: list[dict[str, Any]] = []
    for position, (online, query) in enumerate(zip(online_episodes, query_episodes, strict=True)):
        online_path = output / f"episode_{position:03d}.online.json"
        query_path = output / f"episode_{position:03d}.query.json"
        _write_json_exclusive(online_path, {"schema_version": ONLINE_EPISODE_SCHEMA, **online})
        _write_json_exclusive(query_path, {"schema_version": QUERY_VAULT_SCHEMA, **query})
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
    _write_json_exclusive(index_path, {"schema_version": EPISODE_INDEX_SCHEMA, "episodes": index})
    return index


def _compile_episodes(
    pilot_manifest_path: str | Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build pure references without opening any RGB, depth, normal, or query file."""

    source = Path(pilot_manifest_path)
    try:
        pilot = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid pilot manifest JSON: {source}") from error
    if not isinstance(pilot, dict) or not isinstance(pilot.get("episodes"), list):
        raise ValueError("pilot manifest requires an episodes list")
    if pilot.get("final_holdout_used") is True or pilot.get("diagnostic_test_used") is True:
        raise ValueError(
            "development episode compiler refuses test or final-holdout pilot manifests"
        )

    online_episodes: list[dict[str, Any]] = []
    query_episodes: list[dict[str, Any]] = []
    seen_episode_ids: set[str] = set()
    seen_scene_ids: set[str] = set()
    for source_episode in pilot["episodes"]:
        online, query = _compile_episode(source_episode)
        if online["episode_id"] in seen_episode_ids or online["scene_id"] in seen_scene_ids:
            raise ValueError("pilot compiler requires unique episode_id and scene_id values")
        seen_episode_ids.add(online["episode_id"])
        seen_scene_ids.add(online["scene_id"])
        online_episodes.append(online)
        query_episodes.append(query)
    if not online_episodes:
        raise ValueError("pilot manifest contains no episodes")
    return online_episodes, query_episodes


def _compile_episode(source: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    if not isinstance(source, dict):
        raise ValueError("pilot episode entries must be objects")
    scene_id = _identifier(source.get("scene_id"), "scene_id")
    split_id = _identifier(source.get("split"), "split")
    if any(token in split_id.lower() for token in _FORBIDDEN_SPLIT_TOKENS):
        raise ValueError(f"development compiler refuses split: {split_id}")
    camera = _identifier(source.get("camera"), "camera")
    image_size = _image_size(source.get("image_size"))
    episode_id = f"{split_id}--{scene_id}--{camera}"
    vault_id = f"qv-{_digest({'episode_id': episode_id, 'query': source.get('query')})[:24]}"
    warmup = _online_records(source.get("warmup"), "warmup")
    stream = _online_records(source.get("stream"), "stream")
    query = _query_records(source.get("query"))
    all_ids = [record["frame_id"] for record in (*warmup, *stream, *query)]
    if not warmup or not stream or not query:
        raise ValueError("pilot episode requires non-empty warmup, stream, and query segments")
    if len(all_ids) != len(set(all_ids)) or all_ids != sorted(all_ids):
        raise ValueError("pilot frame_ids must be unique and in chronological order")
    online = {
        "episode_id": episode_id,
        "scene_id": scene_id,
        "split_id": split_id,
        "query_vault_id": vault_id,
        "image_size": image_size,
        "declared_length": len(warmup) + len(stream),
        "warmup": warmup,
        "stream": stream,
    }
    vault = {
        "episode_id": episode_id,
        "scene_id": scene_id,
        "split_id": split_id,
        "query_vault_id": vault_id,
        "image_size": image_size,
        "query": query,
    }
    return online, vault


def _online_records(value: Any, segment: str) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise ValueError(f"pilot {segment} must be a list")
    return [_reference_record(record, include_depth=False) for record in value]


def _query_records(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise ValueError("pilot query must be a list")
    return [_reference_record(record, include_depth=True) for record in value]


def _reference_record(value: Any, *, include_depth: bool) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("pilot frame records must be objects")
    frame_id = value.get("frame_id")
    if type(frame_id) is not int or frame_id < 0:
        raise ValueError("pilot frame_id must be a non-negative integer")
    result = {
        "frame_id": frame_id,
        "intrinsics": _matrix(value.get("intrinsics"), (3, 3), "intrinsics"),
        "c2w": _matrix(value.get("c2w"), (4, 4), "c2w"),
        "rgb": _path_reference(value.get("rgb"), "rgb"),
    }
    if include_depth:
        result["depth"] = _path_reference(value.get("depth"), "depth")
    return result


def _matrix(value: Any, shape: tuple[int, int], name: str) -> list[list[float]]:
    if not isinstance(value, list) or len(value) != shape[0]:
        raise ValueError(f"pilot {name} must be a {shape[0]}x{shape[1]} matrix")
    matrix: list[list[float]] = []
    for row in value:
        if not isinstance(row, list) or len(row) != shape[1]:
            raise ValueError(f"pilot {name} must be a {shape[0]}x{shape[1]} matrix")
        converted: list[float] = []
        for item in row:
            if type(item) not in {int, float}:
                raise ValueError(f"pilot {name} must contain numeric values")
            converted.append(float(item))
        matrix.append(converted)
    if not all(math.isfinite(item) for row in matrix for item in row):
        raise ValueError(f"pilot {name} must contain finite values")
    if name == "intrinsics" and (matrix[0][0] <= 0 or matrix[1][1] <= 0):
        raise ValueError("pilot intrinsics has non-positive focal length")
    if shape == (4, 4) and matrix[3] != [0.0, 0.0, 0.0, 1.0]:
        raise ValueError("pilot c2w must have homogeneous final row")
    return matrix


def _path_reference(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"pilot {name} must be a non-empty reference")
    return value


def _image_size(value: Any) -> list[int]:
    if (
        not isinstance(value, list)
        or len(value) != 2
        or any(type(item) is not int or item < 2 for item in value)
    ):
        raise ValueError("pilot image_size must be [height, width]")
    return list(value)


def _identifier(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value or any(char in value for char in "\\/"):
        raise ValueError(f"pilot {name} must be a simple non-empty identifier")
    return value


def _digest(value: Any) -> str:
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _write_json_exclusive(path: Path, value: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pilot-manifest",
        type=Path,
        default=PROJECT / "outputs/experiment_preflight_20260918/pilot_manifest.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT / "outputs/dynamic_ttt_pilot_20260919/episodes",
    )
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.relative_to(PROJECT / "outputs")
    index = compile_pilot_manifest(args.pilot_manifest, output_dir)
    print(
        json.dumps(
            {
                "episode_index": str(output_dir / "episode_index.json"),
                "episodes": len(index),
            }
        )
    )


if __name__ == "__main__":
    main()
