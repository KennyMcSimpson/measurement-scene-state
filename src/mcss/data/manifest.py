"""Versioned on-disk scene manifests independent of source dataset layout."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ManifestFrame:
    frame_id: int
    rgb: str
    intrinsics: list[list[float]]
    c2w: list[list[float]]
    depth: str | None = None
    normal: str | None = None
    point: str | None = None
    visibility: str | None = None

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "frame_id": self.frame_id,
            "rgb": self.rgb,
            "intrinsics": self.intrinsics,
            "c2w": self.c2w,
        }
        for name in ("depth", "normal", "point", "visibility"):
            value = getattr(self, name)
            if value is not None:
                result[name] = value
        return result


@dataclass(frozen=True)
class SceneManifest:
    scene_id: str
    image_size: tuple[int, int]
    bounds: tuple[tuple[float, float, float], tuple[float, float, float]]
    frames: tuple[ManifestFrame, ...]
    root: Path | None = None
    coordinate_convention: str = "opencv_c2w"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "scene_id": self.scene_id,
            "coordinate_convention": self.coordinate_convention,
            "image_size": list(self.image_size),
            "bounds": [list(self.bounds[0]), list(self.bounds[1])],
            "frames": [frame.to_dict() for frame in self.frames],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], root: str | Path) -> SceneManifest:
        if data.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"expected manifest schema_version {SCHEMA_VERSION}")
        scene_id = data.get("scene_id")
        if not isinstance(scene_id, str) or not scene_id or any(char in scene_id for char in "\\/"):
            raise ValueError("scene_id must be a simple non-empty name")
        if data.get("coordinate_convention") != "opencv_c2w":
            raise ValueError("manifest coordinate_convention must be opencv_c2w")
        image_size = _parse_image_size(data.get("image_size"))
        bounds = _parse_bounds(data.get("bounds"))
        raw_frames = data.get("frames")
        if not isinstance(raw_frames, list) or not raw_frames:
            raise ValueError("manifest frames must be a non-empty list")
        root_path = Path(root).resolve()
        frames = tuple(_parse_frame(frame, root_path) for frame in raw_frames)
        frame_ids = [frame.frame_id for frame in frames]
        if len(set(frame_ids)) != len(frame_ids):
            raise ValueError("manifest frame_id values must be unique")
        return cls(
            scene_id=scene_id,
            image_size=image_size,
            bounds=bounds,
            frames=frames,
            root=root_path,
        )


def save_manifest(manifest: SceneManifest, path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return output


def load_manifest(path: str | Path) -> SceneManifest:
    manifest_path = Path(path)
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid manifest JSON: {manifest_path}") from error
    if not isinstance(data, dict):
        raise ValueError("manifest JSON root must be an object")
    return SceneManifest.from_dict(data, manifest_path.parent)


def manifest_relative_path(root: str | Path, value: str) -> Path:
    """Resolve a manifest reference while forbidding absolute and escaping paths."""

    if not isinstance(value, str) or not value:
        raise ValueError("manifest paths must be non-empty strings")
    normalized = value.replace("\\", "/")
    pure = PurePosixPath(normalized)
    if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
        raise ValueError("manifest paths must be safe relative paths")
    if len(pure.parts[0]) >= 2 and pure.parts[0][1] == ":":
        raise ValueError("manifest paths must be relative, not drive paths")
    root_path = Path(root).resolve()
    result = (root_path / Path(*pure.parts)).resolve()
    try:
        result.relative_to(root_path)
    except ValueError as error:
        raise ValueError("manifest paths must be safe relative paths") from error
    return result


def _parse_frame(data: Any, root: Path) -> ManifestFrame:
    if not isinstance(data, dict):
        raise ValueError("every manifest frame must be an object")
    frame_id = data.get("frame_id")
    if not isinstance(frame_id, int) or frame_id < 0:
        raise ValueError("frame_id must be a non-negative integer")
    rgb = _parse_path(data.get("rgb"), root)
    optional = {
        name: _parse_optional_path(data.get(name), root)
        for name in ("depth", "normal", "point", "visibility")
    }
    intrinsics = _parse_matrix(data.get("intrinsics"), (3, 3), "intrinsics")
    c2w = _parse_matrix(data.get("c2w"), (4, 4), "c2w")
    if not np.allclose(np.asarray(c2w)[3], [0.0, 0.0, 0.0, 1.0], atol=1e-5):
        raise ValueError("c2w must have a homogeneous final row")
    return ManifestFrame(frame_id, rgb, intrinsics, c2w, **optional)


def _parse_path(value: Any, root: Path) -> str:
    manifest_relative_path(root, value)
    return value.replace("\\", "/")


def _parse_optional_path(value: Any, root: Path) -> str | None:
    if value is None:
        return None
    return _parse_path(value, root)


def _parse_matrix(value: Any, shape: tuple[int, int], name: str) -> list[list[float]]:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite {shape[0]}x{shape[1]} matrix")
    return array.astype(float).tolist()


def _parse_image_size(value: Any) -> tuple[int, int]:
    if (
        not isinstance(value, list)
        or len(value) != 2
        or not all(isinstance(item, int) for item in value)
    ):
        raise ValueError("image_size must be [height, width]")
    height, width = value
    if height < 2 or width < 2:
        raise ValueError("image_size values must be at least two")
    return height, width


def _parse_bounds(value: Any) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (2, 3) or not np.isfinite(array).all() or not np.all(array[1] > array[0]):
        raise ValueError("bounds must be finite [[minx, miny, minz], [maxx, maxy, maxz]]")
    return tuple(map(tuple, array.astype(float)))  # type: ignore[return-value]
