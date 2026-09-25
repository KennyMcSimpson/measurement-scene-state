"""Metadata and lazy RGB access for the pinned DL3DV-140 protocol.

The benchmark has two deliberately separate surfaces:

* :func:`prepare_benchmark_metadata` reads only ``transforms.json`` files and
  the pinned Long-LRM split.  It never opens an image.
* :class:`Dl3dvOnlineObservationSource` opens one image when that observation
  arrives.  Query images are exposed through callbacks only after a
  :class:`~mcss.dynamic.types.SealedScene` has been supplied.

The coordinate convention and image processing follow the pinned Long-LRM/
tttLRM converter.  The online gauge is intentionally different from tttLRM:
it is a frozen warmup-only baseline normalization, because query and future
poses are not available to a causal episode.
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from PIL import Image

from mcss.dynamic.types import OnlineObservation, SealedScene
from mcss.evaluation.sealed_rgb import RGBQuery
from mcss.types import Cameras

DL3DV_SCENE_COUNT = 140
DL3DV_TARGET_STRIDE = 8
DL3DV_WARMUP_LENGTH = 4
DL3DV_SOURCE_IMAGE_SIZE = (540, 960)  # H, W; images_4
DL3DV_IMAGES4_DOWNSAMPLE = 4
DL3DV_EVAL_IMAGE_SIZE = (536, 960)  # H, W; tttLRM evaluation config
DL3DV_WARMUP_TARGET_EXTENT_M = 1.9428480059945201
DL3DV_MIN_WARMUP_BASELINE = 1.0e-6
DL3DV_LOCAL_BOUNDS_M = ((-6.0, -4.0, 0.05), (6.0, 4.0, 12.05))
DL3DV_METADATA_SCHEMA = "mcss.dl3dv.benchmark_metadata.v1"
DL3DV_GAUGE_SCHEMA = "mcss.dl3dv.warmup_gauge.v1"
# These identify the public Long-LRM split, rather than the separate DL3DV
# dataset revision used for the RGB files.
DL3DV_SPLIT_REVISION = "0e89431a964b8a7e2fdec24e3ca5ec74a26dcafc"
DL3DV_SPLIT_FILE_SHA1 = "ae0ddcd09b64753210c46978124b926dd223fa98"
# Ordinary SHA-256 of the pinned split bytes.  The 40-character value above is
# the Git blob SHA-1 and must not be used as a SHA-256 substitute.
DL3DV_SPLIT_SHA256 = "4b052aef605fda183683175de66250aaaf337e50003131dbf0113ea781a3145a"
DL3DV_RIGID_ROTATION_TOLERANCE = 1e-3
DL3DV_GAUGE_CALIBRATION = "warmup_gauge_calibration_v2.json"
DL3DV_GAUGE_VALID_WINDOWS = 332
DL3DV_GAUGE_EXCLUDED_WINDOWS = 31

ProtocolMode = Literal["full", "ar"]
_FULL_INPUT_COUNTS = frozenset({16, 32, 64})
_AR_INPUT_COUNTS = frozenset({32, 64})


class Dl3dvProtocolError(ValueError):
    """The pinned split or online episode violates the DL3DV protocol."""


def _as_float_tuple(values: Sequence[Any], length: int, name: str) -> tuple[float, ...]:
    if not isinstance(values, Sequence) or len(values) != length:
        raise Dl3dvProtocolError(f"{name} must contain exactly {length} values")
    result = tuple(float(value) for value in values)
    if not all(math.isfinite(value) for value in result):
        raise Dl3dvProtocolError(f"{name} must contain finite values")
    return result


def _validate_homogeneous(matrix: np.ndarray, name: str) -> np.ndarray:
    value = np.asarray(matrix, dtype=np.float64)
    if value.shape != (4, 4) or not np.isfinite(value).all():
        raise Dl3dvProtocolError(f"{name} must be a finite 4x4 matrix")
    if not np.allclose(value[3], (0.0, 0.0, 0.0, 1.0), atol=1e-6, rtol=1e-6):
        raise Dl3dvProtocolError(f"{name} must have homogeneous final row [0,0,0,1]")
    try:
        np.linalg.inv(value)
    except np.linalg.LinAlgError as error:
        raise Dl3dvProtocolError(f"{name} must be invertible") from error
    return value


def _validate_rigid_camera(matrix: np.ndarray, name: str) -> np.ndarray:
    """Validate a camera pose as a proper rigid transform in ``SE(3)``."""

    value = _validate_homogeneous(matrix, name)
    rotation = value[:3, :3]
    if not np.allclose(
        rotation.T @ rotation,
        np.eye(3, dtype=np.float64),
        atol=DL3DV_RIGID_ROTATION_TOLERANCE,
        rtol=DL3DV_RIGID_ROTATION_TOLERANCE,
    ):
        raise Dl3dvProtocolError(
            f"{name} rotation is not orthonormal within "
            f"{DL3DV_RIGID_ROTATION_TOLERANCE:g}"
        )
    determinant = float(np.linalg.det(rotation))
    if not math.isclose(
        determinant,
        1.0,
        abs_tol=DL3DV_RIGID_ROTATION_TOLERANCE,
        rel_tol=DL3DV_RIGID_ROTATION_TOLERANCE,
    ):
        raise Dl3dvProtocolError(
            f"{name} rotation must have determinant +1, got {determinant:.6g}"
        )
    return value


def official_pose_transform(transform_matrix: Sequence[Sequence[float]]) -> np.ndarray:
    """Apply the exact c2w transform used by Long-LRM's DL3DV converter."""

    c2w = _validate_homogeneous(np.asarray(transform_matrix), "transform_matrix").copy()
    c2w[0:3, 1:3] *= -1.0
    c2w = c2w[[1, 0, 2, 3], :]
    c2w[2, :] *= -1.0
    return _validate_homogeneous(c2w, "converted c2w")


def official_intrinsics(
    fxfycxcy: Sequence[float], *, source_size: tuple[int, int]
) -> np.ndarray:
    """Build the raw OpenCV intrinsic matrix from ``fx, fy, cx, cy``."""

    values = _as_float_tuple(fxfycxcy, 4, "fxfycxcy")
    height, width = source_size
    if height < 2 or width < 2:
        raise Dl3dvProtocolError("source image size must be at least 2x2")
    fx, fy, cx, cy = values
    if fx <= 0.0 or fy <= 0.0:
        raise Dl3dvProtocolError("source focal lengths must be positive")
    return np.asarray([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)


def _cv2() -> Any:
    try:
        import cv2
    except ImportError as error:  # pragma: no cover - environment dependent
        raise RuntimeError("opencv-python is required for DL3DV undistortion") from error
    return cv2


def undistorted_intrinsics(
    raw_intrinsics: np.ndarray,
    distortion: Sequence[float],
    *,
    source_size: tuple[int, int],
) -> np.ndarray:
    """Return the converter's ``getOptimalNewCameraMatrix(alpha=0)`` result.

    This operation depends only on camera metadata and image dimensions.  It
    does not read any image pixels, so it is safe during metadata preparation.
    """

    height, width = source_size
    if height < 2 or width < 2:
        raise Dl3dvProtocolError("source image size must be at least 2x2")
    cv2 = _cv2()
    distortion_array = np.asarray(_as_float_tuple(distortion, 4, "distortion"), dtype=np.float64)
    matrix, _ = cv2.getOptimalNewCameraMatrix(
        np.asarray(raw_intrinsics, dtype=np.float64),
        distortion_array,
        (width, height),
        0,
        (width, height),
    )
    result = np.asarray(matrix, dtype=np.float64)
    if result.shape != (3, 3) or not np.isfinite(result).all():
        raise Dl3dvProtocolError("undistorted intrinsic matrix is invalid")
    return result


def resize_and_center_crop(
    image: Image.Image,
    target_size: tuple[int, int] = DL3DV_EVAL_IMAGE_SIZE,
    fxfycxcy: Sequence[float] | None = None,
) -> tuple[Image.Image, tuple[float, float, float, float] | None]:
    """Match tttLRM's literal anisotropic resize/crop implementation.

    The public notes describe this as resizing to cover.  The pinned source
    actually uses independent ``scale_x`` and ``scale_y`` values, rounds each
    resized dimension, and then center-crops.  This function preserves that
    behavior and applies the same transform to intrinsics.
    """

    target_height, target_width = target_size
    if target_height < 2 or target_width < 2:
        raise Dl3dvProtocolError("target image size must be at least 2x2")
    original_width, original_height = image.size
    if original_width < 2 or original_height < 2:
        raise Dl3dvProtocolError("source image must be at least 2x2")
    scale_x = target_width / original_width
    scale_y = target_height / original_height
    new_width = int(round(original_width * scale_x))
    new_height = int(round(original_height * scale_y))
    resized = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    left = (new_width - target_width) // 2
    top = (new_height - target_height) // 2
    cropped = resized.crop((left, top, left + target_width, top + target_height)).convert("RGB")
    if fxfycxcy is None:
        return cropped, None
    fx, fy, cx, cy = _as_float_tuple(fxfycxcy, 4, "fxfycxcy")
    return cropped, (
        fx * scale_x,
        fy * scale_y,
        cx * scale_x - left,
        cy * scale_y - top,
    )


def _codec_roundtrip(image_bgr: np.ndarray, suffix: str) -> np.ndarray:
    """Emulate converter ``cv2.imwrite`` followed by dataset PIL loading."""

    cv2 = _cv2()
    extension = suffix.lower() or ".png"
    if extension not in {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}:
        raise Dl3dvProtocolError(f"unsupported DL3DV image extension: {suffix!r}")
    ok, encoded = cv2.imencode(extension, image_bgr)
    if not ok:
        raise Dl3dvProtocolError(f"OpenCV could not encode undistorted image as {extension}")
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if decoded is None:
        raise Dl3dvProtocolError("OpenCV could not decode its undistorted image round-trip")
    return decoded


def preprocess_dl3dv_rgb(
    image_path: str | Path,
    frame: Dl3dvFrame,
    *,
    target_size: tuple[int, int] = DL3DV_EVAL_IMAGE_SIZE,
) -> tuple[torch.Tensor, np.ndarray]:
    """Load and preprocess one frame using the official converter/data path.

    The returned tensor is channel-first RGB ``float32`` in ``[0, 1]`` and the
    returned intrinsic matrix matches the resized/cropped image.  This function
    is intentionally called only by an observation or query callback.
    """

    cv2 = _cv2()
    source = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if source is None:
        raise FileNotFoundError(f"DL3DV RGB image could not be read: {image_path}")
    height, width = int(source.shape[0]), int(source.shape[1])
    expected_height, expected_width = frame.source_image_size
    if (height, width) != (expected_height, expected_width):
        raise Dl3dvProtocolError(
            f"DL3DV source image size differs from transforms.json: {(height, width)} != "
            f"{(expected_height, expected_width)}"
        )
    raw_intrinsics = frame.raw_intrinsics_matrix
    distortion = np.asarray(frame.distortion, dtype=np.float64)
    new_intrinsics, _ = cv2.getOptimalNewCameraMatrix(
        raw_intrinsics, distortion, (width, height), 0, (width, height)
    )
    undistorted = cv2.undistort(source, raw_intrinsics, distortion, None, new_intrinsics)
    # The converter writes the undistorted BGR array using the original suffix,
    # after which tttLRM opens the resulting file with PIL.  Reproduce that
    # codec boundary so JPEG scenes retain the same small loss as the official path.
    roundtripped = _codec_roundtrip(undistorted, Path(frame.rgb_relpath).suffix)
    rgb = Image.fromarray(cv2.cvtColor(roundtripped, cv2.COLOR_BGR2RGB), mode="RGB")
    cropped, fxfycxcy = resize_and_center_crop(
        rgb,
        target_size,
        (
            float(new_intrinsics[0, 0]),
            float(new_intrinsics[1, 1]),
            float(new_intrinsics[0, 2]),
            float(new_intrinsics[1, 2]),
        ),
    )
    assert fxfycxcy is not None
    array = np.asarray(cropped, dtype=np.float32).copy()
    tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous().div(255.0)
    intrinsic = official_intrinsics(fxfycxcy, source_size=target_size)
    return tensor, intrinsic


@dataclass(frozen=True)
class Dl3dvProtocol:
    """One supported Long-LRM input selection condition."""

    mode: ProtocolMode
    input_count: int
    fold_size: int = DL3DV_TARGET_STRIDE
    warmup_length: int = DL3DV_WARMUP_LENGTH

    def __post_init__(self) -> None:
        if self.mode not in ("full", "ar"):
            raise Dl3dvProtocolError("DL3DV mode must be 'full' or 'ar'")
        allowed = _FULL_INPUT_COUNTS if self.mode == "full" else _AR_INPUT_COUNTS
        if self.input_count not in allowed:
            raise Dl3dvProtocolError(
                f"{self.mode} mode supports input counts {sorted(allowed)}, got {self.input_count}"
            )
        if self.fold_size != DL3DV_TARGET_STRIDE or self.fold_size < 1:
            raise Dl3dvProtocolError("the pinned DL3DV protocol uses fold_size=8")
        if self.warmup_length != DL3DV_WARMUP_LENGTH:
            raise Dl3dvProtocolError("the causal DL3DV adapter requires four warmup views")

    @property
    def split_key(self) -> str:
        return f"fold_{self.fold_size}_kmeans_{self.input_count}_input"

    @property
    def name(self) -> str:
        return f"{self.mode}{self.input_count}"


@dataclass(frozen=True)
class LongLRMSplitRecord:
    scene_name: str
    input_indices_by_count: Mapping[int, tuple[int, ...]]

    def input_indices(self, input_count: int) -> tuple[int, ...]:
        try:
            return self.input_indices_by_count[input_count]
        except KeyError as error:
            raise Dl3dvProtocolError(
                f"Long-LRM split has no input array for N={input_count}: {self.scene_name}"
            ) from error

    def to_dict(self) -> dict[str, Any]:
        return {
            "scene_name": self.scene_name,
            **{
                f"fold_8_kmeans_{count}_input": list(indices)
                for count, indices in sorted(self.input_indices_by_count.items())
            },
        }


def load_long_lrm_split(
    path: str | Path,
    *,
    expected_scene_count: int = DL3DV_SCENE_COUNT,
    expected_sha256: str | None = DL3DV_SPLIT_SHA256,
) -> tuple[LongLRMSplitRecord, ...]:
    """Read, authenticate, and structurally validate the Long-LRM split JSON.

    The default path is intentionally pinned to the public split bytes used by
    this benchmark.  ``expected_sha256=None`` is available only for isolated
    structural fixtures; production metadata preparation keeps the default.
    """

    source = Path(path)
    split_bytes = source.read_bytes()
    actual_sha256 = hashlib.sha256(split_bytes).hexdigest()
    if expected_sha256 is not None and actual_sha256 != expected_sha256:
        raise Dl3dvProtocolError(
            "Long-LRM split SHA256 does not match the pinned revision: "
            f"expected={expected_sha256}, actual={actual_sha256}"
        )
    try:
        value = json.loads(split_bytes.decode("utf-8"))
    except json.JSONDecodeError as error:
        raise Dl3dvProtocolError(f"invalid Long-LRM split JSON: {source}") from error
    if not isinstance(value, list):
        raise Dl3dvProtocolError("Long-LRM split root must be a list")
    if expected_scene_count is not None and len(value) != expected_scene_count:
        raise Dl3dvProtocolError(
            f"expected {expected_scene_count} Long-LRM scenes, got {len(value)}"
        )
    records: list[LongLRMSplitRecord] = []
    seen: set[str] = set()
    for row_index, row in enumerate(value):
        if not isinstance(row, Mapping):
            raise Dl3dvProtocolError(f"Long-LRM split row {row_index} must be an object")
        scene_name = row.get("scene_name")
        if not isinstance(scene_name, str) or not scene_name or any(
            char in scene_name for char in "\\/"
        ):
            raise Dl3dvProtocolError(f"invalid scene_name at Long-LRM row {row_index}")
        if scene_name in seen:
            raise Dl3dvProtocolError(f"duplicate Long-LRM scene_name: {scene_name}")
        seen.add(scene_name)
        arrays: dict[int, tuple[int, ...]] = {}
        for count in (16, 32, 64, 128):
            key = f"fold_8_kmeans_{count}_input"
            raw_indices = row.get(key)
            if not isinstance(raw_indices, list):
                raise Dl3dvProtocolError(f"{scene_name} is missing {key}")
            indices = tuple(raw_indices)
            if any(type(index) is not int or index < 0 for index in indices):
                raise Dl3dvProtocolError(f"{scene_name} has invalid indices in {key}")
            if len(indices) != count or len(set(indices)) != count:
                raise Dl3dvProtocolError(f"{scene_name} has invalid length or duplicates in {key}")
            arrays[count] = indices
        records.append(LongLRMSplitRecord(scene_name, arrays))
    return tuple(records)


def target_frame_indices(frame_count: int, stride: int = DL3DV_TARGET_STRIDE) -> tuple[int, ...]:
    """Return the exact every-eighth target order used by tttLRM."""

    if type(frame_count) is not int or frame_count < 1:
        raise Dl3dvProtocolError("frame_count must be a positive integer")
    if type(stride) is not int or stride < 1:
        raise Dl3dvProtocolError("target stride must be a positive integer")
    return tuple(range(0, frame_count, stride))


def _validate_scene_indices(
    frame_count: int,
    input_indices: Sequence[int],
    *,
    target_stride: int,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    targets = target_frame_indices(frame_count, target_stride)
    target_set = set(targets)
    inputs = tuple(input_indices)
    if len(set(inputs)) != len(inputs):
        raise Dl3dvProtocolError("DL3DV input indices must be unique")
    if any(type(index) is not int or index < 0 or index >= frame_count for index in inputs):
        raise Dl3dvProtocolError("DL3DV input indices must be in the scene frame range")
    overlap = sorted(target_set.intersection(inputs))
    if overlap:
        raise Dl3dvProtocolError(f"DL3DV context/query overlap at frame {overlap[0]}")
    available = frame_count - len(targets)
    if len(inputs) > available:
        raise Dl3dvProtocolError(
            f"DL3DV scene has {available} non-target frames but needs {len(inputs)} inputs"
        )
    return inputs, targets


def validate_complete_coverage(
    scenes: Sequence[Dl3dvSceneMetadata],
    *,
    expected_scene_count: int = DL3DV_SCENE_COUNT,
    expected_scene_names: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Validate that metadata covers the complete benchmark exactly once."""

    if len(scenes) != expected_scene_count:
        raise Dl3dvProtocolError(
            f"expected complete DL3DV coverage of {expected_scene_count} scenes, got {len(scenes)}"
        )
    names = [scene.scene_id for scene in scenes]
    if len(names) != len(set(names)):
        raise Dl3dvProtocolError("DL3DV metadata contains duplicate scene IDs")
    if expected_scene_names is not None:
        expected = set(expected_scene_names)
        actual = set(names)
        if actual != expected:
            missing = sorted(expected - actual)
            extra = sorted(actual - expected)
            raise Dl3dvProtocolError(
                "DL3DV scene coverage differs from pinned split; "
                f"missing={missing[:3]}, extra={extra[:3]}"
            )
    return {
        "scene_count": len(scenes),
        "scene_ids": tuple(names),
        "target_stride": scenes[0].protocol.fold_size if scenes else DL3DV_TARGET_STRIDE,
        "query_counts": {scene.scene_id: len(scene.query_indices) for scene in scenes},
    }


@dataclass(frozen=True)
class Dl3dvFrame:
    scene_id: str
    frame_id: int
    rgb_relpath: str
    c2w: tuple[tuple[float, ...], ...]
    raw_fxfycxcy: tuple[float, float, float, float]
    distortion: tuple[float, float, float, float]
    source_image_size: tuple[int, int]

    def __post_init__(self) -> None:
        if not isinstance(self.scene_id, str) or not self.scene_id or any(
            char in self.scene_id for char in "\\/"
        ):
            raise Dl3dvProtocolError("DL3DV frame scene_id must be a simple identifier")
        if type(self.frame_id) is not int or self.frame_id < 0:
            raise Dl3dvProtocolError("DL3DV frame_id must be a non-negative integer")
        _frame_path(self.rgb_relpath)
        source_size = self.source_image_size
        if (
            not isinstance(source_size, tuple)
            or len(source_size) != 2
            or any(type(value) is not int or value < 2 for value in source_size)
        ):
            raise Dl3dvProtocolError(
                "DL3DV source_image_size must be a tuple of integers at least two"
            )
        _validate_rigid_camera(np.asarray(self.c2w, dtype=np.float64), "frame c2w")
        _as_float_tuple(self.raw_fxfycxcy, 4, "raw_fxfycxcy")
        _as_float_tuple(self.distortion, 4, "distortion")

    @property
    def raw_intrinsics_matrix(self) -> np.ndarray:
        return official_intrinsics(self.raw_fxfycxcy, source_size=self.source_image_size)

    @property
    def c2w_array(self) -> np.ndarray:
        return _validate_rigid_camera(np.asarray(self.c2w, dtype=np.float64), "frame c2w")

    def to_dict(self) -> dict[str, Any]:
        return {
            "scene_id": self.scene_id,
            "frame_id": self.frame_id,
            "rgb_relpath": self.rgb_relpath,
            "c2w": [list(row) for row in self.c2w],
            "raw_fxfycxcy": list(self.raw_fxfycxcy),
            "distortion": list(self.distortion),
            "source_image_size": list(self.source_image_size),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Dl3dvFrame:
        return cls(
            str(value["scene_id"]),
            int(value["frame_id"]),
            _frame_path(value["rgb_relpath"]),
            tuple(tuple(float(item) for item in row) for row in value["c2w"]),
            tuple(_as_float_tuple(value["raw_fxfycxcy"], 4, "raw_fxfycxcy")),
            tuple(_as_float_tuple(value["distortion"], 4, "distortion")),
            tuple(int(item) for item in value["source_image_size"]),
        )


@dataclass(frozen=True)
class Dl3dvSceneMetadata:
    scene_id: str
    scene_root: Path
    frames: tuple[Dl3dvFrame, ...]
    input_indices: tuple[int, ...]
    query_indices: tuple[int, ...]
    protocol: Dl3dvProtocol
    split_id: str = "test"
    target_size: tuple[int, int] = DL3DV_EVAL_IMAGE_SIZE
    source_schema: str = "Long-LRM/tttLRM transforms.json"

    def __post_init__(self) -> None:
        if not self.scene_id or any(char in self.scene_id for char in "\\/"):
            raise Dl3dvProtocolError("DL3DV scene_id must be a simple non-empty identifier")
        if not isinstance(self.protocol, Dl3dvProtocol):
            raise Dl3dvProtocolError("DL3DV scene metadata requires a Dl3dvProtocol")
        if not self.frames:
            raise Dl3dvProtocolError("DL3DV scene metadata requires at least one frame")
        if self.source_schema != "Long-LRM/tttLRM transforms.json":
            raise Dl3dvProtocolError(
                "DL3DV metadata source_schema is not the pinned transforms format"
            )
        rgb_paths = [frame.rgb_relpath for frame in self.frames]
        if any(
            not isinstance(path, str)
            or not path.startswith("images_4/")
            or Path(path).is_absolute()
            or ".." in Path(path).parts
            for path in rgb_paths
        ):
            raise Dl3dvProtocolError("DL3DV RGB paths must stay under images_4")
        if len(rgb_paths) != len(set(rgb_paths)):
            raise Dl3dvProtocolError("DL3DV frames must not alias RGB paths")
        for expected_id, frame in enumerate(self.frames):
            if frame.scene_id != self.scene_id or frame.frame_id != expected_id:
                raise Dl3dvProtocolError("DL3DV frame IDs and scene IDs must follow source order")
            if frame.source_image_size != self.frames[0].source_image_size:
                raise Dl3dvProtocolError("DL3DV frames must share one source image size")
            _validate_rigid_camera(np.asarray(frame.c2w, dtype=np.float64), "frame c2w")
        if (
            not isinstance(self.target_size, tuple)
            or len(self.target_size) != 2
            or any(type(value) is not int or value < 2 for value in self.target_size)
        ):
            raise Dl3dvProtocolError("DL3DV target_size must be a tuple of integers at least two")
        if len(self.input_indices) != self.protocol.input_count:
            raise Dl3dvProtocolError(
                "DL3DV metadata input_indices must match the protocol input_count"
            )
        inputs, targets = _validate_scene_indices(
            len(self.frames), self.input_indices, target_stride=self.protocol.fold_size
        )
        if tuple(self.query_indices) != targets:
            raise Dl3dvProtocolError("DL3DV query_indices must be every eighth source frame")
        object.__setattr__(self, "input_indices", inputs)
        object.__setattr__(self, "query_indices", targets)

    @property
    def arrival_indices(self) -> tuple[int, ...]:
        return tuple(sorted(self.input_indices))

    @property
    def warmup_indices(self) -> tuple[int, ...]:
        return self.arrival_indices[: self.protocol.warmup_length]

    @property
    def stream_indices(self) -> tuple[int, ...]:
        return self.arrival_indices[self.protocol.warmup_length :]

    @property
    def episode_id(self) -> str:
        return f"dl3dv140--{self.scene_id}--{self.protocol.name}"

    @property
    def query_vault_id(self) -> str:
        return f"dl3dv140-query--{self.scene_id}--{self.protocol.name}"

    @property
    def raw_source_size(self) -> tuple[int, int]:
        return self.frames[0].source_image_size

    def frame(self, frame_id: int) -> Dl3dvFrame:
        if type(frame_id) is not int or frame_id < 0 or frame_id >= len(self.frames):
            raise Dl3dvProtocolError(f"frame_id is outside scene metadata: {frame_id}")
        frame = self.frames[frame_id]
        if frame.frame_id != frame_id:
            raise Dl3dvProtocolError("DL3DV frame IDs must follow source frame order")
        return frame

    def to_dict(self) -> dict[str, Any]:
        return {
            "scene_id": self.scene_id,
            "scene_root": str(self.scene_root),
            "frames": [frame.to_dict() for frame in self.frames],
            "input_indices": list(self.input_indices),
            "query_indices": list(self.query_indices),
            "protocol": {
                "mode": self.protocol.mode,
                "input_count": self.protocol.input_count,
                "fold_size": self.protocol.fold_size,
                "warmup_length": self.protocol.warmup_length,
            },
            "split_id": self.split_id,
            "target_size": list(self.target_size),
            "source_schema": self.source_schema,
            "coordinate_protocol": {
                "pose_transform": "official Long-LRM c2w conversion",
                "gauge": "warmup-only max baseline, frozen before stream",
                "target_extent_m": DL3DV_WARMUP_TARGET_EXTENT_M,
                "gauge_calibration": DL3DV_GAUGE_CALIBRATION,
                "gauge_valid_rigid_train_windows": DL3DV_GAUGE_VALID_WINDOWS,
                "gauge_excluded_invalid_rigid_windows": DL3DV_GAUGE_EXCLUDED_WINDOWS,
                "metric_depth_known": False,
                "rgb_only": True,
                "local_bounds_m": [list(v) for v in DL3DV_LOCAL_BOUNDS_M],
                "tttLRM_scale_difference": (
                    "tttLRM normalizes all selected input and target poses; this adapter uses "
                    "the first four warmup poses only"
                ),
            },
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Dl3dvSceneMetadata:
        protocol_value = value["protocol"]
        protocol = Dl3dvProtocol(
            str(protocol_value["mode"]),
            int(protocol_value["input_count"]),
            int(protocol_value.get("fold_size", DL3DV_TARGET_STRIDE)),
            int(protocol_value.get("warmup_length", DL3DV_WARMUP_LENGTH)),
        )
        return cls(
            str(value["scene_id"]),
            Path(value["scene_root"]),
            tuple(Dl3dvFrame.from_dict(frame) for frame in value["frames"]),
            tuple(int(item) for item in value["input_indices"]),
            tuple(int(item) for item in value["query_indices"]),
            protocol,
            str(value.get("split_id", "test")),
            tuple(int(item) for item in value.get("target_size", DL3DV_EVAL_IMAGE_SIZE)),
            str(value.get("source_schema", "Long-LRM/tttLRM transforms.json")),
        )


def _frame_path(value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise Dl3dvProtocolError("DL3DV frame file_path must be a non-empty string")
    normalized = value.replace("\\", "/")
    path = Path(normalized)
    if path.is_absolute() or any(part == ".." for part in path.parts):
        raise Dl3dvProtocolError(f"unsafe DL3DV frame file_path: {value!r}")
    parts = tuple(part for part in path.parts if part not in ("", "."))
    if not parts or len(parts) == 1:
        raise Dl3dvProtocolError(
            "DL3DV frame file_path must point to an image filename"
        )
    # The pinned converter discards any parent directories and keeps only the
    # basename under ``images_4``.
    return f"images_4/{parts[-1]}"


def _scene_fxfycxcy(scene: Mapping[str, Any], frame: Mapping[str, Any]) -> tuple[float, ...]:
    values = [frame.get(name, scene.get(name)) for name in ("fl_x", "fl_y", "cx", "cy")]
    if any(value is None for value in values):
        raise Dl3dvProtocolError("transforms.json must provide fl_x, fl_y, cx, and cy")
    return _as_float_tuple(values, 4, "frame intrinsics")


def _scene_distortion(scene: Mapping[str, Any], frame: Mapping[str, Any]) -> tuple[float, ...]:
    values = [frame.get(name, scene.get(name, 0.0)) for name in ("k1", "k2", "p1", "p2")]
    return _as_float_tuple(values, 4, "frame distortion")


def _resolve_images4_geometry(
    transforms_size: tuple[int, int], *, require_source_size: bool
) -> tuple[tuple[int, int], tuple[float, float]]:
    """Resolve the ``images_4`` size and metadata-to-pixel intrinsic scales.

    DL3DV's transforms retain the original 4K dimensions while the benchmark
    tree stores the level-4 images at 960x540.  The public Long-LRM converter
    scales intrinsics from the transform dimensions to the decoded image
    dimensions before undistortion.  This inference uses metadata only.
    """

    height, width = transforms_size
    target_height, target_width = DL3DV_SOURCE_IMAGE_SIZE
    if transforms_size == DL3DV_SOURCE_IMAGE_SIZE:
        return transforms_size, (1.0, 1.0)
    factor = DL3DV_IMAGES4_DOWNSAMPLE
    if height % factor == 0 and width % factor == 0:
        source_size = (height // factor, width // factor)
        if min(source_size) >= 2:
            scale = 1.0 / factor
            return source_size, (scale, scale)
    if require_source_size:
        raise Dl3dvProtocolError(
            "DL3DV transforms dimensions must describe an images_4 source "
            f"or a {factor}x original-resolution multiple, got {transforms_size}"
        )
    return transforms_size, (1.0, 1.0)


def _read_png_size_header(path: Path) -> tuple[int, int] | None:
    """Read PNG dimensions without decoding or loading image pixels."""

    try:
        with path.open("rb") as handle:
            header = handle.read(24)
    except FileNotFoundError:
        return None
    if len(header) < 24 or header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR":
        return None
    width, height = struct.unpack(">II", header[16:24])
    if width < 2 or height < 2:
        raise Dl3dvProtocolError(f"DL3DV PNG has invalid dimensions: {path}")
    return int(height), int(width)


def prepare_scene_metadata(
    scene_root: str | Path,
    split_record: LongLRMSplitRecord | Mapping[str, Any],
    protocol: Dl3dvProtocol,
    *,
    split_id: str = "test",
    target_size: tuple[int, int] = DL3DV_EVAL_IMAGE_SIZE,
    require_source_size: bool = False,
) -> Dl3dvSceneMetadata:
    """Prepare one scene from metadata only; no RGB file is opened."""

    root = Path(scene_root)
    transforms_path = root / "nerfstudio" / "transforms.json"
    try:
        transforms = json.loads(transforms_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise
    except json.JSONDecodeError as error:
        raise Dl3dvProtocolError(f"invalid DL3DV transforms JSON: {transforms_path}") from error
    if not isinstance(transforms, Mapping):
        raise Dl3dvProtocolError("DL3DV transforms.json root must be an object")
    try:
        width = int(transforms["w"])
        height = int(transforms["h"])
    except (KeyError, TypeError, ValueError) as error:
        raise Dl3dvProtocolError("transforms.json must provide integer w and h") from error
    transforms_size = (height, width)
    if min(transforms_size) < 2:
        raise Dl3dvProtocolError("DL3DV source image size must be at least 2x2")
    source_size, intrinsic_scales = _resolve_images4_geometry(
        transforms_size, require_source_size=require_source_size
    )
    raw_frames = transforms.get("frames")
    if not isinstance(raw_frames, list) or not raw_frames:
        raise Dl3dvProtocolError("transforms.json must contain a non-empty frames list")
    scene_id = root.name
    if isinstance(split_record, LongLRMSplitRecord):
        record = split_record
    else:
        scene_name = split_record.get("scene_name")
        arrays = {
            count: tuple(split_record[f"fold_8_kmeans_{count}_input"])
            for count in (16, 32, 64, 128)
            if f"fold_8_kmeans_{count}_input" in split_record
        }
        if not isinstance(scene_name, str):
            raise Dl3dvProtocolError("split_record must contain scene_name")
        record = LongLRMSplitRecord(scene_name, arrays)
    if record.scene_name != scene_id:
        raise Dl3dvProtocolError(
            f"scene folder and Long-LRM scene_name differ: {scene_id!r} != {record.scene_name!r}"
        )
    raw_metadata_frames: list[Dl3dvFrame] = []
    for frame_id, raw_frame in enumerate(raw_frames):
        if not isinstance(raw_frame, Mapping):
            raise Dl3dvProtocolError(f"DL3DV frame {frame_id} must be an object")
        path = _frame_path(raw_frame.get("file_path"))
        frame_source_size = source_size
        frame_intrinsic_scales = intrinsic_scales
        header_size = _read_png_size_header(root / "nerfstudio" / Path(path))
        if header_size is not None:
            frame_source_size = header_size
            frame_intrinsic_scales = (
                frame_source_size[1] / width,
                frame_source_size[0] / height,
            )
        c2w = official_pose_transform(raw_frame.get("transform_matrix"))
        raw_intrinsics = _scene_fxfycxcy(transforms, raw_frame)
        scale_x, scale_y = frame_intrinsic_scales
        raw_intrinsics = (
            raw_intrinsics[0] * scale_x,
            raw_intrinsics[1] * scale_y,
            raw_intrinsics[2] * scale_x,
            raw_intrinsics[3] * scale_y,
        )
        raw_metadata_frames.append(
            Dl3dvFrame(
                scene_id,
                frame_id,
                path,
                tuple(tuple(float(item) for item in row) for row in c2w),
                raw_intrinsics,
                _scene_distortion(transforms, raw_frame),
                frame_source_size,
            )
        )
    input_indices = record.input_indices(protocol.input_count)
    input_indices, query_indices = _validate_scene_indices(
        len(raw_metadata_frames), input_indices, target_stride=protocol.fold_size
    )
    if len(input_indices) < protocol.warmup_length:
        raise Dl3dvProtocolError("DL3DV input count is smaller than the warmup prefix")
    return Dl3dvSceneMetadata(
        scene_id,
        root,
        tuple(raw_metadata_frames),
        input_indices,
        query_indices,
        protocol,
        split_id,
        target_size,
    )


@dataclass(frozen=True)
class Dl3dvBenchmarkMetadata:
    scenes: tuple[Dl3dvSceneMetadata, ...]
    raw_root: Path
    split_path: Path
    split_sha256: str
    protocol: Dl3dvProtocol
    split_revision: str = DL3DV_SPLIT_REVISION

    def coverage(self) -> dict[str, Any]:
        return validate_complete_coverage(
            self.scenes,
            expected_scene_names=(scene.scene_name for scene in self.split_records),
        )

    @property
    def split_records(self) -> tuple[LongLRMSplitRecord, ...]:
        return load_long_lrm_split(self.split_path)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": DL3DV_METADATA_SCHEMA,
            "raw_root": str(self.raw_root),
            "split_path": str(self.split_path),
            "split_sha256": self.split_sha256,
            "split_revision": self.split_revision,
            "split_file_sha1": DL3DV_SPLIT_FILE_SHA1,
            "protocol": {
                "mode": self.protocol.mode,
                "input_count": self.protocol.input_count,
                "fold_size": self.protocol.fold_size,
                "warmup_length": self.protocol.warmup_length,
            },
            "coverage": self.coverage(),
            "scenes": [scene.to_dict() for scene in self.scenes],
        }


def prepare_benchmark_metadata(
    raw_root: str | Path,
    split_path: str | Path,
    protocol: Dl3dvProtocol,
    *,
    output_path: str | Path | None = None,
    split_id: str = "test",
    target_size: tuple[int, int] = DL3DV_EVAL_IMAGE_SIZE,
    require_source_size: bool = True,
) -> Dl3dvBenchmarkMetadata:
    """Prepare all 140 scene manifests without touching benchmark image pixels."""

    root = Path(raw_root)
    split_file = Path(split_path)
    split_records = load_long_lrm_split(split_file)
    expected_names = {record.scene_name for record in split_records}
    scenes: list[Dl3dvSceneMetadata] = []
    for record in split_records:
        scene_root = root / record.scene_name
        if not scene_root.is_dir():
            raise FileNotFoundError(f"DL3DV scene directory is missing: {scene_root}")
        scenes.append(
            prepare_scene_metadata(
                scene_root,
                record,
                protocol,
                split_id=split_id,
                target_size=target_size,
                require_source_size=require_source_size,
            )
        )
    metadata = Dl3dvBenchmarkMetadata(
        tuple(scenes),
        root,
        split_file,
        hashlib.sha256(split_file.read_bytes()).hexdigest(),
        protocol,
    )
    validate_complete_coverage(
        metadata.scenes,
        expected_scene_names=expected_names,
    )
    if output_path is not None:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(metadata.to_dict(), indent=2, sort_keys=True, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )
    return metadata


def load_benchmark_metadata(path: str | Path) -> Dl3dvBenchmarkMetadata:
    source = Path(path)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise Dl3dvProtocolError(f"invalid DL3DV metadata JSON: {source}") from error
    if not isinstance(value, Mapping) or value.get("schema") != DL3DV_METADATA_SCHEMA:
        raise Dl3dvProtocolError(f"unexpected DL3DV metadata schema: {source}")
    split_revision = value.get("split_revision")
    if split_revision != DL3DV_SPLIT_REVISION:
        raise Dl3dvProtocolError(
            "DL3DV metadata split_revision is not the pinned Long-LRM revision: "
            f"expected={DL3DV_SPLIT_REVISION}, actual={split_revision}"
        )
    stored_split_hash = value.get("split_sha256")
    if stored_split_hash != DL3DV_SPLIT_SHA256:
        raise Dl3dvProtocolError(
            "DL3DV metadata split_sha256 is not the pinned split digest: "
            f"expected={DL3DV_SPLIT_SHA256}, actual={stored_split_hash}"
        )
    stored_split_sha1 = value.get("split_file_sha1", DL3DV_SPLIT_FILE_SHA1)
    if stored_split_sha1 != DL3DV_SPLIT_FILE_SHA1:
        raise Dl3dvProtocolError(
            "DL3DV metadata split_file_sha1 is not the pinned Git blob identity: "
            f"expected={DL3DV_SPLIT_FILE_SHA1}, actual={stored_split_sha1}"
        )
    protocol_value = value.get("protocol")
    if not isinstance(protocol_value, Mapping):
        raise Dl3dvProtocolError("DL3DV metadata protocol must be an object")
    protocol = Dl3dvProtocol(
        str(protocol_value["mode"]),
        int(protocol_value["input_count"]),
        int(protocol_value.get("fold_size", DL3DV_TARGET_STRIDE)),
        int(protocol_value.get("warmup_length", DL3DV_WARMUP_LENGTH)),
    )
    try:
        split_path = Path(value["split_path"])
        raw_root = Path(value["raw_root"])
        raw_scene_values = value["scenes"]
    except (KeyError, TypeError, ValueError) as error:
        raise Dl3dvProtocolError("DL3DV metadata is missing required paths or scenes") from error
    if not split_path.is_file():
        raise FileNotFoundError(f"DL3DV split file is missing: {split_path}")
    actual_split_hash = hashlib.sha256(split_path.read_bytes()).hexdigest()
    if actual_split_hash != stored_split_hash or actual_split_hash != DL3DV_SPLIT_SHA256:
        raise Dl3dvProtocolError("DL3DV split file does not match the pinned SHA256")
    split_records = load_long_lrm_split(split_path)
    records_by_scene = {record.scene_name: record for record in split_records}
    if not isinstance(raw_scene_values, list):
        raise Dl3dvProtocolError("DL3DV metadata scenes must be a list")
    scenes = tuple(Dl3dvSceneMetadata.from_dict(item) for item in raw_scene_values)
    if protocol != Dl3dvProtocol(
        str(protocol_value["mode"]),
        int(protocol_value["input_count"]),
        int(protocol_value.get("fold_size", DL3DV_TARGET_STRIDE)),
        int(protocol_value.get("warmup_length", DL3DV_WARMUP_LENGTH)),
    ):
        raise Dl3dvProtocolError("DL3DV metadata protocol is internally inconsistent")
    for scene in scenes:
        record = records_by_scene.get(scene.scene_id)
        if record is None:
            raise Dl3dvProtocolError(
                f"DL3DV metadata scene is absent from the pinned split: {scene.scene_id}"
            )
        expected_root = (raw_root / scene.scene_id).resolve(strict=False)
        if Path(scene.scene_root).resolve(strict=False) != expected_root:
            raise Dl3dvProtocolError(
                f"DL3DV scene_root does not match raw_root for {scene.scene_id}"
            )
        if scene.protocol != protocol:
            raise Dl3dvProtocolError("DL3DV metadata scenes do not share the top-level protocol")
        if scene.input_indices != record.input_indices(protocol.input_count):
            raise Dl3dvProtocolError(
                f"DL3DV input_indices differ from the pinned split for {scene.scene_id}"
            )
        if scene.query_indices != target_frame_indices(len(scene.frames), protocol.fold_size):
            raise Dl3dvProtocolError(
                f"DL3DV query_indices differ from the pinned target rule for {scene.scene_id}"
            )
    metadata = Dl3dvBenchmarkMetadata(
        scenes,
        raw_root,
        split_path,
        str(stored_split_hash),
        protocol,
        str(split_revision),
    )
    validate_complete_coverage(
        metadata.scenes,
        expected_scene_names=records_by_scene,
    )
    return metadata


@dataclass(frozen=True)
class WarmupGauge:
    """Frozen scale and first-camera frame for one causal episode."""

    target_extent_m: float
    observed_max_baseline: float
    scale: float
    anchor_c2w: np.ndarray

    def __post_init__(self) -> None:
        for name, value in (
            ("target_extent_m", self.target_extent_m),
            ("observed_max_baseline", self.observed_max_baseline),
            ("scale", self.scale),
        ):
            if not math.isfinite(float(value)) or float(value) <= 0.0:
                raise Dl3dvProtocolError(f"warmup gauge {name} must be finite and positive")
        object.__setattr__(
            self,
            "anchor_c2w",
            _validate_rigid_camera(
                np.asarray(self.anchor_c2w, dtype=np.float64), "gauge anchor"
            ).copy(),
        )

    def transform(self, c2w: np.ndarray) -> np.ndarray:
        """Express a pose in the first warmup frame and apply frozen translation scale."""

        relative = np.linalg.solve(self.anchor_c2w, _validate_rigid_camera(c2w, "c2w")).copy()
        relative[:3, 3] *= self.scale
        return _validate_homogeneous(relative, "gauged c2w")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": DL3DV_GAUGE_SCHEMA,
            "target_extent_m": self.target_extent_m,
            "calibration": DL3DV_GAUGE_CALIBRATION,
            "calibration_valid_rigid_train_windows": DL3DV_GAUGE_VALID_WINDOWS,
            "calibration_excluded_invalid_rigid_windows": DL3DV_GAUGE_EXCLUDED_WINDOWS,
            "observed_max_baseline": self.observed_max_baseline,
            "scale": self.scale,
            "anchor_c2w": self.anchor_c2w.tolist(),
            "metric_depth_known": False,
            "rgb_only": True,
        }


def compute_warmup_gauge(
    c2ws: Sequence[np.ndarray],
    *,
    target_extent_m: float = DL3DV_WARMUP_TARGET_EXTENT_M,
    minimum_baseline: float = DL3DV_MIN_WARMUP_BASELINE,
) -> WarmupGauge:
    """Compute the frozen four-view baseline gauge without future/query poses."""

    if len(c2ws) != DL3DV_WARMUP_LENGTH:
        raise Dl3dvProtocolError("warmup gauge requires exactly four prefix poses")
    if not math.isfinite(float(target_extent_m)) or float(target_extent_m) <= 0.0:
        raise Dl3dvProtocolError("target_extent_m must be finite and positive")
    if not math.isfinite(float(minimum_baseline)) or float(minimum_baseline) <= 0.0:
        raise Dl3dvProtocolError("minimum_baseline must be finite and positive")
    poses = [_validate_rigid_camera(np.asarray(value), "warmup c2w") for value in c2ws]
    first_center = poses[0][:3, 3]
    observed = max(float(np.linalg.norm(pose[:3, 3] - first_center)) for pose in poses)
    if observed < minimum_baseline:
        raise Dl3dvProtocolError(
            f"warmup camera baseline is too small for a stable gauge: {observed:.3e}"
        )
    scale = float(target_extent_m) / observed
    return WarmupGauge(float(target_extent_m), observed, scale, poses[0])


class Dl3dvOnlineObservationSource:
    """Causal DL3DV observations with a post-seal lazy query callback seam."""

    def __init__(
        self,
        scene: Dl3dvSceneMetadata,
        *,
        device: str | torch.device = "cpu",
        target_extent_m: float = DL3DV_WARMUP_TARGET_EXTENT_M,
        minimum_baseline: float = DL3DV_MIN_WARMUP_BASELINE,
    ) -> None:
        self.metadata = scene
        self.device = torch.device(device)
        warmup_frames = [scene.frame(index) for index in scene.warmup_indices]
        self.gauge = compute_warmup_gauge(
            [frame.c2w_array for frame in warmup_frames],
            target_extent_m=target_extent_m,
            minimum_baseline=minimum_baseline,
        )
        self._warmup_cache: tuple[OnlineObservation, ...] | None = None
        self._stream_cursor = 0
        self._sealed: SealedScene | None = None
        self._query_cache: dict[int, OnlineObservation] = {}

    @property
    def episode_id(self) -> str:
        return self.metadata.episode_id

    @property
    def scene_id(self) -> str:
        return self.metadata.scene_id

    @property
    def split_id(self) -> str:
        return self.metadata.split_id

    @property
    def query_vault_id(self) -> str:
        return self.metadata.query_vault_id

    @property
    def image_size(self) -> tuple[int, int]:
        return self.metadata.target_size

    @property
    def stream_steps(self) -> int:
        return len(self.metadata.stream_indices)

    @property
    def query_count(self) -> int:
        return len(self.metadata.query_indices)

    @property
    def remaining_steps(self) -> int:
        return self.stream_steps - self._stream_cursor

    @property
    def sealed(self) -> bool:
        return self._sealed is not None

    def warmup(self) -> tuple[OnlineObservation, ...]:
        if self._sealed is not None:
            raise ValueError("DL3DV source is sealed")
        if self._warmup_cache is None:
            self._warmup_cache = tuple(
                self._load_observation(self.metadata.frame(index))
                for index in self.metadata.warmup_indices
            )
        return self._warmup_cache

    def next_observation(self) -> OnlineObservation:
        if self._sealed is not None:
            raise ValueError("DL3DV source is sealed")
        if self._stream_cursor >= self.stream_steps:
            raise StopIteration("DL3DV online stream is exhausted")
        frame = self.metadata.frame(self.metadata.stream_indices[self._stream_cursor])
        self._stream_cursor += 1
        return self._load_observation(frame)

    def _load_observation(self, frame: Dl3dvFrame) -> OnlineObservation:
        image_path = self.metadata.scene_root / "nerfstudio" / Path(frame.rgb_relpath)
        rgb, intrinsic = preprocess_dl3dv_rgb(
            image_path,
            frame,
            target_size=self.metadata.target_size,
        )
        c2w = self.gauge.transform(frame.c2w_array)
        camera = Cameras(
            torch.from_numpy(intrinsic).to(self.device, dtype=torch.float32),
            torch.from_numpy(c2w).to(self.device, dtype=torch.float32),
            self.metadata.target_size,
        )
        return OnlineObservation(
            self.metadata.scene_id,
            frame.frame_id,
            rgb.to(self.device, dtype=torch.float32),
            camera,
        )

    def queries_for_sealed(self, sealed: SealedScene) -> tuple[RGBQuery, ...]:
        """Return lazy target callbacks after the runner has sealed this episode."""

        if not isinstance(sealed, SealedScene):
            raise TypeError("queries_for_sealed requires a SealedScene")
        if self._sealed is not None:
            raise ValueError("DL3DV query callbacks were already opened")
        if self._stream_cursor != self.stream_steps:
            raise ValueError("DL3DV query callbacks require the online stream to be exhausted")
        expected_observed = tuple(self.metadata.arrival_indices)
        if tuple(sealed.observed_ids) != expected_observed:
            raise ValueError("sealed observed_ids do not match DL3DV context arrival order")
        if (
            sealed.episode_id != self.episode_id
            or sealed.scene_id != self.scene_id
            or sealed.split_id != self.split_id
            or sealed.query_vault_id != self.query_vault_id
        ):
            raise ValueError("sealed scene identity does not match DL3DV metadata")
        self._sealed = sealed
        result: list[RGBQuery] = []
        for frame_id in self.metadata.query_indices:
            result.append(
                RGBQuery.for_sealed(
                    sealed,
                    frame_id,
                    lambda frame_id=frame_id: self._query_observation(frame_id).camera,
                    lambda frame_id=frame_id: self._query_observation(frame_id).rgb,
                    self.metadata.target_size,
                )
            )
        return tuple(result)

    # Explicit alias for callers that use the source/evaluator terminology.
    open_query_callbacks = queries_for_sealed

    def _query_observation(self, frame_id: int) -> OnlineObservation:
        if self._sealed is None:
            raise RuntimeError("DL3DV query supplier is unavailable before seal")
        if frame_id not in self.metadata.query_indices:
            raise Dl3dvProtocolError(f"frame {frame_id} is not a DL3DV query frame")
        if frame_id not in self._query_cache:
            self._query_cache[frame_id] = self._load_observation(self.metadata.frame(frame_id))
        return self._query_cache[frame_id]


# Compatibility names make the boundary easy to discover without changing the
# existing generic OnlineEpisodeSource contract.
DL3DVProtocol = Dl3dvProtocol
DL3DVSceneMetadata = Dl3dvSceneMetadata
DL3DVOnlineObservationSource = Dl3dvOnlineObservationSource
Dl3dvEpisodeSource = Dl3dvOnlineObservationSource
DL3DVBenchmarkMetadata = Dl3dvBenchmarkMetadata
prepare_dl3dv_metadata = prepare_benchmark_metadata


__all__ = [
    "DL3DV_EVAL_IMAGE_SIZE",
    "DL3DV_GAUGE_CALIBRATION",
    "DL3DV_GAUGE_EXCLUDED_WINDOWS",
    "DL3DV_GAUGE_SCHEMA",
    "DL3DV_GAUGE_VALID_WINDOWS",
    "DL3DV_IMAGES4_DOWNSAMPLE",
    "DL3DV_LOCAL_BOUNDS_M",
    "DL3DV_METADATA_SCHEMA",
    "DL3DV_MIN_WARMUP_BASELINE",
    "DL3DV_RIGID_ROTATION_TOLERANCE",
    "DL3DV_SCENE_COUNT",
    "DL3DV_SOURCE_IMAGE_SIZE",
    "DL3DV_SPLIT_REVISION",
    "DL3DV_SPLIT_FILE_SHA1",
    "DL3DV_SPLIT_SHA256",
    "DL3DV_TARGET_STRIDE",
    "DL3DV_WARMUP_LENGTH",
    "DL3DV_WARMUP_TARGET_EXTENT_M",
    "Dl3dvBenchmarkMetadata",
    "DL3DVOnlineObservationSource",
    "DL3DVProtocol",
    "DL3DVSceneMetadata",
    "DL3DVBenchmarkMetadata",
    "Dl3dvEpisodeSource",
    "Dl3dvFrame",
    "Dl3dvOnlineObservationSource",
    "Dl3dvProtocol",
    "Dl3dvProtocolError",
    "Dl3dvSceneMetadata",
    "LongLRMSplitRecord",
    "WarmupGauge",
    "compute_warmup_gauge",
    "load_benchmark_metadata",
    "load_long_lrm_split",
    "official_intrinsics",
    "official_pose_transform",
    "preprocess_dl3dv_rgb",
    "prepare_benchmark_metadata",
    "prepare_dl3dv_metadata",
    "prepare_scene_metadata",
    "resize_and_center_crop",
    "target_frame_indices",
    "undistorted_intrinsics",
    "validate_complete_coverage",
]
