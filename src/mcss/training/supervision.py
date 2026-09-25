"""Training-private query supervision with strict train-scene identity checks."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch import Tensor

from mcss.types import Cameras

QUERY_VAULT_SCHEMA = "mcss.dynamic.query_vault.v1"
_EPISODE_FIELDS = frozenset(
    {
        "schema_version",
        "episode_id",
        "scene_id",
        "split_id",
        "query_vault_id",
        "image_size",
        "query",
    }
)
_QUERY_FIELDS = frozenset({"frame_id", "intrinsics", "c2w", "rgb", "depth"})


@dataclass(frozen=True)
class TrainingSupervision:
    """Raw-world query cameras and labels available only to offline training."""

    episode_id: str
    scene_id: str
    query_vault_id: str
    frame_ids: tuple[int, ...]
    query_cameras: Cameras
    query_rgb: Tensor
    query_depth: Tensor

    @classmethod
    def from_query_file(
        cls,
        path: str | Path,
        *,
        episode_id: str,
        scene_id: str,
        query_vault_id: str,
        context_frame_ids: tuple[int, ...],
        device: str | torch.device = "cpu",
    ) -> TrainingSupervision:
        """Open one train-only query artifact after identity and overlap validation."""

        artifact = _read_query_artifact(path)
        _validate_identity(artifact, episode_id, scene_id, query_vault_id)
        image_size = _validate_image_size(artifact["image_size"])
        records = _validate_query_records(artifact["query"], image_size, context_frame_ids)
        target_device = torch.device(device)
        cameras = _query_cameras(records, image_size, target_device)
        rgb = torch.stack(
            [_load_rgb(record["rgb"], image_size, target_device) for record in records]
        )
        depth = torch.stack(
            [_load_depth(record["depth"], image_size, target_device) for record in records]
        )
        return cls(
            episode_id=episode_id,
            scene_id=scene_id,
            query_vault_id=query_vault_id,
            frame_ids=tuple(record["frame_id"] for record in records),
            query_cameras=cameras,
            query_rgb=rgb.unsqueeze(0),
            query_depth=depth.unsqueeze(0),
        )


def _read_query_artifact(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    try:
        artifact = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid training query JSON: {source}") from error
    if not isinstance(artifact, dict):
        raise ValueError("training query JSON root must be an object")
    _require_exact_fields(artifact, _EPISODE_FIELDS, "training query artifact")
    if artifact["schema_version"] != QUERY_VAULT_SCHEMA:
        raise ValueError(f"expected training query schema {QUERY_VAULT_SCHEMA}")
    return artifact


def _validate_identity(
    artifact: dict[str, Any], episode_id: str, scene_id: str, query_vault_id: str
) -> None:
    for name, expected in {
        "episode_id": episode_id,
        "scene_id": scene_id,
        "query_vault_id": query_vault_id,
    }.items():
        if not isinstance(expected, str) or not expected:
            raise ValueError(f"expected {name} must be a nonempty string")
        if artifact[name] != expected:
            raise ValueError(f"training query {name} does not match the online episode")
    if artifact["split_id"] != "train":
        raise ValueError("training supervision requires a training split query artifact")


def _validate_query_records(
    value: Any, image_size: tuple[int, int], context_frame_ids: tuple[int, ...]
) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise ValueError("training query artifact requires a nonempty query list")
    context_ids = set(context_frame_ids)
    if len(context_ids) != len(context_frame_ids) or any(
        type(frame_id) is not int or frame_id < 0 for frame_id in context_frame_ids
    ):
        raise ValueError("context frame IDs must be unique nonnegative integers")
    records: list[dict[str, Any]] = []
    seen: set[int] = set()
    for record in value:
        if not isinstance(record, dict):
            raise ValueError("training query records must be objects")
        _require_exact_fields(record, _QUERY_FIELDS, "training query frame")
        frame_id = record["frame_id"]
        if type(frame_id) is not int or frame_id < 0:
            raise ValueError("training query frame IDs must be nonnegative integers")
        if frame_id in seen:
            raise ValueError("training query frame IDs must be unique")
        if frame_id in context_ids:
            raise ValueError("training query frame IDs must be disjoint from context")
        seen.add(frame_id)
        for name in ("rgb", "depth"):
            if not isinstance(record[name], str) or not record[name]:
                raise ValueError(f"training query {name} reference must be a nonempty path string")
        _camera_from_record(record, image_size, torch.device("cpu"))
        records.append(record)
    if [record["frame_id"] for record in records] != sorted(seen):
        raise ValueError("training query frame IDs must be in ascending order")
    return records


def _query_cameras(
    records: list[dict[str, Any]], image_size: tuple[int, int], device: torch.device
) -> Cameras:
    cameras = [_camera_from_record(record, image_size, device) for record in records]
    return Cameras(
        torch.stack([camera.intrinsics for camera in cameras]).unsqueeze(0),
        torch.stack([camera.c2w for camera in cameras]).unsqueeze(0),
        image_size,
    )


def _camera_from_record(
    record: dict[str, Any], image_size: tuple[int, int], device: torch.device
) -> Cameras:
    try:
        intrinsics = torch.tensor(record["intrinsics"], dtype=torch.float32, device=device)
        c2w = torch.tensor(record["c2w"], dtype=torch.float32, device=device)
    except (TypeError, ValueError) as error:
        raise ValueError("training query cameras must contain numeric matrices") from error
    return Cameras(intrinsics, c2w, image_size)


def _load_rgb(path: str, image_size: tuple[int, int], device: torch.device) -> Tensor:
    source = Path(path)
    with Image.open(source) as image:
        image = image.convert("RGB")
        if (image.height, image.width) != image_size:
            raise ValueError(f"training query RGB size does not match image_size: {source}")
        array = np.asarray(image, dtype=np.float32).copy()
    return torch.from_numpy(array).permute(2, 0, 1).contiguous().div(255.0).to(device)


def _load_depth(path: str, image_size: tuple[int, int], device: torch.device) -> Tensor:
    value = np.load(Path(path), allow_pickle=False)
    if value.shape != image_size:
        raise ValueError("training query depth size does not match image_size")
    return torch.from_numpy(np.asarray(value, dtype=np.float32).copy()).unsqueeze(0).to(device)


def _validate_image_size(value: Any) -> tuple[int, int]:
    if (
        not isinstance(value, list)
        or len(value) != 2
        or any(type(item) is not int or item < 2 for item in value)
    ):
        raise ValueError(
            "training query image_size must be [height, width] with values at least two"
        )
    return value[0], value[1]


def _require_exact_fields(value: dict[str, Any], expected: frozenset[str], name: str) -> None:
    actual = frozenset(value)
    if actual != expected:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        raise ValueError(f"{name} fields mismatch; missing={missing}, unexpected={unexpected}")
