"""Sequential online episode readers with no query or label access."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from mcss.dynamic.types import OnlineObservation
from mcss.types import Cameras

ONLINE_EPISODE_SCHEMA = "mcss.dynamic.online_episode.v1"
ONLINE_EPISODE_COLLECTION_SCHEMA = "mcss.dynamic.online_episodes.v1"
_COLLECTION_FIELDS = frozenset({"schema_version", "episodes"})
_EPISODE_FIELDS = frozenset(
    {
        "episode_id",
        "scene_id",
        "split_id",
        "query_vault_id",
        "image_size",
        "declared_length",
        "warmup",
        "stream",
    }
)
_FRAME_FIELDS = frozenset({"frame_id", "intrinsics", "c2w", "rgb"})
_SINGLE_FILE_FIELDS = frozenset({"schema_version", *_EPISODE_FIELDS})


class OnlineEpisodeSource:
    """Expose one episode as warmup plus a strictly sequential stream.

    This class intentionally has no random-access operation and never loads a stream image
    until that observation arrives at ``next_observation``.
    """

    def __init__(self, spec: dict[str, Any], *, device: torch.device) -> None:
        self._device = device
        self._spec = _validate_episode(spec)
        self._warmup_records = tuple(self._spec["warmup"])
        self._stream_records = tuple(self._spec["stream"])
        self._warmup_cache: tuple[OnlineObservation, ...] | None = None
        self._cursor = 0

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        device: str | torch.device = "cpu",
        *,
        episode_id: str | None = None,
    ) -> OnlineEpisodeSource:
        """Load one strict online-episode artifact without opening any referenced image."""

        artifact = _read_json(path)
        if frozenset(artifact) == _SINGLE_FILE_FIELDS:
            if artifact["schema_version"] != ONLINE_EPISODE_SCHEMA:
                raise ValueError(f"expected online episode schema {ONLINE_EPISODE_SCHEMA}")
            selected = {name: artifact[name] for name in _EPISODE_FIELDS}
            if episode_id is not None and selected["episode_id"] != episode_id:
                raise ValueError(f"online episode_id not found: {episode_id}")
            return cls(selected, device=torch.device(device))

        _require_exact_fields(artifact, _COLLECTION_FIELDS, "online episode collection")
        if artifact["schema_version"] != ONLINE_EPISODE_COLLECTION_SCHEMA:
            raise ValueError(f"expected online episode schema {ONLINE_EPISODE_COLLECTION_SCHEMA}")
        episodes = artifact["episodes"]
        if not isinstance(episodes, list) or not episodes:
            raise ValueError("online episode collection requires a non-empty episodes list")
        selected = [episode for episode in episodes if isinstance(episode, dict)]
        if len(selected) != len(episodes):
            raise ValueError("online episode entries must be objects")
        if episode_id is not None:
            selected = [episode for episode in selected if episode.get("episode_id") == episode_id]
            if len(selected) != 1:
                raise ValueError(f"online episode_id not found uniquely: {episode_id}")
        elif len(selected) != 1:
            raise ValueError(
                "episode_id is required when an online artifact contains multiple episodes"
            )
        return cls(selected[0], device=torch.device(device))

    @property
    def episode_id(self) -> str:
        return self._spec["episode_id"]

    @property
    def scene_id(self) -> str:
        return self._spec["scene_id"]

    @property
    def split_id(self) -> str:
        return self._spec["split_id"]

    @property
    def query_vault_id(self) -> str:
        return self._spec["query_vault_id"]

    @property
    def remaining_steps(self) -> int:
        return len(self._stream_records) - self._cursor

    def warmup(self) -> tuple[OnlineObservation, ...]:
        """Return the arrived warmup observations, loading only their RGB files once."""

        if self._warmup_cache is None:
            self._warmup_cache = tuple(
                self._read_observation(record) for record in self._warmup_records
            )
        return self._warmup_cache

    def next_observation(self) -> OnlineObservation:
        """Load exactly the next arrived stream RGB, or raise ``StopIteration`` when exhausted."""

        if self._cursor >= len(self._stream_records):
            raise StopIteration("online stream is exhausted")
        record = self._stream_records[self._cursor]
        self._cursor += 1
        return self._read_observation(record)

    def _read_observation(self, record: dict[str, Any]) -> OnlineObservation:
        image_size = tuple(self._spec["image_size"])
        rgb_path = Path(record["rgb"])
        with Image.open(rgb_path) as image:
            image = image.convert("RGB")
            if (image.height, image.width) != image_size:
                raise ValueError(f"RGB size does not match episode image_size: {rgb_path}")
            rgb = torch.from_numpy(np.asarray(image, dtype=np.float32).copy())
        rgb = (rgb.permute(2, 0, 1).contiguous() / 255.0).to(self._device)
        camera = _camera_from_record(record, image_size, self._device)
        return OnlineObservation(
            scene_id=self.scene_id,
            frame_id=record["frame_id"],
            rgb=rgb,
            camera=camera,
        )


def _read_json(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid online episode JSON: {source}") from error
    if not isinstance(value, dict):
        raise ValueError("online episode JSON root must be an object")
    return value


def _validate_episode(value: dict[str, Any]) -> dict[str, Any]:
    _require_exact_fields(value, _EPISODE_FIELDS, "online episode")
    for field in ("episode_id", "scene_id", "split_id", "query_vault_id"):
        _require_identifier(value[field], field)
    image_size = _validate_image_size(value["image_size"])
    declared_length = value["declared_length"]
    if type(declared_length) is not int or declared_length < 1:
        raise ValueError("declared_length must be a positive integer")
    warmup = _validate_frames(value["warmup"], image_size, "warmup")
    stream = _validate_frames(value["stream"], image_size, "stream")
    if not warmup:
        raise ValueError("online episode requires at least one warmup observation")
    frame_ids = [record["frame_id"] for record in (*warmup, *stream)]
    if len(frame_ids) != len(set(frame_ids)):
        raise ValueError("online warmup and stream frame_ids must not overlap")
    if frame_ids != sorted(frame_ids):
        raise ValueError("online frame_ids must be in arrival order")
    if declared_length != len(frame_ids):
        raise ValueError("declared_length must equal warmup plus stream observation count")
    return value


def _validate_frames(value: Any, image_size: tuple[int, int], name: str) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise ValueError(f"online {name} must be a list")
    records: list[dict[str, Any]] = []
    for record in value:
        if not isinstance(record, dict):
            raise ValueError(f"online {name} records must be objects")
        _require_exact_fields(record, _FRAME_FIELDS, f"online {name} frame")
        if type(record["frame_id"]) is not int or record["frame_id"] < 0:
            raise ValueError("online frame_id must be a non-negative integer")
        if not isinstance(record["rgb"], str) or not record["rgb"]:
            raise ValueError("online RGB reference must be a non-empty path string")
        _camera_from_record(record, image_size, torch.device("cpu"))
        records.append(record)
    return records


def _camera_from_record(
    record: dict[str, Any], image_size: tuple[int, int], device: torch.device
) -> Cameras:
    try:
        intrinsics = torch.tensor(record["intrinsics"], dtype=torch.float32, device=device)
        c2w = torch.tensor(record["c2w"], dtype=torch.float32, device=device)
    except (TypeError, ValueError) as error:
        raise ValueError("online camera references must contain numeric matrices") from error
    return Cameras(intrinsics=intrinsics, c2w=c2w, image_size=image_size)


def _validate_image_size(value: Any) -> tuple[int, int]:
    if (
        not isinstance(value, list)
        or len(value) != 2
        or any(type(item) is not int or item < 2 for item in value)
    ):
        raise ValueError("image_size must be [height, width] with values at least two")
    return value[0], value[1]


def _require_identifier(value: Any, name: str) -> None:
    if not isinstance(value, str) or not value or any(char in value for char in "\\/"):
        raise ValueError(f"{name} must be a simple non-empty identifier")


def _require_exact_fields(value: dict[str, Any], expected: frozenset[str], name: str) -> None:
    actual = frozenset(value)
    if actual != expected:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        raise ValueError(f"{name} fields mismatch; missing={missing}, unexpected={unexpected}")
