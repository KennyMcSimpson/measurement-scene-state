"""Read query labels only after a scene-state seal has been validated."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch import Tensor

from mcss.dynamic.types import SealedScene, hash_scene_state
from mcss.measurements import FixedMeasurementRenderer
from mcss.types import Cameras

QUERY_VAULT_SCHEMA = "mcss.dynamic.query_vault.v1"
_COLLECTION_FIELDS = frozenset({"schema_version", "episodes"})
_EPISODE_FIELDS = frozenset(
    {"episode_id", "scene_id", "split_id", "query_vault_id", "image_size", "query"}
)
_QUERY_FIELDS = frozenset({"frame_id", "intrinsics", "c2w", "rgb", "depth"})
_SINGLE_FILE_FIELDS = frozenset({"schema_version", *_EPISODE_FIELDS})


def evaluate_sealed(
    sealed: SealedScene,
    query_manifest_path: str | Path,
    measurement: FixedMeasurementRenderer,
    device: str | torch.device | None = None,
) -> dict[str, Any]:
    """Evaluate RGB, depth, opacity and coverage from an immutable sealed scene.

    Query metadata and image files remain unopened until the seal's state hash and identity
    fields are valid. The hash is checked again after rendering to catch state mutation by a
    misbehaving measurement implementation.
    """

    _validate_sealed(sealed)
    state_hash = hash_scene_state(sealed.scene_state)
    if state_hash != sealed.state_hash:
        raise ValueError("sealed scene state_hash does not match the current scene_state")

    vault_episodes = _read_vault(query_manifest_path)
    query_episode = _select_query_episode(vault_episodes, sealed)
    image_size = _validate_image_size(query_episode["image_size"])
    records = _validate_query_records(query_episode["query"], image_size, sealed.observed_ids)

    target_device = (
        torch.device(device) if device is not None else sealed.scene_state.density_logits.device
    )
    state = sealed.scene_state.to(target_device)
    anchor_c2w = sealed.anchor_c2w.to(device=target_device, dtype=state.density_logits.dtype)
    world_to_anchor = torch.linalg.inv(anchor_c2w)
    per_query: list[dict[str, float | int]] = []
    with torch.no_grad():
        for record in records:
            camera = _query_camera(record, image_size, target_device, world_to_anchor)
            predictions = measurement(state, camera, {"rgb", "depth", "visibility"})
            target_rgb = _load_rgb(record["rgb"], image_size, target_device)
            target_depth = _load_depth(record["depth"], image_size, target_device)
            per_query.append(
                _query_metrics(
                    record["frame_id"],
                    predictions["rgb"],
                    predictions["depth"],
                    predictions["visibility"],
                    target_rgb,
                    target_depth,
                )
            )

    if hash_scene_state(sealed.scene_state) != state_hash:
        raise RuntimeError("sealed scene_state was modified during query evaluation")
    return {"per_query": per_query, "averages": _average_metrics(per_query)}


def _validate_sealed(sealed: SealedScene) -> None:
    if not isinstance(sealed, SealedScene):
        raise TypeError("evaluate_sealed requires a SealedScene")
    for name in (
        "episode_id",
        "scene_id",
        "split_id",
        "query_vault_id",
        "state_hash",
        "checkpoint_hash",
        "config_hash",
        "fast_state_hash",
    ):
        _require_identifier(getattr(sealed, name), name)
    if not sealed.observed_ids or any(
        type(item) is not int or item < 0 for item in sealed.observed_ids
    ):
        raise ValueError("sealed observed_ids must be non-empty non-negative integers")
    if len(sealed.observed_ids) != len(set(sealed.observed_ids)):
        raise ValueError("sealed observed_ids must be unique")
    anchor = sealed.anchor_c2w
    if anchor.shape != (4, 4) or not torch.isfinite(anchor).all():
        raise ValueError("sealed anchor_c2w must be a finite 4x4 transform")
    expected_row = torch.tensor([0.0, 0.0, 0.0, 1.0], device=anchor.device, dtype=anchor.dtype)
    if not torch.allclose(anchor[3], expected_row, atol=1e-5, rtol=1e-5):
        raise ValueError("sealed anchor_c2w must have a homogeneous final row")


def _read_vault(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    try:
        vault = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid query vault JSON: {source}") from error
    if not isinstance(vault, dict):
        raise ValueError("query vault JSON root must be an object")
    if frozenset(vault) == _SINGLE_FILE_FIELDS:
        if vault["schema_version"] != QUERY_VAULT_SCHEMA:
            raise ValueError(f"expected query vault schema {QUERY_VAULT_SCHEMA}")
        return [{name: vault[name] for name in _EPISODE_FIELDS}]
    _require_exact_fields(vault, _COLLECTION_FIELDS, "query vault collection")
    if vault["schema_version"] != QUERY_VAULT_SCHEMA:
        raise ValueError(f"expected query vault schema {QUERY_VAULT_SCHEMA}")
    episodes = vault["episodes"]
    if (
        not isinstance(episodes, list)
        or not episodes
        or not all(isinstance(item, dict) for item in episodes)
    ):
        raise ValueError("query vault requires a non-empty episodes list of objects")
    return episodes


def _select_query_episode(vault: list[dict[str, Any]], sealed: SealedScene) -> dict[str, Any]:
    matches = [
        episode
        for episode in vault
        if episode.get("episode_id") == sealed.episode_id
    ]
    if len(matches) != 1:
        raise ValueError("query vault does not contain the sealed episode exactly once")
    episode = matches[0]
    _require_exact_fields(episode, _EPISODE_FIELDS, "query vault episode")
    for name in ("episode_id", "scene_id", "split_id", "query_vault_id"):
        _require_identifier(episode[name], name)
        if episode[name] != getattr(sealed, name):
            raise ValueError(f"query vault {name} does not match sealed scene")
    return episode


def _validate_query_records(
    value: Any, image_size: tuple[int, int], observed_ids: tuple[int, ...]
) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise ValueError("query vault episode requires a non-empty query list")
    records: list[dict[str, Any]] = []
    seen: set[int] = set()
    for record in value:
        if not isinstance(record, dict):
            raise ValueError("query vault records must be objects")
        _require_exact_fields(record, _QUERY_FIELDS, "query vault frame")
        frame_id = record["frame_id"]
        if type(frame_id) is not int or frame_id < 0:
            raise ValueError("query frame_id must be a non-negative integer")
        if frame_id in seen or frame_id in observed_ids:
            raise ValueError("query frame_ids must be unique and disjoint from sealed observed_ids")
        seen.add(frame_id)
        for name in ("rgb", "depth"):
            if not isinstance(record[name], str) or not record[name]:
                raise ValueError(f"query {name} reference must be a non-empty path string")
        _query_camera(record, image_size, torch.device("cpu"), torch.eye(4))
        records.append(record)
    if [record["frame_id"] for record in records] != sorted(seen):
        raise ValueError("query frame_ids must be in ascending order")
    return records


def _query_camera(
    record: dict[str, Any],
    image_size: tuple[int, int],
    device: torch.device,
    world_to_anchor: Tensor,
) -> Cameras:
    try:
        intrinsics = torch.tensor(record["intrinsics"], dtype=torch.float32, device=device)
        c2w = torch.tensor(record["c2w"], dtype=torch.float32, device=device)
    except (TypeError, ValueError) as error:
        raise ValueError("query camera references must contain numeric matrices") from error
    world_to_anchor = world_to_anchor.to(device=device, dtype=c2w.dtype)
    local_c2w = world_to_anchor @ c2w
    return Cameras(
        intrinsics.unsqueeze(0).unsqueeze(0),
        local_c2w.unsqueeze(0).unsqueeze(0),
        image_size,
    )


def _load_rgb(path: str, image_size: tuple[int, int], device: torch.device) -> Tensor:
    source = Path(path)
    with Image.open(source) as image:
        image = image.convert("RGB")
        if (image.height, image.width) != image_size:
            raise ValueError(f"query RGB size does not match image_size: {source}")
        value = torch.from_numpy(np.asarray(image, dtype=np.float32).copy())
    return (value.permute(2, 0, 1).contiguous() / 255.0).to(device).unsqueeze(0).unsqueeze(0)


def _load_depth(path: str, image_size: tuple[int, int], device: torch.device) -> Tensor:
    source = Path(path)
    value = np.load(source, allow_pickle=False)
    if value.shape != image_size:
        raise ValueError(f"query depth size does not match image_size: {source}")
    value = np.asarray(value, dtype=np.float32)
    return torch.from_numpy(value.copy()).to(device).unsqueeze(0).unsqueeze(0).unsqueeze(0)


def _query_metrics(
    frame_id: int,
    prediction_rgb: Tensor,
    prediction_depth: Tensor,
    opacity: Tensor,
    target_rgb: Tensor,
    target_depth: Tensor,
) -> dict[str, float | int]:
    valid_rgb = torch.isfinite(target_rgb).all(dim=-3, keepdim=True)
    if valid_rgb.any():
        rgb_mask = valid_rgb.expand_as(prediction_rgb)
        rgb_mse = float((prediction_rgb[rgb_mask] - target_rgb[rgb_mask]).square().mean().item())
        rgb_psnr = float("inf") if rgb_mse == 0.0 else -10.0 * math.log10(max(rgb_mse, 1e-12))
    else:
        rgb_mse = float("nan")
        rgb_psnr = float("nan")
    valid_depth = torch.isfinite(target_depth) & (target_depth > 0)
    if valid_depth.any():
        relative_error = (prediction_depth - target_depth).abs() / target_depth.abs().clamp_min(
            1e-8
        )
        depth_abs_rel = float(relative_error[valid_depth].mean().item())
    else:
        depth_abs_rel = float("nan")
    return {
        "frame_id": frame_id,
        "rgb_mse": rgb_mse,
        "rgb_psnr": rgb_psnr,
        "depth_abs_rel": depth_abs_rel,
        "opacity_mean": float(opacity.mean().item()),
        "coverage": float((opacity > 1e-6).to(dtype=torch.float32).mean().item()),
    }


def _average_metrics(per_query: list[dict[str, float | int]]) -> dict[str, float]:
    names = ("rgb_mse", "rgb_psnr", "depth_abs_rel", "opacity_mean", "coverage")
    averages: dict[str, float] = {}
    for name in names:
        values = [float(record[name]) for record in per_query if math.isfinite(float(record[name]))]
        averages[name] = float(sum(values) / len(values)) if values else float("nan")
    return averages


def _validate_image_size(value: Any) -> tuple[int, int]:
    if (
        not isinstance(value, list)
        or len(value) != 2
        or any(type(item) is not int or item < 2 for item in value)
    ):
        raise ValueError("query image_size must be [height, width] with values at least two")
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
