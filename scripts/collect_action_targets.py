"""Collect offline all-action teacher rows for dynamic train episodes."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import torch

from mcss.training.action_teacher import (
    DEFAULT_HORIZON,
    DEFAULT_MAX_UNITS,
    DEFAULT_RAY_CHUNK_SIZE,
    DEFAULT_RENDER_SAMPLES,
    collect_action_targets,
)
from mcss.training.experiment import load_train_episode_index

PROJECT = Path(__file__).resolve().parents[1]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cpu")
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument("--prefix-stride", type=int, default=1)
    parser.add_argument("--max-prefixes-per-episode", type=int)
    parser.add_argument(
        "--selected-prefixes",
        "--selected_prefixes",
        type=Path,
        default=None,
        dest="selected_prefixes",
    )
    parser.add_argument("--rollin-policy", type=str, default="OFF")
    parser.add_argument("--rollin-seed", type=int, default=0)
    parser.add_argument("--max-units", type=float, default=DEFAULT_MAX_UNITS)
    parser.add_argument(
        "--renderer-samples",
        "--render-samples",
        type=int,
        default=DEFAULT_RENDER_SAMPLES,
        dest="renderer_samples",
    )
    parser.add_argument(
        "--ray-chunk-size",
        type=int,
        default=DEFAULT_RAY_CHUNK_SIZE,
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)

    index_path = args.index.resolve()
    entries = load_train_episode_index(index_path)
    selected = _load_selected_prefixes(args.selected_prefixes)
    result = collect_action_targets(
        entries,
        args.checkpoint,
        args.output,
        index_path=index_path,
        device=_resolve_device(args.device),
        horizon=args.horizon,
        max_episodes=args.max_episodes,
        prefix_stride=args.prefix_stride,
        max_prefixes_per_episode=args.max_prefixes_per_episode,
        selected_prefixes=selected,
        rollin_policy=args.rollin_policy,
        rollin_seed=args.rollin_seed,
        max_units=args.max_units,
        renderer_samples=args.renderer_samples,
        ray_chunk_size=args.ray_chunk_size,
        resume=args.resume,
    )
    print(json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False))
    return 0


def _resolve_device(value: str) -> torch.device:
    if value == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        return torch.device("cuda")
    if value == "auto" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _load_selected_prefixes(path: Path | None) -> dict[str, list[int]] | None:
    if path is None:
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid selected-prefix JSON: {path}") from error
    if isinstance(value, Mapping):
        if set(value) == {"episodes"}:
            value = value["episodes"]
        if not isinstance(value, Mapping):
            raise ValueError("selected-prefix episodes must be an object")
        return _normalize_selected_mapping(value)
    if isinstance(value, list):
        result: dict[str, list[int]] = {}
        for index, record in enumerate(value):
            if not isinstance(record, Mapping):
                raise ValueError(f"selected-prefix record {index} must be an object")
            if set(record) != {"episode_id", "prefix_steps"}:
                raise ValueError(
                    f"selected-prefix record {index} fields must be episode_id and prefix_steps"
                )
            episode_id = record["episode_id"]
            if not isinstance(episode_id, str) or not episode_id:
                raise ValueError(f"selected-prefix record {index} has an invalid episode_id")
            if episode_id in result:
                raise ValueError(f"selected-prefix record duplicates episode_id: {episode_id}")
            result[episode_id] = _normalize_steps(record["prefix_steps"], episode_id)
        return result
    raise ValueError("selected-prefix JSON must be an episode mapping or record list")


def _normalize_selected_mapping(value: Mapping[object, object]) -> dict[str, list[int]]:
    result: dict[str, list[int]] = {}
    for episode_id, steps in value.items():
        if not isinstance(episode_id, str) or not episode_id:
            raise ValueError("selected-prefix episode IDs must be nonempty strings")
        if episode_id in result:
            raise ValueError(f"selected-prefix mapping duplicates episode_id: {episode_id}")
        result[episode_id] = _normalize_steps(steps, episode_id)
    return result


def _normalize_steps(value: object, episode_id: str) -> list[int]:
    if isinstance(value, (str, bytes)) or not isinstance(value, list):
        raise ValueError(f"selected prefixes for {episode_id!r} must be a list of integers")
    if any(type(step) is not int or step < 0 for step in value):
        raise ValueError(f"selected prefixes for {episode_id!r} must be nonnegative integers")
    if value != sorted(set(value)) or not value:
        raise ValueError(f"selected prefixes for {episode_id!r} must be sorted and unique")
    return list(value)


if __name__ == "__main__":
    raise SystemExit(main())
