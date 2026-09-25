"""Run the configured two-stage, train-only dynamic carrier experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from mcss.training.experiment import (
    DynamicExperimentConfig,
    dynamic_experiment_config_from_mapping,
    run_dynamic_experiment,
)

PROJECT = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--episode-index",
        type=Path,
        default=PROJECT / "outputs/dynamic_training_run_20260919/episodes/episode_index_train.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT / "outputs/dynamic_training_run_20260919/training",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--config-json",
        type=Path,
        help=(
            "JSON object containing DynamicExperimentConfig fields, including "
            "nested carrier/write"
        ),
    )
    parser.add_argument("--init-checkpoint", type=Path)
    parser.add_argument("--prepared-root", type=Path)
    parser.add_argument("--stage-a-steps", type=int)
    parser.add_argument("--stage-b-steps", type=int)
    parser.add_argument("--renderer-samples", type=int)
    parser.add_argument("--ray-chunk-size", type=int)
    parser.add_argument("--warmup-steps", type=int)
    parser.add_argument("--stream-steps", type=int)
    parser.add_argument("--query-count", type=int)
    parser.add_argument("--action-schedule-mode", choices=("legacy_fixed", "mixed"))
    parser.add_argument("--stage-b-actions", nargs="+")
    args = parser.parse_args(argv)

    output_dir = args.output_dir.resolve()
    output_dir.relative_to(PROJECT / "outputs")
    overrides = {
        name: value
        for name, value in {
            "stage_a_steps": args.stage_a_steps,
            "stage_b_steps": args.stage_b_steps,
            "renderer_samples": args.renderer_samples,
            "ray_chunk_size": args.ray_chunk_size,
            "warmup_steps": args.warmup_steps,
            "stream_steps": args.stream_steps,
            "query_count": args.query_count,
            "action_schedule_mode": args.action_schedule_mode,
            "stage_b_actions": args.stage_b_actions,
        }.items()
        if value is not None
    }
    if args.config_json is None:
        config = DynamicExperimentConfig(**overrides)
    else:
        config = dynamic_experiment_config_from_mapping(
            json.loads(args.config_json.read_text(encoding="utf-8")),
            overrides=overrides,
        )
    run_kwargs = {
        "device": _resolve_device(args.device),
        "resume": args.resume,
        "config": config,
        "init_checkpoint": args.init_checkpoint,
    }
    if args.prepared_root is not None:
        run_kwargs["prepared_root"] = args.prepared_root
    result = run_dynamic_experiment(
        args.episode_index,
        output_dir,
        **run_kwargs,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _resolve_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        return torch.device("cuda")
    if requested == "auto" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


if __name__ == "__main__":
    raise SystemExit(main())
