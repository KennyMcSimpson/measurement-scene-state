"""Evaluate one strict dynamic checkpoint across selected development episodes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from mcss.evaluation.checkpoint_report import (
    DEFAULT_POLICIES,
    DEFAULT_RAY_CHUNK_SIZE,
    POLICY_NAMES,
    evaluate_dynamic_checkpoint,
    load_episode_index,
)

PROJECT = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--index",
        type=Path,
        default=PROJECT / "outputs/dynamic_ttt_pilot_20260919/episodes/episode_index.json",
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--render-samples", type=int, default=16)
    parser.add_argument("--ray-chunk-size", type=int, default=DEFAULT_RAY_CHUNK_SIZE)
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument("--residual-threshold", type=float, default=0.03)
    parser.add_argument("--policy-checkpoint", type=Path)
    parser.add_argument("--warmup-count", type=int, default=4)
    parser.add_argument("--stream-count", type=int, default=8)
    parser.add_argument("--query-count", type=int, default=4)
    parser.add_argument(
        "--policies",
        nargs="+",
        choices=sorted(POLICY_NAMES),
        default=list(DEFAULT_POLICIES),
    )
    args = parser.parse_args(argv)
    output = args.output.resolve()
    output.relative_to(PROJECT / "outputs")
    torch.set_num_threads(4)
    report = evaluate_dynamic_checkpoint(
        load_episode_index(args.index),
        args.checkpoint,
        output,
        device=args.device,
        policies=args.policies,
        render_samples=args.render_samples,
        ray_chunk_size=args.ray_chunk_size,
        residual_threshold=args.residual_threshold,
        max_episodes=args.max_episodes,
        policy_checkpoint=args.policy_checkpoint,
        warmup_count=args.warmup_count,
        stream_count=args.stream_count,
        query_count=args.query_count,
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "report": str(output / "report.json"),
                "summary": str(output / "summary.csv"),
                "records": len(report["records"]),
                "actual_subset": report["actual_subset"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
