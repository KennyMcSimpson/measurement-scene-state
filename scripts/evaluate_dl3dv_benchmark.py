"""Evaluate a dynamic checkpoint on the pinned DL3DV-140 benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mcss.evaluation.dl3dv_report import evaluate_dl3dv_benchmark


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--mode", choices=("full", "ar"), default="full")
    parser.add_argument("--input-count", type=int, default=32)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--policy",
        choices=("OFF", "FUSE", "COMPLETE", "ALL", "RESIDUAL_THRESHOLD", "LEARNED"),
        default="OFF",
    )
    parser.add_argument("--policy-checkpoint", type=Path)
    parser.add_argument("--residual-threshold", type=float, default=0.03)
    parser.add_argument("--max-units", type=float, default=1e12)
    parser.add_argument(
        "--max-scenes",
        type=int,
        help="diagnostic subset size; its output is explicitly partial",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--without-lpips",
        action="store_true",
        help="omit the optional VGG-LPIPS metric for an explicitly reduced diagnostic run",
    )
    parser.add_argument("--data-hash")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = evaluate_dl3dv_benchmark(
        args.metadata,
        args.checkpoint,
        args.output,
        mode=args.mode,
        input_count=args.input_count,
        device=args.device,
        policy=args.policy,
        policy_checkpoint=args.policy_checkpoint,
        residual_threshold=args.residual_threshold,
        max_units=args.max_units,
        max_scenes=args.max_scenes,
        resume=args.resume,
        include_lpips=not args.without_lpips,
        data_hash=args.data_hash,
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "benchmark_claim": report["benchmark_claim"],
                "output": str(args.output),
                "failure_counts": report["failure_counts"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
