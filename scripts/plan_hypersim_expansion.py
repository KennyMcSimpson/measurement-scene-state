"""Create and size-check the official-split Hypersim expansion plan."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mcss.data.hypersim_expansion import (
    build_expansion_plan,
    estimate_expansion_storage,
    write_expansion_plan,
)

GIB = 1024**3


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("data/hypersim_raw/official_metadata/metadata_images_split_scene_v1.csv"),
    )
    parser.add_argument(
        "--observed-subset", type=Path, default=Path("configs/hypersim_subset_v1.csv")
    )
    parser.add_argument(
        "--partitions-output", type=Path, default=Path("configs/hypersim_er_partitions.csv")
    )
    parser.add_argument(
        "--exclusions-output",
        type=Path,
        default=Path("configs/hypersim_er_final_exclusions.csv"),
    )
    parser.add_argument("--raw-root", type=Path, default=Path("data/hypersim_raw"))
    parser.add_argument(
        "--baseline-prepared-root", type=Path, default=Path("data/hypersim_prepared")
    )
    parser.add_argument(
        "--target-prepared-root", type=Path, default=Path("data/hypersim_er_prepared")
    )
    parser.add_argument("--reserve-gib", type=float, default=100.0)
    parser.add_argument("--safety-factor", type=float, default=1.25)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.reserve_gib < 0:
        raise ValueError("--reserve-gib must be non-negative")
    plan = build_expansion_plan(args.metadata, args.observed_subset)
    estimate = estimate_expansion_storage(
        plan,
        raw_root=args.raw_root,
        baseline_prepared_root=args.baseline_prepared_root,
        target_prepared_root=args.target_prepared_root,
        reserve_bytes=int(args.reserve_gib * GIB),
        safety_factor=args.safety_factor,
    )
    if not args.dry_run:
        write_expansion_plan(plan, args.partitions_output, args.exclusions_output)
    print(
        json.dumps(
            {
                "dry_run": args.dry_run,
                "metadata_sha256": plan.metadata_sha256,
                "official_scene_counts": plan.official_counts,
                "protocol_scene_counts": plan.counts,
                "storage": estimate.as_dict(),
                "outputs": None
                if args.dry_run
                else {
                    "partitions": str(args.partitions_output.resolve()),
                    "exclusions": str(args.exclusions_output.resolve()),
                },
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if estimate.fits_budget else 2


if __name__ == "__main__":
    raise SystemExit(main())
