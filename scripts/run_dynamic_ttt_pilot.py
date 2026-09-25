"""Run modular untrained train/dev controls; never use this as a final test entry point."""

import argparse
import json
from pathlib import Path

from mcss.data.pilot_provenance import validate_pilot_entries, write_pilot_provenance
from mcss.evaluation.streaming_report import run_pilot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument(
        "--policies",
        nargs="+",
        default=["OFF", "FUSE", "COMPLETE", "ALL"],
        choices=["OFF", "FUSE", "COMPLETE", "ALL", "RESIDUAL_THRESHOLD"],
    )
    args = parser.parse_args()
    data = json.loads(args.index.read_text(encoding="utf-8"))
    index = data["episodes"] if isinstance(data, dict) else data
    if args.max_episodes is not None:
        if args.max_episodes < 1:
            parser.error("--max-episodes must be positive")
        index = index[: args.max_episodes]
    provenance = validate_pilot_entries(index)
    report = run_pilot(
        index, args.output, device=args.device, seed=args.seed, policies=args.policies
    )
    provenance_path = write_pilot_provenance(args.output / "manifest_provenance.json", provenance)
    print(
        json.dumps(
            {
                "status": report["status"],
                "runs": report["runs"],
                "report": str(args.output / "report.json"),
                "manifest_provenance": str(provenance_path),
            }
        )
    )


if __name__ == "__main__":
    main()
