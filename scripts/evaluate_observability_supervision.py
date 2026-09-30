"""Evaluate frozen context-supervision states after every phase state is sealed."""

import argparse
import json
from pathlib import Path

from mcss.mechanism_pilot.observability_supervision_evaluation import evaluate_supervision

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--phase", choices=("baseline", "formal"), required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    rows = evaluate_supervision(args.root, args.phase, args.device)
    print({"phase": args.phase, "rows": len(rows)})

    if args.phase == "baseline":
        reproduction = json.loads((Path(args.root) / "baseline_reproduction.json").read_text())
        print({"BASELINE_REPRO_STATUS": reproduction["status"]})
        if reproduction["status"] != "PASS":
            raise SystemExit(2)
