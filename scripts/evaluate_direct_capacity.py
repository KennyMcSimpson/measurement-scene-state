"""Evaluate only a fully completed, sealed direct-capacity phase."""

import argparse

from mcss.mechanism_pilot.direct_capacity_evaluation import evaluate_capacity

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--phase", required=True, choices=("context", "oracle"))
    parser.add_argument("--device", default="cuda", choices=("cpu", "cuda"))
    args = parser.parse_args()
    rows = evaluate_capacity(args.root, phase=args.phase, device=args.device)
    print(f"{args.phase}: {len(rows)} evaluation rows completed")
