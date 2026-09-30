#!/usr/bin/env python3
"""Evaluate frozen budget states only after every planned phase chain has completed."""

import argparse
import json

from mcss.mechanism_pilot.optimization_bounds_evaluation import evaluate_optimization_bounds

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True)
    p.add_argument("--phase", choices=["context", "oracle", "secondary"], default="context")
    p.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    a = p.parse_args()
    rows = evaluate_optimization_bounds(a.root, a.phase, a.device)
    print(json.dumps({"status": "PASS", "phase": a.phase, "rows": len(rows)}))
