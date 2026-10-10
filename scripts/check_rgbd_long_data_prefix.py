#!/usr/bin/env python3
"""V18 built-in control: steps 1-6000 of every seed must equal V14 C1 bit-for-bit.

For each seed: training.jsonl lines 1-6000 equal V14 C1's except the wall-clock 'seconds' field,
and the step-6000 parameters equal V14 C1's step_006000.pt tensor by tensor (the files differ
because each records its own preregistration hash). Writes audit/v14_prefix_control.json and exits
nonzero on any mismatch, which stops the runner before any EVAL prediction is made.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from mcss.mechanism_pilot.geometry_carrier_experiment import verify_lock
from mcss.mechanism_pilot.small_training import sha, write_json

PREFIX = 6000


def strip(row):
    return {k: v for k, v in row.items() if k != "seconds"}


def compare_logs(mine, theirs, steps=PREFIX):
    if not Path(mine).is_file():
        return {"equal": False, "reason": f"missing {Path(mine).name}"}
    a = Path(mine).read_text().splitlines()[:steps]
    b = Path(theirs).read_text().splitlines()[:steps]
    if len(a) != steps or len(b) != steps:
        return {"equal": False, "reason": f"lines {len(a)} vs {len(b)}"}
    for i, (x, y) in enumerate(zip(a, b, strict=True), start=1):
        if strip(json.loads(x)) != strip(json.loads(y)):
            return {"equal": False, "first_mismatch_step": i}
    return {"equal": True, "steps": steps}


def compare_parameters(mine, theirs):
    if not Path(mine).is_file():
        return {"equal": False, "reason": f"missing {Path(mine).name}"}
    a = torch.load(mine, map_location="cpu", weights_only=True)["carrier"]
    b = torch.load(theirs, map_location="cpu", weights_only=True)["carrier"]
    if sorted(a) != sorted(b):
        return {"equal": False, "reason": "parameter names differ"}
    unequal = [k for k in a if not torch.equal(a[k], b[k])]
    return {"equal": not unequal, "tensors": len(a), "unequal": unequal}


def check(root):
    root = Path(root).resolve()
    _, config = verify_lock(root)
    results, ok = {}, True
    for seed in config["seeds"]:
        ref = config["control"]["references"][str(seed)]
        if (
            sha(Path(ref["training_log"])) != ref["training_log_sha256"]
            or sha(Path(ref["step_006000"])) != ref["step_006000_sha256"]
        ):
            raise PermissionError(f"V14 C1 control reference changed: {seed}")
        run = root / "checkpoints" / f"L_{seed}"
        logs = compare_logs(run / "training.jsonl", ref["training_log"])
        params = compare_parameters(run / "step_006000.pt", ref["step_006000"])
        results[str(seed)] = {"training_log": logs, "step_006000_parameters": params}
        ok = ok and logs["equal"] and params["equal"]
    status = "PASS" if ok else "FAIL"
    write_json(root / "audit/v14_prefix_control.json", {"status": status, "seeds": results})
    print("V14_PREFIX_CONTROL", status, json.dumps(results), flush=True)
    return ok


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    sys.exit(0 if check(parser.parse_args().root) else 1)
