"""Initialize an empty run from the frozen 16-grid bounds protocol; no media reads."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path

from mcss.mechanism_pilot.small_training import sha, write_json

FILES = (
    "preregistration.json",
    "config.json",
    "state_plan.json",
    "scene_manifest.json",
    "bounds_contract.json",
    "bounds_lock.json",
    "optimization_budget_lock.json",
    "decision_rules.json",
    "train_depth_prior.json",
    "baseline_results.json",
    "baseline_reproduction.json",
    "audit/analysis_source_lock.json",
    "audit/previous_experiment_seal.json",
)


def reference_bytes(reference, name):
    path = reference / name
    if path.is_file():
        return path.read_bytes()
    compressed = Path(str(path) + ".gz")
    if compressed.is_file():
        return gzip.decompress(compressed.read_bytes())
    raise FileNotFoundError(path)


def initialize(reference, output):
    reference, output = Path(reference), Path(output)
    if output.exists():
        raise FileExistsError("A new empty run directory is required")
    payloads = {name: reference_bytes(reference, name) for name in FILES}
    prereg = json.loads((reference / "preregistration.json").read_text())
    for name, digest in prereg["locked_file_sha256"].items():
        if sha(reference / name) != digest:
            raise PermissionError(f"Frozen protocol changed: {name}")
    for record in ("config.json", "audit/analysis_source_lock.json"):
        for path, digest in json.loads((reference / record).read_text())["source_sha256"].items():
            if sha(Path(path)) != digest:
                raise PermissionError(f"Frozen source changed: {path}")
    if json.loads((reference / "baseline_reproduction.json").read_text())["status"] != "PASS":
        raise PermissionError("Baseline reproduction must pass")
    output.mkdir(parents=True)
    for name in FILES:
        destination = output / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payloads[name])
    for name in ("raw", "checkpoints", "figures"):
        (output / name).mkdir()
    write_json(
        output / "audit/reproduction_provenance.json",
        {
            "reference": str(reference.resolve()),
            "copied_file_sha256": {
                name: hashlib.sha256(payloads[name]).hexdigest() for name in FILES
            },
            "historical_baseline_and_bounds_reused": True,
            "new_optimization_required": True,
            "media_read": False,
            "historical_timestamps_not_reproduced": True,
        },
    )
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(initialize(args.reference, args.output))
