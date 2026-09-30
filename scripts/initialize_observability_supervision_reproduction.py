"""Create a fresh run from the frozen protocol, without any media or model access."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path

from mcss.mechanism_pilot.small_training import sha, write_json


def payload(reference, name):
    p = reference / name
    if p.is_file():
        return p.read_bytes()
    return gzip.decompress(Path(str(p) + ".gz").read_bytes())


def initialize(reference, output):
    reference, output = Path(reference), Path(output)
    if output.exists():
        raise FileExistsError("A fresh output directory is required")
    prereg = json.loads(payload(reference, "preregistration.json"))
    names = [
        *prereg["locked_file_sha256"],
        "preregistration.json",
        "audit/previous_experiment_seal.json",
        "audit/historical_baseline_raw_reproduction.json",
    ]
    values = {name: payload(reference, name) for name in names}
    for name, h in prereg["locked_file_sha256"].items():
        if hashlib.sha256(values[name]).hexdigest() != h:
            raise PermissionError("Frozen protocol changed: " + name)
    config = json.loads(values["config.json"])
    for p, h in config["source_sha256"].items():
        if sha(Path(p)) != h:
            raise PermissionError("Use archived scientific source: " + p)
    output.mkdir(parents=True)
    for name, b in values.items():
        p = output / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b)
    for name in ("raw", "audit", "figures", "checkpoints"):
        (output / name).mkdir(exist_ok=True)
    write_json(
        output / "audit/reproduction_provenance.json",
        {
            "reference": str(reference.resolve()),
            "copied_sha256": {name: hashlib.sha256(b).hexdigest() for name, b in values.items()},
            "historical_protocol_replayed": True,
            "new_optimization_required": True,
            "historical_timestamps_not_recreated": True,
            "media_or_weights_read": False,
        },
    )
    return output


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    print(initialize(a.reference, a.output))
