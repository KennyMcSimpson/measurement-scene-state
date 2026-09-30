"""Copy an already frozen capacity protocol into a new, empty run directory.

This does not tune, open media, or replay historical pilot choices. Data paths in
scene_manifest remain the archived paths; the referenced data must be available.
"""

import argparse
import json
import shutil
from pathlib import Path

from mcss.mechanism_pilot.small_training import sha, write_json

FILES = (
    "preregistration.json",
    "config.json",
    "state_plan.json",
    "scene_manifest.json",
    "capacity_contract.json",
    "optimization_contract.json",
    "renderer_contract.json",
    "resolution_contract.json",
    "predeclared_geometry_bounds.json",
    "train_depth_prior.json",
    "baseline_results.json",
    "baseline_integrity.json",
    "optimization_sanity.json",
)


def initialize(reference, output):
    reference, output = Path(reference), Path(output)
    if output.exists():
        raise FileExistsError("Use a new directory; never overwrite an experiment")
    for name in FILES:
        if not (reference / name).is_file():
            raise FileNotFoundError(reference / name)
    config = json.loads((reference / "config.json").read_text())
    prereg = json.loads((reference / "preregistration.json").read_text())
    if sha(reference / "config.json") != prereg["config_sha256"]:
        raise PermissionError("Reference config changed")
    if sha(reference / "state_plan.json") != prereg["state_plan_sha256"]:
        raise PermissionError("Reference state plan changed")
    if sha(reference / "scene_manifest.json") != config["scene_manifest_sha256"]:
        raise PermissionError("Reference scene manifest changed")
    for path, digest in config["source_sha256"].items():
        if sha(Path(path)) != digest:
            raise PermissionError(f"Frozen source changed: {path}")
    if json.loads((reference / "baseline_integrity.json").read_text())["status"] != "PASS":
        raise PermissionError("Baseline replay must have passed")
    output.mkdir(parents=True)
    for name in FILES:
        shutil.copyfile(reference / name, output / name)
    for sub in ("audit", "raw", "figures", "checkpoints"):
        (output / sub).mkdir()
    write_json(
        output / "audit/reproduction_provenance.json",
        {
            "reference": str(reference.resolve()),
            "copied_file_sha256": {name: sha(reference / name) for name in FILES},
            "historical_baseline_and_pilot_reused": True,
            "new_direct_state_optimization_required": True,
            "media_read": False,
        },
    )
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(initialize(args.reference, args.output))
