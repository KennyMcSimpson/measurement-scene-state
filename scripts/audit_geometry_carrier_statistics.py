#!/usr/bin/env python3
"""Re-execute saved-raw analysis in isolation and byte-compare scientific JSON.

This independently executes the frozen analyzer; it does not claim a separately
implemented statistical estimator. No query media, weights or models are loaded.
"""

import argparse
import gzip
import hashlib
import importlib.util
import json
import tempfile
from pathlib import Path

SCIENTIFIC_JSON = (
    "static_results.json",
    "wrong_scene_results.json",
    "state_use_results.json",
    "bootstrap_results.json",
    "qualification_results.json",
    "observability_analysis.json",
)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def saved_bytes(path):
    if path.is_file():
        return path.read_bytes(), path
    compressed = Path(str(path) + ".gz")
    return gzip.decompress(compressed.read_bytes()), compressed


def audit(root, output=None):
    root = Path(root).resolve()
    output = Path(output) if output else root / "audit/statistics_reproduction_audit.json"
    # Snapshot originals before analysis; the runner receives a separate output directory.
    expected = {name: saved_bytes(root / name) for name in SCIENTIFIC_JSON}
    analyzer = Path(__file__).with_name("analyze_geometry_carrier.py").resolve()
    spec = importlib.util.spec_from_file_location(
        "geometry_carrier_reproduction_analyzer", analyzer
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    comparisons = {}
    with tempfile.TemporaryDirectory(prefix="geometry-carrier-statistics-audit-") as directory:
        reproduced = Path(directory)
        module.run(root, reproduced)
        provenance = json.loads((reproduced / "statistics_reproduction.json").read_text())
        for name, (original, original_path) in expected.items():
            actual = (reproduced / name).read_bytes()
            comparisons[name] = {
                "byte_exact": actual == original,
                "original_decoded_sha256": digest(original),
                "reproduced_sha256": digest(actual),
                "original_file": str(original_path),
            }
            current, _ = saved_bytes(root / name)
            if current != original:
                raise RuntimeError(f"Original scientific result changed during audit: {name}")
        # Provenance identifies the actual plain/gzip raw used. Check it remains unchanged.
        for mapping in ("input_sha256", "source_sha256"):
            for path, sha256 in provenance[mapping].items():
                if digest(Path(path).read_bytes()) != sha256:
                    raise RuntimeError(f"Analysis input/source changed during audit: {path}")
    passed = all(row["byte_exact"] for row in comparisons.values())
    report = {
        "status": "PASS" if passed else "FAIL",
        "scientific_json_count": len(SCIENTIFIC_JSON),
        "all_scientific_json_byte_exact": passed,
        "comparisons": comparisons,
        "raw_only": True,
        "media_or_checkpoint_loaded": False,
        "new_model_forward": False,
        "original_artifacts_modified": False,
        "reproduction_inputs_sha256": provenance["input_sha256"],
        "reproduction_source_sha256": provenance["source_sha256"],
        "audit_script_sha256": digest(Path(__file__).read_bytes()),
        "FINAL_HOLDOUT_TOUCHED": False,
        "meaning": (
            "Independent raw replay of same estimator, not independent estimator implementation"
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if not passed:
        raise RuntimeError("Scientific JSON reproduction failed; report preserved")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.root, args.output), indent=2))
