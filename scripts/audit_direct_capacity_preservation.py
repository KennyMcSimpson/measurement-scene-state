#!/usr/bin/env python3
"""Recheck frozen predecessors without reopening protected holdout media bytes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mcss.mechanism_pilot.small_training import sha, write_json

CHECKPOINT = Path(
    "outputs/EXP-3D-20260927-centered-training-1000a-600b-v1/training/phase_b_final.pt"
)
EXPECTED = "be7b8b6d2ef366cad245732b9ff227802db596da65082ac0e160f24541579e42"


def audit(root):
    root = Path(root)
    seal_path = root / "audit/old_artifact_seal.json"
    seal = json.loads(seal_path.read_text())
    failures, heldout_metadata_only = [], []
    fresh = 0
    for name, record in seal["files"].items():
        path = Path(name)
        if not path.is_file():
            failures.append({"path": name, "reason": "missing"})
            continue
        stat = path.stat()
        if record["mode"].startswith("previously_verified"):
            heldout_metadata_only.append(name)
            if stat.st_size != record["size"] or stat.st_mtime_ns != record["mtime_ns"]:
                failures.append({"path": name, "reason": "protected_media_stat_changed"})
            continue
        fresh += 1
        if sha(path) != record["sha256"]:
            failures.append({"path": name, "reason": "SHA256_changed"})
    current_hash = sha(CHECKPOINT)
    baseline = json.loads((root / "baseline_integrity.json").read_text())
    config_path = root / "config.json"
    config = json.loads(config_path.read_text()) if config_path.exists() else {}
    frozen_source_failures = [
        name
        for name, digest in config.get("source_sha256", {}).items()
        if sha(Path(name)) != digest
    ]
    report = {
        "status": "PASS"
        if not failures
        and not frozen_source_failures
        and current_hash == EXPECTED
        and baseline["status"] == "PASS"
        else "FAIL",
        "old_file_count": len(seal["files"]),
        "freshly_rehashed_old_files": fresh,
        "protected_media_metadata_only_count": len(heldout_metadata_only),
        "protected_media_bytes_reopened": False,
        "protected_media_hash_assurance": (
            "Inherited earlier verified digest plus unchanged file size/mtime; "
            "not claimed as a fresh byte-level rehash this round."
        ),
        "old_artifact_failures": failures,
        "frozen_optimizer_source_failures": frozen_source_failures,
        "checkpoint_sha256": current_hash,
        "checkpoint_expected_sha256": EXPECTED,
        "baseline_rows": baseline["rows"],
        "baseline_max_metric_absolute_delta": baseline["max_metric_absolute_delta"],
        "baseline_camera_hashes_exact": baseline["camera_hashes_all_exact"],
        "baseline_prediction_hashes_exact": baseline["prediction_hashes_all_exact"],
        "old_artifact_seal_sha256": sha(seal_path),
        "new_carrier_trained": False,
        "dynamic_ttt_run": False,
    }
    write_json(root / "audit/preservation_recheck.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.root), indent=2))
