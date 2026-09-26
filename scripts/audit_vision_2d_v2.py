#!/usr/bin/env python3
"""Audit frozen V1 artifacts and V2 locks; never read annotation/media datasets."""

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", default="/tmp/mcss_v2_paths.json")
    args = parser.parse_args()
    paths = json.loads(Path(args.paths).read_text())
    report = Path(paths["report"])
    original = json.loads((report / "v1_artifact_hashes.json").read_text())
    checked = {p: (ROOT / p).is_file() and sha(ROOT / p) == h for p, h in original.items()}
    old_lock = json.loads(
        (
            ROOT / "docs/experiments/EXP-2D-20260926-opportunity-selector-v1/validation_lock.json"
        ).read_text()
    )
    # Enumerate the originally locked paths only; newly added V2 modules do not alter V1.
    source_map = old_lock.get("source_hashes", old_lock.get("sources", {}))
    if not source_map:
        source_map = old_lock.get("hashes", {})
    source_checks = {p: (ROOT / p).is_file() and sha(ROOT / p) == h for p, h in source_map.items()}
    prereg = json.loads((report / "preregistration.json").read_text())
    checks = {
        "all_v1_artifacts_unchanged": all(checked.values()),
        "v1_locked_sources_unchanged": bool(source_checks) and all(source_checks.values()),
        "config_matches_preregistration": sha(report / "config.json") == prereg["config_sha256"],
        "old_raw_audit_pass": json.loads((report / "old_raw_audit.json").read_text())["status"]
        == "PASS",
    }
    implementation = json.loads((report / "implementation_lock.json").read_text())
    implementation_checks = {
        p: Path(p).is_file() and sha(p) == digest
        for p, digest in implementation["source_hashes"].items()
    }
    selector = json.loads((report / "selector_lock.json").read_text())
    access = json.loads((report / "discovery_access.json").read_text())
    independent = json.loads((report / "independent_data_audit.json").read_text())
    predictions = json.loads((report / "predictions_manifest.json").read_text())
    selectors = json.loads((report / "selectors.json").read_text())
    rows = [json.loads(line) for line in (report / "discovery_rows.jsonl").read_text().splitlines()]
    checks.update(
        {
            "v2_implementation_frozen": bool(implementation_checks)
            and all(implementation_checks.values()),
            "selector_artifact_frozen": sha(report / "selectors.json")
            == selector["selectors_sha256"],
            "discovery_raw_frozen": sha(report / "discovery_rows.jsonl")
            == selector["discovery_sha256"],
            "implementation_binds_selector_lock": sha(report / "selector_lock.json")
            == implementation["selector_lock_sha256"],
            "discovery_only_target_gt_reads_zero": access["target_reads"] == [],
            "source_annotation_reads_24": len(access["source_reads"]) == 24,
            "discovery_raw_coverage": len(rows) == 384 and len({r["sequence"] for r in rows}) == 12,
            "exact_50_51_feature_contract": len(selectors["GateOnly"]["model"]["feature_names"])
            == 50
            and len(selectors["CycleGate"]["model"]["feature_names"]) == 51,
            "protected_sets_untouched": independent["reserve_touched"] is False
            and independent["davis_official_val_touched"] is False,
        }
    )
    if independent["status"] == "BLOCKED_NO_INDEPENDENT_DATA":
        checks["no_fabricated_confirmation"] = (
            predictions["complete"] is False
            and predictions["predictions"] == []
            and not (report / "raw_results.jsonl").exists()
            and not (report / "prediction_lock.json").exists()
            and not (report / "evaluation_started.json").exists()
        )
    logs = {}
    for p in sorted(report.glob("*access*.json")):
        logs[p.name] = json.loads(p.read_text())
    for p in sorted(report.glob("*target*log*.json")):
        logs[p.name] = json.loads(p.read_text())
    result = {
        "schema": "mcss.v2.integrity.v1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "v1_artifact_count": len(checked),
        "v1_artifact_checks": checked,
        "v1_source_checks": source_checks,
        "v2_implementation_checks": implementation_checks,
        "access_evidence": logs,
        "scope": (
            "artifact immutability/config audit; confirmation state and target lock "
            "audited separately by runner"
        ),
        "hashes": {
            str(p.relative_to(report)): sha(p)
            for p in sorted(report.rglob("*"))
            if p.is_file() and p.name not in ("integrity.json", "CHECKSUMS.sha256")
        },
    }
    (report / "integrity.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "status": result["status"],
                "checks": checks,
                "v1_artifact_count": len(checked),
                "source_count": len(source_checks),
            }
        )
    )
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
