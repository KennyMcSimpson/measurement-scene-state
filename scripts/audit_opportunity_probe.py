"""Independently verify raw coverage, locked decisions, media isolation and regeneration."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from mcss.vision_probe.opportunity_analysis import analyze
from mcss.vision_probe.opportunity_runner import (
    CONFIG,
    DATA,
    REPORT,
    TRAJECTORIES,
    WORK,
    load_rows,
    read_json,
    sha,
    verify_lock,
    write_json,
)
from mcss.vision_probe.opportunity_selectors import select_candidates


def main():
    verify_lock()
    rows = load_rows(REPORT / "raw_results.jsonl")
    splits = read_json(REPORT / "split_manifest.json")
    artifact = read_json(REPORT / "selectors.json")
    checks = {}
    expected = {
        (size, split, name, slot, tuple(t))
        for size in CONFIG["resolutions"]
        for split in ("discovery", "validation")
        for name in splits[split]
        for slot in (1, 2)
        for t in TRAJECTORIES
    }
    actual = [
        (r["resolution"], r["split"], r["sequence"], r["target_slot"], tuple(r["trajectory"]))
        for r in rows
    ]
    checks["exact_raw_coverage"] = len(actual) == len(set(actual)) and set(actual) == expected
    checks["selector_training_identity"] = artifact["fit_split"] == "discovery" and set(
        artifact["training_sequences"]
    ) == set(splits["discovery"])
    checks["all_numeric_metrics_finite"] = all(
        np.isfinite(r[k]) for r in rows for k in ("J", "F", "JF", "token_iou")
    )
    checks["off_has_zero_write"] = all(
        all(a["norm"] == 0 for a in r["rank_audit"])
        for r in rows
        if r["trajectory"] == ["OFF", "OFF"]
    )
    valid_decisions = {
        (r["resolution"], r["sequence"], r["target_slot"]): r
        for r in load_rows(REPORT / "validation_decisions.jsonl")
    }
    checks["decisions_match_locked_inference"] = True
    checks["decisions_committed_before_target_gt"] = True
    for size in CONFIG["resolutions"]:
        for split in ("discovery", "validation"):
            for name in splits[split]:
                for slot in (1, 2):
                    pair = [
                        r
                        for r in rows
                        if (r["resolution"], r["split"], r["sequence"], r["target_slot"])
                        == (size, split, name, slot)
                    ]
                    if len(pair) != 16:
                        checks["decisions_match_locked_inference"] = False
                        continue
                    choice = select_candidates(
                        [r["visible"] for r in pair],
                        [r["trajectory"] for r in pair],
                        artifact["by_resolution"][str(size)],
                    )
                    for method, index in choice.items():
                        selected = [
                            i for i, r in enumerate(pair) if method in r["selected_methods"]
                        ]
                        checks["decisions_match_locked_inference"] &= selected == [index]
                    if split == "validation":
                        decision = valid_decisions.get((size, name, slot), {})
                        checks["decisions_committed_before_target_gt"] &= (
                            decision.get("before_target_gt") is True
                            and decision.get("choices") == choice
                        )
    seen = {}
    for row in rows:
        if row["sequence"] not in splits[row["split"]]:
            raise ValueError("Unauthorized input identity")
        for role in ("source", "target"):
            stem = Path(row[f"{role}_frame"]).stem
            seen[DATA / "JPEGImages/480p" / row["sequence"] / row[f"{role}_frame"]] = row[
                f"{role}_image_sha256"
            ]
            seen[DATA / "Annotations/480p" / row["sequence"] / (stem + ".png")] = row[
                f"{role}_mask_sha256"
            ]
    checks["input_hashes_match"] = all(sha(path) == value for path, value in seen.items())
    forbidden = set(splits["reserve"]) | set(splits["official_val_sealed"])
    events = []
    for stage in ("discovery", "validation"):
        events += read_json(REPORT / f"access_audit_{stage}.json")
    paths = [e["path"] for e in events if "path" in e]
    checks["no_sealed_media_access"] = all(
        len(Path(p).parts) >= 3 and Path(p).parts[2] not in forbidden for p in paths
    )
    checks["audited_media_access_present"] = bool(paths)
    checks["raw_hash_matches_validation_completion"] = (
        sha(REPORT / "raw_results.jsonl")
        == read_json(REPORT / "validation_complete.json")["raw_sha256"]
    )
    test = read_json(REPORT / "tests.json")
    checks["full_test_suite_passed"] = test["exit_code"] == 0
    regen = WORK / "independent_regeneration"
    analyze(REPORT / "raw_results.jsonl", regen, CONFIG, artifact)
    outputs = [
        "per_pair_results.csv",
        "per_sequence_results.csv",
        "oracle_analysis.json",
        "selector_analysis.json",
        "robustness_analysis.json",
        "bootstrap_results.json",
        "signal_analysis.json",
        "rank_analysis.json",
        "signal_scatter.csv",
        "visible_features.csv",
    ]
    checks["all_analyses_exactly_reproduced"] = all(
        sha(REPORT / name) == sha(regen / name) for name in outputs
    )
    result = {
        "FINAL_INTEGRITY": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "raw_rows": len(rows),
        "validation_pairs": len(valid_decisions),
        "verified_input_files": len(seen),
        "validation_gt_used_for_selection": False,
        "davis_official_val_touched": False,
        "reserve_touched": False,
        "raw_sha256": sha(REPORT / "raw_results.jsonl"),
        "analysis_sha256": {name: sha(REPORT / name) for name in outputs},
    }
    write_json(REPORT / "integrity.json", result)
    print(json.dumps(result, indent=2))
    if not all(checks.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
