"""Regenerate V2 descriptive statistics exclusively from saved JSON artifacts.

No feature extraction, media reads, selector fitting or selector inference lives
here. Decisions must already be stored; missing decisions are reported rather
than reconstructed from task labels. Old validation remains exploratory.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from mcss.vision_probe.v2_statistics import TRAJECTORIES, global_diagnostic, selector_summary


def _read(path: Path) -> Any:
    return json.loads(path.read_text())


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _find(name: str, report: Path, work: Path) -> Path | None:
    return next((p for p in (report / name, work / name) if p.is_file()), None)


def _csv(path: Path, rows: list[dict]) -> None:
    fields = sorted(set().union(*(row.keys() for row in rows))) if rows else ["status"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    k: json.dumps(v, sort_keys=True) if isinstance(v, (list, dict)) else v
                    for k, v in row.items()
                }
            )


def _key(row: dict) -> tuple[str, str]:
    return str(row["sequence"]), str(row["target_slot"])


def _trajectory(value: Any) -> tuple[str, str]:
    return tuple(value.split("/")) if isinstance(value, str) else tuple(value)


def _macro(rows: list[dict], field: str) -> float | None:
    if not rows or any(field not in r or r[field] is None for r in rows):
        return None
    groups = defaultdict(list)
    for row in rows:
        groups[str(row["sequence"])].append(float(row[field]))
    return float(np.mean([np.mean(values) for values in groups.values()]))


def old_raw_diagnostics(v1: Path) -> tuple[dict, dict]:
    """Recheck saved V1 coverage/hashes/numeric claims without any media access."""
    raw = _rows(v1 / "raw_results.jsonl")
    artifacts = _read(v1 / "selectors.json")
    fixed = artifacts["by_resolution"]["448"]["discovery_best_fixed"]
    diagnostics = {}
    for split, name in (("discovery", "discovery448"), ("validation", "historical_validation448")):
        selected = [r for r in raw if r["resolution"] == 448 and r["split"] == split]
        diagnostics[name] = {
            **global_diagnostic(selected, fixed),
            "evidence_scope": "EXPLORATORY_OLD_RAW",
            "historical_split": split,
            "independent_confirmation": False,
        }
    val = [r for r in raw if r["resolution"] == 448 and r["split"] == "validation"]
    means = diagnostics["historical_validation448"]["means"]
    selectors = {
        m: selector_summary(val, [r for r in val if m in r["selected_methods"]], fixed)
        for m in ("rgb_selector", "visible_selector")
    }
    table = {
        "OFF/OFF": means["OFF"],
        "Discovery-fixed": means["discovery_fixed16"],
        "Constant oracle": means["constant_oracle"],
        "Dynamic oracle": means["dynamic_oracle"],
        "RGB selector": selectors["rgb_selector"]["mean_J"],
        "Visible selector": selectors["visible_selector"]["mean_J"],
    }
    readme = (v1 / "README.md").read_text()
    checks = []
    for method, value in table.items():
        match = re.search(r"^\| " + re.escape(method) + r" \| ([0-9.]+) \|", readme, re.M)
        reported = float(match.group(1)) if match else None
        checks.append(
            {
                "method": method,
                "reported": reported,
                "recomputed": value,
                "pass": reported is not None and abs(reported - value) <= 5.00001e-7,
            }
        )
    split = _read(v1 / "split_manifest.json")
    expected = {
        (size, part, seq, str(slot), t)
        for size in (224, 448)
        for part in ("discovery", "validation")
        for seq in split[part]
        for slot in (1, 2)
        for t in TRAJECTORIES
    }
    observed = [
        (r["resolution"], r["split"], r["sequence"], str(r["target_slot"]), tuple(r["trajectory"]))
        for r in raw
    ]
    digest = _sha(v1 / "raw_results.jsonl")
    integrity = _read(v1 / "integrity.json")
    completion = _read(v1 / "validation_complete.json")
    lock = _read(v1 / "validation_lock.json")
    validations = {
        "full_manifest_coverage": len(observed) == len(set(observed)) and set(observed) == expected,
        "README_rounding_matches": all(c["pass"] for c in checks),
        "raw_matches_integrity": digest == integrity["raw_sha256"],
        "raw_matches_completion": digest == completion["raw_sha256"],
        **{
            f"{name}_matches_lock": _sha(v1 / name) == lock[key]
            for name, key in (
                ("selectors.json", "selectors_sha256"),
                ("split_manifest.json", "split_sha256"),
                ("config.json", "config_sha256"),
            )
        },
    }
    status_path = v1 / "STATUS.md"
    status_checks = []
    if status_path.is_file():
        published = dict(re.findall(r"^([A-Z0-9_]+)=\n([^\n]+)", status_path.read_text(), re.M))
        val_diag = diagnostics["historical_validation448"]
        visible = selectors["visible_selector"]
        status_values = {
            "OFF_J": means["OFF"],
            "BEST_FIXED_J": means["discovery_fixed16"],
            "CONSTANT_ORACLE_J": means["constant_oracle"],
            "DYNAMIC_ORACLE_J": means["dynamic_oracle"],
            "DYNAMIC_ORACLE_MINUS_OFF": val_diag["gaps"]["write_opportunity"]["mean"],
            "DYNAMIC_ORACLE_MINUS_CONSTANT": val_diag["gaps"]["nonconstant_extra"]["mean"],
            "RGB_SELECTOR_J": selectors["rgb_selector"]["mean_J"],
            "VISIBLE_SELECTOR_J": visible["mean_J"],
            "VISIBLE_SELECTOR_MINUS_OFF": visible["net_gain"],
            "VISIBLE_SELECTOR_MINUS_FIXED": visible["comparisons"]["vs_discovery_fixed16"]["mean"],
            "VISIBLE_SELECTOR_ORACLE_CAPTURE": visible["positive_capture"],
            "HARMFUL_WRITE_RATE": visible["harmful_write_rate"],
            "TOP1_SEQUENCE_POSITIVE_GAIN_SHARE": val_diag["gaps"]["write_opportunity"][
                "top1_positive_share"
            ],
            "LEAVE_ONE_SEQUENCE_OUT_MIN_GAIN": val_diag["gaps"]["write_opportunity"]["loso_min"],
        }
        status_checks = [
            {
                "field": name,
                "reported": float(published[name]),
                "recomputed": value,
                "pass": abs(float(published[name]) - value) <= 1e-12,
            }
            for name, value in status_values.items()
        ]
        validations["STATUS_full_precision_matches"] = all(r["pass"] for r in status_checks)
    global_report = {
        "schema": "mcss.v2.old_raw_global16.v1",
        "source_raw_sha256": digest,
        "source_raw": str(v1 / "raw_results.jsonl"),
        "fixed_trajectory_provenance": {
            "path": str(v1 / "selectors.json"),
            "sha256": _sha(v1 / "selectors.json"),
            "trajectory": fixed,
            "fit_split": "discovery",
        },
        "diagnostics": diagnostics,
        "scope": "Old saved raw only; no media reads, selector inference or fitting",
    }
    audit = {
        "schema": "mcss.v2.old_raw_audit.v1",
        "status": "PASS" if all(validations.values()) else "FAIL",
        "checks": validations,
        "README_J_comparisons": checks,
        "STATUS_full_precision_comparisons": status_checks,
        "raw_rows": len(raw),
        "expected_rows": len(expected),
        "source_raw_sha256": digest,
        "selector_recomputations": selectors,
        "scope": "Numeric/artifact audit, not historical input-media re-audit or new confirmation",
    }
    return global_report, audit


def _selected(groups: dict, decisions: list[dict]) -> list[dict]:
    mapping = {}
    for decision in decisions:
        key = _key(decision)
        if key in mapping:
            raise ValueError(f"Duplicate saved decision: {key}")
        mapping[key] = _trajectory(decision["trajectory"])
    if set(mapping) != set(groups):
        raise ValueError("Stored decisions do not cover every raw pair exactly")
    return [
        next(r for r in candidates if tuple(r["trajectory"]) == mapping[key])
        for key, candidates in sorted(groups.items())
    ]


def _size_sensitivity(chosen: list[dict], off: list[dict]) -> dict:
    """Describe retained-object effects; never remove rows from primary scores."""
    baseline = {_key(row): row for row in off}
    buckets = defaultdict(list)
    for row in chosen:
        old = {o["object_id"]: o for o in baseline[_key(row)].get("objects", [])}
        for obj in row.get("objects", []):
            if obj["object_id"] not in old:
                raise ValueError("Object identities differ across candidate evaluations")
            item = {
                "sequence": row["sequence"],
                "target_slot": row["target_slot"],
                "object_id": obj["object_id"],
                "J": obj["J"],
                "gain": obj["J"] - old[obj["object_id"]]["J"],
                "pixel_count": obj.get("pixel_count"),
                "pixel_fraction": obj.get("pixel_fraction"),
                "token_count_224": obj.get("token_count_224"),
                "token_count_448": obj.get("token_count_448"),
            }
            buckets[obj["size_bucket"]].append(item)
    result = {}
    for bucket, objects in sorted(buckets.items()):
        pairs = defaultdict(list)
        for row in objects:
            pairs[_key(row)].append(row)
        pair_rows = [
            {"sequence": key[0], "gain": float(np.mean([x["gain"] for x in values]))}
            for key, values in pairs.items()
        ]
        result[bucket] = {
            "object_rows": objects,
            "n_objects": len(objects),
            "n_pairs_with_bucket": len(pairs),
            "n_sequences_with_bucket": len({r["sequence"] for r in objects}),
            "conditional_sequence_macro_gain": _macro(pair_rows, "gain"),
        }
    return {
        "status": "AVAILABLE" if buckets else "NO_OBJECT_ROWS_IN_RAW",
        "buckets": result,
        "scope": (
            "Descriptive conditional bucket means; all objects/pairs remain in primary analysis"
        ),
    }


def analyze_saved_rows(
    rows: list[dict],
    decisions: dict[str, list[dict]],
    fixed_trajectory: Any,
    *,
    seed=20260926,
    draws=10000,
    tol=1e-12,
) -> dict:
    """Analyze a single split/resolution, requiring stored deployment decisions."""
    diagnostic = global_diagnostic(rows, fixed_trajectory, seed, draws, tol)
    if diagnostic["status"] != "COMPLETE":
        return {"status": "RAW_INCOMPLETE", "global_diagnostic": diagnostic}
    groups = defaultdict(list)
    for row in rows:
        groups[_key(row)].append(row)
    for group in groups.values():
        group.sort(key=lambda r: TRAJECTORIES.index(tuple(r["trajectory"])))
    off = [next(r for r in g if tuple(r["trajectory"]) == ("OFF", "OFF")) for g in groups.values()]
    methods = dict(decisions)
    methods["OFF"] = off
    if fixed_trajectory is not None:
        methods["discovery_fixed16"] = [
            next(r for r in g if tuple(r["trajectory"]) == _trajectory(fixed_trajectory))
            for g in groups.values()
        ]
    methods["constant_oracle"] = [
        max((r for r in g if r["trajectory"][0] == r["trajectory"][1]), key=lambda r: r["J"])
        for g in groups.values()
    ]
    methods["dynamic_oracle"] = [max(g, key=lambda r: r["J"]) for g in groups.values()]
    global_path = _trajectory(diagnostic["hindsight_global16_trajectory"])
    methods["hindsight_global16"] = [
        next(r for r in g if tuple(r["trajectory"]) == global_path) for g in groups.values()
    ]
    summaries, pair_results, sequence_results, sensitivity = {}, [], [], {}
    for method, stored in methods.items():
        chosen = _selected(groups, stored)
        summary = selector_summary(rows, stored, fixed_trajectory, seed, draws, tol)
        summary["secondary"] = {field: _macro(chosen, field) for field in ("F", "JF", "token_iou")}
        summary["method_scope"] = (
            "ANSWER_VISIBLE_DIAGNOSTIC"
            if method in ("constant_oracle", "dynamic_oracle", "hindsight_global16")
            else "STORED_DECISION_OR_FIXED_BASELINE"
        )
        if method == "hindsight_global16":
            summary["comparisons"] = {}
            summary["interval_note"] = (
                "Do not bootstrap the full-sample winner as fixed; "
                "sample-dependence intervals/LOSO are in global_diagnostic with reoptimization"
            )
        summaries[method] = summary
        sensitivity[method] = _size_sensitivity(chosen, off)
        off_by_key = {_key(r): r for r in off}
        for row in chosen:
            pair_results.append(
                {
                    "method": method,
                    "sequence": row["sequence"],
                    "target_slot": row["target_slot"],
                    "trajectory": row["trajectory"],
                    **{k: row.get(k) for k in ("J", "F", "JF", "token_iou")},
                    "delta_OFF": row["J"] - off_by_key[_key(row)]["J"],
                }
            )
        for seq in summary["per_sequence"]:
            subset = [r for r in chosen if r["sequence"] == seq["sequence"]]
            sequence_results.append(
                {"method": method, **seq, "F": _macro(subset, "F"), "JF": _macro(subset, "JF")}
            )
    for suffix in ("", "_outer_OOF_before_fallback"):
        cycle, gate = "CycleGate" + suffix, "GateOnly" + suffix
        if cycle in summaries and gate in summaries:
            cycle_rows = {_key(r): r for r in _selected(groups, methods[cycle])}
            gate_rows = {_key(r): r for r in _selected(groups, methods[gate])}
            seq = defaultdict(list)
            for key, row in cycle_rows.items():
                seq[key[0]].append(row["J"] - gate_rows[key]["J"])
            values = np.array([np.mean(seq[s]) for s in sorted(seq)])
            samples = np.random.default_rng(seed).integers(0, len(values), (draws, len(values)))
            summaries[cycle]["secondary_vs_GateOnly"] = {
                "mean": float(values.mean()),
                "ci95": np.quantile(values[samples].mean(axis=1), [0.025, 0.975]).tolist(),
                "scope": "Secondary descriptive paired sequence comparison, not primary H2",
            }
    return {
        "status": "COMPLETE",
        "global_diagnostic": diagnostic,
        "selector_analysis": summaries,
        "per_pair_results": pair_results,
        "per_sequence_results": sequence_results,
        "object_size_sensitivity": sensitivity,
    }


def _stored_decisions(rows: list[dict]) -> dict[str, list[dict]]:
    result = defaultdict(list)
    for row in rows:
        for method in row.get("selected_methods", []):
            result[str(method)].append(row)
    return dict(result)


def build_report(report_dir: Path, work_dir: Path, v1_report_dir: Path | None = None) -> dict:
    """Write deterministic JSON/CSV summaries; never fabricate confirmation rows."""
    report, work = Path(report_dir).resolve(), Path(work_dir).resolve()
    if report == work or not report.is_dir():
        raise ValueError("Use distinct existing V2 report and work directories")
    config = _read(report / "config.json")
    if "v2" not in str(config.get("schema", "")).lower():
        raise ValueError("Refusing to write summaries into a non-V2 report directory")
    root = Path(__file__).resolve().parents[3]
    v1 = v1_report_dir or root / "docs/experiments/EXP-2D-20260926-opportunity-selector-v1"
    old, old_audit = old_raw_diagnostics(v1)
    _write(report / "global_trajectory_diagnostics.json", old)
    _write(report / "old_raw_audit.json", old_audit)
    fixed = old["fixed_trajectory_provenance"]["trajectory"]
    raw_path = _find("raw_results.jsonl", report, work)
    raw = _rows(raw_path) if raw_path else []
    discovery_path = _find("discovery_rows.jsonl", report, work)
    if discovery_path:
        discovery_rows = _rows(discovery_path)
        old_discovery = {
            (_key(r), tuple(r["trajectory"])): r
            for r in _rows(v1 / "raw_results.jsonl")
            if r["split"] == "discovery" and r["resolution"] == 448
        }
        for row in discovery_rows:
            reference = old_discovery[(_key(row), tuple(row["trajectory"]))]
            if abs(row["J"] - reference["J"]) > 1e-12:
                raise ValueError("Discovery reward differs from saved V1 raw")
            # Only reuse saved secondary/object scores; no new target evaluation.
            for field in ("F", "JF", "token_iou", "objects"):
                if field not in row and field in reference:
                    row[field] = reference[field]
        if any(r.get("split") == "discovery" for r in raw):
            raise ValueError("Duplicate discovery sources; raw must contain independent rows only")
        raw.extend(discovery_rows)
    groups = defaultdict(list)
    for row in raw:
        groups[(str(row.get("split", "UNKNOWN")), str(row.get("resolution", 448)))].append(row)
    nested_path = _find("nested_cv_results.json", report, work)
    nested = _read(nested_path) if nested_path else {}
    results, all_pairs, all_sequences = {}, [], []
    for (split, resolution), subset in sorted(groups.items()):
        decisions = _stored_decisions(subset)
        saved_decisions_path = _find(
            "discovery_decisions.jsonl" if split == "discovery" else "predictions_manifest.json",
            report,
            work,
        )
        if saved_decisions_path:
            saved_pairs = (
                _rows(saved_decisions_path)
                if split == "discovery"
                else _read(saved_decisions_path).get("predictions", [])
            )
            saved = defaultdict(list)
            for pair in saved_pairs:
                paths = pair["trajectories"]
                if len(paths) != 16 or {tuple(t) for t in paths} != set(TRAJECTORIES):
                    raise ValueError("Saved prediction trajectories are incomplete")
                for name, index in pair["choices"].items():
                    if type(index) is not int or not 0 <= index < 16:
                        raise ValueError("Saved decision index is invalid")
                    saved[name].append(
                        {
                            "sequence": pair["sequence"],
                            "target_slot": pair["target_slot"],
                            "trajectory": paths[index],
                        }
                    )
            for name, values in decisions.items():
                expected = {_key(r): tuple(r["trajectory"]) for r in values}
                actual = {_key(r): tuple(r["trajectory"]) for r in saved.get(name, [])}
                if expected != actual:
                    raise ValueError("Raw selected_methods disagrees with saved predictions")
            decisions = dict(saved)
        elif split != "discovery":
            raise ValueError("Independent analysis requires the saved predictions manifest")
        if split == "discovery":
            for name, result in nested.items():
                if isinstance(result, dict) and result.get("outer_oof_decisions"):
                    decisions[f"{name}_outer_OOF_before_fallback"] = result["outer_oof_decisions"]
        result = analyze_saved_rows(
            subset,
            decisions,
            fixed,
            seed=config.get("bootstrap_seed", 20260926),
            draws=config.get("bootstrap_samples", 10000),
            tol=config.get("tie_tolerance", 1e-12),
        )
        result["scope"] = (
            "DISCOVERY_EXPLORATORY" if split == "discovery" else "CONFIRMATION_PENDING_AUDIT"
        )
        results[f"{resolution}/{split}"] = result
        for label, target in (
            ("per_pair_results", all_pairs),
            ("per_sequence_results", all_sequences),
        ):
            target.extend(
                {"split": split, "resolution": resolution, **r} for r in result.get(label, [])
            )
    audit_path = _find("independent_data_audit.json", report, work)
    independent = _read(audit_path) if audit_path else {}
    if not independent:
        prediction_path = _find("predictions_manifest.json", report, work)
        if prediction_path:
            placeholder = _read(prediction_path)
            if "BLOCKED" in str(placeholder.get("status", "")):
                independent = {
                    "status": placeholder["status"],
                    "confirmation_allowed": False,
                    "status_source": "blocked predictions placeholder; data audit pending",
                }
    has_test = any(
        split not in ("discovery", "fit", "validation", "historical_validation", "UNKNOWN")
        for split, _ in groups
    )
    data_status = independent.get("status", independent.get("independence_status", "UNVERIFIED"))
    verified = (
        independent.get("independence_status") == "VERIFIED"
        or independent.get("confirmation_allowed") is True
        or data_status == "VERIFIED"
    )
    if "BLOCKED" in str(data_status):
        confirmation_status = "BLOCKED_NO_INDEPENDENT_DATA"
    elif not verified:
        confirmation_status = "BLOCKED_UNVERIFIED_INDEPENDENCE"
    elif not has_test:
        confirmation_status = "PENDING_INDEPENDENT_EVALUATION"
    else:
        confirmation_status = "COMPLETED_PENDING_INTEGRITY"
    blocked = not has_test or not verified or "BLOCKED" in confirmation_status
    combined_global = {
        "old_raw": old,
        "v2": {k: r["global_diagnostic"] for k, r in results.items()},
        "confirmation_status": confirmation_status,
    }
    _write(report / "global_trajectory_diagnostic.json", combined_global)
    _write(
        report / "selector_analysis.json",
        {k: r.get("selector_analysis", {}) for k, r in results.items()},
    )
    _write(
        report / "bootstrap_results.json",
        {
            "seed": config.get("bootstrap_seed", 20260926),
            "draws": config.get("bootstrap_samples", 10000),
            "primary_unit": "sequence",
            "hindsight_reoptimized_per_resample": True,
            "primary_H2": ["CycleGate-OFF", "CycleGate-discovery_fixed16"],
            "scope": (
                "95% descriptive;97.5% Bonferroni only for two preregistered CycleGate comparisons"
            ),
            "comparisons": {
                k: {m: s["comparisons"] for m, s in r.get("selector_analysis", {}).items()}
                for k, r in results.items()
            },
        },
    )
    _write(
        report / "robustness_analysis.json",
        {k: r.get("object_size_sensitivity", {}) for k, r in results.items()},
    )
    _csv(report / "per_pair_results.csv", all_pairs)
    _csv(report / "per_sequence_results.csv", all_sequences)
    # Summarize recorded costs only; no low-write-rate/low-compute inference.
    cost_sources = {
        name: path
        for name in ("cost_analysis.json", "discovery_cost.json")
        if (path := _find(name, report, work)) is not None
    }
    costs = {name: _read(path) for name, path in cost_sources.items()}
    cost_totals = {}
    for filename, payload in costs.items():
        recorded = payload if isinstance(payload, list) else payload.get("pairs", [])
        if not isinstance(recorded, list):
            continue
        numeric = defaultdict(list)
        for row in recorded:
            for name, value in row.items():
                if (
                    name != "target_slot"
                    and isinstance(value, (int, float))
                    and not isinstance(value, bool)
                ):
                    numeric[name].append(value)
        cost_totals[filename] = {
            name: (max(values) if "peak" in name else sum(values))
            for name, values in numeric.items()
        }
    cost_summary = {
        "component_totals": cost_totals,
        "timing_note": (
            "Components may overlap: cycle_seconds includes correspondence_seconds; "
            "do not add overlapping timings into an invented wall total"
        ),
        "status": "RECORDED" if costs else "NO_COST_RECORDS_FOUND",
        "records": costs,
        "sources": {
            name: {"path": str(path), "sha256": _sha(path)} for name, path in cost_sources.items()
        },
        "interpretation": (
            "Candidates cost compute even when OFF is selected; no matched-budget claim"
        ),
    }
    _write(report / "cost_read_summary.json", cost_summary)
    summary = {
        "confirmation_status": confirmation_status,
        "data_independence_status": data_status,
        "raw_status": "PRESENT" if raw else "NO_V2_RAW_RESULTS",
        "raw_rows": len(raw),
        "raw_sha256": _sha(raw_path) if raw_path else None,
        "discovery_raw_sha256": _sha(discovery_path) if discovery_path else None,
        "old_raw_audit_status": old_audit["status"],
        "analysis_groups": list(results),
        "independent_test_numbers": None
        if blocked
        else {
            k: {m: s["mean_J"] for m, s in r.get("selector_analysis", {}).items()}
            for k, r in results.items()
            if not k.endswith("/discovery")
        },
        "interpretation": (
            "Discovery OOF is pre-fallback; fitted discovery deployment "
            "is not independent performance"
        ),
        "no_media_reads": True,
        "no_training_or_inference": True,
    }
    _write(report / "report_numbers.json", summary)
    return summary
