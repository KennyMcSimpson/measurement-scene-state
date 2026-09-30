#!/usr/bin/env python3
"""Byte-exact JSON-only reproduction audit for observability/supervision attribution."""

import argparse
import hashlib
import importlib.util
import json
import math
import tempfile
from pathlib import Path

SCIENTIFIC_FILES = (
    "observability_analysis.json",
    "supervision_analysis.json",
    "gap_closure_analysis.json",
    "wrong_scene_analysis.json",
    "bootstrap_results.json",
    "cost_analysis.json",
    "decision_summary.json",
)


def load_runner():
    path = Path(__file__).resolve().with_name("analyze_observability_supervision.py")
    spec = importlib.util.spec_from_file_location("frozen_observability_analyzer", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.run


def numeric_comparison(original, reproduced):
    """Report numerical error and structural mismatches; byte equality remains decisive."""
    maximum = 0.0
    numeric_count = 0
    mismatch_count = 0
    examples = []

    def mismatch(path, kind):
        nonlocal mismatch_count
        mismatch_count += 1
        if len(examples) < 20:
            examples.append({"path": path, "kind": kind})

    def visit(left, right, path):
        nonlocal maximum, numeric_count
        if isinstance(left, bool) or isinstance(right, bool):
            if type(left) is not type(right) or left != right:
                mismatch(path, "boolean_or_type")
        elif isinstance(left, (int, float)) and isinstance(right, (int, float)):
            if not math.isfinite(left) or not math.isfinite(right):
                mismatch(path, "nonfinite_number")
                return
            numeric_count += 1
            maximum = max(maximum, abs(left - right))
            if type(left) is not type(right):
                mismatch(path, "numeric_type")
        elif isinstance(left, dict) and isinstance(right, dict):
            if set(left) != set(right):
                mismatch(path, "dictionary_keys")
            for key in sorted(set(left) & set(right)):
                visit(left[key], right[key], f"{path}.{key}")
        elif isinstance(left, list) and isinstance(right, list):
            if len(left) != len(right):
                mismatch(path, "list_length")
            for i, (a, b) in enumerate(zip(left, right, strict=False)):
                visit(a, b, f"{path}[{i}]")
        elif type(left) is not type(right) or left != right:
            mismatch(path, "value_or_type")

    visit(original, reproduced, "$")
    return {
        "max_abs_numeric_error": maximum,
        "numeric_values_compared": numeric_count,
        "structural_or_nonnumeric_mismatches": mismatch_count,
        "mismatch_examples": examples,
    }


def compare_file(original, reproduced):
    original_bytes, reproduced_bytes = original.read_bytes(), reproduced.read_bytes()
    return {
        "original": str(original.resolve()),
        "reproduced_filename": reproduced.name,
        "byte_exact": original_bytes == reproduced_bytes,
        "original_sha256": hashlib.sha256(original_bytes).hexdigest(),
        "reproduced_sha256": hashlib.sha256(reproduced_bytes).hexdigest(),
        **numeric_comparison(json.loads(original_bytes), json.loads(reproduced_bytes)),
    }


def run_audit(root, reference=None, *, runner=None):
    root = Path(root).resolve()
    source = Path(reference).resolve() if reference is not None else root
    output = (
        root
        / "audit"
        / (
            "published_statistics_reproduction_audit.json"
            if reference is not None
            else "statistics_reproduction_audit.json"
        )
    )
    report = {
        "status": "FAIL",
        "comparison_original_root": str(root),
        "analysis_input_root": str(source),
        "published_reference": reference is not None,
        "raw_only": True,
        "media_or_state_checkpoint_loaded": False,
        "access_scope": "Frozen analyzer loads saved JSON/gzip only, never image/depth/state files",
        "comparison_rule": "All seven scientific JSON artifacts must be byte-exact",
        "files": {},
    }
    try:
        for name in SCIENTIFIC_FILES:
            if not (root / name).is_file():
                raise FileNotFoundError(f"Formal analysis must finish before audit: {root / name}")
        runner = load_runner() if runner is None else runner
        with tempfile.TemporaryDirectory(
            prefix="mcss-observability-statistics-reproduction-"
        ) as temp:
            reproduced = Path(temp) / "analysis"
            runner(source, reproduced)
            for name in SCIENTIFIC_FILES:
                report["files"][name] = compare_file(root / name, reproduced / name)
            report["input_provenance"] = json.loads(
                (reproduced / "statistics_reproduction.json").read_text()
            )
        values = list(report["files"].values())
        report["compared_files"] = len(values)
        report["max_abs_numeric_error"] = max(row["max_abs_numeric_error"] for row in values)
        report["byte_mismatch_count"] = sum(not row["byte_exact"] for row in values)
        report["structural_or_nonnumeric_mismatches"] = sum(
            row["structural_or_nonnumeric_mismatches"] for row in values
        )
        if (
            report["byte_mismatch_count"] == 0
            and report["max_abs_numeric_error"] == 0
            and report["structural_or_nonnumeric_mismatches"] == 0
        ):
            report["status"] = "PASS"
    except Exception as error:
        report["error"] = {"type": type(error).__name__, "message": str(error)}
    report["audit_source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--reference", type=Path, help="Optional published JSON/gzip raw archive")
    args = parser.parse_args()
    report = run_audit(args.root, args.reference)
    print(
        json.dumps(
            {
                key: report.get(key)
                for key in (
                    "status",
                    "compared_files",
                    "byte_mismatch_count",
                    "max_abs_numeric_error",
                    "error",
                )
            },
            indent=2,
        )
    )
    raise SystemExit(0 if report["status"] == "PASS" else 1)
