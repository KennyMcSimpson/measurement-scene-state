#!/usr/bin/env python3
"""Independently rerun frozen JSON-only analysis and compare all scientific artifacts.

Reads saved JSON/CSV (including their gzip archives) through the frozen analyzer.
It never loads RGB/depth media, model weights, optimized states or checkpoints.
"""

import argparse
import hashlib
import importlib.util
import json
import math
import tempfile
from pathlib import Path

SCIENTIFIC_FILES = (
    "optimization_results.json",
    "optimization_sufficiency.json",
    "bounds_analysis.json",
    "query_oracle_diagnostic.json",
    "bootstrap_results.json",
    "cost_analysis.json",
)


def load_runner():
    path = Path(__file__).resolve().with_name("analyze_optimization_bounds.py")
    spec = importlib.util.spec_from_file_location("frozen_optimization_bounds_analyzer", path)
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
    report_path = (
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
        "media_or_checkpoint_loaded": False,
        "access_scope": "JSON/CSV/gzip only; no media or state deserialization",
        "comparison_rule": "Every required scientific JSON must match byte-for-byte",
        "files": {},
    }
    try:
        expected = [(root, name) for name in SCIENTIFIC_FILES]
        expected += [
            (root / "secondary_analysis", name)
            for name in (*SCIENTIFIC_FILES, "secondary_decision_audit.json")
        ]
        for directory, name in expected:
            if not (directory / name).is_file():
                raise FileNotFoundError(f"Both phases must finish before audit: {directory / name}")
        runner = load_runner() if runner is None else runner
        with tempfile.TemporaryDirectory(
            prefix="mcss-optimization-bounds-reproduction-"
        ) as temporary:
            primary_output, secondary_output = (
                Path(temporary) / "primary",
                Path(temporary) / "secondary",
            )
            runner(source, primary_output, include_secondary=False)
            runner(source, secondary_output, include_secondary=True)
            for phase, directory, reproduced_directory, filenames in (
                ("primary", root, primary_output, SCIENTIFIC_FILES),
                (
                    "secondary",
                    root / "secondary_analysis",
                    secondary_output,
                    (*SCIENTIFIC_FILES, "secondary_decision_audit.json"),
                ),
            ):
                for name in filenames:
                    report["files"][f"{phase}/{name}"] = compare_file(
                        directory / name, reproduced_directory / name
                    )
            for phase, directory in (("primary", primary_output), ("secondary", secondary_output)):
                metadata = directory / "statistics_reproduction.json"
                report[f"{phase}_input_provenance"] = json.loads(metadata.read_text())
        comparisons = list(report["files"].values())
        report["compared_files"] = len(comparisons)
        report["max_abs_numeric_error"] = max(r["max_abs_numeric_error"] for r in comparisons)
        report["byte_mismatch_count"] = sum(not r["byte_exact"] for r in comparisons)
        report["structural_or_nonnumeric_mismatches"] = sum(
            r["structural_or_nonnumeric_mismatches"] for r in comparisons
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
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, required=True, help="Original complete experiment outputs"
    )
    parser.add_argument(
        "--reference", type=Path, help="Optional published raw archive to reanalyze"
    )
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
