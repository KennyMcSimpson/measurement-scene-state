"""Synthetic-only byte-exact reconstruction audit wrapper tests."""

import importlib.util
import json
from pathlib import Path

import pytest


def audit_module():
    source = Path(__file__).resolve().parents[1] / "scripts/audit_optimization_bounds_statistics.py"
    spec = importlib.util.spec_from_file_location("bounds_statistics_audit", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fixture(root, names):
    for phase in (root, root / "secondary_analysis"):
        phase.mkdir(parents=True, exist_ok=True)
        for name in names:
            (phase / name).write_text(json.dumps({"x": [0.25, 0.5], "flag": True}, sort_keys=True))
    (root / "secondary_analysis/secondary_decision_audit.json").write_text('{"status":"PASS"}')


@pytest.mark.parametrize("reference_enabled", [False, True])
def test_two_reproduction_runs_compare_original_and_preserve_reference_routing(
    tmp_path, reference_enabled
):
    module = audit_module()
    root = tmp_path / "original"
    fixture(root, module.SCIENTIFIC_FILES)
    reference = tmp_path / "published" if reference_enabled else None
    if reference is not None:
        reference.mkdir()
    calls = []

    def runner(source, output, include_secondary=False):
        calls.append((source, output, include_secondary))
        output.mkdir()
        originals = root / "secondary_analysis" if include_secondary else root
        names = (
            (*module.SCIENTIFIC_FILES, "secondary_decision_audit.json")
            if include_secondary
            else module.SCIENTIFIC_FILES
        )
        for name in names:
            (output / name).write_bytes((originals / name).read_bytes())
        (output / "statistics_reproduction.json").write_text('{"raw_only":true}')

    report = module.run_audit(root, reference, runner=runner)
    assert report["status"] == "PASS"
    assert report["compared_files"] == 13 and report["max_abs_numeric_error"] == 0
    assert [call[2] for call in calls] == [False, True]
    assert calls[0][1] != calls[1][1]
    assert all(call[0] == (reference or root).resolve() for call in calls)
    filename = (
        "published_statistics_reproduction_audit.json"
        if reference_enabled
        else "statistics_reproduction_audit.json"
    )
    assert json.loads((root / "audit" / filename).read_text())["status"] == "PASS"


@pytest.mark.parametrize("mutation,expected_error", [("numeric", 0.5), ("whitespace", 0)])
def test_numerical_and_byte_only_mismatch_both_fail(tmp_path, mutation, expected_error):
    module = audit_module()
    fixture(tmp_path, module.SCIENTIFIC_FILES)

    def runner(source, output, include_secondary=False):
        output.mkdir()
        originals = source / "secondary_analysis" if include_secondary else source
        names = (
            (*module.SCIENTIFIC_FILES, "secondary_decision_audit.json")
            if include_secondary
            else module.SCIENTIFIC_FILES
        )
        for name in names:
            payload = (originals / name).read_text()
            if name == "cost_analysis.json" and not include_secondary:
                payload = payload.replace("0.5", "1.0") if mutation == "numeric" else payload + "\n"
            (output / name).write_text(payload)
        (output / "statistics_reproduction.json").write_text('{"raw_only":true}')

    report = module.run_audit(tmp_path, runner=runner)
    assert report["status"] == "FAIL"
    assert report["byte_mismatch_count"] == 1
    assert report["max_abs_numeric_error"] == expected_error


def test_incomplete_secondary_does_not_invoke_analysis(tmp_path):
    module = audit_module()

    def forbidden(*args, **kwargs):
        raise AssertionError("Analyzer cannot run before both published reports exist")

    report = module.run_audit(tmp_path, runner=forbidden)
    assert report["status"] == "FAIL" and report["error"]["type"] == "FileNotFoundError"
