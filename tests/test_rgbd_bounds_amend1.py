"""V7 Amendment 1: the amended estimator differs from the frozen one by one assertion only."""

import difflib
import importlib.util
import inspect
import json
from pathlib import Path

import pytest
from test_geometry_carrier_statistics import fixture, region_fixture

from mcss.mechanism_pilot import geometry_carrier_statistics as frozen


def script(name):
    path = Path(__file__).parents[1] / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def inputs(query=None):
    data = fixture()
    kwargs = {
        "seeds": data[4]["seeds"],
        "state_use_audit": data[4]["state_use_audit"],
        "static_matched_rows": data[4]["static_matched_rows"],
    }
    return (query or data[0], data[1], data[2], data[3]), kwargs


def test_only_the_two_assertion_lines_are_removed():
    module = script("analyze_rgbd_bounds_carrier_amend1.py")
    old = inspect.getsource(frozen.analyze)
    new = old.replace(module.CHECK, "")
    removed = [
        line
        for line in difflib.ndiff(old.splitlines(), new.splitlines())
        if line.startswith(("- ", "+ "))
    ]
    assert len(removed) == 2 and all(line.startswith("- ") for line in removed)
    assert "ray-hit fractions" in removed[1]


def test_equal_hit_fractions_give_identical_results():
    args, kwargs = inputs()
    amended = script("analyze_rgbd_bounds_carrier_amend1.py").amended_analyze()
    expected = json.dumps(frozen.analyze(*args, **kwargs), sort_keys=True)
    assert json.dumps(amended(*args, **kwargs), sort_keys=True) == expected


def test_unequal_hit_fractions_stop_only_the_frozen_estimator():
    args, _ = inputs()
    query = [dict(r) for r in args[0]]
    for row in query:
        if row["variant"] == "C1" and row["method"] == "direct":
            row["ray_hitfraction"] = row["ray_hitfraction"] * 0.5
    args, kwargs = inputs(query)
    with pytest.raises(ValueError, match="ray-hit fractions"):
        frozen.analyze(*args, **kwargs)
    amended = script("analyze_rgbd_bounds_carrier_amend1.py").amended_analyze()
    assert "static_results" in amended(*args, **kwargs)


def test_amended_analysis_and_audit_reproduce_byte_exactly(tmp_path):
    data = fixture()
    values = {
        "scene_split": data[3],
        "training_contract": {"seeds": data[4]["seeds"], "steps": 6000},
        "raw/dev_query_results": data[0],
        "raw/dev_context_results": data[1],
        "raw/dev_matched_results": data[2],
        "raw/dev_region_results": region_fixture(data),
        "raw/dev_static_matched_results": data[4]["static_matched_rows"],
        "audit/dev_state_use": data[4]["state_use_audit"],
        "training_curves": {"training": [], "dev": []},
        "cost_analysis": {"variants": {v: {"training_seconds": 1.0} for v in ("C0", "C1")}},
    }
    for name, value in values.items():
        path = tmp_path / (name + ".json")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
    script("analyze_rgbd_bounds_carrier_amend1.py").run(tmp_path)
    fractions = json.loads((tmp_path / "ray_hit_fractions.json").read_text())
    assert fractions["descriptive_only"] and set(fractions["per_variant_scene"]) >= {"C0", "C1"}
    provenance = json.loads((tmp_path / "statistics_reproduction.json").read_text())
    assert "Amendment 1" in provenance["amendment"]
    assert any(
        p.endswith("analyze_rgbd_bounds_carrier_amend1.py") for p in provenance["source_sha256"]
    )
    script("audit_rgbd_bounds_statistics_amend1.py").audit(tmp_path)
    audit = json.loads((tmp_path / "audit/statistics_reproduction_audit.json").read_text())
    assert audit["status"] == "PASS" and audit["all_scientific_json_byte_exact"]


def test_resume_runner_executes_only_the_amended_phases():
    module = script("run_rgbd_bounds_experiment_amend1.py")
    assert [s for _, s in module.PHASES] == [
        "scripts/analyze_rgbd_bounds_carrier_amend1.py",
        "scripts/audit_rgbd_bounds_statistics_amend1.py",
    ]
    source = Path(module.__file__).read_text()
    assert (
        "train_rgbd_bounds_carrier" not in source and "evaluate_rgbd_bounds_carrier" not in source
    )
