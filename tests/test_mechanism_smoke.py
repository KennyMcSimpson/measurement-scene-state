"""End-to-end synthetic integration checks, never carrier qualification tests."""

import importlib.util
import json
from pathlib import Path

import pytest

from mcss.mechanism_pilot.statistics import analyze_history, scene_macro

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_3d_mechanism_smoke.py"


def script_module():
    spec = importlib.util.spec_from_file_location("mechanism_smoke_script", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read(directory, name):
    return json.loads((directory / name).read_text())


@pytest.fixture(scope="module")
def smoke_runs(tmp_path_factory):
    module = script_module()
    first = tmp_path_factory.mktemp("mechanism-first")
    second = tmp_path_factory.mktemp("mechanism-replay")
    module.run(first)
    module.run(second)
    return module, first, second


def test_all_candidate_seals_precede_any_query_access(smoke_runs):
    _, first, _ = smoke_runs
    events = read(first, "access_log.json")["events"]
    seal_positions = [i for i, row in enumerate(events) if row["event"] == "seal"]
    reads = [(i, row) for i, row in enumerate(events) if row["event"] != "seal"]
    assert len(seal_positions) == 33
    assert len(reads) == 12
    assert max(seal_positions) < min(i for i, _ in reads)
    assert all(row["status"] == "completed" for _, row in reads)
    manifest = read(first, "state_manifest.json")
    assert {row["candidate_id"] for row in events if row["event"] == "seal"} == set(manifest)
    assert all(not set(row["observed_ids"]) & {8, 9} for row in manifest.values())


def test_report_counts_and_statistics_recompute_from_saved_rows(smoke_runs):
    _, first, _ = smoke_runs
    summary = read(first, "summary.json")
    dynamic = read(first, "controlled_history_results.json")
    static = read(first, "static_state_results.json")
    assert summary["scientific_scenes"] == 0
    assert summary["dynamic_query_rows"] == len(dynamic) == 48
    assert summary["static_query_rows"] == len(static) == 30
    assert summary["controlled_pairs"] == len({r["scene_id"] for r in dynamic}) == 3
    assert read(first, "history_analysis.json") == analyze_history(
        dynamic, draws=10000, seed=20260927, tie_tolerance=1e-8
    )
    static_analysis = read(first, "static_analysis.json")
    for method in ("A", "B", "anchor", "prior", "wrong_scene"):
        for metric in ("depth_absrel", "rgb_psnr", "depth_rmse", "coverage"):
            assert static_analysis[method][metric] == scene_macro(
                [r for r in static if r["method"] == method], metric
            )
    for scene in range(3):
        for query in (8, 9):
            rows = [
                r
                for r in static
                if r["scene_id"] == f"synthetic-{scene}" and r["query_id"] == query
            ]
            assert len({row["query_camera_hash"] for row in rows}) == 1


def test_seeded_replay_matches_numeric_artifacts_not_wall_clock(smoke_runs):
    _, first, second = smoke_runs
    for name in (
        "run_lock.json",
        "state_manifest.json",
        "prediction_hashes.json",
        "controlled_history_results.json",
        "static_state_results.json",
        "history_analysis.json",
        "numerical_history_effect.json",
        "static_analysis.json",
    ):
        assert read(first, name) == read(second, name), name


def test_existing_run_is_never_overwritten(smoke_runs):
    module, first, _ = smoke_runs
    previous = (first / "run_lock.json").read_bytes()
    with pytest.raises(FileExistsError, match="overwrite"):
        module.run(first)
    assert (first / "run_lock.json").read_bytes() == previous
