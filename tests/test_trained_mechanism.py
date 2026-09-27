"""Tiny fixture verifies checkpoint-driven diagnostic boundaries, not research quality."""

import importlib.util
import json
from pathlib import Path

import pytest
import torch

from mcss.dynamic.carrier import DynamicSceneCarrier
from mcss.dynamic.checkpoint import save_dynamic_checkpoint
from mcss.dynamic.config import WriteConfig
from mcss.dynamic.write_rule import DirectWriteRule
from mcss.mechanism_pilot.spatial import carrier_config_for_spatial_mode
from mcss.mechanism_pilot.statistics import analyze_history


def load_module(relative, name):
    spec = importlib.util.spec_from_file_location(
        name, Path(__file__).resolve().parents[1] / relative
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def run_fixture(tmp_path_factory):
    root = tmp_path_factory.mktemp("trained-mechanism")
    fixture = load_module("tests/test_small_3d_training.py", "small_training_fixture")
    manifest = fixture.manifest_fixture(root)
    # Shift every pose: missing query/continuation anchoring then becomes observable.
    for scene in manifest["scenes"]:
        for frame in scene["frames"]:
            frame["c2w"][0][3] = 11.0
            frame["c2w"][2][3] = -7.0
    path = root / "manifest.json"
    path.write_text(json.dumps(manifest))
    torch.manual_seed(20260927)
    carrier = DynamicSceneCarrier(carrier_config_for_spatial_mode("anchor-centered"))
    writer = DirectWriteRule(carrier.config, WriteConfig())
    checkpoint = root / "fixture.pt"
    save_dynamic_checkpoint(
        checkpoint, carrier, writer, phase="fixture", provenance={"scope": "fixture"}
    )
    runner = load_module("scripts/run_trained_3d_mechanism.py", "trained_mechanism")
    output = root / "run"
    runner.run(path, checkpoint, output, device="cpu", engineering_fixture=True)
    return runner, output, path, checkpoint


def read(output, name):
    return json.loads((output / name).read_text())


def test_seals_precede_queries_and_no_query_in_runtime(run_fixture):
    runner, output, path, _ = run_fixture
    events = read(output, "sealed_query_access_log.json")
    assert all(e["event"] == "seal" for e in events[:33])
    assert all(e["event"] != "seal" and e["status"] == "completed" for e in events[33:])
    manifest = read(output, "state_manifest.json")
    assert len(manifest) == 33
    assert all(set(r["observed_ids"]).issubset(set(range(8))) for r in manifest.values())
    access = [json.loads(s) for s in (output / "label_access.jsonl").read_text().splitlines()]
    assert all(r["kind"] == "rgb_camera" for r in access if r["stage"] == "state_construction")
    data = runner.PilotData(path, "cpu", [])
    for i in (8, 9, 12, 15):
        with pytest.raises(PermissionError):
            data.observation("fixture0", i)


def test_fixed_cohorts_recompute_and_query_camera_not_changed(run_fixture):
    runner, output, _, _ = run_fixture
    static = read(output, "static_state_results.json")
    dynamic = read(output, "controlled_history_results.json")
    assert len(static) == 90 and len(dynamic) == 144
    for cohort in runner.COHORTS:
        subset = [r for r in dynamic if r["cohort"] == cohort]
        assert analyze_history(subset) == read(output, f"history_analysis_{cohort}.json")
        assert runner.summarize_static([r for r in static if r["cohort"] == cohort]) == read(
            output, f"static_analysis_{cohort}.json"
        )
    for scene in ("fixture0", "fixture1", "fixture2"):
        for query in (8, 9, 12, 13, 14, 15):
            assert (
                len(
                    {
                        r["query_camera_hash"]
                        for r in static + dynamic
                        if r["scene_id"] == scene and r["query_id"] == query
                    }
                )
                == 1
            )
    assert all(r["coverage"] == 1.0 for r in static)


def test_checkpoint_immutable_and_no_overwrite(run_fixture):
    runner, output, path, checkpoint = run_fixture
    summary = read(output, "summary.json")
    assert summary["checkpoint_unchanged"] and summary["source_data_unchanged"]
    assert summary["candidate_isolation"] and summary["controlled_pairs"] == 3
    with pytest.raises(FileExistsError):
        runner.run(path, checkpoint, output, device="cpu", engineering_fixture=True)
