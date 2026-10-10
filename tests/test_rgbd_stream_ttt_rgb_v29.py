"""Core B V29: RGB-only stream reads, the sealed run with them, the audit and the branch rule."""

import copy
import importlib.util
import json
from pathlib import Path

import pytest
import torch
from test_rgbd_stream_write import stream_fixture
from test_rgbd_stream_write_v3 import carrier

from mcss.mechanism_pilot import rgbd_stream_ttt_rgb_v29 as v29
from mcss.mechanism_pilot import rgbd_stream_ttt_v7 as v7
from mcss.mechanism_pilot import rgbd_stream_ttt_v8 as v8


def script(name):
    path = Path(__file__).parents[1] / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_policy_rows_follow_core_b_v7s_format():
    assert v29.POLICY_SPECS["NO_STREAM"] == v7.POLICY_SPECS["NO_STREAM"]
    assert v29.POLICY_SPECS["CTX_TTT_100"] == v7.POLICY_SPECS["CTX_TTT_100"]
    assert v29.POLICY_SPECS["OFF_RGB"] == v7.POLICY_SPECS["OFF"]
    assert v29.POLICY_SPECS["TTT_RGB_100"] == v7.POLICY_SPECS["TTT_100"]
    assert v29.CONTROLS == {"TTT_RGB_100_WRONG_SCENE": "TTT_RGB_100"}


def test_rgb_only_streams_never_open_depth_and_carry_invalid_depth(tmp_path):
    manifest = stream_fixture(tmp_path)
    record = next(r for r in manifest["scenes"] if r["split"] == "DEV")
    access = []
    ids = tuple(r["scene_id"] for r in manifest["scenes"])
    loader = v29.RGBOnlyStreamLoader(
        record,
        manifest["image_size"],
        tmp_path,
        "cpu",
        access,
        allowed_scene_ids=ids,
        holdout_scene_ids=(),
    )
    stream = loader.stream("A")
    assert len(stream) == 4 and torch.equal(stream.depths, torch.zeros_like(stream.depths))
    assert {e["purpose"] for e in access} == {v29.PURPOSE}
    assert all("depth" not in e["channels"] for e in access)


def test_sealed_run_reads_streams_as_rgb_only_and_restores_the_loader(tmp_path):
    torch.set_num_threads(1)
    manifest = stream_fixture(tmp_path)
    dev = [r for r in manifest["scenes"] if r["split"] == "DEV"]
    cohorts = {}
    for name, record in zip(("EVAL_V3", "EVAL_V4"), dev, strict=True):
        cohort = copy.deepcopy(manifest)
        cohort["scenes"] = [{**copy.deepcopy(record), "split": name}]
        cohorts[name] = (cohort, tmp_path)
    expected = {name: name for name in cohorts}
    specs = {name: (s, sup, min(steps, 2)) for name, (s, sup, steps) in v29.POLICY_SPECS.items()}
    original = v8.StreamRGBDLoader
    out = tmp_path / "evaluation"
    rows = v29.evaluate_v29(carrier(), cohorts, expected, out, 7, "cpu", specs)
    assert v8.StreamRGBDLoader is original
    assert len(rows) == 2 * 2 * 2 * (len(specs) + len(v29.CONTROLS))
    access = json.loads((out / "GT_access.json").read_text())
    marker = next(i for i, e in enumerate(access) if e.get("event") == "ALL_STREAM_STATES_SEALED")
    run = script("run_rgbd_stream_ttt_rgb_v29.py")
    assert run.rgb_only_audit(access, marker) > 0
    with pytest.raises(AssertionError):
        run.rgb_only_audit([{"purpose": "STREAM_RGBD", "channels": ["RGB", "depth"]}], 1)


def test_branch_follows_the_two_primary_gates():
    run = script("run_rgbd_stream_ttt_rgb_v29.py")
    assert run.branch("SUPPORTED", "SUPPORTED") == "A"
    assert run.branch("SUPPORTED", "PARTIAL") == "B"
    assert run.branch("NOT_ESTABLISHED", "SUPPORTED") == "C"
    assert run.branch("PARTIAL", "NOT_ESTABLISHED") == "D"
