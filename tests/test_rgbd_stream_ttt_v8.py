"""Core B V8: the policy table is Core B V7's, the sealed two-cohort run and the branch rule."""

import copy
import importlib.util
import json
from pathlib import Path

import pytest
import torch
from test_rgbd_stream_write import stream_fixture
from test_rgbd_stream_write_v3 import carrier

from mcss.mechanism_pilot import rgbd_stream_ttt_v7 as v7
from mcss.mechanism_pilot import rgbd_stream_ttt_v8 as v8


def script(name):
    path = Path(__file__).parents[1] / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_every_policy_row_is_core_b_v7s():
    assert set(v8.POLICY_SPECS) <= set(v7.POLICY_SPECS)
    assert all(v8.POLICY_SPECS[k] == v7.POLICY_SPECS[k] for k in v8.POLICY_SPECS)
    assert v8.CONTROLS == {"TTT_100_WRONG_SCENE": "TTT_100"}
    assert v8.LEARNING_RATE == v7.LEARNING_RATE == 1e-3


def test_two_cohorts_are_sealed_before_query_depth(tmp_path):
    torch.set_num_threads(1)
    manifest = stream_fixture(tmp_path)
    dev = [r for r in manifest["scenes"] if r["split"] == "DEV"]
    cohorts = {}
    for name, record in zip(("EVAL_V3", "EVAL_V4"), dev, strict=True):
        cohort = copy.deepcopy(manifest)
        cohort["scenes"] = [{**copy.deepcopy(record), "split": name}]
        cohorts[name] = (cohort, tmp_path)
    expected = {name: name for name in cohorts}
    specs = {name: (s, sup, min(steps, 2)) for name, (s, sup, steps) in v8.POLICY_SPECS.items()}
    out = tmp_path / "evaluation"
    rows = v8.evaluate_v8(carrier(), cohorts, expected, out, 7, "cpu", specs)
    assert len(rows) == 2 * 2 * 2 * (len(specs) + len(v8.CONTROLS))
    assert {r["cohort"] for r in rows} == set(cohorts)
    access = json.loads((out / "GT_access.json").read_text())
    marker = next(i for i, e in enumerate(access) if e.get("event") == "ALL_STREAM_STATES_SEALED")
    early = [e for e in access[:marker] if "depth" in e.get("channels", [])]
    assert {e["purpose"] for e in early} == {"CONTEXT_ONLY_RGBD", "STREAM_RGBD"}
    assert all(e["frame_id"] not in (8, 9) for e in early)
    meta = json.loads((out / "state_hashes.json").read_text())
    donors = {m["scene_id"]: m["donor_scene_id"] for m in meta.values() if "donor_scene_id" in m}
    assert donors == {"capacity0": "capacity1", "capacity1": "capacity0"}
    ctx = [m for m in meta.values() if m["policy"] == "OFF_CTX_TTT_100"]
    assert all(
        m["supervised_frame_ids"] == m["warmup_frame_ids"] and m["stream_frame_ids"] for m in ctx
    )


def rows_for(values):
    rows = []
    for seed in (1, 2):
        for i in range(8):
            for role in ("A", "B"):
                for qid in (8, 9):
                    for policy, value in values.items():
                        rows.append(
                            {
                                "seed": seed,
                                "scene_id": f"s{i}",
                                "role": role,
                                "query_id": qid,
                                "policy": policy,
                                "depth_absrel": value + 0.002 * ((i + qid + seed) % 3),
                            }
                        )
    return rows


def test_branches_follow_the_two_primary_gates():
    analyze = script("run_rgbd_stream_ttt_source_v8.py").analyze
    base = {
        "NO_STREAM": 0.30,
        "OFF": 0.24,
        "OFF_CTX_TTT_100": 0.23,
        "TTT": 0.225,
        "TTT_100": 0.21,
        "TTT_100_WRONG_SCENE": 0.25,
    }
    assert analyze(rows_for(base), [1, 2])["INTERPRETATION_BRANCH"] == "A"
    same_source = {**base, "OFF_CTX_TTT_100": 0.21}
    assert analyze(rows_for(same_source), [1, 2])["INTERPRETATION_BRANCH"] == "B"
    no_gain = {**base, "TTT_100": 0.24, "OFF_CTX_TTT_100": 0.25}
    assert analyze(rows_for(no_gain), [1, 2])["INTERPRETATION_BRANCH"] == "C"
    flat = {**base, "TTT_100": 0.24, "OFF_CTX_TTT_100": 0.24}
    result = analyze(rows_for(flat), [1, 2])
    assert result["INTERPRETATION_BRANCH"] == "D"
    assert result["contrasts"]["STREAM_DOSE"]["mean"] == pytest.approx(-0.015)
