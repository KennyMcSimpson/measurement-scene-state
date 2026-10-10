"""Core B V7: the policy table reproduces Core B V6, dose arms, and the sealed fresh-cohort run."""

import copy
import importlib.util
import json
from pathlib import Path

import pytest
import torch
from test_rgbd_stream_write import stream_fixture
from test_rgbd_stream_write_v3 import carrier, contexts

from mcss.dynamic.types import hash_scene_state, hash_value
from mcss.mechanism_pilot import rgbd_stream_ttt_context_v6 as v6
from mcss.mechanism_pilot import rgbd_stream_ttt_v7 as v7

V6_SETTINGS = {"steps": 10, "learning_rate": 1e-3}


def test_the_five_base_policies_are_core_b_v6s():
    warm, stream, bounds, _ = contexts()
    model = carrier()
    before = hash_value(model.state_dict())
    for policy in v6.POLICIES:
        theirs = v6.unroll_v6(model, warm, stream, bounds, policy, "e", V6_SETTINGS, 7)
        mine = v7.unroll_v7(model, warm, stream, bounds, policy, "e", 7)
        assert hash_scene_state(mine[0]) == hash_scene_state(theirs[0]), policy
        assert mine[4] == pytest.approx(theirs[4])
        assert mine[3].source_frames == theirs[3].source_frames
    assert hash_value(model.state_dict()) == before


def test_dose_arms_follow_their_table_row():
    warm, stream, bounds, _ = contexts()
    model = carrier()
    specs = {**v7.POLICY_SPECS, "TTT_4": (True, "stream", 4), "CTX_TTT_3": (False, "context", 3)}
    four = v7.unroll_v7(model, warm, stream, bounds, "TTT_4", "e", 7, specs)
    three = v7.unroll_v7(model, warm, stream, bounds, "CTX_TTT_3", "e", 7, specs)
    assert four[2].step == 4 and len(four[4]) == 4
    assert three[2].step == 3 and three[3].source_frames == [o.frame_id for o in warm]
    with pytest.raises(ValueError):
        v7.unroll_v7(model, warm, stream, bounds, "TTT_4", "e", 7)


def test_every_policy_and_control_is_sealed_before_query_depth(tmp_path):
    torch.set_num_threads(1)
    manifest = stream_fixture(tmp_path)
    cohort = copy.deepcopy(manifest)
    cohort["scenes"] = [r for r in cohort["scenes"] if r["split"] == "DEV"]
    specs = {name: (s, sup, min(steps, 2)) for name, (s, sup, steps) in v7.POLICY_SPECS.items()}
    out = tmp_path / "evaluation"
    rows = v7.evaluate_v7(carrier(), {"X": (cohort, tmp_path)}, {"X": "DEV"}, out, 7, "cpu", specs)
    assert len(rows) == 2 * 2 * 2 * (len(specs) + len(v7.CONTROLS))
    access = json.loads((out / "GT_access.json").read_text())
    marker = next(i for i, e in enumerate(access) if e.get("event") == "ALL_STREAM_STATES_SEALED")
    early = [e for e in access[:marker] if "depth" in e.get("channels", [])]
    assert {e["purpose"] for e in early} == {"CONTEXT_ONLY_RGBD", "STREAM_RGBD"}
    assert all(e["frame_id"] not in (8, 9) for e in early)
    meta = json.loads((out / "state_hashes.json").read_text())
    for control in v7.CONTROLS:
        donors = {
            m["scene_id"]: m["donor_scene_id"] for m in meta.values() if m["policy"] == control
        }
        assert donors == {"capacity0": "capacity1", "capacity1": "capacity0"}
    ttt = [m for m in meta.values() if m["policy"] == "TTT"]
    assert all(
        m["supervised_frame_ids"] == m["stream_frame_ids"] and m["ttt_steps"] == 2 for m in ttt
    )
    ctx = [m for m in meta.values() if m["policy"] == "CTX_TTT"]
    assert all(
        m["stream_frame_ids"] == [] and m["supervised_frame_ids"] == m["warmup_frame_ids"]
        for m in ctx
    )


def script(name):
    path = Path(__file__).parents[1] / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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
    analyze = script("run_rgbd_stream_ttt_fresh_v7.py").analyze
    base = {
        "NO_STREAM": 0.30,
        "CTX_TTT": 0.29,
        "CTX_TTT_100": 0.28,
        "OFF": 0.24,
        "OFF_CTX_TTT": 0.235,
        "OFF_CTX_TTT_100": 0.23,
        "TTT": 0.225,
        "TTT_100": 0.22,
        "TTT_WRONG_SCENE": 0.25,
        "TTT_100_WRONG_SCENE": 0.25,
        "CTX_TTT_WRONG_SCENE": 0.31,
        "CTX_TTT_100_WRONG_SCENE": 0.31,
    }
    assert analyze(rows_for(base), [1, 2])["INTERPRETATION_BRANCH"] == "A"
    only_stream = {**base, "CTX_TTT_100": 0.30}
    assert analyze(rows_for(only_stream), [1, 2])["INTERPRETATION_BRANCH"] == "B"
    only_context = {**base, "TTT_100": 0.24}
    assert analyze(rows_for(only_context), [1, 2])["INTERPRETATION_BRANCH"] == "C"
    flat = {**base, "TTT_100": 0.24, "CTX_TTT_100": 0.30}
    result = analyze(rows_for(flat), [1, 2])
    assert result["INTERPRETATION_BRANCH"] == "D"
    assert result["contrasts"]["STREAM_DOSE"]["mean"] == pytest.approx(-0.015)
