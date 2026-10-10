"""Core B V6: context-only TTT, Core B V4's policies unchanged, the seal and the branch rule."""

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
from mcss.mechanism_pilot import rgbd_stream_ttt_v4 as v4

SETTINGS = {"steps": 3, "learning_rate": 1e-2}


def script(name):
    path = Path(__file__).parents[1] / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_no_stream_off_and_ttt_are_core_b_v4s(tmp_path):
    warm, stream, bounds, _ = contexts()
    model = carrier()
    for policy in ("NO_STREAM", "OFF", "TTT"):
        generator = v4.episode_generator(7, "e") if policy == "TTT" else None
        theirs = v4.unroll_ttt(model, warm, stream, bounds, policy, "e", SETTINGS, generator)
        mine = v6.unroll_v6(model, warm, stream, bounds, policy, "e", SETTINGS, 7)
        assert hash_scene_state(mine[0]) == hash_scene_state(theirs[0])
        assert mine[4] == pytest.approx(theirs[4])


def test_context_ttt_writes_only_fast_matrices_from_the_context_frames():
    warm, stream, bounds, _ = contexts()
    model = carrier()
    before = hash_value(model.state_dict())
    no_stream = v6.unroll_v6(model, warm, stream, bounds, "NO_STREAM", "e", SETTINGS, 7)
    ctx = v6.unroll_v6(model, warm, stream, bounds, "CTX_TTT", "e", SETTINGS, 7)
    off_ctx = v6.unroll_v6(model, warm, stream, bounds, "OFF_CTX_TTT", "e", SETTINGS, 7)
    ttt = v6.unroll_v6(model, warm, stream, bounds, "TTT", "e", SETTINGS, 7)
    assert len(ctx[4]) == SETTINGS["steps"] and ctx[2].step == SETTINGS["steps"]
    assert ctx[4][-1] < ctx[4][0]
    assert hash_scene_state(ctx[0]) != hash_scene_state(no_stream[0])
    assert hash_scene_state(off_ctx[0]) != hash_scene_state(ttt[0])
    warm_ids, stream_ids = [o.frame_id for o in warm], [o.frame_id for o in stream]
    assert ctx[3].source_frames == warm_ids
    assert off_ctx[3].source_frames == warm_ids + stream_ids
    assert hash_value(model.state_dict()) == before
    again = v6.unroll_v6(model, warm, stream, bounds, "CTX_TTT", "e", SETTINGS, 7)
    assert again[4] == pytest.approx(ctx[4])


def one_cohort(manifest, root):
    cohort = copy.deepcopy(manifest)
    cohort["scenes"] = [r for r in cohort["scenes"] if r["split"] == "DEV"]
    return {"TRAIN_EXT": (cohort, root)}, {"TRAIN_EXT": "DEV"}


def test_every_policy_and_control_is_sealed_before_query_depth(tmp_path):
    torch.set_num_threads(1)
    manifest = stream_fixture(tmp_path)
    cohorts, expected = one_cohort(manifest, tmp_path)
    out = tmp_path / "evaluation"
    rows = v6.evaluate_v6(carrier(), SETTINGS, cohorts, expected, out, 7, "cpu")
    names = (*v6.POLICIES, *v6.CONTROLS)
    assert len(rows) == 2 * 2 * 2 * len(names)
    access = json.loads((out / "GT_access.json").read_text())
    marker = next(i for i, e in enumerate(access) if e.get("event") == "ALL_STREAM_STATES_SEALED")
    early = [e for e in access[:marker] if "depth" in e.get("channels", [])]
    assert {e["purpose"] for e in early} == {"CONTEXT_ONLY_RGBD", "STREAM_RGBD"}
    assert all(e["frame_id"] not in (8, 9) for e in early)
    meta = json.loads((out / "state_hashes.json").read_text())
    for control, policy in v6.CONTROLS.items():
        donors = {
            m["scene_id"]: m["donor_scene_id"] for m in meta.values() if m["policy"] == control
        }
        assert donors == {"capacity0": "capacity1", "capacity1": "capacity0"}
        assert {m["donor_policy"] for m in meta.values() if m["policy"] == control} == {policy}
    ctx = [m for m in meta.values() if m["policy"] == "CTX_TTT"]
    assert all(
        m["stream_frame_ids"] == [] and m["supervised_frame_ids"] == m["warmup_frame_ids"]
        for m in ctx
    )
    ttt = [m for m in meta.values() if m["policy"] == "TTT"]
    assert all(m["supervised_frame_ids"] == m["stream_frame_ids"] for m in ttt)
    with pytest.raises(PermissionError):
        v6.cohort_records(cohorts, {"TRAIN_EXT": "TRAIN"})


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
    analyze = script("run_rgbd_stream_ttt_context_v6.py").analyze
    base = {
        "NO_STREAM": 0.30,
        "CTX_TTT": 0.28,
        "OFF": 0.24,
        "OFF_CTX_TTT": 0.23,
        "TTT": 0.22,
        "TTT_WRONG_SCENE": 0.25,
        "CTX_TTT_WRONG_SCENE": 0.31,
    }
    assert analyze(rows_for(base), [1, 2])["INTERPRETATION_BRANCH"] == "A"
    assert analyze(rows_for({**base, "CTX_TTT": 0.30}), [1, 2])["INTERPRETATION_BRANCH"] == "B"
    assert analyze(rows_for({**base, "TTT": 0.24}), [1, 2])["INTERPRETATION_BRANCH"] == "C"
    result = analyze(rows_for({**base, "TTT": 0.24, "CTX_TTT": 0.30}), [1, 2])
    assert result["INTERPRETATION_BRANCH"] == "D"
    assert result["contrasts"]["SUPERVISION_SOURCE"]["mean"] == pytest.approx(-0.01)
