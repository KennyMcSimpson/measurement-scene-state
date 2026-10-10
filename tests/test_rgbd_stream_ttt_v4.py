"""Core B V4: gradient test-time writes, the TRAIN-only selection rule and the branch rule."""

import importlib.util
from pathlib import Path

import pytest
from test_rgbd_stream_write_v3 import carrier, contexts

from mcss.dynamic.types import hash_scene_state, hash_value
from mcss.mechanism_pilot import rgbd_stream_ttt_evaluation_v4 as evaluation
from mcss.mechanism_pilot import rgbd_stream_ttt_v4 as v4


def script(name):
    path = Path(__file__).parents[1] / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_ttt_writes_only_the_fast_matrices_and_lowers_the_stream_loss():
    warm, stream, bounds, _ = contexts()
    model = carrier()
    before = hash_value(model.state_dict())
    settings = {"steps": 8, "learning_rate": 1e-2}
    generator = v4.episode_generator(7, "e")
    state, _, fast, _, losses = v4.unroll_ttt(
        model, warm, stream, bounds, "TTT", "e", settings, generator
    )
    off, _, off_fast, _, none = v4.unroll_ttt(model, warm, stream, bounds, "OFF", "e", None, None)
    assert len(losses) == 8 and none == [] and fast.step == 8 and off_fast.step == 0
    assert float(fast.delta_fuse.abs().sum() + fast.delta_complete.abs().sum()) > 0
    assert float(off_fast.delta_fuse.abs().sum()) == 0.0
    assert losses[-1] < losses[0]
    assert hash_scene_state(state) != hash_scene_state(off)
    assert hash_value(model.state_dict()) == before
    assert all(not p.requires_grad for p in model.parameters())
    again = v4.unroll_ttt(
        model, warm, stream, bounds, "TTT", "e", settings, v4.episode_generator(7, "e")
    )
    assert again[4] == pytest.approx(losses)


def test_no_stream_and_off_match_core_b_v3():
    from mcss.mechanism_pilot import rgbd_stream_write_v3 as v3

    warm, stream, bounds, _ = contexts()
    model = carrier()
    rule = v3.make_write_rule(model, 1)
    for policy in ("NO_STREAM", "OFF"):
        mine = v4.unroll_ttt(model, warm, stream, bounds, policy, "e", None, None)[0]
        theirs = v3.unroll(model, rule, warm, stream, bounds, policy, "e")[0]
        assert hash_scene_state(mine) == hash_scene_state(theirs)


def test_selection_rule_and_episode_order():
    grid = v4.GRID
    assert evaluation.choose({0: 0.01, 1: 0.03, 2: 0.03, 3: -0.1}, grid) == 1
    assert evaluation.choose({0: 0.02, 1: 0.01, 2: 0.02, 3: 0.0}, grid) == 0
    eligible = [f"ai_{i:03d}_001/{r}" for i in range(40) for r in ("A", "B")]
    picked = evaluation.selection_episodes(eligible, 32)
    assert len(picked) == 32 and picked == evaluation.selection_episodes(
        list(reversed(eligible)), 32
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
    analyze = script("run_rgbd_stream_ttt_v4.py").analyze
    base = {"NO_STREAM": 0.33, "OFF": 0.24, "TTT": 0.20, "TTT_WRONG_SCENE": 0.26}
    result = analyze(rows_for(base), [1, 2])
    assert result["contrasts"]["TTT_BEYOND_BASELINE"]["mean"] == pytest.approx(0.04)
    assert result["INTERPRETATION_BRANCH"] == "A"
    result = analyze(rows_for({**base, "TTT_WRONG_SCENE": 0.20}), [1, 2])
    assert result["INTERPRETATION_BRANCH"] == "B"
    result = analyze(rows_for({**base, "TTT": 0.25}), [1, 2])
    assert result["INTERPRETATION_BRANCH"] == "C"
