"""Core B V11: online writes per arriving frame, one-shot rows, the sealed run and the branch."""

import copy
import importlib.util
import json
from pathlib import Path

import pytest
import torch
from test_rgbd_stream_write import stream_fixture
from test_rgbd_stream_write_v3 import carrier, contexts

from mcss.dynamic.types import hash_scene_state, hash_value
from mcss.mechanism_pilot import rgbd_stream_ttt_v4 as v4
from mcss.mechanism_pilot import rgbd_stream_ttt_v7 as v7
from mcss.mechanism_pilot import rgbd_stream_ttt_v11 as v11


def script(name):
    path = Path(__file__).parents[1] / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_one_shot_rows_are_core_b_v7s():
    for name in ("NO_STREAM", "OFF", "TTT_100"):
        assert v11.POLICY_SPECS[name] == v7.POLICY_SPECS[name]
    assert v11.POLICY_SPECS["ONLINE_100"] == (True, "online", 100)
    assert v11.POLICY_SPECS["ONLINE_NEW_100"] == (True, "online_new", 100)
    assert v11.POLICY_SPECS["ALL_TTT_100"] == (True, "all", 100)
    assert v11.CONTROLS == {"ONLINE_100_WRONG_SCENE": "ONLINE_100"}


def test_one_shot_policies_unroll_exactly_as_core_b_v7():
    warm, stream, bounds, _ = contexts()
    model = carrier()
    specs = {"NO_STREAM": (False, None, 0), "OFF": (True, None, 0), "TTT_100": (True, "stream", 3)}
    for policy in specs:
        mine = v11.unroll_v11(model, warm, stream, bounds, policy, "e", 7, specs)
        theirs = v7.unroll_v7(model, warm, stream, bounds, policy, "e", 7, specs)
        assert hash_scene_state(mine[0]) == hash_scene_state(theirs[0])
        assert mine[4] == pytest.approx(theirs[4])


def test_online_write_on_a_single_frame_is_the_one_shot_write():
    warm, stream, bounds, _ = contexts(n_stream=1)
    model = carrier()
    batch = v4.cached_episode(model, warm, stream, bounds, "e", True)
    fast_b, losses_b = v4.ttt_fast(model, batch, stream, 3, 1e-3, v4.episode_generator(7, "g"))
    online = v4.cached_episode(model, warm, stream, bounds, "e", False)
    fast_o, losses_o = v11.ttt_online(
        model, online, stream, 3, 1e-3, v4.episode_generator(7, "g"), True
    )
    assert losses_o == losses_b and fast_o.step == fast_b.step == 3
    assert torch.equal(fast_o.delta_fuse, fast_b.delta_fuse)
    assert torch.equal(fast_o.delta_complete, fast_b.delta_complete)
    with torch.no_grad():
        assert hash_scene_state(online.run(fast_o, "materialize")) == hash_scene_state(
            batch.run(fast_b, "materialize")
        )


def test_online_policies_cache_every_frame_and_differ_only_in_replay():
    warm, stream, bounds, _ = contexts()
    model = carrier()
    before = hash_value(model.state_dict())
    specs = {name: (s, sup, 8) for name, (s, sup, _) in v11.POLICY_SPECS.items()}
    replay = v11.unroll_v11(model, warm, stream, bounds, "ONLINE_100", "e", 7, specs)
    newest = v11.unroll_v11(model, warm, stream, bounds, "ONLINE_NEW_100", "e", 7, specs)
    both = v11.unroll_v11(model, warm, stream, bounds, "ALL_TTT_100", "e", 7, specs)
    ids = [o.frame_id for o in warm] + [o.frame_id for o in stream]
    for result in (replay, newest, both):
        assert result[3].source_frames == ids and result[2].step == 8 and len(result[4]) == 8
    assert hash_scene_state(replay[0]) != hash_scene_state(newest[0])
    assert hash_value(model.state_dict()) == before
    uneven = {"ONLINE_100": (True, "online", 6)}
    with pytest.raises(ValueError):
        v11.unroll_v11(model, warm, stream, bounds, "ONLINE_100", "e", 7, uneven)
    again = v11.unroll_v11(model, warm, stream, bounds, "ONLINE_100", "e", 7, specs)
    assert again[4] == pytest.approx(replay[4])


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
    specs = {name: (s, sup, min(steps, 4)) for name, (s, sup, steps) in v11.POLICY_SPECS.items()}
    out = tmp_path / "evaluation"
    rows = v11.evaluate_v11(carrier(), cohorts, expected, out, 7, "cpu", specs)
    assert len(rows) == 2 * 2 * 2 * (len(specs) + len(v11.CONTROLS))
    access = json.loads((out / "GT_access.json").read_text())
    marker = next(i for i, e in enumerate(access) if e.get("event") == "ALL_STREAM_STATES_SEALED")
    early = [e for e in access[:marker] if "depth" in e.get("channels", [])]
    assert {e["purpose"] for e in early} == {"CONTEXT_ONLY_RGBD", "STREAM_RGBD"}
    assert all(e["frame_id"] not in (8, 9) for e in early)
    meta = json.loads((out / "state_hashes.json").read_text())
    donors = {m["scene_id"]: m["donor_scene_id"] for m in meta.values() if "donor_scene_id" in m}
    assert donors == {"capacity0": "capacity1", "capacity1": "capacity0"}
    for m in meta.values():
        if m["policy"] == "ALL_TTT_100":
            assert m["supervised_frame_ids"] == m["warmup_frame_ids"] + m["stream_frame_ids"]
        if m["policy"] in ("ONLINE_100", "ONLINE_NEW_100"):
            assert m["supervised_frame_ids"] == m["stream_frame_ids"] and m["fast_step"] == 4


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
                                "cohort": "EVAL_V3",
                                "scene_id": f"s{i}",
                                "role": role,
                                "query_id": qid,
                                "policy": policy,
                                "depth_absrel": value + 0.002 * ((i + qid + seed) % 3),
                            }
                        )
    return rows


def test_branches_follow_the_online_gain_and_the_midpoint():
    analyze = script("run_rgbd_stream_ttt_online_v11.py").analyze
    base = {
        "NO_STREAM": 0.30,
        "OFF": 0.24,
        "TTT_100": 0.22,
        "ONLINE_100": 0.222,
        "ONLINE_NEW_100": 0.23,
        "ALL_TTT_100": 0.215,
        "ONLINE_100_WRONG_SCENE": 0.25,
    }
    result = analyze(rows_for(base), [1, 2])
    assert result["INTERPRETATION_BRANCH"] == "A"
    assert result["statuses"]["P3_REPLAY_GAIN_STATUS"] == "SUPPORTED"
    assert result["contrasts"]["ONLINE_RETAINS_HALF"]["mean"] == pytest.approx(0.008)
    assert result["contrasts"]["ONLINE_VS_ONE_SHOT"]["mean"] == pytest.approx(0.002)
    weak = {**base, "ONLINE_100": 0.235}
    assert analyze(rows_for(weak), [1, 2])["INTERPRETATION_BRANCH"] == "B"
    none = {**base, "ONLINE_100": 0.24, "ONLINE_NEW_100": 0.24}
    result = analyze(rows_for(none), [1, 2])
    assert result["INTERPRETATION_BRANCH"] == "C"
    assert result["statuses"]["P3_REPLAY_GAIN_STATUS"] != "SUPPORTED"
