"""V23: CTX_TTT is Core B V6's, the dose arms, the static seal boundary and the branch rule."""

import copy
import importlib.util
import json
from pathlib import Path

import pytest
import torch
from test_rgbd_stream_write import stream_fixture
from test_rgbd_stream_write_v3 import carrier, contexts

from mcss.dynamic.types import hash_scene_state
from mcss.mechanism_pilot import rgbd_static_ttt_v23 as v23
from mcss.mechanism_pilot import rgbd_stream_ttt_context_v6 as v6


def script(name):
    path = Path(__file__).parents[1] / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_no_stream_and_ctx_ttt_are_core_b_v6s():
    warm, stream, bounds, _ = contexts()
    model = carrier()
    settings = {"steps": v23.STEPS["CTX_TTT"], "learning_rate": v23.LEARNING_RATE}
    for policy in ("NO_STREAM", "CTX_TTT"):
        theirs = v6.unroll_v6(model, warm, stream, bounds, policy, "e", settings, 7)
        mine = v23.unroll_static(model, warm, bounds, policy, "e", 7)
        assert hash_scene_state(mine[0]) == hash_scene_state(theirs[0])
        assert mine[4] == pytest.approx(theirs[4])
        assert mine[3].source_frames == [o.frame_id for o in warm]


def test_dose_arms_take_their_own_steps(monkeypatch):
    monkeypatch.setitem(v23.STEPS, "CTX_TTT_100", 6)
    warm, _, bounds, _ = contexts()
    model = carrier()
    ten = v23.unroll_static(model, warm, bounds, "CTX_TTT", "e", 7)
    six = v23.unroll_static(model, warm, bounds, "CTX_TTT_100", "e", 7)
    assert ten[2].step == 10 and len(ten[4]) == 10
    assert six[2].step == 6 and len(six[4]) == 6
    assert hash_scene_state(ten[0]) != hash_scene_state(six[0])


def test_static_states_are_sealed_before_query_depth_without_streams(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    monkeypatch.setitem(v23.STEPS, "CTX_TTT", 2)
    monkeypatch.setitem(v23.STEPS, "CTX_TTT_30", 3)
    monkeypatch.setitem(v23.STEPS, "CTX_TTT_100", 4)
    manifest = stream_fixture(tmp_path)
    cohort = copy.deepcopy(manifest)
    cohort["scenes"] = [r for r in cohort["scenes"] if r["split"] == "DEV"]
    out = tmp_path / "evaluation"
    rows = v23.evaluate_static(carrier(), {"X": (cohort, tmp_path)}, {"X": "DEV"}, out, 7, "cpu")
    assert len(rows) == 2 * 2 * 2 * (len(v23.POLICIES) + len(v23.CONTROLS))
    access = json.loads((out / "GT_access.json").read_text())
    marker = next(i for i, e in enumerate(access) if e.get("event") == "ALL_STATIC_STATES_SEALED")
    early = [e for e in access[:marker] if "depth" in e.get("channels", [])]
    assert {e["purpose"] for e in early} == {"CONTEXT_ONLY_RGBD"}
    assert all(e["frame_id"] not in (5, 6, 7, 8, 9, 10) for e in early)
    meta = json.loads((out / "state_hashes.json").read_text())
    donors = {m["scene_id"]: m["donor_scene_id"] for m in meta.values() if "donor_scene_id" in m}
    assert donors == {"capacity0": "capacity1", "capacity1": "capacity0"}
    assert {m["policy"]: m["fast_step"] for m in meta.values() if "fast_step" in m} == {
        "NO_STREAM": 0,
        "CTX_TTT": 2,
        "CTX_TTT_30": 3,
        "CTX_TTT_100": 4,
    }


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
    run = script("run_rgbd_static_ttt_eval_v23.py")
    base = {
        "NO_STREAM": 0.30,
        "CTX_TTT": 0.29,
        "CTX_TTT_30": 0.285,
        "CTX_TTT_100": 0.28,
        "CTX_TTT_WRONG_SCENE": 0.30,
    }
    analyze = run.analyze
    assert (
        analyze(rows_for(base), [1, 2], run.CONTRASTS, run.PRIMARY)["INTERPRETATION_BRANCH"] == "A"
    )
    same = {**base, "CTX_TTT_WRONG_SCENE": 0.29}
    assert (
        analyze(rows_for(same), [1, 2], run.CONTRASTS, run.PRIMARY)["INTERPRETATION_BRANCH"] == "B"
    )
    flat = {**base, "CTX_TTT": 0.30}
    assert (
        analyze(rows_for(flat), [1, 2], run.CONTRASTS, run.PRIMARY)["INTERPRETATION_BRANCH"] == "C"
    )
    result = analyze(rows_for(base), [1, 2], run.CONTRASTS, run.PRIMARY)
    assert result["contrasts"]["DOSE_100_VS_10"]["mean"] == pytest.approx(0.01)
