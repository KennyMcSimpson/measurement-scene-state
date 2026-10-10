"""Core B V9: Core B V7's policy rows plus TTT_300, the sealed TRAIN-EXT run and the branch rule."""

import copy
import importlib.util
import json
from pathlib import Path

import pytest
import torch
from test_rgbd_stream_write import stream_fixture
from test_rgbd_stream_write_v3 import carrier

from mcss.mechanism_pilot import rgbd_stream_ttt_v7 as v7
from mcss.mechanism_pilot import rgbd_stream_ttt_v9 as v9


def script(name):
    path = Path(__file__).parents[1] / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_policy_rows_are_core_b_v7s_plus_300_stream_steps():
    assert set(v9.POLICY_SPECS) - set(v7.POLICY_SPECS) == {"TTT_300"}
    assert all(v9.POLICY_SPECS[k] == v7.POLICY_SPECS[k] for k in v9.POLICY_SPECS if k != "TTT_300")
    assert v9.POLICY_SPECS["TTT_300"] == (True, "stream", 300)
    assert v9.CONTROLS == {"TTT_300_WRONG_SCENE": "TTT_300"}
    assert v9.LEARNING_RATE == v7.LEARNING_RATE == 1e-3


def test_every_policy_and_control_is_sealed_before_query_depth(tmp_path):
    torch.set_num_threads(1)
    manifest = stream_fixture(tmp_path)
    cohort = copy.deepcopy(manifest)
    cohort["scenes"] = [r for r in cohort["scenes"] if r["split"] == "DEV"]
    cohorts, expected = {"TRAIN_EXT": (cohort, tmp_path)}, {"TRAIN_EXT": "DEV"}
    specs = {name: (s, sup, min(steps, 2)) for name, (s, sup, steps) in v9.POLICY_SPECS.items()}
    specs["TTT_300"] = (True, "stream", 3)
    out = tmp_path / "evaluation"
    rows = v9.evaluate_v9(carrier(), cohorts, expected, out, 7, "cpu", specs)
    assert len(rows) == 2 * 2 * 2 * (len(specs) + len(v9.CONTROLS))
    access = json.loads((out / "GT_access.json").read_text())
    marker = next(i for i, e in enumerate(access) if e.get("event") == "ALL_STREAM_STATES_SEALED")
    early = [e for e in access[:marker] if "depth" in e.get("channels", [])]
    assert {e["purpose"] for e in early} == {"CONTEXT_ONLY_RGBD", "STREAM_RGBD"}
    assert all(e["frame_id"] not in (8, 9) for e in early)
    meta = json.loads((out / "state_hashes.json").read_text())
    donors = {m["scene_id"]: m["donor_scene_id"] for m in meta.values() if "donor_scene_id" in m}
    assert donors == {"capacity0": "capacity1", "capacity1": "capacity0"}
    assert {m["donor_policy"] for m in meta.values() if "donor_scene_id" in m} == {"TTT_300"}
    steps = {m["policy"]: m["fast_step"] for m in meta.values() if "fast_step" in m}
    assert steps == {"NO_STREAM": 0, "OFF": 0, "OFF_CTX_TTT_100": 2, "TTT_100": 2, "TTT_300": 3}


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
                                "cohort": "TRAIN_EXT",
                                "scene_id": f"s{i}",
                                "role": role,
                                "query_id": qid,
                                "policy": policy,
                                "depth_absrel": value + 0.002 * ((i + qid + seed) % 3),
                            }
                        )
    return rows


def test_branches_follow_the_two_primary_gates():
    analyze = script("run_rgbd_stream_ttt_dose_v9.py").analyze
    base = {
        "NO_STREAM": 0.30,
        "OFF": 0.24,
        "OFF_CTX_TTT_100": 0.23,
        "TTT_100": 0.21,
        "TTT_300": 0.20,
        "TTT_300_WRONG_SCENE": 0.25,
    }
    assert analyze(rows_for(base), [1, 2])["INTERPRETATION_BRANCH"] == "A"
    same_source = {**base, "OFF_CTX_TTT_100": 0.21}
    assert analyze(rows_for(same_source), [1, 2])["INTERPRETATION_BRANCH"] == "B"
    saturated = {**base, "TTT_300": 0.21}
    assert analyze(rows_for(saturated), [1, 2])["INTERPRETATION_BRANCH"] == "C"
    flat = {**base, "TTT_300": 0.21, "OFF_CTX_TTT_100": 0.21}
    result = analyze(rows_for(flat), [1, 2])
    assert result["INTERPRETATION_BRANCH"] == "D"
    assert result["contrasts"]["STREAM_TTT_300"]["mean"] == pytest.approx(0.03)


def test_core_b_v6_rows_are_matched_only_for_policies_without_ttt():
    agreement = script("run_rgbd_stream_ttt_dose_v9.py").row_agreement
    mine = rows_for({"NO_STREAM": 0.30, "OFF": 0.24, "TTT_100": 0.21})
    theirs = rows_for({"NO_STREAM": 0.30, "OFF": 0.25, "TTT_100": 0.0})
    result = agreement(mine, theirs[: len(theirs) // 2])
    assert result["rows"] == 2 * 8 * 2 * 2 * 2
    assert result["rows_compared"] == result["rows"] // 2
    assert result["rows_identical"] == result["rows_compared"] // 2
    assert result["max_abs_difference"] == pytest.approx(0.01)
    assert agreement(mine, [])["rows_compared"] == 0
