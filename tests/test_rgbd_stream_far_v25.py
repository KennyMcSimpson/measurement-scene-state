"""V25: far queries never enter the stream, the hybrids, and the branch rule of the comparison."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from mcss.mechanism_pilot.rgbd_stream_ttt_v4 import stream_frame_ids


def script(name):
    path = Path(__file__).parents[1] / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def far_fixture():
    roles = {
        "anchor": 0,
        "context_a": [0, 1, 2],
        "context_b": [0, 3, 4],
        "primary_query": [14, 15],
        "query": [14, 15],
        "secondary_query": [],
        "stream": [],
        "far_query": [32, 49],
    }
    frames = [{"frame_id": i} for i in (*range(16), 32, 49)]
    return {"scene_id": "s", "split": "EVAL_FAR", "roles": roles, "frames": frames}


def test_far_queries_become_the_evaluated_queries_and_never_enter_the_stream():
    v25 = script("analyze_rgbd_stream_far_v25.py")
    record = far_fixture()
    assert {32, 49} & set(stream_frame_ids(record, "A"))
    derived = v25.far_record(record)
    assert derived["roles"]["primary_query"] == [32, 49]
    assert derived["roles"]["query"] == [14, 15]
    for key in ("anchor", "context_a", "context_b"):
        assert derived["roles"][key] == record["roles"][key]
    near = {**record, "frames": [f for f in record["frames"] if f["frame_id"] < 16]}
    for role in ("A", "B"):
        assert stream_frame_ids(derived, role) == stream_frame_ids(near, role)
    assert record["roles"]["primary_query"] == [14, 15]


def test_hybrids_follow_the_frozen_near_and_opacity_rules():
    v25 = script("analyze_rgbd_stream_far_v25.py")
    near = np.array([[True, False], [False, False]])
    opacity = np.array([[1.0, 0.2], [0.9, 0.6]])
    harmonic = np.full((2, 2), 5.0)
    carrier = np.arange(4.0).reshape(2, 2)
    assert np.array_equal(v25.hfill(near, harmonic, carrier), [[5.0, 1.0], [2.0, 3.0]])
    assert np.array_equal(v25.ahfill(near, opacity, harmonic, carrier), [[5.0, 5.0], [2.0, 3.0]])
    assert v25.sci(None) == "—" and v25.sci(0.5) == "5.00e-01"


def rows_for(values):
    rows = []
    for seed in (1, 2):
        for i in range(8):
            for role in ("A", "B"):
                for qid in (32, 49):
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


def test_branches_follow_the_two_primary_gates():
    v25 = script("analyze_rgbd_stream_far_v25.py")
    names = {name for pair in v25.CONTRASTS.values() for name in pair}
    base = dict.fromkeys(names, 0.30)
    base.update({"REPROJ7_HARMONIC": 0.30, "HFILL8_7_OFF": 0.27, "HFILL8_7_TTT_100": 0.26})
    result = v25.analyze(rows_for(base), [1, 2])
    assert result["INTERPRETATION_BRANCH"] == "A"
    assert result["contrasts"]["HYBRID_VS_MODEL_FREE_FAR"]["mean"] == pytest.approx(0.04)
    no_write_gain = {**base, "HFILL8_7_OFF": 0.26}
    assert v25.analyze(rows_for(no_write_gain), [1, 2])["INTERPRETATION_BRANCH"] == "B"
    strong_baseline = {**base, "REPROJ7_HARMONIC": 0.25}
    assert v25.analyze(rows_for(strong_baseline), [1, 2])["INTERPRETATION_BRANCH"] == "C"
    neither = {**no_write_gain, "REPROJ7_HARMONIC": 0.25}
    assert v25.analyze(rows_for(neither), [1, 2])["INTERPRETATION_BRANCH"] == "D"
