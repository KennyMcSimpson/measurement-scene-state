"""V24: the hybrids' pixel rules, the 7-frame points and the branch rule of the comparison."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch
from test_rgbd_stream_write_v3 import contexts

from mcss.mechanism_pilot.rgbd_evidence_carrier import RGBDContext


def script(name):
    path = Path(__file__).parents[1] / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_hybrids_take_the_harmonic_fill_on_near_and_low_opacity_pixels():
    v24 = script("analyze_rgbd_stream_baselines_v24.py")
    near = np.array([[True, False], [False, False]])
    opacity = np.array([[1.0, 0.2], [0.9, 0.6]])
    harmonic = np.full((2, 2), 5.0)
    carrier = np.arange(4.0).reshape(2, 2)
    assert np.array_equal(v24.hfill(near, harmonic, carrier), [[5.0, 1.0], [2.0, 3.0]])
    assert np.array_equal(v24.ahfill(near, opacity, harmonic, carrier), [[5.0, 5.0], [2.0, 3.0]])
    assert v24.absrel(carrier + 1, carrier + 1, near) == 0.0
    assert v24.absrel(carrier, carrier, np.zeros((2, 2), dtype=bool)) is None


def test_seven_frame_points_add_the_stream_frames_to_the_context():
    geo = script("run_rgbd_geometric_baselines.py")
    warm, stream, _, _ = contexts()
    seven = RGBDContext([*warm, *stream], torch.cat((warm.depths, stream.depths)))
    three, both = geo.context_points(warm), geo.context_points(seven)
    assert len(both) == len(three) + len(geo.context_points(stream))
    assert torch.equal(both[: len(three)], three)


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


def test_branches_follow_the_two_primary_gates():
    v24 = script("analyze_rgbd_stream_baselines_v24.py")
    names = {name for pair in v24.CONTRASTS.values() for name in pair}
    base = dict.fromkeys(names, 0.25)
    base.update({"REPROJ7_HARMONIC": 0.20, "HFILL8_7_OFF": 0.19, "HFILL8_7_TTT_100": 0.18})
    result = v24.analyze(rows_for(base), [1, 2])
    assert result["INTERPRETATION_BRANCH"] == "A"
    assert result["contrasts"]["HYBRID_VS_MODEL_FREE"]["mean"] == pytest.approx(0.02)
    strong_baseline = {**base, "REPROJ7_HARMONIC": 0.17}
    assert v24.analyze(rows_for(strong_baseline), [1, 2])["INTERPRETATION_BRANCH"] == "B"
    no_write_gain = {**base, "HFILL8_7_OFF": 0.18}
    assert v24.analyze(rows_for(no_write_gain), [1, 2])["INTERPRETATION_BRANCH"] == "C"
    neither = {**no_write_gain, "REPROJ7_HARMONIC": 0.18}
    assert v24.analyze(rows_for(neither), [1, 2])["INTERPRETATION_BRANCH"] == "D"
