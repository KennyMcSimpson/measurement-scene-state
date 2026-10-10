"""V27: the MAX rule and the hybrids' pixel rules, and the branch rule of the confirmation."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest


def script(name):
    path = Path(__file__).parents[1] / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_maxfill_keeps_near_pixels_and_takes_the_farther_depth_elsewhere():
    v27 = script("analyze_rgbd_hybrid_confirm_v27.py")
    near = np.array([[True, False], [False, False]])
    harmonic = np.array([[5.0, 2.0], [2.0, 2.0]])
    carrier = np.array([[1.0, 3.0], [1.5, 2.0]])
    assert np.array_equal(v27.maxfill(near, harmonic, carrier), [[5.0, 3.0], [2.0, 2.0]])
    assert np.array_equal(v27.hfill(near, harmonic, carrier), [[5.0, 3.0], [1.5, 2.0]])
    opacity = np.array([[1.0, 0.9], [0.1, 0.9]])
    assert np.array_equal(v27.ahfill(near, opacity, harmonic, carrier), [[5.0, 3.0], [2.0, 2.0]])


def rows_for(values):
    rows = []
    for seed in (1, 2):
        for i in range(8):
            for role in ("A", "B"):
                for qid in (14, 15):
                    for policy, value in values.items():
                        rows.append(
                            {
                                "seed": seed,
                                "cohort": "TRAIN_EXT" if i < 5 else "EVAL_V5",
                                "scene_id": f"s{i}",
                                "role": role,
                                "query_id": qid,
                                "policy": policy,
                                "depth_absrel": value + 0.002 * ((i + qid + seed) % 3),
                            }
                        )
    return rows


def test_branches_follow_the_two_primary_gates():
    v27 = script("analyze_rgbd_hybrid_confirm_v27.py")
    names = {name for pair in v27.CONTRASTS.values() for name in pair}
    base = dict.fromkeys(names, 0.25)
    base.update({"REPROJ7_HARMONIC": 0.20, "MAXFILL8_7_TTT_100": 0.19, "HFILL8_7_TTT_100": 0.18})
    result = v27.analyze(rows_for(base), [1, 2])
    assert result["INTERPRETATION_BRANCH"] == "A"
    assert result["contrasts"]["MAXFILL_VS_HYBRID"]["mean"] == pytest.approx(-0.01)
    only_max = {**base, "HFILL8_7_TTT_100": 0.21}
    assert v27.analyze(rows_for(only_max), [1, 2])["INTERPRETATION_BRANCH"] == "B"
    only_hybrid = {**base, "MAXFILL8_7_TTT_100": 0.21}
    assert v27.analyze(rows_for(only_hybrid), [1, 2])["INTERPRETATION_BRANCH"] == "C"
    neither = {**only_hybrid, "HFILL8_7_TTT_100": 0.20}
    assert v27.analyze(rows_for(neither), [1, 2])["INTERPRETATION_BRANCH"] == "D"
