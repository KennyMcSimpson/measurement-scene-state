"""V22: the explanation rule, the branch rule and the region AbsRel."""

import importlib.util
from pathlib import Path

import numpy as np


def script():
    path = Path(__file__).parents[1] / "scripts" / "analyze_rgbd_depth_readout_v22.py"
    spec = importlib.util.spec_from_file_location("analyze_rgbd_depth_readout_v22", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_explanation_and_branch_rules():
    v22 = script()
    raw = {"mean": 0.06, "ci95": [0.04, 0.08]}
    small = {"mean": 0.0, "ci95": [-0.01, 0.02]}
    assert v22.decide(raw, small, "SUPPORTED") == ("EXPLAINED", "A")
    assert v22.decide(raw, small, "PARTIAL") == ("EXPLAINED", "B")
    assert v22.decide(raw, {"mean": 0.02, "ci95": [0.0, 0.031]}, "SUPPORTED") == (
        "NOT_EXPLAINED",
        "C",
    )
    assert v22.decide(raw, {"mean": 0.02, "ci95": [0.0, 0.029]}, "SUPPORTED") == ("EXPLAINED", "A")


def test_region_absrel_ignores_empty_masks():
    v22 = script()
    gt, pred = np.array([[1.0, 2.0]]), np.array([[1.5, 2.0]])
    assert v22.absrel(pred, gt, np.array([[True, True]])) == 0.25
    assert v22.absrel(pred, gt, np.zeros((1, 2), bool)) is None
