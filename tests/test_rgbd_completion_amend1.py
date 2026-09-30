"""V11 Amendment 1: only float-rounding RGB excursions are clipped; all else is unchanged."""

import re
from pathlib import Path

import numpy as np
import pytest
import torch

from mcss.mechanism_pilot import rgbd_completion_evaluation_amend1 as amended
from mcss.mechanism_pilot.statistics import measurement_metrics

ROOT = Path(__file__).parents[1]


def pred(values):
    """A rendered-output dict holding a 2x2 RGB image (renderer layout [1, 1, H, W, 3])."""
    rgb = torch.tensor(values, dtype=torch.float32).reshape(1, 1, 2, 2, 3)
    return {"rgb": rgb, "depth": torch.ones(1, 1, 1, 2, 2)}


def metrics(rgb):
    ones = torch.ones(2, 2)
    return measurement_metrics(rgb[0, 0], rgb[0, 0].clamp(0, 1), ones, ones, ones)


def test_values_inside_the_interval_are_returned_bit_identical():
    original = pred([[0.0, 0.5, 1.0], [0.25, 0.75, 1.0], [0.1, 0.2, 0.3], [1.0, 1.0, 1.0]])
    assert amended.clip_rounding(original) is original


def test_one_ulp_overshoot_is_clipped_and_then_passes_the_frozen_metric_guard():
    one_ulp = float(np.nextafter(np.float32(1.0), np.float32(2.0)))
    assert one_ulp - 1.0 == pytest.approx(1.1920928955078125e-07)
    raw = pred([[one_ulp, 0.5, 0.2], [0.1, -1e-8, 1.0], [0.3, 0.3, 0.3], [0.9, 0.8, 0.7]])
    with pytest.raises(ValueError, match="RGB must be in"):
        metrics(raw["rgb"])
    clipped = amended.clip_rounding(raw)["rgb"]
    assert float(clipped.max()) == 1.0 and float(clipped.min()) == 0.0
    assert int((clipped != raw["rgb"]).sum()) == 2
    metrics(clipped)


def test_excursions_beyond_rounding_still_raise_and_nonfinite_is_left_to_the_frozen_guard():
    base = [[0.5, 0.5, 0.5]] * 3
    with pytest.raises(ValueError, match="beyond float rounding"):
        amended.clip_rounding(pred([[1.00001, 0.5, 0.5], *base]))
    with pytest.raises(ValueError, match="beyond float rounding"):
        amended.clip_rounding(pred([[-0.001, 0.5, 0.5], *base]))
    nan = amended.clip_rounding(pred([[float("nan"), 0.5, 1.0], *base]))["rgb"]
    assert torch.isnan(nan).sum() == 1
    with pytest.raises(ValueError, match="finite"):
        metrics(nan)


def normalized(path, drop=()):
    """Source text without whitespace, trailing commas or the registered amendment tokens."""
    text = (ROOT / path).read_text()
    for token in drop:
        text = text.replace(token, "")
    text = re.sub(r"\s+", "", text)
    return re.sub(r",([\]\)])", r"\1", text)


def docstring_body(path):
    text = (ROOT / path).read_text()
    return text[text.index('"""') : text.index('"""', text.index('"""') + 3) + 3]


def test_the_amended_files_differ_from_the_frozen_ones_only_where_registered():
    frozen = normalized("src/mcss/mechanism_pilot/rgbd_completion_evaluation.py")
    amended_text = (
        ROOT / "src/mcss/mechanism_pilot/rgbd_completion_evaluation_amend1.py"
    ).read_text()
    start = amended_text.index("\n\nRGB_ROUNDING_TOLERANCE")
    end = amended_text.index("\n\n\ndef _control")
    body = amended_text[:start] + amended_text[end:]
    assert body.count("clip_rounding(renderer(") == 2
    body = body.replace("clip_rounding(renderer(", "renderer(").replace(
        '("rgb", "depth", "visibility")))', '("rgb", "depth", "visibility"))'
    )
    first = body.index('"""') + 3
    body = body[:first] + body[body.index("\n", first) :]
    frozen_text = (ROOT / "src/mcss/mechanism_pilot/rgbd_completion_evaluation.py").read_text()
    first = frozen_text.index('"""') + 3
    frozen_text = frozen_text[:first] + frozen_text[frozen_text.index("\n", first) :]
    assert re.sub(r"\s+", "", body) == re.sub(r"\s+", "", frozen_text)
    assert frozen  # the frozen evaluator itself is untouched and importable
    for frozen_script, amended_script in (
        (
            "scripts/train_rgbd_completion_carrier.py",
            "scripts/train_rgbd_completion_carrier_amend1.py",
        ),
        (
            "scripts/evaluate_rgbd_completion_carrier.py",
            "scripts/evaluate_rgbd_completion_carrier_amend1.py",
        ),
        (
            "scripts/run_rgbd_completion_experiment.py",
            "scripts/run_rgbd_completion_experiment_amend1.py",
        ),
    ):
        drop = (docstring_body(amended_script), "_amend1")
        assert normalized(amended_script, drop) == normalized(
            frozen_script, (docstring_body(frozen_script),)
        )
        assert "_amend1" in (ROOT / amended_script).read_text()
