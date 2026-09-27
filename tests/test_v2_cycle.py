import inspect

import numpy as np
import pytest
import torch

from mcss.vision_probe.v2_cycle import UnsupportedSourceAnnotation, source_cycle


def test_independent_direction_roundtrip_not_forced_identity():
    source = torch.eye(3).reshape(1, 3, 3)
    target = source[:, [1, 0]].clone()
    result = source_cycle(source, target, np.array([[1, 0, 2]]), "s", "t")
    # Source token 2 has no corresponding target: ties choose target 0 (source token 1).
    assert result["per_object"][0]["soft_iou"] == 1
    assert result["per_object"][1]["soft_iou"] == 0
    assert result["L_cycle"] == 0.5
    assert result["cost"]["correspondence_calls"] == 2


def test_foreground_objects_equal_weight_not_background_or_area():
    source = torch.eye(4).reshape(1, 4, 4)
    target = source[:, :1].clone()
    result = source_cycle(source, target, np.array([[1, 1, 1, 2]]), "s", "t")
    assert result["per_object"][0]["soft_iou"] == 0.75
    assert result["per_object"][1]["soft_iou"] == 0
    assert result["L_cycle"] == 0.625


def test_area_occupancy_product_soft_iou_retains_small_object():
    features = torch.ones(1, 1, 2)
    result = source_cycle(features, features, np.array([[0, 0], [0, 7]]), "s", "t")
    # a=b=.25, intersection=.0625, union=.4375. Identity on fractional occupancy is not IoU 1.
    assert result["per_object"][0]["object_id"] == 7
    assert result["per_object"][0]["soft_iou"] == pytest.approx(1 / 7)


def test_zero_features_finite_and_do_not_fabricate_identity():
    zeros = torch.zeros(2, 2, 3)
    mask = np.array([[0, 1], [0, 0]])
    result = source_cycle(zeros, zeros, mask, "s", "t")
    assert result["L_cycle"] == 1
    assert result["target_to_source_unique"] == 1
    assert result["source_to_target_unique"] == 1
    assert result["zero_source_tokens"] == 4


def test_source_annotation_required_distinct_frames_and_target_gt_impossible_api():
    f = torch.ones(1, 1, 2)
    for mask in (None, np.zeros((2, 2), dtype=int), np.full((2, 2), 255)):
        with pytest.raises(UnsupportedSourceAnnotation, match="UNSUPPORTED"):
            source_cycle(f, f, mask, "s", "t")
    with pytest.raises(ValueError, match="different frames"):
        source_cycle(f, f, np.ones((2, 2), dtype=int), "s", "s")
    assert set(inspect.signature(source_cycle).parameters) == {
        "source_features",
        "target_features",
        "source_mask",
        "source_id",
        "target_id",
    }
    with pytest.raises(TypeError):
        source_cycle(f, f, np.ones((1, 1), dtype=int), "s", "t", target_mask=np.ones((1, 1)))
