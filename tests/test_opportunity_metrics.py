import inspect
import json

import numpy as np
import pytest
import torch

from mcss.vision_probe.geometry import mask_propagation_metrics
from mcss.vision_probe.opportunity_metrics import (
    binary_scores,
    correspondence,
    evaluate_pair,
    size_bucket,
    transfer_probabilities,
)


def _identity_features(height=4, width=4):
    return torch.eye(height * width).reshape(height, width, -1)


def test_matching_is_mask_free_and_respects_spatial_permutation():
    assert list(inspect.signature(correspondence).parameters) == [
        "source_features",
        "target_features",
    ]
    source = _identity_features()
    result = correspondence(source, source.flip(1))
    torch.testing.assert_close(result, torch.arange(16).reshape(4, 4).flip(1))


def test_pixel_transfer_identity_rectangular_original_size_and_json():
    features = _identity_features()
    mask = np.zeros((40, 80), dtype=np.uint8)
    mask[:, :40] = 1
    result = evaluate_pair(features, features, mask, mask)
    assert result["J"] == result["F"] == result["JF"] == 1.0
    assert result["foreground_pixel_count"] == 1600
    assert result["foreground_token_counts"] == {"224": 128, "448": 512}
    assert result["objects"][0]["size_bucket"] == "large"
    json.dumps(result, allow_nan=False)


def test_target_labels_cannot_change_matching_or_transferred_prediction(monkeypatch):
    import mcss.vision_probe.opportunity_metrics as metrics

    features = _identity_features()
    source = np.zeros((16, 16), dtype=np.uint8)
    source[:, :8] = 1
    original = metrics.transfer_probabilities
    captured = []

    def record(*args, **kwargs):
        result = original(*args, **kwargs)
        captured.append(result)
        return result

    monkeypatch.setattr(metrics, "transfer_probabilities", record)
    correct = evaluate_pair(features, features, source, source)
    changed_target = np.full_like(source, 2)
    changed = evaluate_pair(features, features, source, changed_target)
    np.testing.assert_array_equal(captured[0][1], captured[1][1])
    assert set(captured[1]) == {1}  # target-only object never enters prediction
    assert correct["J"] == 1 and changed["J"] == 0
    assert [row["object_id"] for row in changed["objects"]] == [1, 2]


def test_area_occupancy_is_not_nearest_label_and_tiny_objects_are_retained():
    source = np.zeros((16, 16), dtype=np.uint8)
    source[0, 0] = 7
    features = _identity_features(1, 1)
    probability = transfer_probabilities(
        source, (1, 1), torch.zeros(1, 1, dtype=torch.long), (16, 16)
    )
    np.testing.assert_allclose(probability[7], 1 / 256)
    result = evaluate_pair(features, features, source, source)
    assert result["J"] == 0.0
    assert len(result["objects"]) == 1
    assert result["objects"][0]["pixel_count"] == 1
    assert result["objects"][0]["size_bucket"] == "tiny"


def test_binary_scores_empty_shifted_and_disjoint_geometry():
    empty = np.zeros((100, 100), dtype=bool)
    square = empty.copy()
    square[20:40, 20:40] = True
    assert binary_scores(empty, empty) == (1.0, 1.0)
    assert binary_scores(square, empty) == (0.0, 0.0)
    assert binary_scores(empty, square) == (0.0, 0.0)
    assert binary_scores(square, square) == (1.0, 1.0)
    shifted = np.roll(square, 1, axis=1)
    jaccard, boundary_f = binary_scores(shifted, square)
    assert jaccard == pytest.approx(380 / 420)
    assert boundary_f == 1.0  # ceil(.008 * sqrt(2) * 100) == 2
    assert binary_scores(np.roll(square, 50, axis=1), square) == (0.0, 0.0)


def test_void_does_not_count_as_an_object_or_foreground():
    features = _identity_features()
    mask = np.zeros((4, 4), dtype=np.uint8)
    mask[:, :2] = 1
    mask[:, 3] = 255
    result = evaluate_pair(features, features, mask, mask)
    assert result["J"] == result["F"] == 1.0
    assert result["foreground_pixel_count"] == 8
    assert [row["object_id"] for row in result["objects"]] == [1]


def test_legacy_token_iou_matches_existing_metric():
    source = _identity_features()
    target = source.flip(1)
    labels = torch.tensor([[0, 0, 1, 1]] * 4)
    expected = mask_propagation_metrics(source, target, labels, labels)["mean_iou"]
    result = evaluate_pair(source, target, labels.numpy(), labels.numpy())
    assert result["token_iou"] == expected


@pytest.mark.parametrize(
    "fraction,expected",
    [
        (0.0, "tiny"),
        (0.0049, "tiny"),
        (0.005, "small"),
        (0.0199, "small"),
        (0.02, "medium"),
        (0.099, "medium"),
        (0.1, "large"),
    ],
)
def test_fixed_size_bucket_boundaries(fraction, expected):
    assert size_bucket(fraction) == expected


def test_source_ignore_does_not_filter_primary_matching():
    features = _identity_features(1, 2)
    torch.testing.assert_close(correspondence(features, features), torch.tensor([[0, 1]]))
    mask = np.array([[255, 1]], dtype=np.uint8)
    result = evaluate_pair(features, features, mask, np.array([[1, 0]], dtype=np.uint8))
    assert result["J"] == 0.0


def test_bad_features_and_masks_fail_explicitly():
    features = _identity_features()
    with pytest.raises(ValueError, match="finite"):
        correspondence(features, features * float("nan"))
    with pytest.raises(ValueError, match="integer mask"):
        evaluate_pair(features, features, np.ones((4, 4)), np.ones((4, 4)))
