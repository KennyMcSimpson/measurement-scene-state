"""Pixel-space mask transfer for the opportunity study, not official DAVIS J&F.

J is primary. F uses binary erosion contours, Euclidean matching within
ceil(0.008 * image diagonal), and image-exterior background. Both-empty masks
score J=F=1; exactly one empty scores zero. Pairs with no foreground IDs score
zero and contain no object rows. Void (255) target pixels are excluded from
J and boundary evaluation; contours adjacent to void are excluded. Object
means weight source/target-union IDs equally, including objects lost by token
sampling. Size fractions use the full original target image area.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
from PIL import Image
from scipy import ndimage
from torch import Tensor
from torch.nn import functional as F

from mcss.vision_probe.geometry import mask_propagation_metrics


def _features(value: Tensor, name: str) -> None:
    if not isinstance(value, Tensor) or value.ndim != 3 or min(value.shape) < 1:
        raise ValueError(f"{name} must have nonempty [H, W, C] shape")
    if not value.is_floating_point() or not torch.isfinite(value).all():
        raise ValueError(f"{name} must contain finite floating-point features")


@torch.no_grad()
def correspondence(source_features: Tensor, target_features: Tensor) -> Tensor:
    """Return target [H,W] nearest source flat indices; no masks are accepted.

    Ties choose the lowest source flat index. Zero descriptors are permitted
    and consequently tie. Matching always includes every source token.
    """
    _features(source_features, "source_features")
    _features(target_features, "target_features")
    if source_features.shape[-1] != target_features.shape[-1]:
        raise ValueError("source and target feature dimensions must match")
    if source_features.device != target_features.device:
        raise ValueError("source and target features must share a device")
    source = F.normalize(source_features.reshape(-1, source_features.shape[-1]).float(), dim=-1)
    target = F.normalize(target_features.reshape(-1, target_features.shape[-1]).float(), dim=-1)
    return (target @ source.T).argmax(dim=-1).reshape(target_features.shape[:2])


def _mask(value: np.ndarray, name: str) -> np.ndarray:
    value = np.asarray(value)
    if value.ndim != 2 or min(value.shape) < 1 or value.dtype.kind not in "biu":
        raise ValueError(f"{name} must be a nonempty two-dimensional integer mask")
    if value.min() < 0 or value.max() > np.iinfo(np.int32).max:
        raise ValueError(f"{name} contains unsupported object IDs")
    return value


def _ids(mask: np.ndarray) -> list[int]:
    return [int(value) for value in np.unique(mask) if value not in (0, 255)]


def _nearest_labels(mask: np.ndarray, shape: tuple[int, int]) -> Tensor:
    """Use PIL nearest exactly as the original probe's _mask_grid helper."""
    resized = Image.fromarray(mask.astype(np.int32)).resize(
        (shape[1], shape[0]), Image.Resampling.NEAREST
    )
    return torch.from_numpy(np.array(resized, dtype=np.int64))


def size_bucket(fraction: float) -> str:
    """Frozen target-pixel-fraction thresholds; no data-dependent quantiles."""
    if fraction < 0.005:
        return "tiny"
    if fraction < 0.02:
        return "small"
    if fraction < 0.10:
        return "medium"
    return "large"


def transfer_probabilities(
    source_mask: np.ndarray,
    source_grid_shape: tuple[int, int],
    nearest_source: Tensor,
    target_size: tuple[int, int],
) -> dict[int, np.ndarray]:
    """Propagate each source object's occupancy without target labels.

    Area downsampling yields source occupancy, correspondence gathers target
    occupancy, and bilinear interpolation with align_corners=False restores
    the original target spatial size. This is a per-object binary readout;
    probabilities for different objects are not made mutually exclusive.
    """
    source_mask = _mask(source_mask, "source_mask")
    if len(source_grid_shape) != 2 or min(source_grid_shape) <= 0:
        raise ValueError("source_grid_shape must contain positive H,W")
    if len(target_size) != 2 or min(target_size) <= 0:
        raise ValueError("target_size must contain positive H,W")
    if nearest_source.ndim != 2 or nearest_source.numel() == 0:
        raise ValueError("nearest_source must be a nonempty target grid")
    if nearest_source.dtype != torch.long:
        raise ValueError("nearest_source must contain integer torch.long indices")
    indices = nearest_source.detach().cpu()
    if indices.min() < 0 or indices.max() >= math.prod(source_grid_shape):
        raise ValueError("nearest_source contains an out-of-range source index")
    result = {}
    for object_id in _ids(source_mask):
        binary = torch.from_numpy((source_mask == object_id).astype(np.float32))[None, None]
        occupancy = F.interpolate(binary, size=source_grid_shape, mode="area").reshape(-1)
        target = occupancy[indices][None, None]
        probability = F.interpolate(target, size=target_size, mode="bilinear", align_corners=False)
        result[object_id] = probability[0, 0].numpy()
    return result


def binary_scores(
    prediction: np.ndarray, target: np.ndarray, valid: np.ndarray | None = None
) -> tuple[float, float]:
    """Compute region J and approximate boundary F on original-size binaries."""
    prediction = np.asarray(prediction, dtype=bool)
    target = np.asarray(target, dtype=bool)
    if prediction.ndim != 2 or prediction.shape != target.shape or not prediction.size:
        raise ValueError("prediction and target must be aligned nonempty 2D masks")
    if valid is None:
        valid = np.ones_like(target)
    else:
        valid = np.asarray(valid, dtype=bool)
        if valid.shape != target.shape:
            raise ValueError("valid must match the target shape")
    prediction = prediction & valid
    target = target & valid
    union = np.count_nonzero(prediction | target)
    jaccard = np.count_nonzero(prediction & target) / union if union else 1.0
    if not prediction.any() or not target.any():
        return float(jaccard), float(not prediction.any() and not target.any())
    structure = np.ones((3, 3), dtype=bool)
    # A void edge is an annotation boundary, not an object boundary. Preserve
    # image-edge contours by treating exterior pixels as valid for this mask.
    safe = ndimage.binary_erosion(valid, structure=structure, border_value=1)
    predicted_boundary = (
        prediction & ~ndimage.binary_erosion(prediction, structure=structure, border_value=0)
    ) & safe
    target_boundary = (
        target & ~ndimage.binary_erosion(target, structure=structure, border_value=0)
    ) & safe
    if not predicted_boundary.any() or not target_boundary.any():
        return float(jaccard), float(not predicted_boundary.any() and not target_boundary.any())
    tolerance = math.ceil(0.008 * math.hypot(*target.shape))
    to_target = ndimage.distance_transform_edt(~target_boundary)
    to_prediction = ndimage.distance_transform_edt(~predicted_boundary)
    precision = float(np.mean(to_target[predicted_boundary] <= tolerance))
    recall = float(np.mean(to_prediction[target_boundary] <= tolerance))
    boundary_f = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return float(jaccard), boundary_f


@torch.no_grad()
def evaluate_pair(
    source_features: Tensor,
    target_features: Tensor,
    source_mask: np.ndarray,
    target_mask: np.ndarray,
) -> dict[str, Any]:
    """Evaluate one candidate, keeping target labels outside the matching path.

    token_iou intentionally retains old PIL-nearest/ignore-label semantics;
    it can omit tiny objects erased by nearest sampling, unlike pixel J/F.
    Foreground token counts also use nearest label sampling at 16 and 32 grids.
    """
    nearest = correspondence(source_features, target_features)
    source_mask = _mask(source_mask, "source_mask")
    target_mask = _mask(target_mask, "target_mask")
    probabilities = transfer_probabilities(
        source_mask, tuple(source_features.shape[:2]), nearest, target_mask.shape
    )
    target_grids = {size: _nearest_labels(target_mask, (size, size)).numpy() for size in (16, 32)}
    valid = target_mask != 255
    objects = []
    for object_id in sorted(set(_ids(source_mask)) | set(_ids(target_mask))):
        probability = probabilities.get(object_id)
        predicted = (
            np.zeros_like(target_mask, dtype=bool) if probability is None else probability >= 0.5
        )
        target = target_mask == object_id
        jaccard, boundary_f = binary_scores(predicted, target, valid)
        pixel_count = int(np.count_nonzero(target))
        fraction = pixel_count / target_mask.size
        objects.append(
            {
                "object_id": object_id,
                "J": jaccard,
                "F": boundary_f,
                "JF": (jaccard + boundary_f) / 2,
                "pixel_count": pixel_count,
                "pixel_fraction": fraction,
                "size_bucket": size_bucket(fraction),
                "token_count_224": int(np.count_nonzero(target_grids[16] == object_id)),
                "token_count_448": int(np.count_nonzero(target_grids[32] == object_id)),
            }
        )
    source_labels = _nearest_labels(source_mask, tuple(source_features.shape[:2]))
    target_labels = _nearest_labels(target_mask, tuple(target_features.shape[:2]))
    legacy = (
        mask_propagation_metrics(source_features, target_features, source_labels, target_labels)
        if (source_labels != 255).any()
        else {"mean_iou": 0.0}
    )
    foreground = (target_mask != 0) & valid
    return {
        **{
            key: float(np.mean([row[key] for row in objects])) if objects else 0.0
            for key in ("J", "F", "JF")
        },
        "token_iou": float(legacy["mean_iou"]),
        "foreground_pixel_count": int(np.count_nonzero(foreground)),
        "foreground_pixel_fraction": float(np.mean(foreground)),
        "foreground_token_counts": {
            str(resolution): int(np.count_nonzero((grid != 0) & (grid != 255)))
            for resolution, grid in ((224, target_grids[16]), (448, target_grids[32]))
        },
        "objects": objects,
    }
