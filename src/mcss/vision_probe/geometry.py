"""Homography correspondences and discrete patch matching metrics."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor
from torch.nn import functional as F


def homography_correspondence(
    homo: Tensor,
    src_original_size: tuple[int, int],
    dst_original_size: tuple[int, int],
    grid_size: int = 16,
    image_size: int | tuple[int, int] = 224,
) -> tuple[Tensor, Tensor]:
    """Map source token centers through an original-image homography.

    Image sizes are ``(width, height)``.  Token centers use cell-center
    coordinates over the resized image domain, so an identity homography maps
    each source token to its integer target-grid coordinate.
    """

    matrix = _as_homography(homo)
    src_width, src_height = _positive_size(src_original_size, "src_original_size")
    dst_width, dst_height = _positive_size(dst_original_size, "dst_original_size")
    if isinstance(grid_size, bool) or not isinstance(grid_size, int) or grid_size <= 0:
        raise ValueError("grid_size must be a positive integer")
    image_width, image_height = _image_size(image_size)

    coordinate_dtype = matrix.dtype if matrix.is_floating_point() else torch.float32
    matrix = matrix.to(dtype=coordinate_dtype)
    device = matrix.device
    y, x = torch.meshgrid(
        (torch.arange(grid_size, device=device, dtype=coordinate_dtype) + 0.5)
        * (image_height / grid_size),
        (torch.arange(grid_size, device=device, dtype=coordinate_dtype) + 0.5)
        * (image_width / grid_size),
        indexing="ij",
    )
    source_resized = torch.stack((x, y), dim=-1).reshape(-1, 2)
    source_original = source_resized * torch.tensor(
        (src_width / image_width, src_height / image_height),
        device=device,
        dtype=coordinate_dtype,
    )
    homogeneous = torch.cat(
        (
            source_original,
            torch.ones((source_original.shape[0], 1), device=device, dtype=coordinate_dtype),
        ),
        dim=-1,
    )
    mapped_homogeneous = homogeneous @ matrix.T
    denominator = mapped_homogeneous[:, 2]
    if not torch.isfinite(denominator).all() or denominator.abs().le(
        torch.finfo(coordinate_dtype).eps
    ).any():
        raise ValueError("homography produces a non-finite or zero homogeneous denominator")
    destination_original = mapped_homogeneous[:, :2] / denominator.unsqueeze(-1)
    destination_grid = destination_original / torch.tensor(
        (dst_width, dst_height), device=device, dtype=coordinate_dtype
    ) * grid_size - 0.5
    valid = (
        torch.isfinite(destination_grid).all(dim=-1)
        & (destination_grid[:, 0] >= 0)
        & (destination_grid[:, 0] <= grid_size - 1)
        & (destination_grid[:, 1] >= 0)
        & (destination_grid[:, 1] <= grid_size - 1)
    )
    source_indices = torch.arange(grid_size * grid_size, device=device, dtype=torch.long)
    return source_indices[valid], destination_grid[valid].to(dtype=torch.float32)


def matching_metrics(
    source_feat: Tensor,
    target_feat: Tensor,
    valid_src_indices: Tensor,
    gt_dst_xy: Tensor,
) -> dict[str, float | int]:
    """Evaluate discrete cosine nearest-neighbor matching on valid source tokens."""

    _validate_features(source_feat, "source_feat")
    _validate_features(target_feat, "target_feat")
    if source_feat.shape[-1] != target_feat.shape[-1]:
        raise ValueError("source_feat and target_feat must have the same feature dimension")
    if source_feat.device != target_feat.device:
        raise ValueError("source_feat and target_feat must be on the same device")
    if not isinstance(valid_src_indices, Tensor) or valid_src_indices.ndim != 1:
        raise ValueError("valid_src_indices must have shape [N]")
    if valid_src_indices.dtype != torch.long:
        raise ValueError("valid_src_indices must have dtype torch.long")
    if not isinstance(gt_dst_xy, Tensor) or gt_dst_xy.ndim != 2 or gt_dst_xy.shape[-1] != 2:
        raise ValueError("gt_dst_xy must have shape [N, 2]")
    if valid_src_indices.shape[0] != gt_dst_xy.shape[0]:
        raise ValueError("valid_src_indices and gt_dst_xy must contain the same number of points")
    if not torch.isfinite(gt_dst_xy).all():
        raise ValueError("gt_dst_xy must contain only finite values")
    if valid_src_indices.device != source_feat.device:
        valid_src_indices = valid_src_indices.to(source_feat.device)
    if gt_dst_xy.device != source_feat.device:
        gt_dst_xy = gt_dst_xy.to(source_feat.device)
    if valid_src_indices.numel() and (
        valid_src_indices.min() < 0
        or valid_src_indices.max() >= source_feat.shape[0] * source_feat.shape[1]
    ):
        raise ValueError("valid_src_indices contains an out-of-range source index")

    n_points = int(valid_src_indices.numel())
    if n_points == 0:
        return {"pck_1": 0.0, "pck_2": 0.0, "mean_error": float("nan"), "n_points": 0}

    source_flat = F.normalize(source_feat.reshape(-1, source_feat.shape[-1]), dim=-1)
    target_flat = F.normalize(target_feat.reshape(-1, target_feat.shape[-1]), dim=-1)
    scores = source_flat[valid_src_indices] @ target_flat.T
    predicted_flat_indices = scores.argmax(dim=1)
    target_width = target_feat.shape[1]
    predicted_xy = torch.stack(
        (
            predicted_flat_indices.remainder(target_width),
            torch.div(predicted_flat_indices, target_width, rounding_mode="floor"),
        ),
        dim=-1,
    ).to(dtype=gt_dst_xy.dtype)
    error = torch.linalg.vector_norm(predicted_xy - gt_dst_xy, dim=-1)
    return {
        "pck_1": float((error <= 1.0).float().mean().item()),
        "pck_2": float((error <= 2.0).float().mean().item()),
        "mean_error": float(error.mean().item()),
        "n_points": n_points,
    }


def mask_propagation_metrics(
    source_features: Tensor,
    target_features: Tensor,
    source_labels: Tensor,
    target_labels: Tensor,
) -> dict[str, float | int]:
    """Propagate source labels to a target token grid with cosine nearest neighbors.

    The source descriptor map and labels are treated as the frozen reference.
    Each target token selects one source token, and the selected source label is
    compared with the target label.  ``mean_iou`` is the mean over foreground
    object IDs present in either label map; ``foreground_iou`` is the binary
    foreground IoU.  Objects whose predicted/target union is empty are skipped.
    """

    _validate_features(source_features, "source_features")
    _validate_features(target_features, "target_features")
    if source_features.shape[-1] != target_features.shape[-1]:
        raise ValueError("source_features and target_features must have the same feature dimension")
    if source_features.device != target_features.device:
        raise ValueError("source_features and target_features must be on the same device")
    _validate_labels(source_labels, source_features.shape[:2], "source_labels")
    _validate_labels(target_labels, target_features.shape[:2], "target_labels")

    device = source_features.device
    source_labels = source_labels.to(device=device)
    target_labels = target_labels.to(device=device)
    source_flat = F.normalize(source_features.reshape(-1, source_features.shape[-1]), dim=-1)
    target_flat = F.normalize(target_features.reshape(-1, target_features.shape[-1]), dim=-1)
    source_flat_labels = source_labels.reshape(-1)
    source_valid = source_flat_labels != 255
    if not source_valid.any():
        raise ValueError("source_labels must contain at least one non-ignore token")
    nearest_source = (target_flat @ source_flat[source_valid].T).argmax(dim=1)
    propagated = source_flat_labels[source_valid][nearest_source].reshape(target_labels.shape)

    target_valid = target_labels != 255
    if target_valid.any():
        accuracy = float(
            (propagated[target_valid] == target_labels[target_valid]).float().mean().item()
        )
    else:
        accuracy = 0.0
    source_ids = source_labels[(source_labels != 0) & (source_labels != 255)]
    target_ids = target_labels[(target_labels != 0) & (target_labels != 255)]
    if source_ids.numel() or target_ids.numel():
        object_ids = torch.unique(torch.cat((source_ids, target_ids)))
    else:
        object_ids = torch.empty(0, dtype=source_labels.dtype, device=device)

    object_ious: list[Tensor] = []
    for object_id in object_ids:
        predicted_mask = (propagated == object_id) & target_valid
        target_mask = (target_labels == object_id) & target_valid
        union = predicted_mask | target_mask
        if not union.any():
            continue
        intersection = (predicted_mask & target_mask).sum().to(dtype=torch.float32)
        object_ious.append(intersection / union.sum().to(dtype=torch.float32))

    foreground_predicted = (propagated != 0) & target_valid
    foreground_target = (target_labels != 0) & target_valid
    foreground_union = foreground_predicted | foreground_target
    if foreground_union.any():
        foreground_iou = float(
            (foreground_predicted & foreground_target)
            .sum()
            .float()
            .div(foreground_union.sum())
            .item()
        )
    else:
        foreground_iou = 0.0
    mean_iou = float(torch.stack(object_ious).mean().item()) if object_ious else 0.0
    return {
        "mean_iou": mean_iou,
        "foreground_iou": foreground_iou,
        "n_objects": len(object_ious),
        "accuracy": accuracy,
    }


def _as_homography(value: Tensor) -> Tensor:
    if not isinstance(value, Tensor):
        try:
            value = torch.as_tensor(value)
        except (TypeError, ValueError) as error:
            raise TypeError("homo must be convertible to a tensor") from error
    if value.shape != (3, 3):
        raise ValueError("homo must have shape [3, 3]")
    if not value.is_floating_point():
        value = value.float()
    if not torch.isfinite(value).all():
        raise ValueError("homo must contain only finite values")
    if torch.linalg.matrix_rank(value) < 3:
        raise ValueError("homo must be invertible")
    return value


def _positive_size(value: Sequence[int], label: str) -> tuple[float, float]:
    if not isinstance(value, Sequence) or len(value) != 2:
        raise ValueError(f"{label} must be (width, height)")
    width, height = value
    if isinstance(width, bool) or isinstance(height, bool):
        raise ValueError(f"{label} values must be positive")
    try:
        width_float, height_float = float(width), float(height)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} values must be positive") from error
    if not torch.isfinite(torch.tensor((width_float, height_float))).all() or min(
        width_float, height_float
    ) <= 0:
        raise ValueError(f"{label} values must be positive")
    return width_float, height_float


def _image_size(value: int | tuple[int, int]) -> tuple[float, float]:
    if isinstance(value, bool):
        raise ValueError("image_size must be positive")
    if isinstance(value, int):
        if value <= 0:
            raise ValueError("image_size must be positive")
        return float(value), float(value)
    return _positive_size(value, "image_size")


def _validate_features(value: Tensor, label: str) -> None:
    if not isinstance(value, Tensor) or value.ndim != 3:
        raise ValueError(f"{label} must have shape [H, W, C]")
    if min(value.shape) <= 0:
        raise ValueError(f"{label} dimensions must be positive")
    if not value.is_floating_point():
        raise ValueError(f"{label} must use a floating-point dtype")
    if not torch.isfinite(value).all():
        raise ValueError(f"{label} must contain only finite values")


def _validate_labels(value: Tensor, expected_shape: tuple[int, int], label: str) -> None:
    if not isinstance(value, Tensor) or value.ndim != 2 or tuple(value.shape) != expected_shape:
        raise ValueError(f"{label} must have shape {expected_shape}")
    if value.dtype == torch.bool or value.is_floating_point() or value.is_complex():
        raise ValueError(f"{label} must use an integer dtype")
    if value.numel() and value.min() < 0:
        raise ValueError(f"{label} must contain non-negative object IDs")
