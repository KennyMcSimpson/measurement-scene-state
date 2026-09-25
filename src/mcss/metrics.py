"""Standard RGB, geometry, point-map, and visibility metrics."""

from __future__ import annotations

import math
from collections.abc import Mapping

import torch
import torch.nn.functional as functional
from torch import Tensor


def compute_metrics(
    predictions: Mapping[str, Tensor],
    targets: Mapping[str, Tensor],
    *,
    bounds: Tensor | None = None,
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    support = targets.get("support")
    if support is not None:
        valid_support = torch.isfinite(support) & (support > 0.5)
        metrics.update(
            {
                "support/coverage": float(valid_support.float().mean().item()),
                "support/valid_pixels": float(valid_support.sum().item()),
                "support/total_pixels": float(valid_support.numel()),
            }
        )
    if "rgb" in predictions and "rgb" in targets:
        metrics.update(_rgb_metrics(predictions["rgb"], targets["rgb"], support))
    if "depth" in predictions and "depth" in targets:
        metrics.update(_depth_metrics(predictions["depth"], targets["depth"], support))
    if "normal" in predictions and "normal" in targets:
        metrics.update(_normal_metrics(predictions["normal"], targets["normal"], support))
    if "point" in predictions and "point" in targets:
        metrics.update(
            _point_metrics(
                predictions["point"],
                targets["point"],
                targets.get("visibility"),
                support,
                bounds,
            )
        )
    if "visibility" in predictions and "visibility" in targets:
        metrics.update(
            _visibility_metrics(predictions["visibility"], targets["visibility"], support)
        )
    return metrics


def _rgb_metrics(prediction: Tensor, target: Tensor, support: Tensor | None) -> dict[str, float]:
    valid = torch.isfinite(target).all(dim=-3, keepdim=True)
    valid = _apply_support(valid, support)
    valid_values = valid.expand_as(target)
    valid_pixels = float(valid.sum().item())
    if not valid.any():
        return {
            "rgb/psnr": float("nan"),
            "rgb/ssim": float("nan"),
            "rgb/mse": float("nan"),
            "rgb/valid_pixels": 0.0,
        }
    prediction_values = prediction[valid_values]
    target_values = target[valid_values]
    mse = (prediction_values - target_values).square().mean().item()
    psnr = float("inf") if mse == 0.0 else -10.0 * math.log10(max(mse, 1e-12))
    ssim = _masked_windowed_ssim(prediction, target, valid)
    return {
        "rgb/psnr": psnr,
        "rgb/ssim": float(ssim),
        "rgb/mse": mse,
        "rgb/valid_pixels": valid_pixels,
    }


def _depth_metrics(prediction: Tensor, target: Tensor, support: Tensor | None) -> dict[str, float]:
    valid = torch.isfinite(target) & (target > 0)
    valid = _apply_support(valid, support)
    if not valid.any():
        return {
            "depth/abs_rel": float("nan"),
            "depth/rmse": float("nan"),
            "depth/delta1": float("nan"),
            "depth/valid_pixels": 0.0,
        }
    error = (prediction - target).abs()
    abs_rel = _masked_mean(error / target.abs().clamp_min(1e-8), valid)
    rmse = _masked_mean((prediction - target).square(), valid).sqrt()
    ratio = torch.maximum(
        prediction.clamp_min(1e-8) / target.clamp_min(1e-8),
        target.clamp_min(1e-8) / prediction.clamp_min(1e-8),
    )
    delta1 = _masked_mean((ratio < 1.25).to(target.dtype), valid)
    return {
        "depth/abs_rel": float(abs_rel.item()),
        "depth/rmse": float(rmse.item()),
        "depth/delta1": float(delta1.item()),
        "depth/valid_pixels": float(valid.sum().item()),
    }


def _normal_metrics(prediction: Tensor, target: Tensor, support: Tensor | None) -> dict[str, float]:
    prediction = functional.normalize(prediction, dim=-3, eps=1e-8)
    target = functional.normalize(target, dim=-3, eps=1e-8)
    valid = torch.isfinite(target).all(dim=-3, keepdim=True) & (
        target.square().sum(dim=-3, keepdim=True) > 1e-8
    )
    valid = _apply_support(valid, support)
    cosine = (prediction * target).sum(dim=-3, keepdim=True).clamp(-1.0, 1.0)
    angle = torch.rad2deg(torch.acos(cosine))
    return {
        "normal/mean_angle": float(_masked_mean(angle, valid).item()),
        "normal/acc_5": float(_masked_mean((angle < 5.0).to(target.dtype), valid).item()),
        "normal/acc_11.25": float(_masked_mean((angle < 11.25).to(target.dtype), valid).item()),
        "normal/acc_22.5": float(_masked_mean((angle < 22.5).to(target.dtype), valid).item()),
        "normal/valid_pixels": float(valid.sum().item()),
    }


def _point_metrics(
    prediction: Tensor,
    target: Tensor,
    visibility: Tensor | None,
    support: Tensor | None,
    bounds: Tensor | None,
) -> dict[str, float]:
    pred_points = prediction.permute(0, 1, 3, 4, 2).reshape(-1, 3)
    target_points = target.permute(0, 1, 3, 4, 2).reshape(-1, 3)
    valid = torch.isfinite(target_points).all(dim=-1)
    if visibility is not None:
        valid = valid & (visibility.permute(0, 1, 3, 4, 2).reshape(-1) > 0.5)
    if support is not None:
        valid = valid & (
            torch.isfinite(support).permute(0, 1, 3, 4, 2).reshape(-1)
            & (support.permute(0, 1, 3, 4, 2).reshape(-1) > 0.5)
        )
    point_error = torch.linalg.vector_norm(pred_points - target_points, dim=-1)
    valid_errors = point_error[valid]
    pred_points = pred_points[valid]
    target_points = target_points[valid]
    if not len(pred_points) or not len(target_points):
        return {
            "point/chamfer": float("nan"),
            "point/chamfer_m2": float("nan"),
            "point/mae_m": float("nan"),
            "point/rmse_m": float("nan"),
            "point/nrmse_support_diag": float("nan"),
            "point/fscore_0.05": float("nan"),
            "point/fscore_0.25": float("nan"),
            "point/fscore_0.50": float("nan"),
            "point/valid_points": 0.0,
        }
    mae = valid_errors.mean()
    rmse = valid_errors.square().mean().sqrt()
    normalized_rmse = float("nan")
    if bounds is not None:
        diagonal = torch.linalg.vector_norm(bounds[:, 1] - bounds[:, 0], dim=-1).mean()
        normalized_rmse = float((rmse / diagonal.clamp_min(1e-8)).item())
    max_points = 2048
    pred_points = _uniform_spatial_subsample(pred_points, max_points)
    target_points = _uniform_spatial_subsample(target_points, max_points)
    distances = torch.cdist(pred_points, target_points)
    pred_to_target = distances.min(dim=1).values
    target_to_pred = distances.min(dim=0).values
    fscores = {}
    for threshold in (0.05, 0.25, 0.50):
        precision = (pred_to_target < threshold).float().mean()
        recall = (target_to_pred < threshold).float().mean()
        fscore = 2 * precision * recall / (precision + recall).clamp_min(1e-8)
        fscores[f"point/fscore_{threshold:.2f}"] = float(fscore.item())
    chamfer = pred_to_target.square().mean() + target_to_pred.square().mean()
    return {
        "point/chamfer": float(chamfer.item()),
        "point/chamfer_m2": float(chamfer.item()),
        "point/mae_m": float(mae.item()),
        "point/rmse_m": float(rmse.item()),
        "point/nrmse_support_diag": normalized_rmse,
        "point/valid_points": float(valid.sum().item()),
        **fscores,
    }


def _visibility_metrics(
    prediction: Tensor, target: Tensor, support: Tensor | None
) -> dict[str, float]:
    pred = prediction > 0.5
    truth = target > 0.5
    valid = torch.isfinite(target)
    valid = _apply_support(valid, support)
    true_positive = (pred & truth & valid).sum().item()
    false_positive = (pred & ~truth & valid).sum().item()
    false_negative = (~pred & truth & valid).sum().item()
    union = true_positive + false_positive + false_negative
    iou = 1.0 if union == 0 else true_positive / union
    f1_denominator = 2 * true_positive + false_positive + false_negative
    f1 = 1.0 if f1_denominator == 0 else (2 * true_positive) / f1_denominator
    return {
        "visibility/iou": float(iou),
        "visibility/f1": float(f1),
        "visibility/valid_pixels": float(valid.sum().item()),
    }


def _masked_mean(value: Tensor, mask: Tensor) -> Tensor:
    mask = mask.to(value.dtype)
    while mask.ndim < value.ndim:
        mask = mask.unsqueeze(-3)
    return (value * mask).sum() / mask.sum().clamp_min(1.0)


def _masked_windowed_ssim(prediction: Tensor, target: Tensor, valid: Tensor) -> float:
    prediction = prediction.float()
    target = target.float()
    batch_views = math.prod(prediction.shape[:-3])
    channels, height, width = prediction.shape[-3:]
    prediction = prediction.reshape(batch_views, channels, height, width)
    target = target.reshape(batch_views, channels, height, width)
    mask = valid.reshape(batch_views, 1, height, width).expand(-1, channels, -1, -1)
    mask_values = mask.to(prediction.dtype)
    kernel_size = 11
    coordinates = torch.arange(kernel_size, device=prediction.device, dtype=prediction.dtype)
    coordinates = coordinates - (kernel_size - 1) / 2.0
    gaussian = torch.exp(-(coordinates.square()) / (2.0 * 1.5**2))
    gaussian = gaussian / gaussian.sum()
    kernel = (gaussian[:, None] * gaussian[None, :]).reshape(1, 1, kernel_size, kernel_size)

    def smooth(values: Tensor) -> Tensor:
        filtered = functional.conv2d(
            values.reshape(-1, 1, height, width),
            kernel,
            padding=kernel_size // 2,
        )
        return filtered.reshape(batch_views, channels, height, width)

    normalizer = smooth(mask_values).clamp_min(1e-8)
    prediction_values = torch.where(mask, prediction, torch.zeros_like(prediction))
    target_values = torch.where(mask, target, torch.zeros_like(target))
    mean_prediction = smooth(prediction_values) / normalizer
    mean_target = smooth(target_values) / normalizer
    variance_prediction = (
        smooth(prediction_values.square()) / normalizer - mean_prediction.square()
    ).clamp_min(0.0)
    variance_target = (
        smooth(target_values.square()) / normalizer - mean_target.square()
    ).clamp_min(0.0)
    covariance = (
        smooth(prediction_values * target_values) / normalizer - mean_prediction * mean_target
    )
    c1, c2 = 0.01**2, 0.03**2
    score = ((2.0 * mean_prediction * mean_target + c1) * (2.0 * covariance + c2)) / (
        (mean_prediction.square() + mean_target.square() + c1)
        * (variance_prediction + variance_target + c2)
    )
    return float(score[mask].mean().clamp(-1.0, 1.0).item())


def _uniform_spatial_subsample(points: Tensor, max_points: int) -> Tensor:
    if len(points) <= max_points:
        return points
    bins = torch.arange(max_points, device=points.device, dtype=torch.long)
    indices = ((2 * bins + 1) * len(points)) // (2 * max_points)
    return points[indices]


def _apply_support(mask: Tensor, support: Tensor | None) -> Tensor:
    if support is None:
        return mask
    return mask & torch.isfinite(support) & (support > 0.5)
