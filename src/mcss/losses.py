"""Masked modality losses used by the training engine."""

from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn.functional as functional
from torch import Tensor, nn


class MeasurementLoss(nn.Module):
    """Combine only the requested, available measurements with explicit validity masks."""

    def __init__(self, weights: Mapping[str, float], *, epsilon: float = 1e-3) -> None:
        super().__init__()
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")
        self.weights = {
            name: float(weight) for name, weight in weights.items() if float(weight) > 0
        }
        self.epsilon = epsilon

    def forward(
        self, predictions: Mapping[str, Tensor], targets: Mapping[str, Tensor]
    ) -> tuple[Tensor, dict[str, Tensor]]:
        if not predictions:
            raise ValueError("predictions cannot be empty")
        anchor = next(iter(predictions.values()))
        terms: dict[str, Tensor] = {}
        for name, weight in self.weights.items():
            if name not in predictions or name not in targets:
                continue
            terms[name] = _modality_loss(name, predictions, targets, self.epsilon)
            terms[name] = terms[name] * weight
        if not terms:
            return anchor.sum() * 0.0, {}
        total = torch.stack(list(terms.values())).sum()
        return total, terms


def _modality_loss(
    name: str, predictions: Mapping[str, Tensor], targets: Mapping[str, Tensor], epsilon: float
) -> Tensor:
    prediction = predictions[name]
    target = targets[name]
    if name == "rgb":
        valid = torch.isfinite(target).all(dim=-3, keepdim=True)
        valid = _apply_support(valid, targets)
        error = torch.sqrt((prediction - target).square() + epsilon**2)
        return _masked_mean(error, valid)
    if name == "depth":
        valid = torch.isfinite(target) & (target > 0)
        valid = _apply_support(valid, targets)
        relative = (prediction - target).abs() / target.abs().clamp_min(epsilon)
        return _masked_mean(relative, valid)
    if name == "normal":
        prediction = functional.normalize(prediction, dim=-3, eps=epsilon)
        target = functional.normalize(target, dim=-3, eps=epsilon)
        valid = torch.isfinite(target).all(dim=-3, keepdim=True) & (
            target.square().sum(dim=-3, keepdim=True) > epsilon
        )
        valid = _apply_support(valid, targets)
        cosine = (prediction * target).sum(dim=-3, keepdim=True).clamp(-1.0, 1.0)
        return _masked_mean(1.0 - cosine, valid)
    if name == "point":
        valid = torch.isfinite(target).all(dim=-3, keepdim=True)
        if "visibility" in targets:
            valid = valid & (targets["visibility"] > 0.5)
        valid = _apply_support(valid, targets)
        return _masked_mean((prediction - target).abs().mean(dim=-3, keepdim=True), valid)
    if name == "visibility":
        valid = torch.isfinite(target)
        valid = _apply_support(valid, targets)
        with torch.autocast(device_type=prediction.device.type, enabled=False):
            bce = functional.binary_cross_entropy(
                prediction.float().clamp(epsilon, 1 - epsilon),
                target.float(),
                reduction="none",
            )
        return _masked_mean(bce, valid)
    if name == "uncertainty":
        valid = _apply_support(torch.isfinite(target), targets)
        return _masked_mean(torch.sqrt((prediction - target).square() + epsilon**2), valid)
    raise ValueError(f"unsupported modality loss: {name}")


def _masked_mean(value: Tensor, mask: Tensor) -> Tensor:
    mask = torch.broadcast_to(mask, value.shape).to(value.dtype)
    while mask.ndim < value.ndim:
        mask = mask.unsqueeze(-3)
    denominator = mask.sum().clamp_min(1.0)
    return (value * mask).sum() / denominator


def _apply_support(mask: Tensor, targets: Mapping[str, Tensor]) -> Tensor:
    support = targets.get("support")
    if support is None:
        return mask
    return mask & torch.isfinite(support) & (support > 0.5)
