"""Mechanism diagnostics for measurement completeness and predictor independence."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any

from torch import Tensor, nn

from mcss.measurements import FixedMeasurementRenderer


@dataclass(frozen=True)
class NoBypassReport:
    passed: bool
    renderer_parameter_count: int
    forbidden_forward_arguments: tuple[str, ...]
    reasons: tuple[str, ...] = ()


def no_bypass_audit(model: nn.Module) -> NoBypassReport:
    forbidden = {
        "target_rgb",
        "target_depth",
        "target_normal",
        "target_point",
        "target_visibility",
        "target_uncertainty",
    }
    try:
        parameters = list(inspect.signature(model.forward).parameters)
    except (TypeError, ValueError):
        parameters = []
    forbidden_arguments = tuple(sorted(forbidden.intersection(parameters)))
    renderers = [
        module for module in model.modules() if isinstance(module, FixedMeasurementRenderer)
    ]
    parameter_count = sum(
        parameter.numel() for renderer in renderers for parameter in renderer.parameters()
    )
    reasons: list[str] = []
    if not renderers:
        reasons.append("no FixedMeasurementRenderer found")
    if parameter_count:
        reasons.append("fixed renderer contains trainable parameters")
    if forbidden_arguments:
        reasons.append("forward signature exposes target supervision")
    return NoBypassReport(not reasons, parameter_count, forbidden_arguments, tuple(reasons))


def measurement_generalization_matrix(records: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    matrix: dict[str, dict[str, float]] = {}
    for record in records:
        trained = str(record["trained"])
        evaluated = str(record["evaluated"])
        matrix.setdefault(trained, {})[evaluated] = float(record["score"])
    return matrix


def heldout_measurement_gap(
    matrix: dict[str, dict[str, float]],
    *,
    trained_measurements: str,
    seen_name: str,
    heldout_name: str,
) -> float:
    row = matrix[trained_measurements]
    return float(row[seen_name] - row[heldout_name])


def query_equivariance_error(
    reference: dict[str, Tensor], permuted: dict[str, Tensor], permutation: Tensor
) -> dict[str, float]:
    if permutation.ndim != 1:
        raise ValueError("permutation must be one-dimensional")
    errors: dict[str, float] = {}
    for name, value in reference.items():
        if name not in permuted:
            continue
        expected = value.index_select(1, permutation.to(value.device))
        errors[name] = float((expected - permuted[name]).abs().mean().item())
    return errors


def cross_context_consistency(
    first: dict[str, Tensor], second: dict[str, Tensor]
) -> dict[str, float]:
    """Compare predictions from two context subsets on identical target queries."""

    return {
        name: float((value - second[name]).abs().mean().item())
        for name, value in first.items()
        if name in second
    }


def intervention_specificity(
    baseline: dict[str, Tensor],
    density_intervention: dict[str, Tensor],
    color_intervention: dict[str, Tensor],
) -> dict[str, float]:
    density_to_depth = _effect(baseline, density_intervention, "depth")
    color_to_rgb = _effect(baseline, color_intervention, "rgb")
    density_to_rgb = _effect(baseline, density_intervention, "rgb")
    color_to_depth = _effect(baseline, color_intervention, "depth")
    denominator = density_to_depth + color_to_rgb + density_to_rgb + color_to_depth
    return {
        "density_to_depth": _ratio(density_to_depth, density_to_depth),
        "color_to_rgb": _ratio(color_to_rgb, color_to_rgb),
        "cross_talk": _ratio(density_to_rgb + color_to_depth, denominator),
    }


def _effect(baseline: dict[str, Tensor], intervention: dict[str, Tensor], name: str) -> float:
    if name not in baseline or name not in intervention:
        return 0.0
    return float((intervention[name] - baseline[name]).abs().mean().item())


def _ratio(numerator: float, denominator: float) -> float:
    return 0.0 if denominator <= 1e-12 else numerator / denominator
