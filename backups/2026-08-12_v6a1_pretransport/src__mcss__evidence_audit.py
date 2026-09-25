"""Post-forward calibration audits for deterministic scene-state evidence."""

from __future__ import annotations

import torch
from torch import Tensor

from mcss.geometry import generate_rays, intersect_aabb
from mcss.measurements import _sample_volume
from mcss.types import Cameras, SceneState


def classify_depth_samples(
    distances: Tensor,
    target_depth: Tensor,
    *,
    surface_band: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Classify fixed ray samples around a target-visible metric ray depth."""

    if distances.ndim != 3:
        raise ValueError("distances must have shape [B, R, S]")
    if target_depth.shape != distances.shape[:2] + (1,):
        raise ValueError("target_depth must have shape [B, R, 1]")
    if surface_band.shape != (distances.shape[0], 1, 1):
        raise ValueError("surface_band must have shape [B, 1, 1]")
    if not torch.isfinite(surface_band).all() or (surface_band <= 0).any():
        raise ValueError("surface_band must be finite and positive")

    lower = target_depth - surface_band
    upper = target_depth + surface_band
    free = distances < lower
    surface = (distances >= lower) & (distances <= upper)
    behind = distances > upper
    return free, surface, behind


def binary_ranking_metrics(
    scores: Tensor,
    labels: Tensor,
    *,
    reliability_bins: int = 10,
) -> dict[str, object]:
    """Return exact rank metrics and calibration statistics for binary labels."""

    scores = scores.detach().float().reshape(-1).cpu()
    labels = labels.detach().bool().reshape(-1).cpu()
    if scores.shape != labels.shape:
        raise ValueError("scores and labels must contain the same number of values")
    if scores.numel() == 0:
        return _empty_binary_metrics(reliability_bins)
    scores = _clamp_probability_roundoff(scores, name="scores")
    if reliability_bins < 2:
        raise ValueError("reliability_bins must be at least two")

    positives = int(labels.sum())
    negatives = int((~labels).sum())
    auroc = _binary_auroc(scores, labels) if positives and negatives else None
    auprc = _binary_average_precision(scores, labels) if positives else None
    numeric_labels = labels.float()
    brier = float((scores - numeric_labels).square().mean())

    reliability: list[dict[str, float | int | None]] = []
    ece = 0.0
    for index in range(reliability_bins):
        lower = index / reliability_bins
        upper = (index + 1) / reliability_bins
        if index == reliability_bins - 1:
            selected = (scores >= lower) & (scores <= upper)
        else:
            selected = (scores >= lower) & (scores < upper)
        count = int(selected.sum())
        mean_confidence = float(scores[selected].mean()) if count else None
        positive_rate = float(numeric_labels[selected].mean()) if count else None
        if count and mean_confidence is not None and positive_rate is not None:
            ece += count / scores.numel() * abs(mean_confidence - positive_rate)
        reliability.append(
            {
                "lower": lower,
                "upper": upper,
                "count": count,
                "mean_confidence": mean_confidence,
                "positive_rate": positive_rate,
            }
        )
    return {
        "count": scores.numel(),
        "positive_count": positives,
        "negative_count": negatives,
        "auroc": auroc,
        "auprc": auprc,
        "brier": brier,
        "ece": ece,
        "reliability": reliability,
    }


def error_by_risk_deciles(
    risk: Tensor,
    absolute_error: Tensor,
    absolute_relative_error: Tensor,
    *,
    bins: int = 10,
) -> dict[str, object]:
    """Summarize whether target-independent ray risk orders held-out depth error."""

    risk = risk.detach().float().reshape(-1).cpu()
    absolute_error = absolute_error.detach().float().reshape(-1).cpu()
    absolute_relative_error = absolute_relative_error.detach().float().reshape(-1).cpu()
    if risk.shape != absolute_error.shape or risk.shape != absolute_relative_error.shape:
        raise ValueError("risk and error tensors must contain the same number of values")
    if bins < 2:
        raise ValueError("bins must be at least two")
    if not all(
        torch.isfinite(value).all()
        for value in (risk, absolute_error, absolute_relative_error)
    ):
        raise ValueError("risk and error tensors must be finite")
    if ((risk < 0.0) | (risk > 1.0)).any():
        raise ValueError("risk must be within [0, 1]")
    if (absolute_error < 0.0).any() or (absolute_relative_error < 0.0).any():
        raise ValueError("errors must be non-negative")

    order = torch.argsort(risk, stable=True)
    ordered_risk = risk[order]
    ordered_absolute = absolute_error[order]
    ordered_relative = absolute_relative_error[order]
    deciles: list[dict[str, float | int | None]] = []
    for index in range(bins):
        start = risk.numel() * index // bins
        stop = risk.numel() * (index + 1) // bins
        count = stop - start
        deciles.append(
            {
                "index": index,
                "count": count,
                "mean_risk": float(ordered_risk[start:stop].mean()) if count else None,
                "mean_absolute_error_m": (
                    float(ordered_absolute[start:stop].mean()) if count else None
                ),
                "mean_absolute_relative_error": (
                    float(ordered_relative[start:stop].mean()) if count else None
                ),
            }
        )
    absolute_means = [
        float(item["mean_absolute_error_m"])
        for item in deciles
        if item["mean_absolute_error_m"] is not None
    ]
    relative_means = [
        float(item["mean_absolute_relative_error"])
        for item in deciles
        if item["mean_absolute_relative_error"] is not None
    ]
    return {
        "count": risk.numel(),
        "risk_semantics": "one minus maximum fixed-ray localization support",
        "deciles": deciles,
        "absolute_error_monotonic_violations": _monotonic_violations(absolute_means),
        "absolute_relative_error_monotonic_violations": _monotonic_violations(
            relative_means
        ),
        "absolute_error_spearman": _ordered_spearman(absolute_means),
        "absolute_relative_error_spearman": _ordered_spearman(relative_means),
    }


@torch.no_grad()
def audit_evidence(
    state: SceneState,
    cameras: Cameras,
    target_depth: Tensor,
    target_visibility: Tensor | None = None,
    *,
    predicted_depth: Tensor | None = None,
    n_samples: int = 256,
) -> dict[str, object]:
    """Audit V5 correspondence confidence against hidden target-visible ray depth."""

    if state.dual_evidence is not None:
        confidence_volume = state.dual_evidence.surface_localization_support
        evidence_field = "dual_evidence.surface_localization_support"
    elif state.evidence is not None:
        confidence_volume = state.evidence.confidence
        evidence_field = "evidence.confidence"
    else:
        raise ValueError("audit_evidence requires StateEvidence or DualEvidence")
    if n_samples < 2:
        raise ValueError("n_samples must be at least two")
    if len(cameras.leading_shape) != 2:
        raise ValueError("cameras must have shape [B, V, ...]")
    batch_size, view_count = cameras.leading_shape
    height, width = cameras.image_size
    expected = (batch_size, view_count, 1, height, width)
    if target_depth.shape != expected:
        raise ValueError(f"target_depth must have shape {expected}")
    if target_visibility is not None and target_visibility.shape != expected:
        raise ValueError(f"target_visibility must have shape {expected}")
    if predicted_depth is not None and predicted_depth.shape != expected:
        raise ValueError(f"predicted_depth must have shape {expected}")
    if state.density_logits.shape[0] != batch_size:
        raise ValueError("state and target cameras must share batch size")

    origins, directions = generate_rays(cameras)
    ray_count = view_count * height * width
    origins = origins.reshape(batch_size, ray_count, 3)
    directions = directions.reshape(batch_size, ray_count, 3)
    near, far, hit = intersect_aabb(origins, directions, state.bounds)
    depth = target_depth[:, :, 0].reshape(batch_size, ray_count)
    if target_visibility is None:
        visible = torch.ones_like(depth, dtype=torch.bool)
    else:
        raw_visibility = target_visibility[:, :, 0].reshape(batch_size, ray_count)
        visible = torch.isfinite(raw_visibility) & (raw_visibility > 0.5)

    finite_positive_depth = torch.isfinite(depth) & (depth > 0.0)
    invalid_depth = ~finite_positive_depth
    invisible = finite_positive_depth & ~visible
    target_in_bounds = hit & (depth >= near) & (depth <= far)
    missed_bounds = finite_positive_depth & visible & ~target_in_bounds
    valid_rays = finite_positive_depth & visible & target_in_bounds

    fractions = (
        torch.arange(n_samples, device=origins.device, dtype=origins.dtype) + 0.5
    ) / n_samples
    distances = near.unsqueeze(-1) + (far - near).unsqueeze(-1) * fractions
    points = origins.unsqueeze(-2) + directions.unsqueeze(-2) * distances.unsqueeze(-1)
    confidence = _sample_volume(
        confidence_volume.float(), points.float(), state.bounds.float()
    ).squeeze(-1)
    confidence = _clamp_probability_roundoff(
        confidence,
        name="sampled evidence confidence",
    )

    depth_size, height_size, width_size = state.spatial_shape
    extent = state.bounds[:, 1].float() - state.bounds[:, 0].float()
    axis_sizes = torch.tensor(
        [width_size, height_size, depth_size], device=extent.device, dtype=extent.dtype
    )
    surface_band = 0.5 * (extent / axis_sizes).amax(dim=-1).view(batch_size, 1, 1)
    free, surface, behind = classify_depth_samples(
        distances.float(), depth.float().unsqueeze(-1), surface_band=surface_band
    )
    valid_samples = valid_rays.unsqueeze(-1)
    masks = {
        "free": free & valid_samples,
        "surface": surface & valid_samples,
        "behind": behind & valid_samples,
    }
    scores = {name: confidence[mask].float().cpu() for name, mask in masks.items()}
    bins = {name: _distribution(values) for name, values in scores.items()}
    comparisons = {
        "surface_vs_free": _comparison(scores["surface"], scores["free"]),
        "surface_vs_behind": _comparison(scores["surface"], scores["behind"]),
    }
    risk_report = None
    if predicted_depth is not None:
        prediction = predicted_depth[:, :, 0].reshape(batch_size, ray_count).float()
        valid_prediction = torch.isfinite(prediction) & (prediction >= 0.0)
        error_valid = valid_rays & valid_prediction
        absolute_error = (prediction - depth.float()).abs()
        absolute_relative_error = absolute_error / depth.float().clamp_min(1e-6)
        ray_risk = 1.0 - confidence.amax(dim=-1)
        risk_report = error_by_risk_deciles(
            ray_risk[error_valid],
            absolute_error[error_valid],
            absolute_relative_error[error_valid],
        )
    bands = surface_band.reshape(-1).cpu()
    band_value: float | list[float]
    if bands.numel() == 1 or torch.allclose(bands, bands[:1]):
        band_value = float(bands[0])
    else:
        band_value = [float(value) for value in bands]
    return {
        "schema_version": "mcss.evidence_audit.v1",
        "evidence_field": evidence_field,
        "label_semantics": "target-visible surface-near along fixed target-camera rays",
        "n_samples": n_samples,
        "surface_band_m": band_value,
        "rays": {
            "total": depth.numel(),
            "valid": int(valid_rays.sum()),
            "invalid_target_depth": int(invalid_depth.sum()),
            "target_invisible": int(invisible.sum()),
            "missed_state_bounds": int(missed_bounds.sum()),
        },
        "bins": bins,
        "comparisons": comparisons,
        "error_by_risk": risk_report,
    }


def _comparison(positive_scores: Tensor, negative_scores: Tensor) -> dict[str, object]:
    scores = torch.cat((positive_scores, negative_scores))
    labels = torch.cat(
        (
            torch.ones(positive_scores.numel(), dtype=torch.bool),
            torch.zeros(negative_scores.numel(), dtype=torch.bool),
        )
    )
    return binary_ranking_metrics(scores, labels)


def _clamp_probability_roundoff(values: Tensor, *, name: str) -> Tensor:
    if not torch.isfinite(values).all():
        raise ValueError(f"{name} must be finite")
    minimum = float(values.min())
    maximum = float(values.max())
    probability_tolerance = 1e-6
    if minimum < -probability_tolerance or maximum > 1.0 + probability_tolerance:
        raise ValueError(
            f"{name} must be within [0, 1]; observed [{minimum}, {maximum}]"
        )
    return values.clamp(0.0, 1.0)


def _distribution(values: Tensor) -> dict[str, object]:
    values = values.detach().float().reshape(-1).cpu()
    if values.numel() == 0:
        return {"count": 0, "mean": None, "std": None, "deciles": {}}
    quantiles = torch.linspace(0.0, 1.0, 11)
    quantile_values = torch.quantile(values, quantiles)
    return {
        "count": values.numel(),
        "mean": float(values.mean()),
        "std": float(values.std(unbiased=False)),
        "deciles": {
            f"p{index * 10:02d}": float(value)
            for index, value in enumerate(quantile_values)
        },
    }


def _monotonic_violations(values: list[float]) -> int:
    return sum(right + 1e-8 < left for left, right in zip(values, values[1:], strict=False))


def _ordered_spearman(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    tensor = torch.tensor(values, dtype=torch.float64)
    unique, inverse, counts = torch.unique(
        tensor, sorted=True, return_inverse=True, return_counts=True
    )
    del unique
    cumulative = counts.cumsum(dim=0)
    rank_start = cumulative - counts
    ranks = ((rank_start + cumulative - 1).to(torch.float64) * 0.5)[inverse]
    ordered = torch.arange(tensor.numel(), dtype=torch.float64)
    ordered -= ordered.mean()
    ranks -= ranks.mean()
    denominator = torch.linalg.vector_norm(ordered) * torch.linalg.vector_norm(ranks)
    if denominator <= 0:
        return None
    return float((ordered * ranks).sum() / denominator)


def _binary_auroc(scores: Tensor, labels: Tensor) -> float:
    unique, inverse, counts = torch.unique(
        scores, sorted=True, return_inverse=True, return_counts=True
    )
    del unique
    cumulative = counts.cumsum(dim=0)
    rank_start = cumulative - counts + 1
    average_ranks = (rank_start + cumulative).float() * 0.5
    ranks = average_ranks[inverse]
    positives = int(labels.sum())
    negatives = labels.numel() - positives
    rank_sum = ranks[labels].sum()
    auc = (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)
    return float(auc)


def _binary_average_precision(scores: Tensor, labels: Tensor) -> float:
    order = torch.argsort(scores, descending=True, stable=True)
    ordered_labels = labels[order].float()
    cumulative_positive = ordered_labels.cumsum(dim=0)
    ranks = torch.arange(1, labels.numel() + 1, dtype=torch.float32)
    precision = cumulative_positive / ranks
    return float((precision * ordered_labels).sum() / ordered_labels.sum().clamp_min(1.0))


def _empty_binary_metrics(reliability_bins: int) -> dict[str, object]:
    if reliability_bins < 2:
        raise ValueError("reliability_bins must be at least two")
    return {
        "count": 0,
        "positive_count": 0,
        "negative_count": 0,
        "auroc": None,
        "auprc": None,
        "brier": None,
        "ece": None,
        "reliability": [
            {
                "lower": index / reliability_bins,
                "upper": (index + 1) / reliability_bins,
                "count": 0,
                "mean_confidence": None,
                "positive_rate": None,
            }
            for index in range(reliability_bins)
        ],
    }
