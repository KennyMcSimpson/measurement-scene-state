"""Training-only context-subset geometry consistency for the unchanged V5 state."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch
import torch.nn.functional as functional
from torch import Tensor

from mcss.measurements import FixedMeasurementRenderer
from mcss.types import Cameras, SceneState


@dataclass(frozen=True)
class ContextSubset:
    """One context tensor/camera set with a single view removed in place."""

    rgb: Tensor
    cameras: Cameras
    dropped_index: int
    retained_indices: tuple[int, ...]


@dataclass(frozen=True)
class ContextSubsetGeometryResult:
    """Differentiable consistency loss plus detached audit telemetry."""

    loss: Tensor
    terms: dict[str, Tensor]
    diagnostics: dict[str, float]
    audit_diagnostics: dict[str, float]
    per_sample_diagnostics: tuple[dict[str, float], ...]


def drop_context_view(
    context_rgb: Tensor,
    cameras: Cameras,
    *,
    dropped_index: int,
) -> ContextSubset:
    """Remove one view without changing the established coordinate frame or order."""

    if context_rgb.ndim != 5 or context_rgb.shape[2] != 3:
        raise ValueError("context_rgb must have shape [B, V, 3, H, W]")
    batch_size, view_count = context_rgb.shape[:2]
    if cameras.leading_shape != (batch_size, view_count):
        raise ValueError("context cameras must align with context_rgb")
    if cameras.device != context_rgb.device:
        raise ValueError("context cameras and RGB must be on the same device")
    if view_count < 2:
        raise ValueError("context view deletion requires at least two views")
    if not 0 <= dropped_index < view_count:
        raise ValueError("dropped_index must select an existing context view")

    retained_indices = tuple(index for index in range(view_count) if index != dropped_index)
    indices = torch.tensor(retained_indices, device=context_rgb.device, dtype=torch.long)
    return ContextSubset(
        rgb=context_rgb.index_select(1, indices),
        cameras=cameras.select_views(indices),
        dropped_index=dropped_index,
        retained_indices=retained_indices,
    )


def ramped_context_subset_geometry_weight(
    max_weight: float,
    *,
    optimizer_step: int,
    warmup_steps: int,
) -> float:
    """Ramp against optimizer updates, not gradient-accumulation micro-steps."""

    if max_weight < 0.0:
        raise ValueError("max_weight must be non-negative")
    if optimizer_step < 1:
        raise ValueError("optimizer_step must be positive")
    if warmup_steps < 0:
        raise ValueError("warmup_steps must be non-negative")
    if max_weight == 0.0 or warmup_steps == 0:
        return float(max_weight)
    return float(max_weight) * min(optimizer_step / warmup_steps, 1.0)


def compute_context_subset_geometry(
    full_state: SceneState,
    subset_state: SceneState,
    full_predictions: Mapping[str, Tensor],
    target_cameras: Cameras,
    renderer: FixedMeasurementRenderer,
    *,
    residual_saturation_threshold: float,
    collect_audit_diagnostics: bool = True,
) -> ContextSubsetGeometryResult:
    """Compare subset completion geometry against a detached full-context teacher."""

    if full_state.evidence is None or subset_state.evidence is None:
        raise ValueError("context-subset geometry requires V5 StateEvidence on both paths")
    if residual_saturation_threshold <= 0.0:
        raise ValueError("residual_saturation_threshold must be positive")
    if "depth" not in full_predictions or "visibility" not in full_predictions:
        raise ValueError("full_predictions must contain depth and visibility")

    full_evidence = full_state.evidence
    subset_evidence = subset_state.evidence
    reference_shape = full_state.density_logits.shape
    if subset_state.density_logits.shape != reference_shape:
        raise ValueError("full and subset geometry grids must have identical shapes")
    if full_evidence.density_residual.shape != subset_evidence.density_residual.shape:
        raise ValueError("full and subset density residuals must have identical shapes")

    shared_support = (
        subset_evidence.provenance.detach().gt(1e-6).sum(dim=1, keepdim=True) >= 2
    )
    stable_evidence = (
        1.0
        - (
            full_evidence.confidence.detach()
            - subset_evidence.confidence.detach()
        ).abs()
    ).clamp(0.0, 1.0)
    overlap_weight = (
        shared_support.to(dtype=full_state.density_logits.dtype) * stable_evidence
    ).detach()

    teacher_density = full_state.density_logits.detach()
    subset_on_full_evidence = (
        full_evidence.base_density_logits.detach()
        + full_evidence.completion_gate.detach() * subset_evidence.density_residual
    )
    counterfactual_density = (
        overlap_weight * subset_on_full_evidence
        + (1.0 - overlap_weight) * teacher_density
    )
    counterfactual_state = SceneState(
        density_logits=counterfactual_density,
        color=full_state.color.detach(),
        log_variance=full_state.log_variance.detach(),
        bounds=full_state.bounds.detach(),
    )
    counterfactual_predictions = renderer(
        counterfactual_state,
        target_cameras,
        {"depth", "visibility"},
    )

    teacher_depth = full_predictions["depth"].detach().float()
    teacher_visibility = full_predictions["visibility"].detach().float()
    counterfactual_depth = counterfactual_predictions["depth"].float()
    counterfactual_visibility = counterfactual_predictions["visibility"].float()
    voxel_spacing = _mean_metric_voxel_spacing(
        full_state.bounds.detach(),
        full_state.spatial_shape,
    )
    depth_error = functional.smooth_l1_loss(
        counterfactual_depth / voxel_spacing,
        teacher_depth / voxel_spacing,
        reduction="none",
    )
    visible_mass = teacher_visibility.sum()
    depth_loss = (depth_error * teacher_visibility).sum() / visible_mass.clamp_min(1.0)
    visibility_loss = functional.smooth_l1_loss(
        counterfactual_visibility,
        teacher_visibility,
    )
    loss = depth_loss + visibility_loss

    residual_difference = (
        full_evidence.density_residual.detach()
        - subset_evidence.density_residual.detach()
    ).abs()
    diagnostics = {
        "full_evidence_mean": _detached_mean(full_evidence.confidence),
        "subset_evidence_mean": _detached_mean(subset_evidence.confidence),
        "evidence_abs_diff": _detached_mean(
            (full_evidence.confidence - subset_evidence.confidence).abs()
        ),
        "full_completion_gate_mean": _detached_mean(full_evidence.completion_gate),
        "subset_completion_gate_mean": _detached_mean(subset_evidence.completion_gate),
        "shared_coverage": _detached_mean(shared_support),
        "overlap_weight_mean": _detached_mean(overlap_weight),
        "residual_abs_diff": _detached_weighted_mean(
            residual_difference,
            overlap_weight,
        ),
        "counterfactual_depth_drift_voxels": _detached_weighted_mean(
            (counterfactual_depth.detach() - teacher_depth).abs() / voxel_spacing,
            teacher_visibility,
        ),
        "visibility_drift": _detached_mean(
            (counterfactual_visibility.detach() - teacher_visibility).abs()
        ),
    }
    audit_diagnostics: dict[str, float] = {}
    per_sample_diagnostics: tuple[dict[str, float], ...] = ()
    if collect_audit_diagnostics:
        audit_diagnostics = {
            "full_residual_saturation_fraction": _detached_mean(
                full_evidence.density_residual.detach().abs()
                >= residual_saturation_threshold
            ),
            "subset_residual_saturation_fraction": _detached_mean(
                subset_evidence.density_residual.detach().abs()
                >= residual_saturation_threshold
            ),
            **_residual_quantile_diagnostics(
                "full_residual_abs",
                full_evidence.density_residual,
            ),
            **_residual_quantile_diagnostics(
                "subset_residual_abs",
                subset_evidence.density_residual,
            ),
        }
        per_sample_diagnostics = tuple(
            {
                "full_evidence_mean": _detached_mean(full_evidence.confidence[index]),
                "subset_evidence_mean": _detached_mean(subset_evidence.confidence[index]),
                "full_completion_gate_mean": _detached_mean(
                    full_evidence.completion_gate[index]
                ),
                "subset_completion_gate_mean": _detached_mean(
                    subset_evidence.completion_gate[index]
                ),
                "full_residual_abs": _detached_mean(
                    full_evidence.density_residual[index].abs()
                ),
                "subset_residual_abs": _detached_mean(
                    subset_evidence.density_residual[index].abs()
                ),
                "residual_abs_diff": _detached_weighted_mean(
                    residual_difference[index],
                    overlap_weight[index],
                ),
                "shared_coverage": _detached_mean(shared_support[index]),
                "overlap_weight_mean": _detached_mean(overlap_weight[index]),
            }
            for index in range(reference_shape[0])
        )
    return ContextSubsetGeometryResult(
        loss=loss,
        terms={"depth": depth_loss, "visibility": visibility_loss},
        diagnostics=diagnostics,
        audit_diagnostics=audit_diagnostics,
        per_sample_diagnostics=per_sample_diagnostics,
    )


def _mean_metric_voxel_spacing(
    bounds: Tensor,
    spatial_shape: tuple[int, int, int],
) -> Tensor:
    depth, height, width = spatial_shape
    xyz_counts = bounds.new_tensor([width, height, depth])
    spacing = ((bounds[:, 1] - bounds[:, 0]) / xyz_counts).mean(dim=-1)
    return spacing.float().view(-1, 1, 1, 1, 1).clamp_min(1e-6)


def _detached_mean(values: Tensor) -> float:
    return float(values.detach().float().mean().item())


def _detached_weighted_mean(values: Tensor, weights: Tensor) -> float:
    values_float = values.detach().float()
    weights_float = weights.detach().float()
    weight_sum = weights_float.sum()
    if weight_sum.item() == 0.0:
        return 0.0
    return float((values_float * weights_float).sum().div(weight_sum).item())


def _residual_quantile_diagnostics(prefix: str, residual: Tensor) -> dict[str, float]:
    probabilities = torch.arange(0.1, 1.0, 0.1, device=residual.device)
    quantiles = torch.quantile(residual.detach().float().abs().reshape(-1), probabilities)
    return {
        f"{prefix}_p{index * 10:02d}": float(value.item())
        for index, value in enumerate(quantiles, start=1)
    }
