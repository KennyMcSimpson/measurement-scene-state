import pytest
import torch

from mcss.context_subset_consistency import (
    compute_context_subset_geometry,
    drop_context_view,
    ramped_context_subset_geometry_weight,
)
from mcss.geometry import look_at, make_intrinsics
from mcss.measurements import FixedMeasurementRenderer
from mcss.types import Cameras, SceneState, StateEvidence


def _four_context_cameras() -> Cameras:
    intrinsics = make_intrinsics((4, 4), 35.0).view(1, 1, 3, 3).repeat(1, 4, 1, 1)
    intrinsics[0, :, 0, 0] += torch.arange(4)
    c2w = torch.eye(4).view(1, 1, 4, 4).repeat(1, 4, 1, 1)
    c2w[0, :, 0, 3] = torch.arange(4)
    return Cameras(intrinsics, c2w, (4, 4))


def _target_cameras() -> Cameras:
    intrinsics = make_intrinsics((4, 4), 35.0).view(1, 1, 3, 3)
    c2w = look_at(torch.tensor([0.0, 0.0, -3.0]), torch.zeros(3)).view(1, 1, 4, 4)
    return Cameras(intrinsics, c2w, (4, 4))


def _evidence_state(
    *,
    residual: torch.Tensor,
    confidence: torch.Tensor,
    provenance: torch.Tensor,
) -> SceneState:
    unknown = 1.0 - confidence
    gate = 0.1 + 0.9 * unknown
    base_density = 4.0 * confidence - 2.0
    density = base_density + gate * residual
    color = torch.full((1, 3, *confidence.shape[-3:]), 0.5)
    evidence = StateEvidence(
        confidence=confidence,
        unknown_probability=unknown,
        completion_gate=gate,
        provenance=provenance,
        base_density_logits=base_density,
        base_color=color,
        density_residual=residual,
        color_logit_residual=torch.zeros_like(color),
    )
    return SceneState(
        density_logits=density,
        color=color,
        log_variance=torch.full_like(density, -3.0),
        bounds=torch.tensor([[[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]]]),
        evidence=evidence,
    )


def test_drop_context_view_preserves_retained_rgb_camera_order() -> None:
    rgb = torch.arange(4, dtype=torch.float32).view(1, 4, 1, 1, 1).expand(1, 4, 3, 4, 4)
    cameras = _four_context_cameras()

    subset = drop_context_view(rgb, cameras, dropped_index=2)

    assert subset.dropped_index == 2
    assert subset.retained_indices == (0, 1, 3)
    torch.testing.assert_close(subset.rgb[:, :, 0, 0, 0], torch.tensor([[0.0, 1.0, 3.0]]))
    torch.testing.assert_close(
        subset.cameras.intrinsics,
        cameras.intrinsics[:, [0, 1, 3]],
    )
    torch.testing.assert_close(subset.cameras.c2w, cameras.c2w[:, [0, 1, 3]])


@pytest.mark.parametrize("dropped_index", [-1, 4])
def test_drop_context_view_rejects_invalid_index(dropped_index: int) -> None:
    rgb = torch.zeros(1, 4, 3, 4, 4)

    with pytest.raises(ValueError, match="dropped_index"):
        drop_context_view(rgb, _four_context_cameras(), dropped_index=dropped_index)


def test_context_subset_weight_uses_optimizer_update_warmup() -> None:
    assert ramped_context_subset_geometry_weight(0.0, optimizer_step=1, warmup_steps=4) == 0.0
    assert ramped_context_subset_geometry_weight(0.1, optimizer_step=1, warmup_steps=4) == 0.025
    assert ramped_context_subset_geometry_weight(0.1, optimizer_step=4, warmup_steps=4) == 0.1
    assert ramped_context_subset_geometry_weight(0.1, optimizer_step=8, warmup_steps=4) == 0.1
    assert ramped_context_subset_geometry_weight(0.1, optimizer_step=1, warmup_steps=0) == 0.1


def test_context_subset_geometry_updates_only_subset_residual() -> None:
    shape = (1, 1, 4, 4, 4)
    full_confidence = torch.full(shape, 0.5, requires_grad=True)
    subset_confidence = torch.full(shape, 0.45, requires_grad=True)
    full_residual = torch.zeros(shape, requires_grad=True)
    subset_residual = torch.full(shape, 1.5, requires_grad=True)
    full_provenance = torch.full((1, 4, 4, 4, 4), 0.25)
    subset_provenance = torch.full((1, 3, 4, 4, 4), 1.0 / 3.0)
    full_state = _evidence_state(
        residual=full_residual,
        confidence=full_confidence,
        provenance=full_provenance,
    )
    subset_state = _evidence_state(
        residual=subset_residual,
        confidence=subset_confidence,
        provenance=subset_provenance,
    )
    renderer = FixedMeasurementRenderer(n_samples=12, ray_chunk_size=32)
    target_cameras = _target_cameras()
    full_predictions = renderer(full_state, target_cameras, {"depth", "visibility"})

    result = compute_context_subset_geometry(
        full_state,
        subset_state,
        full_predictions,
        target_cameras,
        renderer,
        residual_saturation_threshold=3.8,
    )

    assert torch.isfinite(result.loss)
    assert result.loss > 0.0
    assert set(result.terms) == {"depth", "visibility"}
    assert result.diagnostics["shared_coverage"] == 1.0
    assert 0.0 < result.diagnostics["overlap_weight_mean"] < 1.0
    assert "full_residual_abs_p50" in result.audit_diagnostics
    assert len(result.per_sample_diagnostics) == 1
    result.loss.backward()
    assert subset_residual.grad is not None
    assert subset_residual.grad.abs().sum() > 0.0
    assert full_residual.grad is None
    assert full_confidence.grad is None
    assert subset_confidence.grad is None


def test_context_subset_geometry_zero_overlap_is_finite_zero() -> None:
    shape = (1, 1, 4, 4, 4)
    full_state = _evidence_state(
        residual=torch.zeros(shape),
        confidence=torch.full(shape, 0.5),
        provenance=torch.full((1, 4, 4, 4, 4), 0.25),
    )
    subset_residual = torch.full(shape, 3.0, requires_grad=True)
    subset_state = _evidence_state(
        residual=subset_residual,
        confidence=torch.zeros(shape),
        provenance=torch.zeros(1, 3, 4, 4, 4),
    )
    renderer = FixedMeasurementRenderer(n_samples=12, ray_chunk_size=32)
    target_cameras = _target_cameras()
    full_predictions = renderer(full_state, target_cameras, {"depth", "visibility"})

    result = compute_context_subset_geometry(
        full_state,
        subset_state,
        full_predictions,
        target_cameras,
        renderer,
        residual_saturation_threshold=3.8,
    )

    assert result.diagnostics["shared_coverage"] == 0.0
    assert result.diagnostics["overlap_weight_mean"] == 0.0
    torch.testing.assert_close(result.loss, torch.zeros_like(result.loss))
    result.loss.backward()
    assert subset_residual.grad is not None
    torch.testing.assert_close(subset_residual.grad, torch.zeros_like(subset_residual.grad))


def test_context_subset_geometry_skips_expensive_audit_when_not_logging() -> None:
    shape = (1, 1, 4, 4, 4)
    full_state = _evidence_state(
        residual=torch.zeros(shape),
        confidence=torch.full(shape, 0.5),
        provenance=torch.full((1, 4, 4, 4, 4), 0.25),
    )
    subset_state = _evidence_state(
        residual=torch.ones(shape, requires_grad=True),
        confidence=torch.full(shape, 0.45),
        provenance=torch.full((1, 3, 4, 4, 4), 1.0 / 3.0),
    )
    renderer = FixedMeasurementRenderer(n_samples=12, ray_chunk_size=32)
    target_cameras = _target_cameras()
    full_predictions = renderer(full_state, target_cameras, {"depth", "visibility"})

    result = compute_context_subset_geometry(
        full_state,
        subset_state,
        full_predictions,
        target_cameras,
        renderer,
        residual_saturation_threshold=3.8,
        collect_audit_diagnostics=False,
    )

    assert torch.isfinite(result.loss)
    assert result.audit_diagnostics == {}
    assert result.per_sample_diagnostics == ()
