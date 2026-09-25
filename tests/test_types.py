import pytest
import torch

from mcss.types import (
    Cameras,
    DualEvidence,
    SceneState,
    StateAppearance,
    StateEvidence,
    TransportEvidence,
)


def test_cameras_validate_shape_and_homogeneous_row() -> None:
    intrinsics = torch.eye(3).view(1, 1, 3, 3)
    c2w = torch.eye(4).view(1, 1, 4, 4)
    cameras = Cameras(intrinsics=intrinsics, c2w=c2w, image_size=(8, 10))

    assert cameras.leading_shape == (1, 1)
    assert cameras.image_size == (8, 10)

    invalid = c2w.clone()
    invalid[..., 3, 0] = 1.0
    with pytest.raises(ValueError, match="homogeneous"):
        Cameras(intrinsics=intrinsics, c2w=invalid, image_size=(8, 10))


def test_scene_state_rejects_inconsistent_typed_fields() -> None:
    bounds = torch.tensor([[[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]]])
    density = torch.zeros(1, 1, 4, 4, 4)
    color = torch.zeros(1, 3, 4, 4, 4)
    variance = torch.zeros(1, 1, 4, 4, 4)

    state = SceneState(density, color, variance, bounds)
    assert state.spatial_shape == (4, 4, 4)

    with pytest.raises(ValueError, match="spatial"):
        SceneState(density, color[..., :-1], variance, bounds)


def _state_evidence() -> StateEvidence:
    shape = (1, 1, 4, 4, 4)
    confidence = torch.full(shape, 0.75)
    unknown = 1.0 - confidence
    provenance = torch.full((1, 2, 4, 4, 4), 0.5)
    return StateEvidence(
        confidence=confidence,
        unknown_probability=unknown,
        completion_gate=unknown,
        provenance=provenance,
        base_density_logits=torch.zeros(shape),
        base_color=torch.full((1, 3, 4, 4, 4), 0.5),
        density_residual=torch.full(shape, 0.1),
        color_logit_residual=torch.full((1, 3, 4, 4, 4), 0.2),
    )


def test_scene_state_evidence_validates_and_transfers_with_state() -> None:
    bounds = torch.tensor([[[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]]])
    density = torch.zeros(1, 1, 4, 4, 4)
    color = torch.zeros(1, 3, 4, 4, 4)
    variance = torch.zeros(1, 1, 4, 4, 4)
    evidence = _state_evidence()

    state = SceneState(density, color, variance, bounds, evidence=evidence)
    transferred = state.to(dtype=torch.float64)

    assert transferred.evidence is not None
    assert transferred.evidence.confidence.dtype == torch.float64
    torch.testing.assert_close(
        transferred.evidence.provenance.sum(dim=1),
        torch.ones(1, 4, 4, 4, dtype=torch.float64),
    )


def test_state_evidence_rejects_invalid_probabilities_and_provenance() -> None:
    evidence = _state_evidence()

    with pytest.raises(ValueError, match="confidence"):
        StateEvidence(
            **{
                **evidence.__dict__,
                "confidence": torch.full_like(evidence.confidence, 1.1),
            }
        )
    with pytest.raises(ValueError, match="provenance"):
        StateEvidence(
            **{
                **evidence.__dict__,
                "provenance": torch.full_like(evidence.provenance, 0.25),
            }
        )


def _state_appearance() -> StateAppearance:
    shape = (1, 1, 8, 8, 8)
    confidence = torch.full(shape, 0.6)
    unknown = 1.0 - confidence
    return StateAppearance(
        confidence=confidence,
        unknown_probability=unknown,
        completion_gate=0.1 + 0.9 * unknown,
        provenance=torch.full((1, 2, 8, 8, 8), 0.5),
        base_color=torch.full((1, 3, 8, 8, 8), 0.25),
        color_logit_residual=torch.full((1, 3, 8, 8, 8), 0.4),
    )


def test_typed_appearance_reconstructs_color_and_transfers_with_scene_state() -> None:
    bounds = torch.tensor([[[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]]])
    density = torch.zeros(1, 1, 4, 4, 4)
    native_color = torch.zeros(1, 3, 4, 4, 4)
    appearance = _state_appearance()

    state = SceneState(
        density,
        native_color,
        torch.zeros_like(density),
        bounds,
        appearance=appearance,
    )
    transferred = state.to(dtype=torch.float64)

    expected = torch.sigmoid(
        torch.logit(appearance.base_color) + appearance.completion_gate * 0.4
    )
    torch.testing.assert_close(appearance.color, expected)
    assert state.appearance_shape == (8, 8, 8)
    assert transferred.appearance is not None
    assert transferred.appearance.base_color.dtype == torch.float64
    torch.testing.assert_close(
        transferred.appearance.provenance.sum(dim=1),
        torch.ones(1, 8, 8, 8, dtype=torch.float64),
    )


def test_typed_appearance_rejects_invalid_ranges_and_provenance() -> None:
    appearance = _state_appearance()

    with pytest.raises(ValueError, match="confidence"):
        StateAppearance(
            **{
                **appearance.__dict__,
                "confidence": torch.full_like(appearance.confidence, -0.1),
            }
        )
    with pytest.raises(ValueError, match="provenance"):
        StateAppearance(
            **{
                **appearance.__dict__,
                "provenance": torch.full_like(appearance.provenance, 0.25),
            }
        )


def _dual_evidence() -> DualEvidence:
    shape = (1, 1, 4, 4, 4)
    appearance_confidence = torch.full(shape, 0.75)
    surface_peakness = torch.full(shape, 0.4)
    localization_support = appearance_confidence * surface_peakness
    return DualEvidence(
        surface_localization_support=localization_support,
        appearance_confidence=appearance_confidence,
        surface_peakness=surface_peakness,
        localization_unknown_probability=1.0 - localization_support,
        appearance_unknown_probability=1.0 - appearance_confidence,
        localization_completion_gate=0.1 + 0.9 * (1.0 - localization_support),
        appearance_completion_gate=0.1 + 0.9 * (1.0 - appearance_confidence),
        provenance=torch.full((1, 2, 4, 4, 4), 0.5),
        base_density_logits=4.0 * localization_support - 2.0,
        base_color=torch.full((1, 3, 4, 4, 4), 0.5),
        density_residual=torch.full(shape, 0.2),
        color_logit_residual=torch.full((1, 3, 4, 4, 4), 0.3),
    )


def test_dual_evidence_validates_reconstructs_and_transfers_with_state() -> None:
    evidence = _dual_evidence()
    density = evidence.base_density_logits + (
        evidence.localization_completion_gate * evidence.density_residual
    )
    color = torch.sigmoid(
        torch.logit(evidence.base_color)
        + evidence.appearance_completion_gate * evidence.color_logit_residual
    )
    state = SceneState(
        density_logits=density,
        color=color,
        log_variance=torch.zeros_like(density),
        bounds=torch.tensor([[[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]]]),
        dual_evidence=evidence,
    )

    transferred = state.to(dtype=torch.float64)

    assert transferred.dual_evidence is not None
    assert transferred.dual_evidence.appearance_confidence.dtype == torch.float64
    torch.testing.assert_close(evidence.density_logits, density)
    torch.testing.assert_close(evidence.color, color)
    assert torch.all(
        evidence.surface_localization_support <= evidence.appearance_confidence
    )


def test_dual_evidence_rejects_support_above_appearance_confidence() -> None:
    evidence = _dual_evidence()

    with pytest.raises(ValueError, match="surface_localization_support"):
        DualEvidence(
            **{
                **evidence.__dict__,
                "surface_localization_support": torch.full_like(
                    evidence.surface_localization_support, 0.9
                ),
            }
        )


def _transport_evidence() -> TransportEvidence:
    shape = (1, 1, 4, 4, 4)
    transported_appearance = torch.full(shape, 0.5)
    surface_peakness = torch.full(shape, 0.4)
    localization_support = transported_appearance * surface_peakness
    appearance_confidence = torch.full(shape, 0.75)
    return TransportEvidence(
        surface_localization_support=localization_support,
        transported_appearance_confidence=transported_appearance,
        surface_peakness=surface_peakness,
        transport_coverage=torch.full(shape, 3.0),
        appearance_confidence=appearance_confidence,
        localization_unknown_probability=1.0 - localization_support,
        appearance_unknown_probability=1.0 - appearance_confidence,
        localization_completion_gate=0.1 + 0.9 * (1.0 - localization_support),
        appearance_completion_gate=0.1 + 0.9 * (1.0 - appearance_confidence),
        provenance=torch.full((1, 2, 4, 4, 4), 0.5),
        base_density_logits=4.0 * localization_support - 2.0,
        base_color=torch.full((1, 3, 4, 4, 4), 0.5),
        density_residual=torch.full(shape, 0.2),
        color_logit_residual=torch.full((1, 3, 4, 4, 4), 0.3),
    )


def test_transport_evidence_validates_reconstructs_and_transfers_with_state() -> None:
    evidence = _transport_evidence()
    state = SceneState(
        density_logits=evidence.density_logits,
        color=evidence.color,
        log_variance=torch.zeros_like(evidence.density_logits),
        bounds=torch.tensor([[[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]]]),
        transport_evidence=evidence,
    )

    transferred = state.to(dtype=torch.float64)

    assert transferred.transport_evidence is not None
    assert transferred.transport_evidence.transport_coverage.dtype == torch.float64
    torch.testing.assert_close(
        evidence.surface_localization_support,
        evidence.transported_appearance_confidence * evidence.surface_peakness,
    )
    torch.testing.assert_close(evidence.density_logits, state.density_logits)
    torch.testing.assert_close(evidence.color, state.color)


def test_transport_evidence_rejects_invalid_identity_and_negative_coverage() -> None:
    evidence = _transport_evidence()

    with pytest.raises(ValueError, match="surface_localization_support"):
        TransportEvidence(
            **{
                **evidence.__dict__,
                "surface_localization_support": torch.full_like(
                    evidence.surface_localization_support, 0.9
                ),
            }
        )
    with pytest.raises(ValueError, match="transport_coverage"):
        TransportEvidence(
            **{
                **evidence.__dict__,
                "transport_coverage": torch.full_like(evidence.transport_coverage, -1.0),
            }
        )


def test_scene_state_rejects_multiple_native_evidence_payloads() -> None:
    transport = _transport_evidence()
    dual = _dual_evidence()

    with pytest.raises(ValueError, match="mutually exclusive"):
        SceneState(
            density_logits=transport.density_logits,
            color=transport.color,
            log_variance=torch.zeros_like(transport.density_logits),
            bounds=torch.tensor([[[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]]]),
            dual_evidence=dual,
            transport_evidence=transport,
        )
