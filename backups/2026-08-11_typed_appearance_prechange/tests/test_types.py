import pytest
import torch

from mcss.types import Cameras, SceneState, StateEvidence


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
