import pytest
import torch

from mcss.engine import _state_diagnostics, _state_regularization
from mcss.geometry import look_at, make_intrinsics
from mcss.model.evidence_encoder import _cross_view_evidence
from mcss.model.system import ModelConfig, build_model
from mcss.types import Cameras, SceneState, StateEvidence


def _model_inputs() -> tuple[torch.Tensor, Cameras, Cameras, torch.Tensor]:
    torch.manual_seed(13)
    context_rgb = torch.rand(1, 2, 3, 8, 8)
    intrinsics = make_intrinsics((8, 8), 60.0)
    context_poses = torch.stack(
        (
            look_at(torch.tensor([0.0, 0.0, -3.0]), torch.zeros(3)),
            look_at(torch.tensor([2.0, 0.0, -2.0]), torch.zeros(3)),
        )
    )
    target_pose = look_at(torch.tensor([-2.0, 0.0, -2.0]), torch.zeros(3))
    context_cameras = Cameras(
        intrinsics.expand(1, 2, 3, 3).clone(), context_poses.unsqueeze(0), (8, 8)
    )
    target_cameras = Cameras(
        intrinsics.view(1, 1, 3, 3), target_pose.view(1, 1, 4, 4), (8, 8)
    )
    bounds = torch.tensor([[[-1.2, -1.2, -1.2], [1.2, 1.2, 1.2]]])
    return context_rgb, context_cameras, target_cameras, bounds


def test_cross_view_evidence_prefers_agreement_and_masks_invalid_views() -> None:
    features = torch.tensor(
        [[[[1.0, 0.0], [1.0, 0.0]], [[1.0, 0.0], [-1.0, 0.0]]]]
    )
    rgb = torch.tensor(
        [[[[0.2, 0.3, 0.4], [0.1, 0.1, 0.1]], [[0.2, 0.3, 0.4], [0.9, 0.9, 0.9]]]]
    )
    valid = torch.ones(1, 2, 2, dtype=torch.bool)

    confidence, provenance = _cross_view_evidence(features, rgb, valid, temperature=0.1)

    assert confidence.shape == (1, 2, 1)
    assert provenance.shape == (1, 2, 2, 1)
    assert confidence[0, 0, 0] > confidence[0, 1, 0]
    torch.testing.assert_close(provenance.sum(dim=1), torch.ones(1, 2, 1))

    one_view_valid = torch.tensor([[[True], [False]]])
    confidence, provenance = _cross_view_evidence(
        features[:, :, :1], rgb[:, :, :1], one_view_valid, temperature=0.1
    )
    torch.testing.assert_close(confidence, torch.zeros_like(confidence))
    assert provenance[0, 1, 0, 0] == 0.0
    assert provenance[0, 0, 0, 0] == 1.0


def test_cross_view_evidence_cannot_optimize_its_own_confidence() -> None:
    features = torch.tensor(
        [[[[1.0, 0.0]], [[0.8, 0.2]]]],
        requires_grad=True,
    )
    rgb = torch.tensor([[[[0.2, 0.3, 0.4]], [[0.2, 0.3, 0.4]]]])
    valid = torch.ones(1, 2, 1, dtype=torch.bool)

    confidence, provenance = _cross_view_evidence(features, rgb, valid, temperature=0.1)

    assert not confidence.requires_grad
    assert not provenance.requires_grad


def test_evidence_residual_state_is_auditable_and_reconstructable() -> None:
    config = ModelConfig(
        mode="fixed",
        state_architecture="evidence_residual",
        voxel_resolution=8,
        image_feature_dim=8,
        state_feature_dim=8,
        refinement_blocks=1,
        evidence_temperature=0.1,
        observed_residual_floor=0.2,
        completion_residual_scale=1.5,
        n_samples=8,
        ray_chunk_size=128,
    )
    model = build_model(config)

    output = model(*_model_inputs())
    evidence = output.state.evidence

    assert evidence is not None
    assert evidence.confidence.shape == (1, 1, 8, 8, 8)
    assert evidence.provenance.shape == (1, 2, 8, 8, 8)
    assert (evidence.completion_gate >= 0.2).all()
    assert (evidence.completion_gate <= 1.0).all()
    assert evidence.density_residual.abs().max() <= 1.5
    assert evidence.color_logit_residual.abs().max() <= 1.5
    assert evidence.base_density_logits.min() >= -2.0001
    assert evidence.base_density_logits.max() <= 2.0001
    empty_provenance = evidence.provenance.sum(dim=1, keepdim=True) == 0
    if empty_provenance.any():
        empty_color = empty_provenance.expand_as(evidence.base_color)
        torch.testing.assert_close(
            evidence.base_color[empty_color],
            torch.full_like(evidence.base_color[empty_color], 0.5),
        )

    reconstructed_density = (
        evidence.base_density_logits + evidence.completion_gate * evidence.density_residual
    )
    base_logits = torch.logit(evidence.base_color.clamp(1e-4, 1.0 - 1e-4))
    reconstructed_color = torch.sigmoid(
        base_logits + evidence.completion_gate * evidence.color_logit_residual
    )
    torch.testing.assert_close(output.state.density_logits, reconstructed_density)
    torch.testing.assert_close(output.state.color, reconstructed_color)
    assert all(torch.isfinite(value).all() for value in output.predictions.values())

    loss = sum(value.mean() for value in output.predictions.values())
    loss.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.requires_grad]
    assert any(gradient is not None for gradient in gradients)
    assert all(torch.isfinite(gradient).all() for gradient in gradients if gradient is not None)


def test_evidence_fields_are_predictor_independent_across_an_optimizer_step() -> None:
    config = ModelConfig(
        mode="fixed",
        state_architecture="evidence_residual",
        voxel_resolution=8,
        image_feature_dim=8,
        state_feature_dim=8,
        refinement_blocks=1,
        n_samples=8,
        ray_chunk_size=128,
    )
    model = build_model(config)
    inputs = _model_inputs()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)

    before = model(*inputs)
    assert before.state.evidence is not None
    confidence = before.state.evidence.confidence.detach().clone()
    provenance = before.state.evidence.provenance.detach().clone()
    loss = sum(value.mean() for value in before.predictions.values())
    loss.backward()
    optimizer.step()
    after = model(*inputs)

    assert after.state.evidence is not None
    torch.testing.assert_close(after.state.evidence.confidence, confidence)
    torch.testing.assert_close(after.state.evidence.provenance, provenance)


def test_evidence_residual_regularization_preserves_legacy_and_penalizes_rewriting() -> None:
    shape = (1, 1, 4, 4, 4)
    bounds = torch.tensor([[[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]]])
    density = torch.zeros(shape)
    color = torch.full((1, 3, 4, 4, 4), 0.5)
    variance = torch.zeros(shape)
    legacy = SceneState(density, color, variance, bounds)

    legacy_unweighted = _state_regularization(legacy, evidence_residual_weight=0.0)
    legacy_weighted = _state_regularization(legacy, evidence_residual_weight=1.0)
    torch.testing.assert_close(legacy_unweighted, legacy_weighted)
    assert _state_diagnostics(legacy) == {}

    confidence = torch.full(shape, 0.8)
    unknown = 1.0 - confidence
    evidence = StateEvidence(
        confidence=confidence,
        unknown_probability=unknown,
        completion_gate=unknown,
        provenance=torch.full((1, 2, 4, 4, 4), 0.5),
        base_density_logits=density,
        base_color=color,
        density_residual=torch.full(shape, 0.25),
        color_logit_residual=torch.full_like(color, 0.5),
    )
    state = SceneState(density, color, variance, bounds, evidence=evidence)

    unweighted = _state_regularization(state, evidence_residual_weight=0.0)
    weighted = _state_regularization(state, evidence_residual_weight=0.01)
    diagnostics = _state_diagnostics(state)

    assert weighted > unweighted
    assert diagnostics["state/evidence_mean"] == pytest.approx(0.8)
    assert diagnostics["state/unknown_mean"] == pytest.approx(0.2)
    assert diagnostics["state/completion_gate_mean"] == pytest.approx(0.2)
    assert diagnostics["state/density_residual_abs"] == pytest.approx(0.25)
    assert diagnostics["state/color_residual_abs"] == pytest.approx(0.5)
