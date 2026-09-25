import pytest
import torch

from mcss.engine import _state_diagnostics, _state_regularization
from mcss.geometry import look_at, make_intrinsics
from mcss.model.evidence_encoder import (
    _candidate_profile_telemetry,
    _cross_view_evidence,
    _normalized_center_peakness,
    _positive_candidate_excess,
    _surface_peakness,
    _surface_peakness_with_telemetry,
    _trilinear_splat_candidates,
)
from mcss.model.system import ModelConfig, build_model
from mcss.types import (
    Cameras,
    DualEvidence,
    SceneState,
    StateAppearance,
    StateEvidence,
    TransportEvidence,
)


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


def test_center_peakness_is_zero_for_flat_valid_profiles() -> None:
    scores = torch.zeros(1, 3, 5)
    valid = torch.ones_like(scores, dtype=torch.bool)

    peakness, contributes = _normalized_center_peakness(
        scores,
        valid,
        temperature=0.05,
    )

    torch.testing.assert_close(peakness, torch.zeros_like(peakness), atol=1e-6, rtol=0)
    assert contributes.all()


def test_center_peakness_uses_valid_candidate_count_and_rejects_invalid_centers() -> None:
    scores = torch.tensor(
        [[[0.0, 0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0, 0.0]]]
    )
    valid = torch.tensor(
        [[[True, False, True, True, False], [True, True, False, True, True]]]
    )

    peakness, contributes = _normalized_center_peakness(
        scores,
        valid,
        temperature=0.05,
    )

    assert peakness[0, 0] > 0.99
    assert contributes.tolist() == [[True, False]]
    assert peakness[0, 1] == 0.0


def test_candidate_profile_telemetry_distinguishes_flat_and_shifted_winners() -> None:
    scores = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.1, 0.2, 0.8, 0.0],
        ]
    )
    valid = torch.ones_like(scores, dtype=torch.bool)
    observer_count = torch.full_like(scores, 3, dtype=torch.int64)

    report = _candidate_profile_telemetry(
        scores,
        valid,
        observer_count,
        offsets=torch.tensor([-2, -1, 0, 1, 2]),
        temperature=0.05,
        appearance_confidence=torch.tensor([0.0, 1.0]),
        observer_capacity=3,
    )

    assert report["profiles"] == {
        "total": 2,
        "contributing": 2,
        "contributing_fraction": 1.0,
    }
    assert report["valid_candidate_count_histogram"] == {"5": 2}
    assert report["observer_count_histogram"] == {"3": 10}
    assert report["competition"]["center_argmax_inclusive_rate"] == pytest.approx(0.5)
    assert report["competition"]["center_unique_argmax_rate"] == pytest.approx(0.0)
    assert report["competition"]["center_argmax_fractional_rate"] == pytest.approx(0.1)
    assert report["competition"]["chance_argmax_rate"] == pytest.approx(0.2)
    assert report["competition"]["center_argmax_excess_over_chance"] == pytest.approx(-0.1)
    assert report["competition"]["unique_winner_fraction"] == pytest.approx(0.5)
    assert report["competition"]["winner_offset_histogram"] == {"1": 1}
    assert report["competition"]["noncenter_positive_excess_fraction"] == pytest.approx(1.0)
    assert report["competition"]["zero_positive_excess_profile_count"] == 1
    assert report["competition"]["score_range"]["p50"] == pytest.approx(0.4)
    assert report["competition"]["softmax_entropy_normalized"]["max"] == pytest.approx(1.0)
    assert report["strata"]["appearance_positive"]["profiles"] == 1
    assert report["strata"]["appearance_positive"]["competition"][
        "winner_offset_histogram"
    ] == {"1": 1}
    assert report["strata"]["appearance_top_decile"]["profiles"] == 1
    assert report["counterfactuals"]["valid_observer_mean"]["competition"][
        "center_argmax_fractional_rate"
    ] == pytest.approx(0.1)
    assert report["counterfactuals"]["min_two_observers"]["profiles"][
        "contributing"
    ] == 2


def test_candidate_profile_telemetry_handles_no_contributing_profiles() -> None:
    scores = torch.zeros(2, 5)
    valid = torch.zeros_like(scores, dtype=torch.bool)
    observer_count = torch.zeros_like(scores, dtype=torch.int64)

    report = _candidate_profile_telemetry(
        scores,
        valid,
        observer_count,
        offsets=torch.tensor([-2, -1, 0, 1, 2]),
        temperature=0.05,
    )

    assert report["profiles"]["contributing"] == 0
    assert report["competition"] == {}


def test_surface_peakness_telemetry_preserves_the_production_result() -> None:
    context_rgb, context_cameras, _, bounds = _model_inputs()
    appearance_confidence = torch.linspace(0.0, 1.0, 4**3).reshape(1, 1, 4, 4, 4)
    kwargs = {
        "evidence_temperature": 0.1,
        "surface_peak_temperature": 0.05,
        "chunk_size": 16,
    }

    expected = _surface_peakness(
        context_rgb,
        context_cameras,
        bounds,
        (4, 4, 4),
        **kwargs,
    )
    actual, report = _surface_peakness_with_telemetry(
        context_rgb,
        context_cameras,
        bounds,
        (4, 4, 4),
        appearance_confidence=appearance_confidence,
        **kwargs,
    )

    torch.testing.assert_close(actual, expected)
    assert report["profiles"]["total"] == 2 * 4 * 4 * 4
    assert sum(report["valid_candidate_count_histogram"].values()) == 2 * 4 * 4 * 4
    assert report["surface_peakness"]["count"] == 4 * 4 * 4
    assert "appearance_positive" in report["strata"]
    assert "appearance_top_decile" in report["strata"]


def test_trilinear_splat_places_exact_center_mass_in_one_voxel() -> None:
    points = torch.tensor([[[[0.5, 0.5, 0.5]]]])
    values = torch.tensor([[[2.0]]])
    valid = torch.ones_like(values, dtype=torch.bool)
    bounds = torch.tensor([[[0.0, 0.0, 0.0], [2.0, 2.0, 2.0]]])

    volume = _trilinear_splat_candidates(points, values, valid, bounds, (2, 2, 2))

    expected = torch.zeros(1, 8)
    expected[0, 0] = 2.0
    torch.testing.assert_close(volume, expected)


def test_trilinear_splat_splits_midpoint_mass_across_eight_voxels() -> None:
    points = torch.tensor([[[[1.0, 1.0, 1.0]]]])
    values = torch.ones(1, 1, 1)
    valid = torch.ones_like(values, dtype=torch.bool)
    bounds = torch.tensor([[[0.0, 0.0, 0.0], [2.0, 2.0, 2.0]]])

    volume = _trilinear_splat_candidates(points, values, valid, bounds, (2, 2, 2))

    torch.testing.assert_close(volume, torch.full((1, 8), 0.125))
    assert volume.sum().item() == pytest.approx(1.0)


def test_trilinear_splat_renormalizes_mass_at_the_aabb_boundary() -> None:
    points = torch.tensor([[[[0.0, 0.0, 0.0]]]])
    values = torch.tensor([[[3.0]]])
    valid = torch.ones_like(values, dtype=torch.bool)
    bounds = torch.tensor([[[0.0, 0.0, 0.0], [2.0, 2.0, 2.0]]])

    volume = _trilinear_splat_candidates(points, values, valid, bounds, (2, 2, 2))

    expected = torch.zeros(1, 8)
    expected[0, 0] = 3.0
    torch.testing.assert_close(volume, expected)
    assert volume.sum().item() == pytest.approx(3.0)


def test_trilinear_splat_reuses_weights_for_multiple_value_channels() -> None:
    points = torch.tensor([[[[0.5, 0.5, 0.5]]]])
    values = torch.tensor([[[[2.0, 3.0]]]])
    valid = torch.ones(1, 1, 1, dtype=torch.bool)
    bounds = torch.tensor([[[0.0, 0.0, 0.0], [2.0, 2.0, 2.0]]])

    volume = _trilinear_splat_candidates(points, values, valid, bounds, (2, 2, 2))

    expected = torch.zeros(1, 2, 8)
    expected[0, :, 0] = torch.tensor([2.0, 3.0])
    torch.testing.assert_close(volume, expected)


def test_positive_candidate_excess_is_zero_for_flat_and_tracks_offcenter_peak() -> None:
    scores = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0],
        ]
    )
    valid = torch.ones_like(scores, dtype=torch.bool)

    excess, contributes = _positive_candidate_excess(scores, valid, temperature=0.05)

    torch.testing.assert_close(excess[0], torch.zeros(5), atol=1e-6, rtol=0)
    assert excess[1, 3] > 0.99
    assert excess[1, 2] == 0.0
    assert contributes.tolist() == [True, True]


def test_positive_excess_is_splatted_to_the_offcenter_candidate_location() -> None:
    points = torch.tensor(
        [
            [
                [
                    [0.5, 0.5, 0.5],
                    [0.5, 0.5, 0.5],
                    [0.5, 0.5, 0.5],
                    [1.5, 0.5, 0.5],
                    [0.5, 0.5, 0.5],
                ]
            ]
        ]
    )
    scores = torch.tensor([[[0.0, 0.0, 0.0, 1.0, 0.0]]])
    valid = torch.ones_like(scores, dtype=torch.bool)
    bounds = torch.tensor([[[0.0, 0.0, 0.0], [3.0, 1.0, 1.0]]])
    excess, contributes = _positive_candidate_excess(
        scores, valid, temperature=0.05
    )

    transported = _trilinear_splat_candidates(
        points,
        excess,
        valid & contributes[..., None],
        bounds,
        (1, 1, 3),
    )

    assert transported[0, 0].item() == pytest.approx(0.0, abs=1e-6)
    assert transported[0, 1].item() > 0.99


def _transport_model() -> torch.nn.Module:
    return build_model(
        ModelConfig(
            mode="fixed",
            state_architecture="dual_evidence_transport",
            voxel_resolution=4,
            image_feature_dim=8,
            state_feature_dim=8,
            refinement_blocks=1,
            evidence_temperature=0.1,
            surface_peak_temperature=0.05,
            observed_residual_floor=0.2,
            completion_residual_scale=1.5,
            appearance_resolution_scale=2,
            n_samples=4,
            ray_chunk_size=64,
        )
    )


def test_transport_architecture_keeps_parameters_and_requires_two_other_observers() -> None:
    v5 = build_model(
        ModelConfig(
            mode="fixed",
            state_architecture="evidence_residual",
            voxel_resolution=4,
            image_feature_dim=8,
            state_feature_dim=8,
            refinement_blocks=1,
            appearance_resolution_scale=2,
            n_samples=4,
            ray_chunk_size=64,
        )
    )
    model = _transport_model()

    output = model(*_model_inputs())
    evidence = output.state.transport_evidence

    assert set(model.state_dict()) == set(v5.state_dict())
    assert evidence is not None
    assert output.state.dual_evidence is None
    assert output.state.evidence is None
    torch.testing.assert_close(
        evidence.surface_localization_support,
        evidence.transported_appearance_confidence * evidence.surface_peakness,
    )
    torch.testing.assert_close(
        evidence.surface_localization_support,
        torch.zeros_like(evidence.surface_localization_support),
    )
    torch.testing.assert_close(output.state.density_logits, evidence.density_logits)
    torch.testing.assert_close(output.state.color, evidence.color)
    assert not evidence.surface_localization_support.requires_grad


def test_transport_evidence_is_invariant_to_context_view_permutation() -> None:
    model = _transport_model().eval()
    context_rgb, context_cameras, target_cameras, bounds = _model_inputs()
    context_rgb = torch.cat((context_rgb, context_rgb[:, :1]), dim=1)
    context_cameras = Cameras(
        torch.cat((context_cameras.intrinsics, context_cameras.intrinsics[:, :1]), dim=1),
        torch.cat((context_cameras.c2w, context_cameras.c2w[:, :1]), dim=1),
        context_cameras.image_size,
    )
    permutation = torch.tensor([2, 0, 1])

    original = model(context_rgb, context_cameras, target_cameras, bounds).state
    permuted = model(
        context_rgb[:, permutation],
        context_cameras.select_views(permutation),
        target_cameras,
        bounds,
    ).state

    assert original.transport_evidence is not None
    assert permuted.transport_evidence is not None
    torch.testing.assert_close(
        original.transport_evidence.surface_localization_support,
        permuted.transport_evidence.surface_localization_support,
        atol=1e-5,
        rtol=1e-5,
    )
    torch.testing.assert_close(
        original.transport_evidence.transport_coverage,
        permuted.transport_evidence.transport_coverage,
        atol=1e-5,
        rtol=1e-5,
    )
    torch.testing.assert_close(
        original.transport_evidence.provenance[:, permutation],
        permuted.transport_evidence.provenance,
        atol=1e-5,
        rtol=1e-5,
    )


def test_transport_evidence_cannot_change_across_an_optimizer_step() -> None:
    model = _transport_model()
    context_rgb, context_cameras, target_cameras, bounds = _model_inputs()
    context_rgb = torch.cat((context_rgb, context_rgb[:, :1]), dim=1)
    context_cameras = Cameras(
        torch.cat((context_cameras.intrinsics, context_cameras.intrinsics[:, :1]), dim=1),
        torch.cat((context_cameras.c2w, context_cameras.c2w[:, :1]), dim=1),
        context_cameras.image_size,
    )
    inputs = (context_rgb, context_cameras, target_cameras, bounds)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)

    before = model(*inputs)
    assert before.state.transport_evidence is not None
    frozen = {
        "localization": before.state.transport_evidence.surface_localization_support.clone(),
        "transported_appearance": (
            before.state.transport_evidence.transported_appearance_confidence.clone()
        ),
        "peakness": before.state.transport_evidence.surface_peakness.clone(),
        "coverage": before.state.transport_evidence.transport_coverage.clone(),
        "appearance": before.state.transport_evidence.appearance_confidence.clone(),
        "provenance": before.state.transport_evidence.provenance.clone(),
    }
    loss = sum(value.mean() for value in before.predictions.values())
    loss.backward()
    optimizer.step()
    after = model(*inputs)

    assert after.state.transport_evidence is not None
    torch.testing.assert_close(
        after.state.transport_evidence.surface_localization_support,
        frozen["localization"],
    )
    torch.testing.assert_close(
        after.state.transport_evidence.transported_appearance_confidence,
        frozen["transported_appearance"],
    )
    torch.testing.assert_close(
        after.state.transport_evidence.surface_peakness,
        frozen["peakness"],
    )
    torch.testing.assert_close(
        after.state.transport_evidence.transport_coverage,
        frozen["coverage"],
    )
    torch.testing.assert_close(
        after.state.transport_evidence.appearance_confidence,
        frozen["appearance"],
    )
    torch.testing.assert_close(
        after.state.transport_evidence.provenance,
        frozen["provenance"],
    )


def test_dual_evidence_architecture_keeps_v5_parameter_contract() -> None:
    common = dict(
        mode="fixed",
        voxel_resolution=4,
        image_feature_dim=8,
        state_feature_dim=8,
        refinement_blocks=1,
        appearance_resolution_scale=2,
        n_samples=4,
        ray_chunk_size=64,
    )
    v5 = build_model(ModelConfig(state_architecture="evidence_residual", **common))
    v6a = build_model(
        ModelConfig(
            state_architecture="dual_evidence",
            surface_peak_temperature=0.05,
            **common,
        )
    )

    assert set(v6a.state_dict()) == set(v5.state_dict())
    load_result = v5.load_state_dict(v5.state_dict(), strict=True)
    assert load_result.missing_keys == []
    assert load_result.unexpected_keys == []


def _dual_evidence_model() -> torch.nn.Module:
    return build_model(
        ModelConfig(
            mode="fixed",
            state_architecture="dual_evidence",
            voxel_resolution=4,
            image_feature_dim=8,
            state_feature_dim=8,
            refinement_blocks=1,
            evidence_temperature=0.1,
            surface_peak_temperature=0.05,
            observed_residual_floor=0.2,
            completion_residual_scale=1.5,
            appearance_resolution_scale=2,
            n_samples=4,
            ray_chunk_size=64,
        )
    )


def test_dual_evidence_forward_uses_separate_density_and_appearance_gates() -> None:
    model = _dual_evidence_model()

    output = model(*_model_inputs())
    evidence = output.state.dual_evidence

    assert evidence is not None
    assert output.state.evidence is None
    assert output.state.appearance is not None
    assert evidence.surface_localization_support.shape == (1, 1, 4, 4, 4)
    assert evidence.appearance_confidence.shape == (1, 1, 4, 4, 4)
    assert evidence.surface_peakness.shape == (1, 1, 4, 4, 4)
    assert not evidence.surface_localization_support.requires_grad
    assert not evidence.appearance_confidence.requires_grad
    assert not evidence.surface_peakness.requires_grad
    torch.testing.assert_close(
        evidence.surface_localization_support,
        evidence.appearance_confidence * evidence.surface_peakness,
    )
    torch.testing.assert_close(output.state.density_logits, evidence.density_logits)
    torch.testing.assert_close(output.state.color, evidence.color)
    assert (evidence.localization_completion_gate >= 0.2).all()
    assert (evidence.appearance_completion_gate >= 0.2).all()
    assert all(torch.isfinite(value).all() for value in output.predictions.values())

    loss = sum(value.mean() for value in output.predictions.values())
    loss.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.requires_grad]
    assert any(gradient is not None for gradient in gradients)
    assert all(torch.isfinite(gradient).all() for gradient in gradients if gradient is not None)


def test_dual_evidence_is_invariant_to_context_view_permutation() -> None:
    model = _dual_evidence_model().eval()
    context_rgb, context_cameras, target_cameras, bounds = _model_inputs()
    permutation = torch.tensor([1, 0])

    original = model(context_rgb, context_cameras, target_cameras, bounds).state
    permuted = model(
        context_rgb[:, permutation],
        context_cameras.select_views(permutation),
        target_cameras,
        bounds,
    ).state

    assert original.dual_evidence is not None
    assert permuted.dual_evidence is not None
    torch.testing.assert_close(
        original.dual_evidence.surface_localization_support,
        permuted.dual_evidence.surface_localization_support,
        atol=1e-5,
        rtol=1e-5,
    )
    torch.testing.assert_close(
        original.dual_evidence.appearance_confidence,
        permuted.dual_evidence.appearance_confidence,
        atol=1e-5,
        rtol=1e-5,
    )
    torch.testing.assert_close(
        original.dual_evidence.surface_peakness,
        permuted.dual_evidence.surface_peakness,
        atol=1e-5,
        rtol=1e-5,
    )
    torch.testing.assert_close(
        original.dual_evidence.provenance[:, permutation],
        permuted.dual_evidence.provenance,
        atol=1e-5,
        rtol=1e-5,
    )


def test_dual_evidence_cannot_change_across_an_optimizer_step() -> None:
    model = _dual_evidence_model()
    inputs = _model_inputs()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)

    before = model(*inputs)
    assert before.state.dual_evidence is not None
    frozen = {
        "localization": before.state.dual_evidence.surface_localization_support.clone(),
        "appearance": before.state.dual_evidence.appearance_confidence.clone(),
        "peakness": before.state.dual_evidence.surface_peakness.clone(),
        "provenance": before.state.dual_evidence.provenance.clone(),
    }
    loss = sum(value.mean() for value in before.predictions.values())
    loss.backward()
    optimizer.step()
    after = model(*inputs)

    assert after.state.dual_evidence is not None
    torch.testing.assert_close(
        after.state.dual_evidence.surface_localization_support,
        frozen["localization"],
    )
    torch.testing.assert_close(
        after.state.dual_evidence.appearance_confidence,
        frozen["appearance"],
    )
    torch.testing.assert_close(after.state.dual_evidence.surface_peakness, frozen["peakness"])
    torch.testing.assert_close(after.state.dual_evidence.provenance, frozen["provenance"])


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
    assert output.state.appearance is None
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


def test_highres_appearance_is_auditable_reconstructable_and_differentiable() -> None:
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
        appearance_resolution_scale=2,
        n_samples=8,
        ray_chunk_size=128,
    )
    model = build_model(config)

    output = model(*_model_inputs())
    appearance = output.state.appearance

    assert appearance is not None
    assert output.state.spatial_shape == (8, 8, 8)
    assert output.state.appearance_shape == (16, 16, 16)
    assert appearance.confidence.shape == (1, 1, 16, 16, 16)
    assert appearance.provenance.shape == (1, 2, 16, 16, 16)
    assert appearance.base_color.shape == (1, 3, 16, 16, 16)
    assert (appearance.completion_gate >= 0.2).all()
    assert (appearance.completion_gate <= 1.0).all()
    assert appearance.color_logit_residual.abs().max() <= 1.5
    torch.testing.assert_close(
        appearance.color_logit_residual,
        torch.zeros_like(appearance.color_logit_residual),
    )
    expected_color = torch.sigmoid(torch.logit(appearance.base_color.clamp(1e-4, 1 - 1e-4)))
    torch.testing.assert_close(appearance.color, expected_color)
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
        appearance_resolution_scale=2,
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
    assert before.state.appearance is not None
    appearance_confidence = before.state.appearance.confidence.detach().clone()
    appearance_provenance = before.state.appearance.provenance.detach().clone()
    appearance_base_color = before.state.appearance.base_color.detach().clone()
    loss = sum(value.mean() for value in before.predictions.values())
    loss.backward()
    optimizer.step()
    after = model(*inputs)

    assert after.state.evidence is not None
    assert after.state.appearance is not None
    torch.testing.assert_close(after.state.evidence.confidence, confidence)
    torch.testing.assert_close(after.state.evidence.provenance, provenance)
    torch.testing.assert_close(after.state.appearance.confidence, appearance_confidence)
    torch.testing.assert_close(after.state.appearance.provenance, appearance_provenance)
    torch.testing.assert_close(after.state.appearance.base_color, appearance_base_color)


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

    appearance_confidence = torch.full((1, 1, 8, 8, 8), 0.75)
    appearance = StateAppearance(
        confidence=appearance_confidence,
        unknown_probability=1.0 - appearance_confidence,
        completion_gate=torch.full_like(appearance_confidence, 0.325),
        provenance=torch.full((1, 2, 8, 8, 8), 0.5),
        base_color=torch.full((1, 3, 8, 8, 8), 0.5),
        color_logit_residual=torch.ones(1, 3, 8, 8, 8),
    )
    appearance_state = SceneState(
        density,
        color,
        variance,
        bounds,
        evidence=evidence,
        appearance=appearance,
    )

    appearance_weighted = _state_regularization(
        appearance_state, evidence_residual_weight=0.01
    )
    appearance_diagnostics = _state_diagnostics(appearance_state)

    expected_rewrite = torch.tensor(0.01 * (0.8 * 0.25 + 0.75 * 1.0))
    torch.testing.assert_close(appearance_weighted, unweighted + expected_rewrite)
    assert appearance_diagnostics["state/appearance_evidence_mean"] == pytest.approx(0.75)
    assert appearance_diagnostics["state/appearance_unknown_mean"] == pytest.approx(0.25)
    assert appearance_diagnostics["state/appearance_completion_gate_mean"] == pytest.approx(0.325)
    assert appearance_diagnostics["state/appearance_color_residual_abs"] == pytest.approx(1.0)
    assert appearance_diagnostics["state/appearance_resolution_scale"] == pytest.approx(2.0)


def test_dual_evidence_regularization_uses_localization_for_density_and_appearance_for_color(
) -> None:
    shape = (1, 1, 4, 4, 4)
    localization = torch.full(shape, 0.2)
    appearance_confidence = torch.full(shape, 0.8)
    peakness = localization / appearance_confidence
    base_color = torch.full((1, 3, 4, 4, 4), 0.5)
    dual = DualEvidence(
        surface_localization_support=localization,
        appearance_confidence=appearance_confidence,
        surface_peakness=peakness,
        localization_unknown_probability=1.0 - localization,
        appearance_unknown_probability=1.0 - appearance_confidence,
        localization_completion_gate=1.0 - localization,
        appearance_completion_gate=1.0 - appearance_confidence,
        provenance=torch.full((1, 2, 4, 4, 4), 0.5),
        base_density_logits=torch.zeros(shape),
        base_color=base_color,
        density_residual=torch.full(shape, 0.5),
        color_logit_residual=torch.ones_like(base_color),
    )
    state = SceneState(
        density_logits=dual.density_logits,
        color=dual.color,
        log_variance=torch.zeros(shape),
        bounds=torch.tensor([[[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]]]),
        dual_evidence=dual,
    )

    unweighted = _state_regularization(state, evidence_residual_weight=0.0)
    weighted = _state_regularization(state, evidence_residual_weight=0.01)
    diagnostics = _state_diagnostics(state)

    expected_rewrite = torch.tensor(0.01 * (0.2 * 0.5 + 0.8 * 1.0))
    torch.testing.assert_close(weighted, unweighted + expected_rewrite)
    assert diagnostics["state/surface_localization_support_mean"] == pytest.approx(0.2)
    assert diagnostics["state/appearance_confidence_mean"] == pytest.approx(0.8)
    assert diagnostics["state/surface_peakness_mean"] == pytest.approx(0.25)
    assert diagnostics["state/surface_localization_support_p50"] == pytest.approx(0.2)
    assert diagnostics["state/appearance_confidence_p50"] == pytest.approx(0.8)
    assert diagnostics["state/surface_peakness_p50"] == pytest.approx(0.25)
    assert diagnostics["state/surface_peakness_nonzero_fraction"] == pytest.approx(1.0)
    assert diagnostics["state/localization_completion_gate_mean"] == pytest.approx(0.8)
    assert diagnostics["state/appearance_completion_gate_mean"] == pytest.approx(0.2)


def test_transport_regularization_and_diagnostics_use_transported_localization() -> None:
    shape = (1, 1, 4, 4, 4)
    localization = torch.full(shape, 0.2)
    transported_appearance = torch.full(shape, 0.4)
    appearance_confidence = torch.full(shape, 0.8)
    base_color = torch.full((1, 3, 4, 4, 4), 0.5)
    transport = TransportEvidence(
        surface_localization_support=localization,
        transported_appearance_confidence=transported_appearance,
        surface_peakness=localization / transported_appearance,
        transport_coverage=torch.full(shape, 3.0),
        appearance_confidence=appearance_confidence,
        localization_unknown_probability=1.0 - localization,
        appearance_unknown_probability=1.0 - appearance_confidence,
        localization_completion_gate=1.0 - localization,
        appearance_completion_gate=1.0 - appearance_confidence,
        provenance=torch.full((1, 2, 4, 4, 4), 0.5),
        base_density_logits=torch.zeros(shape),
        base_color=base_color,
        density_residual=torch.full(shape, 0.5),
        color_logit_residual=torch.ones_like(base_color),
    )
    state = SceneState(
        density_logits=transport.density_logits,
        color=transport.color,
        log_variance=torch.zeros(shape),
        bounds=torch.tensor([[[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]]]),
        transport_evidence=transport,
    )

    unweighted = _state_regularization(state, evidence_residual_weight=0.0)
    weighted = _state_regularization(state, evidence_residual_weight=0.01)
    diagnostics = _state_diagnostics(state)

    expected_rewrite = torch.tensor(0.01 * (0.2 * 0.5 + 0.8 * 1.0))
    torch.testing.assert_close(weighted, unweighted + expected_rewrite)
    assert diagnostics["state/surface_localization_support_mean"] == pytest.approx(0.2)
    assert diagnostics["state/transported_appearance_confidence_mean"] == pytest.approx(0.4)
    assert diagnostics["state/surface_peakness_mean"] == pytest.approx(0.5)
    assert diagnostics["state/transport_coverage_mean"] == pytest.approx(3.0)
    assert diagnostics["state/transport_coverage_nonzero_fraction"] == pytest.approx(1.0)
    assert diagnostics["state/localization_density_rewrite_mean"] == pytest.approx(0.1)
