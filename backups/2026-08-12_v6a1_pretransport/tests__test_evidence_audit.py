import math
from dataclasses import replace

import pytest
import torch

from mcss.evidence_audit import (
    audit_evidence,
    binary_ranking_metrics,
    classify_depth_samples,
    error_by_risk_deciles,
)
from mcss.geometry import look_at, make_intrinsics
from mcss.types import Cameras, DualEvidence, SceneState, StateEvidence


def _auditable_case() -> tuple[SceneState, Cameras, torch.Tensor, torch.Tensor]:
    resolution = 4
    shape = (1, 1, resolution, resolution, resolution)
    bounds = torch.tensor([[[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]]])
    confidence = torch.full(shape, 0.5)
    color = torch.full((1, 3, resolution, resolution, resolution), 0.5)
    evidence = StateEvidence(
        confidence=confidence,
        unknown_probability=1.0 - confidence,
        completion_gate=1.0 - confidence,
        provenance=torch.ones(shape),
        base_density_logits=torch.zeros(shape),
        base_color=color,
        density_residual=torch.zeros(shape),
        color_logit_residual=torch.zeros_like(color),
    )
    state = SceneState(
        density_logits=torch.zeros(shape),
        color=color,
        log_variance=torch.zeros(shape),
        bounds=bounds,
        evidence=evidence,
    )
    image_size = (2, 2)
    camera = Cameras(
        make_intrinsics(image_size, 20.0).view(1, 1, 3, 3),
        look_at(torch.tensor([0.0, 0.0, -2.0]), torch.zeros(3)).view(1, 1, 4, 4),
        image_size,
    )
    target_depth = torch.full((1, 1, 1, *image_size), 2.0)
    target_visibility = torch.ones_like(target_depth)
    return state, camera, target_depth, target_visibility


def test_depth_bins_use_inclusive_surface_band() -> None:
    distances = torch.tensor([[[1.0, 1.8, 2.0, 2.2, 3.0]]])
    target = torch.tensor([[[2.0]]])

    free, surface, behind = classify_depth_samples(
        distances,
        target,
        surface_band=torch.tensor([[[0.25]]]),
    )

    assert free.tolist() == [[[True, False, False, False, False]]]
    assert surface.tolist() == [[[False, True, True, True, False]]]
    assert behind.tolist() == [[[False, False, False, False, True]]]


def test_binary_ranking_metrics_are_exact_for_perfect_ordering() -> None:
    scores = torch.tensor([0.95, 0.8, 0.2, 0.05])
    labels = torch.tensor([True, True, False, False])

    metrics = binary_ranking_metrics(scores, labels, reliability_bins=5)

    assert metrics["auroc"] == pytest.approx(1.0)
    assert metrics["auprc"] == pytest.approx(1.0)
    assert metrics["brier"] == pytest.approx(float(((scores - labels.float()) ** 2).mean()))
    assert metrics["ece"] >= 0.0
    assert sum(item["count"] for item in metrics["reliability"]) == 4


def test_binary_metrics_clamp_only_roundoff_scale_probability_overshoot() -> None:
    scores = torch.tensor([-1e-7, 0.25, 0.75, 1.0 + 1e-7])
    labels = torch.tensor([False, False, True, True])

    metrics = binary_ranking_metrics(scores, labels)

    assert metrics["auroc"] == pytest.approx(1.0)
    with pytest.raises(ValueError, match="observed"):
        binary_ranking_metrics(torch.tensor([0.0, 1.001]), labels[:2])


def test_error_by_risk_deciles_detect_monotonic_geometric_error() -> None:
    risk = torch.linspace(0.0, 1.0, 100)
    absolute_error = 2.0 * risk
    absolute_relative_error = risk

    report = error_by_risk_deciles(
        risk,
        absolute_error,
        absolute_relative_error,
        bins=5,
    )

    assert report["count"] == 100
    assert len(report["deciles"]) == 5
    assert report["absolute_error_monotonic_violations"] == 0
    assert report["absolute_relative_error_monotonic_violations"] == 0
    assert report["absolute_error_spearman"] == pytest.approx(1.0)


def test_audit_evidence_returns_finite_surface_free_and_behind_statistics() -> None:
    state, cameras, target_depth, target_visibility = _auditable_case()

    report = audit_evidence(
        state,
        cameras,
        target_depth,
        target_visibility,
        n_samples=16,
    )

    assert report["schema_version"] == "mcss.evidence_audit.v1"
    assert report["surface_band_m"] == pytest.approx(0.25)
    assert report["rays"]["valid"] == 4
    assert report["bins"]["surface"]["count"] > 0
    assert report["bins"]["free"]["count"] > 0
    assert report["bins"]["behind"]["count"] > 0
    assert report["bins"]["surface"]["mean"] == pytest.approx(0.5)
    for comparison in ("surface_vs_free", "surface_vs_behind"):
        assert math.isfinite(report["comparisons"][comparison]["auroc"])
        assert math.isfinite(report["comparisons"][comparison]["auprc"])


def test_audit_rejects_missing_evidence_and_counts_invalid_target_rays() -> None:
    state, cameras, target_depth, target_visibility = _auditable_case()
    with pytest.raises(ValueError, match="StateEvidence"):
        audit_evidence(
            replace(state, evidence=None),
            cameras,
            target_depth,
            target_visibility,
        )

    target_depth[..., 0, 0] = 0.0
    target_visibility[..., 0, 1] = 0.0
    report = audit_evidence(
        state,
        cameras,
        target_depth,
        target_visibility,
        n_samples=16,
    )
    assert report["rays"]["invalid_target_depth"] == 1
    assert report["rays"]["target_invisible"] == 1
    assert report["rays"]["valid"] == 2


def test_audit_uses_dual_localization_support_and_reports_prediction_independent_risk() -> None:
    state, cameras, target_depth, target_visibility = _auditable_case()
    assert state.evidence is not None
    appearance = state.evidence.confidence
    peakness = torch.full_like(appearance, 0.5)
    localization = appearance * peakness
    dual = DualEvidence(
        surface_localization_support=localization,
        appearance_confidence=appearance,
        surface_peakness=peakness,
        localization_unknown_probability=1.0 - localization,
        appearance_unknown_probability=1.0 - appearance,
        localization_completion_gate=1.0 - localization,
        appearance_completion_gate=1.0 - appearance,
        provenance=state.evidence.provenance,
        base_density_logits=state.evidence.base_density_logits,
        base_color=state.evidence.base_color,
        density_residual=state.evidence.density_residual,
        color_logit_residual=state.evidence.color_logit_residual,
    )
    dual_state = replace(state, evidence=None, dual_evidence=dual)
    predicted_depth = target_depth + 0.25

    report = audit_evidence(
        dual_state,
        cameras,
        target_depth,
        target_visibility,
        predicted_depth=predicted_depth,
        n_samples=16,
    )

    assert report["evidence_field"] == "dual_evidence.surface_localization_support"
    assert report["bins"]["surface"]["mean"] == pytest.approx(0.25)
    assert report["error_by_risk"]["count"] == 4
    assert sum(item["count"] for item in report["error_by_risk"]["deciles"]) == 4


def test_audit_clamps_sampled_probability_roundoff_before_ray_risk() -> None:
    state, cameras, target_depth, target_visibility = _auditable_case()
    assert state.evidence is not None
    confidence = torch.ones_like(state.evidence.confidence)
    evidence = replace(
        state.evidence,
        confidence=confidence,
        unknown_probability=torch.zeros_like(confidence),
        completion_gate=torch.zeros_like(confidence),
    )

    report = audit_evidence(
        replace(state, evidence=evidence),
        cameras,
        target_depth,
        target_visibility,
        predicted_depth=target_depth,
        n_samples=16,
    )

    risk_report = report["error_by_risk"]
    assert risk_report is not None
    mean_risks = [
        item["mean_risk"]
        for item in risk_report["deciles"]
        if item["mean_risk"] is not None
    ]
    assert mean_risks == pytest.approx([0.0] * len(mean_risks))
