import torch
from torch import nn

from mcss.diagnostics import (
    intervention_specificity,
    measurement_generalization_matrix,
    no_bypass_audit,
    query_equivariance_error,
)
from mcss.measurements import FixedMeasurementRenderer


class _AuditableModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.renderer = FixedMeasurementRenderer(n_samples=4)
        self.encoder = nn.Linear(2, 2)

    def forward(self, context_rgb, context_cameras, target_cameras, bounds):
        raise NotImplementedError


def test_no_bypass_audit_accepts_parameter_free_renderer_and_label_free_signature() -> None:
    report = no_bypass_audit(_AuditableModel())

    assert report.passed
    assert report.renderer_parameter_count == 0
    assert report.forbidden_forward_arguments == ()


def test_query_equivariance_and_intervention_specificity_have_known_values() -> None:
    reference = {"depth": torch.tensor([[[[[1.0]]], [[[2.0]]]]])}
    permuted = {"depth": torch.tensor([[[[[2.0]]], [[[1.0]]]]])}
    error = query_equivariance_error(reference, permuted, torch.tensor([1, 0]))
    assert error["depth"] == 0.0

    baseline = {"rgb": torch.zeros(1), "depth": torch.zeros(1)}
    density_change = {"rgb": torch.zeros(1), "depth": torch.ones(1)}
    color_change = {"rgb": torch.ones(1), "depth": torch.zeros(1)}
    specificity = intervention_specificity(baseline, density_change, color_change)
    assert specificity["density_to_depth"] == 1.0
    assert specificity["color_to_rgb"] == 1.0
    assert specificity["cross_talk"] == 0.0


def test_measurement_matrix_keeps_train_and_heldout_axes() -> None:
    records = [
        {"trained": "rgb+depth", "evaluated": "rgb", "score": 20.0},
        {"trained": "rgb+depth", "evaluated": "normal", "score": 12.0},
    ]
    matrix = measurement_generalization_matrix(records)

    assert matrix["rgb+depth"]["rgb"] == 20.0
    assert matrix["rgb+depth"]["normal"] == 12.0
