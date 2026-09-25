import math

import torch

from mcss.losses import MeasurementLoss
from mcss.metrics import compute_metrics


def _targets() -> dict[str, torch.Tensor]:
    return {
        "rgb": torch.full((1, 1, 3, 4, 4), 0.5),
        "depth": torch.full((1, 1, 1, 4, 4), 2.0),
        "normal": torch.tensor([0.0, 0.0, 1.0]).view(1, 1, 3, 1, 1).expand(1, 1, 3, 4, 4),
        "point": torch.zeros(1, 1, 3, 4, 4),
        "visibility": torch.ones(1, 1, 1, 4, 4),
        "uncertainty": torch.full((1, 1, 1, 4, 4), 0.1),
    }


def test_perfect_predictions_have_zero_geometry_error() -> None:
    targets = _targets()
    metrics = compute_metrics(targets, targets)

    assert math.isinf(metrics["rgb/psnr"])
    assert metrics["depth/abs_rel"] == 0.0
    assert metrics["depth/rmse"] == 0.0
    assert metrics["depth/delta1"] == 1.0
    assert metrics["normal/mean_angle"] < 0.03
    assert metrics["visibility/iou"] == 1.0
    assert metrics["visibility/f1"] == 1.0


def test_measurement_loss_masks_invalid_depth_and_is_differentiable() -> None:
    targets = _targets()
    targets["depth"][..., 0, 0] = 0.0
    predictions = {name: value.clone().requires_grad_(True) for name, value in targets.items()}
    predictions["depth"].data[..., 0, 0] = 100.0
    objective = MeasurementLoss({"rgb": 1.0, "depth": 1.0, "normal": 1.0, "visibility": 1.0})

    loss, terms = objective(predictions, targets)

    assert terms["depth"] == 0.0
    loss.backward()
    assert predictions["depth"].grad is not None


def test_visibility_loss_is_safe_inside_cuda_autocast() -> None:
    if not torch.cuda.is_available():
        return
    prediction = torch.full((1, 1, 1, 2, 2), 0.75, device="cuda", requires_grad=True)
    target = torch.ones_like(prediction)
    objective = MeasurementLoss({"visibility": 1.0})

    with torch.autocast(device_type="cuda", enabled=True):
        loss, terms = objective({"visibility": prediction}, {"visibility": target})
    loss.backward()

    assert torch.isfinite(loss)
    assert torch.isfinite(terms["visibility"])
    assert prediction.grad is not None


def test_support_mask_excludes_out_of_support_errors_from_every_loss() -> None:
    targets = _targets()
    support = torch.zeros_like(targets["visibility"])
    support[..., 0, 0] = 1.0
    targets["support"] = support
    predictions = {name: value.clone() for name, value in targets.items() if name != "support"}
    outside = ~support.bool()
    predictions["rgb"] = torch.where(outside.expand_as(predictions["rgb"]), 0.0, predictions["rgb"])
    predictions["depth"] = torch.where(outside, 100.0, predictions["depth"])
    predictions["normal"] = torch.where(
        outside.expand_as(predictions["normal"]),
        -predictions["normal"],
        predictions["normal"],
    )
    predictions["point"] = torch.where(
        outside.expand_as(predictions["point"]), 100.0, predictions["point"]
    )
    predictions["visibility"] = torch.where(outside, 0.0, predictions["visibility"])
    objective = MeasurementLoss(
        {"rgb": 1.0, "depth": 1.0, "normal": 1.0, "point": 1.0, "visibility": 1.0}
    )
    baseline, _ = objective(
        {name: target for name, target in targets.items() if name != "support"}, targets
    )

    loss, terms = objective(predictions, targets)

    torch.testing.assert_close(loss, baseline)
    assert terms["rgb"] < 0.002
    assert terms["depth"] == 0.0
    assert terms["normal"] == 0.0
    assert terms["point"] == 0.0


def test_metrics_report_support_coverage_counts_and_metric_point_scales() -> None:
    targets = _targets()
    support = torch.zeros_like(targets["visibility"])
    support[..., 0, 0] = 1.0
    targets["support"] = support
    predictions = {name: value.clone() for name, value in targets.items() if name != "support"}
    outside = ~support.bool()
    predictions["rgb"] = torch.where(outside.expand_as(predictions["rgb"]), 0.0, predictions["rgb"])
    predictions["depth"] = torch.where(outside, 100.0, predictions["depth"])
    predictions["point"] = torch.where(
        outside.expand_as(predictions["point"]), 100.0, predictions["point"]
    )
    bounds = torch.tensor([[[-4.0, -3.0, 0.1], [4.0, 3.0, 8.1]]])

    metrics = compute_metrics(predictions, targets, bounds=bounds)

    assert metrics["support/coverage"] == 1.0 / 16.0
    assert metrics["support/valid_pixels"] == 1.0
    assert metrics["support/total_pixels"] == 16.0
    assert metrics["depth/valid_pixels"] == 1.0
    assert metrics["depth/rmse"] == 0.0
    assert metrics["point/valid_points"] == 1.0
    assert metrics["point/rmse_m"] == 0.0
    assert metrics["point/nrmse_support_diag"] == 0.0
    assert metrics["point/fscore_0.05"] == 1.0
    assert metrics["point/fscore_0.25"] == 1.0
    assert metrics["point/fscore_0.50"] == 1.0


def test_ssim_distinguishes_local_structure_with_identical_global_moments() -> None:
    target = torch.full((1, 1, 3, 16, 16), 0.5)
    y, x = torch.meshgrid(torch.arange(16), torch.arange(16), indexing="ij")
    checkerboard = ((x + y) % 2).float()
    split = torch.zeros(16, 16)
    split[:8] = 1.0
    checkerboard = checkerboard.view(1, 1, 1, 16, 16).expand_as(target)
    split = split.view(1, 1, 1, 16, 16).expand_as(target)
    torch.testing.assert_close(checkerboard.mean(), split.mean())
    torch.testing.assert_close(checkerboard.var(unbiased=False), split.var(unbiased=False))

    checkerboard_ssim = compute_metrics({"rgb": checkerboard}, {"rgb": target})["rgb/ssim"]
    split_ssim = compute_metrics({"rgb": split}, {"rgb": target})["rgb/ssim"]

    assert not math.isclose(checkerboard_ssim, split_ssim, abs_tol=1e-3)


def test_point_cloud_metrics_sample_errors_across_the_full_point_map() -> None:
    target = torch.zeros(1, 1, 3, 64, 64)
    prediction = target.clone()
    prediction[:, :, 2, 32:, :] = 2.0
    visibility = torch.ones(1, 1, 1, 64, 64)

    metrics = compute_metrics(
        {"point": prediction},
        {"point": target, "visibility": visibility},
    )

    assert metrics["point/valid_points"] == 4096.0
    assert metrics["point/chamfer_m2"] > 0.5
    assert metrics["point/fscore_0.50"] < 0.9
