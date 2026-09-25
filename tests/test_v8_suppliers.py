from pathlib import Path

import numpy as np
import pytest
import torch

from mcss import v8_suppliers
from mcss.geometry_supplier import GeometrySupplierRequest
from mcss.types import Cameras
from mcss.v8_suppliers import EMVSNetSupplier, UniDepthV2SmallSupplier


def _request() -> GeometrySupplierRequest:
    rgb = torch.rand(4, 3, 8, 10)
    K = torch.tensor([[[8.0, 0.0, 4.5], [0.0, 8.0, 3.5], [0.0, 0.0, 1.0]]]).repeat(4, 1, 1)
    c2w = torch.eye(4).repeat(4, 1, 1)
    c2w[:, 0, 3] = torch.arange(4, dtype=torch.float32) * 0.1
    return GeometrySupplierRequest((4, 1, 3, 2), rgb, Cameras(K, c2w, (8, 10)))


class _FakeUniDepth:
    def parameters(self):
        return iter(())

    def infer(self, rgb, K):
        assert tuple(rgb.shape) == (3, 8, 10)
        assert tuple(K.shape) == (3, 3)
        return {
            "depth": torch.ones(1, 1, 8, 10),
            "points": torch.ones(1, 3, 8, 10),
            "confidence": torch.full((1, 1, 8, 10), 0.7),
        }

    def to(self, _device):
        return self

    def eval(self):
        return self


class _FakeEMVSNet:
    def __call__(self, images, projection, depth_values):
        del projection
        batch, _views, _channels, height, width = images.shape
        depth_bins = depth_values.shape[1]
        primary = torch.zeros(batch, depth_bins, height, width)
        auxiliary = torch.zeros_like(primary)
        primary[:, 7] = 1.0
        auxiliary[:, 19] = 1.0
        evidential = torch.zeros(batch, 4, height, width)
        evidential[:, 0] = depth_values[:, 31, None, None]
        evidential[:, 1] = 2.0
        evidential[:, 2] = 3.0
        evidential[:, 3] = 4.0
        return primary, evidential, auxiliary


def test_unidepth_adapter_preserves_context_views_and_shapes(tmp_path: Path) -> None:
    supplier = UniDepthV2SmallSupplier(code_root=tmp_path, device="cpu")
    supplier._model = _FakeUniDepth()
    result = supplier.infer(_request())
    assert result.frame_ids == (1, 2, 3, 4)
    assert result.depth_z_m.shape == (4, 8, 10)
    assert result.points_cam_m.shape == (4, 8, 10, 3)
    assert result.unit_provenance == "metric_m"


def test_emvsnet_requires_frozen_64_bin_checkpoint(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="64-bin"):
        EMVSNetSupplier(code_root=tmp_path, checkpoint=tmp_path / "x.ckpt", depth_bins=128)


def test_emvsnet_center_normalization_is_per_image_and_channel() -> None:
    base = torch.arange(24, dtype=torch.float32).reshape(1, 3, 2, 4)
    images = torch.cat([base, base * 3.0 + 17.0], dim=0)
    normalized = v8_suppliers.center_emvsnet_images(images)
    assert torch.allclose(normalized.mean(dim=(-2, -1)), torch.zeros(2, 3), atol=1e-6)
    assert torch.allclose(
        normalized.var(dim=(-2, -1), correction=0), torch.ones(2, 3), atol=1e-5
    )


def test_emvsnet_depth_hypotheses_are_frozen_and_label_independent(tmp_path: Path) -> None:
    supplier = EMVSNetSupplier(code_root=tmp_path, checkpoint=tmp_path / "x.ckpt")
    values = supplier.depth_hypotheses_m(device=torch.device("cpu"))
    assert values.shape == (64,)
    assert values[0].item() == pytest.approx(v8_suppliers.EMVSNET_DEPTH_NEAR_M)
    assert values[-1].item() == pytest.approx(v8_suppliers.EMVSNET_DEPTH_FAR_M)


def test_emvsnet_projection_is_equivariant_to_joint_metric_scaling() -> None:
    request = _request().canonical()
    depth_m = torch.tensor([0.2, 1.5, 7.0], dtype=torch.float32)
    projection_m = v8_suppliers.emvsnet_projection_coordinates(
        request.cameras, translation_scale=1.0
    )
    projection_mm = v8_suppliers.emvsnet_projection_coordinates(
        request.cameras, translation_scale=1000.0
    )
    relative_m = projection_m[1] @ torch.linalg.inv(projection_m[0])
    relative_mm = projection_mm[1] @ torch.linalg.inv(projection_mm[0])

    pixel = torch.tensor([3.0, 2.0, 1.0], dtype=torch.float32)
    points_m = relative_m[:3, :3] @ pixel[:, None] * depth_m[None]
    points_m += relative_m[:3, 3:4]
    points_mm = relative_mm[:3, :3] @ pixel[:, None] * (depth_m * 1000.0)[None]
    points_mm += relative_mm[:3, 3:4]
    assert torch.allclose(
        points_m[:2] / points_m[2:3], points_mm[:2] / points_mm[2:3], atol=1e-5
    )


def test_emvsnet_der_risk_uses_official_epistemic_formula() -> None:
    evidential = torch.zeros(1, 4, 1, 1)
    evidential[:, 1] = 2.0  # nu
    evidential[:, 2] = 3.0  # alpha
    evidential[:, 3] = 4.0  # beta
    risk = v8_suppliers.emvsnet_epistemic_risk(evidential, method="der")
    assert risk.item() == pytest.approx(1.0)

    evidential[:, 1] = 4.0
    lower_risk = v8_suppliers.emvsnet_epistemic_risk(evidential, method="der")
    assert lower_risk.item() < risk.item()


def test_emvsnet_depth_uses_training_and_validation_probability_volume(tmp_path: Path) -> None:
    request = _request()
    request = type(request)(
        request.frame_ids,
        torch.nn.functional.interpolate(request.rgb, size=(128, 160), mode="bilinear"),
        type(request.cameras)(
            request.cameras.intrinsics,
            request.cameras.c2w,
            (128, 160),
        ),
    )
    supplier = EMVSNetSupplier(
        code_root=tmp_path,
        checkpoint=tmp_path / "x.ckpt",
        device="cpu",
    )
    supplier._model = _FakeEMVSNet()

    result = supplier.infer(request)

    expected = supplier.depth_hypotheses_m(device=torch.device("cpu"))[7].item()
    auxiliary = supplier.depth_hypotheses_m(device=torch.device("cpu"))[19].item()
    assert np.allclose(result.depth_z_m[0], expected)
    assert not np.allclose(result.depth_z_m[0], auxiliary)
