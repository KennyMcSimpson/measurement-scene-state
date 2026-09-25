import inspect
import json
from dataclasses import asdict

import pytest
import torch

from mcss.geometry_certificates import (
    CertificateConfig,
    GeometryCertificate,
    build_certificates,
    fundamental_matrix,
    triangulate_dlt,
)
from mcss.types import Cameras


def _cameras() -> Cameras:
    intrinsics = torch.tensor(
        [[[80.0, 0.0, 31.5], [0.0, 80.0, 31.5], [0.0, 0.0, 1.0]]]
    ).repeat(3, 1, 1)
    poses = torch.eye(4).repeat(3, 1, 1)
    poses[1, 0, 3] = 0.5
    poses[2, 1, 3] = 0.5
    return Cameras(intrinsics, poses, (64, 64))


def test_builder_signature_cannot_accept_labels() -> None:
    parameters = set(inspect.signature(build_certificates).parameters)

    assert parameters == {"context_rgb", "context_cameras", "config"}
    assert not any(name in parameters for name in ("depth", "target_rgb", "normal", "point"))


def test_fundamental_matrix_obeys_epipolar_constraint() -> None:
    cameras = _cameras()
    point = torch.tensor([[0.1, 0.2, 3.0]])
    pixels = []
    for view in range(2):
        w2c = torch.linalg.inv(cameras.c2w[view])
        camera_point = point @ w2c[:3, :3].T + w2c[:3, 3]
        projected = camera_point @ cameras.intrinsics[view].T
        pixels.append(projected[:, :2] / projected[:, 2:])
    homogeneous_0 = torch.cat((pixels[0], torch.ones(1, 1)), dim=-1)
    homogeneous_1 = torch.cat((pixels[1], torch.ones(1, 1)), dim=-1)

    matrix = fundamental_matrix(cameras, 0, 1)

    residual = homogeneous_1 @ matrix @ homogeneous_0.T
    torch.testing.assert_close(residual, torch.zeros_like(residual), atol=1e-5, rtol=0)


def test_dlt_triangulation_recovers_metric_point_and_positive_depth() -> None:
    cameras = _cameras()
    point = torch.tensor([0.1, 0.2, 3.0])
    observations = []
    for view in range(2):
        projection = cameras.intrinsics[view] @ torch.linalg.inv(cameras.c2w[view])[:3]
        homogeneous = projection @ torch.cat((point, torch.ones(1)))
        observations.append(homogeneous[:2] / homogeneous[2])

    result = triangulate_dlt(observations[0], observations[1], cameras, 0, 1)

    torch.testing.assert_close(result.point, point, atol=1e-4, rtol=1e-4)
    assert result.cheirality
    assert result.reprojection_error_px < 1e-4
    assert result.triangulation_angle_deg > 1.0


def test_certificate_is_json_serializable_and_frozen() -> None:
    certificate = GeometryCertificate(
        source_view=0,
        source_xy=(10.0, 12.0),
        matched_view=1,
        matched_xy=(11.0, 12.0),
        verification_view=2,
        verification_xy=(10.5, 12.5),
        point=(0.0, 0.0, 3.0),
        descriptor_distance=0.1,
        reciprocal_error_px=0.0,
        cycle_error_px=0.0,
        reprojection_error_px=0.1,
        verification_error_px=0.2,
        triangulation_angle_deg=5.0,
    )

    assert json.loads(json.dumps(asdict(certificate)))["point"] == [0.0, 0.0, 3.0]
    with pytest.raises((AttributeError, TypeError)):
        certificate.point = (1.0, 0.0, 3.0)  # type: ignore[misc]


def test_config_rejects_nonpositive_or_inconsistent_thresholds() -> None:
    with pytest.raises(ValueError, match="grid_stride"):
        CertificateConfig(grid_stride=0)
    with pytest.raises(ValueError, match="verification"):
        CertificateConfig(reprojection_threshold_px=2.0, verification_threshold_px=1.0)

