import numpy as np
import pytest

from mcss.mechanism_pilot.calibration import calibrated_intrinsics


def row(matrix):
    return {f"M_cam_from_uv_{i}{j}": matrix[i][j] for i in range(3) for j in range(3)}


def test_shifted_anisotropic_rays_match_published_formula():
    m = np.array([[0.58, 0.02, 0.09], [0, 0.43, -0.04], [0, 0, -1.0]])
    h, w = 128, 160
    k = calibrated_intrinsics(row(m), (h, w))
    for x, y in [(0, 0), (35, 77), (w - 1, h - 1)]:
        expected = np.diag([1, -1, -1]) @ m @ [2 * (x + 0.5) / w - 1, 1 - 2 * (y + 0.5) / h, 1]
        actual = np.linalg.inv(k) @ [x, y, 1]
        np.testing.assert_allclose(
            actual / np.linalg.norm(actual), expected / np.linalg.norm(expected), atol=1e-12
        )


def test_resize_scales_axes_independently_with_half_pixel_centers():
    m = row([[0.57735, 0, 0], [0, 0.433013, 0], [0, 0, -1]])
    original = calibrated_intrinsics(m, (768, 1024))
    small = calibrated_intrinsics(m, (128, 160))
    assert small[0, 0] == pytest.approx(original[0, 0] * 160 / 1024)
    assert small[1, 1] == pytest.approx(original[1, 1] * 128 / 768)
    assert small[0, 2] == pytest.approx((original[0, 2] + 0.5) * 160 / 1024 - 0.5)
    assert small[0, 0] != pytest.approx(small[1, 1])


def test_invalid_calibration_never_falls_back_to_fov():
    with pytest.raises(ValueError):
        calibrated_intrinsics(row(np.zeros((3, 3))), (128, 160))
    with pytest.raises(KeyError):
        calibrated_intrinsics({}, (128, 160))
