import numpy as np
import pytest

from mcss.dense_geometry_certificates import (
    DenseCertificateConfig,
    build_dense_certificates,
    read_dense_certificate_cache,
    write_dense_certificate_cache,
)
from mcss.geometry_supplier import GeometrySupplierResult


def _supplier(depths: tuple[float, ...]) -> GeometrySupplierResult:
    view_count = len(depths)
    height = width = 5
    intrinsics = np.repeat(
        np.asarray([[[4.0, 0.0, 2.0], [0.0, 4.0, 2.0], [0.0, 0.0, 1.0]]]),
        view_count,
        axis=0,
    ).astype(np.float32)
    c2w = np.repeat(np.eye(4, dtype=np.float32)[None], view_count, axis=0)
    return GeometrySupplierResult(
        frame_ids=tuple(range(view_count)),
        depth_along_ray=np.stack(
            [np.full((height, width), depth, dtype=np.float32) for depth in depths]
        ),
        confidence=np.ones((view_count, height, width), dtype=np.float32) * 4.0,
        valid_mask=np.ones((view_count, height, width), dtype=np.bool_),
        intrinsics=intrinsics,
        c2w=c2w,
    )


def _config() -> DenseCertificateConfig:
    return DenseCertificateConfig(
        grid_stride=2,
        min_depth_m=0.1,
        max_depth_m=20.0,
        depth_absolute_tolerance_m=0.1,
        depth_relative_tolerance=0.0,
        roundtrip_threshold_px=0.25,
        surface_half_width_m=0.05,
        free_space_margin_m=0.05,
        minimum_supporting_views=1,
        maximum_conflicting_views=0,
    )


def test_consistent_known_camera_depths_create_typed_certificates() -> None:
    certificates = build_dense_certificates(_supplier((2.0, 2.0, 2.0)), _config())

    assert certificates.certified.all()
    assert np.all(certificates.support_count == 2)
    assert np.all(certificates.conflict_count == 0)
    assert np.allclose(certificates.free_end_m, 1.95)
    assert np.allclose(certificates.surface_near_m, 1.95)
    assert np.allclose(certificates.surface_far_m, 2.05)
    assert certificates.provenance.shape[-1] == 3
    assert certificates.provenance.all()
    assert np.all((certificates.risk >= 0.0) & (certificates.risk <= 1.0))


def test_nearer_target_is_occlusion_but_farther_target_is_conflict() -> None:
    occluded = build_dense_certificates(_supplier((2.0, 1.0)), _config())
    conflicted = build_dense_certificates(_supplier((2.0, 3.0)), _config())

    assert np.all(occluded.occlusion_count[0] == 1)
    assert not occluded.certified[0].any()
    assert np.all(conflicted.conflict_count[0] == 1)
    assert not conflicted.certified[0].any()


def test_invalid_supplier_pixels_never_receive_write_authority() -> None:
    supplier = _supplier((2.0, 2.0))
    supplier.valid_mask[0] = False
    certificates = build_dense_certificates(supplier, _config())

    assert not certificates.raw_valid[0].any()
    assert not certificates.certified[0].any()
    assert np.all(certificates.free_end_m[0] == 0.0)


def test_frame_remapping_after_permutation_preserves_certificates() -> None:
    original = _supplier((2.0, 2.0, 2.0))
    permutation = np.asarray([2, 0, 1])
    permuted = GeometrySupplierResult(
        frame_ids=tuple(original.frame_ids[index] for index in permutation),
        depth_along_ray=original.depth_along_ray[permutation],
        confidence=original.confidence[permutation],
        valid_mask=original.valid_mask[permutation],
        intrinsics=original.intrinsics[permutation],
        c2w=original.c2w[permutation],
    ).canonical()

    first = build_dense_certificates(original, _config())
    second = build_dense_certificates(permuted, _config())

    assert first.frame_ids == second.frame_ids
    assert np.array_equal(first.certified, second.certified)
    assert np.allclose(first.world_points, second.world_points)


def test_dense_certificate_cache_is_atomic_verified_and_nonoverwriting(tmp_path) -> None:
    certificates = build_dense_certificates(_supplier((2.0, 2.0)), _config())
    destination = tmp_path / "certificates"

    write_dense_certificate_cache(
        destination,
        certificates,
        _config(),
        supplier_arrays_sha256="a" * 64,
    )
    loaded, metadata = read_dense_certificate_cache(destination)

    assert np.array_equal(loaded.certified, certificates.certified)
    assert metadata["supplier_arrays_sha256"] == "a" * 64
    with pytest.raises(FileExistsError):
        write_dense_certificate_cache(
            destination,
            certificates,
            _config(),
            supplier_arrays_sha256="a" * 64,
        )
    arrays = destination / "arrays.npz"
    arrays.write_bytes(arrays.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="hash"):
        read_dense_certificate_cache(destination)
