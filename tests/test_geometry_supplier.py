import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from mcss.geometry_supplier import (
    CodeDependency,
    GeometrySupplierRequest,
    GeometrySupplierResult,
    SupplierIdentity,
    SupplierRunConfig,
    read_supplier_cache,
    write_supplier_cache,
)
from mcss.types import Cameras


def _request(frame_ids: tuple[int, ...] = (20, 10)) -> GeometrySupplierRequest:
    rgb = torch.stack(
        [torch.full((3, 4, 5), float(frame_id) / 100.0) for frame_id in frame_ids]
    )
    intrinsics = torch.tensor(
        [[[4.0, 0.0, 2.0], [0.0, 4.0, 1.5], [0.0, 0.0, 1.0]]] * len(frame_ids)
    )
    c2w = torch.eye(4).repeat(len(frame_ids), 1, 1)
    c2w[:, 0, 3] = torch.tensor(frame_ids, dtype=torch.float32) / 10.0
    return GeometrySupplierRequest(frame_ids, rgb, Cameras(intrinsics, c2w, (4, 5)))


def _result(request: GeometrySupplierRequest) -> GeometrySupplierResult:
    canonical = request.canonical()
    view_count = len(canonical.frame_ids)
    depth = np.stack(
        [np.full((4, 5), float(frame_id), dtype=np.float32) for frame_id in canonical.frame_ids]
    )
    return GeometrySupplierResult(
        frame_ids=canonical.frame_ids,
        depth_along_ray=depth,
        confidence=np.ones((view_count, 4, 5), dtype=np.float32),
        valid_mask=np.ones((view_count, 4, 5), dtype=np.bool_),
        intrinsics=canonical.cameras.intrinsics.numpy(),
        c2w=canonical.cameras.c2w.numpy(),
    )


def test_request_canonicalizes_by_frame_id_and_hashes_content() -> None:
    request = _request()
    canonical = request.canonical()

    assert canonical.frame_ids == (10, 20)
    assert torch.allclose(canonical.rgb[:, 0, 0, 0], torch.tensor([0.1, 0.2]))
    assert canonical.input_sha256 == _request((10, 20)).input_sha256
    changed = _request()
    changed.rgb[0, 0, 0, 0] += 0.01
    assert changed.input_sha256 != request.input_sha256


def test_request_rejects_duplicate_frames_and_mismatched_views() -> None:
    with pytest.raises(ValueError, match="unique"):
        _request((10, 10))

    request = _request()
    with pytest.raises(ValueError, match="view"):
        GeometrySupplierRequest(request.frame_ids, request.rgb[:1], request.cameras)


def test_atomic_supplier_cache_round_trip_refuses_overwrite_and_detects_tamper(
    tmp_path: Path,
) -> None:
    request = _request().canonical()
    result = _result(request)
    identity = SupplierIdentity(
        model_id="facebook/map-anything",
        model_revision="model-revision",
        code_repository="facebookresearch/map-anything",
        code_revision="code-revision",
        dependencies=(
            CodeDependency(
                name="dinov2",
                repository="facebookresearch/dinov2",
                revision="dinov2-revision",
                source_sha256="a" * 64,
            ),
        ),
    )
    config = SupplierRunConfig()
    destination = tmp_path / "supplier"

    write_supplier_cache(destination, result, identity, request.input_sha256, config)
    loaded, metadata = read_supplier_cache(destination)

    assert loaded.frame_ids == (10, 20)
    assert np.array_equal(loaded.depth_along_ray, result.depth_along_ray)
    assert metadata["input_sha256"] == request.input_sha256
    assert metadata["identity"]["model_revision"] == "model-revision"
    assert metadata["identity"]["dependencies"] == [
        {
            "name": "dinov2",
            "repository": "facebookresearch/dinov2",
            "revision": "dinov2-revision",
            "source_sha256": "a" * 64,
        }
    ]
    assert not list(tmp_path.glob(".supplier.tmp-*"))
    with pytest.raises(FileExistsError):
        write_supplier_cache(destination, result, identity, request.input_sha256, config)

    arrays = destination / "arrays.npz"
    arrays.write_bytes(arrays.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="hash"):
        read_supplier_cache(destination)


def test_cache_metadata_cannot_claim_different_frames(tmp_path: Path) -> None:
    request = _request().canonical()
    destination = tmp_path / "supplier"
    write_supplier_cache(
        destination,
        _result(request),
        SupplierIdentity("model", "revision", "repo", "code"),
        request.input_sha256,
        SupplierRunConfig(),
    )
    metadata_path = destination / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["frame_ids"] = [999, 1000]
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match="metadata"):
        read_supplier_cache(destination)


def test_result_rejects_nonfinite_depth_and_shape_mismatch() -> None:
    request = _request().canonical()
    result = _result(request)
    bad_depth = result.depth_along_ray.copy()
    bad_depth[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        replace(result, depth_along_ray=bad_depth)
    with pytest.raises(ValueError, match="shape"):
        replace(result, confidence=result.confidence[:, :-1])
