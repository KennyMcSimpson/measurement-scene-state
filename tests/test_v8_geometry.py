import json
from pathlib import Path

import numpy as np
import pytest

from mcss import v8_geometry
from mcss.v8_geometry import (
    V8Proposal,
    classify_dual_proposals,
    depth_z_to_points_cam,
    emvsnet_mm_to_m,
    read_v8_proposal_cache,
    write_v8_proposal_cache,
)


def _proposal(name: str, depth: float = 2.0) -> V8Proposal:
    depth_z = np.full((2, 3), depth, dtype=np.float32)
    points = depth_z_to_points_cam(
        depth_z,
        np.asarray([[4.0, 0.0, 1.0], [0.0, 4.0, 0.5], [0.0, 0.0, 1.0]], dtype=np.float32),
    )
    return V8Proposal(
        supplier=name,
        frame_ids=(0, 1, 2),
        depth_z_m=np.repeat(depth_z[None], 3, axis=0),
        points_cam_m=np.repeat(points[None], 3, axis=0),
        native_confidence=np.full((3, 2, 3), 0.8, dtype=np.float32),
        native_risk=np.full((3, 2, 3), 0.2, dtype=np.float32),
        valid_mask=np.ones((3, 2, 3), dtype=np.bool_),
        intrinsics=np.repeat(
            np.asarray(
                [[[4.0, 0.0, 1.0], [0.0, 4.0, 0.5], [0.0, 0.0, 1.0]]],
                dtype=np.float32,
            ),
            3,
            axis=0,
        ),
        c2w=np.repeat(np.eye(4, dtype=np.float32)[None], 3, axis=0),
        unit_provenance="metric_m",
        coordinate_convention="opencv_camera_frame",
    )


def test_depth_z_conversion_and_emvsnet_unit_equivariance() -> None:
    intrinsics = np.asarray(
        [[4.0, 0.0, 1.0], [0.0, 4.0, 0.5], [0.0, 0.0, 1.0]], dtype=np.float32
    )
    depth_m = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    points_m = depth_z_to_points_cam(depth_m, intrinsics)
    assert points_m.shape == (2, 2, 3)
    assert np.allclose(points_m[..., 2], depth_m)
    assert np.allclose(emvsnet_mm_to_m(depth_m * 1000.0), depth_m)


def test_ray_distance_to_depth_z_matches_unit_ray_geometry() -> None:
    intrinsics = np.asarray(
        [[4.0, 0.0, 1.0], [0.0, 4.0, 0.5], [0.0, 0.0, 1.0]], dtype=np.float32
    )
    ray_distance = np.full((2, 3), 5.0, dtype=np.float32)
    depth_z = v8_geometry.ray_distance_to_depth_z(ray_distance, intrinsics)
    points = depth_z_to_points_cam(depth_z, intrinsics)
    assert np.allclose(np.linalg.norm(points, axis=-1), ray_distance, atol=1e-6)


def test_proposal_rejects_points_whose_z_coordinate_disagrees_with_depth() -> None:
    proposal = _proposal("unidepth")
    proposal.points_cam_m[..., 2] += 0.25
    with pytest.raises(ValueError, match="z coordinate"):
        V8Proposal(**proposal.__dict__)


def test_dual_state_labels_conflict_and_quiet_joint_failure() -> None:
    first = _proposal("unidepth", depth=2.0)
    second = _proposal("emvsnet", depth=2.01)
    state = classify_dual_proposals(first, second, disagreement_threshold_m=0.1)
    assert state["status"] == "model-supported"

    conflict = classify_dual_proposals(
        first,
        _proposal("emvsnet", depth=4.0),
        disagreement_threshold_m=0.1,
    )
    assert conflict["status"] == "conflict"

    quiet = classify_dual_proposals(
        first,
        _proposal("emvsnet", depth=2.0),
        disagreement_threshold_m=0.1,
        intervention_delta_m=0.0,
        joint_failure=True,
    )
    assert quiet["status"] == "quiet_joint_failure"


def test_v8_cache_is_sealed_and_refuses_overwrite(tmp_path: Path) -> None:
    proposal = _proposal("unidepth")
    destination = tmp_path / "proposal"
    write_v8_proposal_cache(destination, proposal, metadata={"window": "window_000"})
    loaded, metadata = read_v8_proposal_cache(destination)
    assert loaded.supplier == proposal.supplier
    assert metadata["sealed"] is True
    assert (destination / "SEAL.json").is_file()
    with pytest.raises(FileExistsError, match="overwrite"):
        write_v8_proposal_cache(destination, proposal, metadata={"window": "window_000"})

    seal = json.loads((destination / "SEAL.json").read_text(encoding="utf-8"))
    assert seal["arrays_sha256"] == metadata["arrays_sha256"]
    assert metadata["sealed_at_utc"].endswith("+00:00")
