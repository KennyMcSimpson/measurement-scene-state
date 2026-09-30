"""Synthetic geometry only; no real query media, optimizer or holdout reads."""

from dataclasses import replace

import numpy as np
import pytest
import torch

from mcss.dynamic.types import hash_scene_state
from mcss.geometry import generate_rays
from mcss.mechanism_pilot.context_observability import audit_context_observability
from mcss.mechanism_pilot.direct_capacity_contracts import ObservationBatch, StateSealBarrier
from mcss.mechanism_pilot.direct_capacity_optimization import DirectState
from mcss.mechanism_pilot.visibility import common_visibility_masks
from mcss.types import Cameras


def fixture(n_views=3, sealed=True):
    k = torch.tensor([[2.0, 0.0, 1.0], [0.0, 2.0, 1.0], [0.0, 0.0, 1.0]], dtype=torch.float64)
    pose = torch.eye(4, dtype=torch.float64)
    camera = Cameras(k, pose, (3, 3))
    _, rays = generate_rays(camera)
    depth = 2 / rays[..., 2]
    query = ObservationBatch(
        "s",
        (9,),
        torch.zeros(1, 1, 3, 3, 3),
        Cameras(k[None, None], pose[None, None], (3, 3)),
        depth[None, None, None],
        "SEALED_QUERY_EVALUATION",
    )
    context = ObservationBatch(
        "s",
        tuple(range(n_views)),
        torch.zeros(1, n_views, 3, 3, 3),
        Cameras(
            k[None, None].repeat(1, n_views, 1, 1),
            pose[None, None].repeat(1, n_views, 1, 1),
            (3, 3),
        ),
        depth[None, None, None].repeat(1, n_views, 1, 1, 1),
        "CONTEXT_ONLY_RGBD",
    )
    bounds = torch.tensor([[-3.0, -3.0, -3.0], [3.0, 3.0, 3.0]], dtype=torch.float64)
    state = DirectState(bounds, 2).state()
    barrier = StateSealBarrier(["sealed"])
    if sealed:
        barrier.seal("sealed", "s", state)
    return dict(
        barrier=barrier,
        scene_id="s",
        query=query,
        context=context,
        bounds=bounds,
        anchor_c2w=context.cameras.c2w[0, 0],
        allowed_scene_ids=["s"],
        holdout_scene_ids=["protected"],
    )


def test_seal_and_holdout_guards_precede_query_processing():
    args = fixture(sealed=False)
    args["query"] = None
    with pytest.raises(PermissionError, match="All planned"):
        audit_context_observability(**args)
    args = fixture()
    args["scene_id"] = "protected"
    args["query"] = None
    with pytest.raises(PermissionError, match="Holdout"):
        audit_context_observability(**args)


def test_all_views_visible_full_arrays_and_state_immutable():
    args = fixture()
    before = hash_scene_state(args["barrier"].state("sealed"))
    result = audit_context_observability(**args)
    assert np.all(result["obs_class"] == 2)
    assert np.all(result["visible_view_count"] == 3)
    assert result["world_xyz"].shape == (3, 3, 3)
    assert result["projected_uv"].shape == (3, 3, 3, 2)
    assert np.allclose(result["world_xyz"][1, 1], [0, 0, 2])
    assert np.all(result["bounds_inside"])
    assert np.allclose(result["max_triangulation_angle"], 0, atol=1e-5)
    assert np.all(result["triangulation_angle_bin"] == 0)
    assert before == hash_scene_state(args["barrier"].state("sealed"))


def test_visible_occluded_conflict_mutually_exclusive_and_old_visibility_exact():
    args = fixture()
    depth = args["context"].depth.clone()
    depth[:, 1] *= 0.5
    depth[:, 2] *= 1.5
    args["context"] = replace(args["context"], depth=depth)
    result = audit_context_observability(**args)
    assert np.all(result["obs_class"] == 1)
    assert np.all(result["visible_support"][0])
    assert np.all(result["occluded_by_context_surface"][1])
    assert np.all(result["depth_conflict_front"][2])
    assert np.isnan(result["max_triangulation_angle"]).all()
    assert np.all(result["triangulation_angle_bin"] == -1)
    q = args["query"]
    qc = Cameras(q.cameras.intrinsics[0, 0], q.cameras.c2w[0, 0], (3, 3))
    pairs = [
        (
            Cameras(
                args["context"].cameras.intrinsics[0, j], args["context"].cameras.c2w[0, j], (3, 3)
            ),
            depth[0, j, 0],
        )
        for j in range(3)
    ]
    old = common_visibility_masks(qc, q.depth[0, 0, 0], pairs, pairs)
    assert np.array_equal(old["depth_consistent_common"].numpy(), result["visible_support"].any(0))


def test_invalid_depth_keeps_pixel_and_missing_context_depth_never_visible():
    args = fixture(1)
    qdepth = args["query"].depth.clone()
    qdepth[0, 0, 0, 0, 0] = float("nan")
    args["query"] = replace(args["query"], depth=qdepth)
    cdepth = args["context"].depth.clone()
    cdepth[0, 0, 0, 1, 1] = 0
    args["context"] = replace(args["context"], depth=cdepth)
    result = audit_context_observability(**args)
    assert result["obs_class"][0, 0] == -1
    assert np.array_equal(result["pixel_xy"][0, 0], [0, 0])
    assert np.isnan(result["world_xyz"][0, 0]).all()
    assert result["obs_class"][1, 1] == 0
    assert not result["context_depth_valid"][0, 1, 1]
    assert result["query_valid"].sum() == 8


def test_triangulation_visible_pairs_and_anchor_bounds_coordinates():
    args = fixture(2)
    poses = args["context"].cameras.c2w.clone()
    poses[0, 0, 0, 3], poses[0, 1, 0, 3] = -0.1, 0.1
    args["context"] = replace(
        args["context"], cameras=Cameras(args["context"].cameras.intrinsics, poses, (3, 3))
    )
    args["anchor_c2w"] = poses[0, 0]
    args["bounds"] = torch.tensor([[0.05, -0.1, 1.9], [0.15, 0.1, 2.1]], dtype=torch.float64)
    result = audit_context_observability(**args)
    assert result["visible_view_count"][1, 1] == 2
    assert np.isclose(result["max_triangulation_angle"][1, 1], np.degrees(2 * np.arctan(0.1 / 2)))
    assert result["triangulation_angle_bin"][1, 1] == 1
    assert result["max_triangulation_angle_bin"][1, 1] == 1
    assert result["median_triangulation_angle_bin"][1, 1] == 1
    assert np.isclose(result["camera_baseline"][1, 1], 0.2)
    assert result["bounds_inside"][1, 1]
    assert np.isclose(result["nearest_context_distance"][1, 1], np.sqrt(4.01))


def test_reject_oracle_supervision_and_wrong_anchor():
    args = fixture()
    args["query"] = replace(
        args["query"], supervision="DIAGNOSTIC_QUERY_SUPERVISED_ORACLE_NOT_CONTEXT_ONLY"
    )
    with pytest.raises(PermissionError, match="sealed query"):
        audit_context_observability(**args)
    args = fixture()
    args["anchor_c2w"] = args["anchor_c2w"].clone()
    args["anchor_c2w"][0, 3] = 1
    with pytest.raises(PermissionError, match="anchor"):
        audit_context_observability(**args)


def cli_module():
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "observability_cli", Path(__file__).parents[1] / "scripts/audit_context_observability.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_region_sufficient_statistics_preserve_empty_and_overall():
    cli = cli_module()
    arrays = audit_context_observability(**fixture())
    row = dict(scene_id="s", role="A", variant="S0", method="direct", query_id=9)
    depth = arrays["query_gt_depth"] * 1.2
    rows, error = cli.region_rows(row, depth, arrays)
    by_name = {r["region"]: r for r in rows}
    assert by_name["OVERALL"]["pixel_count"] == 9
    assert np.isclose(by_name["OVERALL"]["absrel_mean"], 0.2)
    assert by_name["OBS0"]["absrel_mean"] is None
    assert by_name["OBS0"]["absrel_sum"] == 0
    assert by_name["OBS0"]["total_valid"] == 9
    assert len(rows) == 11
    summary = cli.observability_summary("s", "A", 9, arrays, error)
    assert summary["spearman_visible_count_absrel"] is None
    assert summary["region_counts"]["OBS2PLUS"] == 9


def test_region_correlation_all_valid_pixels_and_nonfinite_predictions_rejected():
    cli = cli_module()
    arrays = audit_context_observability(**fixture())
    arrays["visible_view_count"] = np.arange(9).reshape(3, 3)
    errors = np.arange(9).reshape(3, 3).astype(float)
    result = cli.observability_summary("s", "A", 9, arrays, errors)
    assert np.isclose(result["spearman_visible_count_absrel"], 1)
    with pytest.raises(ValueError, match="Finite depth"):
        cli.region_rows({}, np.full((3, 3), np.nan), arrays)


def test_previous_prediction_must_exist_in_old_seal_and_match_hash(tmp_path):
    import hashlib

    cli = cli_module()
    path = tmp_path / "prediction.npz"
    path.write_bytes(b"fixture")
    digest = hashlib.sha256(b"fixture").hexdigest()
    assert cli.verify_previous(path, {str(path): {"sha256": digest}}) == digest
    with pytest.raises(PermissionError, match="seal"):
        cli.verify_previous(path, {})
    path.write_bytes(b"changed")
    with pytest.raises(PermissionError, match="changed"):
        cli.verify_previous(path, {str(path): {"sha256": digest}})


def test_formal_barrier_delegates_gate_before_any_checkpoint_load(tmp_path, monkeypatch):
    cli = cli_module()
    calls = []

    def blocked(root, phase):
        assert phase == "formal"
        calls.append(phase)
        raise PermissionError("All136 must complete")

    monkeypatch.setattr(cli, "ensure_phase", blocked)
    monkeypatch.setattr(
        torch, "load", lambda *a, **kw: pytest.fail("No checkpoint load before gate")
    )
    with pytest.raises(PermissionError, match="All136"):
        cli.sealed_formal_barrier(tmp_path, {}, {})
    assert calls == ["formal"]
