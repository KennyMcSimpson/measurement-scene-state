"""Metadata-only cohort and camera-only frame-role regression tests."""

import importlib.util
from pathlib import Path

import pytest
import torch

from mcss.mechanism_pilot.optimization_bounds_contracts import FrozenTrainingPrior


def module():
    source = Path(__file__).resolve().parents[1] / "scripts/prepare_geometry_carrier_data.py"
    spec = importlib.util.spec_from_file_location("geometry_carrier_data", source)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def cohort_fixture():
    old = [f"ai_{i:03d}_001" for i in range(1, 21)]
    candidates = [f"ai_{i:03d}_001" for i in range(21, 36)]
    protected = ["ai_099_001"]
    assets = {
        s: {"archive_file": s + ".rar", "asset_file": "asset"} for s in old + candidates + protected
    }
    rows = [
        {"scene_name": s, "protocol_partition": "train", "previously_observed": "True"}
        for s in candidates
    ]
    allowed = {s: list(range(16)) for s in candidates}
    trajectories = {s + "_cam_00": {"Scene type": "living room"} for s in candidates}
    return old, rows, assets, allowed, trajectories, protected


def test_metadata_split_deterministic_and_old_bounds_prior_remains_train():
    m = module()
    args = cohort_fixture()
    train, dev, new = m.select_cohort(*args)
    assert (train, dev, new) == m.select_cohort(*args)
    assert len(train) == 24 and len(dev) == 8 and len(new) == 12
    assert set(m.OLD_THREE) <= set(train)
    assert not set(train) & set(dev)
    assert not {s.split("_")[1] for s in train} & {s.split("_")[1] for s in dev}


@pytest.mark.parametrize("reason", ["bad", "short", "protected", "asset", "volume", "partition"])
def test_candidate_rejects_invalid_or_protected_metadata(reason):
    m = module()
    old, rows, assets, allowed, trajectories, protected = cohort_fixture()
    first = rows[0]["scene_name"]
    if reason == "bad":
        trajectories[first + "_cam_00"]["Scene type"] = "BAD: invalid camera"
    elif reason == "short":
        allowed[first] = list(range(15))
    elif reason == "protected":
        protected.append(first)
    elif reason == "asset":
        assets[first] = assets[protected[0]]
    elif reason == "partition":
        rows[0]["protocol_partition"] = "final_holdout"
    else:
        replacement = "ai_001_002"
        rows[0]["scene_name"] = replacement
        assets[replacement] = assets[first]
        allowed[replacement] = allowed[first]
        trajectories[replacement + "_cam_00"] = trajectories[first + "_cam_00"]
        first = replacement
    _, _, new = m.select_cohort(old, rows, assets, allowed, trajectories, protected)
    assert first not in new


def test_insufficient_candidates_fail_instead_of_searching_other_pool():
    m = module()
    old, rows, assets, allowed, trajectories, protected = cohort_fixture()
    with pytest.raises(ValueError, match="Insufficient"):
        m.select_cohort(old, rows[:11], assets, allowed, trajectories, protected)


def test_old_protected_scene_is_never_reassigned():
    m = module()
    old, rows, assets, allowed, trajectories, protected = cohort_fixture()
    with pytest.raises(PermissionError, match="Protected"):
        m.select_cohort(old, rows, assets, allowed, trajectories, protected + [old[0]])


def camera_fixture():
    frames = [
        {
            "frame_id": i,
            "intrinsics": [[100.0, 0.0, 79.5], [0.0, 100.0, 63.5], [0.0, 0.0, 1.0]],
            "c2w": torch.eye(4).tolist(),
            "depth": "must-not-read.npy",
            "rgb": "must-not-read.png",
        }
        for i in range(8)
    ]
    quality = {i: {"eligible": True} for i in range(8)}
    return frames, quality


def test_frame_roles_use_no_query_camera_or_media_and_are_deterministic():
    m = module()
    frames, quality = camera_fixture()
    prior = FrozenTrainingPrior(0.2, 5.0, "f" * 64)
    roles, support = m.camera_only_roles("synthetic", frames, quality, prior)
    assert roles["context_a"] == [0, 1, 2]
    assert roles["context_b"] == [0, 3, 4]
    assert roles["primary_query"] == [6, 7]
    assert support["A_supported_candidates"] >= 2 and support["B_supported_candidates"] >= 2
    for frame in frames[-2:]:
        frame["intrinsics"] = "FORBIDDEN QUERY INTRINSICS"
        frame["c2w"] = "FORBIDDEN QUERY CAMERA"
        frame["depth"] = object()
        frame["rgb"] = object()
    assert (roles, support) == m.camera_only_roles("synthetic", frames, quality, prior)


def test_insufficient_quality_frames_fail_without_replacement():
    m = module()
    frames, quality = camera_fixture()
    quality[0]["eligible"] = quality[1]["eligible"] = False
    with pytest.raises(ValueError, match="Fewer than seven"):
        m.camera_only_roles("synthetic", frames, quality, FrozenTrainingPrior(0.2, 5.0, "f" * 64))
