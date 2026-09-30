"""EVAL-V4 metadata rule: EVAL-V3's rule with every read EVAL-V3 candidate added to the used set."""

import importlib.util
from pathlib import Path

import pytest


def script():
    path = Path(__file__).parents[1] / "scripts" / "prepare_rgbd_eval_v4_data.py"
    spec = importlib.util.spec_from_file_location("prepare_rgbd_eval_v4_data", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def row(scene, split="train", observed="False"):
    return {
        "scene_name": scene,
        "official_split": split,
        "protocol_partition": split,
        "previously_observed": observed,
    }


def inputs():
    current = {
        "scenes": [
            {"scene_id": "ai_001_001", "split": "TRAIN"},
            {"scene_id": "ai_002_001", "split": "DEV"},
        ]
    }
    v1_lock = {
        "FRESH_V1_candidates": ["ai_003_001"],
        "current_scene_ids": ["ai_001_001", "ai_002_001"],
        "TRAIN_X_candidates": ["ai_001_005"],
    }
    fresh_v2_lock = {"FRESH_V2_candidates": ["ai_004_001"]}
    eval_v3_lock = {
        "EVAL_V3_candidates": ["ai_005_001", "ai_005_002", "ai_006_001", "ai_005_003"],
        "used_scene_ids": ["ai_001_001", "ai_002_001", "ai_003_001", "ai_001_005", "ai_004_001"],
    }
    integrity = {
        "status": "PASS",
        "budget_stop": False,
        "EVAL_V3": 2,
        "failures": {"ai_006_001": {"status": "INELIGIBLE_SOURCE_GEOMETRY"}},
        "never_read_full_volume": ["ai_005_003"],
    }
    return current, v1_lock, fresh_v2_lock, eval_v3_lock, integrity


def test_every_read_eval_v3_candidate_becomes_used_and_the_fence_is_unchanged():
    m = script()
    current, v1_lock, fresh_v2_lock, eval_v3_lock, integrity = inputs()
    used, fence = m.eval_v4_sets(current, v1_lock, fresh_v2_lock, eval_v3_lock, integrity)
    assert {"ai_005_001", "ai_005_002", "ai_006_001"} <= used
    assert "ai_005_003" not in used
    assert fence == m.eval_v3_sets(current, v1_lock, fresh_v2_lock)[1]
    names = ["ai_005_003", "ai_005_002", "ai_007_001", "ai_002_009"]
    rows = [row(s) for s in names]
    assets = {s: {"archive_file": f"A_{s}.rar", "asset_file": "0"} for s in names}
    allowed = {s: list(range(16)) for s in names}
    trajectories = {f"{s}_cam_00": {"Scene type": "OK"} for s in names}
    selected = m.select_candidates(
        rows, assets, allowed, trajectories, protected=set(), excluded=set(), used=used, fence=fence
    )
    # the never-read candidate and a new volume stay eligible; read and fenced scenes do not
    assert sorted(selected) == ["ai_005_003", "ai_007_001"]


def test_inconsistent_eval_v3_records_are_refused():
    m = script()
    _, _, _, eval_v3_lock, integrity = inputs()
    assert m.eval_v3_read(eval_v3_lock, integrity) == {"ai_005_001", "ai_005_002", "ai_006_001"}
    with pytest.raises(PermissionError):
        m.eval_v3_read(eval_v3_lock, {**integrity, "budget_stop": True})
    with pytest.raises(ValueError):
        m.eval_v3_read(eval_v3_lock, {**integrity, "never_read_full_volume": ["ai_009_009"]})
    with pytest.raises(ValueError):
        m.eval_v3_read(eval_v3_lock, {**integrity, "EVAL_V3": 1})
