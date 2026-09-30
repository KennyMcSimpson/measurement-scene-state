"""EVAL-V3 metadata rule: FRESH-V2's rule with DEV and FRESH volumes (not TRAIN) as the fence."""

import importlib.util
from pathlib import Path


def script():
    path = Path(__file__).parents[1] / "scripts" / "prepare_rgbd_eval_v3_data.py"
    spec = importlib.util.spec_from_file_location("prepare_rgbd_eval_v3_data", path)
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


def test_used_and_fence_sets():
    m = script()
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
    fresh_v2_lock = {"FRESH_V2_candidates": ["ai_004_001", "ai_004_009"]}
    used, fence = m.eval_v3_sets(current, v1_lock, fresh_v2_lock)
    assert used == {
        "ai_001_001",
        "ai_002_001",
        "ai_003_001",
        "ai_001_005",
        "ai_004_001",
        "ai_004_009",
    }
    assert fence == {"ai_002_001", "ai_003_001", "ai_004_001", "ai_004_009"}
    rows = [row(s) for s in ("ai_001_002", "ai_002_002", "ai_003_002", "ai_004_002", "ai_005_001")]
    names = [r["scene_name"] for r in rows]
    assets = {s: {"archive_file": f"A_{s}.rar", "asset_file": "0"} for s in names}
    allowed = {s: list(range(16)) for s in names}
    trajectories = {f"{s}_cam_00": {"Scene type": "OK"} for s in names}
    selected = m.select_candidates(
        rows, assets, allowed, trajectories, protected=set(), excluded=set(), used=used, fence=fence
    )
    # a TRAIN volume stays eligible; DEV and FRESH volumes are fenced
    assert sorted(selected) == ["ai_001_002", "ai_005_001"]
    assert selected == sorted(selected, key=m.selection_key)


def test_used_scenes_and_shared_assets_are_never_selected():
    m = script()
    rows = [row("ai_001_002"), row("ai_001_003"), row("ai_001_004", observed="True")]
    names = [r["scene_name"] for r in rows] + ["ai_001_001"]
    assets = {s: {"archive_file": f"A_{s}.rar", "asset_file": "0"} for s in names}
    assets["ai_001_003"] = dict(assets["ai_001_001"])  # the training scene's source asset
    allowed = {s: list(range(16)) for s in names}
    trajectories = {f"{s}_cam_00": {"Scene type": "OK"} for s in names}
    selected = m.select_candidates(
        rows,
        assets,
        allowed,
        trajectories,
        protected=set(),
        excluded=set(),
        used={"ai_001_001"},
        fence=set(),
    )
    assert selected == ["ai_001_002"]
