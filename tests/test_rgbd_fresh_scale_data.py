"""FRESH-V1 / TRAIN-X metadata-only cohort rules (no media, no network)."""

import importlib.util
from pathlib import Path


def module():
    path = Path(__file__).parents[1] / "scripts" / "prepare_rgbd_fresh_scale_data.py"
    spec = importlib.util.spec_from_file_location("fresh_scale_data", path)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def row(name, split, partition, observed="False"):
    return {
        "scene_name": name,
        "official_split": split,
        "protocol_partition": partition,
        "previously_observed": observed,
    }


def fixture():
    partitions = [
        row("ai_001_001", "train", "train", "True"),  # current TRAIN
        row("ai_002_001", "train", "train", "True"),  # current DEV
        row("ai_003_001", "val", "val"),  # fresh
        row("ai_003_002", "val", "val"),  # same volume as a fresh scene
        row("ai_004_001", "val", "val", "True"),  # observed
        row("ai_005_001", "val", "val"),  # protected
        row("ai_001_005", "val", "val"),  # current volume
        row("ai_006_001", "val", "val"),  # asset shared with protected
        row("ai_007_001", "val", "val"),  # too few frames
        row("ai_008_001", "val", "val"),  # fresh
        row("ai_001_002", "train", "train"),
        row("ai_001_003", "train", "train"),
        row("ai_001_004", "train", "train"),
        row("ai_002_002", "train", "train"),  # DEV volume
        row("ai_003_003", "train", "train"),  # fresh volume
        row("ai_009_001", "train", "train"),
    ]
    assets = {
        r["scene_name"]: {"archive_file": r["scene_name"], "asset_file": "a"} for r in partitions
    }
    assets["ai_006_001"] = {"archive_file": "ai_005_001", "asset_file": "a"}
    allowed = {r["scene_name"]: list(range(16)) for r in partitions}
    allowed["ai_007_001"] = list(range(10))
    trajectories = {s + "_cam_00": {"Scene type": "OK"} for s in allowed}
    return partitions, assets, allowed, trajectories


def test_fresh_rule_keeps_only_independent_unique_volume_scenes():
    m = module()
    partitions, assets, allowed, trajectories = fixture()
    fresh = m.select_fresh(
        partitions,
        assets,
        allowed,
        trajectories,
        protected={"ai_005_001"},
        excluded=set(),
        current={"ai_001_001", "ai_002_001"},
    )
    assert fresh == ["ai_003_001", "ai_008_001"]


def test_train_x_rule_respects_dev_fresh_volumes_and_per_volume_cap():
    m = module()
    partitions, assets, allowed, trajectories = fixture()
    selected = m.select_train_x(
        partitions,
        assets,
        allowed,
        trajectories,
        protected={"ai_005_001"},
        excluded=set(),
        current={"ai_001_001", "ai_002_001"},
        dev={"ai_002_001"},
        fresh=["ai_003_001", "ai_008_001"],
    )
    assert "ai_002_002" not in selected and "ai_003_003" not in selected
    assert sum(s.startswith("ai_001_") for s in selected) == m.TRAIN_X_PER_VOLUME
    assert "ai_009_001" in selected and len(selected) == len(set(selected))
