"""FRESH-V2 metadata rule: never observed, unused, volume-fenced, asset-unique, hash order."""

import importlib.util
from collections import Counter
from pathlib import Path


def script():
    path = Path(__file__).parents[1] / "scripts" / "prepare_rgbd_fresh_v2_data.py"
    spec = importlib.util.spec_from_file_location("prepare_rgbd_fresh_v2_data", path)
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


def test_selection_applies_every_metadata_rule_in_hash_order():
    m = script()
    rows = [
        row("ai_001_002"),  # volume fenced by the TRAIN/DEV scene ai_001_001
        row("ai_002_001"),
        row("ai_002_002"),
        row("ai_002_003", observed="True"),
        row("ai_002_004", split="val"),
        row("ai_002_005"),  # protected
        row("ai_002_006"),  # already used
        row("ai_002_007"),  # shares its source asset with ai_002_001
        row("ai_002_008"),  # fewer than 16 frames
        row("ai_002_009"),  # BAD trajectory
        row("ai_003_001"),
    ]
    names = [r["scene_name"] for r in rows]
    assets = {s: {"archive_file": f"A_{s}.rar", "asset_file": "0"} for s in names}
    assets["ai_002_007"] = dict(assets["ai_002_001"])
    allowed = {s: list(range(16)) for s in names}
    allowed["ai_002_008"] = list(range(15))
    trajectories = {f"{s}_cam_00": {"Scene type": "OK"} for s in names}
    trajectories["ai_002_009_cam_00"] = {"Scene type": "BAD"}
    selected = m.select_fresh_v2(
        rows,
        assets,
        allowed,
        trajectories,
        protected={"ai_002_005"},
        excluded=set(),
        used={"ai_002_006", "ai_001_001"},
        fence={"ai_001_001"},
    )
    duplicate = min(("ai_002_001", "ai_002_007"), key=m.selection_key)
    expected = sorted(["ai_002_002", "ai_003_001", duplicate], key=m.selection_key)
    assert selected == expected


def test_a_full_volume_is_never_read_again():
    m = script()
    counts = Counter({"002": m.PER_VOLUME})
    assert m.volume_full(counts, "ai_002_010")
    assert not m.volume_full(counts, "ai_003_001")
