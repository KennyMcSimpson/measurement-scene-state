"""TRAIN-EXT candidate rule: splits, exclusions, DEV-volume fence, asset uniqueness, order."""

import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "prepare_rgbd_train_ext_data.py"
spec = importlib.util.spec_from_file_location("prepare_rgbd_train_ext_data", SCRIPT)
ext = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ext)


def row(sid, split="train", partition=None):
    return {"scene_name": sid, "official_split": split, "protocol_partition": partition or split}


def assets_for(ids, shared=()):
    """One unique asset per scene; scenes in `shared` reuse the asset of the first one."""
    out = {sid: {"archive_file": f"AI_{sid}.rar", "asset_file": "01"} for sid in ids}
    for sid in shared[1:]:
        out[sid] = out[shared[0]]
    return out


def run(rows, **kw):
    ids = [r["scene_name"] for r in rows]
    options = {
        "assets": assets_for(ids),
        "allowed": {sid: list(range(16)) for sid in ids},
        "trajectories": {f"{sid}_cam_00": {"Scene type": "OK"} for sid in ids},
        "excluded_sets": [],
        "blocked_assets": set(),
        "fence": set(),
    }
    options.update(kw)
    return ext.select_candidates(rows, **options)


def test_train_and_val_are_kept_but_test_and_mismatched_partitions_are_not():
    rows = [
        row("ai_001_001"),
        row("ai_002_001", "val"),
        row("ai_003_001", "test", "final_holdout"),
        row("ai_004_001", "train", "val"),
    ]
    assert sorted(run(rows)) == ["ai_001_001", "ai_002_001"]


def test_exclusions_fence_short_and_bad_trajectories_are_dropped():
    rows = [row(f"ai_00{i}_001") for i in range(1, 7)]
    ids = [r["scene_name"] for r in rows]
    allowed = {sid: list(range(16)) for sid in ids} | {"ai_004_001": list(range(15))}
    trajectories = {f"{sid}_cam_00": {"Scene type": "OK"} for sid in ids}
    trajectories["ai_005_001_cam_00"] = {"Scene type": "BAD"}
    kept = run(
        rows,
        excluded_sets=[{"ai_001_001"}, {"ai_002_001"}],
        fence={"003"},
        allowed=allowed,
        trajectories=trajectories,
    )
    assert kept == ["ai_006_001"]


def test_source_assets_are_unique_against_blocked_and_within_the_set():
    rows = [row(f"ai_00{i}_001") for i in range(1, 5)]
    ids = [r["scene_name"] for r in rows]
    assets = assets_for(ids, shared=("ai_001_001", "ai_002_001"))
    blocked = {ext.asset_identity(assets["ai_003_001"])}
    kept = run(rows, assets=assets, blocked_assets=blocked)
    assert "ai_003_001" not in kept and "ai_004_001" in kept
    assert len({"ai_001_001", "ai_002_001"} & set(kept)) == 1


def test_order_is_the_seeded_hash_order():
    rows = [row(f"ai_0{i:02d}_001") for i in range(1, 20)]
    kept = run(rows)
    assert kept == sorted(kept, key=ext.selection_key)
    assert ext.SELECTION_SEED == "TRAIN-EXT-20260930" and ext.TRAIN_EXT_MIN == 40
