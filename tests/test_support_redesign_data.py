import pytest

from mcss.mechanism_pilot.support_redesign_data import safe_source_assets, validate_roles_disjoint


def test_safe_asset_join_never_executes_external_code(tmp_path):
    f = tmp_path / "source.py"
    f.write_text(
        "raise Exception('do not execute')\n"
        "scenes.append({'name':'a','archive_file':'a.rar',"
        "'asset_file':os.path.join('folder','asset')})\n"
    )
    assert safe_source_assets(f)["a"]["asset_file"] == "folder/asset"
    f.write_text("scenes.append({'name':'a','asset_file':__import__('os').system('false')})")
    with pytest.raises(ValueError, match="Unsupported"):
        safe_source_assets(f)


def test_holdout_cannot_overlap_train_dev_exposed_asset():
    names = ["ai_001_001", "ai_004_001", "ai_012_001", "ai_013_001"]
    assets = {s: {"archive_file": s + ".rar", "asset_file": "asset"} for s in names}
    validate_roles_disjoint([names[0]], [names[1]], [names[2]], [names[3]], assets)
    with pytest.raises(ValueError, match="overlap"):
        validate_roles_disjoint([names[0]], [names[1]], [names[2]], [names[1]], assets)
    assets[names[3]] = assets[names[0]].copy()
    with pytest.raises(ValueError, match="asset overlap"):
        validate_roles_disjoint([names[0]], [names[1]], [names[2]], [names[3]], assets)


def test_distinct_scene_same_volume_is_rejected():
    names = ["ai_001_001", "ai_004_001", "ai_012_001", "ai_012_002"]
    assets = {s: {"archive_file": s + ".rar", "asset_file": "asset"} for s in names}
    with pytest.raises(ValueError, match="volume overlap"):
        validate_roles_disjoint([names[0]], [names[1]], [names[2]], [names[3]], assets)
