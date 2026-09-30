import numpy as np
import pytest
import torch

from mcss.mechanism_pilot.unseen_data import (
    frame_quality,
    select_roles,
    source_assets,
    validate_unseen,
)


def test_upstream_literal_only_never_executes(tmp_path):
    p = tmp_path / "upstream.py"
    p.write_text(
        "raise RuntimeError('must not execute')\n"
        "scenes.append({'name':'a','archive_file':'x','asset_file':'y'})\n"
        "scenes.append({'name':'b','asset_file':do_not_run()})\n"
    )
    assert list(source_assets(p)) == ["a"]


def test_unseen_source_asset_and_scene_disjointness():
    assets = {
        "ai_001_001": {"archive_file": "train.rar", "asset_file": "x"},
        "ai_004_001": {"archive_file": "dev.rar", "asset_file": "x"},
    }
    validate_unseen(["ai_001_001"], ["ai_004_001"], assets)
    with pytest.raises(ValueError, match="checkpoint training"):
        validate_unseen(["ai_001_001"], ["ai_001_001"], assets)
    assets["ai_004_001"]["archive_file"] = "TRAIN.RAR"
    with pytest.raises(ValueError, match="asset identity"):
        validate_unseen(["ai_001_001"], ["ai_004_001"], assets)


def test_quality_rule_has_no_model_score_and_rejects_black_native_data():
    depth = np.ones((4, 4))
    assert not frame_quality(np.zeros((4, 4, 3)), depth)["eligible"]
    assert frame_quality(np.full((4, 4, 3), 3), depth)["eligible"]
    depth[0, 0] = np.nan
    assert not frame_quality(np.full((4, 4, 3), 255), depth)["eligible"]


def test_role_selection_keeps_query_outside_context():
    frames = [
        {
            "frame_id": i,
            "intrinsics": [[100.0, 0.0, 79.5], [0.0, 100.0, 63.5], [0.0, 0.0, 1.0]],
            "c2w": torch.eye(4).tolist(),
        }
        for i in range(8)
    ]
    quality = {i: {"eligible": True} for i in range(8)}
    roles, diag = select_roles(frames, quality, torch.tensor([[0.0, 0.0, 1.0], [0.1, 0.0, 1.0]]))
    assert diag["status"] == "ELIGIBLE"
    assert roles["primary_query"] == [6, 7]
    assert not set(roles["primary_query"]) & (set(roles["context_a"]) | set(roles["context_b"]))
    assert set(roles["context_a"]) & set(roles["context_b"]) == {0}
    assert roles["context_a"] == [0, 1, 2]
    assert roles["context_b"] == [0, 3, 4]
