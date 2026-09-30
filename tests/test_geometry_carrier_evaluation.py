"""DEV evaluation must seal RGB-only shared states before any target depth read."""

import copy
import json

import pytest
import torch
from test_direct_capacity_evaluation import setup

from mcss.dynamic.carrier import DynamicSceneCarrier
from mcss.dynamic.config import CarrierConfig
from mcss.dynamic.types import hash_scene_state, hash_value
from mcss.mechanism_pilot.direct_capacity_contracts import StateSealBarrier
from mcss.mechanism_pilot.geometry_carrier_evaluation import evaluate_model, selection_summary
from mcss.mechanism_pilot.small_training import write_json


def fixture(root):
    setup(root)
    manifest = json.loads((root / "scene_manifest.json").read_text())
    manifest["protected_scene_ids"] = manifest.pop("FINAL_HOLDOUT_PROHIBITED")
    for record in manifest["scenes"]:
        record["split"] = "DEV"
        record["physical_scene_id"] = record["scene_id"]
        record["historically_exposed"] = True
    train = copy.deepcopy(manifest["scenes"][0])
    train["scene_id"] = train["physical_scene_id"] = "train_unused"
    train["split"] = "TRAIN"
    for frame in train["frames"]:
        frame["depth"] = "/DO_NOT_OPEN_TRAIN_DEPTH"
    manifest["scenes"].append(train)
    write_json(root / "train_depth_prior.json", {"near_m": 0.1, "far_m": 6.0})
    torch.manual_seed(11)
    return manifest, DynamicSceneCarrier(CarrierConfig(grid_size=(16, 16, 16)))


def test_dev_all_sealed_before_depth_controls_and_shared_queries(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    manifest, carrier = fixture(tmp_path)
    old_hash = hash_value(carrier.state_dict())
    import numpy as np

    original = np.load
    count = []
    oldseal = StateSealBarrier.seal

    def seal(self, *args):
        oldseal(self, *args)
        count.append(1)

    def load(path, *args, **kwargs):
        assert len(count) == 8
        assert "DO_NOT_OPEN" not in str(path)
        return original(path, *args, **kwargs)

    monkeypatch.setattr(StateSealBarrier, "seal", seal)
    monkeypatch.setattr(np, "load", load)
    out = tmp_path / "evaluation"
    result = evaluate_model(carrier, manifest, tmp_path, out, "C1", 5, 100, "cpu", diagnostics=True)
    assert len(result["query_rows"]) == 40 and len(result["context_rows"]) == 12
    assert hash_value(carrier.state_dict()) == old_hash and carrier.training
    access = json.loads((out / "GT_access.json").read_text())
    position = next(i for i, e in enumerate(access) if e.get("event") == "ALL_DEV_STATES_SEALED")
    assert all("depth" not in e.get("channels", []) for e in access[:position])
    assert all(e.get("scene_id", "") != "train_unused" for e in access)
    states = torch.load(out / "states.pt", weights_only=False)
    for sid in ("capacity0", "capacity1"):
        for role in ("A", "B"):
            rows = [
                r
                for r in result["query_rows"]
                if r["scene_id"] == sid and r["construction"] == role
            ]
            direct = [r for r in rows if r["method"] == "direct"]
            assert {r["used_state_hash"] for r in direct} == {
                hash_scene_state(states[f"{sid}/{role}"])
            }
            for fid in (8, 9):
                assert len({r["query_camera_hash"] for r in rows if r["query_id"] == fid}) == 1
            assert all(r["donor_scene_id"] != sid for r in rows if r["method"] == "wrong_scene")
            assert all(r["opacity"] == 0 for r in rows if r["method"] == "zero")
    assert len(list((out / "predictions").glob("*.npz"))) == 40
    with pytest.raises(FileExistsError):
        evaluate_model(carrier, manifest, tmp_path, out, "C0", 5, 100, "cpu")


def test_fresh_only_and_protected_dev_denied(tmp_path):
    manifest, carrier = fixture(tmp_path)
    for record in manifest["scenes"]:
        record["split"] = "FRESH_QUALIFICATION"
        record["historically_exposed"] = False
        record["independence_verified"] = True
    with pytest.raises(PermissionError, match="DEV scenes"):
        evaluate_model(carrier, manifest, tmp_path, tmp_path / "blocked", "C0", 1, 0, "cpu")
    manifest["scenes"][0]["split"] = "DEV"
    manifest["protected_scene_ids"].append(manifest["scenes"][0]["scene_id"])
    with pytest.raises(PermissionError, match="Protected"):
        evaluate_model(carrier, manifest, tmp_path, tmp_path / "blocked", "C0", 1, 0, "cpu")


def test_coverage_gate_tracks_geometric_hits_and_nohit_is_not_collapse():
    rows = []
    for role in ("A", "B"):
        for query in (8, 9):
            rows.append(
                {
                    "scene_id": "nohit",
                    "role": role,
                    "query_id": query,
                    "method": "direct",
                    "depth_absrel": 1.0,
                    "rgb_mse": 0.1,
                    "gray_rgb_mse": 0.1,
                    "coverage": 0.0,
                    "ray_hitfraction": 0.0,
                    "hit_count": 0,
                    "hit_opacity": None,
                }
            )
    result = selection_summary(rows, rows, 100)
    assert result["eligible"] and result["no_hit_scenes"] == ["nohit"]
    hitrows = [
        {**r, "hit_count": 10, "ray_hitfraction": 0.2, "coverage": 0.195, "hit_opacity": 0.8}
        for r in rows
    ]
    assert not selection_summary(hitrows, hitrows, 100)["eligible"]
    hitrows = [{**r, "coverage": 0.2} for r in hitrows]
    assert selection_summary(hitrows, hitrows, 100)["eligible"]
    hitrows = [{**r, "hit_opacity": 0.04} for r in hitrows]
    assert not selection_summary(hitrows, hitrows, 100)["eligible"]
