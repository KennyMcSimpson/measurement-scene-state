"""Experiment role firewalls, immutable locks, checkpoint restoration and selection."""

import copy
from pathlib import Path

import pytest
import torch
from test_geometry_carrier_evaluation import fixture

from mcss.dynamic.types import hash_value
from mcss.mechanism_pilot.geometry_carrier_experiment import (
    SceneData,
    load_checkpoint,
    make_carrier,
    require_fresh_lock,
    save_checkpoint,
    select_checkpoint,
    validate_split,
    verify_lock,
)
from mcss.mechanism_pilot.small_training import sha, write_json


def freeze_fixture(root, manifest):
    write_json(root / "scene_split.json", manifest)
    write_json(root / "training_contract.json", {"variants": ["C0", "C1"], "seeds": [7]})
    write_json(
        root / "preregistration.json",
        {
            "status": "FROZEN_BEFORE_FORMAL_TRAINING",
            "input_sha256": {
                str(root / n): sha(root / n) for n in ("scene_split.json", "training_contract.json")
            },
            "source_sha256": {str(Path(__file__).resolve()): sha(Path(__file__))},
        },
    )


def test_physical_identity_protected_and_exposure_firewall(tmp_path):
    manifest, _ = fixture(tmp_path)
    validate_split(manifest)
    duplicate = copy.deepcopy(manifest)
    duplicate["scenes"][1]["physical_scene_id"] = duplicate["scenes"][0]["physical_scene_id"]
    with pytest.raises(PermissionError, match="Duplicate"):
        validate_split(duplicate)
    protected = copy.deepcopy(manifest)
    protected["protected_scene_ids"].append(protected["scenes"][0]["scene_id"])
    with pytest.raises(PermissionError, match="Protected"):
        validate_split(protected)
    exposed = copy.deepcopy(manifest)
    exposed["scenes"][0]["split"] = "FRESH_QUALIFICATION"
    exposed["scenes"][0]["independence_verified"] = True
    with pytest.raises(PermissionError, match="Exposed/unverified"):
        validate_split(exposed)


def test_dev_target_denied_without_depth_read(tmp_path, monkeypatch):
    manifest, _ = fixture(tmp_path)
    data = SceneData(manifest["scenes"][0], manifest, tmp_path, "cpu", [])

    def denied(*a, **kw):
        raise AssertionError("DEV target media was read")

    monkeypatch.setattr(data.media, "read", denied)
    with pytest.raises(PermissionError, match="DEV/fresh target"):
        data.target(0)
    assert all(not hasattr(o, "depth") for o in data.context("A"))
    fresh = copy.deepcopy(manifest["scenes"][0])
    fresh["split"] = "FRESH_QUALIFICATION"
    with pytest.raises(PermissionError, match="Fresh data"):
        SceneData(fresh, manifest, tmp_path, "cpu", [])


def test_lock_mutation_and_fresh_selection_gate(tmp_path):
    manifest, _ = fixture(tmp_path)
    manifest["fresh_status"] = "BLOCKED_INDEPENDENCE_UNRESOLVED"
    freeze_fixture(tmp_path, manifest)
    assert verify_lock(tmp_path)[0] == manifest
    with pytest.raises(PermissionError, match="BLOCKED_INDEPENDENCE"):
        require_fresh_lock(tmp_path)
    write_json(tmp_path / "training_contract.json", {"steps": 1})
    with pytest.raises(PermissionError, match="Frozen input/source changed"):
        verify_lock(tmp_path)


def test_checkpoint_roundtrip_and_earliest_eligible_independent_selection(tmp_path):
    torch.set_num_threads(1)
    model = make_carrier(7, "cpu")
    digest = save_checkpoint(
        tmp_path / "model.pt", model, variant="C1", seed=7, step=100, lock_sha256="a" * 64
    )
    restored, payload = load_checkpoint(tmp_path / "model.pt", "cpu")
    assert digest == sha(tmp_path / "model.pt")
    assert hash_value(restored.state_dict()) == hash_value(model.state_dict())
    assert payload["test_time_input"] == "RGB+CAMERA" and payload["writer_trained"] is False

    def candidate(step, score, eligible=True):
        return {
            "step": step,
            "query_absrel": score,
            "eligible": eligible,
            "partition": "DEV",
            "checkpoint": f"model{step}.pt",
            "checkpoint_sha256": "a" * 64,
        }

    selected = select_checkpoint(
        [candidate(300, 0.1), candidate(100, 0.1 + 1e-9), candidate(50, 0.01, False)]
    )
    assert selected["selected_step"] == 100
    assert select_checkpoint([candidate(20, 0.01, False)])["status"] == "NO_ELIGIBLE_CHECKPOINT"
    with pytest.raises(FileExistsError):
        save_checkpoint(
            tmp_path / "model.pt", model, variant="C1", seed=7, step=100, lock_sha256="a" * 64
        )


def test_data_tampering_and_fresh_candidate_selection_rejected(tmp_path):
    from mcss.mechanism_pilot.geometry_carrier_experiment import verify_data

    data = tmp_path / "synthetic_data.bin"
    data.write_bytes(b"original")
    write_json(tmp_path / "preregistration.json", {"data_sha256": {str(data): sha(data)}})
    verify_data(tmp_path)
    data.write_bytes(b"modified")
    with pytest.raises(PermissionError, match="Locked TRAIN/DEV data changed"):
        verify_data(tmp_path)
    with pytest.raises(PermissionError, match="DEV-only provenance"):
        select_checkpoint(
            [
                {
                    "partition": "FRESH_QUALIFICATION",
                    "eligible": True,
                    "query_absrel": 0.01,
                    "step": 50,
                    "checkpoint": "fresh.pt",
                    "checkpoint_sha256": "a" * 64,
                }
            ]
        )
    with pytest.raises(PermissionError, match="DEV-only provenance"):
        select_checkpoint([{"eligible": True, "query_absrel": 0.01}])
