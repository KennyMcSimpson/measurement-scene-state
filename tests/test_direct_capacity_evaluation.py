"""Synthetic post-seal evaluator tests; no real capacity-development query access."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from mcss.dynamic.types import hash_scene_state
from mcss.mechanism_pilot.direct_capacity_evaluation import evaluate_capacity
from mcss.mechanism_pilot.small_training import sha
from mcss.types import SceneState


def setup(tmp_path):
    records, specs = [], []
    for index in range(2):
        sid = f"capacity{index}"
        record = {
            "scene_id": sid,
            "cohort_role": "CAPACITY_DEV_EXPOSED",
            "roles": {"context_a": [0, 1, 2], "context_b": [0, 3, 4], "primary_query": [8, 9]},
            "frames": [],
        }
        for fid in [0, 1, 2, 3, 4, 8, 9]:
            rgb, depth = tmp_path / f"{sid}_{fid}.png", tmp_path / f"{sid}_{fid}.npy"
            Image.fromarray(np.full((4, 4, 3), 30 + index * 20, np.uint8)).save(rgb)
            np.save(depth, np.full((4, 4), 3.0, np.float32))
            pose = np.eye(4)
            pose[:3, 3] = [index * 3.0 + fid * 0.02, index * 2.0, 0.1]
            record["frames"].append(
                {
                    "frame_id": fid,
                    "rgb": str(rgb),
                    "depth": str(depth),
                    "intrinsics": [[4.0, 0.0, 2.0], [0.0, 4.0, 2.0], [0.0, 0.0, 1.0]],
                    "c2w": pose.tolist(),
                }
            )
        records.append(record)
        for role in ["A", "B"]:
            state = SceneState(
                torch.full((1, 1, 8, 8, 8), -2.0 + index),
                torch.full((1, 3, 8, 8, 8), 0.3 + index * 0.1),
                torch.zeros(1, 1, 8, 8, 8),
                torch.tensor([[[-6.0, -4.0, -6.0], [6.0, 4.0, 6.0]]]),
            )
            directory = tmp_path / "checkpoints" / f"{sid}_{role}"
            directory.mkdir(parents=True)
            torch.save(state, directory / "state.pt")
            (directory / "summary.json").write_text(
                json.dumps({"state_hash": hash_scene_state(state)})
            )
            specs.append(
                {
                    "key": f"{sid}/{role}",
                    "scene_id": sid,
                    "role": role,
                    "track": "RGB_ONLY",
                    "grid": 8,
                    "samples": 64,
                    "bounds_mode": "CURRENT_BOUNDS",
                    "phase": "context",
                    "state_path": str((directory / "state.pt").relative_to(tmp_path)),
                    "optimization_path": str(directory.relative_to(tmp_path)),
                }
            )
    (tmp_path / "scene_manifest.json").write_text(
        json.dumps({"scenes": records, "image_size": [4, 4], "FINAL_HOLDOUT_PROHIBITED": ["held0"]})
    )
    (tmp_path / "state_plan.json").write_text(json.dumps({"states": specs}))
    source_hashes = {str(__file__): sha(Path(__file__))}
    (tmp_path / "config.json").write_text(json.dumps({"source_sha256": source_hashes}))
    for spec in specs:
        directory = tmp_path / spec["optimization_path"]
        (directory / "completion.json").write_text(
            json.dumps(
                {
                    "plan_sha256": sha(tmp_path / "state_plan.json"),
                    "state_file_sha256": sha(directory / "state.pt"),
                    "spec": spec,
                    "phase": "context",
                    "source_hashes": source_hashes,
                }
            )
        )
    (tmp_path / "context_optimization_complete.json").write_text(
        json.dumps(
            {
                "status": "PASS",
                "phase": "context",
                "states": len(specs),
                "input_hashes": {
                    str(tmp_path / name): sha(tmp_path / name)
                    for name in ["scene_manifest.json", "state_plan.json", "config.json"]
                },
            }
        )
    )
    return specs


def test_missing_planned_state_stops_before_query_read(tmp_path, monkeypatch):
    specs = setup(tmp_path)
    (tmp_path / specs[-1]["state_path"]).unlink()

    def no_query(*args, **kwargs):
        raise AssertionError("Query reached before complete phase")

    monkeypatch.setattr(
        "mcss.mechanism_pilot.direct_capacity_evaluation.CapacityEvaluator.query", no_query
    )
    with pytest.raises(PermissionError, match="Every planned optimization"):
        evaluate_capacity(tmp_path, device="cpu")
    assert not (tmp_path / "raw/context_predictions").exists()


def test_full_synthetic_phase_preserves_state_camera_and_raw_predictions(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    setup(tmp_path)
    from mcss.mechanism_pilot.direct_capacity_contracts import StateSealBarrier

    original_seal = StateSealBarrier.seal
    sealed_count = []

    def observed_seal(barrier, *args):
        original_seal(barrier, *args)
        sealed_count.append(1)

    def guarded_sha(path):
        if str(path).endswith((".png", ".npy")):
            assert len(sealed_count) == 4
        return sha(path)

    monkeypatch.setattr(StateSealBarrier, "seal", observed_seal)
    monkeypatch.setattr("mcss.mechanism_pilot.direct_capacity_evaluation.sha", guarded_sha)
    rows = evaluate_capacity(tmp_path, device="cpu")
    assert len(rows) == 4 * 2 * 6
    contexts = json.loads((tmp_path / "raw/context_context_results.json").read_text())
    assert len(contexts) == 12
    events = json.loads((tmp_path / "raw/context_seal_events.json").read_text())
    assert [r["event"] for r in events[:4]] == ["seal"] * 4
    assert all(r["event"] == "query_camera_GT" for r in events[4:])
    access = json.loads((tmp_path / "raw/context_GT_access.json").read_text())
    assert all(
        r["purpose"]
        in [
            "POST_SEAL_CONTEXT_EVALUATION_NOT_OPTIMIZER",
            "SEALED_QUERY_EVALUATION",
            "HASH_INPUT_AFTER_ALL_STATES_SEALED",
        ]
        for r in access
    )
    for key in {r["key"] for r in rows}:
        for fid in [8, 9]:
            rr = [r for r in rows if r["key"] == key and r["query_id"] == fid]
            assert len({r["query_camera_hash"] for r in rr}) == 1
            assert all((tmp_path / r["prediction_path"]).is_file() for r in rr)
            direct = next(r for r in rr if r["method"] == "direct")
            assert direct["prediction_change_from_direct"]["depth"]["rms"] == 0.0
            wrong = next(r for r in rr if r["method"] == "wrong_scene")
            assert wrong["donor_scene_id"] != wrong["scene_id"]
    integrity = json.loads((tmp_path / "raw/context_evaluation_integrity.json").read_text())
    assert integrity["all_states_immutable"] and integrity["all_frozen_inputs_unchanged"]
    assert not integrity["final_holdout_touched"] and not integrity["dynamic_ttt_run"]


def test_holdout_plan_rejected_without_media(tmp_path):
    specs = setup(tmp_path)
    specs[0]["scene_id"] = "held0"
    (tmp_path / "state_plan.json").write_text(json.dumps({"states": specs}))
    with pytest.raises(PermissionError, match="holdout"):
        evaluate_capacity(tmp_path, device="cpu")


def test_all_state_files_exist_but_phase_incomplete_denies_media_hashing(tmp_path, monkeypatch):
    setup(tmp_path)
    (tmp_path / "context_optimization_complete.json").unlink()
    original_sha = sha

    def guarded_sha(path):
        if str(path).endswith((".png", ".npy")):
            raise AssertionError("Query/media bytes hashed before complete phase")
        return original_sha(path)

    monkeypatch.setattr("mcss.mechanism_pilot.direct_capacity_evaluation.sha", guarded_sha)
    with pytest.raises(PermissionError, match="completion marker"):
        evaluate_capacity(tmp_path, device="cpu")


def test_state_completion_hash_mismatch_denies_evaluation(tmp_path):
    specs = setup(tmp_path)
    path = tmp_path / specs[0]["optimization_path"] / "completion.json"
    value = json.loads(path.read_text())
    value["state_file_sha256"] = "wrong"
    path.write_text(json.dumps(value))
    with pytest.raises(PermissionError, match="completion provenance"):
        evaluate_capacity(tmp_path, device="cpu")
