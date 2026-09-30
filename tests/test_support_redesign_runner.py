"""Tiny actual-renderer oracle run and pre-model holdout firewall checks."""

import json

import numpy as np
import pytest
import torch
from PIL import Image

from mcss.dynamic.carrier import DynamicSceneCarrier
from mcss.dynamic.checkpoint import save_dynamic_checkpoint
from mcss.dynamic.config import WriteConfig
from mcss.dynamic.types import hash_scene_state
from mcss.dynamic.write_rule import DirectWriteRule
from mcss.mechanism_pilot.small_training import sha
from mcss.mechanism_pilot.spatial import carrier_config_for_spatial_mode
from mcss.mechanism_pilot.support_redesign_contracts import EXPOSED_SCENES
from mcss.mechanism_pilot.support_redesign_runner import run_development


def setup_manifests(tmp_path, *, media):
    dev_ids = [f"dev{i}" for i in range(8)]
    roles = {
        "TRAIN_SCENES": ["fixture_train"],
        "REDESIGN_DEV_SCENES": dev_ids,
        "REDESIGN_DEV_EXPOSED": list(EXPOSED_SCENES),
        "FINAL_HOLDOUT_SCENES": [f"holdout{i}" for i in range(8)],
    }
    (tmp_path / "scene_role_lock.json").write_text(json.dumps(roles))
    paths = []
    for cohort, ids in [("exposed", EXPOSED_SCENES), ("dev", dev_ids)]:
        scenes = []
        for index, sid in enumerate(ids):
            scene = {
                "scene_id": sid,
                "split": "dev",
                "roles": {"context_a": [0, 1, 2], "context_b": [0, 3, 4], "primary_query": [8, 9]},
                "frames": [],
            }
            if media:
                for fid in [0, 1, 2, 3, 4, 8, 9]:
                    rgb, depth = tmp_path / f"{sid}_{fid}.png", tmp_path / f"{sid}_{fid}.npy"
                    Image.fromarray(np.full((8, 8, 3), 70 + index * 10, np.uint8)).save(rgb)
                    np.save(depth, np.full((8, 8), 3.0, np.float32))
                    pose = np.eye(4)
                    angle = index * 0.13
                    pose[:3, :3] = [
                        [np.cos(angle), 0.0, np.sin(angle)],
                        [0.0, 1.0, 0.0],
                        [-np.sin(angle), 0.0, np.cos(angle)],
                    ]
                    pose[:3, 3] = [index * 4 + fid * 0.025, index * 2.0, 0.1]
                    scene["frames"].append(
                        {
                            "frame_id": fid,
                            "rgb": str(rgb),
                            "depth": str(depth),
                            "intrinsics": [[8.0, 0.0, 4.0], [0.0, 8.0, 4.0], [0.0, 0.0, 1.0]],
                            "c2w": pose.tolist(),
                        }
                    )
            scenes.append(scene)
        path = tmp_path / f"{cohort}.json"
        path.write_text(json.dumps({"image_size": [8, 8], "scenes": scenes}))
        paths.append(path)
    return paths[1], paths[0]


def test_holdout_manifest_refused_before_model_or_media(tmp_path, monkeypatch):
    dev, exposed = setup_manifests(tmp_path, media=False)
    payload = json.loads(dev.read_text())
    payload["scenes"][0]["scene_id"] = "holdout0"
    dev.write_text(json.dumps(payload))
    calls = []

    def forbidden(*args, **kwargs):
        calls.append("called")
        raise AssertionError("Model was reached before identity firewall")

    monkeypatch.setattr(
        "mcss.mechanism_pilot.support_redesign_runner.load_dynamic_checkpoint", forbidden
    )
    with pytest.raises(PermissionError):
        run_development(
            tmp_path, dev, exposed, tmp_path / "missing.pt", tmp_path / "run", device="cpu"
        )
    assert calls == []
    assert not (tmp_path / "run").exists()


def test_tiny_full_oracle_run_seals_audits_geometry_and_never_writes(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    dev, exposed = setup_manifests(tmp_path, media=True)
    config = carrier_config_for_spatial_mode("anchor-centered")
    checkpoint = tmp_path / "checkpoint.pt"
    save_dynamic_checkpoint(
        checkpoint,
        DynamicSceneCarrier(config),
        DirectWriteRule(config, WriteConfig()),
        phase="B",
        provenance={"training_scenes": ["fixture_train"]},
    )
    checkpoint_hash = sha(checkpoint)
    (tmp_path / "preregistration.json").write_text(
        json.dumps({"checkpoint_sha256": checkpoint_hash})
    )
    (tmp_path / "qualified_frame_role_lock.json").write_text(
        json.dumps(
            {
                "initial_scene_role_lock_sha256": sha(tmp_path / "scene_role_lock.json"),
                "manifest_hashes": {"dev_manifest.json": sha(dev)},
            }
        )
    )
    (tmp_path / "oracle_adequacy_clarification.json").write_text("{}")

    def no_writes(*args, **kwargs):
        raise AssertionError("Dynamic write in support-only protocol")

    monkeypatch.setattr(DirectWriteRule, "propose", no_writes)
    output = tmp_path / "run"
    rows = run_development(tmp_path, dev, exposed, checkpoint, output, device="cpu")
    assert len(rows) == 600
    states = torch.load(output / "sealed_states.pt", weights_only=False)
    assert len(states) == 180
    assert all(hash_scene_state(s.scene_state) == s.state_hash for s in states.values())
    access = json.loads((output / "access_log.json").read_text())
    first_oracle = next(
        i for i, r in enumerate(access) if r["kind"] == "ORACLE_GEOMETRY_GT_DEPTH_CAMERA"
    )
    assert len([r for r in access[:first_oracle] if r["kind"] == "SEALED_STATE"]) == 45
    assert all(
        r["state_key"].startswith("R0/")
        for r in access[:first_oracle]
        if r["kind"] == "SEALED_STATE"
    )
    first_eval = next(i for i, r in enumerate(access) if r["kind"] == "QUERY_RGB_DEPTH_GT")
    assert sum(r["kind"] == "SEALED_STATE" for r in access[:first_eval]) == 180
    assert all(
        r["stage"] == "PRESEAL_DIAGNOSTIC_EXCEPTION" and not r["deployable"]
        for r in access
        if r["kind"] == "ORACLE_GEOMETRY_GT_DEPTH_CAMERA"
    )
    plans = torch.load(output / "geometry_plans.pt", weights_only=False)
    for key, plan in plans.items():
        assert plan.points.shape == (128, 3) and plan.candidate_ids.shape == (128,)
        assert torch.all(plan.points >= plan.bounds[0]) and torch.all(plan.points <= plan.bounds[1])
        expected = (plan.points - plan.bounds[0]) / (plan.bounds[1] - plan.bounds[0]) * 2 - 1
        assert torch.allclose(plan.normalized_xyz, expected, atol=1e-6)
        assert torch.equal(states[key].scene_state.bounds[0], plan.bounds)
    for sid in sorted({r["scene_id"] for r in rows}):
        for query in [8, 9]:
            subset = [r for r in rows if r["scene_id"] == sid and r["query_id"] == query]
            assert len({r["query_camera_hash"] for r in subset}) == 1
            assert all(r["candidate_count"] == 128 for r in subset)
            assert all(r["donor_scene_id"] != sid for r in subset if r["method"] == "wrong_scene")
    assert len(list((output / "predictions").glob("*.npz"))) == 600
    integrity = json.loads((output / "integrity.json").read_text())
    assert integrity["parameters_unchanged"] and integrity["checkpoint_unchanged"]
    assert integrity["source_input_hashes_unchanged"] and integrity["writes"] == 0
    assert not integrity["final_holdout_access"] and not integrity["dynamic_ttt_run"]
    assert sha(checkpoint) == checkpoint_hash
