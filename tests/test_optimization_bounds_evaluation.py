"""Synthetic fixtures only: never load formal exposed/holdout media."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from mcss.dynamic.types import hash_scene_state
from mcss.mechanism_pilot.optimization_bounds_evaluation import evaluate_optimization_bounds
from mcss.mechanism_pilot.small_training import sha
from mcss.types import SceneState


def fixture(root, phase="context"):
    scenes, chains, specs = [], [], []
    bounds = [[-6.0, -4.0, -6.0], [6.0, 4.0, 6.0]]
    for i in range(2):
        sid = f"scene{i}"
        scene = {
            "scene_id": sid,
            "cohort_role": "CAPACITY_DEV_EXPOSED",
            "roles": {"context_a": [0, 1, 2], "context_b": [0, 3, 4], "primary_query": [8, 9]},
            "frames": [],
        }
        for fid in (0, 1, 2, 3, 4, 8, 9):
            rgb = root / f"{sid}_{fid}.png"
            depth = root / f"{sid}_{fid}.npy"
            Image.fromarray(np.full((4, 4, 3), 50 + i * 20, np.uint8)).save(rgb)
            np.save(depth, np.full((4, 4), 3.0, np.float32))
            pose = torch.eye(4)
            pose[0, 3] = i * 2 + fid * 0.01
            scene["frames"].append(
                {
                    "frame_id": fid,
                    "rgb": str(rgb),
                    "depth": str(depth),
                    "intrinsics": [[4.0, 0.0, 1.5], [0.0, 4.0, 1.5], [0.0, 0.0, 1.0]],
                    "c2w": pose.tolist(),
                }
            )
        scenes.append(scene)
        roles = ("joint_query",) if phase == "oracle" else ("A", "B")
        for role in roles:
            chain_key = f"{sid}_{role}"
            folder = root / "checkpoints" / chain_key
            folder.mkdir(parents=True)
            chains.append(
                {
                    "key": chain_key,
                    "scene_id": sid,
                    "phase": phase,
                    "optimization_path": str(folder.relative_to(root)),
                }
            )
            for budget in (1000, 3000):
                for selection in ("FIXED_BUDGET", "CONTEXT_SELECTED"):
                    key = f"{chain_key}_{budget}_{selection}"
                    dest = folder / key
                    dest.mkdir()
                    state = SceneState(
                        torch.full((1, 1, 16, 16, 16), -2.0 + i + budget / 10000),
                        torch.full((1, 3, 16, 16, 16), 0.2 + i * 0.1),
                        torch.full((1, 1, 16, 16, 16), -3.0),
                        torch.tensor([bounds]),
                    )
                    torch.save(state, dest / "state.pt")
                    (dest / "summary.json").write_text(
                        json.dumps({"state_hash": hash_scene_state(state)})
                    )
                    specs.append(
                        {
                            "key": key,
                            "chain_key": chain_key,
                            "scene_id": sid,
                            "phase": phase,
                            "role": role,
                            "track": {
                                "context": "RGBD",
                                "oracle": "QUERY_ORACLE",
                                "secondary": "RGB_ONLY",
                            }[phase],
                            "grid": 16,
                            "samples": 64,
                            "bounds_mode": "CURRENT_BOUNDS",
                            "bounds": bounds,
                            "budget": budget,
                            "selection": selection,
                            "state_path": str((dest / "state.pt").relative_to(root)),
                            "optimization_path": str(dest.relative_to(root)),
                        }
                    )
    (root / "scene_manifest.json").write_text(
        json.dumps(
            {"scenes": scenes, "image_size": [4, 4], "FINAL_HOLDOUT_PROHIBITED": ["holdout"]}
        )
    )
    (root / "state_plan.json").write_text(json.dumps({"chains": chains, "states": specs}))
    sources = {str(Path(__file__).resolve()): sha(Path(__file__))}
    (root / "config.json").write_text(json.dumps({"source_sha256": sources}))
    for chain in chains:
        states = {}
        for spec in specs:
            if spec["chain_key"] != chain["key"]:
                continue
            folder = root / spec["optimization_path"]
            states[spec["key"]] = {
                "state_file_sha256": sha(folder / "state.pt"),
                "summary_sha256": sha(folder / "summary.json"),
                "state_hash": json.loads((folder / "summary.json").read_text())["state_hash"],
            }
        (root / chain["optimization_path"] / "completion.json").write_text(
            json.dumps(
                {
                    "phase": phase,
                    "chain_key": chain["key"],
                    "plan_sha256": sha(root / "state_plan.json"),
                    "config_sha256": sha(root / "config.json"),
                    "source_hashes": sources,
                    "states": states,
                }
            )
        )
    (root / f"{phase}_optimization_complete.json").write_text(
        json.dumps(
            {
                "status": "PASS",
                "phase": phase,
                "states": len(specs),
                "input_hashes": {
                    str(root / n): sha(root / n)
                    for n in ["scene_manifest.json", "state_plan.json", "config.json"]
                },
            }
        )
    )
    return specs, chains


def test_phase_complete_required_before_media(tmp_path, monkeypatch):
    fixture(tmp_path)
    (tmp_path / "context_optimization_complete.json").unlink()

    def denied(*args, **kwargs):
        raise AssertionError("query reached")

    monkeypatch.setattr(
        "mcss.mechanism_pilot.optimization_bounds_evaluation.CapacityEvaluator.query", denied
    )
    with pytest.raises(PermissionError, match="completion marker"):
        evaluate_optimization_bounds(tmp_path, device="cpu")


def test_chain_state_or_summary_mutation_blocks_before_media(tmp_path):
    specs, _ = fixture(tmp_path)
    summary = tmp_path / specs[0]["optimization_path"] / "summary.json"
    summary.write_text(summary.read_text() + " ")
    with pytest.raises(PermissionError, match="changed after chain completion"):
        evaluate_optimization_bounds(tmp_path, device="cpu")


def test_all_budget_states_sealed_and_wrong_donor_matched(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    specs, _ = fixture(tmp_path)
    from mcss.mechanism_pilot.direct_capacity_contracts import StateSealBarrier

    count = []
    original = StateSealBarrier.seal

    def observe(self, *args):
        original(self, *args)
        count.append(1)

    def checked_sha(path):
        if Path(path).suffix in (".png", ".npy"):
            assert len(count) == len(specs)
        return sha(path)

    monkeypatch.setattr(StateSealBarrier, "seal", observe)
    monkeypatch.setattr("mcss.mechanism_pilot.optimization_bounds_evaluation.sha", checked_sha)
    rows = evaluate_optimization_bounds(tmp_path, device="cpu")
    assert len(rows) == len(specs) * 2 * 2
    expected = {
        s["key"]: json.loads((tmp_path / s["optimization_path"] / "summary.json").read_text())[
            "state_hash"
        ]
        for s in specs
    }
    for s in specs:
        for fid in (8, 9):
            values = [r for r in rows if r["key"] == s["key"] and r["query_id"] == fid]
            assert len({r["query_camera_hash"] for r in values}) == 1
            direct = next(r for r in values if r["method"] == "direct")
            wrong = next(r for r in values if r["method"] == "wrong_scene")
            assert direct["used_state_hash"] == expected[s["key"]]
            donor = next(
                x
                for x in specs
                if x["scene_id"] == wrong["donor_scene_id"]
                and all(
                    x[k] == s[k] for k in ("budget", "selection", "role", "track", "bounds_mode")
                )
            )
            assert wrong["used_state_hash"] == expected[donor["key"]]
    assert (
        len(json.loads((tmp_path / "raw/context_context_results.json").read_text()))
        == len(specs) * 3
    )


def test_oracle_shares_both_queries_and_has_no_wrong_scene(tmp_path):
    torch.set_num_threads(1)
    specs, _ = fixture(tmp_path, phase="oracle")
    rows = evaluate_optimization_bounds(tmp_path, phase="oracle", device="cpu")
    assert len(rows) == len(specs) * 2
    assert {r["method"] for r in rows} == {"direct"}
    for s in specs:
        assert len({r["used_state_hash"] for r in rows if r["key"] == s["key"]}) == 1


def test_holdout_guard_rejects_before_source_or_media_access(tmp_path):
    fixture(tmp_path)
    path = tmp_path / "scene_manifest.json"
    manifest = json.loads(path.read_text())
    manifest["FINAL_HOLDOUT_PROHIBITED"].append("scene0")
    path.write_text(json.dumps(manifest))
    with pytest.raises(PermissionError, match="holdout stays closed"):
        evaluate_optimization_bounds(tmp_path, device="cpu")
    assert not (tmp_path / "raw").exists()


def test_secondary_depth_metrics_are_post_seal_only(tmp_path):
    torch.set_num_threads(1)
    specs, _ = fixture(tmp_path, phase="secondary")
    rows = evaluate_optimization_bounds(tmp_path, phase="secondary", device="cpu")
    assert len(rows) == len(specs) * 4
    access = json.loads((tmp_path / "raw/secondary_GT_access.json").read_text())
    assert all(
        event["purpose"]
        in (
            "HASH_INPUT_AFTER_ALL_STATES_SEALED",
            "POST_SEAL_CONTEXT_EVALUATION_NOT_OPTIMIZER",
            "SEALED_QUERY_EVALUATION",
        )
        for event in access
    )
    assert all(row["track"] == "RGB_ONLY" for row in rows)
