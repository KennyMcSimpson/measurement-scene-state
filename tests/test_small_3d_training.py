import json
from copy import deepcopy

import numpy as np
import pytest
import torch
from PIL import Image

from mcss.dynamic.checkpoint import load_dynamic_checkpoint
from mcss.mechanism_pilot.small_training import (
    SmallTrainingConfig,
    TrainScene,
    run_small_training,
    schedule,
    validate_manifest,
)


def manifest_fixture(tmp_path):
    scenes = []
    for i in range(3):
        frames = []
        for frame_id in range(16):
            rgb = tmp_path / f"{i}_{frame_id}.png"
            depth = tmp_path / f"{i}_{frame_id}.npy"
            Image.fromarray(np.full((8, 8, 3), 80 + i * 10, dtype=np.uint8)).save(rgb)
            np.save(depth, np.full((8, 8), 3.0, dtype=np.float32))
            frames.append(
                {
                    "frame_id": frame_id,
                    "rgb": str(rgb),
                    "depth": str(depth),
                    "intrinsics": [[8.0, 0.0, 4.0], [0.0, 8.0, 4.0], [0.0, 0.0, 1.0]],
                    "c2w": np.eye(4).tolist(),
                }
            )
        scenes.append(
            {
                "scene_id": f"fixture{i}",
                "split": "train",
                "frames": frames,
                "roles": {
                    "context_a": [0, 1, 2],
                    "context_b": [0, 3, 4],
                    "stream": [5, 6],
                    "query": [12, 13, 14, 15],
                },
            }
        )
    return {"schema": "mcss.small_training.data.v1", "image_size": [8, 8], "scenes": scenes}


def test_schedule_does_not_confound_scene_action():
    per_scene = {i: set() for i in range(3)}
    for step in range(36):
        scene, query, action = schedule(step, 3)
        per_scene[scene].add((query, action))
    assert all(len(pairs) == 12 for pairs in per_scene.values())


def test_train_manifest_rejects_dev_query_reuse_and_duplicates(tmp_path):
    manifest = manifest_fixture(tmp_path)
    validate_manifest(manifest, engineering_fixture=True)
    bad = deepcopy(manifest)
    bad["scenes"][0]["split"] = "dev"
    with pytest.raises(ValueError, match="train scenes"):
        validate_manifest(bad, engineering_fixture=True)
    bad = deepcopy(manifest)
    bad["scenes"][0]["roles"]["query"][0] = 0
    with pytest.raises(ValueError, match="Frame roles"):
        validate_manifest(bad, engineering_fixture=True)
    bad = deepcopy(manifest)
    bad["scenes"][1]["scene_id"] = bad["scenes"][0]["scene_id"]
    with pytest.raises(ValueError, match="distinct"):
        validate_manifest(bad, engineering_fixture=True)
    scene = TrainScene(manifest["scenes"][0], (8, 8), tmp_path, "cpu", [])
    with pytest.raises(ValueError, match="cannot construct"):
        scene.observations("query")


@pytest.mark.parametrize("spatial_mode", ["legacy-forward", "anchor-centered"])
def test_two_stage_training_roundtrip_and_label_contract(tmp_path, spatial_mode):
    from mcss.mechanism_pilot.spatial import carrier_config_for_spatial_mode

    carrier_config = carrier_config_for_spatial_mode(spatial_mode)
    torch.set_num_threads(1)
    manifest = manifest_fixture(tmp_path)
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    output = tmp_path / "run"
    summary = run_small_training(
        path,
        output,
        device="cpu",
        engineering_fixture=True,
        carrier_config=carrier_config,
        config=SmallTrainingConfig(
            stage_a_steps=1, stage_b_steps=1, renderer_samples=4, ray_chunk_size=64
        ),
    )
    assert summary["source_unchanged"] and summary["data_unchanged"]
    assert summary["n_dev_scenes"] == summary["n_test_scenes"] == 0
    assert len(json.loads((output / "train_evaluation.json").read_text())) == 36
    initial, initial_writer, _ = load_dynamic_checkpoint(output / "initial.pt")
    final, final_writer, _ = load_dynamic_checkpoint(output / "phase_b_final.pt")
    assert initial.config == final.config == carrier_config
    assert json.loads((output / "lock.json").read_text())["carrier"]["local_bounds_m"] == [
        list(v) for v in carrier_config.local_bounds_m
    ]
    assert any(not torch.equal(v, final.state_dict()[k]) for k, v in initial.state_dict().items())
    assert any(
        not torch.equal(v, final_writer.state_dict()[k])
        for k, v in initial_writer.state_dict().items()
    )
    rows = [json.loads(line) for line in (output / "training.jsonl").read_text().splitlines()]
    assert len(rows) == 2 and rows[1]["writer_gradient_norm_after_clip"] > 0
    access = json.loads((output / "label_access.json").read_text())
    assert all(row["split"] == "train" for row in access)
    assert all(row["frame_id"] >= 12 for row in access if "depth" in row["kind"])
    with pytest.raises(FileExistsError):
        run_small_training(path, output, device="cpu")
