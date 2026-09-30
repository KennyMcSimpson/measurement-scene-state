"""Independent static evaluation identity, sealing and scene-statistic regressions."""

import json
from copy import deepcopy

import numpy as np
import pytest
import torch
from PIL import Image

from mcss.dynamic.carrier import DynamicSceneCarrier
from mcss.dynamic.checkpoint import save_dynamic_checkpoint
from mcss.dynamic.config import CarrierConfig, WriteConfig
from mcss.dynamic.types import hash_scene_state, hash_value
from mcss.dynamic.write_rule import DirectWriteRule
from mcss.measurements import FixedMeasurementRenderer
from mcss.mechanism_pilot.static_evaluation import (
    METRICS,
    run_static,
    summarize,
    validate_static_roles,
)


def scene_record(scene_id="scene0"):
    return {
        "scene_id": scene_id,
        "split": "train",
        "roles": {
            "context_a": [0, 1, 2],
            "context_b": [0, 3, 4],
            "primary_query": [8, 9],
            "secondary_query": [12],
        },
    }


@pytest.mark.parametrize(
    "kind",
    [
        "primary_leak",
        "secondary_leak",
        "cohort_overlap",
        "training_identity",
        "nondev",
        "missing_physical",
    ],
)
def test_role_and_unseen_identity_firewall(kind):
    row = scene_record()
    row.update(split="dev", physical_scene_id="asset0")
    trained = ["trained_other"]
    if kind == "primary_leak":
        row["roles"]["primary_query"] = [2, 9]
    elif kind == "secondary_leak":
        row["roles"]["secondary_query"] = [3]
    elif kind == "cohort_overlap":
        row["roles"]["secondary_query"] = [8]
    elif kind == "training_identity":
        trained = ["scene0"]
    elif kind == "nondev":
        row["split"] = "test"
    else:
        del row["physical_scene_id"]
    with pytest.raises((ValueError, PermissionError)):
        validate_static_roles(row, trained, unseen=True)


def test_scene_equal_summary_and_scene_bootstrap_not_query_weighted():
    rows = []
    for scene, n, a in [("many_queries", 7, 1.0), ("one_query", 1, 3.0)]:
        for query in range(n):
            for method, offset in [("A", 0.0), ("B", 0.2), ("anchor", 1.0), ("wrong_scene", 2.0)]:
                row = {
                    "scene_id": scene,
                    "query_id": query,
                    "cohort": "primary",
                    "method": method,
                    **dict.fromkeys(METRICS, a + offset),
                }
                for region in [
                    "common_visible",
                    "inside_volume_supported",
                    "inside_volume_unsupported",
                    "outside_volume",
                ]:
                    row[region] = None
                rows.append(row)
    result = summarize(rows)["primary"]
    assert result["methods"]["A"]["full_image"]["metrics"]["depth_absrel"]["mean"] == 2.0
    gain = result["paired_depth_absrel"]["anchor_minus_A"]
    assert gain["mean"] == 1.0 and gain["ci95"] == [1.0, 1.0]
    assert gain["unit"] == "scene" and gain["n_scenes"] == 2 and gain["draws"] == 10000
    assert result["methods"]["A"]["common_visible"]["metrics"]["depth_absrel"]["mean"] is None
    assert summarize(deepcopy(rows)) == summarize(rows)


def test_static_end_to_end_seals_before_queries_and_preserves_wrong_scene_camera(
    tmp_path, monkeypatch
):
    torch.set_num_threads(1)
    scenes = []
    for scene_index in range(3):
        scene = scene_record(f"scene{scene_index}")
        scene["frames"] = []
        angle = 0.2 * scene_index
        rotation = np.array(
            [
                [np.cos(angle), 0.0, np.sin(angle)],
                [0.0, 1.0, 0.0],
                [-np.sin(angle), 0.0, np.cos(angle)],
            ]
        )
        for fid in [0, 1, 2, 3, 4, 8, 9, 12]:
            rgb, depth = (
                tmp_path / f"{scene_index}_{fid}.png",
                tmp_path / f"{scene_index}_{fid}.npy",
            )
            Image.fromarray(np.full((4, 4, 3), 50 + 30 * scene_index, np.uint8)).save(rgb)
            np.save(depth, np.full((4, 4), 3.0, np.float32))
            pose = np.eye(4)
            pose[:3, :3] = rotation
            pose[:3, 3] = [scene_index * 4.0 + fid * 0.02, scene_index * 2.0, 0.4 * scene_index]
            scene["frames"].append(
                {
                    "frame_id": fid,
                    "rgb": str(rgb),
                    "depth": str(depth),
                    "intrinsics": [[4.0, 0.0, 2.0], [0.0, 4.0, 2.0], [0.0, 0.0, 1.0]],
                    "c2w": pose.tolist(),
                }
            )
        scenes.append(scene)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"image_size": [4, 4], "scenes": scenes}))
    config = CarrierConfig()
    checkpoint = tmp_path / "carrier.pt"
    save_dynamic_checkpoint(
        checkpoint,
        DynamicSceneCarrier(config),
        DirectWriteRule(config, WriteConfig()),
        phase="B",
        provenance={"training_scenes": [s["scene_id"] for s in scenes]},
    )
    calls = []
    original_forward = FixedMeasurementRenderer.forward

    def spy(renderer, state, camera, measurements):
        calls.append((hash_scene_state(state), hash_value(camera), camera.c2w.clone()))
        return original_forward(renderer, state, camera, measurements)

    def no_writes(*args, **kwargs):
        raise AssertionError("Static protocol invoked a dynamic write")

    monkeypatch.setattr(FixedMeasurementRenderer, "forward", spy)
    monkeypatch.setattr(DirectWriteRule, "propose", no_writes)
    output = tmp_path / "results"
    rows = run_static(manifest, checkpoint, output, device="cpu")
    assert len(rows) == 3 * 3 * 8
    events = json.loads((output / "sealed_query_access.json").read_text())
    assert all(e["event"] == "seal" for e in events[:18])
    assert all(e["event"] != "seal" for e in events[18:])
    stored = torch.load(output / "sealed_states.pt", weights_only=False)
    state_manifest = json.loads((output / "state_manifest.json").read_text())
    for key, state in stored.items():
        assert hash_scene_state(state.scene_state) == state_manifest[key]["state_hash"]
        assert not set(state.observed_ids).intersection([8, 9, 12])
    rendered_rows = [row for row in rows if row["method"] != "prior"]
    assert len(calls) == len(rendered_rows)
    for index, scene in enumerate(scenes):
        sid = scene["scene_id"]
        for query in [8, 9, 12]:
            group = [r for r in rows if r["scene_id"] == sid and r["query_id"] == query]
            assert len({r["query_camera_hash"] for r in group}) == 1
            paired_calls = {
                r["method"]: call
                for r, call in zip(rendered_rows, calls, strict=True)
                if r["scene_id"] == sid and r["query_id"] == query
            }
            assert torch.equal(paired_calls["A"][2], paired_calls["wrong_scene"][2])
            assert (
                paired_calls["wrong_scene"][0]
                == stored[f"{scenes[(index + 1) % 3]['scene_id']}/A"].state_hash
            )
            for row in group:
                counts = row["support_bucket_counts"]
                assert (
                    sum(
                        counts[k]
                        for k in [
                            "inside_volume_supported",
                            "inside_volume_unsupported",
                            "outside_volume",
                        ]
                    )
                    == row["depth_valid_count"]
                )
    integrity = json.loads((output / "integrity.json").read_text())
    assert integrity["states_sealed_before_query"] and integrity["writes"] == 0
    assert integrity["inputs_unchanged"] and not integrity["dynamic_ttt_run"]
