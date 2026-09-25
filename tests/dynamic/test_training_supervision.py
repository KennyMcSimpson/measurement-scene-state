import json
import math

import numpy as np
import pytest
import torch
from PIL import Image

from mcss.training.smoke import DynamicTrainingSmokeConfig
from mcss.training.supervision import TrainingSupervision


def _write_query_manifest(tmp_path, *, split_id: str = "train", frame_id: int = 9):
    rgb_path = tmp_path / "query.png"
    depth_path = tmp_path / "query.npy"
    Image.fromarray(np.full((4, 6, 3), 127, dtype=np.uint8)).save(rgb_path)
    np.save(depth_path, np.full((4, 6), 2.0, dtype=np.float32))
    manifest = {
        "schema_version": "mcss.dynamic.query_vault.v1",
        "episode_id": "train--scene-a--cam-00",
        "scene_id": "scene-a",
        "split_id": split_id,
        "query_vault_id": "qv-training",
        "image_size": [4, 6],
        "query": [
            {
                "frame_id": frame_id,
                "intrinsics": [[4.0, 0.0, 2.5], [0.0, 4.0, 1.5], [0.0, 0.0, 1.0]],
                "c2w": np.eye(4).tolist(),
                "rgb": str(rgb_path),
                "depth": str(depth_path),
            }
        ],
    }
    path = tmp_path / "query.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def test_training_supervision_reads_only_matching_train_query(tmp_path) -> None:
    path = _write_query_manifest(tmp_path)

    supervision = TrainingSupervision.from_query_file(
        path,
        episode_id="train--scene-a--cam-00",
        scene_id="scene-a",
        query_vault_id="qv-training",
        context_frame_ids=(0, 1, 2, 3),
    )

    assert supervision.frame_ids == (9,)
    assert supervision.query_cameras.leading_shape == (1, 1)
    assert supervision.query_rgb.shape == (1, 1, 3, 4, 6)
    assert supervision.query_depth.shape == (1, 1, 1, 4, 6)
    torch.testing.assert_close(supervision.query_depth, torch.full((1, 1, 1, 4, 6), 2.0))


@pytest.mark.parametrize(
    ("split_id", "context_frame_ids", "error"),
    [
        ("dev", (), "training split"),
        ("train", (9,), "disjoint"),
    ],
)
def test_training_supervision_rejects_nontrain_or_overlapping_query(
    tmp_path, split_id, context_frame_ids, error
) -> None:
    path = _write_query_manifest(tmp_path, split_id=split_id)

    with pytest.raises(ValueError, match=error):
        TrainingSupervision.from_query_file(
            path,
            episode_id="train--scene-a--cam-00",
            scene_id="scene-a",
            query_vault_id="qv-training",
            context_frame_ids=context_frame_ids,
        )


@pytest.mark.parametrize("learning_rate", [math.nan, math.inf, -math.inf, 0.0])
def test_training_smoke_config_rejects_nonfinite_or_nonpositive_learning_rate(
    learning_rate,
) -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        DynamicTrainingSmokeConfig(learning_rate=learning_rate)
