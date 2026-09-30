import json
from copy import deepcopy

import numpy as np
import pytest
from PIL import Image

from mcss.mechanism_pilot.data_prep import resize_rgb_depth
from mcss.mechanism_pilot.rgb_audit import (
    audit_rgb_chain,
    classify_black_frame,
    rgb_statistics,
)


def test_raw_statistics_preserved_before_preprocessing(tmp_path):
    raw = tmp_path / "raw/ai_003_001/images/scene_cam_00_final_preview"
    prepared = tmp_path / "prepared/ai_003_001"
    raw.mkdir(parents=True)
    prepared.mkdir(parents=True)
    frames = []
    for fid in (0, 1):
        image = np.zeros((7, 9, 3), dtype=np.uint8)
        if fid:
            image[:, :4] = [255, 130, 70]
        source = raw / f"frame.{fid:04d}.color.jpg"
        Image.fromarray(image).save(source)
        with Image.open(source) as im:
            original = np.array(im.convert("RGB"))
        resized, _ = resize_rgb_depth(original, np.ones((7, 9)), (3, 4))
        target = prepared / f"{fid:04d}.png"
        Image.fromarray(resized).save(target)
        frames.append({"frame_id": fid, "rgb": str(target)})
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"image_size": [3, 4], "scenes": [{"scene_id": "ai_003_001", "frames": frames}]})
    )
    output = tmp_path / "audit"
    result = audit_rgb_chain(manifest, output)
    rows = json.loads((output / "frame_audit.json").read_text())
    assert result["ROOT_CAUSE"] == "SOURCE_FRAME_IS_BLACK"
    assert result["black_frames"] == [0]
    assert rows[1]["source_rgb"] == rgb_statistics(original)
    assert rows[1]["source_rgb"]["dimensions_hw"] == [7, 9]
    assert rows[1]["prepared_rgb"]["dimensions_hw"] == [3, 4]
    assert all(r["prepared_matches_recomputed_resize"] for r in rows)
    assert all(r["model_evaluator_tensors_equal"] for r in rows)
    assert len((output / "ai003_frame_audit.csv").read_text().splitlines()) == 11
    with pytest.raises(FileExistsError):
        audit_rgb_chain(manifest, output)


def test_stats_distinguish_black_pixels_from_zero_channels_and_invalid():
    image = np.array([[[0, 0, 0], [0, 255, 0], [1, 1, 1], [2, 2, 2]]], np.uint8)
    stats = rgb_statistics(image)
    assert stats["exact_zero_pixels_percent"] == 25
    assert stats["near_black_pixels_percent"] == 50
    assert stats["unique_rgb_values"] == 4
    invalid = rgb_statistics(np.array([[[np.nan, np.inf, 0]]]), scale=1)
    assert invalid["nan_values"] == invalid["inf_values"] == 1
    assert invalid["exact_zero_pixels_percent"] == 0


@pytest.mark.parametrize(
    "field,value,expected",
    [
        ("frame_mapping_valid", False, "FRAME_MAPPING_ERROR"),
        ("decoder_black_agreement", False, "DECODER_ERROR"),
        ("prepared_matches_recomputed_resize", False, "PREPROCESSING_ERROR"),
    ],
)
def test_quality_root_cause_independent_of_model_score(field, value, expected):
    row = {
        "frame_mapping_valid": True,
        "decoder_black_agreement": True,
        "prepared_matches_recomputed_resize": True,
        "source_rgb": {"max": 0},
        "prepared_rgb": {"max": 0},
        "model_input": {"min": 0, "max": 0, "nan_values": 0, "inf_values": 0},
    }
    assert classify_black_frame(row) == "SOURCE_FRAME_IS_BLACK"
    changed = deepcopy(row)
    changed[field] = value
    assert classify_black_frame(changed) == expected
    changed["model_score"] = 999
    assert classify_black_frame(changed) == expected
