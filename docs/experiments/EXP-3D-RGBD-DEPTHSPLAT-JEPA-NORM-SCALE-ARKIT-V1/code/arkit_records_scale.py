"""ARKitScenes training records for train_jepa_arkit_scale.py (EXP-3D-RGBD-DEPTHSPLAT-JEPA-NORM-SCALE-ARKIT-V1).

TRAIN = the pilot's TRAIN records (EXP-3D-RGBD-ARKIT-DATA-V1, 182 scans) plus every record of four earlier ARKit
evaluation cohorts whose experiments have ended (CONFIRM, SELECT, SYSTEM, SYSTEM2: Training-order positions 360-999, 80
scans each); DEV = the pilot's 8 DEV records. Every training or DEV venue must be distinct and absent from both
evaluation cohorts of this experiment (ARKIT_VAL: Validation fold; ARKIT_SYSTEM3: positions 1000-1359); asserted on
every load.
"""
import json
from pathlib import Path

OUT = Path("/home/xmum26_zonghan/zonghan_o_mirror/measurement-scene-state/outputs")
PILOT = OUT / "EXP-3D-RGBD-ARKIT-DATA-V1/manifest_arkit_trainval.json"
ADDED = {"ARKIT_CONFIRM": OUT / "EXP-3D-RGBD-ARKIT-CONFIRM-DATA-V1/manifest_arkit_confirm.json",
         "ARKIT_SELECT": OUT / "EXP-3D-RGBD-ARKIT-SELECT-DATA-V1/manifest_arkit_select.json",
         "ARKIT_SYSTEM": OUT / "EXP-3D-RGBD-ARKIT-SYSTEM-DATA-V1/manifest_arkit_system.json",
         "ARKIT_SYSTEM2": OUT / "EXP-3D-RGBD-ARKIT-SYSTEM2-DATA-V1/manifest_arkit_system2.json"}
EVALUATION = (OUT / "EXP-3D-RGBD-ARKIT-VAL-DATA-V1/manifest_arkit_val80.json",
              OUT / "EXP-3D-RGBD-ARKIT-SYSTEM3-DATA-V1/manifest_arkit_system3.json")


def load_records():
    scenes = json.loads(PILOT.read_text())["scenes"]
    train = [r for r in scenes if r["split"] == "TRAIN"]
    dev = [r for r in scenes if r["split"] == "DEV"]
    for split, path in ADDED.items():
        added = json.loads(path.read_text())["scenes"]
        if len(added) != 80 or {r["split"] for r in added} != {split}:
            raise ValueError(f"unexpected {split} records")
        train += added
    held = set()
    for path in EVALUATION:
        held |= {r["physical_scene_id"] for r in json.loads(path.read_text())["scenes"]}
    venues = [r["physical_scene_id"] for r in train + dev]
    if len(set(venues)) != len(venues) or len({r["scene_id"] for r in train + dev}) != len(venues):
        raise ValueError("a venue or scene id appears twice in TRAIN/DEV")
    if set(venues) & held:
        raise ValueError("a training or DEV venue is in an evaluation cohort")
    if len(train) != 182 + 4 * 80 or len(dev) != 8:
        raise ValueError("unexpected ARKit TRAIN/DEV sizes")
    return train, dev
