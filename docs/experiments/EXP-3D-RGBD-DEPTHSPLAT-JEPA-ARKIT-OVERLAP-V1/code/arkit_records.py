"""ARKitScenes training records for train_jepa_arkit.py (EXP-3D-RGBD-DEPTHSPLAT-JEPA-ARKIT-V1).

TRAIN and DEV are the pilot's Training-fold records (EXP-3D-RGBD-ARKIT-DATA-V1/manifest_arkit_trainval.json: the first
360 venues of the seeded Training order; the first 8 eligible scans are DEV, the rest TRAIN). Every training or DEV
venue must be absent from both ARKit evaluation cohorts (ARKIT_VAL80: Validation fold; ARKIT_CONFIRM: Training-order
positions 360-519); asserted on every load.
"""
import json
from pathlib import Path

REPO = Path("/home/xmum26_zonghan/zonghan_o_mirror/measurement-scene-state")
MANIFEST = REPO / "outputs/EXP-3D-RGBD-ARKIT-DATA-V1/manifest_arkit_trainval.json"
EVALUATION = (REPO / "outputs/EXP-3D-RGBD-ARKIT-VAL-DATA-V1/manifest_arkit_val80.json",
              REPO / "outputs/EXP-3D-RGBD-ARKIT-CONFIRM-DATA-V1/manifest_arkit_confirm.json")


def load_records():
    scenes = json.loads(MANIFEST.read_text())["scenes"]
    train = [r for r in scenes if r["split"] == "TRAIN"]
    dev = [r for r in scenes if r["split"] == "DEV"]
    held = set()
    for path in EVALUATION:
        held |= {r["physical_scene_id"] for r in json.loads(path.read_text())["scenes"]}
    if {r["physical_scene_id"] for r in train + dev} & held:
        raise ValueError("a training or DEV venue is in an ARKit evaluation cohort")
    if not train or len(dev) != 8:
        raise ValueError("unexpected ARKit TRAIN/DEV records")
    return train, dev
