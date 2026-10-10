"""Probe root for the DepthSplat zero-shot check on DEV (exploratory, development data): a DEV-only
copy of the V7 split and an arms contract with G1 (V21) at step 24000, three seeds."""
import hashlib
import json
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
root.mkdir(parents=True, exist_ok=False)
split = json.loads(Path("outputs/EXP-3D-RGBD-DEPTH-BOUNDS-V7/scene_split.json").read_text())
manifest = {key: split[key] for key in ("image_size", "protected_scene_ids")}
manifest["scenes"] = [r for r in split["scenes"] if r["split"] == "DEV"]
(root / "manifest_dev.json").write_text(json.dumps(manifest))
v17 = json.loads(Path("outputs/EXP-3D-RGBD-LONG-TRAIN-V17/training_contract.json").read_text())["eval"]
g1 = Path("outputs/EXP-3D-RGBD-SEGMENT-DIVERSITY-V21/G1/EXP-3D-RGBD-LONG-TRAIN-V17").resolve()
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
contract = {
    "arms": {"G1": {"module": "mcss.mechanism_pilot.rgbd_long_train_carrier", "root": str(g1),
                    "step": 24000, "preregistration_sha256": sha(g1 / "preregistration.json")}},
    "reference_arm": "G1",
    "seeds": [20260928, 20260929, 20260930],
    "cohorts": {"DEV": {"manifest": str(root / "manifest_dev.json"),
                        "manifest_sha256": sha(root / "manifest_dev.json"), "split": "DEV",
                        "role": "primary_query"}},
    "near_px": v17["near_px"],
    "reprojection_fill_constant_m": v17["reprojection_fill_constant_m"],
    "abstain_opacity_threshold": v17["abstain_opacity_threshold"],
}
(root / "evaluation_contract.json").write_text(json.dumps(contract, indent=2))
print("PROBE_ROOT", root, len(manifest["scenes"]), "DEV scenes")
