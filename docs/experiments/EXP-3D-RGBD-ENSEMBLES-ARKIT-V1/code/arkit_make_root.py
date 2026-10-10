"""EXP-3D-RGBD-ENSEMBLES-ARKIT-V1: evaluation root on the ARKitScenes Validation cohort (80 real iPad RGB-D scans,
outputs/EXP-3D-RGBD-ARKIT-VAL-DATA-V1/manifest_arkit_val80.json, copied), the G1 arms contract (three seeds) and the
frame-role lock (no prediction is made and no query depth is read here). Episodes: every record x role (A, B);
context = the role's frames; stream = the Core B frame-id rule (4 frames evenly spaced over the free frames, shared
by both roles); queries = primary_query. Run through cv_host_run.py from the repository root:
  python arkit_make_root.py ROOT
"""
import hashlib
import json
import shutil
import sys
from pathlib import Path

from mcss.mechanism_pilot.rgbd_stream_write import stream_frame_ids

REPO = Path.cwd()
SOURCE = REPO / "outputs/EXP-3D-RGBD-ARKIT-VAL-DATA-V1/manifest_arkit_val80.json"
G1_ROOT = REPO / "outputs/EXP-3D-RGBD-SEGMENT-DIVERSITY-V21/G1/EXP-3D-RGBD-LONG-TRAIN-V17"
NAME = "ARKIT_VAL"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


root = Path(sys.argv[1]).resolve()
root.mkdir(parents=True, exist_ok=False)
manifest_path = root / "manifest_arkit_val80.json"
shutil.copyfile(SOURCE, manifest_path)
records = json.loads(manifest_path.read_text())["scenes"]
if len(records) != 80 or any(r["split"] != NAME for r in records):
    raise ValueError("unexpected ARKit manifest")
if len({r["physical_scene_id"] for r in records}) != 80:
    raise ValueError("venues must be distinct")
episodes = []
for r in records:
    stream = stream_frame_ids(r, "A")
    if stream is None or stream != stream_frame_ids(r, "B"):
        raise ValueError(f"{r['scene_id']}: no shared four-frame stream")
    for role, key in (("A", "context_a"), ("B", "context_b")):
        context, queries = list(r["roles"][key]), list(r["roles"]["primary_query"])
        if set(stream) & (set(context) | set(queries)) or set(context) & set(queries):
            raise ValueError(f"{r['scene_id']}/{role}: frame roles overlap")
        episodes.append({"scene_id": r["scene_id"], "role": role, "context": context, "stream": list(stream),
                         "queries": queries})
v17 = json.loads((REPO / "outputs/EXP-3D-RGBD-LONG-TRAIN-V17/training_contract.json").read_text())["eval"]
contract = {
    "arms": {"G1": {"module": "mcss.mechanism_pilot.rgbd_long_train_carrier", "root": str(G1_ROOT), "step": 24000,
                    "preregistration_sha256": sha(G1_ROOT / "preregistration.json")}},
    "reference_arm": "G1", "seeds": [20260928, 20260929, 20260930],
    "cohorts": {NAME: {"manifest": str(manifest_path), "manifest_sha256": sha(manifest_path), "split": NAME,
                       "role": "primary_query"}},
    "near_px": v17["near_px"], "reprojection_fill_constant_m": v17["reprojection_fill_constant_m"],
    "abstain_opacity_threshold": v17["abstain_opacity_threshold"],
}
(root / "evaluation_contract.json").write_text(json.dumps(contract, indent=2))
lock = {"rule": "context = role frames; stream = Core B frame-id rule (stream_frame_ids, 4 frames, shared by roles); "
                "queries = primary_query",
        "cohorts": {NAME: {"manifest": str(manifest_path), "manifest_sha256": sha(manifest_path), "episodes": episodes}},
        "source_manifest": {"path": str(SOURCE), "sha256": sha(SOURCE)}}
(root / "frame_role_lock.json").write_text(json.dumps(lock, indent=1))
print("ARKIT_ROOT", root, len(records), "records", len(episodes), "episodes")
