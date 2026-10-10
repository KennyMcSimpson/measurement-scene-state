"""EXP-3D-RGBD-TTT-ENSEMBLE-V1: evaluation root on EVAL34 with the Core B streams (no prediction is made and no query
depth is read here). EVAL34 = the DepthSplat probe's combined EVAL-V3 + EVAL-V4 manifest (copied). Episodes: every
record x role (A, B); context = the role's frames; stream = the Core B frame-id rule (4 frames, shared by both roles),
which must equal the stream roles sealed by Core B V8 (EXP-3D-RGBD-DYNAMIC-TTT-SOURCE-V8); queries = primary_query.
Asserts disjointness from TRAIN72, SEG-V1, DEV and the protected holdout. Run through cv_host_run.py from the
repository root:  python tttens_make_root.py ROOT
"""
import hashlib
import json
import shutil
import sys
from pathlib import Path

from mcss.mechanism_pilot.rgbd_stream_write import stream_frame_ids

REPO = Path.cwd()
SOURCE = Path.home() / "cv_work/depthsplat/probe_eval34/manifest_eval34.json"
V8 = REPO / "outputs/EXP-3D-RGBD-DYNAMIC-TTT-SOURCE-V8/stream_roles.json"
NAME = "EVAL34"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


root = Path(sys.argv[1]).resolve()
root.mkdir(parents=True, exist_ok=False)
manifest_path = root / "manifest_eval34.json"
shutil.copyfile(SOURCE, manifest_path)
records = json.loads(manifest_path.read_text())["scenes"]
split = json.loads((REPO / "outputs/EXP-3D-RGBD-DEPTH-BOUNDS-V7/scene_split.json").read_text())
train = [r for r in split["scenes"] if r["split"] == "TRAIN"]
dev = json.loads((Path.home() / "cv_work/depthsplat/probe_dev/manifest_dev.json").read_text())["scenes"]
used = {r["scene_id"] for r in train + dev} | set(split["protected_scene_ids"])
used_physical = {r["physical_scene_id"] for r in train + dev}
if {r["scene_id"] for r in records} & used or {r["physical_scene_id"] for r in records} & used_physical:
    raise ValueError("an EVAL34 record overlaps TRAIN72 (and so SEG-V1), DEV or the protected holdout")
if any(r["split"] != NAME for r in records) or len(records) != 83:
    raise ValueError("unexpected EVAL34 manifest")
sealed = json.loads(V8.read_text())["episodes"]
episodes = []
for r in records:
    stream = stream_frame_ids(r, "A")
    if stream is None or stream != stream_frame_ids(r, "B"):
        raise ValueError(f"{r['scene_id']}: no shared four-frame stream")
    for role, key in (("A", "context_a"), ("B", "context_b")):
        if sealed.get(f"{r['scene_id']}/{role}") != list(stream):
            raise ValueError(f"{r['scene_id']}/{role}: stream differs from Core B V8")
        context, queries = list(r["roles"][key]), list(r["roles"]["primary_query"])
        if set(stream) & (set(context) | set(queries)):
            raise ValueError(f"{r['scene_id']}/{role}: frame roles overlap")
        episodes.append({"scene_id": r["scene_id"], "role": role, "context": context, "stream": list(stream),
                         "queries": queries})
if len(sealed) != len(episodes):
    raise ValueError("Core B V8 stream roles cover other episodes")
lock = {"rule": "context = role frames; stream = Core B frame-id rule (equal to Core B V8's sealed stream roles); "
                "queries = primary_query",
        "cohorts": {NAME: {"manifest": str(manifest_path), "manifest_sha256": sha(manifest_path), "episodes": episodes}},
        "source_manifest": {"path": str(SOURCE), "sha256": sha(SOURCE)},
        "stream_roles": {"path": str(V8), "sha256": sha(V8)}}
(root / "frame_role_lock.json").write_text(json.dumps(lock, indent=1))
print("TTT_ENSEMBLE_ROOT", root, len(records), "records", len(episodes), "episodes")
