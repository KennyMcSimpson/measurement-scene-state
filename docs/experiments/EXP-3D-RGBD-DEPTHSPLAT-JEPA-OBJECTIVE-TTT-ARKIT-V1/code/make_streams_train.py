"""EXP-3D-RGBD-DEPTHSPLAT-JEPA-OBJECTIVE-TTT-ARKIT-V1: training/streams_TRAIN.json, the four Core B stream frames of
every TRAIN record of EXP-3D-RGBD-ARKIT-DATA-V1 by the frozen stream_frame_ids rule (frame ids only). A record is listed
as missing when the rule gives no stream, the two roles disagree or a stream frame is a primary query.
Run from the repository root (CUDA hidden):  .venv/bin/python make_streams_train.py OUT
"""
import json
import sys
from collections import Counter
from pathlib import Path

from mcss.mechanism_pilot.rgbd_stream_write import stream_frame_ids

manifest = json.loads(Path("outputs/EXP-3D-RGBD-ARKIT-DATA-V1/manifest_arkit_trainval.json").read_text())
table, missing = {}, []
for r in manifest["scenes"]:
    if r["split"] != "TRAIN":
        continue
    a, b = stream_frame_ids(r, "A"), stream_frame_ids(r, "B")
    if a is None or a != b or set(a) & set(r["roles"]["primary_query"]):
        missing.append(r["scene_id"])
        continue
    table[r["scene_id"]] = list(a)
Path(sys.argv[1]).write_text(json.dumps({"rule": "stream_frame_ids(record, A) == (B), Core B rule, TRAIN records of "
                                                  "EXP-3D-RGBD-ARKIT-DATA-V1", "streams": table, "missing": missing},
                                        indent=0))
print("TRAIN streams", len(table), "missing", len(missing), Counter(tuple(v) for v in table.values()).most_common(3))
