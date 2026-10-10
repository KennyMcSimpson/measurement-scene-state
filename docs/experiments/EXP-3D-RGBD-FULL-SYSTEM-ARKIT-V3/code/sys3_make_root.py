"""EXP-3D-RGBD-FULL-SYSTEM-ARKIT-V3: the evaluation files on ARKIT_SYSTEM3 (EXP-3D-RGBD-ARKIT-SYSTEM3-DATA-V1, 160 fresh
scans; manifest used unchanged). Episodes: record x role (A, B); context = role frames; stream = the Core B frame-id
rule (stream_frame_ids, four evenly spaced free frames, shared by both roles); queries = primary_query. The arms
contract of EXP-3D-RGBD-CARRIER-ARKIT-OVERLAP-V1 (G1 and GA, three seeds, step 24000) for this cohort. The selector
seal: per episode S = (p90 - p10) / median of the valid measured depth of its three context frames and the decision
S >= 1.2. Only context depth is read; no prediction is made and no query depth is read.
Run through cv_host_run.py from the repository root (CUDA hidden):  python sys3_make_root.py ROOT
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

from mcss.mechanism_pilot.rgbd_stream_write import stream_frame_ids

DATA = Path("outputs/EXP-3D-RGBD-ARKIT-SYSTEM3-DATA-V1").resolve()
CARRIER = Path("outputs/EXP-3D-RGBD-CARRIER-ARKIT-OVERLAP-V1").resolve()
NAME, THRESHOLD, SCANS = "ARKIT_SYSTEM3", 1.2, 160


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


root = Path(sys.argv[1]).resolve()
if not (root / "audit/plan_locked_at.txt").is_file():
    raise SystemExit("the plan must be locked first")
if (root / "evaluation_contract.json").exists():
    raise SystemExit("evaluation files exist already")
if json.loads((DATA / "preparation_integrity.json").read_text())["status"] != "PASS":
    raise SystemExit("cohort preparation did not pass")
manifest = DATA / "manifest_arkit_system3.json"
records = json.loads(manifest.read_text())["scenes"]
if len(records) != SCANS or len({r["physical_scene_id"] for r in records}) != SCANS:
    raise ValueError(f"expected {SCANS} distinct venues")
carrier = json.loads((CARRIER / "evaluation_contract.json").read_text())
contract = dict(carrier, cohorts={NAME: {"manifest": str(manifest), "manifest_sha256": sha(manifest), "split": NAME,
                                         "role": "primary_query"}})
contract["source_contract"] = {"path": str(CARRIER / "evaluation_contract.json"),
                               "sha256": sha(CARRIER / "evaluation_contract.json")}
(root / "evaluation_contract.json").write_text(json.dumps(contract, indent=2))
episodes, selector = [], {}
for r in records:
    stream = stream_frame_ids(r, "A")
    if stream is None or stream != stream_frame_ids(r, "B"):
        raise ValueError(f"{r['scene_id']}: no shared four-frame stream")
    frames = {f["frame_id"]: f for f in r["frames"]}
    for role, key in (("A", "context_a"), ("B", "context_b")):
        context, queries = list(r["roles"][key]), list(r["roles"]["primary_query"])
        if set(stream) & (set(context) | set(queries)) or set(context) & set(queries):
            raise ValueError(f"{r['scene_id']}/{role}: frame roles overlap")
        episodes.append({"scene_id": r["scene_id"], "role": role, "context": context, "stream": list(stream),
                         "queries": queries})
        depth = np.concatenate([d[np.isfinite(d) & (d > 0)] for d in
                                (np.load(frames[c]["depth"]).squeeze().astype(np.float64) for c in context)])
        spread = float((np.percentile(depth, 90) - np.percentile(depth, 10)) / np.median(depth))
        selector[f"{r['scene_id']}/{role}"] = {"spread": spread, "use_learned": spread >= THRESHOLD}
(root / "frame_role_lock.json").write_text(json.dumps({
    "rule": "context = role frames; stream = Core B frame-id rule (stream_frame_ids, 4 frames, shared by roles); "
            "queries = primary_query",
    "cohorts": {NAME: {"manifest": str(manifest), "manifest_sha256": sha(manifest), "episodes": episodes}}}, indent=1))
(root / "selector_seal.json").write_text(json.dumps({
    "rule": "S = (p90 - p10) / median of the valid measured depth of the episode's three context frames; "
            f"use the learned completion where S >= {THRESHOLD}", "threshold": THRESHOLD,
    "query_depth_read": False, "episodes": selector}, indent=1))
print("SYS3_ROOT", root, len(records), "records", len(episodes), "episodes; selected",
      sum(v["use_learned"] for v in selector.values()))
