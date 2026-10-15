"""EXP-3D-CAJU3D-SYSTEM4-HOLDOUT-V1: the evaluation files on ARKIT_SYSTEM4 (EXP-3D-RGBD-ARKIT-SYSTEM4-DATA-V1, manifest
used unchanged), exactly as FULL-SYSTEM-V3's sys3_make_root.py builds them for ARKIT_SYSTEM3 (no selector seal).
Episodes: record x role (A, B); context = role frames; stream = the Core B frame-id rule (stream_frame_ids, four evenly
spaced free frames, shared by both roles); queries = primary_query. The arms contract of
EXP-3D-RGBD-CARRIER-ARKIT-OVERLAP-V1 for this cohort. Metadata only: no image, depth or prediction is read.
Run through cv_host_run.py from the repository root (CUDA hidden):  python s4_make_root.py ROOT
"""
import hashlib
import json
import sys
from pathlib import Path

from mcss.mechanism_pilot.rgbd_stream_write import stream_frame_ids

DATA = Path("outputs/EXP-3D-RGBD-ARKIT-SYSTEM4-DATA-V1").resolve()
CARRIER = Path("outputs/EXP-3D-RGBD-CARRIER-ARKIT-OVERLAP-V1").resolve()
NAME = "ARKIT_SYSTEM4"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


root = Path(sys.argv[1]).resolve()
if not (root / "audit/plan_locked_at.txt").is_file():
    raise SystemExit("the plan must be locked first")
if (root / "evaluation_contract.json").exists() or (root / "frame_role_lock.json").exists():
    raise SystemExit("evaluation files exist already")
integrity = json.loads((DATA / "preparation_integrity.json").read_text())
if integrity["status"] != "PASS":
    raise SystemExit("cohort preparation did not pass")
manifest = DATA / "manifest_arkit_system4.json"
records = json.loads(manifest.read_text())["scenes"]
if len(records) != integrity["eligible"] or len({r["physical_scene_id"] for r in records}) != len(records):
    raise ValueError("expected one distinct venue per eligible scan")
carrier = json.loads((CARRIER / "evaluation_contract.json").read_text())
contract = dict(carrier, cohorts={NAME: {"manifest": str(manifest), "manifest_sha256": sha(manifest), "split": NAME,
                                         "role": "primary_query"}})
contract["source_contract"] = {"path": str(CARRIER / "evaluation_contract.json"),
                               "sha256": sha(CARRIER / "evaluation_contract.json")}
(root / "evaluation_contract.json").write_text(json.dumps(contract, indent=2))
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
(root / "frame_role_lock.json").write_text(json.dumps({
    "rule": "context = role frames; stream = Core B frame-id rule (stream_frame_ids, 4 frames, shared by roles); "
            "queries = primary_query",
    "cohorts": {NAME: {"manifest": str(manifest), "manifest_sha256": sha(manifest), "episodes": episodes}}}, indent=1))
print("S4_ROOT", root, len(records), "records", len(episodes), "episodes")
