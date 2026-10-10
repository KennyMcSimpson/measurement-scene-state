"""EXP-3D-RGBD-COMPLETION-SELECTOR-ARKIT-V1: the evaluation files. ARKIT_SELECT (EXP-3D-RGBD-ARKIT-SELECT-DATA-V1, 80
fresh scans) with one added role, overlap_query = the first two of frames 6, 7, 9, 11, 12 that the record does not
use as a context frame (either role) or an old query (primary_query); the arms contract of
EXP-3D-RGBD-CARRIER-ARKIT-OVERLAP-V1 (G1 and GA, three seeds, step 24000) for this cohort; the frame-role lock
(record x role; context = role frames; queries = overlap_query; no stream); and the selector seal: for every episode
the context depth spread S = (p90 - p10) / median of the valid measured depth of its three context frames and the
decision S >= 1.2. Only context depth is read; no prediction is made and no query depth is read.
Run from the repository root:  python3 sel_make_root.py ROOT   (ROOT holds PROTOCOL.md and the plan lock)
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

DATA = Path("outputs/EXP-3D-RGBD-ARKIT-SELECT-DATA-V1").resolve()
CARRIER = Path("outputs/EXP-3D-RGBD-CARRIER-ARKIT-OVERLAP-V1").resolve()
NAME, ORDER, THRESHOLD = "ARKIT_SELECT", (6, 7, 9, 11, 12), 1.2


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


root = Path(sys.argv[1]).resolve()
if not (root / "audit/plan_locked_at.txt").is_file():
    raise SystemExit("the plan must be locked first")
if (root / "evaluation_contract.json").exists():
    raise SystemExit("evaluation files exist already")
integrity = json.loads((DATA / "preparation_integrity.json").read_text())
if integrity["status"] != "PASS":
    raise SystemExit(f"cohort preparation status {integrity['status']}")
source = DATA / "manifest_arkit_select.json"
m = json.loads(source.read_text())
records = []
for r in m["scenes"]:
    used = set(r["roles"]["context_a"]) | set(r["roles"]["context_b"]) | set(r["roles"]["primary_query"])
    free = [f for f in ORDER if f not in used and f in {x["frame_id"] for x in r["frames"]}]
    if len(free) < 2:
        raise ValueError(f"{r['scene_id']}: fewer than two free query frames")
    records.append(dict(r, roles=dict(r["roles"], overlap_query=free[:2])))
manifest_path = root / "manifest_arkit_select_overlap.json"
manifest_path.write_text(json.dumps({**{k: v for k, v in m.items() if k != "scenes"}, "scenes": records,
                                     "derived_from": {"path": str(source), "sha256": sha(source),
                                                      "added_role": "overlap_query = first two free of 6, 7, 9, 11, 12"}}))
carrier = json.loads((CARRIER / "evaluation_contract.json").read_text())
contract = dict(carrier, cohorts={NAME: {"manifest": str(manifest_path), "manifest_sha256": sha(manifest_path),
                                         "split": NAME, "role": "overlap_query"}})
contract["source_contract"] = {"path": str(CARRIER / "evaluation_contract.json"),
                               "sha256": sha(CARRIER / "evaluation_contract.json")}
(root / "evaluation_contract.json").write_text(json.dumps(contract, indent=2))
episodes, selector = [], {}
for r in records:
    frames = {f["frame_id"]: f for f in r["frames"]}
    for role, key in (("A", "context_a"), ("B", "context_b")):
        context = list(r["roles"][key])
        episodes.append({"scene_id": r["scene_id"], "role": role, "context": context, "stream": [],
                         "queries": list(r["roles"]["overlap_query"])})
        depth = np.concatenate([d[np.isfinite(d) & (d > 0)] for d in
                                (np.load(frames[c]["depth"]).squeeze().astype(np.float64) for c in context)])
        spread = float((np.percentile(depth, 90) - np.percentile(depth, 10)) / np.median(depth))
        selector[f"{r['scene_id']}/{role}"] = {"spread": spread, "use_learned": spread >= THRESHOLD}
(root / "frame_role_lock.json").write_text(json.dumps({"cohorts": {NAME: {
    "manifest": str(manifest_path), "manifest_sha256": sha(manifest_path), "episodes": episodes}}}, indent=1))
(root / "selector_seal.json").write_text(json.dumps({
    "rule": "S = (p90 - p10) / median of the valid measured depth of the episode's three context frames; "
            f"use the learned completion where S >= {THRESHOLD}", "threshold": THRESHOLD,
    "query_depth_read": False, "episodes": selector}, indent=1))
moved = {r["scene_id"]: r["roles"]["overlap_query"] for r in records if r["roles"]["overlap_query"] != [6, 7]}
chosen = sum(v["use_learned"] for v in selector.values())
print("SEL_ROOT", root, len(records), "records", len(episodes), "episodes; learned used in", chosen, "episodes;",
      "queries moved:", moved)
