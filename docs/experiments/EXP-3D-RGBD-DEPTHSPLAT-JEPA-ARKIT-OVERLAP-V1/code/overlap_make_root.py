"""EXP-3D-RGBD-DEPTHSPLAT-JEPA-ARKIT-OVERLAP-V1: the evaluation root. ARKIT160_OVERLAP = the 160 ARKIT160 records
(EXP-3D-RGBD-DEPTHSPLAT-JEPA-ARKIT-V1 manifest, sha-checked) with one added role, overlap_query = the first two of
frames 6, 7, 9, 11, 12 that the record does not use as a context frame (either role), an old query (primary_query)
or a Core B stream frame (ARKIT_VAL streams of EXP-3D-RGBD-ENSEMBLES-ARKIT-V1); for almost every record frames 6 and 7.
No experiment has evaluated these frames or fed them to a model. Writes the manifest (split relabelled, records
otherwise unchanged), the G1 arms contract (three seeds; role overlap_query) and the frame-role lock (record x role;
context = role frames; queries = overlap_query; no stream). Also copies the TRAIN hit table used for training and,
for information, the same table of the evaluation records.
No prediction is made and no query depth is read. Run from the repository root:  python3 overlap_make_root.py ROOT
"""
import hashlib
import json
import shutil
import sys
from pathlib import Path

SOURCE = Path("outputs/EXP-3D-RGBD-DEPTHSPLAT-JEPA-ARKIT-V1").resolve()
ENSEMBLES = Path("outputs/EXP-3D-RGBD-ENSEMBLES-ARKIT-V1").resolve()
HIT_TABLE = Path.home() / "cv_work/depthsplat/jepa_ttt/hit_shares_TRAIN.json"
NAME, ORDER = "ARKIT160_OVERLAP", (6, 7, 9, 11, 12)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


root = Path(sys.argv[1]).resolve()
root.mkdir(parents=True, exist_ok=False)
source_contract = json.loads((SOURCE / "evaluation_contract.json").read_text())
source_manifest = SOURCE / "manifest_arkit160.json"
if sha(source_manifest) != source_contract["cohorts"]["ARKIT160"]["manifest_sha256"]:
    raise PermissionError("ARKIT160 manifest changed")
m = json.loads(source_manifest.read_text())
streams = {}
for e in json.loads((ENSEMBLES / "frame_role_lock.json").read_text())["cohorts"]["ARKIT_VAL"]["episodes"]:
    streams.setdefault(e["scene_id"], set()).update(e["stream"])
records = []
for r in m["scenes"]:
    used = (set(r["roles"]["context_a"]) | set(r["roles"]["context_b"]) | set(r["roles"]["primary_query"])
            | streams.get(r["scene_id"], set()))
    free = [f for f in ORDER if f not in used and f in {x["frame_id"] for x in r["frames"]}]
    if len(free) < 2:
        raise ValueError(f"{r['scene_id']}: fewer than two free query frames")
    records.append(dict(r, split=NAME, roles=dict(r["roles"], overlap_query=free[:2])))
manifest_path = root / "manifest_arkit160_overlap.json"
manifest_path.write_text(json.dumps({**{k: v for k, v in m.items() if k != "scenes"}, "scenes": records,
                                     "derived_from": {"path": str(source_manifest), "sha256": sha(source_manifest),
                                                      "added_role": "overlap_query = first two free of 6, 7, 9, 11, 12"}}))
contract = dict(source_contract, cohorts={NAME: {"manifest": str(manifest_path), "manifest_sha256": sha(manifest_path),
                                                 "split": NAME, "role": "overlap_query"}})
(root / "evaluation_contract.json").write_text(json.dumps(contract, indent=2))
episodes = [{"scene_id": r["scene_id"], "role": role, "context": list(r["roles"][key]), "stream": [],
             "queries": list(r["roles"]["overlap_query"])}
            for r in records for role, key in (("A", "context_a"), ("B", "context_b"))]
moved = {r["scene_id"]: r["roles"]["overlap_query"] for r in records if r["roles"]["overlap_query"] != [6, 7]}
print("records whose queries are not frames 6 and 7:", moved)
(root / "frame_role_lock.json").write_text(json.dumps({"cohorts": {NAME: {
    "manifest": str(manifest_path), "manifest_sha256": sha(manifest_path), "episodes": episodes}}}, indent=1))
(root / "training").mkdir()
shutil.copyfile(HIT_TABLE, root / "training/hit_shares_TRAIN.json")
# informational: the same table for the evaluation records (context depth and query cameras only)
shutil.copyfile(HIT_TABLE.with_name("hit_shares_ARKIT160.json"), root / "hit_shares_ARKIT160.json")
print("OVERLAP_ROOT", root, len(records), "records", len(episodes), "episodes", 2 * len(episodes), "queries")
