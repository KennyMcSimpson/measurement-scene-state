"""EXP-3D-RGBD-COMPLETION-SELECTOR-FAR-ARKIT-V1: the evaluation files. ARKIT_SELECT (EXP-3D-RGBD-ARKIT-SELECT-DATA-V1, 80
scans; manifest used unchanged) with queries = primary_query (frames 14 and 15); the arms contract of
EXP-3D-RGBD-CARRIER-ARKIT-OVERLAP-V1 (G1 and GA, three seeds, step 24000) for this cohort and role; the frame-role lock
(record x role; context = role frames; queries = primary_query; no stream); and a byte copy of the selector seal of
EXP-3D-RGBD-COMPLETION-SELECTOR-ARKIT-V1 (S depends only on the context, which is the same).
No prediction is made and no query depth is read. Run from the repository root:  python3 far_make_root.py ROOT
"""
import hashlib
import json
import shutil
import sys
from pathlib import Path

DATA = Path("outputs/EXP-3D-RGBD-ARKIT-SELECT-DATA-V1").resolve()
CARRIER = Path("outputs/EXP-3D-RGBD-CARRIER-ARKIT-OVERLAP-V1").resolve()
SELECTOR = Path("outputs/EXP-3D-RGBD-COMPLETION-SELECTOR-ARKIT-V1").resolve()
NAME = "ARKIT_SELECT"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


root = Path(sys.argv[1]).resolve()
if (root / "evaluation_contract.json").exists():
    raise SystemExit("evaluation files exist already")
manifest = DATA / "manifest_arkit_select.json"
records = json.loads(manifest.read_text())["scenes"]
prereg = json.loads((SELECTOR / "preregistration.json").read_text())
if sha(SELECTOR / "selector_seal.json") != prereg["root_files"]["selector_seal.json"]:
    raise PermissionError("the selector seal changed")
if sha(manifest) != prereg["cohort_data"]["manifest_arkit_select.json"]:
    raise PermissionError("the cohort manifest changed")
carrier = json.loads((CARRIER / "evaluation_contract.json").read_text())
contract = dict(carrier, cohorts={NAME: {"manifest": str(manifest), "manifest_sha256": sha(manifest), "split": NAME,
                                         "role": "primary_query"}})
contract["source_contract"] = {"path": str(CARRIER / "evaluation_contract.json"),
                               "sha256": sha(CARRIER / "evaluation_contract.json")}
(root / "evaluation_contract.json").write_text(json.dumps(contract, indent=2))
episodes = []
for r in records:
    if set(r["roles"]["primary_query"]) & (set(r["roles"]["context_a"]) | set(r["roles"]["context_b"])):
        raise ValueError(f"{r['scene_id']}: a query frame is a context frame")
    for role, key in (("A", "context_a"), ("B", "context_b")):
        episodes.append({"scene_id": r["scene_id"], "role": role, "context": list(r["roles"][key]), "stream": [],
                         "queries": list(r["roles"]["primary_query"])})
(root / "frame_role_lock.json").write_text(json.dumps({"cohorts": {NAME: {
    "manifest": str(manifest), "manifest_sha256": sha(manifest), "episodes": episodes}}}, indent=1))
shutil.copyfile(SELECTOR / "selector_seal.json", root / "selector_seal.json")
seal = json.loads((root / "selector_seal.json").read_text())["episodes"]
if set(seal) != {f"{e['scene_id']}/{e['role']}" for e in episodes}:
    raise ValueError("the selector seal does not cover the episodes")
print("FAR_ROOT", root, len(records), "records", len(episodes), "episodes; selected",
      sum(v["use_learned"] for v in seal.values()))
