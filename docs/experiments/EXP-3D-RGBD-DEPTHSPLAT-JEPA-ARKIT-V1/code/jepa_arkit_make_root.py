"""EXP-3D-RGBD-DEPTHSPLAT-JEPA-ARKIT-V1: evaluation root ARKIT160 = ARKIT_VAL80 (Validation fold) + ARKIT_CONFIRM
(Training-order venues 360-519), one manifest (split relabelled ARKIT160, records otherwise unchanged), the G1 arms
contract (three seeds) and the frame-role lock (record x role; context = role frames; queries = primary_query; no
stream). Asserts that no evaluation venue is among the 360 Training venues reserved for training (the pilot
selection). No prediction is made and no query depth is read. Run through cv_host_run.py from the repository root:
  python jepa_arkit_make_root.py ROOT
"""
import hashlib
import json
import sys
from pathlib import Path

REPO = Path.cwd()
SOURCES = {"ARKIT_VAL": REPO / "outputs/EXP-3D-RGBD-ARKIT-VAL-DATA-V1/manifest_arkit_val80.json",
           "ARKIT_CONFIRM": REPO / "outputs/EXP-3D-RGBD-ARKIT-CONFIRM-DATA-V1/manifest_arkit_confirm.json"}
PILOT = REPO / "outputs/EXP-3D-RGBD-ARKIT-DATA-V1/selection.json"
G1_ROOT = REPO / "outputs/EXP-3D-RGBD-SEGMENT-DIVERSITY-V21/G1/EXP-3D-RGBD-LONG-TRAIN-V17"
NAME = "ARKIT160"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


root = Path(sys.argv[1]).resolve()
root.mkdir(parents=True, exist_ok=False)
records, header = [], None
for cohort, path in SOURCES.items():
    m = json.loads(path.read_text())
    header = header or {k: m[k] for k in ("image_size", "depth_semantics")}
    records += [dict(r, split=NAME, source_cohort=cohort) for r in m["scenes"]]
venues = [r["physical_scene_id"] for r in records]
if len(records) != 160 or len(set(venues)) != 160:
    raise ValueError("expected 160 distinct evaluation venues")
reserved = {f"arkit_visit_{s['visit_id']}" for s in json.loads(PILOT.read_text())["selected"]["Training"]}
if set(venues) & reserved:
    raise ValueError("an evaluation venue is reserved for training")
manifest_path = root / "manifest_arkit160.json"
manifest_path.write_text(json.dumps({**header, "protected_scene_ids": [], "scenes": records,
                                     "sources": {k: {"path": str(v), "sha256": sha(v)} for k, v in SOURCES.items()}}))
v17 = json.loads((REPO / "outputs/EXP-3D-RGBD-LONG-TRAIN-V17/training_contract.json").read_text())["eval"]
contract = {
    "arms": {"G1": {"module": "mcss.mechanism_pilot.rgbd_long_train_carrier", "root": str(G1_ROOT), "step": 24000,
                    "preregistration_sha256": sha(G1_ROOT / "preregistration.json")}},
    "reference_arm": "G1", "seeds": [20260928, 20260929, 20260930],
    "cohorts": {NAME: {"manifest": str(manifest_path), "manifest_sha256": sha(manifest_path), "split": NAME,
                       "role": "primary_query"}},
    "near_px": v17["near_px"], "reprojection_fill_constant_m": v17["reprojection_fill_constant_m"],
    "abstain_opacity_threshold": v17["abstain_opacity_threshold"]}
(root / "evaluation_contract.json").write_text(json.dumps(contract, indent=2))
episodes = [{"scene_id": r["scene_id"], "role": role, "context": list(r["roles"][key]), "stream": [],
             "queries": list(r["roles"]["primary_query"])}
            for r in records for role, key in (("A", "context_a"), ("B", "context_b"))]
(root / "frame_role_lock.json").write_text(json.dumps({"cohorts": {NAME: {
    "manifest": str(manifest_path), "manifest_sha256": sha(manifest_path), "episodes": episodes}}}, indent=1))
print("ARKIT160_ROOT", root, len(records), "records", len(episodes), "episodes")
