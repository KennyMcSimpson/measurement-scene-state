"""EXP-3D-RGBD-COMPLETION-ENSEMBLE-V1: evaluation root on EVAL34 (EVAL-V3 + EVAL-V4, 83 scenes; the DepthSplat probe's
combined manifest, copied), the G1 arms contract (three seeds) and the frame-role lock (no prediction is made and no
query depth is read here). Episodes: every record x role (A, B); context = the role's frames; queries =
primary_query; no stream. Asserts disjointness from TRAIN72, SEG-V1, DEV and the protected holdout. Run through
cv_host_run.py from the repository root:  python ens_make_root.py ROOT
"""
import hashlib
import json
import shutil
import sys
from pathlib import Path

REPO = Path.cwd()
SOURCE = Path.home() / "cv_work/depthsplat/probe_eval34/manifest_eval34.json"
G1_ROOT = REPO / "outputs/EXP-3D-RGBD-SEGMENT-DIVERSITY-V21/G1/EXP-3D-RGBD-LONG-TRAIN-V17"
SEEDS = [20260928, 20260929, 20260930]
NAME = "EVAL34"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


root = Path(sys.argv[1]).resolve()
root.mkdir(parents=True, exist_ok=False)
manifest_path = root / "manifest_eval34.json"
shutil.copyfile(SOURCE, manifest_path)
manifest = json.loads(manifest_path.read_text())
records = manifest["scenes"]
split = json.loads((REPO / "outputs/EXP-3D-RGBD-DEPTH-BOUNDS-V7/scene_split.json").read_text())
train = [r for r in split["scenes"] if r["split"] == "TRAIN"]
dev = json.loads((Path.home() / "cv_work/depthsplat/probe_dev/manifest_dev.json").read_text())["scenes"]
used = {r["scene_id"] for r in train + dev} | set(split["protected_scene_ids"])
used_physical = {r["physical_scene_id"] for r in train + dev}
if {r["scene_id"] for r in records} & used or {r["physical_scene_id"] for r in records} & used_physical:
    raise ValueError("an EVAL34 record overlaps TRAIN72 (and so SEG-V1), DEV or the protected holdout")
if any(r["split"] != NAME for r in records) or len(records) != 83:
    raise ValueError("unexpected EVAL34 manifest")
v17 = json.loads((REPO / "outputs/EXP-3D-RGBD-LONG-TRAIN-V17/training_contract.json").read_text())["eval"]
contract = {
    "arms": {"G1": {"module": "mcss.mechanism_pilot.rgbd_long_train_carrier", "root": str(G1_ROOT), "step": 24000,
                    "preregistration_sha256": sha(G1_ROOT / "preregistration.json")}},
    "reference_arm": "G1", "seeds": SEEDS,
    "cohorts": {NAME: {"manifest": str(manifest_path), "manifest_sha256": sha(manifest_path), "split": NAME,
                       "role": "primary_query"}},
    "near_px": v17["near_px"], "reprojection_fill_constant_m": v17["reprojection_fill_constant_m"],
    "abstain_opacity_threshold": v17["abstain_opacity_threshold"],
}
(root / "evaluation_contract.json").write_text(json.dumps(contract, indent=2))
episodes = [{"scene_id": r["scene_id"], "role": role, "context": list(r["roles"][key]), "stream": [],
             "queries": list(r["roles"]["primary_query"])}
            for r in records for role, key in (("A", "context_a"), ("B", "context_b"))]
lock = {"rule": "context = role frames; queries = primary_query; no stream",
        "cohorts": {NAME: {"manifest": str(manifest_path), "manifest_sha256": sha(manifest_path), "episodes": episodes}},
        "source_manifest": {"path": str(SOURCE), "sha256": sha(SOURCE)}}
(root / "frame_role_lock.json").write_text(json.dumps(lock, indent=1))
print("ENSEMBLE_ROOT", root, len(records), "records", len(episodes), "episodes")
