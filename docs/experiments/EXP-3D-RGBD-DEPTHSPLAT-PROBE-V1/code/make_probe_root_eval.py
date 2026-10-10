"""Probe root for a DepthSplat development check on existing evaluation cohorts (exploratory): one combined
manifest of the given cohorts' records (split relabelled NAME, records otherwise unchanged) and an arms contract
with G1 (V21) at step 24000, one seed. Run through cv_host_run.py from the repository root:
  python make_probe_root_eval.py ROOT NAME MANIFEST [MANIFEST ...]
"""
import hashlib
import json
import sys
from pathlib import Path

root, name, sources = Path(sys.argv[1]).resolve(), sys.argv[2], sys.argv[3:]
root.mkdir(parents=True, exist_ok=False)
records, header = [], None
for source in sources:
    m = json.loads(Path(source).read_text())
    if header is None:
        header = {k: v for k, v in m.items() if k not in ("scenes", "schema", "candidate_lock_sha256")}
    if m["image_size"] != header["image_size"] or m["depth_semantics"] != header["depth_semantics"]:
        raise ValueError(f"{source}: image size or depth semantics differ")
    header["protected_scene_ids"] = sorted(set(header["protected_scene_ids"]) | set(m["protected_scene_ids"]))
    for r in m["scenes"]:
        records.append(dict(r, split=name, source_manifest=source))
for key in ("scene_id", "physical_scene_id"):
    values = [r[key] for r in records]
    if len(values) != len(set(values)):
        raise ValueError(f"duplicate {key} across cohorts")
if {r["physical_scene_id"] for r in records} & set(header["protected_scene_ids"]):
    raise ValueError("a record is in the protected holdout")
manifest = dict(header, scenes=records, sources=sources)
(root / f"manifest_{name.lower()}.json").write_text(json.dumps(manifest))
v17 = json.loads(Path("outputs/EXP-3D-RGBD-LONG-TRAIN-V17/training_contract.json").read_text())["eval"]
g1 = Path("outputs/EXP-3D-RGBD-SEGMENT-DIVERSITY-V21/G1/EXP-3D-RGBD-LONG-TRAIN-V17").resolve()
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
contract = {
    "arms": {"G1": {"module": "mcss.mechanism_pilot.rgbd_long_train_carrier", "root": str(g1),
                    "step": 24000, "preregistration_sha256": sha(g1 / "preregistration.json")}},
    "reference_arm": "G1",
    "seeds": [20260928],
    "cohorts": {name: {"manifest": str(root / f"manifest_{name.lower()}.json"),
                       "manifest_sha256": sha(root / f"manifest_{name.lower()}.json"), "split": name,
                       "role": "primary_query"}},
    "near_px": v17["near_px"],
    "reprojection_fill_constant_m": v17["reprojection_fill_constant_m"],
    "abstain_opacity_threshold": v17["abstain_opacity_threshold"],
}
(root / "evaluation_contract.json").write_text(json.dumps(contract, indent=2))
print("PROBE_ROOT", root, len(records), "records", name)
