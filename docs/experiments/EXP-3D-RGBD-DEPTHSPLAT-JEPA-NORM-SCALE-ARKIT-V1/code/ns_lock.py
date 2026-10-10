"""EXP-3D-RGBD-DEPTHSPLAT-JEPA-NORM-SCALE-ARKIT-V1: preregistration.json and audit/locked_at.txt (refuses to overwrite a
lock or to lock once any NS checkpoint, seal or score exists). Before locking it re-derives the training script from
OBJECTIVE-TTT's train_jepa_arkit_norm.py with code/derive_train_scale.py and requires a byte-identical result, requires
the helpers and hit_share_table.py to be the earlier experiments' and re-checks the merged hit table: every TRAIN entry
must equal OBJECTIVE-TTT's TRAIN table and the four added cohort tables must be the ones recorded in training/.
Run from the repository root:  python3 ns_lock.py outputs/EXP-3D-RGBD-DEPTHSPLAT-JEPA-NORM-SCALE-ARKIT-V1
"""
import hashlib
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

OUT = Path("outputs").resolve()
V1 = OUT / "EXP-3D-RGBD-DEPTHSPLAT-JEPA-TTT-V1"
OBJ = OUT / "EXP-3D-RGBD-DEPTHSPLAT-JEPA-OBJECTIVE-TTT-ARKIT-V1"
OVERLAP = OUT / "EXP-3D-RGBD-DEPTHSPLAT-JEPA-ARKIT-OVERLAP-V1"
NF = OUT / "EXP-3D-RGBD-DEPTHSPLAT-JEPA-NORM-FEEDFORWARD-ARKIT-V1"
MANIFESTS = {"CONFIRM": "CONFIRM-DATA-V1/manifest_arkit_confirm.json", "SELECT": "SELECT-DATA-V1/manifest_arkit_select.json",
             "SYSTEM": "SYSTEM-DATA-V1/manifest_arkit_system.json", "SYSTEM2": "SYSTEM2-DATA-V1/manifest_arkit_system2.json",
             "SYSTEM3": "SYSTEM3-DATA-V1/manifest_arkit_system3.json", "VAL": "VAL-DATA-V1/manifest_arkit_val80.json"}
MANIFESTS = {k: OUT / f"EXP-3D-RGBD-ARKIT-{v}" for k, v in MANIFESTS.items()}
PILOT = OUT / "EXP-3D-RGBD-ARKIT-DATA-V1"
RELEASED = Path.home() / "zonghan_o_mirror/_models/depthsplat/depthsplat-gs-base-re10kdl3dv-448x768-randview2-6-f8ddd845.pth"
ADDED_TABLES = ("hit_shares_ARKIT_CONFIRM.json", "hit_shares_ARKIT_SELECT.json", "hit_shares_ARKIT_SYSTEM.json",
                "hit_shares_ARKIT_SYSTEM2.json")


def sha(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


root = Path(sys.argv[1]).resolve()
code, training = root / "code", root / "training"
if (root / "preregistration.json").exists():
    raise SystemExit("already locked")
if any((root / d).exists() for d in ("checkpoints", "ds", "scores")):
    raise SystemExit("NS training, evaluation or score output exists")
with tempfile.TemporaryDirectory() as tmp:
    derived = Path(tmp) / "scale.py"
    subprocess.run([sys.executable, str(code / "derive_train_scale.py"), str(OBJ / "code/train_jepa_arkit_norm.py"),
                    str(derived)], check=True, stdout=subprocess.DEVNULL)
    if sha(derived) != sha(code / "train_jepa_arkit_scale.py"):
        raise SystemExit("train_jepa_arkit_scale.py is not the derivation of OBJECTIVE-TTT's JN script")
for name in ("ds_predict.py", "ds_finetune.py"):
    if sha(code / name) != sha(OBJ / "code" / name):
        raise SystemExit(f"{name} must be OBJECTIVE-TTT's")
if sha(code / "hit_share_table.py") != sha(OVERLAP / "code/hit_share_table.py"):
    raise SystemExit("hit_share_table.py must be OVERLAP-V1's")
merged = json.loads((training / "hit_shares_SCALE.json").read_text())
train_table = json.loads((OBJ / "training/hit_shares_TRAIN.json").read_text())
if any(merged[k] != v for k, v in train_table.items()):
    raise SystemExit("the merged table changed a TRAIN entry")
added = {}
for name in ADDED_TABLES:
    added.update(json.loads((training / name).read_text()))
if set(added) | set(train_table) != set(merged) or any(merged[k] != v for k, v in added.items()):
    raise SystemExit("the merged table is not TRAIN plus the four added tables")
lock = {
    "experiment": "EXP-3D-RGBD-DEPTHSPLAT-JEPA-NORM-SCALE-ARKIT-V1",
    "locked_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
    "protocol_sha256": sha(root / "PROTOCOL.md"),
    "code": {p.name: sha(p) for p in sorted(code.iterdir()) if p.is_file()},
    "root_files": {str(p.relative_to(root)): sha(p) for p in [root / "reused_seals.json",
                                                             root / "eval_SYSTEM3/frame_role_lock.json",
                                                             root / "eval_VAL/frame_role_lock.json",
                                                             *sorted(training.iterdir())]},
    "training_data": {"pilot": {n: sha(PILOT / n) for n in ("manifest_arkit_trainval.json", "candidate_lock.json")},
                      **{n: sha(MANIFESTS[n]) for n in ("CONFIRM", "SELECT", "SYSTEM", "SYSTEM2")}},
    "evaluation_data": {n: sha(MANIFESTS[n]) for n in ("SYSTEM3", "VAL")},
    "objective_preregistration_sha256": sha(OBJ / "preregistration.json"),
    "nf_preregistration_sha256": sha(NF / "preregistration.json"),
    "nf_results_sha256": sha(NF / "results/ARKIT_SYSTEM3_results.json"),
    "released_weights_sha256": sha(RELEASED),
    "v1_code": {n: sha(V1 / "code" / n) for n in ("ds_eval.py", "score_analyze.py")},
    "design_basis": "the NF result (known at design time: JN feed-forward gain PARTIAL); one factor changed (training "
                    "data 182 -> 502 scans); locked before any NS training or evaluation",
    "settings": {"training": "JL1-JL3: train_jepa_arkit_scale.py (JN script, record source arkit_records_scale), seeds "
                             "20261008 / 20261009 / 20261010, JN's settings (4000 steps, lr 5e-6 / ViT 5e-7, warmup "
                             "200, cosine, AdamW wd 1e-4, clip 1.0, 2 targets, min hit 0.05, scale cap 0.1, last step)",
                 "evaluation": "V1's ds_eval.py, CACHE arm only, cap 0.1: JL1-JL3 on ARKIT_SYSTEM3 and ARKIT_VAL; NF's "
                               "JN2 / JN3 on ARKIT_VAL", "alpha_min": 0.5,
                 "seed_readouts": "JNS / JLS = per-query mean over the three seeds of the AbsRel"},
    "primary": {"P1_MORE_DATA_SYSTEM3_ALL": "ARKIT_SYSTEM3 ALL: JNS_CACHE - JLS_CACHE",
                "P2_SCALED_RECIPE_VS_RELEASED_SYSTEM3_ALL": "ARKIT_SYSTEM3 ALL: ZS_CACHE - JLS_CACHE"},
    "statistics": {"unit": "scene", "draws": 10000, "seed": 20260928, "gate": "V2"},
    "rules": ["no setting changes after this lock; amendments only appended and dated",
              "query depth read only by ns_score.py score after every seal is complete",
              "ARKitScenes media outside git; nothing pushed"],
}
(root / "preregistration.json").write_text(json.dumps(lock, indent=1) + "\n")
(root / "audit").mkdir(exist_ok=True)
(root / "audit" / "locked_at.txt").write_text(f"{lock['locked_at']} sha256(preregistration.json)="
                                              f"{sha(root / 'preregistration.json')}\n")
print("LOCKED", lock["locked_at"], sha(root / "preregistration.json"))
