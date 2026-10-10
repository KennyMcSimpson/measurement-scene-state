"""EXP-3D-RGBD-CARRIER-ARKIT-OVERLAP-V1, after GA training: the evaluation files of the root. The frame-role lock of
EXP-3D-RGBD-DEPTHSPLAT-JEPA-ARKIT-OVERLAP-V1 (byte copy; ARKIT160_OVERLAP) and the arms contract: the OVERLAP-V1 contract
with the arms G1 (V21, step 24000, unchanged) and GA (this experiment's root, step 24000), three seeds, the cohort
ARKIT160_OVERLAP (the OVERLAP-V1 manifest, role overlap_query). Refuses unless every GA seed completed and nothing is
sealed yet. No prediction is made and no query depth is read. Run from the repository root:  python3 ga_make_eval.py
"""
import hashlib
import json
import shutil
from pathlib import Path

CARRIER = Path("outputs/EXP-3D-RGBD-CARRIER-ARKIT-OVERLAP-V1").resolve()
OVERLAP = Path("outputs/EXP-3D-RGBD-DEPTHSPLAT-JEPA-ARKIT-OVERLAP-V1").resolve()
GA_ROOT = CARRIER / "GA/EXP-3D-RGBD-LONG-TRAIN-V17"
SEEDS = (20260928, 20260929, 20260930)
STEP = 24000


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


if (CARRIER / "evaluation_contract.json").exists() or (CARRIER / "raw").exists():
    raise SystemExit("evaluation files exist already")
for seed in SEEDS:
    summary = json.loads((GA_ROOT / f"checkpoints/L_{seed}/summary.json").read_text())
    if summary["status"] != "COMPLETE" or summary["steps"] != STEP:
        raise SystemExit(f"GA seed {seed} did not complete")
    if not (GA_ROOT / f"checkpoints/L_{seed}/step_{STEP:06d}.pt").is_file():
        raise SystemExit(f"GA seed {seed}: no step-{STEP} checkpoint")
source = json.loads((OVERLAP / "evaluation_contract.json").read_text())
if set(source["arms"]) != {"G1"} or source["arms"]["G1"]["step"] != STEP:
    raise SystemExit("unexpected OVERLAP-V1 contract")
contract = dict(source)
contract["arms"] = {"G1": source["arms"]["G1"],
                    "GA": {"module": source["arms"]["G1"]["module"], "root": str(GA_ROOT), "step": STEP,
                           "preregistration_sha256": sha(GA_ROOT / "preregistration.json")}}
contract["reference_arm"] = "G1"
contract["carrier_arkit_overlap"] = {"ga_checkpoints": {str(s): sha(GA_ROOT / f"checkpoints/L_{s}/step_{STEP:06d}.pt")
                                                        for s in SEEDS},
                                     "source_contract_sha256": sha(OVERLAP / "evaluation_contract.json")}
(CARRIER / "evaluation_contract.json").write_text(json.dumps(contract, indent=2))
shutil.copyfile(OVERLAP / "frame_role_lock.json", CARRIER / "frame_role_lock.json")
print("GA_EVAL_FILES", CARRIER, "arms", sorted(contract["arms"]))
