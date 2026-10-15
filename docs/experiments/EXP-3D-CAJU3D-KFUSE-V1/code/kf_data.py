"""EXP-3D-CAJU3D-KFUSE-V1: loading one cohort's per-query arrays for fitting or scoring (CPU).

Per query (valid ground-truth pixels only): r7 (geometry seal REPROJ7_HARMONIC, floored at 1e-6), near (near7), f7 and
alpha (Core B's TTT: the sealed released-weights TTT_m1 through caju_logic.core_b), log_p (the H-JEPA seal), gt (through
TeacherAccess, after every prediction and feature is sealed). Every sealed file is checked against its hash.
"""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUTPUTS = HERE.parents[1]
for _path in (OUTPUTS / "EXP-3D-RGBD-DEPTHSPLAT-JEPA-TTT-V1" / "code", OUTPUTS / "EXP-3D-CAJU-HEADROOM-V1" / "code",
              OUTPUTS / "EXP-3D-CAJU3D-ENGINE-V1" / "code"):
    sys.path.insert(0, str(_path))
import engine_score as S  # noqa: E402
from caju_logic import core_b  # noqa: E402


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def jepa_features(directory):
    parts = sorted(Path(directory).glob("kf_jepa_seal_*of*.json"))
    if not parts or len(parts) != int(parts[0].stem.split("of")[1]):
        raise PermissionError(f"{directory}: incomplete set of H-JEPA seals")
    out = {}
    for part in parts:
        seal = json.loads(part.read_text())
        if seal["limit"] or seal["query_media_read"] or seal["query_depth_read"] or seal["access_violations"]:
            raise PermissionError(f"{part}: not the protocol's run")
        out.update(seal["predictions"])
    return out


def load_cohort(root, cohort, access):
    """The cohort's per-query arrays; `access` collects the TeacherAccess reads."""
    lock = json.loads((root / "frame_role_lock.json").read_text())["cohorts"][cohort]
    if sha(lock["manifest"]) != lock["manifest_sha256"]:
        raise PermissionError(f"{cohort}: manifest changed")
    reused = json.loads((root / "reused_seals.json").read_text())[cohort]
    if sha(reused["geometry"]["path"]) != reused["geometry"]["sha256"]:
        raise PermissionError(f"{cohort}: geometry seal changed")
    geometry = json.loads(Path(reused["geometry"]["path"]).read_text())["predictions"]
    zs = S.reused_predictions(reused["ZS"])
    jepa = jepa_features(root / "jepa" / cohort)
    records = {r["scene_id"]: r for r in json.loads(Path(lock["manifest"]).read_text())["scenes"]}
    keys = [f"{e['scene_id']}/{e['role']}/{q}" for e in lock["episodes"] for q in e["queries"]]
    for name, preds in (("geometry", geometry), ("ZS", zs), ("H-JEPA", jepa)):
        missing = [k for k in keys if k not in preds]
        if missing:
            raise PermissionError(f"{cohort}/{name}: {len(missing)} queries not sealed")
    queries = []
    for e in lock["episodes"]:
        teacher = S.TeacherAccess(records[e["scene_id"]], e["queries"], S.IMAGE_SIZE, S.REPO, access)
        for qid in e["queries"]:
            key = f"{e['scene_id']}/{e['role']}/{qid}"
            geo, ttt, jf = S.load_checked(geometry[key]), S.load_checked(zs[key]), S.load_checked(jepa[key])
            gt = teacher.query_depth(qid)
            valid = np.isfinite(gt) & (gt > 0)
            _, f7 = core_b(ttt["TTT_m1_depth_z"], ttt["TTT_m1_alpha"], geo["REPROJ7_HARMONIC"], geo["cosine"])
            queries.append({"key": key, "place": e["place"], "episode": f"{e['scene_id']}/{e['role']}",
                            "gt": gt[valid], "r7": np.maximum(geo["REPROJ7_HARMONIC"][valid].astype(np.float64), 1e-6),
                            "f7": np.maximum(f7[valid], 1e-6), "alpha": ttt["TTT_m1_alpha"][valid].astype(np.float64),
                            "near": geo["near7"][valid].astype(bool), "log_p": jf["log_p"][valid].astype(np.float64),
                            "jepa_hit": jf["hit"][valid].astype(bool)})
    if {a["purpose"] for a in access} - {"SEALED_QUERY_EVALUATION"}:
        raise PermissionError("unexpected read purposes")
    return queries
