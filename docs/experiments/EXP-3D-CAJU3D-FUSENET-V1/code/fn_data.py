"""EXP-3D-CAJU3D-FUSENET-V1: per-query input images of one cohort (CPU, numpy).

base [8, 128, 160] float32, all from arrived data (sealed Core B TTT, sealed geometry):
  0 log R7            REPROJ7_HARMONIC (floored at 1e-6)
  1 log F7            Core B's F7: the TTT ray depth where alpha >= 0.5, else R7
  2 DIS               |log F7 - log R7|
  3 ALPHA             the TTT alpha
  4 NEAR              near7 (R7 is a direct reprojection hit)
  5 COSINE            the geometry seal's cosine
  6 COVERED           alpha >= 0.5
  7 log RAY           the raw TTT ray depth depth_z / max(alpha, 1e-3) / cosine, clipped to [0.1, 20] m
jepa [66, 64, 80] float32 (only when asked): the H-JEPA's z_hat (64), log P, hit at the query cells.
gt [128, 160] float64 with 0 where invalid (read through TeacherAccess, after every input is sealed).
kf: the float64 valid-pixel arrays of KFUSE-V1 (r7, f7, alpha, near, gt), so CORE_B and KF_BASE are scored exactly as
there.
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

ALPHA_MIN = 0.5


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def feature_seals(directory):
    parts = sorted(Path(directory).glob("fn_feature_seal_*of*.json"))
    if not parts or len(parts) != int(parts[0].stem.split("of")[1]):
        raise PermissionError(f"{directory}: incomplete set of H-JEPA feature seals")
    out = {}
    for part in parts:
        seal = json.loads(part.read_text())
        if seal["limit"] or seal["query_media_read"] or seal["query_depth_read"] or seal["access_violations"]:
            raise PermissionError(f"{part}: not the protocol's run")
        out.update(seal["predictions"])
    return out


def base_image(depth_z, alpha, r7, near7, cosine):
    alpha = alpha.astype(np.float64)
    r7 = np.maximum(r7.astype(np.float64), 1e-6)
    _, f7 = core_b(depth_z, alpha, r7, cosine)
    f7 = np.maximum(f7, 1e-6)
    ray = np.clip(depth_z / np.maximum(alpha, 1e-3) / cosine, 0.1, 20.0)
    channels = [np.log(r7), np.log(f7), np.abs(np.log(f7) - np.log(r7)), alpha, near7.astype(np.float64), cosine,
                (alpha >= ALPHA_MIN).astype(np.float64), np.log(ray)]
    return np.stack(channels).astype(np.float32)


def load_cohort(root, cohort, access, with_jepa):
    """List of per-query dicts: key, place, episode, base, jepa (or None), gt."""
    lock = json.loads((root / "frame_role_lock.json").read_text())["cohorts"][cohort]
    if sha(lock["manifest"]) != lock["manifest_sha256"]:
        raise PermissionError(f"{cohort}: manifest changed")
    reused = json.loads((root / "reused_seals.json").read_text())[cohort]
    if sha(reused["geometry"]["path"]) != reused["geometry"]["sha256"]:
        raise PermissionError(f"{cohort}: geometry seal changed")
    geometry = json.loads(Path(reused["geometry"]["path"]).read_text())["predictions"]
    zs = S.reused_predictions(reused["ZS"])
    jepa = feature_seals(root / "features" / cohort) if with_jepa else {}
    records = {r["scene_id"]: r for r in json.loads(Path(lock["manifest"]).read_text())["scenes"]}
    queries = []
    for e in lock["episodes"]:
        teacher = S.TeacherAccess(records[e["scene_id"]], e["queries"], S.IMAGE_SIZE, S.REPO, access)
        for qid in e["queries"]:
            key = f"{e['scene_id']}/{e['role']}/{qid}"
            geo, ttt = S.load_checked(geometry[key]), S.load_checked(zs[key])
            base = base_image(ttt["TTT_m1_depth_z"], ttt["TTT_m1_alpha"], geo["REPROJ7_HARMONIC"], geo["near7"],
                              geo["cosine"])
            features = None
            if with_jepa:
                f = S.load_checked(jepa[key])
                features = np.concatenate([f["z_hat"].astype(np.float32), f["log_p"][None].astype(np.float32),
                                           f["hit"][None].astype(np.float32)])
            gt = teacher.query_depth(qid)
            valid = np.isfinite(gt) & (gt > 0)
            gt = np.where(valid, gt, 0.0)
            _, f7 = core_b(ttt["TTT_m1_depth_z"], ttt["TTT_m1_alpha"], geo["REPROJ7_HARMONIC"], geo["cosine"])
            kf = {"gt": gt[valid], "r7": np.maximum(geo["REPROJ7_HARMONIC"][valid].astype(np.float64), 1e-6),
                  "f7": np.maximum(f7[valid], 1e-6), "alpha": ttt["TTT_m1_alpha"][valid].astype(np.float64),
                  "near": geo["near7"][valid].astype(bool), "log_p": np.zeros(int(valid.sum()))}
            queries.append({"key": key, "place": e["place"], "episode": f"{e['scene_id']}/{e['role']}",
                            "base": base, "jepa": features, "gt": gt, "kf": kf})
    if {a["purpose"] for a in access} - {"SEALED_QUERY_EVALUATION"}:
        raise PermissionError("unexpected read purposes")
    return queries


def query_weights(queries):
    """Weight per query so that the weighted sum of per-query AbsRel equals the place-level mean."""
    places = {}
    for i, q in enumerate(queries):
        places.setdefault(q["place"], {}).setdefault(q["episode"], []).append(i)
    w = np.zeros(len(queries))
    for episodes in places.values():
        for members in episodes.values():
            for i in members:
                w[i] = 1.0 / (len(places) * len(episodes) * len(members))
    return w
