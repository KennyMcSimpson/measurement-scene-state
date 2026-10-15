"""EXP-3D-CAJU3D-SYSTEM4-HOLDOUT-V1: FUSE_V2, the readout trained by the plan's fixed recipe (data, model, readout).

FUSE_V2 keeps KFUSE-V1's frozen table on the covered pixels (alpha >= 0.5) and learns only the uncovered correction:
  log pred = (1 - w_KF) log R7 + w_KF log F7   on covered pixels (w_KF from the frozen table, not learned)
  log pred = log R7 + r,  r = 0.7 tanh(net)    on uncovered pixels
net = the FUSENET-V1 trunk with dilations (1, 2, 4, 8, 16, 1) on the same 8-channel base image; a zero head, so an
untrained V2 reads exactly KF_BASE. Training cohorts: every cohort with Core B predictions except SYSTEM4:
FIT_SYSTEM, FIT_SYSTEM2, CONF_SYSTEM3, CONF_VAL (KFUSE-V1's root) and CAJU_PILOT, CAJU_SEL, CAJU_EVAL (CAJU-V1's
100-step replay and geometry seals).
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
OUTPUTS = HERE.parents[1]
KF_ROOT = OUTPUTS / "EXP-3D-CAJU3D-KFUSE-V1"
CAJU = OUTPUTS / "EXP-3D-CAJU-HEADROOM-V1"
sys.path.insert(0, str(OUTPUTS / "EXP-3D-CAJU3D-FUSENET-V1" / "code"))
sys.path.insert(0, str(KF_ROOT / "code"))
import fn_data as D  # noqa: E402
import kf_logic as K  # noqa: E402
from caju_logic import core_b  # noqa: E402

S = D.S
KF_COHORTS = ("FIT_SYSTEM", "FIT_SYSTEM2", "CONF_SYSTEM3", "CONF_VAL")
CAJU_COHORTS = ("CAJU_PILOT", "CAJU_SEL", "CAJU_EVAL")
TABLES = KF_ROOT / "fusion_tables.json"
TABLES_SHA = "15100550ca6b39763244a69405ec88f8eede4faaa700aff4619557ee1c7e533c"
DILATIONS, WIDTH, R_MAX = (1, 2, 4, 8, 16, 1), 32, 0.7


def kf_weight_map(depth_z, alpha, r7, near7, cosine, tables):
    """[H, W] float32: the frozen KF_BASE weight on covered pixels, 0 elsewhere (unused there)."""
    r7 = np.maximum(r7.astype(np.float64), 1e-6)
    _, f7 = core_b(depth_z, alpha, r7, cosine)
    q = {"r7": r7.ravel(), "f7": np.maximum(f7, 1e-6).ravel(), "alpha": alpha.astype(np.float64).ravel(),
         "near": near7.astype(bool).ravel(), "log_p": np.zeros(r7.size)}
    cov, feats = K.covered_features(q)
    w = np.zeros(r7.size)
    idx = K.bin_index(feats, K.ARMS["KF_BASE"], tables["edges"])
    w[cov] = K.WEIGHTS[np.asarray(tables["weights"]["KF_BASE"])[idx]]
    return w.reshape(r7.shape).astype(np.float32)


def query_item(key, place, episode, depth_z, alpha, r7, near7, cosine, gt, tables):
    valid = np.isfinite(gt) & (gt > 0)
    return {"key": key, "place": place, "episode": episode,
            "base": D.base_image(depth_z, alpha, r7, near7, cosine),
            "w_kf": kf_weight_map(depth_z, alpha, r7, near7, cosine, tables), "gt": np.where(valid, gt, 0.0)}


def load_kf_cohort(cohort, access, tables):
    lock = json.loads((KF_ROOT / "frame_role_lock.json").read_text())["cohorts"][cohort]
    if D.sha(lock["manifest"]) != lock["manifest_sha256"]:
        raise PermissionError(f"{cohort}: manifest changed")
    reused = json.loads((KF_ROOT / "reused_seals.json").read_text())[cohort]
    if D.sha(reused["geometry"]["path"]) != reused["geometry"]["sha256"]:
        raise PermissionError(f"{cohort}: geometry seal changed")
    geometry = json.loads(Path(reused["geometry"]["path"]).read_text())["predictions"]
    zs = S.reused_predictions(reused["ZS"])
    records = {r["scene_id"]: r for r in json.loads(Path(lock["manifest"]).read_text())["scenes"]}
    out = []
    for e in lock["episodes"]:
        teacher = S.TeacherAccess(records[e["scene_id"]], e["queries"], S.IMAGE_SIZE, S.REPO, access)
        for qid in e["queries"]:
            key = f"{e['scene_id']}/{e['role']}/{qid}"
            geo, ttt = S.load_checked(geometry[key]), S.load_checked(zs[key])
            out.append(query_item(key, e["place"], f"{e['scene_id']}/{e['role']}", ttt["TTT_m1_depth_z"],
                                  ttt["TTT_m1_alpha"], geo["REPROJ7_HARMONIC"], geo["near7"], geo["cosine"],
                                  teacher.query_depth(qid), tables))
    return out


def caju_replay():
    out = {}
    for part in sorted((CAJU / "replay").glob("*/replay_seal_*of*.json")):
        seal = json.loads(part.read_text())
        if seal["ttt_steps"] != 100 or not seal["checkpoint_sha256"].startswith("f8ddd845"):
            raise PermissionError(f"{part}: not the released-weights 100-step replay")
        out.update({k: v for k, v in seal["predictions"].items() if v["kind"] == "query"})
    return out


def load_caju_cohort(cohort, access, tables, replay):
    lock = json.loads((CAJU / "frame_role_lock.json").read_text())["cohorts"][cohort]
    if D.sha(lock["manifest"]) != lock["manifest_sha256"]:
        raise PermissionError(f"{cohort}: manifest changed")
    geometry = json.loads((CAJU / "geometry" / f"geometry_seal_{cohort}.json").read_text())["predictions"]
    records = {r["scene_id"]: r for r in json.loads(Path(lock["manifest"]).read_text())["scenes"]}
    out = []
    for e in lock["episodes"]:
        teacher = S.TeacherAccess(records[e["scene_id"]], e["queries"], S.IMAGE_SIZE, S.REPO, access)
        for qid in e["queries"]:
            key = f"{e['scene_id']}/{e['role']}/{qid}"
            ttt, geo = S.load_checked(replay[key]), S.load_checked(geometry[key])
            out.append(query_item(key, e["place"], f"{e['scene_id']}/{e['role']}", ttt["FULL_depth_z"],
                                  ttt["FULL_alpha"], geo["REPROJ7_HARMONIC"], geo["near7"], geo["cosine"],
                                  teacher.query_depth(qid), tables))
    return out


def load_training(access):
    if D.sha(TABLES) != TABLES_SHA:
        raise PermissionError("the KFUSE-V1 fusion tables changed")
    tables = json.loads(TABLES.read_text())
    queries = [q for c in KF_COHORTS for q in load_kf_cohort(c, access, tables)]
    replay = caju_replay()
    queries += [q for c in CAJU_COHORTS for q in load_caju_cohort(c, access, tables, replay)]
    if {a["purpose"] for a in access} - {"SEALED_QUERY_EVALUATION"}:
        raise PermissionError("unexpected read purposes")
    return queries


class V2Net(nn.Module):
    def __init__(self):
        super().__init__()
        layers, cin = [], 8
        for d in DILATIONS:
            layers += [nn.Conv2d(cin, WIDTH, 3, padding=d, dilation=d), nn.GroupNorm(8, WIDTH), nn.GELU()]
            cin = WIDTH
        self.trunk = nn.Sequential(*layers)
        self.head = nn.Conv2d(WIDTH, 1, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, base):
        return R_MAX * torch.tanh(self.head(self.trunk(base))[:, 0])


def log_prediction(base, w_kf, r):
    log_r7, log_f7, covered = base[:, 0], base[:, 1], base[:, 6] > 0.5
    return torch.where(covered, (1 - w_kf) * log_r7 + w_kf * log_f7, log_r7 + r)
