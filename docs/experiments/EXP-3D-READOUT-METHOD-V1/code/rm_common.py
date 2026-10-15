"""EXP-3D-READOUT-METHOD-V1: shared data, features, ground truth, regions and metrics (numpy).

Every input comes from the sealed predictions cached by rm_cache.py (arrived frames + the query camera only); the
query ground truth is read through TeacherAccess and used only as the training target and for scoring.
Regions (fixed by the protocol): COV = TTT alpha >= 0.5; UNC_NEAR = not COV and R7 within 8 px of a reprojection hit
(near7); UNC_FILL = not COV and R7 a fill value. Evaluation-only strata: zero-hit queries, fill distance bins, depth
edges and far depth (both from the ground truth, never an input).
"""

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT.parent
for _p in ("EXP-3D-RGBD-DEPTHSPLAT-JEPA-TTT-V1/code", "EXP-3D-CAJU-HEADROOM-V1/code", "EXP-3D-CAJU3D-ENGINE-V1/code",
           "EXP-3D-CAJU3D-KFUSE-V1/code", "EXP-3D-CAJU3D-FUSENET-V1/code", "EXP-3D-CAJU3D-SYSTEM4-HOLDOUT-V1/code"):
    sys.path.insert(0, str(OUT / _p))
import engine_score as S  # noqa: E402
import fn_data as FD  # noqa: E402
import kf_logic as K  # noqa: E402

COHORTS = ("FIT_SYSTEM", "FIT_SYSTEM2", "CONF_SYSTEM3", "CONF_VAL", "CAJU", "ARKIT_SYSTEM4")
FOLDS = ("CONF_SYSTEM3", "CONF_VAL", "CAJU", "ARKIT_SYSTEM4")
TABLES = OUT / "EXP-3D-CAJU3D-KFUSE-V1" / "fusion_tables.json"
TABLES_SHA = "15100550ca6b39763244a69405ec88f8eede4faaa700aff4619557ee1c7e533c"
ALPHA_MIN, H, W = 0.5, 128, 160
STRATA = ("ALL", "COV", "UNC_NEAR", "UNC_FILL", "ZERO_HIT", "FILL_D0_8", "FILL_D8_32", "FILL_D32", "EDGE", "FAR")
LOCKS = {"FIT_SYSTEM": "EXP-3D-CAJU3D-KFUSE-V1/frame_role_lock.json",
         "FIT_SYSTEM2": "EXP-3D-CAJU3D-KFUSE-V1/frame_role_lock.json",
         "CONF_SYSTEM3": "EXP-3D-CAJU3D-KFUSE-V1/frame_role_lock.json",
         "CONF_VAL": "EXP-3D-CAJU3D-KFUSE-V1/frame_role_lock.json",
         "CAJU": "EXP-3D-CAJU-HEADROOM-V1/frame_role_lock.json",
         "ARKIT_SYSTEM4": "EXP-3D-CAJU3D-SYSTEM4-HOLDOUT-V1/frame_role_lock.json"}


def load_cache(cohort):
    arrays = dict(np.load(ROOT / "cache" / f"{cohort}.npz"))
    rows = json.loads((ROOT / "cache" / f"{cohort}.json").read_text())["queries"]
    return arrays, rows


def load_gt(cohort, rows, access):
    """[Q, H, W] float64 query depth (0 where invalid), read through TeacherAccess episode by episode."""
    locks = json.loads((OUT / LOCKS[cohort]).read_text())["cohorts"]
    lock_name = {"CAJU": None, "ARKIT_SYSTEM4": "ARKIT_SYSTEM4"}.get(cohort, cohort)
    records = {}
    for name, entry in locks.items():
        if lock_name is None and not name.startswith("CAJU_"):
            continue
        if lock_name is not None and name != lock_name:
            continue
        records.update({r["scene_id"]: r for r in json.loads(Path(entry["manifest"]).read_text())["scenes"]})
    episodes = defaultdict(list)
    for i, r in enumerate(rows):
        episodes[r["episode"]].append(i)
    gt = np.zeros((len(rows), H, W))
    for members in episodes.values():
        rec = records[rows[members[0]]["scene_id"]]
        teacher = S.TeacherAccess(rec, [rows[i]["query"] for i in members], S.IMAGE_SIZE, S.REPO, access)
        for i in members:
            g = teacher.query_depth(rows[i]["query"])
            gt[i] = np.where(np.isfinite(g) & (g > 0), g, 0.0)
    return gt


def query_weights(rows):
    """Weight per query: the place-level mean (mean over places of mean over episodes of mean over queries)."""
    places = defaultdict(lambda: defaultdict(list))
    for i, r in enumerate(rows):
        places[r["place"]][r["episode"]].append(i)
    w = np.zeros(len(rows))
    for eps in places.values():
        for members in eps.values():
            for i in members:
                w[i] = 1.0 / (len(places) * len(eps) * len(members))
    return w


def ray(z, a, cosine):
    return np.clip(z / np.maximum(a, 1e-3) / cosine, 0.1, 20.0)


def derived(arrays, i, tables):
    """Per-query maps (float64): r7, f7 (Core B's F7), Core B, the KF readout and the KF weight, regions, rays."""
    r7 = np.maximum(arrays["r7"][i].astype(np.float64), 1e-6)
    cosine = arrays["cosine"][i].astype(np.float64)
    near = arrays["near7"][i]
    tz, ta = arrays["ttt_z"][i].astype(np.float64), arrays["ttt_a"][i].astype(np.float64)
    cov = ta >= ALPHA_MIN
    f7 = np.maximum(np.where(cov, tz / np.maximum(ta, 1e-12) / cosine, r7), 1e-6)
    q = {"r7": r7.ravel(), "f7": f7.ravel(), "alpha": ta.ravel(), "near": near.ravel(), "log_p": np.zeros(r7.size)}
    covf, feats = K.covered_features(q)
    w = np.zeros(r7.size)
    w[covf] = K.WEIGHTS[np.asarray(tables["weights"]["KF_BASE"])[K.bin_index(feats, K.ARMS["KF_BASE"], tables["edges"])]]
    w = w.reshape(r7.shape)
    return {"r7": r7, "f7": f7, "cosine": cosine, "near": near, "cov": cov, "ttt_a": ta,
            "core_b": np.sqrt(r7 * f7), "kf": np.exp((1 - w) * np.log(r7) + w * np.log(f7)), "w_kf": w,
            "ray_ttt": ray(tz, ta, cosine),
            "ray_ff": ray(arrays["ff_z"][i].astype(np.float64), arrays["ff_a"][i].astype(np.float64), cosine),
            "ff_a": arrays["ff_a"][i].astype(np.float64)}


def fill_distance(near):
    """Pixels from the nearest near7 pixel (0 inside near7); 999 when the query has no near7 pixel."""
    if not near.any():
        return np.full(near.shape, 999.0)
    return ndimage.distance_transform_edt(~near)


def strata_masks(d, gt, hit7):
    valid = gt > 0
    dist = fill_distance(d["near"])
    unc_fill = ~d["cov"] & ~d["near"]
    lg = np.log(np.where(valid, gt, 1.0))
    edge = np.zeros_like(valid)
    for axis in (0, 1):
        diff = np.abs(np.diff(lg, axis=axis)) > 0.1
        both = np.logical_and(np.take(valid, range(1, valid.shape[axis]), axis=axis),
                              np.take(valid, range(0, valid.shape[axis] - 1), axis=axis))
        mark = diff & both
        if axis == 0:
            edge[1:, :] |= mark
            edge[:-1, :] |= mark
        else:
            edge[:, 1:] |= mark
            edge[:, :-1] |= mark
    m = {"ALL": valid, "COV": valid & d["cov"], "UNC_NEAR": valid & ~d["cov"] & d["near"], "UNC_FILL": valid & unc_fill,
         "ZERO_HIT": valid & (hit7 == 0), "FILL_D0_8": valid & unc_fill & (dist <= 8),
         "FILL_D8_32": valid & unc_fill & (dist > 8) & (dist <= 32), "FILL_D32": valid & unc_fill & (dist > 32),
         "EDGE": valid & edge, "FAR": valid & (gt > 4.0)}
    return m


def region_sums(pred, gt, masks):
    """{stratum: [sum of AbsRel, pixel count]} for one prediction."""
    err = np.where(gt > 0, np.abs(pred - gt) / np.where(gt > 0, gt, 1.0), 0.0)
    return {k: [float(err[m].sum()), int(m.sum())] for k, m in masks.items()}


def geometry_channels(d, hit7):
    """System A's geometric inputs (10 x H x W, float32): log R7, near7, cosine, fill distance, zero-hit flag,
    p10 / p50 / p90 of log R7 over near7 pixels (0 when none), pixel row and column in [-1, 1]."""
    lr = np.log(d["r7"])
    dist = np.clip(np.log1p(fill_distance(d["near"])) / math.log(64.0), 0.0, 1.5)
    if d["near"].any():
        q10, q50, q90 = np.quantile(lr[d["near"]], [0.1, 0.5, 0.9])
    else:
        q10 = q50 = q90 = 0.0
    rows = np.broadcast_to(np.linspace(-1, 1, H)[:, None], (H, W))
    cols = np.broadcast_to(np.linspace(-1, 1, W)[None, :], (H, W))
    full = lambda v: np.full((H, W), v)  # noqa: E731
    return np.stack([lr, d["near"].astype(np.float64), d["cosine"], dist, full(1.0 if hit7 == 0 else 0.0), full(q10),
                     full(q50), full(q90), rows, cols]).astype(np.float32)


def system_channels(d, kind):
    """The four channels of one DepthSplat system (kind 'ff' = feed-forward, 'ttt' = 100-step TTT)."""
    rz, a = (d["ray_ff"], d["ff_a"]) if kind == "ff" else (d["ray_ttt"], d["ttt_a"])
    lr = np.log(d["r7"])
    lray = np.log(rz)
    return np.stack([lray, a, np.clip(lray - lr, -2.0, 2.0), (a >= ALPHA_MIN).astype(np.float64)]).astype(np.float32)


def v2_base(arrays, i):
    """FUSENET / V2's 8-channel base image (exact V2 inputs)."""
    return FD.base_image(arrays["ttt_z"][i], arrays["ttt_a"][i], arrays["r7"][i], arrays["near7"][i], arrays["cosine"][i])
