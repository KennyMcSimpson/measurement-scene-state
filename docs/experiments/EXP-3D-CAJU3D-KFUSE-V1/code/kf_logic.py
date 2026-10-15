"""EXP-3D-CAJU3D-KFUSE-V1 (CAJU-3D step 3): the Kalman-style per-pixel readout fusion (numpy only).

Core B reads a query pixel as sqrt(R7 x F7): F7 is the TTT ray depth where the TTT alpha >= 0.5 (else R7), R7 the
7-view reprojection (harmonic fill where no view hits). The fusion replaces the fixed 0.5 by a per-pixel weight on
the covered pixels (alpha >= 0.5):  fused = R7^(1 - w) F7^w,  w = table[bin(pixel)];  uncovered pixels keep R7.
A table is a grid of feature bins with one weight per bin (from WEIGHTS), fitted on the FIT cohorts by the summed
place-weighted AbsRel and then frozen. Arms:
  CORE_B   w = 0.5 everywhere (the comparator)
  GEO_U    one weight for every covered pixel
  KF_DIS   DIS (8 bins)
  KF_BASE  DIS (8) x ALPHA (4) x NEAR (2)          non-JEPA: the TTT state's own opacity
  KF_JEPA  DIS (8) x LOGP (4) x NEAR (2)           JEPA: the reverse H-JEPA's predicted log variance at the query
  KF_FULL  DIS (8) x ALPHA (4) x LOGP (4) x NEAR (2)
Features of a covered pixel: DIS = |log F7 - log R7|; ALPHA = the TTT alpha; NEAR = R7 is a direct reprojection hit
(geometry seal near7), not a harmonic fill; LOGP = the H-JEPA's log P at the query cell, predicted from the 7 arrived
views and the query camera only.
"""

import numpy as np

ALPHA_MIN = 0.5
WEIGHTS = np.round(np.linspace(0.0, 1.0, 21), 2)
FEATURE_BINS = {"DIS": 8, "ALPHA": 4, "LOGP": 4}
ARMS = {"GEO_U": (), "KF_DIS": ("DIS",), "KF_BASE": ("DIS", "ALPHA", "NEAR"), "KF_JEPA": ("DIS", "LOGP", "NEAR"),
        "KF_FULL": ("DIS", "ALPHA", "LOGP", "NEAR")}


def covered_features(q):
    """q: dict of valid-pixel arrays r7, f7, alpha, near, log_p. Returns the covered mask and its feature arrays."""
    cov = q["alpha"] >= ALPHA_MIN
    return cov, {"DIS": np.abs(np.log(q["f7"][cov]) - np.log(q["r7"][cov])), "ALPHA": q["alpha"][cov],
                 "LOGP": q["log_p"][cov], "NEAR": q["near"][cov].astype(np.int64)}


def quantile_edges(values, bins):
    """Inner edges of `bins` equal-count bins."""
    return np.quantile(values, np.linspace(0.0, 1.0, bins + 1)[1:-1])


def bin_index(feats, names, edges):
    """Flat bin index of every covered pixel for an arm's feature tuple (NEAR is binary)."""
    index = np.zeros(len(feats["DIS"]), dtype=np.int64)
    for name in names:
        b = feats[name] if name == "NEAR" else np.digitize(feats[name], edges[name])
        index = index * (2 if name == "NEAR" else FEATURE_BINS[name]) + b
    return index


def n_bins(names):
    return int(np.prod([2 if n == "NEAR" else FEATURE_BINS[n] for n in names])) if names else 1


def weight_errors(q, cov):
    """[len(WEIGHTS), n_covered] AbsRel of the fused prediction under every weight."""
    lr, lf, g = np.log(q["r7"][cov]), np.log(q["f7"][cov]), q["gt"][cov]
    return np.abs(np.exp((1.0 - WEIGHTS)[:, None] * lr + WEIGHTS[:, None] * lf) - g) / g


def place_weights(queries):
    """Per query the weight of each of its pixels so that the weighted AbsRel sum equals the cohort's mean over
    places of the mean over episodes of the mean over queries (V1's place aggregation)."""
    places = {}
    for i, q in enumerate(queries):
        places.setdefault(q["place"], {}).setdefault(q["episode"], []).append(i)
    w = {}
    for episodes in places.values():
        for members in episodes.values():
            for i in members:
                w[i] = 1.0 / (len(places) * len(episodes) * len(members) * queries[i]["gt"].size)
    return w


def accumulate(queries, edges, arms=ARMS):
    """Per arm and per place: [bins, len(WEIGHTS)] sums of the weighted AbsRel of the covered pixels."""
    pw = place_weights(queries)
    loss = {arm: {} for arm in arms}
    for i, q in enumerate(queries):
        cov, feats = covered_features(q)
        err = weight_errors(q, cov) * pw[i]
        for arm, names in arms.items():
            idx = bin_index(feats, names, edges)
            m = loss[arm].setdefault(q["place"], np.zeros((n_bins(names), len(WEIGHTS))))
            for j in range(len(WEIGHTS)):
                m[:, j] += np.bincount(idx, weights=err[j], minlength=n_bins(names))
    return loss


def fit_weights(loss_by_place, exclude=None):
    """Per bin the weight index minimizing the summed loss over the places (except `exclude`); empty bins -> 0.5."""
    total = sum(m for p, m in loss_by_place.items() if p != exclude)
    j = np.argmin(total, axis=1)
    empty = ~np.any(total > 0, axis=1)
    j[empty] = int(np.where(WEIGHTS == 0.5)[0][0])
    return j


def readout(q, arm, table):
    """The arm's prediction on the valid pixels of query q. table = {"edges": ..., "weights": {arm: [index]}}."""
    out = q["r7"].copy()
    cov, feats = covered_features(q)
    if arm == "CORE_B":
        w = np.full(cov.sum(), 0.5)
    else:
        w = WEIGHTS[np.asarray(table["weights"][arm])[bin_index(feats, ARMS[arm], table["edges"])]]
    out[cov] = np.exp((1.0 - w) * np.log(q["r7"][cov]) + w * np.log(q["f7"][cov]))
    return out


def branch(p1, p2):
    if p1["status"] == "SUPPORTED" and p2["mean"] > 0:
        return "JEPA_KALMAN_SUPPORTED"
    if p1["status"] in ("SUPPORTED", "PARTIAL"):
        return "JEPA_KALMAN_PARTIAL"
    if p1["status"] == "HARMFUL":
        return "JEPA_KALMAN_HARMFUL"
    return "JEPA_KALMAN_NOT_ESTABLISHED"
