"""EXP-3D-CAJU3D-KFUSE-V1 post-hoc exploration (FIT cohorts only; their truth is fitting data): can the uncovered
pixels (TTT alpha < 0.5, where Core B and every KFUSE arm keep R7) gain from the TTT ray depth depth_z / alpha / cos?

Per uncovered valid pixel: RAY = the TTT ray depth (alpha floored at 1e-3), R7, NEAR (near7), ALPHA. Bins: ALPHA in
[0, .05, .1, .2, .3, .4, .5) x NEAR x DIS (4 quantile bins of |log RAY - log R7|); per bin the weight w of
R7^(1 - w) RAY^w minimizing the place-weighted AbsRel. Reported in-sample and leave-one-place-out on FIT, added to
KF_BASE's frozen covered-pixel readout. Not a confirmation: an input to a possible later preregistration.
Run from the repository root (CUDA hidden, PYTHONPATH=src):  python hole_probe.py ROOT OUT.json
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(ROOT / "code"))
import kf_data as D  # noqa: E402
import kf_logic as K  # noqa: E402
from score_analyze import contrast  # noqa: E402

S = D.S
ALPHA_EDGES = np.array([0.05, 0.1, 0.2, 0.3, 0.4])
DIS_BINS = 4


def load(cohort, access):
    """kf_data.load_cohort plus the raw TTT ray depth on the same valid pixels."""
    queries = D.load_cohort(ROOT, cohort, access)
    reused = json.loads((ROOT / "reused_seals.json").read_text())[cohort]
    zs = S.reused_predictions(reused["ZS"])
    geometry = json.loads(Path(reused["geometry"]["path"]).read_text())["predictions"]
    lock = json.loads((ROOT / "frame_role_lock.json").read_text())["cohorts"][cohort]
    records = {r["scene_id"]: r for r in json.loads(Path(lock["manifest"]).read_text())["scenes"]}
    by_key = {q["key"]: q for q in queries}
    for e in lock["episodes"]:
        teacher = S.TeacherAccess(records[e["scene_id"]], e["queries"], S.IMAGE_SIZE, S.REPO, access)
        for qid in e["queries"]:
            key = f"{e['scene_id']}/{e['role']}/{qid}"
            ttt, geo = S.load_checked(zs[key]), S.load_checked(geometry[key])
            gt = teacher.query_depth(qid)
            valid = np.isfinite(gt) & (gt > 0)
            ray = ttt["TTT_m1_depth_z"] / np.maximum(ttt["TTT_m1_alpha"].astype(np.float64), 1e-3) / geo["cosine"]
            by_key[key]["ray"] = np.maximum(ray[valid], 1e-6)
    return queries


def main():
    tables = json.loads((ROOT / "fusion_tables.json").read_text())
    access, queries = [], []
    for cohort in ("FIT_SYSTEM", "FIT_SYSTEM2"):
        queries += load(cohort, access)
    pw = K.place_weights(queries)
    weights = K.WEIGHTS
    unc = [q["alpha"] < K.ALPHA_MIN for q in queries]
    dis_all = np.concatenate([np.abs(np.log(q["ray"][u]) - np.log(q["r7"][u])) for q, u in zip(queries, unc)])
    dis_edges = np.quantile(dis_all, np.linspace(0, 1, DIS_BINS + 1)[1:-1])
    n_bins = (len(ALPHA_EDGES) + 1) * 2 * DIS_BINS
    loss_by_place, base = {}, {}
    for i, (q, u) in enumerate(zip(queries, unc)):
        a = np.digitize(q["alpha"][u], ALPHA_EDGES)
        near = q["near"][u].astype(np.int64)
        d = np.digitize(np.abs(np.log(q["ray"][u]) - np.log(q["r7"][u])), dis_edges)
        idx = (a * 2 + near) * DIS_BINS + d
        lr, lf, g = np.log(q["r7"][u]), np.log(q["ray"][u]), q["gt"][u]
        err = np.abs(np.exp((1 - weights)[:, None] * lr + weights[:, None] * lf) - g) / g * pw[i]
        m = loss_by_place.setdefault(q["place"], np.zeros((n_bins, len(weights))))
        for j in range(len(weights)):
            m[:, j] += np.bincount(idx, weights=err[j], minlength=n_bins)
        q["hole_idx"] = idx
        # KF_BASE's frozen readout on the covered pixels, R7 on the uncovered ones
        base[i] = K.readout(q, "KF_BASE", tables)
    total = sum(loss_by_place.values())
    j_all = K.fit_weights(loss_by_place)

    def hole_readout(q, i, j_bin):
        out = base[i].copy()
        u = q["alpha"] < K.ALPHA_MIN
        w = weights[j_bin[q["hole_idx"]]]
        out[u] = np.exp((1 - w) * np.log(q["r7"][u]) + w * np.log(q["ray"][u]))
        return out

    rows = []
    for place in sorted(loss_by_place):
        j_loo = K.fit_weights(loss_by_place, exclude=place)
        for i, q in enumerate(queries):
            if q["place"] != place:
                continue
            rows.append({"place": q["place"], "episode": q["episode"], "query": q["key"], "core_b": {
                "KF_BASE": float(np.mean(np.abs(base[i] - q["gt"]) / q["gt"])),
                "KF_BASE_HOLE_INSAMPLE": float(np.mean(np.abs(hole_readout(q, i, j_all) - q["gt"]) / q["gt"])),
                "KF_BASE_HOLE_LOO": float(np.mean(np.abs(hole_readout(q, i, j_loo) - q["gt"]) / q["gt"]))}})
    table = S.place_table(rows)
    out = {"means": {k: S.mean(v) for k, v in table.items()},
           "KF_BASE_minus_HOLE_LOO": contrast(table, "KF_BASE", "KF_BASE_HOLE_LOO")}
    share = total.sum(axis=1)
    out["bins"] = [{"alpha_bin": int(b // (2 * DIS_BINS)), "near": int((b // DIS_BINS) % 2), "dis_bin": int(b % DIS_BINS),
                    "loss_share": float(share[b] / share.sum()), "w": float(weights[j_all[b]]),
                    "gain_vs_r7": float(total[b, 0] - total[b, j_all[b]])} for b in range(n_bins)]
    out["uncovered_error_at_r7"] = float(total[:, 0].sum())
    out["dis_edges"] = dis_edges.tolist()
    Path(sys.argv[2]).write_text(json.dumps(out, indent=1))
    c = out["KF_BASE_minus_HOLE_LOO"]
    print("means", json.dumps({k: round(v, 5) for k, v in out["means"].items()}))
    print("KF_BASE - HOLE_LOO", round(c["mean"], 5), [round(x, 5) for x in c["ci95"]], c["improved_tied_worse"], c["status"])
    print("uncovered place-weighted AbsRel at R7:", round(out["uncovered_error_at_r7"], 5))
    for b in out["bins"]:
        if b["gain_vs_r7"] > 1e-5:
            print("  bin", b)


if __name__ == "__main__":
    main()
