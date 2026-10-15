"""EXP-3D-READOUT-METHOD-V1, Amendment 1 (post hoc: written after the Gate A / Gate B results; AMENDMENT_1.md).

Is TTT needed inside the strongest structure (the KF-table covered readout + V2's uncovered correction)? The same
structure on the released DepthSplat's feed-forward render (the CACHE state, no TTT):
  KF_FF      the KF_BASE table (DIS 8 x ALPHA 4 x NEAR 2) refit on FIT_SYSTEM + FIT_SYSTEM2 through kf_logic exactly as
             KFUSE-V1's stage_fit fitted the frozen table, with the feed-forward F7 and alpha; a replay of the TTT fit
             from the same cache is compared with the frozen table
  B1_FF      KF_FF on the feed-forward covered pixels (alpha_ff >= 0.5), R7 elsewhere
  B7_FF      V2's recipe on the feed-forward render (rm_run.train_family "B7" on the feed-forward data), seeds 20261631-3
  B1_REPLAY  B1 recomputed with the frozen table (must equal Phase 1's B1)
Folds, inner validation places, epochs and optimizer are rm_run's; strata use the protocol's TTT-alpha regions.
  python rm_amend1.py ROOT smoke  (code check on training cohorts only: FIT_SYSTEM -> FIT_SYSTEM2, 1 seed, 1 epoch)
  python rm_amend1.py ROOT lock   (hashes AMENDMENT_1.md and the amendment code; refuses when results exist)
  python rm_amend1.py ROOT run    (refuses without the lock or when a locked file changed)
Run from the repository root with the DepthSplat environment (PYTHONPATH=src), GPU 4.
"""

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import rm_common as C  # noqa: E402  (sets the import paths)
import rm_run as R  # noqa: E402
import kf_logic as K  # noqa: E402

SEEDS = (20261631, 20261632, 20261633)
FIT = ("FIT_SYSTEM", "FIT_SYSTEM2")
LOCKED = ("AMENDMENT_1.md", "code/rm_amend1.py", "code/rm_amend1_stats.py")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def feed_forward(arrays):
    """The cache with the feed-forward render in the TTT slots (rm_common reads ttt_z / ttt_a)."""
    out = dict(arrays)
    out["ttt_z"], out["ttt_a"] = arrays["ff_z"], arrays["ff_a"]
    return out


def kf_queries(arrays, rows, gt):
    """kf_logic's query dicts (valid pixels), F7 / alpha from the render in the TTT slots (rm_common.derived's F7)."""
    qs = []
    for i, r in enumerate(rows):
        valid = gt[i] > 0
        r7 = np.maximum(arrays["r7"][i].astype(np.float64), 1e-6)
        z, a = arrays["ttt_z"][i].astype(np.float64), arrays["ttt_a"][i].astype(np.float64)
        cosine = arrays["cosine"][i].astype(np.float64)
        f7 = np.maximum(np.where(a >= K.ALPHA_MIN, z / np.maximum(a, 1e-12) / cosine, r7), 1e-6)
        qs.append({"place": r["place"], "episode": r["episode"], "key": r["key"], "gt": gt[i][valid],
                   "r7": r7[valid], "f7": f7[valid], "alpha": a[valid], "near": arrays["near7"][i][valid].astype(bool),
                   "log_p": np.zeros(int(valid.sum()))})
    return qs


def fit_table(queries):
    """KFUSE-V1's stage_fit, KF_BASE arm only (its weights do not depend on the other arms)."""
    covered = [K.covered_features(q)[1] for q in queries]
    edges = {n: K.quantile_edges(np.concatenate([f[n] for f in covered]), b).tolist()
             for n, b in K.FEATURE_BINS.items()}
    loss = K.accumulate(queries, edges, {"KF_BASE": K.ARMS["KF_BASE"]})
    return {"edges": edges, "weights": {"KF_BASE": K.fit_weights(loss["KF_BASE"]).tolist()}}


def prepare(arrays, rows, gt, tables, tables_ff):
    """B7_FF's training / scoring arrays (feed-forward) + the frozen-table B1 and the protocol's TTT regions."""
    ff = feed_forward(arrays)
    n = len(rows)
    out = {"base8": np.zeros((n, 8, C.H, C.W), np.float32)}
    for k in ("logr7", "logkf", "cov", "near", "cov_ttt", "logkf_ttt"):
        out[k] = np.zeros((n, C.H, C.W), np.float32)
    for i in range(n):
        d = C.derived(ff, i, tables_ff)
        out["base8"][i] = C.v2_base(ff, i)
        out["logr7"][i], out["logkf"][i] = np.log(d["r7"]), np.log(d["kf"])
        out["cov"][i], out["near"][i] = d["cov"], d["near"]
        dt = C.derived(arrays, i, tables)
        out["cov_ttt"][i], out["logkf_ttt"][i] = dt["cov"], np.log(dt["kf"])
    out["gt"] = gt.astype(np.float32)
    out["zero"] = np.array([r["hit7"] == 0 for r in rows])
    return out


def lock(root):
    if (root / "results/amend1").exists() or (root / "AMENDMENT_1.lock.json").exists():
        raise SystemExit("cannot lock: amendment results or a lock exist already")
    rec = {"amendment": 1, "post_hoc": True, "locked_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
           "files": {n: sha(root / n) for n in LOCKED}, "preregistration_sha256": sha(root / "PREREGISTRATION.json"),
           "primary": "B7_FF - B7 (> 0: TTT helps inside the KF table + V2 structure), pooled 350 places",
           "thresholds": {"ci_lower_gt": 0.0, "min_mean_gain": 0.002, "median_gain_gt": 0.0, "max_top1_share": 0.5},
           "labels": ["TTT_ADDED_VALUE_IN_KF_V2_STRUCTURE=SUPPORTED", "...=HARMFUL (CI upper < 0)",
                      "...=NOT_ESTABLISHED"],
           "seeds": list(SEEDS), "gate_a_and_gate_b_unchanged": True}
    (root / "AMENDMENT_1.lock.json").write_text(json.dumps(rec, indent=1) + "\n")
    (root / "audit").mkdir(exist_ok=True)
    (root / "audit/amendment1_locked_at.txt").write_text(f"{rec['locked_at']} {sha(root / 'AMENDMENT_1.lock.json')}\n")
    print("AMEND1_LOCKED", rec["locked_at"], sha(root / "AMENDMENT_1.lock.json"))


def run(root, smoke=False):
    if not smoke:
        rec = json.loads((root / "AMENDMENT_1.lock.json").read_text())
        for name, h in rec["files"].items():
            if sha(root / name) != h:
                raise PermissionError(f"{name} changed since the amendment lock")
    cohorts, folds = (FIT, ("FIT_SYSTEM2",)) if smoke else (C.COHORTS, C.FOLDS)
    seeds, epochs = (SEEDS[:1], 1) if smoke else (SEEDS, R.EPOCHS)
    if sha(C.TABLES) != C.TABLES_SHA:
        raise PermissionError("the KFUSE-V1 fusion tables changed")
    tables = json.loads(C.TABLES.read_text())
    out_dir = root / ("smoke/amend1" if smoke else "results/amend1")
    out_dir.mkdir(parents=True, exist_ok=smoke)
    device = "cuda"
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    access, started, raw = [], time.time(), {}
    for c in cohorts:
        arrays, rows = C.load_cache(c)
        raw[c] = (arrays, rows, C.load_gt(c, rows, access))
    if {a["purpose"] for a in access} != {"SEALED_QUERY_EVALUATION"}:
        raise PermissionError("unexpected read purposes")
    fit_ff = [q for c in FIT for q in kf_queries(feed_forward(raw[c][0]), raw[c][1], raw[c][2])]
    tables_ff = fit_table(fit_ff)
    fit_ttt = [q for c in FIT for q in kf_queries(raw[c][0], raw[c][1], raw[c][2])]
    replay = fit_table(fit_ttt)
    replay_check = {
        "edge_max_abs_diff": {n: float(np.max(np.abs(np.array(replay["edges"][n]) - np.array(tables["edges"][n]))))
                              for n in ("DIS", "ALPHA")},
        "kf_base_bins_equal": int(np.sum(np.array(replay["weights"]["KF_BASE"]) ==
                                         np.array(tables["weights"]["KF_BASE"]))), "bins": 64,
        "fit_places": len({q["place"] for q in fit_ttt}), "fit_queries": len(fit_ttt)}
    tables_ff.update({"weight_grid": K.WEIGHTS.tolist(), "arms": {"KF_BASE": list(K.ARMS["KF_BASE"])},
                      "fit_cohorts": list(FIT), "fit_places": len({q["place"] for q in fit_ff}),
                      "render": "released DepthSplat feed-forward (CACHE state), no TTT"})
    (out_dir / "fusion_tables_ff.json").write_text(json.dumps(tables_ff, indent=1) + "\n")
    (out_dir / "kf_fit.json").write_text(json.dumps({"replay_of_frozen_ttt_table": replay_check,
                                                     "fusion_tables_ff_sha256": sha(out_dir / "fusion_tables_ff.json"),
                                                     "ff_kf_base_weights": [float(K.WEIGHTS[j]) for j in
                                                                            tables_ff["weights"]["KF_BASE"]],
                                                     "ttt_kf_base_weights": [float(K.WEIGHTS[j]) for j in
                                                                             tables["weights"]["KF_BASE"]]},
                                                    indent=1) + "\n")
    print("KF_FF fitted; replay", json.dumps(replay_check), f"({time.time() - started:.0f}s)", flush=True)
    del fit_ff, fit_ttt
    data, rows = {}, {}
    for c in cohorts:
        data[c], rows[c] = prepare(*raw[c], tables, tables_ff), raw[c][1]
        raw[c] = None
        print(f"prepared {c} ({time.time() - started:.0f}s)", flush=True)
    for n, fold in enumerate(folds):
        train_cohorts = [c for c in cohorts if c != fold]
        pool = {k: np.concatenate([data[c][k] for c in train_cohorts]) for k in ("base8", "logr7", "logkf", "cov", "gt")}
        prow = [r for c in train_cohorts for r in rows[c]]
        places = sorted({r["place"] for r in prow})
        perm = np.random.default_rng(20261500 + n).permutation(len(places))
        val_places = {places[i] for i in perm[:int(round(R.VAL_SHARE * len(places)))]}
        tr = np.array([i for i, r in enumerate(prow) if r["place"] not in val_places])
        va = np.array([i for i, r in enumerate(prow) if r["place"] in val_places])
        qtr = C.query_weights([prow[i] for i in tr])
        qva = C.query_weights([prow[i] for i in va])
        models, log, (_, predict) = R.train_family("B7", pool, tr, va, qtr, qva, device, seeds, epochs)
        del pool
        print(f"{fold}: B7_FF trained ({time.time() - started:.0f}s)", flush=True)
        ev, erow = data[fold], rows[fold]
        preds = {m: np.zeros((len(erow), C.H, C.W), np.float16) for m in ("B1_FF", "B7_FF")}
        metrics = []
        for i in range(len(erow)):
            lp = {"B1_FF": ev["logkf"][i], "B1_REPLAY": ev["logkf_ttt"][i]}
            with torch.no_grad():
                x = torch.tensor(ev["base8"][i:i + 1], device=device)
                a = {k: torch.tensor(ev[k][i:i + 1], device=device) for k in ("logr7", "logkf", "cov", "gt")}
                lp["B7_FF"] = torch.stack([predict(mm, x, a) for mm in models]).mean(0)[0].double().cpu().numpy()
            gt = ev["gt"][i].astype(np.float64)
            masks = C.strata_masks({"near": ev["near"][i] > 0.5, "cov": ev["cov_ttt"][i] > 0.5}, gt, erow[i]["hit7"])
            entry = {"key": erow[i]["key"], "place": erow[i]["place"], "episode": erow[i]["episode"], "fold": fold,
                     "hit7": erow[i]["hit7"], "ff_covered_share": float(ev["cov"][i].mean()),
                     "ttt_covered_share": float(ev["cov_ttt"][i].mean()), "methods": {}}
            for name, v in lp.items():
                pred = np.exp(v)
                if name in preds:
                    preds[name][i] = pred.astype(np.float16)
                entry["methods"][name] = C.region_sums(pred, gt, masks)
            metrics.append(entry)
        np.savez_compressed(out_dir / f"preds_{fold}.npz", **preds)
        (out_dir / f"fold_{fold}.json").write_text(json.dumps({
            "fold": fold, "train_cohorts": train_cohorts, "train_places": len(places) - len(val_places),
            "val_places": sorted(val_places), "train_logs": log, "n_params": sum(p.numel() for p in models[0].parameters()),
            "queries": metrics}, indent=1) + "\n")
        print(f"{fold}: scored {len(erow)} queries ({time.time() - started:.0f}s)", flush=True)
    print("RM_AMEND1_DONE", round(time.time() - started, 1), "s; ground-truth reads", len(access), flush=True)


if __name__ == "__main__":
    {"lock": lock, "run": run, "smoke": lambda r: run(r, smoke=True)}[sys.argv[2]](Path(sys.argv[1]).resolve())
