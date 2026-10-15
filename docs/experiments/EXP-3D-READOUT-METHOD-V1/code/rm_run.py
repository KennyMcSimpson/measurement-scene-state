"""EXP-3D-READOUT-METHOD-V1: Phase 1 (calibration controls vs V2) and Phase 2 (matched system readouts), leave one
cohort out (GPU 4 for the networks, CPU for the fits).

Folds hold out CONF_SYSTEM3, CONF_VAL, CAJU and ARKIT_SYSTEM4 in turn; FIT_SYSTEM and FIT_SYSTEM2 (which fitted the
frozen KF table) always train. In a fold, 10 % of the training places (seed 20261500 + fold index) validate the
networks' epochs; every fit and network uses the remaining training places only; the held-out cohort is scored.
Per held-out query: the AbsRel of every method on every stratum (rm_common.STRATA) and the predictions (float16).
  python rm_run.py ROOT smoke   (FIT_SYSTEM -> FIT_SYSTEM2, 1 seed, 1 epoch: a code check on training data only)
  python rm_run.py ROOT run     (the protocol's run; refuses without the lock)
Run from the repository root with the DepthSplat environment (PYTHONPATH=src).
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
import rm_methods as M  # noqa: E402
import s4_v2 as V2  # noqa: E402

SEEDS = {"B6": (20261501, 20261502, 20261503), "B6W": (20261511, 20261512, 20261513),
         "B7": (20261521, 20261522, 20261523), "SYS_A": (20261601, 20261602, 20261603),
         "SYS_B": (20261611, 20261612, 20261613), "SYS_C": (20261621, 20261622, 20261623)}
EPOCHS, BATCH, LR, WD, VAL_SHARE, MAX_FIT_PIXELS = 30, 8, 1e-3, 1e-4, 0.10, 3_000_000
SIMPLE = ("B2A", "B2B", "B3", "B4", "B5")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(cohort, tables, access):
    """Every per-query array the methods need (float32 stacks) + the rows and the ground truth."""
    arrays, rows = C.load_cache(cohort)
    gt = C.load_gt(cohort, rows, access)
    n = len(rows)
    out = {k: np.zeros((n, *s), np.float32) for k, s in (("base8", (8, C.H, C.W)), ("geo10", (10, C.H, C.W)),
                                                         ("ff4", (4, C.H, C.W)), ("ttt4", (4, C.H, C.W)),
                                                         ("logr7", (C.H, C.W)), ("logkf", (C.H, C.W)),
                                                         ("logcoreb", (C.H, C.W)), ("logcorea", (C.H, C.W)),
                                                         ("cov", (C.H, C.W)), ("near", (C.H, C.W)),
                                                         ("ray", (C.H, C.W)), ("alpha", (C.H, C.W)),
                                                         ("cosine", (C.H, C.W)), ("dist", (C.H, C.W)))}
    for i in range(n):
        d = C.derived(arrays, i, tables)
        out["base8"][i] = C.v2_base(arrays, i)
        out["geo10"][i] = C.geometry_channels(d, rows[i]["hit7"])
        out["ff4"][i] = C.system_channels(d, "ff")
        out["ttt4"][i] = C.system_channels(d, "ttt")
        out["logr7"][i], out["logkf"][i], out["logcoreb"][i] = np.log(d["r7"]), np.log(d["kf"]), np.log(d["core_b"])
        ffcov = d["ff_a"] >= C.ALPHA_MIN
        out["logcorea"][i] = np.log(np.sqrt(d["r7"] * np.where(ffcov, np.maximum(d["ray_ff"], 1e-6), d["r7"])))
        out["cov"][i], out["near"][i] = d["cov"], d["near"]
        out["ray"][i], out["alpha"][i], out["cosine"][i] = d["ray_ttt"], d["ttt_a"], d["cosine"]
        out["dist"][i] = np.clip(np.log1p(C.fill_distance(d["near"])) / np.log(64.0), 0, 1.5)
    out["gt"] = gt.astype(np.float32)
    out["zero"] = np.array([r["hit7"] == 0 for r in rows])
    return out, rows


def uncovered_pixels(data, idx, qw, rng):
    """1-D arrays over the uncovered valid pixels of queries idx, each pixel weighted qw_q / n_valid_q."""
    parts = {k: [] for k in ("r7", "g", "w", "near", "zero", "alpha", "cosine", "ray", "dist", "cov_share",
                             "near_share", "row", "col")}
    rows_ = np.broadcast_to(np.linspace(-1, 1, C.H)[:, None], (C.H, C.W))
    cols_ = np.broadcast_to(np.linspace(-1, 1, C.W)[None, :], (C.H, C.W))
    for j, i in enumerate(idx):
        valid = data["gt"][i] > 0
        m = valid & (data["cov"][i] < 0.5)
        if not m.any():
            continue
        nv = valid.sum()
        parts["r7"].append(np.exp(data["logr7"][i][m]).astype(np.float64))
        parts["g"].append(data["gt"][i][m].astype(np.float64))
        parts["w"].append(np.full(m.sum(), qw[j] / nv))
        parts["near"].append(data["near"][i][m] > 0.5)
        parts["zero"].append(np.full(m.sum(), bool(data["zero"][i])))
        for k in ("alpha", "cosine", "ray", "dist"):
            parts[k].append(data[k][i][m].astype(np.float64))
        parts["cov_share"].append(np.full(m.sum(), float(data["cov"][i].mean())))
        parts["near_share"].append(np.full(m.sum(), float(data["near"][i].mean())))
        parts["row"].append(rows_[m])
        parts["col"].append(cols_[m])
    f = {k: np.concatenate(v) for k, v in parts.items()}
    if len(f["g"]) > MAX_FIT_PIXELS:
        keep = rng.choice(len(f["g"]), MAX_FIT_PIXELS, replace=False)
        scale = len(f["g"]) / MAX_FIT_PIXELS
        f = {k: v[keep] for k, v in f.items()}
        f["w"] = f["w"] * scale
    return f


def query_pixels(data, i):
    """1-D uncovered-pixel features of one query (all pixels, for applying a fitted control)."""
    m = data["cov"][i] < 0.5
    rows_ = np.broadcast_to(np.linspace(-1, 1, C.H)[:, None], (C.H, C.W))
    cols_ = np.broadcast_to(np.linspace(-1, 1, C.W)[None, :], (C.H, C.W))
    n = int(m.sum())
    return m, {"r7": np.exp(data["logr7"][i][m]).astype(np.float64), "near": data["near"][i][m] > 0.5,
               "zero": np.full(n, bool(data["zero"][i])), "alpha": data["alpha"][i][m].astype(np.float64),
               "cosine": data["cosine"][i][m].astype(np.float64), "ray": data["ray"][i][m].astype(np.float64),
               "dist": data["dist"][i][m].astype(np.float64), "cov_share": np.full(n, float(data["cov"][i].mean())),
               "near_share": np.full(n, float(data["near"][i].mean())), "row": rows_[m], "col": cols_[m]}


def train_family(name, data, tr, va, qtr, qva, device, seeds, epochs):
    """Train one network family (3 seeds), epoch chosen on the inner validation places; returns models + log."""
    def inputs(idx):
        if name in ("B6", "B6W", "B7"):
            return torch.tensor(data["base8"][idx], device=device)
        parts = [data["geo10"][idx]]
        if name == "SYS_B":
            parts.append(data["ff4"][idx])
        if name == "SYS_C":
            parts.append(data["ttt4"][idx])
        return torch.tensor(np.concatenate(parts, axis=1), device=device)

    def aux(idx):
        return {k: torch.tensor(data[k][idx], device=device) for k in ("logr7", "logkf", "cov", "gt")}

    def predict(model, x, a):
        if name in ("B6", "B6W", "B7"):
            return torch.where(a["cov"] > 0.5, a["logkf"], a["logr7"] + model(x))
        return a["logr7"] + model(x)

    def make():
        if name == "B6":
            return M.PixelMLP((32, 32))
        if name == "B6W":
            return M.PixelMLP((150, 150, 150))
        if name == "B7":
            return V2.V2Net()
        return M.SysNet({"SYS_A": 10, "SYS_B": 14, "SYS_C": 14}[name])

    xtr, atr, xva, ava = inputs(tr), aux(tr), inputs(va), aux(va)
    wtr = torch.tensor(qtr, dtype=torch.float32, device=device)
    wva = torch.tensor(qva, dtype=torch.float64, device=device)

    def absrel(lp, g):
        valid = g > 0
        err = torch.where(valid, (torch.exp(lp) - g).abs() / g.clamp_min(1e-6), torch.zeros_like(g))
        return err.sum((1, 2)) / valid.sum((1, 2)).clamp_min(1)

    @torch.no_grad()
    def evaluate(model):
        total = 0.0
        for s in range(0, len(va), 32):
            sl = slice(s, s + 32)
            total += float((absrel(predict(model, xva[sl], {k: v[sl] for k, v in ava.items()}), ava["gt"][sl]).double()
                            * wva[sl]).sum())
        return total

    models, log = [], {}
    for seed in seeds:
        torch.manual_seed(seed)
        model = make().to(device)
        opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
        gen = torch.Generator().manual_seed(seed)
        curve = [evaluate(model)]
        best, best_epoch, state = curve[0], 0, {k: v.clone() for k, v in model.state_dict().items()}
        for epoch in range(1, epochs + 1):
            model.train()
            order = torch.randperm(len(tr), generator=gen).to(device)
            for s in range(0, len(order), BATCH):
                b = order[s:s + BATCH]
                err = absrel(predict(model, xtr[b], {k: v[b] for k, v in atr.items()}), atr["gt"][b])
                loss = (err * wtr[b]).sum() / wtr[b].sum()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
            model.eval()
            curve.append(evaluate(model))
            if curve[-1] < best:
                best, best_epoch, state = curve[-1], epoch, {k: v.clone() for k, v in model.state_dict().items()}
        model.load_state_dict(state)
        models.append(model.eval())
        log[str(seed)] = {"best_epoch": best_epoch, "best_val": best, "val_curve": curve}
    del xtr, atr, xva, ava
    return models, log, (inputs, predict)


def main():
    root, stage = Path(sys.argv[1]).resolve(), sys.argv[2]
    if stage == "run" and not (root / "PREREGISTRATION.json").is_file():
        raise SystemExit("lock first")
    if sha(C.TABLES) != C.TABLES_SHA:
        raise PermissionError("the KFUSE-V1 fusion tables changed")
    tables = json.loads(C.TABLES.read_text())
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    cohorts =("FIT_SYSTEM", "FIT_SYSTEM2") if stage == "smoke" else C.COHORTS
    folds = ("FIT_SYSTEM2",) if stage == "smoke" else C.FOLDS
    seeds = {k: v[:1] for k, v in SEEDS.items()} if stage == "smoke" else SEEDS
    epochs = 1 if stage == "smoke" else EPOCHS
    out_dir = root / ("smoke" if stage == "smoke" else "results/phase12")
    out_dir.mkdir(parents=True, exist_ok=True)
    access, data, rows, started = [], {}, {}, time.time()
    for c in cohorts:
        data[c], rows[c] = prepare(c, tables, access)
        print(f"prepared {c}: {len(rows[c])} queries ({time.time() - started:.0f}s)", flush=True)
    if {a["purpose"] for a in access} != {"SEALED_QUERY_EVALUATION"}:
        raise PermissionError("unexpected read purposes")
    for n, fold in enumerate(folds):
        train_cohorts = [c for c in cohorts if c != fold]
        # one pooled training pool
        pool = {k: np.concatenate([data[c][k] for c in train_cohorts]) for k in data[train_cohorts[0]]}
        prow = [r for c in train_cohorts for r in rows[c]]
        places = sorted({r["place"] for r in prow})
        perm = np.random.default_rng(20261500 + n).permutation(len(places))
        val_places = {places[i] for i in perm[:int(round(VAL_SHARE * len(places)))]}
        tr = np.array([i for i, r in enumerate(prow) if r["place"] not in val_places])
        va = np.array([i for i, r in enumerate(prow) if r["place"] in val_places])
        qtr = C.query_weights([prow[i] for i in tr])
        qva = C.query_weights([prow[i] for i in va])
        rng = np.random.default_rng(20261550 + n)
        unc = uncovered_pixels(pool, tr, qtr, rng)
        fits = {"B2A": M.fit_b2(unc)}
        fits["B2B"] = fits["B2A"]
        fits["B3"] = M.fit_b3(unc, fits["B2A"]["c_zero"])
        fits["B4"] = M.fit_b4(unc, device)
        fits["B5"] = M.fit_b5(unc)
        print(f"{fold}: fits done ({time.time() - started:.0f}s)", flush=True)
        nets, logs = {}, {}
        for name in ("B6", "B6W", "B7", "SYS_A", "SYS_B", "SYS_C"):
            models, log, fns = train_family(name, pool, tr, va, qtr[:], qva[:], device, seeds[name], epochs)
            nets[name], logs[name] = (models, fns), log
            print(f"{fold}: {name} trained ({time.time() - started:.0f}s)", flush=True)
        ev, erow = data[fold], rows[fold]
        methods = ("R7", "CORE_A", "B0", "B1") + SIMPLE + ("B6", "B6W", "B7", "SYS_A", "SYS_B", "SYS_C",
                                                             "SYS_C_ON_B", "SYS_B_ON_C")
        preds = {m: np.zeros((len(erow), C.H, C.W), np.float16) for m in methods}
        metrics = []
        for i in range(len(erow)):
            lp = {"R7": ev["logr7"][i], "CORE_A": ev["logcorea"][i], "B0": ev["logcoreb"][i], "B1": ev["logkf"][i]}
            m, f = query_pixels(ev, i)
            for name in SIMPLE:
                x = ev["logkf"][i].astype(np.float64).copy()
                x[m] = M.apply_b(name, fits[name], f)
                lp[name] = x
            with torch.no_grad():
                for name in ("B6", "B6W", "B7", "SYS_A", "SYS_B", "SYS_C"):
                    models, (inputs, predict) = nets[name]
                    x = inputs_eval(ev, i, name, device)
                    a = {k: torch.tensor(ev[k][i:i + 1], device=device) for k in ("logr7", "logkf", "cov", "gt")}
                    lp[name] = torch.stack([predict(mm, x, a) for mm in models]).mean(0)[0].double().cpu().numpy()
                for name, src, swap in (("SYS_C_ON_B", "SYS_C", "SYS_B"), ("SYS_B_ON_C", "SYS_B", "SYS_C")):
                    models, (inputs, predict) = nets[src]
                    x = inputs_eval(ev, i, swap, device)
                    a = {k: torch.tensor(ev[k][i:i + 1], device=device) for k in ("logr7", "logkf", "cov", "gt")}
                    lp[name] = torch.stack([predict(mm, x, a) for mm in models]).mean(0)[0].double().cpu().numpy()
            gt = ev["gt"][i].astype(np.float64)
            masks = C.strata_masks({"near": ev["near"][i] > 0.5, "cov": ev["cov"][i] > 0.5}, gt, erow[i]["hit7"])
            entry = {"key": erow[i]["key"], "place": erow[i]["place"], "episode": erow[i]["episode"], "fold": fold,
                     "hit7": erow[i]["hit7"], "methods": {}}
            for name, v in lp.items():
                pred = np.exp(v)
                preds[name][i] = pred.astype(np.float16)
                entry["methods"][name] = C.region_sums(pred, gt, masks)
            metrics.append(entry)
        np.savez_compressed(out_dir / f"preds_{fold}.npz", **preds)
        (out_dir / f"fold_{fold}.json").write_text(json.dumps({
            "fold": fold, "train_cohorts": train_cohorts, "train_places": len(places) - len(val_places),
            "val_places": sorted(val_places), "fits": fits, "train_logs": logs,
            "n_params": {k: M.n_params(nets[k][0][0]) for k in nets}, "queries": metrics}, indent=1) + "\n")
        print(f"{fold}: scored {len(erow)} queries ({time.time() - started:.0f}s)", flush=True)
    print("RM_RUN_DONE", round(time.time() - started, 1), "s; ground-truth reads", len(access), flush=True)


def inputs_eval(ev, i, name, device):
    if name in ("B6", "B6W", "B7"):
        return torch.tensor(ev["base8"][i:i + 1], device=device)
    parts = [ev["geo10"][i:i + 1]]
    if name == "SYS_B":
        parts.append(ev["ff4"][i:i + 1])
    if name == "SYS_C":
        parts.append(ev["ttt4"][i:i + 1])
    return torch.tensor(np.concatenate(parts, axis=1), device=device)


if __name__ == "__main__":
    main()
