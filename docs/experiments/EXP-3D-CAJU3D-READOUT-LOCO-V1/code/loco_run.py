"""EXP-3D-CAJU3D-READOUT-LOCO-V1: leave-one-cohort-out robustness of the FUSE_V2 recipe (GPU 4 training + CPU scoring).

Folds hold out CONF_SYSTEM3, CONF_VAL, CAJU (PILOT + SEL + EVAL) and ARKIT_SYSTEM4 in turn; FIT_SYSTEM and FIT_SYSTEM2
(which fitted the frozen KF table) only ever train. Per fold: three V2Net seeds trained on every other cohort by
SYSTEM4-HOLDOUT-V1's recipe (10 % of the training places validate the epoch), then CORE_B, KF_BASE and the ensemble
FUSE_LOCO are scored on the held-out cohort's places. Writes results/loco_results.json and results/loco_rows.json.
Run from the repository root with the DepthSplat environment (PYTHONPATH=src):  python loco_run.py ROOT
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
OUTPUTS = HERE.parents[1]
S4 = OUTPUTS / "EXP-3D-CAJU3D-SYSTEM4-HOLDOUT-V1"
sys.path.insert(0, str(S4 / "code"))
import s4_v2 as V  # noqa: E402  (sets the import paths)
import fn_model as M  # noqa: E402
from score_analyze import contrast  # noqa: E402

D, S = V.D, V.S
FOLDS = ("CONF_SYSTEM3", "CONF_VAL", "CAJU", "ARKIT_SYSTEM4")
SEEDS, EPOCHS, BATCH, LR, WD, VAL_SHARE = (20261301, 20261302, 20261303), 30, 8, 1e-3, 1e-4, 0.10


def load_system4(access, tables):
    lock = json.loads((S4 / "frame_role_lock.json").read_text())["cohorts"]["ARKIT_SYSTEM4"]
    if D.sha(lock["manifest"]) != lock["manifest_sha256"]:
        raise PermissionError("SYSTEM4 manifest changed")
    records = {r["scene_id"]: r for r in json.loads(Path(lock["manifest"]).read_text())["scenes"]}
    geometry = json.loads((S4 / "geometry/geometry_seal_ARKIT_SYSTEM4.json").read_text())["predictions"]
    ttt = {}
    for part in sorted((S4 / "ds/ARKIT_SYSTEM4/ZS").glob("ds_seal_*of*.json")):
        ttt.update(json.loads(part.read_text())["predictions"])
    out = []
    for e in lock["episodes"]:
        teacher = S.TeacherAccess(records[e["scene_id"]], e["queries"], S.IMAGE_SIZE, S.REPO, access)
        for qid in e["queries"]:
            key = f"{e['scene_id']}/{e['role']}/{qid}"
            g, t = S.load_checked(geometry[key]), S.load_checked(ttt[key])
            out.append(V.query_item(key, records[e["scene_id"]]["physical_scene_id"], f"{e['scene_id']}/{e['role']}",
                                    t["TTT_m1_depth_z"], t["TTT_m1_alpha"], g["REPROJ7_HARMONIC"], g["near7"],
                                    g["cosine"], teacher.query_depth(qid), tables))
    return out


def train_models(train, device, split_seed):
    places = sorted({q["place"] for q in train})
    perm = np.random.default_rng(split_seed).permutation(len(places))
    val_places = {places[i] for i in perm[:int(round(VAL_SHARE * len(places)))]}
    tr = [q for q in train if q["place"] not in val_places]
    va = [q for q in train if q["place"] in val_places]

    def stack(items, key):
        return torch.tensor(np.stack([q[key] for q in items]), dtype=torch.float32, device=device)

    tb, tw, tg = stack(tr, "base"), stack(tr, "w_kf"), stack(tr, "gt")
    vb, vw, vg = stack(va, "base"), stack(va, "w_kf"), stack(va, "gt")
    tq = torch.tensor(D.query_weights(tr), dtype=torch.float64, device=device)
    vq = torch.tensor(D.query_weights(va), dtype=torch.float64, device=device)

    @torch.no_grad()
    def evaluate(model):
        total = 0.0
        for i in range(0, len(vb), 32):
            err = M.absrel_per_query(V.log_prediction(vb[i:i + 32], vw[i:i + 32], model(vb[i:i + 32])), vg[i:i + 32])
            total += float((err.double() * vq[i:i + 32]).sum())
        return total

    models, runs = [], {}
    for seed in SEEDS:
        torch.manual_seed(seed)
        model = V.V2Net().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
        generator = torch.Generator().manual_seed(seed)
        best, best_epoch, best_state = evaluate(model), 0, {k: v.clone() for k, v in model.state_dict().items()}
        start = best
        for epoch in range(1, EPOCHS + 1):
            model.train()
            order = torch.randperm(len(tb), generator=generator).to(device)
            for i in range(0, len(order), BATCH):
                idx = order[i:i + BATCH]
                err = M.absrel_per_query(V.log_prediction(tb[idx], tw[idx], model(tb[idx])), tg[idx])
                w = tq[idx].float()
                loss = (err * w).sum() / w.sum()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            model.eval()
            score = evaluate(model)
            if score < best:
                best, best_epoch = score, epoch
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
        model.load_state_dict(best_state)
        models.append(model.eval())
        runs[str(seed)] = {"val_kf_base": start, "best_val": best, "best_epoch": best_epoch}
    return models, runs, len(places)


def main():
    root = Path(sys.argv[1]).resolve()
    if (root / "results/loco_results.json").exists():
        raise SystemExit("results exist already")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    if D.sha(V.TABLES) != V.TABLES_SHA:
        raise PermissionError("the KFUSE-V1 fusion tables changed")
    tables = json.loads(V.TABLES.read_text())
    access = []
    cohorts = {c: V.load_kf_cohort(c, access, tables) for c in V.KF_COHORTS}
    replay = V.caju_replay()
    cohorts["CAJU"] = [q for c in V.CAJU_COHORTS for q in V.load_caju_cohort(c, access, tables, replay)]
    cohorts["ARKIT_SYSTEM4"] = load_system4(access, tables)
    results, all_rows, started = {"folds": {}}, [], time.time()
    for n, fold in enumerate(FOLDS):
        train = [q for c, qs in cohorts.items() if c != fold for q in qs]
        models, runs, train_places = train_models(train, device, 20261300 + n)
        rows = []
        for q in cohorts[fold]:
            valid = q["gt"] > 0
            base = torch.tensor(q["base"][None], device=device)
            w_kf = torch.tensor(q["w_kf"][None], device=device)
            with torch.no_grad():
                fuse = torch.stack([V.log_prediction(base, w_kf, m(base)) for m in models]).mean(0)[0]
                kf = V.log_prediction(base, w_kf, torch.zeros_like(base[:, 0]))[0]
            log_r7, log_f7, cov = q["base"][0], q["base"][1], q["base"][6] > 0.5
            core = np.where(cov, 0.5 * (log_r7.astype(np.float64) + log_f7), log_r7)
            preds = {"CORE_B": np.exp(core), "KF_BASE": np.exp(kf.double().cpu().numpy()),
                     "FUSE_LOCO": np.exp(fuse.double().cpu().numpy())}
            gt = q["gt"][valid]
            rows.append({"fold": fold, "place": f"{fold}:{q['place']}", "episode": q["episode"], "query": q["key"],
                         "core_b": {k: float(np.mean(np.abs(v[valid] - gt) / gt)) for k, v in preds.items()}})
        table = S.place_table(rows)
        places = sorted(table["CORE_B"])
        gain = np.array([table["CORE_B"][p] - table["FUSE_LOCO"][p] for p in places])
        results["folds"][fold] = {"train_places": train_places, "runs": runs, "places": len(places),
                                  "means": {k: S.mean(v) for k, v in table.items()},
                                  "core_b_minus_fuse": contrast(table, "CORE_B", "FUSE_LOCO"),
                                  "kf_base_minus_fuse": contrast(table, "KF_BASE", "FUSE_LOCO"),
                                  "median_gain": float(np.median(gain)), "worse_places": int((gain < 0).sum())}
        all_rows += rows
        print(f"{fold}: Core B {results['folds'][fold]['means']['CORE_B']:.5f} -> FUSE_LOCO "
              f"{results['folds'][fold]['means']['FUSE_LOCO']:.5f}; worse {int((gain < 0).sum())}/{len(places)} "
              f"({time.time() - started:.0f}s)", flush=True)
    table = S.place_table(all_rows)
    places = sorted(table["CORE_B"])
    gain = np.array([table["CORE_B"][p] - table["FUSE_LOCO"][p] for p in places])
    results["pooled"] = {"places": len(places), "means": {k: S.mean(v) for k, v in table.items()},
                         "core_b_minus_fuse": contrast(table, "CORE_B", "FUSE_LOCO"),
                         "kf_base_minus_fuse": contrast(table, "KF_BASE", "FUSE_LOCO"),
                         "median_gain": float(np.median(gain)), "worse_places": int((gain < 0).sum()),
                         "gain_quartiles": np.quantile(gain, [0, 0.25, 0.5, 0.75, 1]).tolist()}
    results["ground_truth_reads"] = len(access)
    results["seconds"] = round(time.time() - started, 1)
    (root / "results").mkdir(exist_ok=True)
    (root / "results/loco_results.json").write_text(json.dumps(results, indent=1, sort_keys=True) + "\n")
    (root / "results/loco_rows.json").write_text(json.dumps(all_rows, indent=1, sort_keys=True) + "\n")
    print("LOCO_DONE", json.dumps({k: round(v, 5) for k, v in results["pooled"]["means"].items()}))


if __name__ == "__main__":
    main()
