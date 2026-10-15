"""EXP-3D-CAJU3D-SYSTEM4-HOLDOUT-V1: training FUSE_V2 by the plan's fixed recipe (GPU 4; no SYSTEM4 data is touched).

458 places of 7 cohorts (s4_v2.load_training); 10 % of the places (seed 20261200) are the validation set for the epoch
choice. Three seeds (20261201-3), 30 epochs, Adam (lr 1e-3, weight decay 1e-4), batches of 8 queries, loss = the
place-weighted mean AbsRel (only the uncovered correction is learned). Writes models_v2/, results/v2_train_log.json
and the model hashes in audit/v2_fit_log.txt; refuses when any SYSTEM4 result exists.
Run from the repository root with the DepthSplat environment (PYTHONPATH=src):  python s4_v2_train.py ROOT
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import s4_v2 as V  # noqa: E402  (sets the import paths)
import fn_model as M  # noqa: E402

D = V.D
SEEDS, SPLIT_SEED, VAL_SHARE, EPOCHS, BATCH, LR, WD = (20261201, 20261202, 20261203), 20261200, 0.10, 30, 8, 1e-3, 1e-4


def main():
    root = Path(sys.argv[1]).resolve()
    if (root / "models_v2").exists() or (root / "results/holdout_results.json").exists():
        raise SystemExit("V2 models or holdout results exist already")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    access = []
    queries = V.load_training(access)
    places = sorted({q["place"] for q in queries})
    perm = np.random.default_rng(SPLIT_SEED).permutation(len(places))
    val_places = {places[i] for i in perm[:int(round(VAL_SHARE * len(places)))]}
    train = [q for q in queries if q["place"] not in val_places]
    val = [q for q in queries if q["place"] in val_places]

    def stack(items, key, dtype=torch.float32):
        return torch.tensor(np.stack([q[key] for q in items]), dtype=dtype, device=device)

    tb, tw, tg = stack(train, "base"), stack(train, "w_kf"), stack(train, "gt")
    vb, vw, vg = stack(val, "base"), stack(val, "w_kf"), stack(val, "gt")
    tq = torch.tensor(D.query_weights(train), dtype=torch.float64, device=device)
    vq = torch.tensor(D.query_weights(val), dtype=torch.float64, device=device)

    @torch.no_grad()
    def evaluate(model):
        total = 0.0
        for i in range(0, len(vb), 32):
            err = M.absrel_per_query(V.log_prediction(vb[i:i + 32], vw[i:i + 32], model(vb[i:i + 32])), vg[i:i + 32])
            total += float((err.double() * vq[i:i + 32]).sum())
        return total

    (root / "models_v2").mkdir()
    log = {"places": len(places), "train_places": len(places) - len(val_places), "val_places": sorted(val_places),
           "train_queries": len(train), "val_queries": len(val), "ground_truth_reads": len(access),
           "epochs": EPOCHS, "batch": BATCH, "lr": LR, "weight_decay": WD, "device": device, "runs": {}}
    started = time.time()
    for seed in SEEDS:
        torch.manual_seed(seed)
        model = V.V2Net().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
        generator = torch.Generator().manual_seed(seed)
        curve = [evaluate(model)]  # epoch 0 = KF_BASE
        best, best_epoch, best_state = curve[0], 0, {k: v.clone() for k, v in model.state_dict().items()}
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
            curve.append(evaluate(model))
            if curve[-1] < best:
                best, best_epoch = curve[-1], epoch
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
        path = root / "models_v2" / f"V2_s{seed}.pt"
        torch.save({"seed": seed, "epoch": best_epoch, "state_dict": {k: v.cpu() for k, v in best_state.items()}}, path)
        log["runs"][f"V2_s{seed}"] = {"best_epoch": best_epoch, "best_val": best, "val_curve": curve,
                                      "sha256": D.sha(path)}
        print(f"V2 seed {seed}: val KF_BASE {curve[0]:.5f} -> best {best:.5f} at epoch {best_epoch} "
              f"({time.time() - started:.0f}s)", flush=True)
    log["seconds"] = round(time.time() - started, 1)
    (root / "results").mkdir(exist_ok=True)
    (root / "results/v2_train_log.json").write_text(json.dumps(log, indent=1) + "\n")
    (root / "audit").mkdir(exist_ok=True)
    with (root / "audit/v2_fit_log.txt").open("a") as f:
        stamp = time.strftime("%Y-%m-%d %H:%M:%S %z")
        for name, run in log["runs"].items():
            f.write(f"{stamp} {name} {run['sha256']}\n")
    print("V2_TRAIN_DONE", log["seconds"], "s")


if __name__ == "__main__":
    main()
