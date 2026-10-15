"""EXP-3D-CAJU3D-FUSENET-V1: training the learned readouts on the FIT cohorts (GPU 4; FIT truth only).

FIT = every ARKIT_SYSTEM + ARKIT_SYSTEM2 episode (160 places). The places are split once by seed (SPLIT_SEED): 32
validation places, 128 training places. Arms BASE (base image only) and JEPA (base + the H-JEPA's query-cell
prediction), three seeds each, identical architecture and schedule: Adam (lr 1e-3, weight decay 1e-4), EPOCHS epochs,
batches of 8 queries, loss = the place-weighted mean AbsRel of the batch (fn_model.log_prediction). The checkpoint
of the epoch with the lowest validation place-level AbsRel is kept. Writes models/, results/train_log.json and logs
every model's hash in audit/fit_log.txt; refuses when any confirmation result exists.
Run from the repository root with the DepthSplat environment (PYTHONPATH=src):  python fn_train.py ROOT
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fn_data as D  # noqa: E402
import fn_model as M  # noqa: E402

FIT = ("FIT_SYSTEM", "FIT_SYSTEM2")
ARMS = {"BASE": False, "JEPA": True}
SEEDS = (20261101, 20261102, 20261103)
SPLIT_SEED, N_VAL, EPOCHS, BATCH, LR, WD = 20261100, 32, 30, 8, 1e-3, 1e-4


def split_places(places):
    order = sorted(places)
    perm = np.random.default_rng(SPLIT_SEED).permutation(len(order))
    val = {order[i] for i in perm[:N_VAL]}
    return sorted(set(order) - val), sorted(val)


def tensors(queries, device):
    base = torch.tensor(np.stack([q["base"] for q in queries]), device=device)
    jepa = torch.tensor(np.stack([q["jepa"] for q in queries]), device=device)
    gt = torch.tensor(np.stack([q["gt"] for q in queries]), dtype=torch.float32, device=device)
    return base, jepa, gt


@torch.no_grad()
def evaluate(model, base, jepa, gt, weights, use_jepa):
    total = 0.0
    for i in range(0, len(base), 32):
        logit, r = model(base[i:i + 32], jepa[i:i + 32] if use_jepa else None)
        err = M.absrel_per_query(M.log_prediction(base[i:i + 32], logit, r), gt[i:i + 32])
        total += float((err.double() * weights[i:i + 32]).sum())
    return total


def main():
    root = Path(sys.argv[1]).resolve()
    if (root / "results/confirm_results.json").exists() or (root / "models").exists():
        raise SystemExit("models or confirmation results exist already")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    access, queries = [], []
    for cohort in FIT:
        queries += D.load_cohort(root, cohort, access, with_jepa=True)
    train_places, val_places = split_places({q["place"] for q in queries})
    train = [q for q in queries if q["place"] in set(train_places)]
    val = [q for q in queries if q["place"] in set(val_places)]
    tb, tj, tg = tensors(train, device)
    vb, vj, vg = tensors(val, device)
    tw = torch.tensor(D.query_weights(train), dtype=torch.float64, device=device)
    vw = torch.tensor(D.query_weights(val), dtype=torch.float64, device=device)
    (root / "models").mkdir()
    log = {"fit_cohorts": list(FIT), "train_places": train_places, "val_places": val_places,
           "train_queries": len(train), "val_queries": len(val), "ground_truth_reads": len(access),
           "epochs": EPOCHS, "batch": BATCH, "lr": LR, "weight_decay": WD, "device": device, "runs": {}}
    started = time.time()
    for arm, use_jepa in ARMS.items():
        for seed in SEEDS:
            torch.manual_seed(seed)
            model = M.FuseNet(use_jepa).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
            generator = torch.Generator().manual_seed(seed)
            curve = [evaluate(model, vb, vj, vg, vw, use_jepa)]  # epoch 0 = Core B
            best, best_state, best_epoch = curve[0], {k: v.clone() for k, v in model.state_dict().items()}, 0
            for epoch in range(1, EPOCHS + 1):
                model.train()
                order = torch.randperm(len(tb), generator=generator).to(device)
                for i in range(0, len(order), BATCH):
                    idx = order[i:i + BATCH]
                    logit, r = model(tb[idx], tj[idx] if use_jepa else None)
                    err = M.absrel_per_query(M.log_prediction(tb[idx], logit, r), tg[idx])
                    w = tw[idx].float()
                    loss = (err * w).sum() / w.sum()
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
                model.eval()
                curve.append(evaluate(model, vb, vj, vg, vw, use_jepa))
                if curve[-1] < best:
                    best, best_epoch = curve[-1], epoch
                    best_state = {k: v.clone() for k, v in model.state_dict().items()}
            path = root / "models" / f"{arm}_s{seed}.pt"
            torch.save({"arm": arm, "seed": seed, "use_jepa": use_jepa, "epoch": best_epoch,
                        "state_dict": {k: v.cpu() for k, v in best_state.items()}}, path)
            log["runs"][f"{arm}_s{seed}"] = {"best_epoch": best_epoch, "best_val": best, "val_curve": curve,
                                             "sha256": D.sha(path)}
            print(f"{arm} seed {seed}: val Core B {curve[0]:.5f} -> best {best:.5f} at epoch {best_epoch} "
                  f"({time.time() - started:.0f}s)", flush=True)
    log["seconds"] = round(time.time() - started, 1)
    (root / "results").mkdir(exist_ok=True)
    (root / "results/train_log.json").write_text(json.dumps(log, indent=1) + "\n")
    (root / "audit").mkdir(exist_ok=True)
    with (root / "audit/fit_log.txt").open("a") as f:
        stamp = time.strftime("%Y-%m-%d %H:%M:%S %z")
        for name, run in log["runs"].items():
            f.write(f"{stamp} {name} {run['sha256']}\n")
    print("FN_TRAIN_DONE", log["seconds"], "s")


if __name__ == "__main__":
    main()
