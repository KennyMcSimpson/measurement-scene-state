"""EXP-3D-READOUT-METHOD-V1, Amendment 3 (post hoc; AMENDMENT_3.md): the premise test of Candidate A in an explicit
geometric form, a free-space lower bound from the arrived measurements. Deployable (the 7 arrived RGB-D frames and the
query camera only), non-learned, CPU, applied to the saved Phase 1 predictions; no training.

Carving: a point X is free space when some arrived frame j sees it in front of its measured surface,
  |X - c_j| < D_j(pi_j(X)) (1 - REL) - ABS      (REL = 0.05, ABS = 0.05 m; D_j = measured ray distance, valid pixels)
A query ray o + t d is sampled at T = 128 log-spaced distances in [0.1, 20] m. Its free-space lower bound t_lb is the
last sample of the carved prefix (every sample up to it is free space; 0 when the first sample is not carved): a static
surface cannot lie there. The deployable readout X_FS clamps a prediction X on the protocol's uncovered pixels (TTT
alpha < 0.5) to max(X, t_lb). Predictions (float16, as saved): B7 (V2's recipe), B5 (the strongest simple control), R7.
Descriptive: how often t_lb constrains, how often a prediction or the ground truth lies inside carved space (the latter
= carving errors), per region.
Run from the repository root with the project environment (CUDA hidden, PYTHONPATH=src):
  python rm_amend3.py ROOT smoke   (code check on the smoke fold FIT_SYSTEM2, a training cohort -> smoke/amend3)
  python rm_amend3.py ROOT lock    (hashes AMENDMENT_3.md and this file; refuses when results exist)
  python rm_amend3.py ROOT run     (refuses without the lock or when a locked file changed)
"""

import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT, STAGE = Path(sys.argv[1]).resolve(), sys.argv[2]
sys.path.insert(0, str(ROOT / "code"))
import rm_common as C  # noqa: E402  (sets the import paths)
import rm_stats as RS  # noqa: E402  (reads sys.argv[1:]; its main() does not run)
from mcss.geometry import generate_rays, project_world  # noqa: E402

REL, ABS = 0.05, 0.05
T = np.geomspace(0.1, 20.0, 128)
METHODS = ("B7", "B5", "R7")
LOCKED = ("AMENDMENT_3.md", "code/rm_amend3.py")
LOCK_COHORTS = {"CAJU": ("CAJU_PILOT", "CAJU_SEL", "CAJU_EVAL")}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def lock_episodes(cohort):
    locks = json.loads((C.OUT / C.LOCKS[cohort]).read_text())["cohorts"]
    out, records = {}, {}
    for name in LOCK_COHORTS.get(cohort, (cohort,)):
        out.update({f"{e['scene_id']}/{e['role']}": e for e in locks[name]["episodes"]})
        records.update({r["scene_id"]: r for r in json.loads(Path(locks[name]["manifest"]).read_text())["scenes"]})
    return out, records


def arrived(teacher, episode):
    """(camera, measured ray distance [H, W], camera centre) of the 7 arrived frames."""
    frames = []
    for ids in (episode["context"], episode["stream"]):
        ctx = teacher.frames(ids)
        for obs, depth in zip(ctx, ctx.depths):
            cam = obs.camera.to(dtype=torch.float64)
            frames.append((cam, torch.as_tensor(depth, dtype=torch.float64).numpy(), cam.c2w[:3, 3].numpy()))
    return frames


def carved_samples(frames, o, d):
    """[P, len(T)] free-space flags of the samples along the rays o + t d."""
    pts = (o[:, None, :] + d[:, None, :] * T[None, :, None]).reshape(-1, 3)
    carved = np.zeros(len(pts), bool)
    tp = torch.as_tensor(pts)
    for cam, depth, centre in frames:
        pix, _, inside = project_world(tp, cam)
        pix, inside = np.nan_to_num(pix.numpy(), nan=-1.0, posinf=-1.0, neginf=-1.0), inside.numpy()  # behind: not inside
        h, w = cam.image_size
        u = np.clip(np.round(pix[:, 0]).astype(np.int64), 0, w - 1)
        v = np.clip(np.round(pix[:, 1]).astype(np.int64), 0, h - 1)
        meas = depth[v, u]
        ok = inside & np.isfinite(meas) & (meas > 0)
        carved |= ok & (np.linalg.norm(pts - centre, axis=1) < meas * (1.0 - REL) - ABS)
    return carved.reshape(-1, len(T))


def in_carved(carved, t):
    k = np.clip(np.searchsorted(T, t), 1, len(T) - 1)
    k = np.where(np.abs(np.log(T[k - 1]) - np.log(t)) < np.abs(np.log(T[k]) - np.log(t)), k - 1, k)
    return carved[np.arange(len(t)), k]


def lock():
    if (ROOT / "results/amend3").exists() or (ROOT / "AMENDMENT_3.lock.json").exists():
        raise SystemExit("cannot lock: amendment 3 results or a lock exist already")
    rec = {"amendment": 3, "post_hoc": True, "deployable": True, "locked_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
           "files": {n: sha(ROOT / n) for n in LOCKED},
           "primary": "B7 - B7_FS, pooled 350 places (> 0: the free-space lower bound lowers V2's error)",
           "decision": {"A_PREMISE_FREE_SPACE=SUPPORTED": "mean >= 0.005 and CI lower > 0",
                        "A_PREMISE_FREE_SPACE=NOT_SUPPORTED": "otherwise"},
           "carving": {"rel": REL, "abs_m": ABS, "samples": len(T), "range_m": [0.1, 20.0]}}
    (ROOT / "AMENDMENT_3.lock.json").write_text(json.dumps(rec, indent=1) + "\n")
    (ROOT / "audit/amendment3_locked_at.txt").write_text(f"{rec['locked_at']} {sha(ROOT / 'AMENDMENT_3.lock.json')}\n")
    print("AMEND3_LOCKED", rec["locked_at"], sha(ROOT / "AMENDMENT_3.lock.json"))


def run(smoke):
    if not smoke:
        rec = json.loads((ROOT / "AMENDMENT_3.lock.json").read_text())
        for name, h in rec["files"].items():
            if sha(ROOT / name) != h:
                raise PermissionError(f"{name} changed since the amendment 3 lock")
    torch.set_num_threads(8)
    out_dir = ROOT / ("smoke/amend3" if smoke else "results/amend3")
    out_dir.mkdir(parents=True, exist_ok=smoke)
    access, started, entries = [], time.time(), []
    desc = defaultdict(lambda: defaultdict(float))
    for fold in RS.FOLDS:
        arrays, rows = C.load_cache(fold)
        fj = json.loads((RS.RES / f"fold_{fold}.json").read_text())["queries"]
        if [e["key"] for e in fj] != [r["key"] for r in rows]:
            raise ValueError(f"{fold}: prediction order differs from the cache")
        saved = np.load(RS.RES / f"preds_{fold}.npz")
        preds = {m: saved[m] for m in METHODS}
        gt = C.load_gt(fold, rows, access)
        episodes, records = lock_episodes(fold)
        members = defaultdict(list)
        for i, r in enumerate(rows):
            members[r["episode"]].append(i)
        for ep, idx in members.items():
            e = episodes[ep]
            teacher = C.S.TeacherAccess(records[e["scene_id"]], [rows[i]["query"] for i in idx], C.S.IMAGE_SIZE,
                                        C.S.REPO, access)
            frames = arrived(teacher, e)
            for i in idx:
                cam = teacher.query_camera(rows[i]["query"]).to(dtype=torch.float64)
                o, d = (a.numpy().reshape(-1, 3) for a in generate_rays(cam))
                unc = (arrays["ttt_a"][i] < C.ALPHA_MIN).ravel()
                p = np.flatnonzero(unc)
                carved = carved_samples(frames, o[p], d[p])
                first_open = np.where((~carved).any(1), (~carved).argmax(1), len(T))
                t_lb = np.where(first_open > 0, T[np.maximum(first_open - 1, 0)], 0.0)
                g = gt[i].astype(np.float64)
                masks = C.strata_masks({"near": arrays["near7"][i], "cov": arrays["ttt_a"][i] >= C.ALPHA_MIN}, g,
                                       rows[i]["hit7"])
                entry = {"key": rows[i]["key"], "place": rows[i]["place"], "episode": rows[i]["episode"], "fold": fold,
                         "hit7": rows[i]["hit7"], "methods": {}}
                gv = g.ravel()[p]
                valid = gv > 0
                for m in METHODS:
                    x = np.maximum(preds[m][i].astype(np.float64), 1e-3)
                    xf = x.ravel().copy()
                    xf[p] = np.maximum(xf[p], t_lb)
                    entry["methods"][m] = C.region_sums(x, g, masks)
                    entry["methods"][m + "_FS"] = C.region_sums(xf.reshape(x.shape), g, masks)
                    xv = x.ravel()[p][valid]
                    desc[fold][f"{m}_below_bound"] += float((xv < t_lb[valid]).sum())
                    desc[fold][f"{m}_in_carved"] += float(in_carved(carved[valid], xv).sum())
                desc[fold]["uncovered_valid"] += float(valid.sum())
                desc[fold]["constrained"] += float((t_lb[valid] > 0).sum())
                desc[fold]["gt_below_bound"] += float((gv[valid] < t_lb[valid]).sum())
                desc[fold]["gt_in_carved"] += float(in_carved(carved[valid], gv[valid]).sum())
                desc[fold]["bound_over_gt_sum"] += float((t_lb[valid] / gv[valid]).sum())
                entries.append(entry)
        print(f"{fold}: {len(rows)} queries ({time.time() - started:.0f}s)", flush=True)
    purposes = {a["purpose"] for a in access}
    if purposes - {"SEALED_QUERY_EVALUATION", "QUERY_CAMERA_TEACHER_ONLY", "ORACLE_TEACHER_ONLY"}:
        raise PermissionError(f"unexpected read purposes {sorted(purposes)}")
    (out_dir / "queries.json").write_text(json.dumps(entries) + "\n")
    names = tuple(n for m in METHODS for n in (m, m + "_FS"))
    table = RS.place_table(entries, names)
    means = {m: float(np.mean(list(table[m].values()))) for m in names}
    contrasts = {f"{m}_minus_{m}_FS": {**RS.short(RS.contrast(table, m, m + "_FS")), **RS.summary(table, m, m + "_FS")}
                 for m in METHODS}
    p1 = contrasts["B7_minus_B7_FS"]
    label = "SUPPORTED" if p1["mean"] >= 0.005 and p1["ci95"][0] > 0 else "NOT_SUPPORTED"
    pooled = defaultdict(float)
    for f in desc.values():
        for k, v in f.items():
            pooled[k] += v
    describe = {}
    for name, s in list(desc.items()) + [("POOLED", pooled)]:
        n = max(s["uncovered_valid"], 1.0)
        describe[name] = {"uncovered_valid_pixels": int(s["uncovered_valid"]), "constrained_share": s["constrained"] / n,
                          "mean_bound_over_gt": s["bound_over_gt_sum"] / n,
                          "gt_below_bound_share": s["gt_below_bound"] / n, "gt_in_carved_share": s["gt_in_carved"] / n,
                          **{f"{m}_below_bound_share": s[f"{m}_below_bound"] / n for m in METHODS},
                          **{f"{m}_in_carved_share": s[f"{m}_in_carved"] / n for m in METHODS}}
    regional = {m: {st: RS.pixel_absrel(entries, m, st)[0] for st in ("ALL", "UNC_NEAR", "UNC_FILL", "FILL_D0_8",
                                                                     "FILL_D8_32", "FILL_D32", "ZERO_HIT")}
                for m in names}
    res = {"places": len(table["B7"]), "queries": len(entries), "means": means, "contrasts": contrasts,
           "primary": {"contrast": p1, "label": "A_PREMISE_FREE_SPACE=" + label}, "describe": describe,
           "pooled_pixel_absrel": regional, "read_purposes": sorted(purposes), "reads": len(access),
           "carving": {"rel": REL, "abs_m": ABS, "samples": len(T), "range_m": [0.1, 20.0]}}
    dest = out_dir if smoke else ROOT
    (dest / "AMENDMENT_3_RESULTS.json").write_text(json.dumps(res, indent=1, sort_keys=True) + "\n")
    print("AMEND3", res["primary"]["label"], json.dumps(p1), flush=True)
    print(json.dumps(describe["POOLED"]), flush=True)


if __name__ == "__main__":
    {"lock": lock, "run": lambda: run(False), "smoke": lambda: run(True)}[STAGE]()
