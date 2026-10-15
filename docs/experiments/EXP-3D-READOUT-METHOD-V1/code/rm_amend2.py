"""EXP-3D-READOUT-METHOD-V1, Amendment 2 (post hoc; written before Amendment 1's results were seen; AMENDMENT_2.md):
the premise test of Candidate B (a cross-view shared correction). An ORACLE upper bound, CPU only, on the saved Phase 1
predictions of V2's recipe (B7); no training.

For every held-out episode (two query views) and both directions src -> dst: the ground-truth surface points of src
(its valid pixels) are projected into dst (1-pixel z-buffer, nearest point kept). A dst pixel and its src pixel are a
correspondence when the dst pixel is uncovered (TTT alpha < 0.5, the protocol's region), its ground truth is valid and
the projected ground-truth distance agrees with it within 2 % (|log| < 0.02: the same visible surface). The ground
truth only defines correspondences (an oracle no deployable method has) and scores. On every correspondence:
  B7_F16     V2's dst prediction (float16, as saved)
  SHARED     the geometric mean of the dst prediction and the src prediction transported along the true
             correspondence (the distance from the dst camera of the point src predicts on its ray): one shared
             estimate of the surface point from both views
  BEST       the better of the two (a looser super-oracle, descriptive only)
Per dst query: AbsRel over all its valid pixels with its correspondences replaced; place = mean over episodes of the
mean over queries; V1's paired place bootstrap. Descriptive: correspondence coverage of the uncovered valid pixels,
the correlation of the two views' log errors on correspondences, and V2's cross-view disagreement.
Run from the repository root with the project environment (CUDA hidden, PYTHONPATH=src):
  python rm_amend2.py ROOT smoke   (code check on the smoke fold FIT_SYSTEM2, a training cohort -> smoke/amend2)
  python rm_amend2.py ROOT lock    (hashes AMENDMENT_2.md and this file; refuses when results exist)
  python rm_amend2.py ROOT run     (refuses without the lock or when a locked file changed)
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

AGREE = 0.02
LOCKED = ("AMENDMENT_2.md", "code/rm_amend2.py")
BINS = [0.0, 0.02, 0.05, 0.1, 0.2, 0.5, np.inf]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def records_for(cohort):
    locks = json.loads((C.OUT / C.LOCKS[cohort]).read_text())["cohorts"]
    lock_name = {"CAJU": None, "ARKIT_SYSTEM4": "ARKIT_SYSTEM4"}.get(cohort, cohort)
    records = {}
    for name, entry in locks.items():
        if (lock_name is None and not name.startswith("CAJU_")) or (lock_name is not None and name != lock_name):
            continue
        records.update({r["scene_id"]: r for r in json.loads(Path(entry["manifest"]).read_text())["scenes"]})
    return records


def views(cohort, rows, gt, preds, unc, access):
    """Per query: flat ground truth, prediction, uncovered mask, rays and camera (query cameras via TeacherAccess)."""
    records = records_for(cohort)
    episodes = defaultdict(list)
    for i, r in enumerate(rows):
        episodes[r["episode"]].append(i)
    out = [None] * len(rows)
    for members in episodes.values():
        rec = records[rows[members[0]]["scene_id"]]
        teacher = C.S.TeacherAccess(rec, [rows[i]["query"] for i in members], C.S.IMAGE_SIZE, C.S.REPO, access)
        for i in members:
            cam = teacher.query_camera(rows[i]["query"]).to(dtype=torch.float64)
            o, d = generate_rays(cam)
            out[i] = {"gt": gt[i].ravel(), "pred": np.maximum(preds[i].astype(np.float64), 1e-3).ravel(),
                      "unc": unc[i].ravel(),
                      "o": o.numpy().reshape(-1, 3), "d": d.numpy().reshape(-1, 3), "cam": cam,
                      "c": cam.c2w[:3, 3].numpy()}
    return out, episodes


def correspond(src, dst):
    """Oracle correspondences src -> dst on dst's uncovered valid pixels, and the per-pixel quantities."""
    sv = np.flatnonzero(src["gt"] > 0)
    x = src["o"][sv] + src["d"][sv] * src["gt"][sv, None]
    pix, _, inside = project_world(torch.as_tensor(x), dst["cam"])
    pix, inside = pix.numpy(), inside.numpy()
    u = np.clip(np.round(pix[:, 0]).astype(np.int64), 0, C.W - 1)
    v = np.clip(np.round(pix[:, 1]).astype(np.int64), 0, C.H - 1)
    flat, dist = (v * C.W + u)[inside], np.linalg.norm(x - dst["c"], axis=1)[inside]
    sv = sv[inside]
    order = np.lexsort((dist, flat))
    first = np.r_[True, flat[order][1:] != flat[order][:-1]] if len(order) else np.zeros(0, bool)
    win = order[first]
    ud, us, rgt = flat[win], sv[win], dist[win]
    gd = dst["gt"][ud]
    ok = (gd > 0) & dst["unc"][ud]
    ok[ok] = np.abs(np.log(rgt[ok]) - np.log(gd[ok])) < AGREE
    ud, us = ud[ok], us[ok]
    xp = src["o"][us] + src["d"][us] * src["pred"][us, None]
    r_hat = np.linalg.norm(xp - dst["c"], axis=1)
    pd, gd, ps, gs = dst["pred"][ud], dst["gt"][ud], src["pred"][us], src["gt"][us]
    a_b7, a_tr = np.abs(pd - gd) / gd, np.abs(r_hat - gd) / gd
    a_sh = np.abs(np.sqrt(pd * r_hat) - gd) / gd
    es, ed = np.log(ps) - np.log(gs), np.log(pd) - np.log(gd)
    dis = np.abs(np.log(r_hat) - np.log(pd))
    return {"n": int(len(ud)), "b7": float(a_b7.sum()), "shared": float(a_sh.sum()),
            "best": float(np.minimum(a_b7, a_tr).sum()), "transported": float(a_tr.sum()),
            "es": float(es.sum()), "ed": float(ed.sum()), "es2": float((es ** 2).sum()), "ed2": float((ed ** 2).sum()),
            "esed": float((es * ed).sum()), "dis_hist": np.histogram(dis, bins=BINS)[0].tolist(),
            "src_covered": int((~src["unc"][us]).sum())}


def correlation(s):
    n = s["n"]
    if n < 2:
        return None
    cov = s["esed"] / n - s["es"] / n * s["ed"] / n
    vs, vd = s["es2"] / n - (s["es"] / n) ** 2, s["ed2"] / n - (s["ed"] / n) ** 2
    return float(cov / np.sqrt(vs * vd)) if vs > 0 and vd > 0 else None


def lock():
    if (ROOT / "results/amend2").exists() or (ROOT / "AMENDMENT_2.lock.json").exists():
        raise SystemExit("cannot lock: amendment 2 results or a lock exist already")
    rec = {"amendment": 2, "post_hoc": True, "oracle_upper_bound": True,
           "locked_at": time.strftime("%Y-%m-%d %H:%M:%S %z"), "files": {n: sha(ROOT / n) for n in LOCKED},
           "primary": "B7_F16 - B7_SHARED, pooled 350 places (> 0: sharing the two views' estimates of the same "
                      "uncovered surface points would lower the error, with the true correspondences)",
           "decision": {"B_PREMISE_ACCURACY=SUPPORTED": "mean >= 0.005 and CI lower > 0",
                        "B_PREMISE_ACCURACY=NOT_SUPPORTED": "otherwise"},
           "agreement_log_threshold": AGREE}
    (ROOT / "AMENDMENT_2.lock.json").write_text(json.dumps(rec, indent=1) + "\n")
    (ROOT / "audit/amendment2_locked_at.txt").write_text(f"{rec['locked_at']} {sha(ROOT / 'AMENDMENT_2.lock.json')}\n")
    print("AMEND2_LOCKED", rec["locked_at"], sha(ROOT / "AMENDMENT_2.lock.json"))


def run(smoke):
    if not smoke:
        rec = json.loads((ROOT / "AMENDMENT_2.lock.json").read_text())
        for name, h in rec["files"].items():
            if sha(ROOT / name) != h:
                raise PermissionError(f"{name} changed since the amendment 2 lock")
    torch.set_num_threads(1)
    out_dir = ROOT / ("smoke/amend2" if smoke else "results/amend2")
    out_dir.mkdir(parents=True, exist_ok=smoke)
    access, started, entries, stats = [], time.time(), [], {}
    for fold in RS.FOLDS:
        arrays, rows = C.load_cache(fold)
        fj = json.loads((RS.RES / f"fold_{fold}.json").read_text())["queries"]
        if [e["key"] for e in fj] != [r["key"] for r in rows]:
            raise ValueError(f"{fold}: prediction order differs from the cache")
        preds = np.load(RS.RES / f"preds_{fold}.npz")["B7"]
        gt = C.load_gt(fold, rows, access)
        unc = arrays["ttt_a"] < C.ALPHA_MIN
        vw, episodes = views(fold, rows, gt, preds, unc, access)
        fold_sum = defaultdict(float)
        hist = np.zeros(len(BINS) - 1, np.int64)
        for members in episodes.values():
            if len(members) != 2:
                raise ValueError("an episode without exactly two queries")
            for dst, src in ((members[0], members[1]), (members[1], members[0])):
                s = correspond(vw[src], vw[dst])
                d = vw[dst]
                valid = d["gt"] > 0
                total = float((np.abs(d["pred"][valid] - d["gt"][valid]) / d["gt"][valid]).sum())
                n = int(valid.sum())
                r = rows[dst]
                entries.append({"key": r["key"], "place": r["place"], "episode": r["episode"], "fold": fold,
                                "hit7": r["hit7"], "unc_valid": int((valid & d["unc"]).sum()), "pairs": s["n"],
                                "methods": {"B7_F16": {"ALL": [total, n]},
                                            "B7_SHARED": {"ALL": [total - s["b7"] + s["shared"], n]},
                                            "B7_BEST": {"ALL": [total - s["b7"] + s["best"], n]}}})
                for k in ("n", "b7", "shared", "best", "transported", "es", "ed", "es2", "ed2", "esed", "src_covered"):
                    fold_sum[k] += s[k]
                fold_sum["unc_valid"] += entries[-1]["unc_valid"]
                hist += np.array(s["dis_hist"])
        fold_sum["dis_hist"] = hist.tolist()
        stats[fold] = dict(fold_sum)
        print(f"{fold}: {len(rows)} queries ({time.time() - started:.0f}s)", flush=True)
    purposes = {a["purpose"] for a in access}
    if purposes - {"SEALED_QUERY_EVALUATION", "QUERY_CAMERA_TEACHER_ONLY"}:
        raise PermissionError(f"unexpected read purposes {sorted(purposes)}")
    (out_dir / "queries.json").write_text(json.dumps(entries) + "\n")
    methods = ("B7_F16", "B7_SHARED", "B7_BEST")
    table = RS.place_table(entries, methods)
    means = {m: float(np.mean(list(table[m].values()))) for m in methods}
    contrasts = {f"B7_F16_minus_{m}": {**RS.short(RS.contrast(table, "B7_F16", m)), **RS.summary(table, "B7_F16", m)}
                 for m in methods[1:]}
    p = contrasts["B7_F16_minus_B7_SHARED"]
    label = "SUPPORTED" if p["mean"] >= 0.005 and p["ci95"][0] > 0 else "NOT_SUPPORTED"
    pooled = defaultdict(float)
    for s in stats.values():
        for k, v in s.items():
            if k != "dis_hist":
                pooled[k] += v
    pooled["dis_hist"] = np.sum([s["dis_hist"] for s in stats.values()], axis=0).tolist()
    describe = {}
    for name, s in list(stats.items()) + [("POOLED", dict(pooled))]:
        n = max(s["n"], 1)
        describe[name] = {"correspondences": int(s["n"]), "uncovered_valid_pixels": int(s["unc_valid"]),
                          "coverage_of_uncovered": s["n"] / max(s["unc_valid"], 1),
                          "src_pixel_covered_share": s["src_covered"] / n,
                          "pixel_absrel_on_correspondences": {"B7_F16": s["b7"] / n, "SHARED": s["shared"] / n,
                                                              "TRANSPORTED_ONLY": s["transported"] / n,
                                                              "BEST": s["best"] / n},
                          "log_error_correlation": correlation(s),
                          "disagreement_hist": {"bins": [str(b) for b in BINS], "counts": s["dis_hist"]}}
    saved = [e for f in RS.FOLDS for e in json.loads((RS.RES / f"fold_{f}.json").read_text())["queries"]]
    fold_json_b7 = float(np.mean(list(RS.place_table(saved, ("B7",))["B7"].values())))  # float64 scoring
    res = {"places": len(table["B7_F16"]), "queries": len(entries), "means": means, "contrasts": contrasts,
           "primary": {"contrast": p, "label": "B_PREMISE_ACCURACY=" + label},
           "fold_means": {f: {m: float(np.mean(list(RS.place_table([e for e in entries if e["fold"] == f],
                                                                      methods)[m].values()))) for m in methods}
                          for f in RS.FOLDS},
           "describe": describe, "read_purposes": sorted(purposes), "ground_truth_reads": len(access),
           "agreement_log_threshold": AGREE, "b7_pooled_mean_float64_scoring": fold_json_b7}
    dest = out_dir if smoke else ROOT
    (dest / "AMENDMENT_2_RESULTS.json").write_text(json.dumps(res, indent=1, sort_keys=True) + "\n")
    print("AMEND2", res["primary"]["label"], json.dumps(p), flush=True)
    print(json.dumps(describe["POOLED"]), flush=True)


if __name__ == "__main__":
    {"lock": lock, "run": lambda: run(False), "smoke": lambda: run(True)}[STAGE]()
