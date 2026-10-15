"""EXP-3D-CAJU3D-SYSTEM4-HOLDOUT-V1: the preregistered holdout test (CPU). Reads the readout seal, then the SYSTEM4
truth through TeacherAccess. P1 = CORE_B - FUSE_V2, P2 = CORE_B - FUSE_V1 over the SYSTEM4 places (place =
physical_scene_id; mean over episodes of the mean over queries), V1's paired place bootstrap + V2 gate; secondaries
FUSE_V1 - FUSE_V2, CORE_B - KF_BASE, KF_BASE - FUSE_V2; covered / uncovered pixel AbsRel, worse places, gain quartiles.
Run from the repository root (CUDA hidden, PYTHONPATH=src):  python s4_score.py ROOT
"""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUTPUTS = HERE.parents[1]
for _path in ("EXP-3D-RGBD-DEPTHSPLAT-JEPA-TTT-V1/code", "EXP-3D-CAJU-HEADROOM-V1/code", "EXP-3D-CAJU3D-ENGINE-V1/code"):
    sys.path.insert(0, str(OUTPUTS / _path))
import engine_score as S  # noqa: E402
from score_analyze import contrast  # noqa: E402

COHORT = "ARKIT_SYSTEM4"
NAMES = ("CORE_B", "KF_BASE", "FUSE_V1", "FUSE_V2")
VERDICT = {"SUPPORTED": "SUPPORTED", "PARTIAL": "PARTIAL", "HARMFUL": "HARMFUL"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def conclusion(p1, p2):
    statuses = (p1["status"], p2["status"])
    if all(s == "SUPPORTED" for s in statuses):
        return "LEARNED_READOUT_CONFIRMED"
    if "HARMFUL" in statuses:
        return "LEARNED_READOUT_HARMFUL"
    if any(s in ("SUPPORTED", "PARTIAL") for s in statuses):
        return "LEARNED_READOUT_PARTIAL"
    return "LEARNED_READOUT_NOT_ESTABLISHED"


def main():
    root = Path(sys.argv[1]).resolve()
    seal = json.loads((root / "readouts/readout_seal.json").read_text())
    if seal["query_depth_read"] is not False:
        raise PermissionError("readouts were not sealed blind")
    lock = json.loads((root / "frame_role_lock.json").read_text())["cohorts"][COHORT]
    if sha(lock["manifest"]) != lock["manifest_sha256"]:
        raise PermissionError("manifest changed")
    records = {r["scene_id"]: r for r in json.loads(Path(lock["manifest"]).read_text())["scenes"]}
    keys = [f"{e['scene_id']}/{e['role']}/{q}" for e in lock["episodes"] for q in e["queries"]]
    if sorted(keys) != sorted(seal["predictions"]):
        raise PermissionError("the readout seal does not cover the cohort")
    access, rows = [], []
    pixels = {f"{t}_{n}": [0.0, 0] for t in ("cov", "unc") for n in NAMES}
    for e in lock["episodes"]:
        teacher = S.TeacherAccess(records[e["scene_id"]], e["queries"], S.IMAGE_SIZE, S.REPO, access)
        for qid in e["queries"]:
            key = f"{e['scene_id']}/{e['role']}/{qid}"
            pred = S.load_checked(seal["predictions"][key])
            gt = teacher.query_depth(qid)
            valid = np.isfinite(gt) & (gt > 0)
            cov = pred["covered"][valid]
            row = {"place": records[e["scene_id"]]["physical_scene_id"], "episode": f"{e['scene_id']}/{e['role']}",
                   "query": key, "core_b": {}}
            for n in NAMES:
                err = np.abs(pred[n][valid] - gt[valid]) / gt[valid]
                row["core_b"][n] = float(err.mean())
                for tag, mask in (("cov", cov), ("unc", ~cov)):
                    pixels[f"{tag}_{n}"][0] += float(err[mask].sum())
                    pixels[f"{tag}_{n}"][1] += int(mask.sum())
            rows.append(row)
    if {a["purpose"] for a in access} != {"SEALED_QUERY_EVALUATION"}:
        raise PermissionError("unexpected read purposes")
    table = S.place_table(rows)
    places = sorted(table["CORE_B"])
    p1, p2 = contrast(table, "CORE_B", "FUSE_V2"), contrast(table, "CORE_B", "FUSE_V1")
    gains = {n: np.array([table["CORE_B"][p] - table[n][p] for p in places]) for n in ("FUSE_V1", "FUSE_V2", "KF_BASE")}
    result = {"cohort": COHORT, "places": len(places), "queries": len(rows), "ground_truth_reads": len(access),
              "means": {n: S.mean(table[n]) for n in NAMES},
              "P1_core_b_minus_fuse_v2": p1, "P2_core_b_minus_fuse_v1": p2,
              "secondary": {"FUSE_V1_minus_FUSE_V2": contrast(table, "FUSE_V1", "FUSE_V2"),
                            "CORE_B_minus_KF_BASE": contrast(table, "CORE_B", "KF_BASE"),
                            "KF_BASE_minus_FUSE_V2": contrast(table, "KF_BASE", "FUSE_V2")},
              "pixel_absrel": {k: v[0] / max(v[1], 1) for k, v in pixels.items()},
              "worse_places": {n: int((g < 0).sum()) for n, g in gains.items()},
              "gain_quartiles": {n: np.quantile(g, [0, 0.25, 0.5, 0.75, 1]).tolist() for n, g in gains.items()},
              "conclusion": conclusion(p1, p2)}
    (root / "results").mkdir(exist_ok=True)
    (root / "results/holdout_results.json").write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
    (root / "scores").mkdir(exist_ok=True)
    (root / "scores/rows.json").write_text(json.dumps(rows, indent=1, sort_keys=True) + "\n")
    print("S4_HOLDOUT", result["conclusion"], json.dumps({n: round(v, 5) for n, v in result["means"].items()}))


if __name__ == "__main__":
    main()
