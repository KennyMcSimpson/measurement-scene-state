"""EXP-3D-CAJU3D-KFUSE-REPL-V1: the frozen KFUSE-V1 fusion tables on the 58 CAJU-V1 places (CPU, no GPU, no fitting).

TTT = CAJU-V1's FULL replay (released weights, 7-view CACHE, 100 raw TTT steps = Core B's TTT); geometry = CAJU-V1's
geometry seals (REPROJ7_HARMONIC, near7, cosine); ground truth through TeacherAccess. Arms CORE_B, GEO_U, KF_DIS,
KF_BASE with KFUSE-V1's fusion_tables.json unchanged (the H-JEPA arms are not scored: these places trained it).
P = CORE_B - KF_BASE over the 58 places (V1 paired place bootstrap + V2 gate).
Run from the repository root (CUDA hidden, PYTHONPATH=src):  python repl_score.py ROOT
"""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUTPUTS = HERE.parents[1]
KF = OUTPUTS / "EXP-3D-CAJU3D-KFUSE-V1"
CAJU = OUTPUTS / "EXP-3D-CAJU-HEADROOM-V1"
sys.path.insert(0, str(KF / "code"))
import kf_data as D  # noqa: E402  (sets the import paths)
import kf_logic as K  # noqa: E402
from caju_logic import core_b  # noqa: E402
from score_analyze import contrast  # noqa: E402

S = D.S
COHORTS = ("CAJU_PILOT", "CAJU_SEL", "CAJU_EVAL")
ARMS = ("CORE_B", "GEO_U", "KF_DIS", "KF_BASE")
TABLES_SHA = "15100550ca6b39763244a69405ec88f8eede4faaa700aff4619557ee1c7e533c"
VERDICT = {"SUPPORTED": "REPLICATED", "PARTIAL": "PARTIALLY_REPLICATED", "HARMFUL": "REVERSED"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def replay_queries():
    out = {}
    for part in sorted((CAJU / "replay").glob("*/replay_seal_*of*.json")):
        seal = json.loads(part.read_text())
        if seal["ttt_steps"] != 100 or not seal["checkpoint_sha256"].startswith("f8ddd845"):
            raise PermissionError(f"{part}: not the released-weights 100-step replay")
        out.update({k: v for k, v in seal["predictions"].items() if v["kind"] == "query"})
    return out


def main():
    root = Path(sys.argv[1]).resolve()
    if not (root / "audit/locked_at.txt").is_file():
        raise SystemExit("lock first")
    if sha(KF / "fusion_tables.json") != TABLES_SHA:
        raise PermissionError("the KFUSE-V1 fusion tables changed")
    tables = json.loads((KF / "fusion_tables.json").read_text())
    locks = json.loads((CAJU / "frame_role_lock.json").read_text())["cohorts"]
    replay = replay_queries()
    access, rows = [], []
    for cohort in COHORTS:
        lock = locks[cohort]
        if sha(lock["manifest"]) != lock["manifest_sha256"]:
            raise PermissionError(f"{cohort}: manifest changed")
        geo_seal = CAJU / "geometry" / f"geometry_seal_{cohort}.json"
        geometry = json.loads(geo_seal.read_text())["predictions"]
        records = {r["scene_id"]: r for r in json.loads(Path(lock["manifest"]).read_text())["scenes"]}
        for e in lock["episodes"]:
            teacher = S.TeacherAccess(records[e["scene_id"]], e["queries"], S.IMAGE_SIZE, S.REPO, access)
            for qid in e["queries"]:
                key = f"{e['scene_id']}/{e['role']}/{qid}"
                if replay[key]["cohort"] != cohort:
                    raise PermissionError(f"{key}: replay cohort mismatch")
                ttt, geo = S.load_checked(replay[key]), S.load_checked(geometry[key])
                gt = teacher.query_depth(qid)
                valid = np.isfinite(gt) & (gt > 0)
                _, f7 = core_b(ttt["FULL_depth_z"], ttt["FULL_alpha"], geo["REPROJ7_HARMONIC"], geo["cosine"])
                q = {"gt": gt[valid], "r7": np.maximum(geo["REPROJ7_HARMONIC"][valid].astype(np.float64), 1e-6),
                     "f7": np.maximum(f7[valid], 1e-6), "alpha": ttt["FULL_alpha"][valid].astype(np.float64),
                     "near": geo["near7"][valid].astype(bool), "log_p": np.zeros(int(valid.sum()))}
                rows.append({"cohort": cohort, "place": e["place"], "episode": f"{e['scene_id']}/{e['role']}",
                             "query": key, "core_b": {a: float(np.mean(np.abs(K.readout(q, a, tables) - q["gt"])
                                                                         / q["gt"])) for a in ARMS}})
    if {a["purpose"] for a in access} != {"SEALED_QUERY_EVALUATION"}:
        raise PermissionError("unexpected read purposes")
    table = S.place_table(rows)
    primary = contrast(table, "CORE_B", "KF_BASE")
    result = {"places": len(table["CORE_B"]), "queries": len(rows), "ground_truth_reads": len(access),
              "means": {a: S.mean(table[a]) for a in ARMS},
              "primary_core_b_minus_kf_base": primary,
              "secondary": {"CORE_B_minus_GEO_U": contrast(table, "CORE_B", "GEO_U"),
                            "GEO_U_minus_KF_BASE": contrast(table, "GEO_U", "KF_BASE")},
              "by_cohort": {c: {a: S.mean(S.place_table([r for r in rows if r["cohort"] == c])[a]) for a in ARMS}
                            for c in COHORTS},
              "verdict": VERDICT.get(primary["status"], "NOT_REPLICATED"), "fusion_tables_sha256": TABLES_SHA}
    (root / "results").mkdir(exist_ok=True)
    (root / "results/repl_results.json").write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
    (root / "scores").mkdir(exist_ok=True)
    (root / "scores/rows.json").write_text(json.dumps(rows, indent=1, sort_keys=True) + "\n")
    print("REPL", result["verdict"], json.dumps({a: round(v, 5) for a, v in result["means"].items()}))


if __name__ == "__main__":
    main()
