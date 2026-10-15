"""EXP-3D-CAJU3D-JEPAGATE-V1: scoring, the DEV go/stop and the preregistered confirmation (CPU).

Reuses ENGINE-V1's checked scorer (engine_score.score: reused seals and every file checked against their hashes,
ground truth through TeacherAccess after sealing, Core B readout, place = mean over roles of the mean over queries)
and V1's paired place bootstrap + V2 gate (score_analyze.contrast).
  dev      DEV_SYSTEM + DEV_SYSTEM2: CORE_B (sealed released-weights TTT_m1) and the gate arms JEPA, DEPTH, INC, RANDOM.
           GO iff GATE_JEPA has a lower DEV mean than both CORE_B and GATE_DEPTH (decide); decision.json and its hash
           are written before any confirmation depth is read.
  confirm  CONF_SYSTEM3 (P1) and CONF_VAL (P2): GATE_DEPTH - GATE_JEPA (does the JEPA innovation pick better pixels than
           the cross-view depth disagreement); secondaries CORE_B - GATE_JEPA, CORE_B - GATE_DEPTH.
Run through cv_host_run.py from the repository root (CUDA hidden):  python gate_score.py ROOT dev|confirm
"""

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUTPUTS = HERE.parents[1]
for _path in (OUTPUTS / "EXP-3D-RGBD-DEPTHSPLAT-JEPA-TTT-V1" / "code", OUTPUTS / "EXP-3D-CAJU-HEADROOM-V1" / "code",
              OUTPUTS / "EXP-3D-CAJU3D-ENGINE-V1" / "code", HERE):
    sys.path.insert(0, str(_path))
import engine_score as S  # noqa: E402
from score_analyze import contrast  # noqa: E402

DEV_ARMS = ("JEPA", "DEPTH", "INC", "RANDOM")
CONF_ARMS = ("JEPA", "DEPTH")
DEV = ("DEV_SYSTEM", "DEV_SYSTEM2")
CONF = ("CONF_SYSTEM3", "CONF_VAL")
SHARE, STEPS, SCALE_MAX = 0.10, 100, 0.1


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def decide(means):
    """GO iff the JEPA gate beats both Core B and the depth-consistency gate on the DEV means."""
    return means["GATE_JEPA"] < means["CORE_B"] and means["GATE_JEPA"] < means["GATE_DEPTH"]


def branch(p1, p2):
    if p1["status"] == "SUPPORTED" and p2["mean"] > 0:
        return "JEPA_ADDED_VALUE_SUPPORTED"
    if p1["status"] in ("SUPPORTED", "PARTIAL"):
        return "JEPA_ADDED_VALUE_PARTIAL"
    if p1["status"] == "HARMFUL":
        return "JEPA_ADDED_VALUE_HARMFUL"
    return "JEPA_ADDED_VALUE_NOT_ESTABLISHED"


def gate_predictions(directory, arms):
    parts = sorted(Path(directory).glob("gate_seal_*of*.json"))
    if not parts or len(parts) != int(parts[0].stem.split("of")[1]):
        raise PermissionError(f"{directory}: incomplete set of gate seals")
    predictions, logs = {}, {}
    for part in parts:
        seal = json.loads(part.read_text())
        if (list(seal["arms"]) != list(arms) or seal["limit"] or seal["share"] != SHARE or seal["ttt_steps"] != STEPS
                or seal["scale_max"] != SCALE_MAX or seal["query_depth_read"] or seal["access_violations"]):
            raise PermissionError(f"{part}: not the protocol's run")
        predictions.update(seal["predictions"])
        logs.update(seal["episodes_log"])
    return predictions, logs


def sources(root, cohort, new, arms):
    reused = json.loads((root / "reused_seals.json").read_text())
    zs = S.reused_predictions(reused[cohort]["ZS"])
    src = {"CORE_B": (zs, "TTT_m1"), "ZS_CACHE": (zs, "CACHE")}
    src.update({f"GATE_{a}": (new, a) for a in arms})
    return src


def overlap_summary(logs):
    keys = sorted({k for v in logs.values() for k in v["arms"]["dropped_overlap"]})
    return {k: float(np.mean([v["arms"]["dropped_overlap"][k] for v in logs.values()])) for k in keys}


def stage_dev(root):
    if (root / "decision.json").exists():
        raise SystemExit("the DEV decision exists already")
    new, logs = gate_predictions(root / "gate/DEV", DEV_ARMS)
    rows, _, reads = S.score(root, list(DEV), {c: sources(root, c, new, DEV_ARMS) for c in DEV})
    table = S.place_table(rows)
    means = {k: S.mean(v) for k, v in table.items()}
    go = decide(means)
    stats = {f"CORE_B_minus_{a}": contrast(table, "CORE_B", f"GATE_{a}") for a in DEV_ARMS}
    stats.update({f"{a}_minus_JEPA": contrast(table, f"GATE_{a}", "GATE_JEPA") for a in ("DEPTH", "INC", "RANDOM")})
    S.write(root / "scores/DEV/rows.json", rows)
    S.write(root / "results/dev_results.json", {"places": len(table["CORE_B"]), "means": means, "contrasts": stats,
                                                "dropped_overlap": overlap_summary(logs), "ground_truth_reads": reads,
                                                "go": go})
    S.write(root / "decision.json", {"rule": "GO iff GATE_JEPA's DEV mean is below both CORE_B and GATE_DEPTH",
                                     "go": go, "dev_means": means})
    (root / "audit").mkdir(exist_ok=True)
    with (root / "audit/decision_log.txt").open("a") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S %z')} {sha(root / 'decision.json')}\n")
    print("DEV_DECISION", "GO" if go else "STOP", json.dumps({k: round(v, 5) for k, v in means.items()}))


def stage_confirm(root):
    decision = json.loads((root / "decision.json").read_text())
    if sha(root / "decision.json") != (root / "audit/decision_log.txt").read_text().split()[-1]:
        raise PermissionError("decision.json changed since its hash was logged")
    if not decision["go"]:
        raise SystemExit("DEV said STOP: no confirmation")
    new, logs = gate_predictions(root / "gate/CONF", CONF_ARMS)
    results = {}
    for cohort in CONF:
        rows, _, reads = S.score(root, [cohort], {cohort: sources(root, cohort, new, CONF_ARMS)})
        table = S.place_table(rows)
        results[cohort] = {"places": len(table["CORE_B"]), "means": {k: S.mean(v) for k, v in table.items()},
                           "primary_depth_minus_jepa": contrast(table, "GATE_DEPTH", "GATE_JEPA"),
                           "core_b_minus_jepa": contrast(table, "CORE_B", "GATE_JEPA"),
                           "core_b_minus_depth": contrast(table, "CORE_B", "GATE_DEPTH"), "ground_truth_reads": reads}
        S.write(root / f"scores/{cohort}/rows.json", rows)
    results["dropped_overlap"] = overlap_summary(logs)
    results["branch"] = branch(results["CONF_SYSTEM3"]["primary_depth_minus_jepa"],
                               results["CONF_VAL"]["primary_depth_minus_jepa"])
    S.write(root / "results/confirm_results.json", results)
    print("CONFIRM", results["branch"])


def main():
    root, stage = Path(sys.argv[1]).resolve(), sys.argv[2]
    {"dev": stage_dev, "confirm": stage_confirm}[stage](root)


if __name__ == "__main__":
    main()
