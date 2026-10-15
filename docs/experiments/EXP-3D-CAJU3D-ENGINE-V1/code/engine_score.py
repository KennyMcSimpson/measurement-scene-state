"""EXP-3D-CAJU3D-ENGINE-V1: scoring, the DEV selection and the preregistered confirmation (CPU).

Every stage first checks every prediction it reads (the reused seals against reused_seals.json, every file against its
sealed hash, this experiment's engine seals for settings, completeness and the access audit) and that the new CACHE
renders equal the sealed CACHE renders bit for bit; only then is query depth read (TeacherAccess.query_depth, purpose
logged). Readout of every arm: CAJU Gate 0's caju_logic.core_b, i.e. CORE_B = sqrt(R7 x F7), F7 = ray depth where
alpha >= 0.5 else R7; AbsRel on valid pixels; place value = mean over roles of the mean over queries.
  dev      DEV_SYSTEM + DEV_SYSTEM2: every arm; engine_logic.select_candidate picks C* (or STOP); selection.json and its
           hash are written before any confirmation depth is read.
  confirm  CONF_SYSTEM3 (P1) and CONF_VAL (P2): CORE_B (sealed released-weights TTT_m1) - C*, V1's paired place
           bootstrap and V2 gate (score_analyze.contrast); secondaries C* - its own CACHE, coverage.
Run through cv_host_run.py from the repository root (CUDA hidden):  python engine_score.py ROOT dev|confirm
"""

import hashlib
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUTPUTS = HERE.parents[1]
for _path in (OUTPUTS / "EXP-3D-RGBD-DEPTHSPLAT-JEPA-TTT-V1" / "code", OUTPUTS / "EXP-3D-CAJU-HEADROOM-V1" / "code",
              HERE):
    sys.path.insert(0, str(_path))
import engine_logic as L  # noqa: E402
from caju_logic import absrel, core_b  # noqa: E402
from mcss.measurement_planning.information import TeacherAccess  # noqa: E402
from score_analyze import contrast  # noqa: E402

REPO = Path.cwd()
IMAGE_SIZE = (128, 160)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def clean(value):
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.generic):
        return clean(value.item())
    return value


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(value), indent=1, sort_keys=True, allow_nan=False) + "\n")


def load_checked(item):
    if sha(item["path"]) != item["sha256"]:
        raise PermissionError(f"a sealed prediction changed: {item['path']}")
    return dict(np.load(item["path"]))


def reused_predictions(entry):
    """Merged predictions of a reused seal directory, checked against reused_seals.json."""
    out = {}
    for name, digest in entry["parts"].items():
        path = Path(entry["dir"]) / name
        if sha(path) != digest:
            raise PermissionError(f"reused seal changed: {path}")
        seal = json.loads(path.read_text())
        if seal["query_depth_read"] is not False or seal["scale_max"] != L.SCALE_MAX or seal["limit"]:
            raise PermissionError(f"{path}: not a certified cap-0.1 run")
        out.update(seal["predictions"])
    return out


def engine_predictions(directory, model, arms):
    parts = sorted(Path(directory).glob("engine_seal_*of*.json"))
    if not parts or len(parts) != int(parts[0].stem.split("of")[1]):
        raise PermissionError(f"{directory}: incomplete set of engine seals")
    predictions, logs = {}, {}
    for part in parts:
        seal = json.loads(part.read_text())
        if (seal["model"] != model or list(seal["arms"]) != list(arms) or seal["limit"] or seal["ttt_steps"] != L.STEPS
                or seal["scale_max"] != L.SCALE_MAX or seal["query_depth_read"] or seal["access_violations"]):
            raise PermissionError(f"{part}: not the protocol's run")
        predictions.update(seal["predictions"])
        logs.update(seal["episodes_log"])
    return predictions, logs


def score(root, cohorts, sources):
    """sources: {arm name: (predictions dict, array prefix)}. Returns rows and checks the CACHE identities."""
    locks = json.loads((root / "frame_role_lock.json").read_text())["cohorts"]
    reused = json.loads((root / "reused_seals.json").read_text())
    rows, access, identity = [], [], defaultdict(lambda: {"queries": 0, "max_abs_diff": 0.0})
    for cohort in cohorts:
        lock = locks[cohort]
        if sha(lock["manifest"]) != lock["manifest_sha256"]:
            raise PermissionError(f"{cohort}: manifest changed")
        geo_entry = reused[cohort]["geometry"]
        if sha(geo_entry["path"]) != geo_entry["sha256"]:
            raise PermissionError(f"{cohort}: geometry seal changed")
        geometry = json.loads(Path(geo_entry["path"]).read_text())["predictions"]
        records = {r["scene_id"]: r for r in json.loads(Path(lock["manifest"]).read_text())["scenes"]}
        src = sources[cohort]
        keys = [f"{e['scene_id']}/{e['role']}/{q}" for e in lock["episodes"] for q in e["queries"]]
        for name, (preds, _) in src.items():
            missing = [k for k in keys if k not in preds]
            if missing:
                raise PermissionError(f"{cohort}/{name}: {len(missing)} queries not sealed")
        for e in lock["episodes"]:
            teacher = TeacherAccess(records[e["scene_id"]], e["queries"], IMAGE_SIZE, REPO, access)
            for qid in e["queries"]:
                key = f"{e['scene_id']}/{e['role']}/{qid}"
                geo = load_checked(geometry[key])
                arrays = {name: (load_checked(preds[key]), prefix) for name, (preds, prefix) in src.items()}
                for new, old in (("ZS_CACHE_NEW", "ZS_CACHE"), ("JN1_CACHE_NEW", "JN1_CACHE")):
                    if new in arrays and old in arrays:
                        (a, pa), (b, pb) = arrays[new], arrays[old]
                        d = max(float(np.max(np.abs(a[f"{pa}_{f}"].astype(np.float64) - b[f"{pb}_{f}"])))
                                for f in ("depth_z", "alpha"))
                        identity[new]["queries"] += 1
                        identity[new]["max_abs_diff"] = max(identity[new]["max_abs_diff"], d)
                gt = teacher.query_depth(qid)
                valid = np.isfinite(gt) & (gt > 0)
                row = {"cohort": cohort, "episode": f"{e['scene_id']}/{e['role']}", "place": e["place"],
                       "query": qid, "core_b": {}, "f7": {}, "coverage": {}}
                for name, (data, prefix) in arrays.items():
                    cb, f7 = core_b(data[f"{prefix}_depth_z"], data[f"{prefix}_alpha"], geo["REPROJ7_HARMONIC"],
                                    geo["cosine"])
                    row["core_b"][name] = absrel(cb, gt)
                    row["f7"][name] = absrel(f7, gt)
                    row["coverage"][name] = float((data[f"{prefix}_alpha"] >= L.ALPHA_MIN)[valid].mean())
                row["core_b"]["R7"] = absrel(geo["REPROJ7_HARMONIC"], gt)
                rows.append(row)
    if {a["purpose"] for a in access} != {"SEALED_QUERY_EVALUATION"}:
        raise PermissionError("unexpected read purposes")
    for new, v in identity.items():
        if v["max_abs_diff"] != 0.0:
            raise PermissionError(f"{new}: the recomputed CACHE differs from the sealed one ({v['max_abs_diff']})")
    return rows, {k: dict(v) for k, v in identity.items()}, len(access)


def place_table(rows, metric="core_b"):
    grouped = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for row in rows:
        for name, v in row[metric].items():
            if v is not None and np.isfinite(v):
                grouped[name][row["place"]][row["episode"]].append(v)
    return {n: {p: float(np.mean([np.mean(q) for q in eps.values()])) for p, eps in pl.items()}
            for n, pl in grouped.items()}


def mean(table):
    return float(np.mean(list(table.values())))


def dev_sources(root):
    reused = json.loads((root / "reused_seals.json").read_text())
    zs_new, zs_logs = engine_predictions(root / "engine/DEV/ZS", "ZS", L.DEV_ARMS["ZS"])
    jn_new, jn_logs = engine_predictions(root / "engine/DEV/JN1", "JN1", L.DEV_ARMS["JN1"])
    sources = {}
    for cohort in L.DEV_COHORTS:
        zs_old = reused_predictions(reused[cohort]["ZS"])
        jn_old = reused_predictions(reused[cohort]["JN1_CACHE"])
        src = {"ZS_CACHE": (zs_old, "CACHE"), "ZS_RAW": (zs_old, "TTT_m1"), "JN1_CACHE": (jn_old, "CACHE"),
               "ZS_CACHE_NEW": (zs_new, "CACHE"), "JN1_CACHE_NEW": (jn_new, "CACHE")}
        src.update({f"ZS_{a}": (zs_new, a) for a in L.DEV_ARMS["ZS"] if a != "CACHE"})
        src.update({f"JN1_{a}": (jn_new, a) for a in L.DEV_ARMS["JN1"] if a != "CACHE"})
        sources[cohort] = src
    return sources, {"ZS": zs_logs, "JN1": jn_logs}


def mechanism(logs):
    """Coverage and opacity of the observed views and the query coverage, per arm (mean over DEV episodes)."""
    out = {}
    for model, by_episode in logs.items():
        for arm in L.DEV_ARMS[model]:
            vals = [v["arms"][arm] for v in by_episode.values()]
            out[f"{model}_{arm}"] = {
                "observed_coverage": float(np.mean([x["observed"]["coverage"] for x in vals])),
                "observed_normalized_error": float(np.mean([x["observed"]["normalized_error_covered"] for x in vals
                                                            if x["observed"]["normalized_error_covered"] is not None])),
                "observed_raw_depth_term": float(np.mean([x["observed"]["raw_depth_term"] for x in vals])),
                "opacity_mean": float(np.mean([x["observed"]["opacity_mean"] for x in vals])),
                "opacity_share_ge_0_5": float(np.mean([x["observed"]["opacity_share_ge_0_5"] for x in vals])),
                "query_coverage": float(np.mean([x["query_coverage"] for x in vals])),
                "seconds_median": float(np.median([x["seconds"] for x in vals])),
                "failures": sum(1 for x in vals if x.get("failed_at_step") or x.get("failed_reverted_to_cache")),
                "projected_steps_mean": float(np.mean([x.get("projected_steps", 0) for x in vals]))}
    return out


def stage_dev(root):
    if (root / "selection.json").exists():
        raise SystemExit("the DEV selection exists already")
    sources, logs = dev_sources(root)
    rows, identity, reads = score(root, list(L.DEV_COHORTS), sources)
    table, table_f7 = place_table(rows), place_table(rows, "f7")
    means = {k: mean(v) for k, v in table.items()}
    best, stop = L.select_candidate(means)
    stats = {k: contrast(table, L.COMPARATOR, k) for k in L.CANDIDATES + ("ZS_CACHE", "JN1_CACHE")}
    stats["JN1_NORM_vs_JN1_RAW"] = contrast(table, "JN1_RAW", "JN1_NORM")
    stats["ZS_NORM_vs_ZS_RAW"] = contrast(table, "ZS_RAW", "ZS_NORM")
    write(root / "scores/DEV/rows.json", rows)
    write(root / "results/dev_results.json", {
        "places": len(table[L.COMPARATOR]), "means_core_b": means, "means_f7": {k: mean(v) for k, v in table_f7.items()},
        "contrasts_vs_core_b_and_pairs": stats, "mechanism": mechanism(logs), "cache_identity": identity,
        "ground_truth_reads": reads, "selected": best, "stop": stop})
    write(root / "selection.json", {"rule": "lowest DEV mean place AbsRel (Core B readout) among engine_logic.CANDIDATES; "
                                            "STOP unless it beats ZS_RAW (Core B) on DEV",
                                    "selected": best, "stop": stop, "dev_means": means})
    (root / "audit").mkdir(exist_ok=True)
    with (root / "audit/selection_log.txt").open("a") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S %z')} {sha(root / 'selection.json')}\n")
    print("DEV_SELECTED", best, "STOP" if stop else "GO", json.dumps({k: round(v, 5) for k, v in means.items()}))


def stage_confirm(root):
    selection = json.loads((root / "selection.json").read_text())
    if sha(root / "selection.json") != (root / "audit/selection_log.txt").read_text().split()[-1]:
        raise PermissionError("selection.json changed since its hash was logged")
    if selection["stop"]:
        raise SystemExit("DEV said STOP: no confirmation")
    model, variant = L.arm_of(selection["selected"])
    reused = json.loads((root / "reused_seals.json").read_text())
    new, logs = engine_predictions(root / "engine/CONF" / model, model, ("CACHE", variant))
    sources = {}
    for cohort in L.CONF_COHORTS:
        zs_old = reused_predictions(reused[cohort]["ZS"])
        src = {"CORE_B": (zs_old, "TTT_m1"), "ZS_CACHE": (zs_old, "CACHE"), "CANDIDATE": (new, variant),
               f"{model}_CACHE_NEW": (new, "CACHE")}
        if model == "JN1":
            src["JN1_CACHE"] = (reused_predictions(reused[cohort]["JN1_CACHE"]), "CACHE")
        sources[cohort] = src
    results = {"selected": selection["selected"]}
    for cohort in L.CONF_COHORTS:
        rows, identity, reads = score(root, [cohort], {cohort: sources[cohort]})
        table = place_table(rows)
        base_cache = "ZS_CACHE" if model == "ZS" else "JN1_CACHE"
        results[cohort] = {
            "places": len(table["CORE_B"]), "means": {k: mean(v) for k, v in table.items()},
            "primary": contrast(table, "CORE_B", "CANDIDATE"),
            "candidate_vs_own_cache": contrast(table, base_cache, "CANDIDATE"),
            "candidate_vs_r7": contrast(table, "R7", "CANDIDATE"), "cache_identity": identity,
            "ground_truth_reads": reads,
            "coverage": {k: float(np.mean([r["coverage"][k] for r in rows])) for k in sources[cohort]}}
        write(root / f"scores/{cohort}/rows.json", rows)
    p1, p2 = results["CONF_SYSTEM3"]["primary"], results["CONF_VAL"]["primary"]
    results["branch"] = L.branch(p1["status"], p2)
    write(root / "results/confirm_results.json", results)
    print("CONFIRM", selection["selected"], results["branch"], json.dumps(
        {c: {"mean": round(results[c]["primary"]["mean"], 5), "status": results[c]["primary"]["status"]}
         for c in L.CONF_COHORTS}))


def main():
    root, stage = Path(sys.argv[1]).resolve(), sys.argv[2]
    {"dev": stage_dev, "confirm": stage_confirm}[stage](root)


if __name__ == "__main__":
    main()
