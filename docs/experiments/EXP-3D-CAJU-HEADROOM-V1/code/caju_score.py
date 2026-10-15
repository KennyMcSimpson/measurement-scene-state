"""EXP-3D-CAJU-HEADROOM-V1: scoring and the preregistered Gate 0 analysis (CPU).

Every scoring stage first checks that the cohort's predictions are sealed and unchanged (the geometry seal; the replay
seals with the protocol's settings and no access violation); only then is ground truth read, through
TeacherAccess.query_depth (purpose SEALED_QUERY_EVALUATION, logged).
  pilot    engineering only, never in the formal statistics: CAJU_PILOT rows, V1's independent reference runs at the
           AbsRel level (equivalence), timings.
  sel      needs every formal seal (CAJU_SEL and CAJU_EVAL); scores CAJU_SEL, chooses the best fixed action
           (caju_logic.select_fixed), writes fixed_action_selection.json and logs its hash before any CAJU_EVAL depth
           is read.
  eval     refuses unless fixed_action_selection.json matches its logged hash; scores CAJU_EVAL.
  analyze  no ground truth read: utilities, policies, the project's place bootstrap, the gate, Pareto points, residual
           bins, history and observed damage, runtime, figures.
Run through cv_host_run.py from the repository root (CUDA hidden):  python caju_score.py ROOT pilot|sel|eval|analyze
"""

import hashlib
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from caju_logic import (  # noqa: E402
    ACTIONS,
    BOOT_SEED,
    DRAWS,
    EPISODE_BAND,
    STEPS,
    absrel,
    best_action,
    compute_checks,
    core_b,
    episodes_from_rows,
    fixed_hull_value,
    frontier,
    gate0,
    pareto_points,
    place_means,
    policy_tables,
    quality_checks,
    select_fixed,
    utilities,
)
from mcss.measurement_planning.information import TeacherAccess  # noqa: E402
from mcss.mechanism_pilot.direct_capacity_statistics import describe  # noqa: E402

REPO = Path.cwd()
IMAGE_SIZE = (128, 160)
FORMAL = ("CAJU_SEL", "CAJU_EVAL")
SETTINGS = {"scale_max": 0.1, "ttt_mult": 1.0, "ttt_steps": 100, "snapshot_step": 10}
POLICIES = ("FIXED_OFF", "FIXED_SHORT", "FIXED_FULL", "FIX", "SUFFICIENT", "XSUFFICIENT", "XROLE", "XORACLE",
            "VORACLE", "ORACLE")
DESCRIPTIVE_FEATURES = ("residual_raw_mean", "residual_norm_covered_mean", "residual_norm_covered_p90", "coverage",
                        "rgb_l1", "residual_top10_mass_share", "residual_stream_mean", "depthsplat_vs_measured_absrel",
                        "measured_depth_spread", "camera_baseline_max_m", "camera_angle_max_deg", "valid_fraction")


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


def lock_of(root, cohort):
    lock = json.loads((root / "frame_role_lock.json").read_text())["cohorts"][cohort]
    if sha(lock["manifest"]) != lock["manifest_sha256"]:
        raise PermissionError(f"{cohort}: manifest changed since the frame-role lock")
    return lock


def replay_seals(root, tag):
    parts = sorted((root / "replay" / tag).glob("replay_seal_*of*.json"))
    if not parts:
        raise FileNotFoundError(f"no replay seal under replay/{tag}")
    counts = {p.stem.split("_")[-1].split("of")[1] for p in parts}
    if len(counts) != 1 or len(parts) != int(counts.pop()):
        raise PermissionError(f"replay/{tag}: incomplete set of shard seals")
    predictions, logs, info = {}, {}, []
    for part in parts:
        seal = json.loads(part.read_text())
        if any(seal[k] != v for k, v in SETTINGS.items()) or seal["limit"]:
            raise PermissionError(f"{part}: not the protocol's run")
        if seal["query_depth_read"] or seal["history_depth_read"] or seal["access_violations"]:
            raise PermissionError(f"{part}: the seal does not certify the information contract")
        predictions.update(seal["predictions"])
        logs.update(seal["episodes_log"])
        info.append({k: seal[k] for k in ("shard", "episodes", "targets", "seconds", "max_memory_gb", "gpu",
                                          "replay_sha256", "v1_ds_eval_sha256", "checkpoint_sha256")})
    return {"predictions": predictions, "logs": logs, "parts": info}


def geometry_seal(root, cohort):
    seal = json.loads((root / "geometry" / f"geometry_seal_{cohort}.json").read_text())
    if seal["query_depth_read"] or seal["history_depth_read"]:
        raise PermissionError(f"{cohort}: geometry seal does not certify that no depth of a target was read")
    return seal


def target_keys(lock):
    return [f"{e['scene_id']}/{e['role']}/{f}" for e in lock["episodes"] for f in e["queries"] + e["history"]]


def check_sealed(root, cohort, tag):
    lock = lock_of(root, cohort)
    keys = set(target_keys(lock))
    geometry, replay = geometry_seal(root, cohort), replay_seals(root, tag)
    if keys - set(geometry["predictions"]) or keys - set(replay["predictions"]):
        raise PermissionError(f"{cohort}: not every target is sealed")
    return lock, geometry, replay


def score_cohort(root, cohort, tag, extra=None):
    lock, geometry, replay = check_sealed(root, cohort, tag)
    records = {r["scene_id"]: r for r in json.loads(Path(lock["manifest"]).read_text())["scenes"]}
    access, rows, started = [], [], time.time()
    for e in lock["episodes"]:
        teacher = TeacherAccess(records[e["scene_id"]], e["queries"] + e["history"], IMAGE_SIZE, REPO, access)
        for fid in e["queries"] + e["history"]:
            key = f"{e['scene_id']}/{e['role']}/{fid}"
            geo, pred = load_checked(geometry["predictions"][key]), load_checked(replay["predictions"][key])
            gt = teacher.query_depth(fid)
            r7, cosine, far7 = geo["REPROJ7_HARMONIC"], geo["cosine"], ~geo["near7"].astype(bool)
            row = {"episode": f"{e['scene_id']}/{e['role']}", "place": e["place"], "frame": fid,
                   "kind": "query" if fid in e["queries"] else "history", "r7": absrel(r7, gt),
                   "far7_share": float(far7[np.isfinite(gt) & (gt > 0)].mean()), "loss": {}, "loss_f7": {},
                   "loss_far7": {}, "coverage": {}}
            arms = [(a, pred) for a in ACTIONS]
            if extra is not None and key in extra:
                ref = load_checked(extra[key])
                arms += [(a, ref) for a in sorted({k.rsplit("_", 2)[0] for k in ref if k.endswith("_depth_z")})]
            for arm, source in arms:
                cb, f7 = core_b(source[f"{arm}_depth_z"], source[f"{arm}_alpha"], r7, cosine)
                row["loss"][arm] = absrel(cb, gt)
                row["loss_f7"][arm] = absrel(f7, gt)
                row["loss_far7"][arm] = absrel(cb, gt, far7)
                row["coverage"][arm] = float((source[f"{arm}_alpha"] >= 0.5)[np.isfinite(gt) & (gt > 0)].mean())
            rows.append(row)
    purposes = {a["purpose"] for a in access}
    if purposes != {"SEALED_QUERY_EVALUATION"}:
        raise PermissionError(f"unexpected read purposes {sorted(purposes)}")
    write(root / "scores" / cohort / "rows.json", rows)
    write(root / "scores" / cohort / "ground_truth_reads.json",
          {"reads": [{k: a[k] for k in ("scene_id", "frame_id", "purpose")} for a in access],
           "seconds": round(time.time() - started, 1)})
    print("SCORED", cohort, len(rows), "targets")
    return rows, replay


# ---------------------------------------------------------------- stages


def selection_log(root):
    return root / "audit" / "selection_log.txt"


def check_selection(root):
    path, log = root / "fixed_action_selection.json", selection_log(root)
    if not path.is_file() or not log.is_file():
        raise PermissionError("the fixed action must be selected on CAJU_SEL (stage sel) first")
    if sha(path) != log.read_text().split()[-1]:
        raise PermissionError("fixed_action_selection.json changed since its hash was logged")
    return json.loads(path.read_text())


def stage_pilot(root):
    eq = json.loads((root / "equivalence" / "equivalence_seal.json").read_text())
    rows, replay = score_cohort(root, "CAJU_PILOT", "PILOT", extra=eq["predictions"])
    episodes = episodes_from_rows(rows)
    fixed, u = select_fixed(episodes)
    per_episode, ref = {}, {}
    for row in rows:
        if row["kind"] == "query" and "SHORT_REF" in row["loss"]:
            ref.setdefault(row["episode"], []).append(row["loss"])
    for e, losses in ref.items():
        mean = {k: float(np.mean([x[k] for x in losses])) for k in losses[0]}
        per_episode[e] = {
            "utility_replay": {a: mean["OFF"] - mean[a] for a in ACTIONS},
            "utility_reference": {a: mean["OFF_REF"] - mean[f"{a}_REF"] for a in ACTIONS},
            "utility_reference2": {a: mean["OFF_REF2"] - mean[f"{a}_REF2"] for a in ACTIONS},
        }
    deltas = {a: max(abs(v["utility_replay"][a] - v["utility_reference"][a]) for v in per_episode.values())
              for a in ACTIONS}
    floors = {a: max(abs(v["utility_reference"][a] - v["utility_reference2"][a]) for v in per_episode.values())
              for a in ACTIONS}
    timing = [v["timing_s"] for v in replay["logs"].values()]
    report = {
        "cohort": "CAJU_PILOT", "engineering_only": True, "episodes": len(episodes),
        "place_mean_utility": u, "best_fixed_on_pilot": fixed,
        "action_absrel": {a: float(np.mean(list(place_means(
            {e: float(np.mean([f[a] for f in v["frames"].values()])) for e, v in episodes.items()},
            {e: v["place"] for e, v in episodes.items()}).values()))) for a in ACTIONS},
        "equivalence_utility": {"max_abs_replay_minus_reference": deltas, "max_abs_reference_run_to_run": floors,
                                "per_episode": per_episode,
                                "equivalent": {a: deltas[a] <= 2 * floors[a] + 1e-6 for a in ACTIONS}},
        "timing_median_s": {k: float(np.median([t[k] for t in timing if k in t])) for k in timing[0]},
        "failed_reverted": sum(v["ttt"]["failed_reverted_to_cache"] for v in replay["logs"].values()),
    }
    write(root / "pilot" / "pilot_report.json", report)
    print("PILOT", json.dumps({k: report[k] for k in ("place_mean_utility", "action_absrel")}),
          json.dumps(report["equivalence_utility"]["equivalent"]))


def stage_sel(root):
    for cohort in FORMAL:
        check_sealed(root, cohort, "FORMAL")
    if (root / "fixed_action_selection.json").exists():
        raise SystemExit("the fixed action was selected already")
    rows, _ = score_cohort(root, "CAJU_SEL", "FORMAL")
    episodes = episodes_from_rows(rows)
    fixed, u = select_fixed(episodes)
    write(root / "fixed_action_selection.json", {
        "rule": "the action with the highest mean CAJU_SEL place utility (quality only: AbsRel(OFF) - AbsRel(a) of "
                "the Core B readout, mean over queries, mean over roles); ties to the cheaper action",
        "fixed_action": fixed, "place_mean_utility": u, "places": len({v["place"] for v in episodes.values()}),
        "episodes": len(episodes)})
    log = selection_log(root)
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S %z')} {sha(root / 'fixed_action_selection.json')}\n")
    print("FIXED_ACTION", fixed, json.dumps(u))


def stage_eval(root):
    check_selection(root)
    score_cohort(root, "CAJU_EVAL", "FORMAL")


def spearman(x, y):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    keep = np.isfinite(x) & np.isfinite(y)
    if keep.sum() < 3:
        return None

    def ranks(v):
        order = np.argsort(v, kind="mergesort")
        r = np.empty(len(v))
        r[order] = np.arange(len(v))
        for value in np.unique(v):
            tied = v == value
            r[tied] = r[tied].mean()
        return r

    rx, ry = ranks(x[keep]), ranks(y[keep])
    return float(np.corrcoef(rx, ry)[0, 1]) if rx.std() > 0 and ry.std() > 0 else None


def stats_of(values):
    return describe(values, draws=DRAWS, seed=BOOT_SEED, positive_means_improvement=True)


def stage_analyze(root):
    selection = check_selection(root)
    fixed = selection["fixed_action"]
    rows = {c: json.loads((root / "scores" / c / "rows.json").read_text()) for c in FORMAL}
    replay = replay_seals(root, "FORMAL")
    logs = replay["logs"]
    eval_eps = episodes_from_rows(rows["CAJU_EVAL"])
    hist_eps = episodes_from_rows(rows["CAJU_EVAL"], kind="history")
    place_of = {e: v["place"] for e, v in eval_eps.items()}
    values, costs, picks = policy_tables(eval_eps, fixed)

    def mean_of(table):
        return float(np.mean(list(table.values())))

    def diff(a, b):
        return {p: values[a][p] - values[b][p] for p in values[a]}

    contrasts = {
        "PRIMARY_XORACLE_MINUS_FIX": diff("XORACLE", "FIX"), "ORACLE_MINUS_FIX": diff("ORACLE", "FIX"),
        "VORACLE_MINUS_FIX": diff("VORACLE", "FIX"), "XROLE_MINUS_FIX": diff("XROLE", "FIX"),
        "FULL_MINUS_OFF": values["FIXED_FULL"], "SHORT_MINUS_OFF": values["FIXED_SHORT"],
        "FULL_MINUS_SHORT": diff("FIXED_FULL", "FIXED_SHORT"), "FIX_MINUS_XSUFFICIENT": diff("FIX", "XSUFFICIENT"),
    }
    boot = {k: stats_of(v) for k, v in contrasts.items()}
    boot.update({f"POLICY_{p}": stats_of(values[p]) for p in POLICIES})
    quality = quality_checks(boot["PRIMARY_XORACLE_MINUS_FIX"])
    saving = 1 - mean_of(costs["XSUFFICIENT"]) / STEPS[fixed] if STEPS[fixed] else None
    cheaper = {a: mean_of(values["FIX"]) - mean_of(values[f"FIXED_{a}"]) for a in ACTIONS
               if a != fixed and STEPS[a] <= 0.5 * STEPS[fixed]}
    compute = compute_checks(fixed, saving, boot["FIX_MINUS_XSUFFICIENT"], cheaper)
    status = gate0(quality, compute)

    u_ep = {e: utilities(v["frames"]) for e, v in eval_eps.items()}
    best = {e: best_action(u) for e, u in u_ep.items()}
    ranking = {e: tuple(sorted(ACTIONS, key=lambda a, u=u: (-round(u[a], 12), STEPS[a]))) for e, u in u_ep.items()}
    modal = Counter(ranking.values()).most_common(1)[0][0]
    by_place = {}
    for e in eval_eps:
        by_place.setdefault(place_of[e], []).append(best[e])
    x_picks = Counter(p for v in picks.values() for p in v["XORACLE"])
    episode_frac = {
        "oracle_action_share": {a: float(np.mean([best[e] == a for e in best])) for a in ACTIONS},
        "xoracle_choice_share": {a: x_picks[a] / sum(x_picks.values()) for a in ACTIONS},
        "beneficial_share": {a: float(np.mean([u[a] > EPISODE_BAND for u in u_ep.values()])) for a in ("SHORT", "FULL")},
        "harmful_share": {a: float(np.mean([u[a] < -EPISODE_BAND for u in u_ep.values()])) for a in ("SHORT", "FULL")},
        "full_beats_short_share": float(np.mean([u["FULL"] - u["SHORT"] > EPISODE_BAND for u in u_ep.values()])),
        "short_beats_full_share": float(np.mean([u["SHORT"] - u["FULL"] > EPISODE_BAND for u in u_ep.values()])),
        "modal_ranking": list(modal),
        "non_modal_ranking_share": float(np.mean([r != modal for r in ranking.values()])),
        "places_with_both_roles_same_best_action": float(np.mean([len(set(v)) == 1 for v in by_place.values()
                                                                    if len(v) > 1])),
    }

    hist_u = {e: utilities(v["frames"]) for e, v in hist_eps.items()}
    history = {
        a: {"place_mean_damage": mean_of(place_means({e: -u[a] for e, u in hist_u.items()}, place_of)),
            "damaged_episode_share": float(np.mean([-u[a] > EPISODE_BAND for u in hist_u.values()])),
            "spearman_query_vs_history_utility": spearman([u_ep[e][a] for e in hist_u], [hist_u[e][a] for e in hist_u])}
        for a in ("SHORT", "FULL")}
    eval_logs = {e: logs[e] for e in eval_eps}
    observed = {a: {k: [v["observed_damage"][a][k] for v in eval_logs.values()
                       if v["observed_damage"][a][k] is not None] for k in ("mean_error_increase", "broken_share")}
                for a in ("SHORT", "FULL")}
    observed = {a: {k: {"mean": float(np.mean(x)), "median": float(np.median(x)), "max": float(np.max(x))}
                    for k, x in d.items() if x} for a, d in observed.items()}

    fixed_points = [{"action": a, "steps": float(STEPS[a]), "utility": mean_of(values[f"FIXED_{a}"])} for a in ACTIONS]
    pareto = {"fixed_points": fixed_points}
    for mode in ("in_sample", "cross_query"):
        pts = pareto_points(eval_eps, mode)
        pareto[mode] = {"points": pts, "frontier": frontier(pts),
                        "advantage_over_fixed_mixtures": {
                            str(b): max(p["utility"] for p in pts if p["steps"] <= b + 1e-9)
                            - fixed_hull_value(fixed_points, b) for b in (5, 10, 25, 50, 100)}}

    formal_logs = {e: v for e, v in logs.items() if v["cohort"] in FORMAL}
    residual = {e: v["features"]["residual_raw_mean"] for e, v in formal_logs.items()}
    edges = [float(x) for x in np.percentile(list(residual.values()), [100 / 3, 200 / 3])]

    def bin_of(x):
        return int(np.searchsorted(edges, x, side="right"))

    bins = {a: [{"beneficial": 0, "neutral": 0, "harmful": 0} for _ in range(3)] for a in ("SHORT", "FULL")}
    for e, u in u_ep.items():
        for a in ("SHORT", "FULL"):
            label = "beneficial" if u[a] > EPISODE_BAND else "harmful" if u[a] < -EPISODE_BAND else "neutral"
            bins[a][bin_of(residual[e])][label] += 1
    feature_corr = {f: {a: spearman([logs[e]["features"].get(f) if logs[e]["features"].get(f) is not None
                                     else np.nan for e in u_ep], [u_ep[e][a] for e in u_ep]) for a in ("SHORT", "FULL")}
                    for f in DESCRIPTIVE_FEATURES}

    timing = [v["timing_s"] for v in formal_logs.values()]

    def summary(x):
        x = [v for v in x if v is not None]
        return {"median": float(np.median(x)), "mean": float(np.mean(x)), "p90": float(np.percentile(x, 90)),
                "n": len(x)}

    runtime = {
        "unit": "seconds, CUDA-synchronized, on GPU 4 of hit shared with other processes (time-sliced)",
        "encode": summary([t["encode"] for t in timing]),
        "readout_per_action": summary([t[f"readout_{a}"] for t in timing for a in ACTIONS]),
        "ttt_steps_1_to_10": summary([t["steps_1_to_snapshot"] for t in timing]),
        "ttt_steps_11_to_100": summary([t.get("steps_after_snapshot") for t in timing]),
        "seconds_per_step_1_to_10": summary([t["steps_1_to_snapshot"] / 10 for t in timing]),
        "seconds_per_step_11_to_100": summary([t["steps_after_snapshot"] / 90 for t in timing
                                               if "steps_after_snapshot" in t]),
        "extra_compute_by_action": {
            "OFF": 0.0, "SHORT": float(np.median([t["steps_1_to_snapshot"] for t in timing])),
            "FULL": float(np.median([t["steps_1_to_snapshot"] + t.get("steps_after_snapshot", 0.0) for t in timing]))},
        "shards": replay["parts"],
        "failed_reverted": sum(v["ttt"]["failed_reverted_to_cache"] for v in formal_logs.values()),
    }

    action_absrel = {c: {a: mean_of(place_means(
        {e: float(np.mean([f[a] for f in v["frames"].values()])) for e, v in episodes_from_rows(rows[c]).items()},
        {e: v["place"] for e, v in episodes_from_rows(rows[c]).items()})) for a in ACTIONS} for c in FORMAL}
    action_absrel_f7 = {a: mean_of(place_means(
        {e: float(np.mean([f[a] for f in v["frames"].values()])) for e, v in
         episodes_from_rows(rows["CAJU_EVAL"], metric="loss_f7").items()}, place_of)) for a in ACTIONS}
    headroom = {
        "fixed_action": fixed, "selection": selection, "gate0_status": status,
        "dynamic_action_space": "NO_USEFUL_DYNAMIC_ACTION_SPACE" if status.startswith("NOT_ESTABLISHED") else status,
        "quality_checks": quality, "compute_checks": compute, "compute_saving_fraction": saving,
        "cheaper_fixed_quality_loss": cheaper,
        "policy_place_mean_utility": {p: mean_of(values[p]) for p in POLICIES},
        "policy_place_mean_steps": {p: mean_of(costs[p]) for p in POLICIES},
        "headroom": {k: boot[k]["mean"] for k in ("PRIMARY_XORACLE_MINUS_FIX", "ORACLE_MINUS_FIX",
                                                    "VORACLE_MINUS_FIX", "XROLE_MINUS_FIX")},
        "episode_fractions": episode_frac, "marginal_full_over_short": boot["FULL_MINUS_SHORT"]["mean"],
        "pareto": pareto, "history_damage": history, "observed_damage": observed,
        "residual_bins": {"feature": "residual_raw_mean (pre-update, the TTT step-0 depth term)",
                          "edges": edges, "edge_rule": "tertiles over the 100 formal episodes (no ground truth)",
                          "band": EPISODE_BAND, "counts": bins},
        "feature_spearman_with_utility": feature_corr,
        "places": len(values["FIX"]), "episodes": len(eval_eps),
    }
    utility = {
        "cohort": "CAJU_EVAL", "fixed_action": fixed, "action_place_mean_absrel": action_absrel,
        "action_place_mean_absrel_f7_readout": action_absrel_f7,
        "per_place_utility": {a: values[f"FIXED_{a}"] for a in ACTIONS},
        "per_episode": {e: {"place": place_of[e], "quality_utility": u_ep[e],
                            "history_utility": hist_u.get(e), "observed_damage": logs[e]["observed_damage"],
                            "cost_steps": STEPS, "cost_seconds": {
                                "SHORT": logs[e]["timing_s"]["steps_1_to_snapshot"],
                                "FULL": logs[e]["timing_s"]["steps_1_to_snapshot"]
                                + logs[e]["timing_s"].get("steps_after_snapshot", 0.0)},
                            "fit": logs[e]["fit"], "features": logs[e]["features"],
                            "ttt_failed": logs[e]["ttt"]["failed_reverted_to_cache"]} for e in eval_eps},
        "net_utility": "quality only (lambda_c = lambda_h = 0, fixed before any utility was seen); cost and damage "
                       "are reported separately (Pareto, history_damage, observed_damage)",
    }
    write(root / "results" / "counterfactual_utility.json", utility)
    write(root / "results" / "oracle_headroom.json", headroom)
    write(root / "results" / "bootstrap_results.json", boot)
    write(root / "results" / "runtime_profile.json", runtime)
    figures(root, u_ep, values, boot, pareto, residual, edges, bins, fixed)
    print("GATE0", status, json.dumps({k: round(v, 6) for k, v in headroom["headroom"].items()}))


def figures(root, u_ep, values, boot, pareto, residual, edges, bins, fixed):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = root / "figures"
    out.mkdir(exist_ok=True)
    eps = sorted(u_ep)
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for i, a in enumerate(("SHORT", "FULL")):
        x = np.array([u_ep[e][a] for e in eps])
        ax.scatter(x, np.full(len(x), i) + np.linspace(-0.25, 0.25, len(x)), s=10, label=a)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_yticks([0, 1], ["SHORT (10)", "FULL (100)"])
    ax.set_xlabel("per-episode utility vs OFF (AbsRel(OFF) - AbsRel(action))")
    ax.set_title("Fig. 1  Future-query utility of each action (CAJU_EVAL)")
    fig.tight_layout()
    fig.savefig(out / "fig1_action_utility.png", dpi=150)
    plt.close(fig)

    names = [p for p in POLICIES]
    means = [boot[f"POLICY_{p}"]["mean"] for p in names]
    lo = [m - boot[f"POLICY_{p}"]["ci95"][0] for p, m in zip(names, means, strict=True)]
    hi = [boot[f"POLICY_{p}"]["ci95"][1] - m for p, m in zip(names, means, strict=True)]
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    ax.bar(range(len(names)), means, yerr=[lo, hi], capsize=3, color=["#888"] * 3 + ["#357"] + ["#6a9"] * 6)
    ax.set_xticks(range(len(names)), names, rotation=35, ha="right")
    ax.set_ylabel("place-mean utility vs OFF")
    ax.set_title(f"Fig. 2  Oracles vs the best fixed action ({fixed}), 95% place bootstrap")
    fig.tight_layout()
    fig.savefig(out / "fig2_oracle_vs_fixed.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    fp = sorted(pareto["fixed_points"], key=lambda p: p["steps"])
    ax.plot([p["steps"] for p in fp], [p["utility"] for p in fp], "ko-", label="fixed actions (mixtures)")
    for mode, style in (("in_sample", "C1.--"), ("cross_query", "C0.-")):
        f = pareto[mode]["frontier"]
        ax.plot([p["steps"] for p in f], [p["utility"] for p in f], style, label=f"per-episode oracle ({mode})")
    ax.set_xlabel("mean TTT steps per episode")
    ax.set_ylabel("place-mean utility vs OFF")
    ax.set_title("Fig. 3  Quality-compute Pareto (CAJU_EVAL)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "fig3_pareto.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for a, c in (("SHORT", "C1"), ("FULL", "C0")):
        ax.scatter([residual[e] for e in eps], [u_ep[e][a] for e in eps], s=12, c=c, label=a)
    for x in edges:
        ax.axvline(x, color="gray", lw=0.8, ls=":")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("pre-update depth residual on the 7 arrived frames (relative L1)")
    ax.set_ylabel("utility vs OFF")
    ax.set_title("Fig. 4  Residual size vs true update utility")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "fig4_residual_vs_utility.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.6), sharey=True)
    for ax, a in zip(axes, ("SHORT", "FULL"), strict=True):
        bottom = np.zeros(3)
        for label, color in (("beneficial", "#4a8"), ("neutral", "#bbb"), ("harmful", "#c55")):
            h = np.array([b[label] for b in bins[a]])
            ax.bar(range(3), h, bottom=bottom, color=color, label=label)
            bottom += h
        ax.set_xticks(range(3), ["low", "mid", "high"])
        ax.set_title(f"{a}: episodes per residual tertile")
    axes[0].set_ylabel("episodes")
    axes[1].legend(fontsize=8)
    fig.suptitle("Fig. 5  Beneficial / harmful updates within residual bins (band 0.0005)")
    fig.tight_layout()
    fig.savefig(out / "fig5_residual_bins.png", dpi=150)
    plt.close(fig)


def main():
    root, stage = Path(sys.argv[1]).resolve(), sys.argv[2]
    {"pilot": stage_pilot, "sel": stage_sel, "eval": stage_eval, "analyze": stage_analyze}[stage](root)


if __name__ == "__main__":
    main()
