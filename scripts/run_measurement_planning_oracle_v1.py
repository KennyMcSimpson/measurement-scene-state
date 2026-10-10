#!/usr/bin/env python3
"""EXP-MEASUREMENT-PLANNING-ORACLE-V1: oracle headroom audit of budgeted depth acquisition (no
training).

Task (MEASUREMENT_PLANNING_TASK_V1, preregistration.md): an episode is one record and one context
role. The role's three RGB-D frames are measured; the record's frames that no role uses form the
candidate pool, whose RGB and cameras are free and whose depth is revealed only on request; the
primary queries are scored. A subset S of requested candidates is reconstructed by the frozen
model-free backend (z-buffer of all measured frames + harmonic fill, i.e. REPROJ_HARMONIC over
context + S). Every subset of the pool is evaluated (teacher path), which gives the exact best-K
oracle, the cross-query oracle and the exact random expectation; deployable heuristics act through
the MeasurementEnv only.

Stages (repository root, project environment, through cv_host_run.py):
  lock                   freeze contracts, scene manifest and hashes (refused once results exist)
  tests                  run and record this experiment's tests
  compute --cohort NAME  per-episode raw tables (EVAL34, EVAL34_GAP3, EVAL_FAR, REPLICA, DEV)
  analyze                statistics, gate, figures, final summary
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from mcss.measurement_planning import geometry as g
from mcss.measurement_planning import heuristics as h
from mcss.measurement_planning.information import (
    CANDIDATE_DEPTH_PURPOSE,
    CANDIDATE_RGB_PURPOSE,
    CONTEXT_PURPOSE,
    ROLE_KEYS,
    candidate_pool,
    query_ids,
)
from mcss.measurement_planning.oracle import FIELDS, run_episode
from mcss.mechanism_pilot.direct_capacity_statistics import describe, paired
from mcss.mechanism_pilot.geometry_carrier_statistics import _evidence, surface_status

EXPERIMENT = "EXP-MEASUREMENT-PLANNING-ORACLE-V1"
REPO = g.REPO
# A rehearsal redirects every output (MP_ORACLE_REHEARSAL_OUT) and treats DEV as its primary cohort
# (code-path check only; it prints no numbers). The formal run never sets it.
REHEARSAL = "MP_ORACLE_REHEARSAL_OUT" in os.environ
OUT = Path(os.environ.get("MP_ORACLE_REHEARSAL_OUT", REPO / "outputs" / EXPERIMENT))
PRIMARY = "DEV" if REHEARSAL else "EVAL34"
DOCS = REPO / "docs" / "experiments" / EXPERIMENT
DRAWS, SEED = 10000, 20260928
PRIMARY_BUDGETS = (1, 2, 4)
HEADROOM_FRACTION_MIN = 0.20
REPLAY_TOLERANCE_M = 1e-9  # K = 0 against the sealed REPROJ_HARMONIC (INPAINT-V1 precedent)
GAP = 3
V17_CONTRACT = REPO / "outputs/EXP-3D-RGBD-LONG-TRAIN-V17/training_contract.json"
MANIFESTS = {
    "EVAL_V3": REPO / "outputs/EXP-3D-RGBD-EVAL-V3-DATA/manifest_eval_v3.json",
    "EVAL_V4": REPO / "outputs/EXP-3D-RGBD-EVAL-V4-DATA/manifest_eval_v4.json",
    "EVAL_FAR": REPO / "outputs/EXP-3D-RGBD-EVAL-FAR-DATA/manifest_eval_far.json",
    "REPLICA": REPO / "outputs/EXP-3D-RGBD-REPLICA-DATA-V1/manifest_replica.json",
    "V7_SPLIT": REPO / "outputs/EXP-3D-RGBD-DEPTH-BOUNDS-V7/scene_split.json",
}
SEALS = {
    "EVAL34": Path.home() / "cv_work/depthsplat/probe_eval34/raw/eval/EVAL34/prediction_seal.json",
    "DEV": Path.home() / "cv_work/depthsplat/probe_dev/raw/eval/DEV/prediction_seal.json",
}
COHORTS = {
    # name: (sources, query role, minimum frame gap to queries, use)
    "EVAL34": (("EVAL_V3", "EVAL_V4"), "primary_query", 0, "PRIMARY"),
    "EVAL34_GAP3": (("EVAL_V3", "EVAL_V4"), "primary_query", GAP, "SECONDARY_GAP_POOL"),
    "EVAL_FAR": (("EVAL_FAR",), "far_query", 0, "SECONDARY_FAR_QUERIES"),
    "REPLICA": (("REPLICA",), "primary_query", 0, "DESCRIPTIVE_CROSS_DATASET"),
    "DEV": (("V7_SPLIT",), "primary_query", 0, "REHEARSAL_ONLY"),
}
DIAGNOSTIC = ("ORACLE", "XORACLE", "RANDOM_EXPECTED", "WORST", "QUERY_NEAREST")
CODE = (
    "src/mcss/measurement_planning/__init__.py",
    "src/mcss/measurement_planning/information.py",
    "src/mcss/measurement_planning/geometry.py",
    "src/mcss/measurement_planning/heuristics.py",
    "src/mcss/measurement_planning/oracle.py",
    "scripts/run_measurement_planning_oracle_v1.py",
    "tests/test_measurement_planning_oracle_v1.py",
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def fill_constant():
    return float(read(V17_CONTRACT)["eval"]["reprojection_fill_constant_m"])


# ---------------------------------------------------------------- cohorts and the frame-role lock
def records(name):
    sources, query_role, gap, _ = COHORTS[name]
    out, protected = [], set()
    for source in sources:
        manifest = read(MANIFESTS[source])
        protected |= set(manifest.get("protected_scene_ids", []))
        rows = manifest["scenes"]
        if source == "V7_SPLIT":
            rows = [r for r in rows if r["split"] == "DEV"]
        out.extend(rows)
    ids = [r["scene_id"] for r in out]
    if len(ids) != len(set(ids)) or set(ids) & protected:
        raise PermissionError("Duplicate or protected scene in a cohort")
    image_size = tuple(read(MANIFESTS[sources[0]])["image_size"])
    return out, query_role, gap, image_size


def episodes(name):
    rows, query_role, gap, image_size = records(name)
    jobs = []
    for record in rows:
        pool = candidate_pool(record, gap)
        queries = list(record["roles"][query_role])
        if set(pool) & set(queries) or set(pool) & set(query_ids(record)):
            raise PermissionError("A candidate is a query")
        for role in ("A", "B"):
            jobs.append(
                {
                    "cohort": name,
                    "scene_id": record["scene_id"],
                    "role": role,
                    "context": list(record["roles"][ROLE_KEYS[role]]),
                    "pool": pool,
                    "queries": queries,
                    "image_size": list(image_size),
                }
            )
    return rows, jobs


# ---------------------------------------------------------------- lock
def lock():
    if (OUT / "raw").exists() and any((OUT / "raw").rglob("*.npz")):
        raise PermissionError(
            "Results exist: the contracts are frozen once, before any computation"
        )
    for path in ("PROTOCOL.md", "preregistration.md"):
        if not (OUT / path).exists():
            raise FileNotFoundError(f"{path} must be written before the lock")
    constant = fill_constant()
    scene_manifest, frame_roles = {}, {}
    for name in COHORTS:
        rows, jobs = episodes(name)
        scene_manifest[name] = {
            "use": COHORTS[name][3],
            "scenes": sorted({j["scene_id"] for j in jobs}),
            "episodes": len(jobs),
            "query_role": COHORTS[name][1],
            "min_query_gap": COHORTS[name][2],
            "pool_sizes": sorted({len(j["pool"]) for j in jobs}),
        }
        frame_roles[name] = {
            r["scene_id"]: {
                "roles": {k: v for k, v in r["roles"].items() if v},
                "pool": candidate_pool(r, COHORTS[name][2]),
            }
            for r in rows
        }
    train = {r["scene_id"] for r in read(MANIFESTS["V7_SPLIT"])["scenes"] if r["split"] == "TRAIN"}
    for name, entry in scene_manifest.items():
        if set(entry["scenes"]) & train:
            raise PermissionError(f"{name} overlaps TRAIN72")
    write(OUT / "scene_manifest.json", scene_manifest)
    write(OUT / "frame_role_lock.json", frame_roles)
    write(
        OUT / "task_contract.json",
        {
            "task": "MEASUREMENT_PLANNING_TASK_V1",
            "episode": "one record x one context role (A or B)",
            "initial_context": "the role's three RGB-D frames (context_a or context_b)",
            "candidate_pool": (
                "frames no role uses (not context of either role, query, stream, "
                "ignored, far_query)"
            ),
            "gap_pool": (
                f"EVAL34_GAP3: candidates at least {GAP} frame indices from every primary query"
            ),
            "queries": {k: v[1] for k, v in COHORTS.items()},
            "primary_budgets": list(PRIMARY_BUDGETS),
            "budget_curve": "K = 0 .. pool size",
            "backend": "frozen REPROJ_HARMONIC over initial context + requested candidates",
            "regions": (
                "NEAR / FAR from the initial context's hits (frozen 8 px rule), "
                "fixed for every subset"
            ),
            "metric": "AbsRel on GT-valid query pixels (arms-evaluator rule); "
            "episode = mean over its queries; "
            "scene = mean over roles",
            "primary_metric": "ALL AbsRel",
            "fill_constant_m": constant,
            "cohorts": {k: {"use": v[3], "sources": list(v[0])} for k, v in COHORTS.items()},
        },
    )
    write(
        OUT / "information_contract.json",
        {
            "deployable_may_see": [
                "initial context RGB, depth and cameras",
                "candidate RGB and cameras (all candidates, before any request)",
                "depth of candidates it has requested, within the budget",
            ],
            "deployable_never_sees": [
                "candidate depth before its request",
                "query RGB, depth or camera",
                "the other role's context frames",
                "oracle utilities or any teacher output",
            ],
            "enforcement": "MeasurementEnv holds a sanitized record "
            "(role context + candidates only); "
            "request_depth enforces membership, uniqueness and budget; "
            "every read is logged by purpose",
            "teacher_only": ["ORACLE", "XORACLE", "RANDOM_EXPECTED", "WORST", "QUERY_NEAREST"],
            "purposes": {
                "context": CONTEXT_PURPOSE,
                "candidate_rgb": CANDIDATE_RGB_PURPOSE,
                "candidate_depth": CANDIDATE_DEPTH_PURPOSE,
            },
        },
    )
    write(
        OUT / "baseline_contract.json",
        {
            "deployable": {name: (fn.__doc__ or "").strip() for name, fn in h.DEPLOYABLE.items()},
            "diagnostic": {
                "ORACLE": (
                    "best subset of size K per episode by the episode's mean ALL AbsRel (in-sample)"
                ),
                "XORACLE": (
                    "cross-query oracle: subset chosen on the other query of the "
                    "episode, scored on this one"
                ),
                "RANDOM_EXPECTED": (
                    "exact mean over all subsets of size K (uniform random policy expectation)"
                ),
                "WORST": "worst subset of size K (range only)",
                "QUERY_NEAREST": (
                    "K candidates nearest to any query camera (uses query cameras; not deployable)"
                ),
            },
            "constants": {
                "fill_cap_px": h.FILL_CAP_PX,
                "pose_metres_per_radian": h.POSE_METRES_PER_RADIAN,
                "novelty_relative": h.NOVELTY_RELATIVE,
                "random_seed": h.RANDOM_SEED,
            },
            "best_heuristic_rule": "per K, the deployable selector with the lowest mean "
            "scene ALL AbsRel "
            "(chosen on the same data: conservative for headroom)",
        },
    )
    hashes = {
        "code": {p: sha(REPO / p) for p in CODE},
        "frozen_functions": g.frozen_hashes(),
        "manifests": {k: sha(v) for k, v in MANIFESTS.items()},
        "seals": {k: sha(v) for k, v in SEALS.items() if v.exists()},
        "v17_contract": sha(V17_CONTRACT),
        "documents": {p: sha(OUT / p) for p in ("PROTOCOL.md", "preregistration.md")},
        "contracts": {
            p: sha(OUT / p)
            for p in (
                "task_contract.json",
                "information_contract.json",
                "baseline_contract.json",
                "scene_manifest.json",
                "frame_role_lock.json",
            )
        },
    }
    gate = {
        "primary_cohort": "EVAL34",
        "budgets": list(PRIMARY_BUDGETS),
        "headroom_fraction_min": HEADROOM_FRACTION_MIN,
        "rule": "HEADROOM_SUPPORTED at K iff the cross-query contrast BEST_H_K - XORACLE_K "
        "passes the V2 gate "
        "(SUPPORTED) and (BEST_H_K - ORACLE_K) / (NO_DEPTH - ORACLE_K) >= 0.20 in scene means; "
        "HEADROOM_STATUS = SUPPORTED if any K in (1, 2, 4) is supported, else NOT_ESTABLISHED; "
        "POLICY_TRAINING_ALLOWED iff SUPPORTED",
        "statistics": {"unit": "scene", "draws": DRAWS, "seed": SEED},
    }
    write(
        OUT / "preregistration.json",
        {
            "experiment": EXPERIMENT,
            "gate": gate,
            "hashes": hashes,
            "rehearsal": REHEARSAL,
            "locked_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        },
    )
    print("LOCKED", sha(OUT / "preregistration.json"), flush=True)


def verify_lock():
    pre = read(OUT / "preregistration.json")
    for group, mapping in pre["hashes"].items():
        if group == "frozen_functions":
            current = g.frozen_hashes()
            if current != mapping:
                raise PermissionError("A frozen function changed")
            continue
        if isinstance(mapping, str):
            if sha(V17_CONTRACT) != mapping:
                raise PermissionError("V17 contract changed")
            continue
        for key, digest in mapping.items():
            path = {
                "code": lambda k: REPO / k,
                "manifests": lambda k: MANIFESTS[k],
                "seals": lambda k: SEALS[k],
                "documents": lambda k: OUT / k,
                "contracts": lambda k: OUT / k,
            }[group](key)
            if sha(path) != digest:
                raise PermissionError(f"{group}/{key} changed after the lock")
    return pre


def compute(cohort, workers):
    verify_lock()
    rows, jobs = episodes(cohort)
    by_id = {r["scene_id"]: r for r in rows}
    constant = fill_constant()
    out = OUT / "raw" / cohort
    if out.exists() and any(out.glob("*.npz")):
        raise FileExistsError(f"{cohort} was computed once")
    seal_prefix = "EVAL34" if cohort in ("EVAL34", "EVAL34_GAP3") else cohort
    seal_path = SEALS.get(seal_prefix)
    for job in jobs:
        job["record"] = by_id[job["scene_id"]]
        job["constant"] = constant
        job["repo"] = str(REPO)
        job["out_dir"] = str(out)
        job["seal_prefix"] = seal_prefix
        job["seal_path"] = str(seal_path) if seal_path is not None and seal_path.exists() else None
    started = time.time()
    with Pool(workers) as pool:
        done = pool.map(run_episode, jobs, chunksize=1)
    write(
        out / "_done.json", {"episodes": sorted(done), "seconds": round(time.time() - started, 1)}
    )
    print(
        "COMPUTED", cohort, len(done), "episodes", round(time.time() - started, 1), "s", flush=True
    )


# ---------------------------------------------------------------- analysis
def load(cohort):
    folder = OUT / "raw" / cohort
    episodes_ = {}
    for path in sorted(folder.glob("*.json")):
        if path.name.startswith("_"):
            continue
        meta = read(path)
        npz = folder / f"{path.stem}.npz"
        if sha(npz) != meta["table_sha256"]:
            raise PermissionError(f"{npz} changed")
        meta["table"] = dict(np.load(npz))
        episodes_[path.stem] = meta
    return episodes_


def popcount(bits):
    return bin(bits).count("1")


def mean_over_queries(values):
    """Episode value per subset: mean over the episode's queries, skipping empty queries."""
    with np.errstate(all="ignore"):
        out = np.nanmean(values, axis=1)
    return out


def episode_methods(meta):
    """Per method and K: the chosen subset(s) and the episode's value of every field."""
    table = {f: meta["table"][f] for f in FIELDS}
    pool = meta["pool"]
    n = len(pool)
    sizes = np.array([popcount(b) for b in range(1 << n)])
    score = mean_over_queries(table["all"])
    per_query_all = table["all"]
    index = {f: i for i, f in enumerate(pool)}

    def bits_of(frames):
        return int(sum(1 << index[f] for f in frames))

    def values(bits):
        return {f: float(mean_over_queries(table[f][[bits]])[0]) for f in FIELDS}

    result = {0: {"NO_DEPTH": values(0)}}
    for k in range(1, n + 1):
        members = np.flatnonzero(sizes == k)
        best = int(members[np.argmin(score[members])])
        worst = int(members[np.argmax(score[members])])
        methods = {
            "ORACLE": values(best),
            "WORST": values(worst),
            "RANDOM_EXPECTED": {
                f: float(np.nanmean(mean_over_queries(table[f][members]))) for f in FIELDS
            },
        }
        nq = per_query_all.shape[1]
        cross = defaultdict(list)
        for q in range(nq):
            others = [o for o in range(nq) if o != q]
            if not others:
                continue
            chooser = np.nanmean(per_query_all[members][:, others], axis=1)
            pick = int(members[np.argmin(chooser)])
            for f in FIELDS:
                cross[f].append(table[f][pick, q])
        methods["XORACLE"] = {
            f: float(np.nanmean(v)) if np.isfinite(v).any() else float("nan")
            for f, v in cross.items()
        }
        for name, chosen in meta["selections"].items():
            methods[name] = values(bits_of(chosen[str(k)]))
        methods["QUERY_NEAREST"] = values(
            bits_of(meta["diagnostic_selections"]["QUERY_NEAREST"][str(k)])
        )
        methods["_oracle_frames"] = [pool[i] for i in range(n) if best >> i & 1]
        result[k] = methods
    return result


def scene_table(episodes_):
    """{K: {method: {field: {scene: value}}}}; scene = mean over its roles (NaN roles skipped)."""
    grouped = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
    for meta in episodes_.values():
        for k, methods in episode_methods(meta).items():
            for method, fields in methods.items():
                if method.startswith("_"):
                    continue
                for f, v in fields.items():
                    if v is not None and np.isfinite(v):
                        grouped[k][method][f][meta["scene_id"]].append(v)
    return {
        k: {
            m: {
                f: {s: float(np.mean(v)) for s, v in scenes.items()} for f, scenes in fields.items()
            }
            for m, fields in methods.items()
        }
        for k, methods in grouped.items()
    }


def gate(stats):
    checks = {
        "mean_ci_and_scene_consistency": _evidence(stats),
        "every_leave_one_scene_out_gain_positive": bool(stats["loso"])
        and min(stats["loso"].values()) > 0,
        "positive_top1_share_at_most_0_5": stats["positive_top1_contribution"] is not None
        and stats["positive_top1_contribution"] <= 0.5,
    }
    return surface_status(stats, checks), checks


def contrast(a, b):
    common = sorted(set(a) & set(b))
    stats = paired({s: a[s] for s in common}, {s: b[s] for s in common}, draws=DRAWS, seed=SEED)
    stats.pop("per_scene", None)
    return stats


def headroom(table, budgets, deployable):
    """Per K: best deployable heuristic, in-sample and cross-query headroom, oracle gain, gate."""
    out = {}
    nodepth = table[0]["NO_DEPTH"]["all"]
    for k in budgets:
        if k not in table:
            continue
        rows = table[k]
        means = {m: float(np.mean(list(rows[m]["all"].values()))) for m in deployable if m in rows}
        best = min(means, key=lambda m: (means[m], m))
        best_values = rows[best]["all"]
        d = contrast(best_values, rows["ORACLE"]["all"])
        x = contrast(best_values, rows["XORACLE"]["all"])
        gain = contrast(nodepth, rows["ORACLE"]["all"])
        fraction = d["mean"] / gain["mean"] if gain["mean"] > 0 else None
        status, checks = gate(x)
        supported = (
            status == "SUPPORTED" and fraction is not None and fraction >= HEADROOM_FRACTION_MIN
        )
        out[str(k)] = {
            "best_heuristic": best,
            "heuristic_means_all": means,
            "oracle_minus_best": d,
            "xoracle_minus_best": x,
            "oracle_gain_over_no_depth": gain,
            "best_gain_over_no_depth": contrast(nodepth, best_values),
            "best_vs_random_expected": contrast(rows["RANDOM_EXPECTED"]["all"], best_values),
            "headroom_fraction": fraction,
            "xoracle_gate_status": status,
            "xoracle_gate_checks": checks,
            "headroom_supported": bool(supported),
        }
    return out


def budget_curve(table):
    curve = {}
    for k in sorted(table):
        curve[str(k)] = {
            m: {f: float(np.mean(list(v.values()))) if v else None for f, v in fields.items()}
            for m, fields in table[k].items()
        }
    return curve


def spearman(x, y):
    from scipy.stats import spearmanr

    if len(set(x)) < 2 or len(set(y)) < 2:
        return None
    return float(spearmanr(x, y).statistic)


def utility_structure(episodes_):
    """Single-step utilities U_i = L(no depth) - L({i}) (ALL) against deployable features, within
    episodes."""
    names = (
        "translation_m",
        "rotation_deg",
        "pose_distance",
        "uncovered",
        "fill_distance_px",
        "photometric",
        "novelty3d",
        "pool_index",
    )
    per_scene = defaultdict(lambda: defaultdict(list))
    top1 = defaultdict(list)
    points = []
    oracle_positions = defaultdict(int)
    marginal = defaultdict(list)
    for meta in episodes_.values():
        pool = meta["pool"]
        table = mean_over_queries(meta["table"]["all"])
        far_cov = mean_over_queries(meta["table"]["coverage_far"])
        far_err = mean_over_queries(meta["table"]["far"])
        utility = {f: float(table[0] - table[1 << i]) for i, f in enumerate(pool)}
        best = max(pool, key=lambda f: (utility[f], -f))
        oracle_positions[pool.index(best)] += 1
        feats = meta["features"]
        for name in names:
            xs = [feats[str(f)][name] for f in pool]
            ys = [utility[f] for f in pool]
            rho = spearman(xs, ys)
            if rho is not None:
                per_scene[name][meta["scene_id"]].append(rho)
            pick = max(pool, key=lambda f: (feats[str(f)][name], -f))
            top1[name].append(pick == best)
        for i, f in enumerate(pool):
            points.append(
                {
                    "utility": utility[f],
                    **{name: feats[str(f)][name] for name in names},
                    "far_coverage_gain": float(far_cov[1 << i] - far_cov[0])
                    if np.isfinite(far_cov[0])
                    else None,
                    "far_improvement": float(far_err[0] - far_err[1 << i])
                    if np.isfinite(far_err[0])
                    else None,
                    "scene_id": meta["scene_id"],
                }
            )
        n = len(pool)
        for bits in range(1 << n):
            m = popcount(bits)
            if m > 3:
                continue
            for i in range(n):
                if not bits >> i & 1:
                    marginal[m].append(float(table[bits] - table[bits | 1 << i]))
    correlations = {}
    for name, scenes in per_scene.items():
        values = {s: float(np.mean(v)) for s, v in scenes.items()}
        stats = describe(values, draws=DRAWS, seed=SEED)
        stats.pop("per_scene", None)
        correlations[name] = {
            "mean_within_episode_spearman": stats,
            "top1_agreement": float(np.mean(top1[name])),
        }
    best = max(
        correlations, key=lambda k: abs(correlations[k]["mean_within_episode_spearman"]["mean"])
    )
    rho = abs(correlations[best]["mean_within_episode_spearman"]["mean"])
    label = "STRONG" if rho > 0.6 else "MODERATE" if rho >= 0.3 else "WEAK"
    total = sum(oracle_positions.values())
    modal = max(oracle_positions.values()) / total
    tercile = []
    by_scene = defaultdict(list)
    for p in points:
        by_scene[p["scene_id"]].append(p)
    for group in by_scene.values():
        u = np.array([p["utility"] for p in group])
        c = np.array([p["uncovered"] for p in group])
        hi_c, lo_c = c >= np.quantile(c, 2 / 3), c <= np.quantile(c, 1 / 3)
        hi_u, lo_u = u >= np.quantile(u, 2 / 3), u <= np.quantile(u, 1 / 3)
        tercile.append(
            {
                "high_cov_low_util": int((hi_c & lo_u).sum()),
                "low_cov_high_util": int((lo_c & hi_u).sum()),
                "n": len(group),
            }
        )
    return {
        "correlations": correlations,
        "best_feature": best,
        "predictability_label": label,
        "oracle_best1_positions": {str(k): v for k, v in sorted(oracle_positions.items())},
        "oracle_best1_modal_share": modal,
        "utility_scene_dependent": bool(modal < 0.5),
        "mismatch_counts": {
            "high_uncovered_low_utility": sum(t["high_cov_low_util"] for t in tercile),
            "low_uncovered_high_utility": sum(t["low_cov_high_util"] for t in tercile),
            "candidates": sum(t["n"] for t in tercile),
        },
        "marginal_utility_by_history": {
            str(m): {"mean": float(np.mean(v)), "median": float(np.median(v)), "n": len(v)}
            for m, v in sorted(marginal.items())
        },
        "_points": points,
        "_marginal": {str(m): v for m, v in marginal.items()},
    }


def figures(curve, scenes, util, gates, folder):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    folder.mkdir(parents=True, exist_ok=True)
    ks = sorted(int(k) for k in curve)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for method, style in (
        ("RANDOM_EXPECTED", ":"),
        ("UNIFORM", "--"),
        ("ORACLE", "-"),
        ("XORACLE", "-."),
    ):
        xs = [k for k in ks if k > 0 and method in curve[str(k)]]
        ax.plot(xs, [curve[str(k)][method]["all"] for k in xs], style, marker="o", label=method)
    best = [
        (int(k), v["best_heuristic"]) for k, v in sorted(gates.items(), key=lambda i: int(i[0]))
    ]
    ax.plot(
        [k for k, _ in best],
        [curve[str(k)][m]["all"] for k, m in best],
        "s",
        label="BEST_HEURISTIC (K=1,2,4)",
    )
    ax.axhline(curve["0"]["NO_DEPTH"]["all"], color="grey", lw=1, label="NO_EXTRA_DEPTH")
    ax.set_xlabel("depth budget K (requested candidate frames)")
    ax.set_ylabel("query AbsRel (ALL), scene mean")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(folder / "1_budget_curve.png", dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(1, len(gates), figsize=(4 * len(gates), 3.5), squeeze=False)
    for ax, (k, entry) in zip(axes[0], sorted(gates.items(), key=lambda i: int(i[0])), strict=True):
        rows = scenes[int(k)]
        gap = sorted(
            rows[entry["best_heuristic"]]["all"][s] - rows["ORACLE"]["all"][s]
            for s in rows["ORACLE"]["all"]
        )
        ax.bar(range(len(gap)), gap, color="tab:blue")
        ax.set_title(f"K={k}: {entry['best_heuristic']} - ORACLE")
        ax.set_xlabel("scene (sorted)")
    axes[0][0].set_ylabel("AbsRel gap (ALL)")
    fig.tight_layout()
    fig.savefig(folder / "2_oracle_gap_per_scene.png", dpi=150)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    positions = util["oracle_best1_positions"]
    ax.bar([int(p) for p in positions], list(positions.values()), color="tab:blue")
    ax.set_xlabel("pool index of the oracle's best single candidate (0 = earliest)")
    ax.set_ylabel("episodes")
    fig.tight_layout()
    fig.savefig(folder / "3_oracle_frame_positions.png", dpi=150)
    plt.close(fig)
    pts = util["_points"]
    for _, file, feats in (
        ("uncovered", "4_utility_vs_uncovered.png", ("uncovered",)),
        ("pose", "5_utility_vs_pose.png", ("translation_m", "rotation_deg")),
    ):
        fig, axes = plt.subplots(1, len(feats), figsize=(4.5 * len(feats), 3.5), squeeze=False)
        for ax, feat in zip(axes[0], feats, strict=True):
            ax.scatter([p[feat] for p in pts], [p["utility"] for p in pts], s=4, alpha=0.4)
            ax.set_xlabel(feat)
            ax.set_ylabel("single-step utility (ALL AbsRel reduction)")
        fig.tight_layout()
        fig.savefig(folder / file, dpi=150)
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    marg = util["_marginal"]
    keys = sorted(marg, key=int)
    ax.boxplot([marg[k] for k in keys], showfliers=False)
    ax.set_xticks(range(1, len(keys) + 1), [f"|S|={k}" for k in keys])
    ax.set_ylabel("marginal utility of one more depth frame")
    fig.tight_layout()
    fig.savefig(folder / "6_utility_by_history.png", dpi=150)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(5, 4))
    xs = [
        p["far_coverage_gain"]
        for p in pts
        if p["far_coverage_gain"] is not None and p["far_improvement"] is not None
    ]
    ys = [
        p["far_improvement"]
        for p in pts
        if p["far_coverage_gain"] is not None and p["far_improvement"] is not None
    ]
    ax.scatter(xs, ys, s=4, alpha=0.4)
    ax.set_xlabel("FAR coverage gain of one candidate")
    ax.set_ylabel("FAR AbsRel improvement")
    fig.tight_layout()
    fig.savefig(folder / "7_far_coverage_vs_improvement.png", dpi=150)
    plt.close(fig)


def analyze_cohort(cohort):
    episodes_ = load(cohort)
    scenes = scene_table(episodes_)
    deployable = tuple(h.DEPLOYABLE)
    pool_min = min(len(m["pool"]) for m in episodes_.values())
    budgets = tuple(k for k in PRIMARY_BUDGETS if k <= pool_min)
    gates = headroom(scenes, budgets, deployable)
    util = utility_structure(episodes_)
    curve = budget_curve({k: v for k, v in scenes.items() if k <= pool_min})
    replay = [d for m in episodes_.values() for d in m["replay_k0"].values()]
    integrity = {
        "episodes": len(episodes_),
        "scenes": len({m["scene_id"] for m in episodes_.values()}),
        "selector_audits_ok": all(m["selector_audits_ok"] for m in episodes_.values()),
        "teacher_purposes": sorted(
            {(t[0], tuple(t[1])) for m in episodes_.values() for t in m["teacher_reads"]}, key=str
        ),
        "replay_k0_checked": len(replay),
        "replay_k0_max_abs_diff": max((d["max_abs_diff_harmonic"] for d in replay), default=None),
        "replay_k0_near_equal": all(d["near_equal"] for d in replay) if replay else None,
    }
    return episodes_, scenes, gates, util, curve, integrity


def analyze():
    verify_lock()
    results = {}
    for cohort in COHORTS:
        if (cohort == "DEV") != REHEARSAL or not (OUT / "raw" / cohort / "_done.json").exists():
            continue
        episodes_, scenes, gates, util, curve, integrity = analyze_cohort(cohort)
        folder = OUT / "figures" / cohort
        figures(curve, scenes, util, gates, folder)
        results[cohort] = {
            "use": COHORTS[cohort][3],
            "gates": gates,
            "budget_curve": curve,
            "utility": {k: v for k, v in util.items() if not k.startswith("_")},
            "integrity": integrity,
        }
    primary = results.get(PRIMARY)
    if primary is None:
        raise FileNotFoundError("The primary cohort has not been computed")
    supported = [k for k, v in primary["gates"].items() if v["headroom_supported"]]
    status = "SUPPORTED" if supported else "NOT_ESTABLISHED"
    write(
        OUT / "oracle_results.json",
        {
            c: {
                k: {
                    "ORACLE": r["budget_curve"][k].get("ORACLE"),
                    "XORACLE": r["budget_curve"][k].get("XORACLE"),
                }
                for k in r["budget_curve"]
            }
            for c, r in results.items()
        },
    )
    write(
        OUT / "heuristic_results.json",
        {
            c: {
                k: {m: v for m, v in r["budget_curve"][k].items() if m not in DIAGNOSTIC}
                for k in r["budget_curve"]
            }
            for c, r in results.items()
        },
    )
    write(OUT / "budget_curve.json", {c: r["budget_curve"] for c, r in results.items()})
    write(
        OUT / "bootstrap_results.json",
        {c: {"gates": r["gates"], "utility": r["utility"]} for c, r in results.items()},
    )
    summary = final_summary(results, status, supported)
    write(OUT / "final_summary.json", summary)
    # determinism: a second full pass must give identical numbers
    again = {c: analyze_cohort(c)[2] for c in results}
    deterministic = all(
        json.dumps(again[c], sort_keys=True) == json.dumps(results[c]["gates"], sort_keys=True)
        for c in results
    )
    integrity_ok = (
        deterministic
        and all(r["integrity"]["selector_audits_ok"] for r in results.values())
        and all(
            r["integrity"]["replay_k0_max_abs_diff"] is None
            or (
                r["integrity"]["replay_k0_max_abs_diff"] <= REPLAY_TOLERANCE_M
                and r["integrity"]["replay_k0_near_equal"]
            )
            for r in results.values()
        )
    )
    write(
        OUT / "integrity.json",
        {
            "deterministic_reanalysis": deterministic,
            "replay_tolerance_m": REPLAY_TOLERANCE_M,
            "final_integrity": "PASS" if integrity_ok else "FAIL",
            **{c: r["integrity"] for c, r in results.items()},
            "final_holdout_touched": False,
        },
    )
    print_summary(summary, integrity_ok)


def final_summary(results, status, supported):
    p = results[PRIMARY]
    curve, gates, util = p["budget_curve"], p["gates"], p["utility"]

    def val(k, m):
        entry = curve.get(str(k), {}).get(m)
        return None if entry is None else round(entry["all"], 4)

    competitive = {k: v["best_heuristic"] for k, v in gates.items()}
    return {
        "N_SCENES": p["integrity"]["scenes"],
        "N_EPISODES": p["integrity"]["episodes"],
        "N_CANDIDATES_PER_SCENE": 9,
        "BUDGETS": list(PRIMARY_BUDGETS),
        "NO_DEPTH_ABSREL": val(0, "NO_DEPTH"),
        "RANDOM_K1": val(1, "RANDOM_EXPECTED"),
        "UNIFORM_K1": val(1, "UNIFORM"),
        **{
            f"BEST_HEURISTIC_K{k}": val(k, gates[str(k)]["best_heuristic"]) for k in PRIMARY_BUDGETS
        },
        **{f"ORACLE_K{k}": val(k, "ORACLE") for k in PRIMARY_BUDGETS},
        **{f"XORACLE_K{k}": val(k, "XORACLE") for k in PRIMARY_BUDGETS},
        **{
            f"ORACLE_MINUS_BEST_HEURISTIC_K{k}": round(
                gates[str(k)]["oracle_minus_best"]["mean"], 4
            )
            for k in PRIMARY_BUDGETS
        },
        "HEADROOM_CI": {
            k: [round(x, 4) for x in gates[k]["xoracle_minus_best"]["ci95"]] for k in gates
        },
        "HEADROOM_FRACTION": {
            k: None
            if gates[k]["headroom_fraction"] is None
            else round(gates[k]["headroom_fraction"], 3)
            for k in gates
        },
        "XORACLE_GATE": {k: gates[k]["xoracle_gate_status"] for k in gates},
        "HEADROOM_STATUS": status,
        "HEADROOM_SUPPORTED_BUDGETS": supported,
        "MOST_COMPETITIVE_HEURISTIC": competitive,
        "UTILITY_SCENE_DEPENDENT": util["utility_scene_dependent"],
        "UTILITY_PREDICTABILITY_DIAGNOSTIC": f"{util['predictability_label']} (best feature "
        f"{util['best_feature']})",
        "POLICY_TRAINING_ALLOWED": status == "SUPPORTED",
        "SECONDARY": {
            c: {
                k: {
                    "best": v["best_heuristic"],
                    "fraction": v["headroom_fraction"],
                    "xoracle_gate": v["xoracle_gate_status"],
                    "supported": v["headroom_supported"],
                }
                for k, v in r["gates"].items()
            }
            for c, r in results.items()
            if c != PRIMARY
        },
        "FINAL_HOLDOUT_TOUCHED": False,
        "NEW_DATA_DOWNLOADED": False,
        "GITHUB_PUSHED": False,
        "REPORT_DIR": str(OUT),
    }


def print_summary(s, deterministic):
    tests = read(OUT / "tests.json")["status"] if (OUT / "tests.json").exists() else "NOT_RECORDED"
    lines = ["=" * 60, "MEASUREMENT PLANNING ORACLE V1 FINAL", "=" * 60, ""]
    for key in (
        "N_SCENES",
        "N_CANDIDATES_PER_SCENE",
        "BUDGETS",
        "NO_DEPTH_ABSREL",
        "RANDOM_K1",
        "UNIFORM_K1",
        "BEST_HEURISTIC_K1",
        "ORACLE_K1",
        "BEST_HEURISTIC_K2",
        "ORACLE_K2",
        "BEST_HEURISTIC_K4",
        "ORACLE_K4",
        "ORACLE_MINUS_BEST_HEURISTIC_K1",
        "ORACLE_MINUS_BEST_HEURISTIC_K2",
        "ORACLE_MINUS_BEST_HEURISTIC_K4",
        "HEADROOM_CI",
        "HEADROOM_STATUS",
        "MOST_COMPETITIVE_HEURISTIC",
        "UTILITY_SCENE_DEPENDENT",
        "UTILITY_PREDICTABILITY_DIAGNOSTIC",
        "POLICY_TRAINING_ALLOWED",
    ):
        lines.append(f"{key}={s[key]}")
    lines += [
        "",
        "FINAL_HOLDOUT_TOUCHED=false",
        "NEW_DATA_DOWNLOADED=false",
        "GITHUB_PUSHED=false",
        "",
        f"TESTS={tests}",
        f"FINAL_INTEGRITY={'PASS' if deterministic else 'FAIL'}",
        "",
        f"REPORT_DIR={s['REPORT_DIR']}",
        "=" * 60,
    ]
    text = "\n".join(lines)
    (OUT / "final_summary.txt").write_text(text + "\n")
    if REHEARSAL:
        print(f"REHEARSAL_ANALYSIS_DONE (numbers withheld) integrity={deterministic}", flush=True)
        return
    print(text, flush=True)


def record_tests():
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "tests/test_measurement_planning_oracle_v1.py"],
        cwd=REPO,
        capture_output=True,
        text=True,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
    )
    write(
        OUT / "tests.json",
        {
            "status": "PASS" if result.returncode == 0 else "FAIL",
            "summary": result.stdout.strip().splitlines()[-1:],
            "returncode": result.returncode,
        },
    )
    print(result.stdout[-2000:], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("lock", "tests", "compute", "analyze"))
    parser.add_argument("--cohort", choices=tuple(COHORTS))
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    if args.stage == "lock":
        lock()
    elif args.stage == "tests":
        record_tests()
    elif args.stage == "compute":
        compute(args.cohort, args.workers)
    else:
        analyze()


if __name__ == "__main__":
    main()
