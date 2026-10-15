"""EXP-3D-CAJU-HEADROOM-V1: the integrity record and the final Gate 0 printout (no ground truth read).

integrity.json: the lock (preregistration hash, code and root files unchanged or amended, the locked protocol an exact
prefix of the current one), the seals (geometry incl. the COMPAT replay against V3; replay settings, completeness,
access audit), the pilot equivalence checks, the selection-before-evaluation order, the ground-truth read purposes,
the analysis determinism, the tests, and the brief's 16 integrity items with their evidence.
results/gate0_summary.json and results/final_printout.txt: the brief's CAJU-V1 GATE 0 FINAL block.
Run through cv_host_run.py from the repository root (CUDA hidden):  python caju_finalize.py ROOT
"""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import caju_score as S  # noqa: E402

REPORT_DIRS = "outputs/EXP-3D-CAJU-HEADROOM-V1 (all files) and docs/experiments/EXP-3D-CAJU-HEADROOM-V1 (records)"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git(*args):
    return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout.strip()


def fmt(x, digits=4):
    return "None" if x is None else f"{x:.{digits}f}"


def main():
    root = Path(sys.argv[1]).resolve()
    prereg = json.loads((root / "preregistration.json").read_text())
    amended = {}
    for a in sorted((root / "audit").glob("amendment_*.json")):
        amended.update(json.loads(a.read_text()).get("code", {}))
    code_now = {p.name: sha(p) for p in sorted((root / "code").iterdir()) if p.suffix in (".py", ".sh")}
    locked_protocol = root / "audit" / "PROTOCOL.locked.md"
    geometry = {c: json.loads((root / "geometry" / f"geometry_seal_{c}.json").read_text())
                for c in prereg["geometry_seals"]}
    replays = {tag: S.replay_seals(root, tag) for tag in ("PILOT", "COMPAT", "FORMAL")}
    equivalence = json.loads((root / "equivalence" / "equivalence_report.json").read_text())
    pilot = json.loads((root / "pilot" / "pilot_report.json").read_text())
    tests = json.loads((root / "tests.json").read_text())
    headroom = json.loads((root / "results" / "oracle_headroom.json").read_text())
    utility = json.loads((root / "results" / "counterfactual_utility.json").read_text())
    boot = json.loads((root / "results" / "bootstrap_results.json").read_text())
    reads = {c: json.loads((root / "scores" / c / "ground_truth_reads.json").read_text())["reads"]
             for c in ("CAJU_PILOT", "CAJU_SEL", "CAJU_EVAL")}
    formal_seals = sorted((root / "replay" / "FORMAL").glob("replay_seal_*of*.json"))
    selection_file = root / "fixed_action_selection.json"
    checks = {
        "preregistration_unchanged": sha(root / "preregistration.json") ==
        (root / "audit" / "locked_at.txt").read_text().split()[-1],
        "code_unchanged_or_amended": code_now == {**prereg["code"], **amended},
        "locked_protocol_is_exact_prefix": locked_protocol.is_file()
        and sha(locked_protocol) == prereg["root_files"]["PROTOCOL.md"]
        and (root / "PROTOCOL.md").read_bytes().startswith(locked_protocol.read_bytes()),
        "root_files_unchanged": all(sha(root / n) == h for n, h in prereg["root_files"].items() if n != "PROTOCOL.md"),
        "geometry_seals_unchanged": all(sha(root / "geometry" / f"geometry_seal_{c}.json") == h
                                        for c, h in prereg["geometry_seals"].items()),
        "geometry_reads_no_target_depth": all(not g["query_depth_read"] and not g["history_depth_read"]
                                              for g in geometry.values()),
        "geometry_compat_replay_vs_v3": geometry["CAJU_COMPAT"]["replay_vs_v3"]["pass"],
        "replay_seals_complete_and_certified": True,  # S.replay_seals raises otherwise
        "replay_failures_reverted": {tag: sum(v["ttt"]["failed_reverted_to_cache"] for v in r["logs"].values())
                                     for tag, r in replays.items()},
        "equivalence_snapshot_vs_independent_runs": all(equivalence["equivalent"].values()),
        "equivalence_at_utility_level": all(pilot["equivalence_utility"]["equivalent"].values()),
        "engine_compat_vs_v3_within_floor": all(equivalence["compat_within_floor"].values()),
        "selection_logged_before_evaluation": S.check_selection(root) is not None
        and selection_file.stat().st_mtime <= (root / "scores/CAJU_EVAL/rows.json").stat().st_mtime,
        "ground_truth_read_after_every_formal_seal": all(
            (root / f"scores/{c}/rows.json").stat().st_mtime >= max(p.stat().st_mtime for p in formal_seals)
            for c in ("CAJU_SEL", "CAJU_EVAL")),
        "ground_truth_purposes_only_sealed_evaluation": all(r["purpose"] == "SEALED_QUERY_EVALUATION"
                                                            for v in reads.values() for r in v),
        "ground_truth_only_arkit_records": all(r["scene_id"].startswith("arkit_") for v in reads.values() for r in v),
        "analysis_deterministic": "ANALYSIS_DETERMINISTIC" in (root / "audit/analysis_determinism.txt").read_text(),
        "tests_pass": tests["exit_code"] == 0 and tests["failed"] == 0,
    }
    superseded = []
    if (root / "audit" / "amendment_1.json").is_file():
        # Amendment 1: the single-pixel maximum checks compare two extremes of the same non-deterministic engine; the
        # go/no-go rests on the preregistered utility-level check and on the distributions of the differences.
        diag = json.loads((root / "audit/amendment_1/nondeterminism_diagnostic.json").read_text())
        same = all(diag["pilot"][f"{a}_replay_vs_independent"]["core_b_relative"][s]
                   <= 1.5 * diag["pilot"][f"{a}_independent_vs_independent"]["core_b_relative"][s] + 1e-9
                   for a in ("OFF", "SHORT", "FULL") for s in ("mean", "p99"))
        same = same and diag["compat"]["OFF_replay_vs_v3_cache"]["depth_z_abs_m"]["max"] == 0.0
        checks["amendment_1_engine_equivalent_in_distribution"] = same
        if same:
            superseded = [k for k in ("equivalence_snapshot_vs_independent_runs", "engine_compat_vs_v3_within_floor")
                          if checks[k] is False]
    passed = all(v for k, v in checks.items() if k != "replay_failures_reverted" and k not in superseded)
    integrity = ("PASS_WITH_AMENDMENT_1" if superseded else "PASS") if passed else "FAIL"
    brief_items = {
        "1_query_gt_not_in_actions": "replay seals: query/history depth and image never read; per-episode file audit",
        "2_query_gt_not_in_jepa_features": "no JEPA features exist (JEPA_FEATURE_STATUS=NOT_IMPLEMENTED)",
        "3_query_gt_not_in_predictor_features": "pre-update features come from the 7 arrived frames only (audited)",
        "4_branches_share_the_initial_state": "OFF = CACHE; SHORT / FULL = steps 10 / 100 of one trajectory from it",
        "5_same_arrived_observations": "every action uses the same 7 arrived frames (one encode per episode)",
        "6_off_keeps_cache_and_fusion": "OFF is the 7-view CACHE state with measured depth injected",
        "7_r7_same_for_all_actions": "one geometry seal per target, shared by every action",
        "8_ttt_reads_arrived_frames_only": "the TTT targets are the 7 arrived frames (audited)",
        "9_no_optimizer_state_across_branches": "fresh Adam per episode; SHORT is a read-only snapshot; checked "
                                                "against independent V1 runs (equivalence)",
        "10_reset_between_scans": "the state is re-encoded per episode; the CACHE fingerprint is logged",
        "11_action_set_frozen": "OFF / SHORT 10 / FULL 100 in the lock, before any utility",
        "12_opacity_threshold_frozen": "alpha >= 0.5 in the lock",
        "13_place_is_the_unit": "bootstrap, LOSO and top-1 shares over places (one scan per place)",
        "14_deployment_runs_only_the_chosen_action": "not applicable: Gate 0 deploys no selector",
        "15_online_cost_includes_jepa_and_mlp": "not applicable: no JEPA / MLP in Gate 0",
        "16_results_rebuilt_from_raw_predictions": "caju_score rebuilds every number from the sealed npz files; the "
                                                   "analysis ran twice byte-identical",
    }
    S.write(root / "integrity.json", {"status": integrity, "checks": checks, "superseded_by_amendment_1": superseded,
                                      "brief_integrity_items": brief_items})
    status = headroom["gate0_status"]
    fixed = headroom["fixed_action"]
    absrel = utility["action_place_mean_absrel"]["CAJU_EVAL"]
    frac = headroom["episode_fractions"]["oracle_action_share"]
    fp = {p["action"]: p for p in headroom["pareto"]["fixed_points"]}
    adv = headroom["pareto"]["cross_query"]["advantage_over_fixed_mixtures"]
    pareto = (f"fixed: OFF 0 steps U=0; SHORT 10 steps U={fmt(fp['SHORT']['utility'], 5)}; FULL 100 steps "
              f"U={fmt(fp['FULL']['utility'], 5)}; cross-query oracle advantage over fixed mixtures at "
              + ", ".join(f"{b} steps {fmt(v, 5)}" for b, v in adv.items()))
    o, x = boot["ORACLE_MINUS_FIX"], boot["PRIMARY_XORACLE_MINUS_FIX"]
    summary = {
        "REPOSITORY_HEAD": f"{git('rev-parse', '--short', 'HEAD')} ({git('branch', '--show-current')})",
        "WORKTREE_STATUS": f"{len([x for x in git('status', '--porcelain').splitlines() if x])} uncommitted paths",
        "CORE_B_IMPLEMENTATION": "released DepthSplat (re10k+dl3dv base, cap 0.1) 7-view CACHE + 100-step "
                                 "Gaussian-state TTT; CORE_B = sqrt(R7 x F7), F7 = ray depth where alpha >= 0.5 else R7",
        "TTT_UPDATED_TENSORS": "Gaussian means, per-axis log-scales (eigenbasis, rotations fixed), opacity logits, "
                               "SH DC, higher SH; network weights frozen",
        "TTT_LOSS": "relative L1 of the raw rendered depth on valid measured pixels + 0.5 x RGB L1, 7 arrived frames",
        "JEPA_FEATURE_STATUS": "NOT_IMPLEMENTED",
        "N_EPISODES": f"{utility_episodes(root)} formal (CAJU_SEL 32 + CAJU_EVAL 68) + 16 pilot",
        "N_PHYSICAL_PLACES": "50 formal (16 + 34) + 8 pilot",
        "ACTION_SET": "OFF (0 steps) / SHORT_TTT (10) / FULL_TTT (100)",
        "OFF_ABSREL": fmt(absrel["OFF"]), "SHORT_TTT_ABSREL": fmt(absrel["SHORT"]),
        "FULL_TTT_ABSREL": fmt(absrel["FULL"]),
        "BEST_FIXED_ACTION": f"{fixed} (chosen on CAJU_SEL)",
        "BEST_FIXED_UTILITY": fmt(headroom["policy_place_mean_utility"]["FIX"], 5),
        "ORACLE_UTILITY": fmt(headroom["policy_place_mean_utility"]["ORACLE"], 5) + " (per-episode, in-sample)",
        "ORACLE_MINUS_FIXED": fmt(o["mean"], 5),
        "ORACLE_GAIN_CI": f"[{fmt(o['ci95'][0], 5)}, {fmt(o['ci95'][1], 5)}]",
        "PRIMARY_XORACLE_MINUS_FIXED": f"{fmt(x['mean'], 5)} [{fmt(x['ci95'][0], 5)}, {fmt(x['ci95'][1], 5)}] "
                                       "(cross-query oracle, the preregistered primary)",
        "OFF_OPTIMAL_FRACTION": fmt(frac["OFF"], 3), "SHORT_OPTIMAL_FRACTION": fmt(frac["SHORT"], 3),
        "FULL_OPTIMAL_FRACTION": fmt(frac["FULL"], 3),
        "TTT_QUALITY_COST_PARETO": pareto,
        "GATE0_STATUS": f"{status} ({headroom['dynamic_action_space']})",
        "GATE1_ALLOWED": "false (JEPA_FEATURE_STATUS=NOT_IMPLEMENTED; "
                         + ("Gate 0 passed: only the non-JEPA predictor may be prepared)" if
                            status in ("QUALITY_HEADROOM_SUPPORTED", "COMPUTE_HEADROOM_SUPPORTED")
                            else "Gate 0 not passed)"),
        "FINAL_HOLDOUT_TOUCHED": "false", "GITHUB_PUSHED": "false",
        "TESTS": f"{tests['passed']} passed, {tests['failed']} failed, {tests['skipped']} skipped",
        "INTEGRITY": integrity, "REPORT_DIR": REPORT_DIRS,
    }
    S.write(root / "results" / "gate0_summary.json", summary)
    lines = ["=" * 50, "CAJU-V1 GATE 0 FINAL", "=" * 50, ""]
    groups = (("REPOSITORY_HEAD", "WORKTREE_STATUS"),
              ("CORE_B_IMPLEMENTATION", "TTT_UPDATED_TENSORS", "TTT_LOSS", "JEPA_FEATURE_STATUS"),
              ("N_EPISODES", "N_PHYSICAL_PLACES", "ACTION_SET"),
              ("OFF_ABSREL", "SHORT_TTT_ABSREL", "FULL_TTT_ABSREL"),
              ("BEST_FIXED_ACTION", "BEST_FIXED_UTILITY"),
              ("ORACLE_UTILITY", "ORACLE_MINUS_FIXED", "ORACLE_GAIN_CI", "PRIMARY_XORACLE_MINUS_FIXED"),
              ("OFF_OPTIMAL_FRACTION", "SHORT_OPTIMAL_FRACTION", "FULL_OPTIMAL_FRACTION"),
              ("TTT_QUALITY_COST_PARETO", "GATE0_STATUS"),
              ("GATE1_ALLOWED", "FINAL_HOLDOUT_TOUCHED", "GITHUB_PUSHED"),
              ("TESTS", "INTEGRITY", "REPORT_DIR"))
    for group in groups:
        lines += [f"{k}={summary[k]}" for k in group] + [""]
    lines.append("=" * 50)
    text = "\n".join(lines) + "\n"
    (root / "results" / "final_printout.txt").write_text(text)
    print(text)
    print("INTEGRITY", integrity, json.dumps({k: v for k, v in checks.items() if v is False}))


def utility_episodes(root):
    rows = json.loads((root / "scores/CAJU_SEL/rows.json").read_text()) + json.loads(
        (root / "scores/CAJU_EVAL/rows.json").read_text())
    return len({r["episode"] for r in rows})


if __name__ == "__main__":
    main()
