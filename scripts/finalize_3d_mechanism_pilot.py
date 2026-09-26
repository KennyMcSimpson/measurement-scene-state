"""Assemble honest blocked scientific deliverables; never invent missing results."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NAME = "EXP-3D-20260927-state-history-pilot-v1"
DOC = ROOT / "docs/experiments" / NAME
OUT = ROOT / "outputs" / NAME


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def main():
    audit = json.loads((DOC / "carrier_audit.json").read_text())
    blocked = {
        "status": "NOT_EVALUATED_CARRIER_BLOCKED",
        "scientific_scene_count": 0,
        "reason": "No qualified dynamic checkpoint and no calibrated 3D cohort.",
        "metrics": None,
        "synthetic_results": "smoke/ only; not scientific evidence",
    }
    for name in [
        "static_state_results",
        "controlled_history_results",
        "natural_history_results",
        "policy_branch_rewards",
        "policy_decisions",
    ]:
        write(OUT / "raw" / (name + ".json"), dict(blocked, rows=[]))
    for name in [
        "static_state_analysis",
        "history_interaction_analysis",
        "action_oracle_analysis",
        "bootstrap_results",
        "robustness_analysis",
        "cost_analysis",
    ]:
        write(OUT / "analysis" / (name + ".json"), blocked)
    write(OUT / "analysis/carrier_analysis.json", audit)
    write(
        OUT / "analysis/policy_analysis.json",
        dict(
            blocked,
            POLICY_STAGE_SKIPPED=True,
            FEEDBACK_IDENTIFIABILITY_STATUS="SKIPPED",
            training_runs=0,
        ),
    )
    for name in ["information_contract.json"]:
        (OUT / "audit" / name).write_bytes((DOC / name).read_bytes())
    for name in ["label_access_log", "sealed_query_access_log"]:
        write(
            OUT / "audit" / (name + ".json"),
            {
                "scope": "FORMAL_SCIENTIFIC",
                "events": [],
                "reads": 0,
                "synthetic_access_log": "../smoke/access_log.json",
            },
        )
    write(OUT / "locks/carrier_lock.json", audit)
    write(OUT / "locks/metric_lock.json", json.loads((DOC / "preregistration.json").read_text()))
    write(
        OUT / "locks/policy_lock.json",
        {
            "POLICY_STAGE_SKIPPED": True,
            "reason": "Scientific Phase B unavailable",
            "trained_models": [],
        },
    )
    inputs = ["src/mcss/mechanism_pilot", "scripts", "tests"]
    files = [
        p
        for folder in inputs
        for p in (ROOT / folder).glob("*.py")
        if folder == "src/mcss/mechanism_pilot" or "mechanism" in p.name
    ]
    write(
        OUT / "locks/implementation_lock.json",
        {
            "hashes": {
                str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(files)
            }
        },
    )
    status = {
        "CARRIER_STATUS": "BLOCKED",
        "STATIC_STATE_STATUS": "BLOCKED_CARRIER_NOT_READY",
        "WRITE_OPPORTUNITY_STATUS": "NOT_ESTABLISHED",
        "HISTORY_ACTION_STATUS": "NOT_ESTABLISHED",
        "EVALUATION_STATUS": "NOT_EVALUATED_CARRIER_BLOCKED",
        "status_interpretation": "Not evaluated, not a negative experimental finding.",
        "FEEDBACK_IDENTIFIABILITY_STATUS": "SKIPPED",
        "POLICY_STAGE_SKIPPED": True,
        "N_TRAIN_SCENES": 0,
        "N_DEV_SCENES": 0,
        "N_EVAL_SCENES": 0,
        "2D_BRANCH_STATUS": "SEALED_PENDING_INDEPENDENT_CONFIRMATION",
    }
    for key in [
        "STATIC_R_FIXED",
        "STATIC_R_RESIDUAL",
        "STATIC_ANCHOR",
        "CROSS_SCENE_REPLACEMENT_DELTA",
        "BEST_FIXED_WRITE",
        "WRITE_ORACLE_GAIN",
        "BENEFICIAL_ACTION_FLIPS",
        "HISTORY_INTERACTION_MEAN",
        "HISTORY_INTERACTION_CI",
        "GLOBAL_ACTION_GAP",
        "STATE_DEPENDENT_ACTION_GAP",
        "POLICY_P0_GAIN",
        "POLICY_P1_GAIN",
        "POLICY_P2_GAIN",
        "POLICY_REGRET",
    ]:
        status[key] = None
    status["CONTROLLED_HISTORY_PAIRS"] = 0
    status.update(SEALED_QUERY_LEAKAGE=False, TEST_DEPTH_IN_POLICY=False)
    # Replay existing discovery-only comparator without modifying its old output directory.
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "fair2d", ROOT / "scripts/explore_v2_discovery.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    source = ROOT / "docs/experiments/EXP-2D-opportunity-selector-v2-20260926T015728+0800"
    rows = [json.loads(x) for x in (source / "discovery_rows.jsonl").read_text().splitlines()]
    decisions = mod.crossfit_global(rows)
    old = json.loads(
        (ROOT / "docs/experiments/EXP-2D-followup-20260927/discovery_diagnostics.json").read_text()
    )
    assert decisions == old["LOSO_global_decisions"]
    result = {
        "POSTHOC_FAIR_BASELINE": True,
        "scope": "discovery only; not preregistered",
        "2D_BRANCH_STATUS": status["2D_BRANCH_STATUS"],
        "decisions": decisions,
        "methods": old["methods"],
        "comparisons": old["comparisons"],
        "replayed_decisions_identical": True,
        "media_reads": 0,
        "source_hashes": old["source_hashes"],
    }
    write(OUT / "analysis/2d_outer_fold_fixed16.json", result)
    for key, method in [
        ("2D_OUTER_FOLD_FIXED16", "LOSO_global16"),
        ("2D_GATEONLY_OOF", "GateOnly"),
        ("2D_CYCLEGATE_OOF", "CycleGate"),
    ]:
        summary = old["methods"][method]
        status[key] = sum(x["J"] for x in summary["per_sequence"]) / len(summary["per_sequence"])
    tests_file = OUT / "audit/tests.json"
    tests = json.loads(tests_file.read_text()) if tests_file.exists() else {"status": "PENDING"}
    status["TESTS"] = tests
    initial = json.loads((DOC / "implementation_audit.json").read_text())["initial_hashes"]
    changed = [
        name
        for name, digest in initial.items()
        if not (ROOT / name).is_file()
        or hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest
    ]
    smoke = OUT / "smoke/summary.json"
    status["CANDIDATE_STATE_ISOLATION"] = (
        json.loads(smoke.read_text()).get("candidate_isolation_verified", False)
        if smoke.exists()
        else False
    )
    integrity = {
        "status": "PASS"
        if not changed and tests.get("status") == "PASS" and smoke.exists()
        else "INCOMPLETE",
        "initial_file_checks": len(initial),
        "changed_preexisting_files": changed,
        "scientific_data_reads": 0,
        "formal_results_fabricated": False,
        "synthetic_separate": True,
        "tests": tests.get("status"),
        "interpretation": "Artifact/engineering integrity only; carrier remains blocked.",
    }
    write(OUT / "audit/integrity.json", integrity)
    status["FINAL_INTEGRITY"] = integrity["status"]
    status["REPORT_DIR"] = str(DOC)
    write(DOC / "terminal_summary.json", status)
    text = (
        "3D REVERSE-JEPA + DYNAMIC TTT MECHANISM PILOT\n"
        + "\n".join(f"{k}=" + ("NA" if v is None else json.dumps(v)) for k, v in status.items())
        + "\n"
    )
    (DOC / "terminal_summary.txt").write_text(text)
    (DOC / "STATUS.md").write_text("# Status\n\n```text\n" + text + "```\n")
    print(text)


if __name__ == "__main__":
    main()
