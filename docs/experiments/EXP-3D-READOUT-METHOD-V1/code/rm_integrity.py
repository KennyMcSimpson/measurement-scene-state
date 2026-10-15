"""EXP-3D-READOUT-METHOD-V1: integrity report (standard library only, read-only except INTEGRITY_REPORT.json).

Checks, against the locks and logs written before the results: the locked code, caches and KF table are unchanged;
the analysis code hash was logged before any evaluation result; both analyses replayed byte-identically; the run logs
finished with the expected ground-truth reads; the amendment lock is intact; no weights are stored; no code touches a
protected holdout or downloads data; which files were added after the preregistration lock (and why).
  python3 rm_integrity.py ROOT
"""

import hashlib
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(sys.argv[1]).resolve()
KF_TABLE = ROOT.parent / "EXP-3D-CAJU3D-KFUSE-V1" / "fusion_tables.json"
POST_LOCK = {
    "rm_stats.py": "statistics; written after the preregistration lock, hash logged before any evaluation result "
                   "(audit/stats_code_locked_at.txt)",
    "rm_runtime.py": "runtime profile (no ground truth); first attempt failed in the DepthSplat environment's R7 "
                     "path (audit/runtime.log), rerun as two parts + merge",
    "rm_amend1.py": "Amendment 1 (post hoc, after the Gate A / B results), locked in AMENDMENT_1.lock.json before its run",
    "rm_amend1_stats.py": "Amendment 1 statistics, locked with it",
    "rm_amend2.py": "Amendment 2 (post hoc, Candidate B oracle premise test, CPU), locked in AMENDMENT_2.lock.json",
    "rm_amend3.py": "Amendment 3 (post hoc, Candidate A free-space premise test, CPU), locked in AMENDMENT_3.lock.json",
    "rm_integrity.py": "this report",
}
REQUIRED = ("README.md", "STATUS.md", "CODE_AUDIT.md", "DATA_EXPOSURE_REGISTRY.md", "MODEL_LINEAGE.md", "PROTOCOL.md",
            "PREREGISTRATION.json", "BASELINE_MATRIX.csv", "CALIBRATION_CONTROLS.csv", "TTT_NECESSITY_RESULTS.csv",
            "REGIONAL_ERROR_BREAKDOWN.csv", "BOOTSTRAP_RESULTS.json", "RUNTIME_PROFILE.json",
            "NOVELTY_AND_RELATED_WORK.md", "NEXT_METHOD_PROPOSAL.md", "INTEGRITY_REPORT.json", "commands.sh")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def last_line(path, token):
    lines = [l for l in Path(path).read_text().splitlines() if token in l] if Path(path).is_file() else []
    return lines[-1] if lines else None


def main():
    prereg = json.loads((ROOT / "PREREGISTRATION.json").read_text())
    rep = {"experiment": "EXP-3D-READOUT-METHOD-V1", "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
           "preregistration": {"locked_at": prereg["locked_at"], "sha256": sha(ROOT / "PREREGISTRATION.json"),
                               "locked_at_record": (ROOT / "audit/locked_at.txt").read_text().strip()}}
    rep["locked_code_unchanged"] = {n: sha(ROOT / "code" / n) == h for n, h in prereg["code"].items()}
    rep["locked_caches_unchanged"] = {n: sha(ROOT / "cache" / n) == h for n, h in prereg["caches"].items()}
    rep["kf_table_unchanged"] = sha(KF_TABLE) == prereg["kf_table_sha256"]
    rep["protocol_unchanged"] = sha(ROOT / "PROTOCOL.md") == prereg["protocol_sha256"] == sha(
        ROOT / "audit/PROTOCOL.locked.md")
    stats_rec = (ROOT / "audit/stats_code_locked_at.txt").read_text()
    rep["stats_code"] = {"logged_before_results": "results/phase12 has: )" in stats_rec,
                         "unchanged_since_logged": sha(ROOT / "code/rm_stats.py") in stats_rec}
    rep["analysis_replay"] = {"phase12": (ROOT / "audit/analysis_determinism.txt").read_text().strip(),
                              "amendment1": ((ROOT / "audit/amend1_determinism.txt").read_text().strip()
                                             if (ROOT / "audit/amend1_determinism.txt").is_file() else "NOT_RUN")}
    rep["runs"] = {"phase12": last_line(ROOT / "audit/run.log", "RM_RUN_DONE"),
                   "phase12_exit": last_line(ROOT / "audit/run.log", "RM_RUN_EXIT"),
                   "amendment1": last_line(ROOT / "audit/amend1_run.log", "RM_AMEND1_DONE"),
                   "amendment1_exit": last_line(ROOT / "audit/amend1_run.log", "AMEND1_EXIT"),
                   "runtime_gpu_exit": last_line(ROOT / "audit/runtime_gpu.log", "RUNTIME_GPU_EXIT"),
                   "runtime_first_attempt_exit": last_line(ROOT / "audit/runtime.log", "RUNTIME_EXIT")}
    for k in (1, 2, 3):
        lk = json.loads((ROOT / f"AMENDMENT_{k}.lock.json").read_text())
        res = ROOT / f"AMENDMENT_{k}_RESULTS.json"
        rep[f"amendment{k}_lock"] = {
            "locked_at": lk["locked_at"], "files_unchanged": {n: sha(ROOT / n) == h for n, h in lk["files"].items()},
            "lock_record": (ROOT / f"audit/amendment{k}_locked_at.txt").read_text().strip(),
            "label": json.loads(res.read_text())["primary"]["label"] if res.is_file() else "NOT_RUN"}
    rep["runs"]["amendment2_exit"] = last_line(ROOT / "audit/amend2_run.log", "AMEND2_EXIT")
    rep["runs"]["amendment3_exit"] = last_line(ROOT / "audit/amend3_run.log", "AMEND3_EXIT")
    kf = json.loads((ROOT / "results/amend1/kf_fit.json").read_text()) if (
        ROOT / "results/amend1/kf_fit.json").is_file() else None
    rep["amendment1_kf_replay"] = kf["replay_of_frozen_ttt_table"] if kf else "NOT_RUN"
    a1res = json.loads((ROOT / "AMENDMENT_1_RESULTS.json").read_text()) if (
        ROOT / "AMENDMENT_1_RESULTS.json").is_file() else None
    rep["amendment1_b1_replay_max_abs_difference"] = a1res["b1_replay_max_abs_difference"] if a1res else "NOT_RUN"
    rt = json.loads((ROOT / "RUNTIME_PROFILE.json").read_text())
    rep["runtime_r7_check"] = rt["r7_check"]
    code = {p.name: p.read_text() for p in sorted((ROOT / "code").glob("*.py"))}
    rep["post_lock_code"] = {n: POST_LOCK.get(n, "UNEXPECTED") for n in code if n not in prereg["code"]}
    # a path-like string into Hypersim or a final holdout, or a download call (prose mentions do not count)
    pattern = re.compile(r"[\"'][^\"'\n]*(hypersim|final_holdout)[^\"'\n]*/[^\"'\n]*[\"']|\b(wget|curl)\s|urlretrieve|"
                         r"urlopen|requests\.get|snapshot_download|hf_hub_download", re.I)
    rep["protected_holdout_or_download_in_code"] = sorted(n for n, t in code.items()
                                                          if pattern.search(t) and n != "rm_integrity.py")
    rep["weights_stored"] = sorted(str(p.relative_to(ROOT)) for ext in ("*.pt", "*.pth", "*.ckpt")
                                   for p in ROOT.rglob(ext))
    rep["fresh_data_used"] = prereg["fresh_data_used"]
    rep["final_holdout_touched"] = prereg["final_holdout_touched"]
    rep["deliverables"] = {n: (ROOT / n).is_file() or n == "INTEGRITY_REPORT.json" for n in REQUIRED}
    ok = (all(rep["locked_code_unchanged"].values()) and all(rep["locked_caches_unchanged"].values())
          and rep["kf_table_unchanged"] and rep["protocol_unchanged"] and all(rep["stats_code"].values())
          and rep["analysis_replay"]["phase12"] == "ANALYSIS_DETERMINISTIC"
          and all(all(rep[f"amendment{k}_lock"]["files_unchanged"].values()) for k in (1, 2, 3))
          and "UNEXPECTED" not in rep["post_lock_code"].values()
          and not rep["protected_holdout_or_download_in_code"] and not rep["weights_stored"])
    rep["status"] = "PASS" if ok else "CHECK_FAILED"
    (ROOT / "INTEGRITY_REPORT.json").write_text(json.dumps(rep, indent=1, sort_keys=True) + "\n")
    print("INTEGRITY", rep["status"])


if __name__ == "__main__":
    main()
