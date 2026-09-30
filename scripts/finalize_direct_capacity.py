#!/usr/bin/env python3
"""Archive a completed capacity attribution without publishing checkpoints or old data.

Run only after the report, full tests, postrun audit and statistics reproduction finish.
No media decode, model inference, git commit/push or old-experiment mutation occurs.
"""

from __future__ import annotations

import argparse
import gzip
import json
import shutil
import tarfile
from pathlib import Path

from mcss.mechanism_pilot.small_training import sha, write_json

EXPERIMENT = "EXP-3D-DIRECT-STATE-CAPACITY-V1"
PASS_RECORDS = (
    "baseline_integrity.json",
    "baseline_carrier_reconstruction.json",
    "context_optimization_complete.json",
    "oracle_optimization_complete.json",
    "raw/context_evaluation_integrity.json",
    "raw/oracle_evaluation_integrity.json",
    "audit/postrun_validation.json",
    "audit/preservation_recheck.json",
)
REQUIRED = (
    "README.md",
    "STATUS.md",
    "commands.sh",
    "git_commit.txt",
    "dirty.patch",
    "preregistration.json",
    "config.json",
    "scene_manifest.json",
    "state_plan.json",
    "capacity_contract.json",
    "optimization_contract.json",
    "renderer_contract.json",
    "resolution_contract.json",
    "direct_state_results.json",
    "capacity_analysis.json",
    "resolution_analysis.json",
    "renderer_capacity_analysis.json",
    "optimization_curves.json",
    "bootstrap_results.json",
    "statistics_reproduction.json",
    "tests.json",
)


def read(path):
    return json.loads(path.read_text())


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def check_hashes(mapping):
    for name, digest in mapping.items():
        path = Path(name)
        require(path.is_file() and sha(path) == digest, f"Frozen hash mismatch: {path}")


def validate(root, docs):
    require(
        root.name == EXPERIMENT and docs.name == EXPERIMENT,
        "Only this new experiment output/docs directory may be finalized",
    )
    require(
        root != docs and root not in docs.parents and docs not in root.parents,
        "Output and publication directories must be separate",
    )
    for name in REQUIRED + PASS_RECORDS:
        require((root / name).is_file(), f"Required completed artifact missing: {name}")
    for name in PASS_RECORDS:
        require(read(root / name).get("status") == "PASS", f"Audit not PASS: {name}")
    post = read(root / "audit/postrun_validation.json")
    require(post.get("optimized_state_count") == 425, "All 425 planned states must be audited")
    baseline = read(root / "baseline_integrity.json")
    require(
        baseline["rows"] == 136 and baseline["max_metric_absolute_delta"] <= 1e-6,
        "Frozen carrier baseline reproduction incomplete",
    )
    reconstruction = read(root / "baseline_carrier_reconstruction.json")
    require(
        reconstruction.get("reconstructed_states") == 51
        and all(row["exact"] for row in reconstruction["rows"]),
        "Carrier reconstruction not exact",
    )
    for name in PASS_RECORDS:
        record = read(root / name)
        for forbidden in ("final_holdout_touched", "dynamic_ttt_run", "new_carrier_trained"):
            require(record.get(forbidden, False) is False, f"Forbidden action in {name}")
    tests = read(root / "tests.json")
    full = tests.get("full_suite", tests)
    require(isinstance(full, dict), "Full-suite test evidence must be structured")
    passed = full.get("passed", tests.get("passed", 0))
    require(isinstance(passed, int) and passed > 0, "Positive full-suite passed count required")
    for record in (tests, full):
        require(str(record.get("status", "PASS")).upper() in ("PASS", "PASSED"), "Tests not PASS")
        for field in ("failed", "failures", "errors", "error"):
            require(record.get(field, 0) in (0, None, False), "Test failures/errors remain")
    statistics = read(root / "statistics_reproduction.json")
    require(
        statistics.get("raw_only") is True
        and statistics.get("media_or_checkpoint_loaded") is False,
        "Statistics must be reconstructed from raw only",
    )
    if "status" in statistics:
        require(statistics["status"] == "PASS", "Statistics audit not PASS")
    check_hashes(statistics["inputs_sha256"])
    check_hashes(statistics["source_sha256"])
    for name in ("audit/statistics_reproduction_audit.json", "audit/statistics_audit.json"):
        if (root / name).exists():
            require(read(root / name).get("status") == "PASS", f"Statistics audit failed: {name}")
    check_hashes(read(root / "config.json")["source_sha256"])
    require(
        len(read(root / "scene_manifest.json")["scenes"]) == 17,
        "Exactly 17 exposed scenes required",
    )
    require(
        len(list((root / "figures").glob("*.png"))) >= 6, "Six required scientific figures missing"
    )


def write_gzip(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as output:
        with gzip.GzipFile(filename="", fileobj=output, mode="wb", mtime=0) as compressed:
            with source.open("rb") as raw:
                shutil.copyfileobj(raw, compressed)


def checkpoint_index(root):
    plan = read(root / "state_plan.json")["states"]
    require(len(plan) == 425, "Frozen state plan must contain 425 states")
    rows, summaries = [], {}
    for spec in plan:
        path = Path(spec["state_path"])
        path = path if path.is_absolute() else root / path
        folder = Path(spec["optimization_path"])
        folder = folder if folder.is_absolute() else root / folder
        completion, summary = read(folder / "completion.json"), read(folder / "summary.json")
        digest = sha(path)
        require(
            completion["state_file_sha256"] == digest, "Selected state changed since completion"
        )
        require(
            completion["plan_sha256"] == sha(root / "state_plan.json"), "State plan hash mismatch"
        )
        rows.append(
            {
                "key": spec["key"],
                "phase": spec["phase"],
                "scene_id": spec["scene_id"],
                "state_path": str(path.resolve()),
                "relative_state_path": str(path.relative_to(root)),
                "state_file_sha256": digest,
                "state_tensor_hash": summary["state_hash"],
                "bytes": path.stat().st_size,
                "selected_step": summary["selected_step"],
                "summary_sha256": sha(folder / "summary.json"),
                "completion_sha256": sha(folder / "completion.json"),
            }
        )
        summaries[spec["key"]] = summary
    write_json(
        root / "checkpoints/index.json",
        {
            "n_states": len(rows),
            "states": rows,
            "distribution": "Metadata only in docs; .pt files remain in local experiment output",
        },
    )
    write_json(root / "raw/optimization_summaries.json", summaries)


def source_snapshot(root, repository):
    paths = set(repository.glob("src/mcss/**/*.py"))
    paths.update(repository.glob("scripts/*direct_capacity*.py"))
    paths.update(repository.glob("tests/test_direct_capacity*.py"))
    for name in ("pyproject.toml", "uv.lock"):
        if (repository / name).is_file():
            paths.add(repository / name)
    paths = sorted(p for p in paths if p.is_file())
    sources = {str(p.relative_to(repository)): sha(p) for p in paths}
    write_json(root / "source_manifest.json", sources)
    # Filesystem enumeration deliberately includes new, not-yet-git-tracked source files.
    with (root / "source_snapshot.tar.gz").open("wb") as output:
        with gzip.GzipFile(filename="", fileobj=output, mode="wb", mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w|") as archive:
                for path in paths:
                    info = archive.gettarinfo(str(path), arcname=str(path.relative_to(repository)))
                    info.uid = info.gid = 0
                    info.uname = info.gname = ""
                    info.mtime = 0
                    with path.open("rb") as handle:
                        archive.addfile(info, handle)
    return sources


def finalize(root, docs):
    root, docs = Path(root).resolve(), Path(docs).resolve()
    repository = Path(__file__).resolve().parents[1]
    validate(root, docs)  # No publication writes until completed audits have passed.
    checkpoint_index(root)
    sources = source_snapshot(root, repository)
    docs.mkdir(parents=True, exist_ok=True)
    for path in sorted(root.iterdir()):
        if path.is_file() and (
            path.suffix in (".json", ".md", ".sh", ".txt", ".patch")
            or path.name == "source_snapshot.tar.gz"
        ):
            if path.name not in ("artifact_manifest.json", "integrity.json"):
                shutil.copy2(path, docs / path.name)
    shutil.copytree(root / "figures", docs / "figures", dirs_exist_ok=True)
    (docs / "checkpoints").mkdir(exist_ok=True)
    shutil.copy2(root / "checkpoints/index.json", docs / "checkpoints/index.json")
    for sub in ("raw", "audit"):
        for path in sorted((root / sub).rglob("*")):
            if not path.is_file() or path.suffix not in (
                ".json",
                ".jsonl",
                ".csv",
                ".log",
                ".txt",
                ".patch",
                ".xml",
            ):
                continue
            target = docs / path.relative_to(root)
            if sub == "audit" and path.stat().st_size <= 128 * 1024 and path.suffix == ".json":
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)
            else:
                write_gzip(path, Path(str(target) + ".gz"))
    (docs / "ARCHIVE_CONTENTS.md").write_text(
        "# Archived capacity attribution\n\n"
        f"Complete local experiment output: `{root}`.\n\n"
        "This report archive includes source snapshots (including new untracked code), "
        "contracts, figures, raw numeric results, access/seal logs, audit evidence, and "
        "a 425-state checkpoint hash/path/size index. `.pt` weights, prediction arrays, "
        "and datasets remain in the local output tree and are not copied into the git report. "
        "Compressed JSON/log files use ordinary gzip. `source_snapshot.tar.gz` contains "
        "repository-relative paths; inspect it before extracting into a separate directory.\n\n"
        "Protected holdout media were not opened by this finalizer. Earlier verified "
        "digests and metadata-only preservation checks are explicitly distinguished "
        "from newly computed file hashes in the integrity audit.\n"
    )
    files = {
        str(p.relative_to(root)): sha(p)
        for p in sorted(root.rglob("*"))
        if p.is_file() and p not in (root / "artifact_manifest.json", root / "integrity.json")
    }
    write_json(root / "artifact_manifest.json", files)
    evidence = {
        name: sha(root / name)
        for name in (
            "audit/preservation_recheck.json",
            "audit/postrun_validation.json",
            "tests.json",
            "baseline_integrity.json",
            "baseline_carrier_reconstruction.json",
            "statistics_reproduction.json",
            "checkpoints/index.json",
            "source_manifest.json",
            "source_snapshot.tar.gz",
            "artifact_manifest.json",
        )
    }
    preservation = read(root / "audit/preservation_recheck.json")
    integrity = {
        "status": "PASS",
        "experiment": EXPERIMENT,
        "artifact_count": len(files),
        "checkpoint_states_indexed": 425,
        "source_files_archived": len(sources),
        "evidence_sha256": evidence,
        "old_artifact_preservation_status": preservation["status"],
        "protected_media_bytes_reopened": False,
        "holdout_hash_assurance": preservation["protected_media_hash_assurance"],
        "FINAL_HOLDOUT_TOUCHED": False,
        "DYNAMIC_TTT_RUN": False,
        "NEW_CARRIER_TRAINED": False,
        "baseline_state_reconstruction_exact": True,
        "baseline_metrics_exact": read(root / "baseline_integrity.json")[
            "max_metric_absolute_delta"
        ]
        == 0,
        "full_local_output": str(root),
        "report_directory": str(docs),
        "checkpoint_weights_copied_to_docs": False,
    }
    write_json(root / "integrity.json", integrity)
    for name in ("artifact_manifest.json", "integrity.json"):
        shutil.copy2(root / name, docs / name)
    doc_hashes = {
        str(p.relative_to(docs)): sha(p)
        for p in sorted(docs.rglob("*"))
        if p.is_file() and p.name != "report_manifest.json"
    }
    write_json(docs / "report_manifest.json", doc_hashes)
    return integrity


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--docs", required=True)
    args = parser.parse_args()
    print(json.dumps(finalize(args.root, args.docs), indent=2))
