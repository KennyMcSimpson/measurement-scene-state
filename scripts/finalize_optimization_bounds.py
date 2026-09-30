"""Archive completed 16-grid attribution; no inference, media decode, or checkpoint publishing."""

import argparse
import csv
import gzip
import json
import shutil
import tarfile
from pathlib import Path

from analyze_optimization_bounds import load_trajectories

from mcss.mechanism_pilot.small_training import sha, write_json

EXPERIMENT = "EXP-3D-16G-OPTIMIZATION-BOUNDS-V1"
PHASES = ("context", "oracle", "secondary")


def read(path):
    return json.loads(path.read_text())


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def write_gzip(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as output:
        with gzip.GzipFile(filename="", fileobj=output, mode="wb", mtime=0) as compressed:
            with source.open("rb") as raw:
                shutil.copyfileobj(raw, compressed)


def source_snapshot(root, repository):
    paths = set(repository.glob("src/mcss/**/*.py"))
    paths.update(repository.glob("scripts/*optimization_bounds*.py"))
    paths.update(repository.glob("scripts/*direct_capacity*.py"))
    paths.update(repository.glob("tests/test_*optimization*.py"))
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


def validate(root, docs):
    require(root.name == docs.name == EXPERIMENT, "Wrong experiment directory")
    require(
        root != docs and root not in docs.parents and docs not in root.parents,
        "Output and docs must be separate",
    )
    required = [
        "README.md",
        "STATUS.md",
        "commands.sh",
        "tests.json",
        "statistics_reproduction.json",
        "primary_analysis_complete.json",
        "secondary_analysis/statistics_reproduction.json",
        "audit/statistics_reproduction_audit.json",
    ]
    for name in required:
        require((root / name).is_file(), f"Missing completed artifact: {name}")
    passes = [
        "baseline_reproduction.json",
        "audit/postrun_validation.json",
        "audit/statistics_reproduction_audit.json",
        "secondary_analysis/secondary_decision_audit.json",
    ]
    for phase in PHASES:
        passes += [f"{phase}_optimization_complete.json", f"raw/{phase}_evaluation_integrity.json"]
    for name in passes:
        record = read(root / name)
        require(record.get("status") == "PASS", f"Not PASS: {name}")
        for field in ("final_holdout_touched", "dynamic_ttt_run", "new_carrier_trained"):
            require(not record.get(field, False), f"Forbidden action: {field}")
    post = read(root / "audit/postrun_validation.json")
    require(post["chain_count"] == 153 and post["state_count"] == 782, "Incomplete audit")
    tests = read(root / "tests.json")
    full = tests.get("full_suite", tests)
    require(full.get("passed", 0) > 0, "No full-suite pass evidence")
    for record in (tests, full):
        require(str(record.get("status", "PASS")).upper() in ("PASS", "PASSED"), "Tests failed")
        require(
            all(
                record.get(k, 0) in (0, None, False)
                for k in ("failed", "failures", "errors", "error")
            ),
            "Test errors remain",
        )
    prereg = read(root / "preregistration.json")
    for path, digest in prereg["locked_file_sha256"].items():
        require(sha(root / path) == digest, f"Preregistration hash mismatch: {path}")
    for record in ("config.json", "audit/analysis_source_lock.json"):
        for path, digest in read(root / record)["source_sha256"].items():
            require(sha(Path(path)) == digest, f"Source changed: {path}")
    for name in ("statistics_reproduction.json", "secondary_analysis/statistics_reproduction.json"):
        record = read(root / name)
        require(record["raw_only"] and not record["media_or_checkpoint_loaded"], "Not raw-only")
        for mapping in ("inputs_sha256", "source_sha256"):
            for path, digest in record[mapping].items():
                require(sha(Path(path)) == digest, f"Statistics input changed: {path}")
    seal = read(root / "audit/previous_experiment_seal.json")
    require(len(seal["files"]) == 4974, "Expected 4974 old sealed artifacts")
    for name, record in seal["files"].items():
        path = Path(name)
        require(
            path.stat().st_size == record["bytes"] and sha(path) == record["sha256"],
            f"Previous artifact changed: {name}",
        )
    require(len(list((root / "figures").glob("*.png"))) >= 7, "Seven figures required")


def checkpoint_index(root):
    plan = read(root / "state_plan.json")
    require(len(plan["chains"]) == 153 and len(plan["states"]) == 782, "Incomplete plan")
    chain_map = {s["key"]: s for s in plan["chains"]}
    trajectories = load_trajectories(root, plan, {}, include_secondary=True)
    rows = []
    for key, trajectory in trajectories.items():
        require(trajectory["spec"] == chain_map[key], "Aggregate chain spec differs from plan")
        chain_dir = root / chain_map[key]["optimization_path"]
        require(
            trajectory["summary"] == read(chain_dir / "trajectory_summary.json"),
            "Aggregate trajectory summary differs from saved trajectory",
        )
        checkpoints = read(chain_dir / "full_objective_checkpoints.json")
        if isinstance(checkpoints, dict):
            checkpoints = checkpoints["full_objective_checkpoints"]
        require(trajectory["checkpoints"] == checkpoints, "Aggregate objective checkpoints differ")
        # Trace values are parsed with the same CSV conversion as the frozen statistics CLI.
        with (chain_dir / "optimization_trace.csv").open() as handle:
            trace = [
                {
                    k: (float(v) if k != "kind" and v not in ("", "None") else v)
                    for k, v in row.items()
                }
                for row in csv.DictReader(handle)
            ]
        require(trajectory["trace"] == trace, "Aggregate trace differs from optimizer CSV")
    for spec in plan["states"]:
        folder = root / spec["optimization_path"]
        summary = read(folder / "summary.json")
        completion_path = (
            root / chain_map[spec["chain_key"]]["optimization_path"] / "completion.json"
        )
        completion = read(completion_path)
        state = root / spec["state_path"]
        digest = sha(state)
        require(completion["plan_sha256"] == sha(root / "state_plan.json"), "Plan changed")
        sealed = completion["states"][spec["key"]]
        require(sealed["state_file_sha256"] == digest, "Checkpoint changed")
        require(sealed["summary_sha256"] == sha(folder / "summary.json"), "Summary changed")
        require(summary["state_hash"] == sealed["state_hash"], "State identity mismatch")
        require(
            trajectories[spec["chain_key"]]["budget_summaries"][str(spec["budget"])][
                spec["selection"]
            ]
            == summary,
            "Aggregate budget summary differs",
        )
        rows.append(
            {
                **spec,
                "state_file_sha256": digest,
                "state_tensor_hash": summary["state_hash"],
                "bytes": state.stat().st_size,
                "selected_step": summary["selected_step"],
                "summary_sha256": sha(folder / "summary.json"),
                "completion_sha256": sha(completion_path),
            }
        )
    write_json(root / "raw/trajectory_summaries.json", trajectories)
    write_json(
        root / "checkpoints/index.json",
        {
            "n_states": len(rows),
            "n_chains": 153,
            "states": rows,
            "distribution": "Metadata only; weights remain in local outputs",
        },
    )


def finalize(root, docs):
    root, docs = Path(root).resolve(), Path(docs).resolve()
    validate(root, docs)
    checkpoint_index(root)
    sources = source_snapshot(root, Path(__file__).resolve().parents[1])
    write_json(
        root / "audit/finalizer_preservation_recheck.json",
        {
            "status": "PASS",
            "old_files_unchanged": 4974,
            "previous_seal_sha256": sha(root / "audit/previous_experiment_seal.json"),
            "protected_media_bytes_reopened": False,
        },
    )
    docs.mkdir(parents=True, exist_ok=True)
    for path in sorted(root.iterdir()):
        if path.is_file() and (
            path.suffix in (".json", ".md", ".sh", ".txt", ".patch")
            or path.name == "source_snapshot.tar.gz"
        ):
            if path.name not in ("artifact_manifest.json", "integrity.json"):
                shutil.copy2(path, docs / path.name)
    for sub in ("figures", "secondary_analysis"):
        shutil.copytree(root / sub, docs / sub, dirs_exist_ok=True)
    (docs / "checkpoints").mkdir(exist_ok=True)
    shutil.copy2(root / "checkpoints/index.json", docs / "checkpoints/index.json")
    for sub in ("raw", "audit"):
        for path in sorted((root / sub).rglob("*")):
            if not path.is_file() or path.suffix not in (
                ".json",
                ".jsonl",
                ".csv",
                ".txt",
                ".log",
                ".patch",
                ".xml",
            ):
                continue
            target = docs / path.relative_to(root)
            if sub == "audit" and path.suffix == ".json" and path.stat().st_size <= 131072:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)
            else:
                write_gzip(path, Path(str(target) + ".gz"))
    (docs / "ARCHIVE_CONTENTS.md").write_text(
        "# Portable numeric archive\n\n"
        "The archive contains all numeric query/context rows, complete 153-chain trajectories "
        "(trace, full-objective checkpoints and all budget summaries), protocol, source snapshot, "
        "audits and the 782-state hash index. GPU weights, predictions and datasets "
        "stay in outputs.\n\n"
        "From the repository root, reproduce primary statistics without media or weights:\n\n"
        f"`.venv/bin/python scripts/analyze_optimization_bounds.py --root {docs} "
        "--output /tmp/optimization-bounds-primary-reproduction`\n\n"
        "Add `--include-secondary` and use a different output directory for secondary analysis. "
        "The statistics loader reads compressed raw directly. Historical chronology is evidence "
        "of the original run; copying files does not recreate historical timestamps.\n"
    )
    files = {
        str(p.relative_to(root)): sha(p)
        for p in sorted(root.rglob("*"))
        if p.is_file() and p.name not in ("artifact_manifest.json", "integrity.json")
    }
    write_json(root / "artifact_manifest.json", files)
    integrity = {
        "status": "PASS",
        "experiment": EXPERIMENT,
        "chain_count": 153,
        "checkpoint_states_indexed": 782,
        "old_files_unchanged": 4974,
        "source_files_archived": len(sources),
        "checkpoint_weights_copied_to_docs": False,
        "final_holdout_touched": False,
        "new_carrier_trained": False,
        "dynamic_ttt_run": False,
        "artifact_manifest_sha256": sha(root / "artifact_manifest.json"),
    }
    write_json(root / "integrity.json", integrity)
    for name in ("artifact_manifest.json", "integrity.json"):
        shutil.copy2(root / name, docs / name)
    require(
        not list(docs.rglob("*.pt")) and not list(docs.rglob("*.npz")),
        "Weights/prediction arrays must not enter docs",
    )
    write_json(
        docs / "report_manifest.json",
        {
            str(p.relative_to(docs)): sha(p)
            for p in sorted(docs.rglob("*"))
            if p.is_file() and p.name != "report_manifest.json"
        },
    )
    return integrity


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--docs", required=True)
    args = parser.parse_args()
    print(json.dumps(finalize(args.root, args.docs), indent=2))
