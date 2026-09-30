"""Archive completed geometry attribution without inference or publishing weights."""

import argparse
import gzip
import json
import shutil
import tarfile
from pathlib import Path

from mcss.mechanism_pilot.small_training import sha, write_json

EXPERIMENT = "EXP-3D-CONTEXT-OBSERVABILITY-SUPERVISION-V1"
PHASES = ("baseline", "formal")


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
    paths.update(repository.glob("scripts/*observability*.py"))
    paths.update(repository.glob("scripts/*optimization_bounds*.py"))
    paths.update(repository.glob("scripts/*direct_capacity*.py"))
    paths.update(repository.glob("tests/test_*observability*.py"))
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
    require(
        not list(docs.rglob("*.pt")) and not list(docs.rglob("*.npz")),
        "Destination already contains excluded weights or predictions",
    )
    required = [
        "README.md",
        "STATUS.md",
        "commands.sh",
        "tests.json",
        "statistics_reproduction.json",
        "observability_analysis.json",
        "supervision_analysis.json",
        "gap_closure_analysis.json",
        "wrong_scene_analysis.json",
        "bootstrap_results.json",
        "cost_analysis.json",
        "audit/statistics_reproduction_audit.json",
    ]
    for name in required:
        require((root / name).is_file(), f"Missing completed artifact: {name}")
    passes = [
        "baseline_reproduction.json",
        "audit/postrun_validation.json",
        "audit/statistics_reproduction_audit.json",
        "audit/observability_integrity.json",
    ]
    for phase in PHASES:
        passes += [f"{phase}_optimization_complete.json", f"raw/{phase}_evaluation_integrity.json"]
    for name in passes:
        record = read(root / name)
        require(record.get("status") == "PASS", f"Not PASS: {name}")
        for field in ("final_holdout_touched", "dynamic_ttt_run", "new_carrier_trained"):
            require(
                not record.get(field, False) and not record.get(field.upper(), False),
                f"Forbidden action: {field}",
            )
    post = read(root / "audit/postrun_validation.json")
    require(post["n_states"] == 136, "Incomplete audit")
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
    for record in ("config.json",):
        for path, digest in read(root / record)["source_sha256"].items():
            require(sha(Path(path)) == digest, f"Source changed: {path}")
    for name in ("statistics_reproduction.json",):
        record = read(root / name)
        require(
            record["raw_only"] and not record["media_or_state_checkpoint_loaded"], "Not raw-only"
        )
        for mapping in ("input_sha256", "source_sha256"):
            for path, digest in record[mapping].items():
                require(sha(Path(path)) == digest, f"Statistics input changed: {path}")
    seal = read(root / "audit/previous_experiment_seal.json")
    require(len(seal["files"]) == 11285, "Expected 11285 old sealed artifacts")
    for name, record in seal["files"].items():
        path = Path(name)
        require(
            path.stat().st_size == record["bytes"] and sha(path) == record["sha256"],
            f"Previous artifact changed: {name}",
        )
    require(len(list((root / "figures").glob("*.png"))) >= 7, "Seven figures required")


def checkpoint_index(root):
    specs = read(root / "state_plan.json")["states"]
    require(len(specs) == 136 and len({s["key"] for s in specs}) == 136, "Incomplete state plan")
    rows, summaries, trajectories = [], {}, {}
    for spec in specs:
        folder = root / spec["optimization_path"]
        state = root / spec["state_path"]
        summary, completion = read(folder / "summary.json"), read(folder / "completion.json")
        require(
            completion["status"] == "PASS" and completion["key"] == spec["key"],
            "State completion identity mismatch",
        )
        require(completion["plan_sha256"] == sha(root / "state_plan.json"), "Plan changed")
        require(completion["config_sha256"] == sha(root / "config.json"), "Config changed")
        require(
            completion["state_file_sha256"] == summary["state_file_sha256"] == sha(state),
            "Checkpoint changed",
        )
        require(completion["summary_sha256"] == sha(folder / "summary.json"), "Summary changed")
        require(completion["state_hash"] == summary["state_hash"], "State tensor hash mismatch")
        require(
            summary["variant"] == spec["variant"] and summary["selected_step"] == 10000,
            "Frozen variant/final budget changed",
        )
        checks = read(folder / "full_objective_checkpoints.json")
        trace = read(folder / "trace.json")
        require([c["step"] for c in checks] == list(range(0, 10001, 100)), "Incomplete checkpoints")
        require([r["step"] for r in trace] == [1] + list(range(10, 10001, 10)), "Incomplete trace")
        summaries[spec["key"]] = {"spec": spec, "summary": summary}
        trajectories[spec["key"]] = {
            "spec": spec,
            "summary": summary,
            "full_objective_checkpoints": checks,
            "trace": trace,
            "lock": read(folder / "lock.json"),
            "completion": completion,
            "optimizer_GT_access": read(folder / "access.json"),
            "source_file_sha256": {
                name: sha(folder / name)
                for name in (
                    "trace.json",
                    "full_objective_checkpoints.json",
                    "optimization_trace.csv",
                    "summary.json",
                    "lock.json",
                    "completion.json",
                    "access.json",
                )
            },
        }
        rows.append(
            {
                **spec,
                "bytes": state.stat().st_size,
                "state_file_sha256": sha(state),
                "state_tensor_hash": summary["state_hash"],
                "selected_step": 10000,
                "summary_sha256": sha(folder / "summary.json"),
                "completion_sha256": sha(folder / "completion.json"),
            }
        )
    # Preserve byte identity of a pre-analysis aggregate on repeated finalization.
    aggregate = root / "raw/optimization_summaries.json"
    if aggregate.exists():
        require(read(aggregate) == summaries, "Existing statistics aggregate differs from states")
    else:
        write_json(aggregate, summaries)
    trajectory_path = root / "raw/optimization_trajectories.json"
    if trajectory_path.exists():
        require(read(trajectory_path) == trajectories, "Existing full trajectory archive changed")
    else:
        write_json(trajectory_path, trajectories)
    write_json(
        root / "checkpoints/index.json",
        {
            "n_states": 136,
            "states": rows,
            "distribution": "Metadata only; GPU checkpoint weights remain in local outputs",
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
            "old_files_unchanged": 11285,
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
    for sub in ("figures",):
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
        "The archive contains all numeric query/context rows, complete 136-state trajectories "
        "(trace, full-objective checkpoints and fixed-final summaries), protocol, "
        "source snapshot, "
        "audits and the 136-state hash index. GPU weights, predictions and datasets "
        "stay in outputs.\n\n"
        "From the repository root, reproduce primary statistics without media or weights:\n\n"
        f"`.venv/bin/python scripts/analyze_observability_supervision.py --root {docs} "
        "--output /tmp/context-geometry-statistics-reproduction`\n\n"
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
        "checkpoint_states_indexed": 136,
        "old_files_unchanged": 11285,
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
