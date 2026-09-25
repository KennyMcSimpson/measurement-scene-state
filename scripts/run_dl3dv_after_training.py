"""One finite, dependency-gated DL3DV evaluation after the approved training run.

This driver does not retrain or tune from benchmark results. It waits for the
explicit training and data-verification artifacts, freezes their identities,
checks synthetic full-resolution resource use, then evaluates two named causal
protocols once. Every run requires a fresh output directory.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from run_dynamic_policy_refinement import sha256, write_json

PROJECT = Path(__file__).resolve().parents[1]
DATA_REVISION = "9684e8382278c5e18173c1e72bd246daf287453"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_training(root: Path) -> dict | None:
    """Return completed dependencies; fail on a failed rather than missing run."""
    path = root / "pipeline_status.json"
    if not path.is_file():
        raise FileNotFoundError(f"approved training pipeline has not started: {path}")
    status = read_json(path)
    if status.get("status") == "failed":
        raise RuntimeError(f"training pipeline failed: {status.get('error')}")
    if status.get("status") != "complete":
        return None
    checkpoint = root / "training/phase_b_final.pt"
    policy = root / "policy_refinement/policy_recollected.pt"
    for key, file in (("final_carrier_sha256", checkpoint), ("final_policy_sha256", policy)):
        if sha256(file) != status.get(key):
            raise ValueError(f"completed training artifact identity mismatch: {key}")
    history = read_json(root / "training/status.json")
    if history.get("status") != "complete" or history.get("global_step") != 9000:
        raise ValueError("the approved 9000-step carrier training is incomplete")
    for relative, policies in (
        ("evaluation_dev_fixed/report.json", {"OFF", "FUSE", "COMPLETE", "ALL"}),
        ("policy_refinement/evaluation_dev_recollected/report.json", {"OFF", "LEARNED"}),
    ):
        report = read_json(root / relative)
        if report.get("status") != "complete":
            raise ValueError(f"development evaluation is incomplete: {relative}")
        summaries = report.get("summaries", [])
        if {row["policy"] for row in summaries} != policies:
            raise ValueError(f"development policies are incomplete: {relative}")
        for row in summaries:
            for metric in ("depth_abs_rel", "rgb_psnr"):
                if row.get("metric_valid_scene_counts", {}).get(metric) != 46:
                    raise ValueError(f"development metric missing a scene: {relative}")
                if not math.isfinite(row["metrics_mean"][metric]):
                    raise ValueError(f"development metric is nonfinite: {relative}")
    return {
        "checkpoint": str(checkpoint), "policy": str(policy),
        "checkpoint_sha256": sha256(checkpoint), "policy_sha256": sha256(policy),
        "pipeline_status_sha256": sha256(path),
    }


def validate_download(root: Path, raw_root: Path) -> dict | None:
    """Require the independent verifier, not merely a downloader completion flag."""
    path = root / "completion_verification.json"
    if not path.is_file():
        download_status = root / "status.json"
        if download_status.is_file() and read_json(download_status).get("status") == "failed":
            raise RuntimeError("DL3DV downloader failed before independent verification")
        return None
    verification = read_json(path)
    if verification.get("status") != "verified":
        raise ValueError("DL3DV independent verification has not passed")
    if verification.get("revision") != DATA_REVISION:
        raise ValueError("DL3DV verification revision differs from the approved source")
    if Path(verification["data_root"]).resolve() != raw_root.resolve():
        raise ValueError("DL3DV verification names a different data root")
    if verification.get("file_count") != 12777:
        raise ValueError("expected 140 transforms plus 12637 selected images")
    if verification.get("total_bytes") != 11313259491:
        raise ValueError("verified DL3DV byte total differs from the approved selected files")
    coverage = verification.get("protocol_coverage", {})
    if (
        coverage.get("status") != "complete"
        or coverage.get("mapping") != "transforms_frame_order"
        or set(coverage.get("protocols", [])) != {"full16", "ar32"}
        or coverage.get("required_files") != 12777
        or coverage.get("missing_files") != 0
    ):
        raise ValueError("DL3DV verification has not checked actual protocol frame paths")
    if verification.get("source_hash_verified", {}).get("all_entries") is not True:
        raise ValueError("DL3DV verifier did not confirm every remote file identity")
    if verification.get("download_status", {}).get("entry_sha256_matches") is not True:
        raise ValueError("DL3DV verifier did not reconcile downloader and independent hashes")
    return {"verification": str(path), "verification_sha256": sha256(path)}


def validate_report(report: dict) -> None:
    if report.get("status") != "complete":
        raise ValueError(f"benchmark evaluation incomplete: {report.get('status')}")
    records = report.get("records", [])
    if len(records) != 140 or len({row["scene_id"] for row in records}) != 140:
        raise ValueError("benchmark requires exactly 140 unique scene records")
    if any(row.get("status") != "ok" for row in records):
        raise ValueError("benchmark contains a failed scene")
    for row in records:
        images = row.get("per_image", [])
        if len(images) != row.get("target_count") or not images:
            raise ValueError("benchmark target count mismatch")
        if any(image.get("status") != "ok" for image in images):
            raise ValueError("benchmark contains a failed target image")
        if any(set(image.get("metrics", {})) != {"psnr", "ssim", "lpips"} for image in images):
            raise ValueError("benchmark requires PSNR, SSIM, and VGG LPIPS for every target")
        if any(not math.isfinite(value) for image in images for value in image["metrics"].values()):
            raise ValueError("benchmark contains a nonfinite target metric")


def write_results(output: Path, training_root: Path, reports: dict) -> None:
    lines = [
        "# Dynamic TTT corrected-camera experiment and DL3DV comparison", "",
        "The main method remains a query-independent typed scene state read by a fixed "
        "measurement renderer, with prefix-only dynamic fast-weight updates.", "",
        "## Hypersim development evidence", "",
        "These are development results, not DL3DV results. The old-original and "
        "corrected protocols must not be pooled.", "",
        "| Run / development protocol | Policy | Scenes | Depth AbsRel | RGB PSNR |",
        "|---|---|---:|---:|---:|",
    ]
    inputs = [
        ("Old carrier / original protocol", PROJECT / (
            "outputs/dynamic_full_method_20260920/evaluation_dev_fixed/report.json")),
        ("Old carrier / corrected protocol", training_root / (
            "evaluation_dev_old_carrier/report.json")),
        ("New carrier / corrected protocol", training_root / "evaluation_dev_fixed/report.json"),
        ("32-scene policy pilot / original protocol", PROJECT / (
            "outputs/dynamic_policy_refinement_pilot_20260920/"
            "evaluation_dev_recollected/report.json")),
        ("New carrier + recollected policy / corrected protocol", training_root / (
            "policy_refinement/evaluation_dev_recollected/report.json")),
    ]
    for label, path in inputs:
        if not path.is_file():
            lines.append(f"| {label} (artifact absent) | pending | — | — | — |")
            continue
        for row in read_json(path)["summaries"]:
            metrics = row["metrics_mean"]
            lines.append(
                f"| {label} | {row['policy']} | {row['scene_count']} | "
                f"{metrics['depth_abs_rel']:.6f} | {metrics['rgb_psnr']:.6f} |"
            )
    lines += [
        "", "## DL3DV-140", "",
        "This is a causal adapter comparison, not an exact tttLRM reproduction. "
        "Our normalization uses only four warmup camera poses; the public tttLRM "
        "loader includes selected query cameras in normalization. All local targets "
        "use the official 536 x 960 image and PSNR/SSIM/VGG-LPIPS preprocessing. "
        "The paper does not identify macro versus micro aggregation, so both are retained.",
        "", "| Views / method | Aggregation | PSNR | SSIM | LPIPS |",
        "|---|---|---:|---:|---:|",
        "| 16 / Long-LRM (published) | paper | 22.660 | 0.740 | 0.292 |",
        "| 16 / tttLRM full (published) | paper | 23.600 | 0.784 | 0.255 |",
        "| 32 / Long-LRM (published) | paper | 24.100 | 0.783 | 0.254 |",
        "| 32 / tttLRM AR (published) | paper | 24.310 | 0.803 | 0.237 |",
    ]
    for name, report in reports.items():
        for key, label in (
            ("macro_scene_mean", "scene macro"), ("micro_image_mean", "image micro"),
        ):
            metrics = report[key]["metrics"]
            lines.append(
                f"| {name} / our causal adapter | {label} | {metrics['psnr']:.6f} | "
                f"{metrics['ssim']:.6f} | {metrics['lpips']:.6f} |"
            )
    lines += [
        "", "Published source: https://arxiv.org/html/2602.20160v2#S4.T2", "",
        "The old failed resource preflight and earlier policy-collapse diagnostics are "
        "retained in their original directories. This run's configuration, hashes, "
        "commands, exit codes, scene records, and resource check are stored beside this file.",
        "No checkpoint was selected or refitted from these external benchmark scores.", "",
    ]
    (output / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-root", required=True, type=Path)
    parser.add_argument("--download-root", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--wait-hours", type=float, default=24.0)
    args = parser.parse_args()
    output = args.output.resolve()
    output.relative_to(PROJECT / "outputs")
    if args.wait_hours <= 0 or args.wait_hours > 48:
        raise ValueError("dependency wait must be within (0, 48] hours")
    output.mkdir(parents=True, exist_ok=False)
    status = {
        "schema": "mcss.dl3dv.finite_continuation.v1", "status": "running",
        "stage": "waiting_for_training_and_verified_download", "pid": os.getpid(),
        "started_at": datetime.now(UTC).isoformat(), "stages": [],
        "comparison": "causal adapter comparison, not exact tttLRM reproduction",
    }
    write_json(output / "status.json", status)
    # These hashes are taken before any dependency finishes or benchmark is opened.
    sources = [Path(__file__).resolve(), PROJECT / "scripts/check_dl3dv_resources.py"]
    sources += [PROJECT / "scripts/evaluate_dl3dv_benchmark.py"]
    sources += [PROJECT / "scripts/verify_dl3dv_download.py"]
    sources += [PROJECT / "scripts/run_dynamic_policy_refinement.py"]
    sources += sorted((PROJECT / "src/mcss/dynamic").glob("*.py"))
    sources += [
        PROJECT / "src/mcss/data/dl3dv_benchmark.py",
        PROJECT / "src/mcss/data/dl3dv_download.py",
        PROJECT / "src/mcss/evaluation/dl3dv_report.py",
        PROJECT / "src/mcss/evaluation/sealed_rgb.py",
        PROJECT / "src/mcss/measurements.py", PROJECT / "src/mcss/geometry.py",
        PROJECT / "src/mcss/types.py",
    ]
    try:
        frozen_sources = {str(path): sha256(path) for path in sources}
        split = PROJECT / (
            "outputs/benchmark_protocol_20260920/sources/Long-LRM/data/"
            "dl3dv_fold_8_kmeans_input_idx.json"
        )
        frozen_sources[str(split)] = sha256(split)
        write_json(output / "source_sha256.json", frozen_sources)
        deadline = time.monotonic() + args.wait_hours * 3600
        while True:
            training = validate_training(args.training_root.resolve())
            download = validate_download(args.download_root.resolve(), args.data_root)
            download_status = args.download_root / "status.json"
            if download is None and download_status.is_file() and (
                read_json(download_status).get("status") == "complete"
            ):
                for file, expected in frozen_sources.items():
                    if sha256(Path(file)) != expected:
                        raise ValueError(f"dependency changed before download verification: {file}")
                command = [
                    sys.executable, "-u", str(PROJECT / "scripts/verify_dl3dv_download.py"),
                    "--inventory-dir", str(args.download_root.resolve()),
                    "--data-root", str(args.data_root.resolve()), "--protocol-json", str(split),
                    "--output", str(args.download_root.resolve() / "completion_verification.json"),
                ]
                status["stage"] = "verify_download"
                write_json(output / "status.json", status)
                with (output / "verify_download.stdout.log").open("x", encoding="utf-8") as stdout:
                    error_log = output / "verify_download.stderr.log"
                    with error_log.open("x", encoding="utf-8") as stderr:
                        result = subprocess.run(
                            command, cwd=PROJECT, stdout=stdout, stderr=stderr, check=False,
                        )
                status["stages"].append({
                    "name": "verify_download", "command": command,
                    "returncode": result.returncode, "finished_at": datetime.now(UTC).isoformat(),
                })
                if result.returncode:
                    raise RuntimeError("independent download verification failed; see its log")
                download = validate_download(args.download_root.resolve(), args.data_root)
            if training is not None and download is not None:
                status["dependencies"] = {
                    "training_complete": True, "download_verified": True,
                }
                break
            if time.monotonic() >= deadline:
                raise TimeoutError("dependencies did not complete within the finite wait limit")
            status["dependencies"] = {
                "training_complete": training is not None,
                "download_verified": download is not None,
            }
            status["stage"] = "waiting_for_training_and_verified_download"
            status["updated_at"] = datetime.now(UTC).isoformat()
            write_json(output / "status.json", status)
            time.sleep(30)
        write_json(output / "dependencies.json", {"training": training, "download": download})
        frozen_sources[training["checkpoint"]] = training["checkpoint_sha256"]
        frozen_sources[training["policy"]] = training["policy_sha256"]
        frozen_sources[download["verification"]] = download["verification_sha256"]

        def check_frozen() -> None:
            for file, expected in frozen_sources.items():
                if sha256(Path(file)) != expected:
                    raise ValueError(f"frozen evaluation dependency changed: {file}")

        def run_stage(name: str, script: str, arguments: list[str]) -> None:
            check_frozen()
            command = [sys.executable, "-u", str(PROJECT / "scripts" / script), *arguments]
            stage = {"name": name, "command": command, "started_at": datetime.now(UTC).isoformat()}
            status["stage"] = name
            status["stages"].append(stage)
            write_json(output / "status.json", status)
            env = dict(os.environ, OMP_NUM_THREADS="4", MKL_NUM_THREADS="4")
            with (output / f"{name}.stdout.log").open("x", encoding="utf-8") as stdout:
                with (output / f"{name}.stderr.log").open("x", encoding="utf-8") as stderr:
                    child = subprocess.Popen(
                        command, cwd=PROJECT, env=env, stdout=stdout, stderr=stderr,
                    )
                    stage["pid"] = child.pid
                    write_json(output / "status.json", status)
                    stage["returncode"] = child.wait()
            stage["finished_at"] = datetime.now(UTC).isoformat()
            write_json(output / "status.json", status)
            if stage["returncode"] != 0:
                raise RuntimeError(f"{name} failed; see {name}.stderr.log")

        run_stage("resource_preflight", "check_dl3dv_resources.py", [
            "--checkpoint", training["checkpoint"], "--output", str(output / "resource_preflight"),
            "--device", "cuda:0", "--input-count", "32",
        ])
        resource = read_json(output / "resource_preflight/report.json")
        if resource.get("status") != "complete":
            raise RuntimeError("full-resolution synthetic resource check failed")
        if resource.get("source", {}).get("source_manifest_stable") is not True:
            raise RuntimeError("resource check sources changed during execution")
        from mcss.data.dl3dv_benchmark import (
            Dl3dvProtocol,
            load_benchmark_metadata,
            prepare_benchmark_metadata,
        )

        reports = {}
        for mode, views in (("full", 16), ("ar", 32)):
            check_frozen()
            name = f"{mode}{views}"
            metadata = output / f"metadata_{name}.json"
            prepare_benchmark_metadata(
                args.data_root.resolve(), split, Dl3dvProtocol(mode, views), output_path=metadata,
            )
            load_benchmark_metadata(metadata)
            run_stage(name, "evaluate_dl3dv_benchmark.py", [
                "--metadata", str(metadata), "--checkpoint", training["checkpoint"],
                "--policy", "LEARNED", "--policy-checkpoint", training["policy"],
                "--output", str(output / name), "--device", "cuda:0",
                "--mode", mode, "--input-count", str(views),
                "--data-hash", download["verification_sha256"],
            ])
            report_path = output / name / "report.json"
            report = read_json(report_path)
            validate_report(report)
            reports[name] = {
                "report": str(report_path), "sha256": sha256(report_path),
                "macro_scene_mean": report["macro_scene_mean"],
                "micro_image_mean": report["micro_image_mean"],
            }
        check_frozen()
        write_json(output / "benchmark_results.json", {
            "comparison": status["comparison"], "reports": reports,
            "training": training, "download": download,
            "note": "No benchmark-driven refitting or checkpoint selection was performed.",
        })
        write_results(output, args.training_root.resolve(), reports)
        status.update(status="complete", stage="complete", reports=reports)
    except Exception as error:
        status.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        status["updated_at"] = datetime.now(UTC).isoformat()
        write_json(output / "status.json", status)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
