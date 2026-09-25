"""Fail-closed gates before a finite job can open the external benchmark."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


@pytest.fixture
def continuation(monkeypatch):
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location("benchmark_continuation", scripts / (
        "run_dl3dv_after_training.py"
    ))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_training_gate_waits_and_rejects_failure_and_modified_final_weights(continuation, tmp_path):
    status = tmp_path / "pipeline_status.json"
    put(status, {"status": "running"})
    assert continuation.validate_training(tmp_path) is None
    put(status, {"status": "failed", "error": "bad camera"})
    with pytest.raises(RuntimeError, match="bad camera"):
        continuation.validate_training(tmp_path)
    checkpoint = tmp_path / "training/phase_b_final.pt"
    policy = tmp_path / "policy_refinement/policy_recollected.pt"
    checkpoint.parent.mkdir()
    policy.parent.mkdir()
    checkpoint.write_bytes(b"frozen carrier")
    policy.write_bytes(b"frozen policy")
    put(status, {
        "status": "complete", "final_carrier_sha256": continuation.sha256(checkpoint),
        "final_policy_sha256": continuation.sha256(policy),
    })
    put(tmp_path / "training/status.json", {"status": "complete", "global_step": 9000})
    for relative, policies in (
        ("evaluation_dev_fixed/report.json", ["OFF", "FUSE", "COMPLETE", "ALL"]),
        ("policy_refinement/evaluation_dev_recollected/report.json", ["OFF", "LEARNED"]),
    ):
        put(tmp_path / relative, {
            "status": "complete", "summaries": [{
                "policy": policy_name,
                "metric_valid_scene_counts": {"depth_abs_rel": 46, "rgb_psnr": 46},
                "metrics_mean": {"depth_abs_rel": .4, "rgb_psnr": 15},
            } for policy_name in policies],
        })
    assert continuation.validate_training(tmp_path)["policy"] == str(policy)
    checkpoint.write_bytes(b"different carrier")
    with pytest.raises(ValueError, match="identity mismatch"):
        continuation.validate_training(tmp_path)


def test_download_complete_flag_alone_cannot_open_benchmark(continuation, tmp_path):
    put(tmp_path / "status.json", {"status": "complete"})
    assert continuation.validate_download(tmp_path, tmp_path / "data") is None
    put(tmp_path / "completion_verification.json", {"status": "failed"})
    with pytest.raises(ValueError, match="verification has not passed"):
        continuation.validate_download(tmp_path, tmp_path / "data")


def test_partial_or_failed_metrics_cannot_be_reported_complete(continuation):
    records = [{
        "scene_id": str(index), "status": "ok", "target_count": 1,
        "per_image": [{"status": "ok", "metrics": {"psnr": 20, "ssim": .7, "lpips": .3}}],
    } for index in range(140)]
    continuation.validate_report({"status": "complete", "records": records})
    with pytest.raises(ValueError, match="140 unique"):
        continuation.validate_report({"status": "complete", "records": records[:-1]})
    records[0]["per_image"][0]["metrics"].pop("lpips")
    with pytest.raises(ValueError, match="requires PSNR"):
        continuation.validate_report({"status": "complete", "records": records})


def test_source_hashes_without_protocol_path_coverage_cannot_open_benchmark(
    continuation, tmp_path,
):
    data = tmp_path / "data"
    verification = {
        "status": "verified", "revision": continuation.DATA_REVISION,
        "data_root": str(data), "file_count": 12777, "total_bytes": 11313259491,
        "source_hash_verified": {"all_entries": True},
        "download_status": {"entry_sha256_matches": True},
    }
    path = tmp_path / "completion_verification.json"
    put(path, verification)
    with pytest.raises(ValueError, match="actual protocol frame paths"):
        continuation.validate_download(tmp_path, data)
    verification["protocol_coverage"] = {
        "status": "complete", "mapping": "transforms_frame_order",
        "protocols": ["full16", "ar32"], "required_files": 12777, "missing_files": 0,
    }
    put(path, verification)
    assert continuation.validate_download(tmp_path, data) is not None
    verification["protocol_coverage"]["missing_files"] = 1
    put(path, verification)
    with pytest.raises(ValueError, match="actual protocol frame paths"):
        continuation.validate_download(tmp_path, data)
