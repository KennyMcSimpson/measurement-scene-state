"""Tiny synthetic phase seals, controls, reference matching, and protected-data guards."""

import json
from pathlib import Path

import pytest
import torch
from test_direct_capacity_evaluation import setup as old_setup

from mcss.dynamic.types import hash_scene_state
from mcss.mechanism_pilot.direct_capacity_contracts import StateSealBarrier
from mcss.mechanism_pilot.observability_supervision_evaluation import (
    baseline_comparison,
    ensure_phase,
    evaluate_supervision,
)
from mcss.mechanism_pilot.small_training import sha, write_json


def setup(root):
    torch.set_num_threads(1)
    previous = old_setup(root)
    specs = []
    for old in previous:
        state = torch.load(root / old["state_path"], weights_only=False)
        state.density_logits = (
            state.density_logits.repeat_interleave(2, -1)
            .repeat_interleave(2, -2)
            .repeat_interleave(2, -3)
        )
        state.color = (
            state.color.repeat_interleave(2, -1).repeat_interleave(2, -2).repeat_interleave(2, -3)
        )
        state.log_variance = torch.zeros_like(state.density_logits)
        for variant in ("S0", "S1", "S2", "S3"):
            key = f"{old['scene_id']}__{old['role']}__{variant}"
            folder = root / "checkpoints" / key
            folder.mkdir()
            torch.save(state, folder / "state.pt")
            spec = {
                "key": key,
                "scene_id": old["scene_id"],
                "role": old["role"],
                "variant": variant,
                "grid": 16,
                "samples": 64,
                "bounds": state.bounds.reshape(2, 3).tolist(),
                "state_path": str((folder / "state.pt").relative_to(root)),
                "optimization_path": str(folder.relative_to(root)),
            }
            write_json(
                folder / "summary.json",
                {
                    "state_hash": hash_scene_state(state),
                    "variant": variant,
                    "supervision": "CONTEXT_ONLY_RGBD",
                    "selection": "FIXED_BUDGET",
                    "selected_step": 10000,
                },
            )
            write_json(folder / "access.json", [])
            specs.append(spec)
    sources = {str(Path(__file__).resolve()): sha(Path(__file__))}
    config = {
        "source_sha256": sources,
        "baseline_reproduction": {"mean_atol": 0.001, "per_row_atol": 0.01},
    }
    write_json(root / "config.json", config)
    write_json(root / "state_plan.json", {"states": specs})
    write_json(
        root / "preregistration.json",
        {
            "locked_file_sha256": {
                name: sha(root / name)
                for name in ("config.json", "state_plan.json", "scene_manifest.json")
            }
        },
    )
    for spec in specs:
        folder = root / spec["optimization_path"]
        write_json(
            folder / "completion.json",
            {
                "status": "PASS",
                "key": spec["key"],
                "plan_sha256": sha(root / "state_plan.json"),
                "config_sha256": sha(root / "config.json"),
                "source_sha256": sources,
                "state_file_sha256": sha(folder / "state.pt"),
                "state_hash": json.loads((folder / "summary.json").read_text())["state_hash"],
                "summary_sha256": sha(folder / "summary.json"),
                "access_sha256": sha(folder / "access.json"),
                "query_selection": False,
            },
        )
    for phase, count in [("baseline", 4), ("formal", 16)]:
        write_json(
            root / f"{phase}_optimization_complete.json",
            {
                "status": "PASS",
                "phase": phase,
                "states": count,
                "input_sha256": {
                    str(root / name): sha(root / name)
                    for name in (
                        "config.json",
                        "state_plan.json",
                        "scene_manifest.json",
                        "preregistration.json",
                    )
                },
            },
        )
    write_json(root / "baseline_reproduction.json", {"status": "PASS"})
    return specs, config


def test_full_phase_controls_seal_before_media_and_baseline_replay(tmp_path, monkeypatch):
    specs, config = setup(tmp_path)
    count = []
    original = StateSealBarrier.seal

    def observe(self, *args):
        original(self, *args)
        count.append(1)

    def guarded_sha(path):
        if str(path).endswith((".png", ".npy")):
            assert len(count) == len(specs)
        return sha(path)

    monkeypatch.setattr(StateSealBarrier, "seal", observe)
    monkeypatch.setattr(
        "mcss.mechanism_pilot.observability_supervision_evaluation.sha", guarded_sha
    )
    rows = evaluate_supervision(tmp_path, "formal", "cpu")
    assert len(rows) == 80
    for spec in specs:
        subset = [r for r in rows if r["key"] == spec["key"]]
        for fid in (8, 9):
            assert len({r["query_camera_hash"] for r in subset if r["query_id"] == fid}) == 1
        direct = [r for r in subset if r["method"] == "direct"]
        assert len({r["used_state_hash"] for r in direct}) == 1
        for row in subset:
            assert row["prediction_file_sha256"] == sha(tmp_path / row["prediction_path"])
            if row["method"] == "zero":
                assert row["opacity"] == 0
    events = json.loads((tmp_path / "raw/formal_seal_events.json").read_text())
    assert all(e["event"] == "seal" for e in events[:16])
    assert all(e["event"] == "query_camera_GT" for e in events[16:])
    contexts = json.loads((tmp_path / "raw/context_results.json").read_text())
    query = [r for r in rows if r["variant"] == "S0" and r["method"] == "direct"]
    context = [r for r in contexts if r["variant"] == "S0"]
    write_json(tmp_path / "raw/historical_baseline_query.json", query)
    write_json(tmp_path / "raw/historical_baseline_context.json", context)
    assert baseline_comparison(tmp_path, query, context, config)["status"] == "PASS"
    altered = [{**r, "depth_absrel": r["depth_absrel"] + 0.02} for r in query]
    assert baseline_comparison(tmp_path, altered, context, config)["status"] == "FAIL"
    with pytest.raises(PermissionError, match="Baseline reproduction"):
        ensure_phase(tmp_path, "formal")
    monkeypatch.setattr("mcss.mechanism_pilot.observability_supervision_evaluation.sha", sha)
    baseline = evaluate_supervision(tmp_path, "baseline", "cpu")
    assert len(baseline) == 8
    assert json.loads((tmp_path / "baseline_reproduction.json").read_text())["status"] == "PASS"


def test_incomplete_and_holdout_rejected_before_media(tmp_path, monkeypatch):
    specs, _ = setup(tmp_path)

    def forbid(*args, **kwargs):
        raise AssertionError("Media accessed before guards")

    monkeypatch.setattr("PIL.Image.open", forbid)
    (tmp_path / specs[-1]["state_path"]).unlink()
    with pytest.raises(PermissionError, match="All states"):
        evaluate_supervision(tmp_path, "formal", "cpu")
    manifest = json.loads((tmp_path / "scene_manifest.json").read_text())
    manifest["FINAL_HOLDOUT_PROHIBITED"].append(manifest["scenes"][0]["scene_id"])
    write_json(tmp_path / "scene_manifest.json", manifest)
    with pytest.raises(PermissionError, match="prohibited"):
        evaluate_supervision(tmp_path, "baseline", "cpu")
