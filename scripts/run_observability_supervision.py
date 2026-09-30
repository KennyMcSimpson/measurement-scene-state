"""Run frozen matched context-only supervision states with a baseline reproduction gate."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path

import torch

from mcss.geometry import transform_cameras
from mcss.mechanism_pilot.direct_capacity_contracts import ContextOnlyLoader
from mcss.mechanism_pilot.direct_capacity_optimization import DirectOptimizationConfig
from mcss.mechanism_pilot.geometric_supervision_optimization import optimize_geometric_state
from mcss.mechanism_pilot.small_training import sha, write_json


def read(path):
    return json.loads(Path(path).read_text())


def validate(root, phase):
    config, plan, manifest = (
        read(root / name) for name in ("config.json", "state_plan.json", "scene_manifest.json")
    )
    prereg = read(root / "preregistration.json")
    for name, digest in prereg["locked_file_sha256"].items():
        if sha(root / name) != digest:
            raise PermissionError(f"Frozen input changed: {name}")
    for name, digest in config["source_sha256"].items():
        if sha(Path(name)) != digest:
            raise PermissionError(f"Frozen source changed: {name}")
    records = {r["scene_id"]: r for r in manifest["scenes"]}
    if len(records) != 17 or set(records) & set(manifest["FINAL_HOLDOUT_PROHIBITED"]):
        raise PermissionError("Exposed scene scope changed")
    if any(r["cohort_role"] != "CAPACITY_DEV_EXPOSED" for r in records.values()):
        raise PermissionError("Wrong cohort")
    if phase == "formal":
        marker = root / "baseline_reproduction.json"
        if not marker.exists() or read(marker)["status"] != "PASS":
            raise PermissionError("Fresh S0 reproduction must PASS before new supervision")
    return config, plan, manifest, records


def check(root, spec):
    directory = root / spec["optimization_path"]
    done = read(directory / "completion.json")
    config = read(root / "config.json")
    if (
        done["status"] != "PASS"
        or done["key"] != spec["key"]
        or done["source_sha256"] != config["source_sha256"]
    ):
        raise PermissionError("Completion identity changed")
    for name, field in [("state_plan.json", "plan_sha256"), ("config.json", "config_sha256")]:
        if sha(root / name) != done[field]:
            raise PermissionError("Completion protocol changed")
    for name, field in [
        ("state.pt", "state_file_sha256"),
        ("summary.json", "summary_sha256"),
        ("access.json", "access_sha256"),
    ]:
        if sha(directory / name) != done[field]:
            raise PermissionError("Completed artifact changed")
    if read(directory / "summary.json")["state_hash"] != done["state_hash"]:
        raise PermissionError("State identity mismatch")
    return done


def one(root, phase, key, device):
    config, plan, manifest, records = validate(root, phase)
    spec = next(s for s in plan["states"] if s["key"] == key)
    if (phase == "baseline") != (spec["variant"] == "S0"):
        raise PermissionError("Wrong optimization phase")
    directory = root / spec["optimization_path"]
    if directory.exists():
        check(root, spec)
        return
    access = []
    batch = ContextOnlyLoader(
        records[spec["scene_id"]],
        manifest["image_size"],
        root,
        device,
        track="RGBD",
        capacity_scene_ids=tuple(records),
        holdout_scene_ids=tuple(manifest["FINAL_HOLDOUT_PROHIBITED"]),
        access_log=access,
    ).context(spec["role"])
    inverse = torch.linalg.inv(batch.cameras.c2w[0, 0])
    batch = replace(batch, cameras=transform_cameras(batch.cameras, inverse))
    seed = 20260927 + int(
        hashlib.sha256(f"{spec['scene_id']}|{spec['role']}|RGBD".encode()).hexdigest()[:8], 16
    )
    _, summary = optimize_geometric_state(
        batch,
        torch.tensor(spec["bounds"], dtype=torch.float32, device=device),
        directory,
        variant=spec["variant"],
        seed=seed,
        config=DirectOptimizationConfig(**config["optimization"]),
    )
    write_json(directory / "access.json", access)
    validate(root, phase)
    write_json(
        directory / "completion.json",
        {
            "status": "PASS",
            "key": key,
            "plan_sha256": sha(root / "state_plan.json"),
            "config_sha256": sha(root / "config.json"),
            "source_sha256": config["source_sha256"],
            "state_file_sha256": sha(directory / "state.pt"),
            "state_hash": summary["state_hash"],
            "summary_sha256": sha(directory / "summary.json"),
            "access_sha256": sha(directory / "access.json"),
            "query_selection": False,
            "new_carrier_trained": False,
            "final_holdout_touched": False,
        },
    )


def run(root, phase, device):
    config, plan, _, _ = validate(root, phase)
    specs = [s for s in plan["states"] if (s["variant"] == "S0") == (phase == "baseline")]
    logs = root / "audit" / f"{phase}_optimization_logs"
    logs.mkdir(parents=True, exist_ok=True)

    def worker(spec):
        with (logs / (spec["key"] + ".log")).open("a") as log:
            subprocess.run(
                [
                    sys.executable,
                    __file__,
                    "--root",
                    str(root),
                    "--phase",
                    phase,
                    "--device",
                    device,
                    "--key",
                    spec["key"],
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
                env={**os.environ, "PYTHONFAULTHANDLER": "1"},
            )
        return spec["key"]

    with ThreadPoolExecutor(max_workers=config["parallel_workers"]) as pool:
        for i, f in enumerate(as_completed([pool.submit(worker, s) for s in specs]), 1):
            print(phase, i, "/", len(specs), f.result(), flush=True)
    sealed = specs if phase == "baseline" else plan["states"]
    for spec in sealed:
        check(root, spec)
    validate(root, phase)
    prereg = read(root / "preregistration.json")
    inputs = {
        str((root / name).resolve()): sha(root / name)
        for name in [*prereg["locked_file_sha256"], "preregistration.json"]
    }
    if phase == "formal":
        for sid in sorted({s["scene_id"] for s in sealed}):
            for role in ("A", "B"):
                summaries = [
                    read(root / s["optimization_path"] / "summary.json")
                    for s in sealed
                    if s["scene_id"] == sid and s["role"] == role
                ]
                for field in ("initial_state_hash", "ray_stream_sha256"):
                    if len({s[field] for s in summaries}) != 1:
                        raise PermissionError(f"Matched comparison failed: {field}")
    write_json(
        root / f"{phase}_optimization_complete.json",
        {
            "status": "PASS",
            "phase": phase,
            "states": len(sealed),
            "input_sha256": inputs,
            "all_phase_states_sealed": True,
            "final_holdout_touched": False,
            "new_carrier_trained": False,
            "dynamic_ttt_run": False,
        },
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--phase", choices=["baseline", "formal"], required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--key")
    a = p.parse_args()
    one(a.root, a.phase, a.key, a.device) if a.key else run(a.root, a.phase, a.device)
