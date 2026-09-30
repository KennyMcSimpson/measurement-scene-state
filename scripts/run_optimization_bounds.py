"""Run locked, independent 16-grid trajectories; never use query feedback for context fitting."""

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
from mcss.mechanism_pilot.budget_state_optimization import run_budget_trajectory
from mcss.mechanism_pilot.direct_capacity_contracts import (
    ContextOnlyLoader,
    ExplicitQueryOracleLoader,
)
from mcss.mechanism_pilot.direct_capacity_optimization import DirectOptimizationConfig
from mcss.mechanism_pilot.small_training import sha, write_json

LOCK_FILES = (
    "config.json",
    "state_plan.json",
    "scene_manifest.json",
    "preregistration.json",
    "optimization_budget_lock.json",
    "bounds_lock.json",
    "bounds_contract.json",
    "decision_rules.json",
)


def read(path):
    return json.loads(Path(path).read_text())


def validate(root, phase):
    config, plan, manifest = (read(root / name) for name in LOCK_FILES[:3])
    prereg = read(root / "preregistration.json")
    for name, digest in prereg["locked_file_sha256"].items():
        if sha(root / name) != digest:
            raise PermissionError(f"Pre-registered input changed: {name}")
    for name, digest in config["source_sha256"].items():
        if sha(Path(name)) != digest:
            raise PermissionError(f"Frozen source changed: {name}")
    records = {r["scene_id"]: r for r in manifest["scenes"]}
    if len(records) != 17 or set(records) & set(manifest["FINAL_HOLDOUT_PROHIBITED"]):
        raise PermissionError("Exactly the locked seventeen exposed scenes required")
    if phase in ("oracle", "secondary"):
        marker = root / (
            "raw/context_evaluation_integrity.json"
            if phase == "oracle"
            else "primary_analysis_complete.json"
        )
        if not marker.exists() or read(marker)["status"] != "PASS":
            raise PermissionError(f"Prior phase must finish before {phase}")
    return config, plan, manifest, records


def check_completion(root, chain, config, plan):
    directory = root / chain["optimization_path"]
    done = read(directory / "completion.json")
    if (
        done["chain_key"] != chain["key"]
        or done["phase"] != chain["phase"]
        or done["plan_sha256"] != sha(root / "state_plan.json")
        or done["config_sha256"] != sha(root / "config.json")
        or done["source_hashes"] != config["source_sha256"]
    ):
        raise PermissionError("Completed trajectory differs from frozen plan/source")
    specs = [s for s in plan["states"] if s["chain_key"] == chain["key"]]
    if set(done["states"]) != {s["key"] for s in specs}:
        raise PermissionError("Completed trajectory missing budget/selection states")
    for spec in specs:
        saved = done["states"][spec["key"]]
        if (
            sha(root / spec["state_path"]) != saved["state_file_sha256"]
            or sha(root / spec["optimization_path"] / "summary.json") != saved["summary_sha256"]
        ):
            raise PermissionError("Completed state or selected-objective evidence changed")
    return done


def run_chain(root, phase, key, device):
    config, plan, manifest, records = validate(root, phase)
    chain = next(c for c in plan["chains"] if c["key"] == key and c["phase"] == phase)
    directory = root / chain["optimization_path"]
    if directory.exists():
        check_completion(root, chain, config, plan)
        print("Verified completed trajectory", key, flush=True)
        return
    record, accesses = records[chain["scene_id"]], []
    kwargs = {
        "capacity_scene_ids": tuple(records),
        "holdout_scene_ids": tuple(manifest["FINAL_HOLDOUT_PROHIBITED"]),
        "access_log": accesses,
    }
    if phase == "oracle":
        anchor_batch = ContextOnlyLoader(
            record, manifest["image_size"], root, device, track="RGB_ONLY", **kwargs
        ).context("anchor")
        anchor = anchor_batch.cameras.c2w[0, 0]
        batch = ExplicitQueryOracleLoader(
            record, manifest["image_size"], root, device, scope="QUERY_SUPERVISED_ORACLE", **kwargs
        ).diagnostic_query_supervision()
    else:
        batch = ContextOnlyLoader(
            record, manifest["image_size"], root, device, track=chain["track"], **kwargs
        ).context(chain["role"])
        anchor = batch.cameras.c2w[0, 0]
    batch = replace(batch, cameras=transform_cameras(batch.cameras, torch.linalg.inv(anchor)))
    seed = 20260927 + int(
        hashlib.sha256(
            f"{chain['scene_id']}|{chain['role']}|{chain['track']}".encode()
        ).hexdigest()[:8],
        16,
    )
    bounds = torch.tensor(chain["bounds"], dtype=torch.float32, device=device)
    settings = DirectOptimizationConfig(**config["optimization"])
    run_budget_trajectory(
        batch,
        bounds,
        directory,
        seed=seed,
        budgets=tuple(chain["budgets"]),
        config=settings,
        objective_threshold=0.001,
        gradient_threshold=0.10,
        secondary_sanity=phase == "secondary",
    )
    write_json(directory / "access.json", accesses)
    states = {}
    for spec in plan["states"]:
        if spec["chain_key"] != key:
            continue
        summary_path = root / spec["optimization_path"] / "summary.json"
        summary = read(summary_path)
        states[spec["key"]] = {
            "state_file_sha256": sha(root / spec["state_path"]),
            "state_hash": summary["state_hash"],
            "summary_sha256": sha(summary_path),
        }
    validate(root, phase)
    write_json(
        directory / "completion.json",
        {
            "phase": phase,
            "chain_key": key,
            "plan_sha256": sha(root / "state_plan.json"),
            "config_sha256": sha(root / "config.json"),
            "source_hashes": config["source_sha256"],
            "states": states,
            "query_selection": False,
            "new_carrier_trained": False,
            "final_holdout_touched": False,
        },
    )
    print("Completed trajectory", key, len(states), "states", flush=True)


def run_phase(root, phase, device):
    config, plan, _, _ = validate(root, phase)
    chains = [c for c in plan["chains"] if c["phase"] == phase]
    locked = {str(root / n): sha(root / n) for n in LOCK_FILES}
    logdir = root / "audit" / f"{phase}_trajectory_logs"
    logdir.mkdir(parents=True, exist_ok=True)

    def worker(chain):
        path = logdir / (chain["key"] + ".log")
        with path.open("a") as handle:
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
                    "--chain",
                    chain["key"],
                ],
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=True,
                env={**os.environ, "PYTHONFAULTHANDLER": "1"},
            )
        return chain["key"]

    with ThreadPoolExecutor(max_workers=config["parallel_workers"]) as pool:
        futures = [pool.submit(worker, c) for c in chains]
        for i, future in enumerate(as_completed(futures), 1):
            print(f"{phase} {i}/{len(chains)} {future.result()}", flush=True)
    for chain in chains:
        check_completion(root, chain, config, plan)
    assert all(sha(Path(p)) == h for p, h in locked.items())
    validate(root, phase)
    write_json(
        root / f"{phase}_optimization_complete.json",
        {
            "status": "PASS",
            "phase": phase,
            "chains": len(chains),
            "states": sum(s["phase"] == phase for s in plan["states"]),
            "input_hashes": locked,
            "parallel_workers": config["parallel_workers"],
            "all_budgets_complete_before_query_evaluation": True,
            "final_holdout_touched": False,
            "new_carrier_trained": False,
            "dynamic_ttt_run": False,
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--phase", choices=["context", "oracle", "secondary"], required=True)
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--chain")
    args = parser.parse_args()
    if args.chain:
        run_chain(args.root, args.phase, args.chain, args.device)
    else:
        run_phase(args.root, args.phase, args.device)
