#!/usr/bin/env python3
"""Core B V2: evaluate the final write rule of every seed on DEV, all states sealed first."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from mcss.mechanism_pilot.geometry_carrier_experiment import read, verify_data, verify_lock
from mcss.mechanism_pilot.rgbd_stream_evaluation_v2 import evaluate_streams
from mcss.mechanism_pilot.rgbd_stream_write_v2 import (
    CONTROLS,
    POLICIES,
    load_frozen_carrier,
    load_write_rule,
)
from mcss.mechanism_pilot.small_training import sha, write_json


def evaluate(root, device):
    root = Path(root).resolve()
    manifest, config = verify_lock(root)
    verify_data(root)
    torch.set_num_threads(config["num_threads"])
    carriers = read(root / "carrier_lock.json")["carriers"]
    rows, summaries = [], []
    for seed in config["seeds"]:
        run = root / "checkpoints" / f"write_{seed}"
        summary = read(run / "summary.json")
        if summary["status"] != "COMPLETE" or summary["steps"] != config["steps"]:
            raise PermissionError("Every seed must finish its frozen write-rule budget")
        for name, key in (
            ("rule_step000000.pt", "initial_rule_sha256"),
            ("rule_final.pt", "final_rule_sha256"),
        ):
            if sha(run / name) != summary[key]:
                raise PermissionError(f"Write rule changed: {run / name}")
        lock = carriers[str(seed)]
        carrier, _ = load_frozen_carrier(lock["path"], device)
        trained, payload = load_write_rule(run / "rule_final.pt", device)
        untrained, _ = load_write_rule(run / "rule_step000000.pt", device)
        if payload["lock_sha256"] != sha(root / "preregistration.json"):
            raise PermissionError("Write rule not from the frozen training")
        out = root / "raw" / "dev_stream" / f"seed_{seed}"
        seed_rows = evaluate_streams(
            carrier, {"trained": trained, "untrained": untrained}, manifest, root, out, seed, device
        )
        for row in seed_rows:
            row["prediction_path"] = str((out / row["prediction_path"]).relative_to(root))
        rows.extend(seed_rows)
        summaries.append(summary)
    streams = {s["data_stream_sha256"] for s in summaries}
    names = (*POLICIES, *CONTROLS)
    records = {r["scene_id"]: r for r in manifest["scenes"] if r["split"] == "DEV"}
    expected = {
        (seed, sid, role, qid, name)
        for seed in config["seeds"]
        for sid, record in records.items()
        for role in ("A", "B")
        for qid in record["roles"]["primary_query"]
        for name in names
    }
    actual = {(r["seed"], r["scene_id"], r["role"], r["query_id"], r["policy"]) for r in rows}
    if actual != expected or len(actual) != len(rows):
        raise AssertionError("Incomplete Core B DEV matrix")
    state_ok = all(
        len(
            {
                r["used_state_hash"]
                for r in rows
                if (r["seed"], r["scene_id"], r["role"], r["policy"]) == key
            }
        )
        == 1
        for key in {(r["seed"], r["scene_id"], r["role"], r["policy"]) for r in rows}
    )
    write_json(root / "raw/dev_stream_query_results.json", rows)
    write_json(
        root / "audit/dev_state_use.json",
        {
            "status": "PASS" if state_ok else "FAIL",
            "one_sealed_state_per_seed_scene_role_policy": state_ok,
            "queries_per_state": 2,
            "query_after_state_seal": True,
            "distinct_seed_data_streams": len(streams),
        },
    )
    write_json(
        root / "training_curves.json",
        {
            str(seed): [
                json.loads(line)
                for line in (root / "checkpoints" / f"write_{seed}" / "training.jsonl")
                .read_text()
                .splitlines()
            ]
            for seed in config["seeds"]
        },
    )
    if not state_ok:
        raise AssertionError("State reuse audit failed")
    print("CORE_B_DEV_EVALUATION_PASS", len(rows), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    evaluate(args.root, args.device)
