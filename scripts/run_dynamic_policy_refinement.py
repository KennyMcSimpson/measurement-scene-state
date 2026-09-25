"""Run the train-only random/learned recollection cycle and development evaluation.

This is a finite experiment driver, not a scheduler. It never opens an external
benchmark or final holdout, and each invocation requires a fresh output directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]


def write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def teacher_support(path: Path) -> dict:
    previous = Counter()
    prefixes = Counter()
    identifiers = set()
    episode_prefixes = {}
    written_rows = 0
    with path.open(encoding="utf-8") as source:
        for line in source:
            row = json.loads(line)
            if row["row_id"] in identifiers:
                raise ValueError("duplicate teacher row ID")
            identifiers.add(row["row_id"])
            previous[row["control_input"]["previous_action"]] += 1
            prefixes[row["prefix_step"]] += 1
            scene_prefixes = episode_prefixes.setdefault(row["episode_id"], set())
            if row["prefix_step"] in scene_prefixes:
                raise ValueError("duplicate prefix within a teacher episode")
            scene_prefixes.add(row["prefix_step"])
            written_rows += any(value > 0 for value in row["control_input"]["fast_state_stats"])
    return {
        "rows": len(identifiers), "previous_actions": dict(previous),
        "prefix_counts": dict(prefixes), "rows_with_nonzero_fast_state": written_rows,
        "episode_prefixes": {key: sorted(value) for key, value in episode_prefixes.items()},
    }


def rollout_support(report_path: Path, policy_path: Path) -> dict:
    import torch

    from mcss.dynamic.policy import control_to_features
    from mcss.dynamic.policy_checkpoint import load_policy_checkpoint
    from mcss.dynamic.types import ControlInput

    policy, metadata = load_policy_checkpoint(policy_path)
    normalization = metadata["normalization"]
    mean = torch.tensor(normalization["mean"])
    scale = torch.tensor(normalization["scale"])
    report = json.loads(report_path.read_text(encoding="utf-8"))
    all_z = []
    writes_per_episode = Counter()
    actions = Counter()
    for record in report["records"]:
        if record["policy"] != "LEARNED":
            continue
        writes_per_episode[sum(action != "OFF" for action in record["actions"])] += 1
        for step in record["history"]:
            actions[step["action"]] += 1
            features = control_to_features(ControlInput(**step["control_input"]))
            all_z.append(((features - mean) / scale).abs()[65:73].max())
    values = torch.stack(all_z)
    return {
        "actions": dict(actions), "writes_per_episode": dict(writes_per_episode),
        "dynamic_z_abs_quantiles_p50_p90_p99_max":
            torch.quantile(values, torch.tensor([0.5, 0.9, 0.99, 1.0])).tolist(),
        "scene_summaries": report["summaries"],
        "policy_hidden_dim": policy.hidden_dim,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-index", type=Path, required=True)
    parser.add_argument("--dev-index", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scene-limit", type=int)
    parser.add_argument("--epochs", type=int, default=100)
    args = parser.parse_args()
    output = args.output.resolve()
    output.relative_to(PROJECT / "outputs")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite refinement experiment: {output}")
    if args.scene_limit is not None and args.scene_limit < 2:
        raise ValueError("scene-limit must permit a held-out scene split")
    train_index = args.train_index.resolve()
    dev_index = args.dev_index.resolve()
    checkpoint = args.checkpoint.resolve()
    source = json.loads(train_index.read_text(encoding="utf-8"))
    if source["schema_version"] != "mcss.dynamic.episode_index.v1":
        raise ValueError("unexpected train index schema")
    first_by_scene = {}
    for entry in sorted(source["episodes"], key=lambda value: value["episode_id"]):
        if entry["split_id"] != "train":
            raise ValueError("recollection accepts only train scenes")
        if entry["stream_steps"] != 8 or entry["query_count"] != 4:
            raise ValueError("this refinement cycle requires the frozen 4+8+4 protocol")
        first_by_scene.setdefault(entry["scene_id"], entry)
    # Hash ordering spreads a limited engineering pilot over the scene-ID range.
    scene_ids = sorted(first_by_scene, key=lambda value: hashlib.sha256(value.encode()).hexdigest())
    if args.scene_limit is not None:
        scene_ids = scene_ids[: args.scene_limit]
    selected = [first_by_scene[scene] for scene in scene_ids]
    if len(selected) < 2:
        raise ValueError("not enough train scenes")
    dev_source = json.loads(dev_index.read_text(encoding="utf-8"))
    if any(entry["split_id"] != "dev" for entry in dev_source["episodes"]):
        raise ValueError("development evaluation accepts only dev scenes")
    if set(scene_ids) & {entry["scene_id"] for entry in dev_source["episodes"]}:
        raise ValueError("train/development scene overlap")
    output.mkdir(parents=True)
    selected_index = output / "episode_index_train_scene_balanced.json"
    write_json(selected_index, {"schema_version": source["schema_version"], "episodes": selected})
    source_files = sorted((PROJECT / "src/mcss/dynamic").glob("*.py"))
    source_files += sorted((PROJECT / "src/mcss/training").glob("*.py"))
    source_files += [
        Path(__file__).resolve(),
        PROJECT / "scripts/collect_action_targets.py",
        PROJECT / "scripts/train_action_policy.py",
        PROJECT / "scripts/evaluate_dynamic_checkpoint.py",
        PROJECT / "src/mcss/evaluation/checkpoint_report.py",
        PROJECT / "src/mcss/evaluation/sealed_queries.py",
        PROJECT / "src/mcss/data/episodes.py",
        PROJECT / "src/mcss/data/pilot_provenance.py",
        PROJECT / "src/mcss/geometry.py",
        PROJECT / "src/mcss/measurements.py",
        PROJECT / "src/mcss/types.py",
        PROJECT / "src/mcss/losses.py",
    ]
    plan = {
        "schema": "mcss.dynamic.policy_refinement.v1",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "train_index_sha256": sha256(train_index),
        "dev_index_sha256": sha256(dev_index),
        "selected_index_sha256": sha256(selected_index),
        "scene_ids": scene_ids,
        "scene_count": len(scene_ids),
        "selection": "first window per train scene; deterministic scene-hash order",
        "expected_rows_per_collection": len(scene_ids) * 8,
        "prefixes": list(range(8)),
        "horizon": 2,
        "continuation": "OFF",
        "random_rollin_seed": 20260920,
        "policy_fit_seed": 0,
        "policy_hidden_dim": 32,
        "epochs": args.epochs,
        "source_sha256": {str(path.relative_to(PROJECT)): sha256(path) for path in source_files},
        "scope": "train-only recollection/refit and development assessment; no final benchmark",
    }
    write_json(output / "plan.json", plan)
    status = {"status": "running", "stage": None, "stages": [], "pid": os.getpid()}
    common_collect = [
        "--index", str(selected_index), "--checkpoint", str(checkpoint), "--device", "cuda",
        "--horizon", "2", "--prefix-stride", "1", "--max-prefixes-per-episode", "8",
        "--renderer-samples", "48", "--ray-chunk-size", "2048",
    ]
    common_fit = ["--epochs", str(args.epochs), "--device", "cpu", "--seed", "0"]
    common_eval = [
        "--index", str(dev_index), "--checkpoint", str(checkpoint), "--device", "cuda",
        "--render-samples", "48", "--ray-chunk-size", "2048", "--policies", "OFF", "LEARNED",
    ]
    random_teacher = output / "teacher_random"
    learned_teacher = output / "teacher_learned"
    random_policy = output / "policy_random.pt"
    final_policy = output / "policy_recollected.pt"
    stages = [
        ("collect_random", "collect_action_targets.py", [
            *common_collect, "--output", str(random_teacher), "--rollin-policy", "RANDOM",
            "--rollin-seed", "20260920",
        ]),
        ("fit_random", "train_action_policy.py", [
            "--teacher", str(random_teacher / "action_targets.jsonl"),
            "--output", str(random_policy), *common_fit,
        ]),
        ("evaluate_random", "evaluate_dynamic_checkpoint.py", [
            *common_eval, "--policy-checkpoint", str(random_policy),
            "--output", str(output / "evaluation_dev_random"),
        ]),
        ("collect_learned", "collect_action_targets.py", [
            *common_collect, "--output", str(learned_teacher),
            "--rollin-policy", str(random_policy),
        ]),
        ("fit_recollected", "train_action_policy.py", [
            "--teacher", str(random_teacher / "action_targets.jsonl"),
            str(learned_teacher / "action_targets.jsonl"), "--output", str(final_policy),
            *common_fit,
        ]),
        ("evaluate_recollected", "evaluate_dynamic_checkpoint.py", [
            *common_eval, "--policy-checkpoint", str(final_policy),
            "--output", str(output / "evaluation_dev_recollected"),
        ]),
    ]
    env = dict(os.environ, OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", PYTHONUNBUFFERED="1")
    try:
        for name, script, arguments in stages:
            if sha256(checkpoint) != plan["checkpoint_sha256"]:
                raise RuntimeError("frozen carrier checkpoint changed during refinement")
            for relative, expected in plan["source_sha256"].items():
                if sha256(PROJECT / relative) != expected:
                    raise RuntimeError(f"experiment source changed during refinement: {relative}")
            command = [sys.executable, str(PROJECT / "scripts" / script), *arguments]
            stage = {"name": name, "started_at": datetime.now(UTC).isoformat(), "command": command}
            status["stage"] = name
            status["stages"].append(stage)
            write_json(output / "status.json", status)
            print(f"Starting {name}", flush=True)
            with (output / f"{name}.stdout.log").open("w", encoding="utf-8") as stdout:
                with (output / f"{name}.stderr.log").open("w", encoding="utf-8") as stderr:
                    completed = subprocess.run(
                        command, cwd=PROJECT, env=env, stdout=stdout, stderr=stderr, check=False,
                    )
            stage["returncode"] = completed.returncode
            stage["finished_at"] = datetime.now(UTC).isoformat()
            if completed.returncode:
                raise RuntimeError(f"{name} failed; see {name}.stderr.log")
            if name.startswith("collect_"):
                teacher = random_teacher if name == "collect_random" else learned_teacher
                progress = json.loads((teacher / "progress.json").read_text(encoding="utf-8"))
                if (progress["status"] != "complete"
                        or progress["row_count"] != plan["expected_rows_per_collection"]
                        or set(progress["completed_episode_ids"])
                        != {entry["episode_id"] for entry in selected}):
                    raise RuntimeError(f"{name} failed collection coverage validation")
                support = teacher_support(teacher / "action_targets.jsonl")
                write_json(output / f"{name}.support.json", support)
                expected_episodes = {entry["episode_id"] for entry in selected}
                if (
                    support["rows"] != plan["expected_rows_per_collection"]
                    or set(support["episode_prefixes"]) != expected_episodes
                    or any(
                        value != list(range(8)) for value in support["episode_prefixes"].values()
                    )
                ):
                    raise RuntimeError("teacher does not cover every episode/prefix exactly once")
                if name == "collect_random" and (
                    set(support["previous_actions"]) != {"OFF", "FUSE", "COMPLETE", "ALL"}
                    or support["rows_with_nonzero_fast_state"] == 0
                ):
                    raise RuntimeError("random teacher lacks updated-state/action support")
            if name.startswith("evaluate_"):
                policy = random_policy if name == "evaluate_random" else final_policy
                suffix = "random" if name == "evaluate_random" else "recollected"
                write_json(output / f"{name}.support.json", rollout_support(
                    output / f"evaluation_dev_{suffix}" / "report.json", policy,
                ))
            write_json(output / "status.json", status)
        status["status"] = "complete"
        status["stage"] = "complete"
        status["final_policy_sha256"] = sha256(final_policy)
    except Exception as error:
        status["status"] = "failed"
        status["error"] = str(error)
        raise
    finally:
        write_json(output / "status.json", status)
    print(json.dumps(status, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
