"""Engineering pilot composition: online runner and post-seal evaluator meet only here."""

import copy
import json
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import torch

from mcss.data.episodes import OnlineEpisodeSource
from mcss.dynamic.carrier import DynamicSceneCarrier
from mcss.dynamic.config import CarrierConfig, WriteConfig
from mcss.dynamic.policy import FixedPolicy, ResidualThresholdPolicy
from mcss.dynamic.runner import StreamingRunner
from mcss.dynamic.types import Action, hash_value
from mcss.dynamic.write_rule import DirectWriteRule
from mcss.evaluation.sealed_queries import evaluate_sealed
from mcss.measurements import FixedMeasurementRenderer


def run_pilot(
    index,
    output_dir,
    *,
    device="cpu",
    seed=17,
    policies=("OFF", "FUSE", "COMPLETE", "ALL"),
    max_units=1e12,
):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "report.json").exists():
        raise FileExistsError("Use a new output directory to preserve previous pilot evidence")
    if not index or any(entry["split_id"] not in {"train", "dev"} for entry in index):
        raise ValueError("Engineering pilot only accepts declared train/dev episodes")
    torch.manual_seed(seed)
    torch.set_num_threads(4)
    config, write_config = CarrierConfig(), WriteConfig()
    base_carrier = DynamicSceneCarrier(config).to(device)
    base_write = DirectWriteRule(config, write_config).to(device)
    source_hash = hash_value(
        {"carrier": base_carrier.state_dict(), "write_rule": base_write.state_dict()}
    )
    resolved = {
        "carrier": asdict(config),
        "write": asdict(write_config),
        "seed": seed,
        "device": str(device),
        "policies": list(policies),
        "max_units": max_units,
        "renderer_n_samples": 16,
        "index": index,
        "initialization": "random_untrained_shared_across_policies",
    }
    (output / "resolved_config.json").write_text(
        json.dumps(resolved, indent=2) + "\n", encoding="utf-8"
    )
    records = []
    for entry in index:
        for name in policies:
            source = OnlineEpisodeSource.from_file(entry["online_manifest"], device=device)
            if source.scene_id != entry["scene_id"] or source.split_id != entry["split_id"]:
                raise ValueError("Pilot index and online manifest disagree")
            query_count = int(entry["query_count"])
            warmup = source.warmup()
            policy = (
                ResidualThresholdPolicy()
                if name == "RESIDUAL_THRESHOLD"
                else FixedPolicy(Action(name))
            )
            renderer = FixedMeasurementRenderer(n_samples=16)
            runner = StreamingRunner(
                copy.deepcopy(base_carrier),
                copy.deepcopy(base_write),
                renderer,
                policy,
                max_units=max_units,
            )
            if runner.checkpoint_hash != source_hash:
                raise RuntimeError("Internal controls did not start from identical weights")
            if torch.device(device).type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
                torch.cuda.synchronize(device)
            started = perf_counter()
            runner.reset(
                warmup,
                episode_id=source.episode_id,
                scene_id=source.scene_id,
                split_id=source.split_id,
                query_vault_id=source.query_vault_id,
                stream_steps=source.remaining_steps,
                query_count=query_count,
            )
            while source.remaining_steps:
                runner.step(source.next_observation())
            sealed = runner.seal()
            if torch.device(device).type == "cuda":
                torch.cuda.synchronize(device)
            adaptation_seconds = perf_counter() - started
            evaluation_started = perf_counter()
            metrics = evaluate_sealed(sealed, entry["query_manifest"], renderer)
            if len(metrics["per_query"]) != query_count:
                raise ValueError("Query count differs from the declared pilot protocol")
            if torch.device(device).type == "cuda":
                torch.cuda.synchronize(device)
            evaluation_seconds = perf_counter() - evaluation_started
            # Query count is declared by the protocol, never inferred by the policy from labels.
            for _ in range(query_count):
                runner.budget.record("query_render", runner.budget.render_units)
            peak = (
                torch.cuda.max_memory_allocated(device)
                if torch.device(device).type == "cuda"
                else None
            )
            record = {
                "episode_id": source.episode_id,
                "scene_id": source.scene_id,
                "split_id": source.split_id,
                "policy": name,
                "trained_policy": False,
                "state_hash": sealed.state_hash,
                "checkpoint_hash": sealed.checkpoint_hash,
                "fast_state_hash": sealed.fast_state_hash,
                "observed_ids": list(sealed.observed_ids),
                "adaptation_seconds": adaptation_seconds,
                "evaluation_seconds": evaluation_seconds,
                "peak_cuda_allocated_bytes": peak,
                "metrics": metrics,
                "budget": runner.budget.report(),
                "history": runner.history,
            }
            path = output / f"{source.episode_id}_{name}.json"
            path.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8")
            records.append(record)
            print(
                f"Completed {source.scene_id} {name}: {adaptation_seconds:.2f}s adaptation",
                flush=True,
            )
    groups = defaultdict(list)
    for record in records:
        groups[record["split_id"]].append(record["scene_id"])
    report = {
        "schema": "mcss.dynamic.engineering_pilot.v1",
        "status": "complete",
        "scientific_performance_evidence": False,
        "initialization": resolved["initialization"],
        "purpose": "Validate software execution and isolation, not method accuracy",
        "configuration_hash": hash_value(resolved),
        "shared_checkpoint_hash": source_hash,
        "episodes": len(index),
        "runs": len(records),
        "scenes_by_split": {key: sorted(set(value)) for key, value in groups.items()},
        "final_holdout_used": False,
        "records": records,
    }
    (output / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return report
