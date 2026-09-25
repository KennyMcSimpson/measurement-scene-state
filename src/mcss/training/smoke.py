"""Finite real-scene engineering smoke for offline dynamic training kernels."""

from __future__ import annotations

import json
import math
import platform
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
from torch import Tensor

from mcss.dynamic.carrier import DynamicSceneCarrier
from mcss.dynamic.checkpoint import load_dynamic_checkpoint, save_dynamic_checkpoint
from mcss.dynamic.config import CarrierConfig, WriteConfig
from mcss.dynamic.types import Action, OnlineObservation, hash_value
from mcss.dynamic.write_rule import DirectWriteRule
from mcss.measurements import FixedMeasurementRenderer
from mcss.training.grounded import grounded_measurement_loss, paired_context_measurement_loss
from mcss.training.supervision import TrainingSupervision
from mcss.training.write_unroll import unroll_observed_episode


@dataclass(frozen=True)
class DynamicTrainingSmokeConfig:
    carrier: CarrierConfig = field(default_factory=CarrierConfig)
    write: WriteConfig = field(default_factory=WriteConfig)
    learning_rate: float = 1e-3
    renderer_samples: int = 8
    ray_chunk_size: int = 2048

    def __post_init__(self) -> None:
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        if self.renderer_samples < 2 or self.ray_chunk_size < 1:
            raise ValueError("renderer configuration is invalid")


def run_dynamic_training_smoke(
    warmup: tuple[OnlineObservation, ...],
    stream: tuple[OnlineObservation, ...],
    supervision: TrainingSupervision,
    output_dir: str | Path,
    *,
    device: str | torch.device,
    config: DynamicTrainingSmokeConfig | None = None,
    resource_adaptation: str | None = None,
) -> dict[str, object]:
    """Run one grounded step and two ALL-write steps; this is not convergence training."""

    warmup = tuple(warmup)
    stream = tuple(stream)
    config = config or DynamicTrainingSmokeConfig()
    _validate_smoke_inputs(warmup, stream, supervision)
    target_device = torch.device(device)
    carrier = DynamicSceneCarrier(config.carrier).to(target_device)
    write_rule = DirectWriteRule(config.carrier, config.write).to(target_device)
    renderer = FixedMeasurementRenderer(
        n_samples=config.renderer_samples,
        ray_chunk_size=config.ray_chunk_size,
    ).to(target_device)
    optimizer = torch.optim.Adam(
        [*carrier.parameters(), *write_rule.parameters()], lr=config.learning_rate
    )
    records: list[dict[str, object]] = []

    optimizer.zero_grad(set_to_none=True)
    paired = paired_context_measurement_loss(
        carrier,
        warmup[:2],
        warmup[2:],
        supervision.frame_ids,
        supervision.query_cameras,
        supervision.query_rgb,
        supervision.query_depth,
        renderer,
    )
    records.append(
        _backward_and_step(
            name="grounded_context_pair",
            loss=paired.loss,
            carrier=carrier,
            write_rule=write_rule,
            optimizer=optimizer,
        )
    )

    for index in range(2):
        optimizer.zero_grad(set_to_none=True)
        state, fast, _, anchor_c2w = unroll_observed_episode(
            carrier,
            write_rule,
            warmup,
            stream,
            (Action.ALL, Action.ALL),
            f"{supervision.episode_id}:write-smoke-{index}",
        )
        loss, _ = grounded_measurement_loss(
            state,
            supervision.query_cameras,
            supervision.query_rgb,
            supervision.query_depth,
            anchor_c2w,
            renderer,
        )
        record = _backward_and_step(
            name=f"write_unroll_all_{index + 1}",
            loss=loss,
            carrier=carrier,
            write_rule=write_rule,
            optimizer=optimizer,
        )
        if (
            torch.count_nonzero(fast.delta_fuse) == 0
            or torch.count_nonzero(fast.delta_complete) == 0
        ):
            raise RuntimeError("ALL unroll did not produce nonzero fast writes")
        if float(record["write_gradient_norm"]) <= 0:
            raise RuntimeError("ALL unroll did not produce a write-rule gradient")
        if record["write_rule_hash_before"] == record["write_rule_hash_after"]:
            raise RuntimeError("ALL unroll optimizer step did not change write-rule parameters")
        record["fast_fuse_norm"] = float(fast.delta_fuse.norm().detach().cpu())
        record["fast_complete_norm"] = float(fast.delta_complete.norm().detach().cpu())
        records.append(record)

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output / "dynamic_training_smoke.pt"
    checkpoint_hash = save_dynamic_checkpoint(
        checkpoint_path,
        carrier,
        write_rule,
        phase="engineering-training-smoke-untrained",
        provenance={
            "episode_id": supervision.episode_id,
            "scene_id": supervision.scene_id,
            "split_id": "train",
            "query_vault_id": supervision.query_vault_id,
            "warmup_frame_ids": [observation.frame_id for observation in warmup],
            "stream_frame_ids": [observation.frame_id for observation in stream],
            "query_frame_ids": list(supervision.frame_ids),
            "purpose": "finite engineering smoke; not convergence training",
        },
    )
    restored_carrier, restored_rule, metadata = load_dynamic_checkpoint(
        checkpoint_path, target_device
    )
    original_state_hash = _unroll_state_hash(
        carrier, write_rule, warmup, stream, supervision.episode_id
    )
    restored_state_hash = _unroll_state_hash(
        restored_carrier, restored_rule, warmup, stream, supervision.episode_id
    )
    if original_state_hash != restored_state_hash:
        raise RuntimeError("strict checkpoint roundtrip changed the unroll output state")

    report = {
        "status": "untrained-engineering-smoke",
        "converged": False,
        "steps_completed": len(records),
        "config": asdict(config),
        "resource_adaptation": resource_adaptation,
        "episode": {
            "episode_id": supervision.episode_id,
            "scene_id": supervision.scene_id,
            "split_id": "train",
            "warmup_frame_ids": [observation.frame_id for observation in warmup],
            "stream_frame_ids": [observation.frame_id for observation in stream],
            "query_frame_ids": list(supervision.frame_ids),
            "image_size": list(warmup[0].camera.image_size),
        },
        "steps": records,
        "checkpoint": {
            "path": str(checkpoint_path.resolve()),
            "sha256": checkpoint_hash,
            "roundtrip_metadata": metadata,
            "original_unroll_state_hash": original_state_hash,
            "restored_unroll_state_hash": restored_state_hash,
        },
        "environment": _environment(target_device),
    }
    _write_json(output / "config.json", asdict(config))
    _write_json(output / "report.json", report)
    return report


def _validate_smoke_inputs(
    warmup: tuple[OnlineObservation, ...],
    stream: tuple[OnlineObservation, ...],
    supervision: TrainingSupervision,
) -> None:
    if len(warmup) != 4 or len(stream) != 2:
        raise ValueError("training smoke requires exactly four warmup and two stream observations")
    if len(supervision.frame_ids) != 4:
        raise ValueError("training smoke requires exactly four query frames")
    if warmup[0].scene_id != supervision.scene_id:
        raise ValueError("training supervision scene does not match the online observations")
    observed_ids = {observation.frame_id for observation in (*warmup, *stream)}
    if observed_ids & set(supervision.frame_ids):
        raise ValueError("training query frames overlap observed frames")
    if supervision.query_cameras.device != warmup[0].rgb.device:
        raise ValueError("training supervision and online observations must share a device")


def _backward_and_step(*, name, loss: Tensor, carrier, write_rule, optimizer) -> dict[str, object]:
    if loss.ndim != 0 or not torch.isfinite(loss):
        raise RuntimeError(f"{name} produced a nonfinite scalar loss")
    pre_hashes = _parameter_hashes(carrier, write_rule)
    loss.backward()
    carrier_gradient_norm = _gradient_norm(carrier.parameters(), f"{name} carrier")
    write_gradient_norm = _gradient_norm(write_rule.parameters(), f"{name} write rule")
    optimizer.step()
    post_hashes = _parameter_hashes(carrier, write_rule)
    if pre_hashes == post_hashes:
        raise RuntimeError(f"{name} optimizer step did not change slow parameters")
    return {
        "name": name,
        "loss": float(loss.detach().cpu()),
        "carrier_gradient_norm": carrier_gradient_norm,
        "write_gradient_norm": write_gradient_norm,
        "carrier_hash_before": pre_hashes["carrier"],
        "carrier_hash_after": post_hashes["carrier"],
        "write_rule_hash_before": pre_hashes["write_rule"],
        "write_rule_hash_after": post_hashes["write_rule"],
    }


def _gradient_norm(parameters, name: str) -> float:
    gradients = [parameter.grad for parameter in parameters if parameter.grad is not None]
    if not gradients:
        return 0.0
    if not all(torch.isfinite(gradient).all() for gradient in gradients):
        raise RuntimeError(f"{name} has nonfinite gradients")
    squared_norm = sum(gradient.detach().float().square().sum() for gradient in gradients)
    return float(torch.sqrt(squared_norm).cpu())


def _parameter_hashes(carrier, write_rule) -> dict[str, str]:
    return {
        "carrier": hash_value(carrier.state_dict()),
        "write_rule": hash_value(write_rule.state_dict()),
    }


def _unroll_state_hash(carrier, write_rule, warmup, stream, episode_id: str) -> str:
    with torch.no_grad():
        state, _, _, _ = unroll_observed_episode(
            carrier,
            write_rule,
            warmup,
            stream,
            (Action.ALL, Action.ALL),
            f"{episode_id}:roundtrip",
        )
    return hash_value(state)


def _environment(device: torch.device) -> dict[str, object]:
    return {
        "device": str(device),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cuda_device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
    }


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
