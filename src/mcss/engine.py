"""Reproducible training and evaluation engine for scene-state experiments."""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import time
from collections.abc import Iterator, Mapping
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

from mcss.config import DatasetConfig, ExperimentConfig
from mcss.context_subset_consistency import (
    compute_context_subset_geometry,
    drop_context_view,
    ramped_context_subset_geometry_weight,
)
from mcss.data.collate import collate_scene_examples
from mcss.data.manifest_dataset import ManifestSceneDataset
from mcss.data.synthetic import SyntheticSceneDataset
from mcss.losses import MeasurementLoss
from mcss.metrics import compute_metrics
from mcss.model.system import MeasurementCompleteSystem, build_model
from mcss.types import SceneBatch, SceneExample, SceneState

DEFAULT_LOSS_WEIGHTS = {
    "rgb": 1.0,
    "depth": 1.0,
    "normal": 0.5,
    "point": 0.25,
    "visibility": 0.5,
    "uncertainty": 0.1,
}


class _ScalarTracker:
    """Track log-window means and an EMA that can be restored from checkpoints."""

    def __init__(self, *, ema_decay: float) -> None:
        if not 0.0 <= ema_decay < 1.0:
            raise ValueError("ema_decay must be in [0, 1)")
        self.ema_decay = float(ema_decay)
        self.window_sums: dict[str, float] = {}
        self.window_count = 0
        self.ema: dict[str, float] = {}

    def update(self, values: Mapping[str, float]) -> None:
        self.window_count += 1
        for name, raw_value in values.items():
            value = float(raw_value)
            self.window_sums[name] = self.window_sums.get(name, 0.0) + value
            previous = self.ema.get(name)
            self.ema[name] = (
                value
                if previous is None
                else self.ema_decay * previous + (1.0 - self.ema_decay) * value
            )

    def snapshot(self, *, reset_window: bool) -> dict[str, float]:
        result = {
            **{
                f"window/{name}": total / max(self.window_count, 1)
                for name, total in self.window_sums.items()
            },
            **{f"ema/{name}": value for name, value in self.ema.items()},
        }
        if reset_window:
            self.window_sums.clear()
            self.window_count = 0
        return result

    def state_dict(self) -> dict[str, object]:
        return {
            "ema_decay": self.ema_decay,
            "window_sums": dict(self.window_sums),
            "window_count": self.window_count,
            "ema": dict(self.ema),
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        saved_decay = float(state.get("ema_decay", self.ema_decay))
        if not math.isclose(saved_decay, self.ema_decay):
            raise ValueError("checkpoint telemetry ema_decay does not match config")
        self.window_sums = _float_mapping(state.get("window_sums", {}), "window_sums")
        self.window_count = int(state.get("window_count", 0))
        self.ema = _float_mapping(state.get("ema", {}), "ema")


class Trainer:
    """Own model, optimizer, data, output artifacts, and deterministic runtime state."""

    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        seed_everything(config.seed)
        self.device = resolve_device(config.device)
        self.output_dir = Path(config.training.output_dir)
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.model: MeasurementCompleteSystem = build_model(config.model).to(self.device)
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
        )
        self.amp_enabled = config.training.amp and self.device.type == "cuda"
        self.amp_dtype = (
            torch.bfloat16 if config.training.amp_dtype == "bfloat16" else torch.float16
        )
        if (
            self.amp_enabled
            and self.amp_dtype == torch.bfloat16
            and not torch.cuda.is_bf16_supported()
        ):
            raise RuntimeError(
                "bfloat16 AMP was requested but this CUDA device does not support it"
            )
        self.scaler = torch.amp.GradScaler(
            "cuda", enabled=self.amp_enabled and self.amp_dtype == torch.float16
        )
        self.optimizer_step = 0
        self.total_optimizer_steps = (
            config.training.max_steps // config.training.gradient_accumulation
        )
        self.telemetry = _ScalarTracker(ema_decay=config.training.ema_decay)
        self.context_subset_generator: torch.Generator | None = None
        if config.training.context_subset_geometry_weight > 0.0:
            self.context_subset_generator = torch.Generator(device="cpu")
            self.context_subset_generator.manual_seed(config.seed + 0x5C5C)
        weights = {
            name: DEFAULT_LOSS_WEIGHTS.get(name, 1.0) for name in config.training.measurements
        }
        self.objective = MeasurementLoss(weights)
        self.dataset = build_dataset(config.dataset, seed=config.seed)
        self.loader = DataLoader(
            self.dataset,
            batch_size=config.training.batch_size,
            shuffle=True,
            num_workers=config.training.num_workers,
            pin_memory=self.device.type == "cuda",
            collate_fn=collate_scene_examples,
            drop_last=False,
        )
        self.step = 0
        self.loaded_checkpoint_path: Path | None = None
        if config.training.resume:
            self.load_checkpoint(config.training.resume, load_optimizer=True)

    def train(self) -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        _atomic_json(self.output_dir / "resolved_config.json", asdict(self.config))
        self.model.train()
        batches = _cycle(self.loader)
        batches_per_epoch = len(self.loader)
        self.optimizer.zero_grad(set_to_none=True)
        latest_checkpoint = self._checkpoint_path(self.step)
        for step in range(self.step + 1, self.config.training.max_steps + 1):
            step_started = time.perf_counter()
            batch = next(batches).to(self.device)
            with torch.autocast(
                device_type=self.device.type,
                dtype=self.amp_dtype,
                enabled=self.amp_enabled,
            ):
                output = self.model(
                    batch.context_rgb, batch.context_cameras, batch.target_cameras, batch.bounds
                )
                predictions = output.predictions
                targets = batch.targets()
                loss, terms = self.objective(predictions, targets)
                regularization = _state_regularization(
                    output.state,
                    evidence_residual_weight=self.config.training.evidence_residual_weight,
                )
                objective_total = loss + regularization
                context_subset_result = None
                context_subset_metadata = None
                context_subset_weight = 0.0
                if self.context_subset_generator is not None:
                    dropped_index = int(
                        torch.randint(
                            batch.context_rgb.shape[1],
                            (1,),
                            generator=self.context_subset_generator,
                        ).item()
                    )
                    subset = drop_context_view(
                        batch.context_rgb,
                        batch.context_cameras,
                        dropped_index=dropped_index,
                    )
                    subset_state = self.model.state_encoder(
                        subset.rgb,
                        subset.cameras,
                        batch.bounds,
                    )
                    context_subset_result = compute_context_subset_geometry(
                        output.state,
                        subset_state,
                        predictions,
                        batch.target_cameras,
                        self.model.renderer,
                        residual_saturation_threshold=(
                            0.95 * self.config.model.completion_residual_scale
                        ),
                        collect_audit_diagnostics=(
                            step % self.config.training.log_every == 0
                        ),
                    )
                    context_subset_weight = ramped_context_subset_geometry_weight(
                        self.config.training.context_subset_geometry_weight,
                        optimizer_step=self.optimizer_step + 1,
                        warmup_steps=self.config.training.warmup_steps,
                    )
                    objective_total = (
                        objective_total + context_subset_weight * context_subset_result.loss
                    )
                    context_subset_metadata = {
                        "dropped_index": subset.dropped_index,
                        "retained_indices": list(subset.retained_indices),
                        "per_scene": [
                            {
                                "scene_id": (
                                    batch.scene_ids[index]
                                    if batch.scene_ids
                                    else None
                                ),
                                **diagnostics,
                            }
                            for index, diagnostics in enumerate(
                                context_subset_result.per_sample_diagnostics
                            )
                        ],
                    }
                total = objective_total / self.config.training.gradient_accumulation
            self.scaler.scale(total).backward()
            optimizer_updated = False
            optimizer_skipped = False
            grad_norm: float | None = None
            if step % self.config.training.gradient_accumulation == 0:
                next_optimizer_step = self.optimizer_step + 1
                learning_rate = self._learning_rate(next_optimizer_step)
                for parameter_group in self.optimizer.param_groups:
                    parameter_group["lr"] = learning_rate
                self.scaler.unscale_(self.optimizer)
                norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                grad_norm = float(norm.detach().item())
                scale_before = float(self.scaler.get_scale())
                self.scaler.step(self.optimizer)
                self.scaler.update()
                scale_after = float(self.scaler.get_scale())
                optimizer_updated = not self.scaler.is_enabled() or scale_after >= scale_before
                optimizer_skipped = not optimizer_updated
                if optimizer_updated:
                    self.optimizer_step = next_optimizer_step
                self.optimizer.zero_grad(set_to_none=True)
            self.step = step
            scalar_values = {
                "loss": float(loss.detach().item()),
                "regularization": float(regularization.detach().item()),
                "loss/total": float(objective_total.detach().item()),
                **{f"loss/{name}": float(value.detach().item()) for name, value in terms.items()},
                **_state_diagnostics(output.state),
                "time/step_seconds": time.perf_counter() - step_started,
            }
            if context_subset_result is not None:
                scalar_values.update({
                    "loss/context_subset_geometry": float(
                        context_subset_result.loss.detach().item()
                    ),
                    "loss/context_subset_depth": float(
                        context_subset_result.terms["depth"].detach().item()
                    ),
                    "loss/context_subset_visibility": float(
                        context_subset_result.terms["visibility"].detach().item()
                    ),
                    "context_subset/effective_weight": context_subset_weight,
                    "context_subset/weighted_loss": float(
                        (context_subset_weight * context_subset_result.loss).detach().item()
                    ),
                    **{
                        f"context_subset/{name}": value
                        for name, value in context_subset_result.diagnostics.items()
                    },
                })
            self.telemetry.update(scalar_values)
            if step % self.config.training.log_every == 0:
                epoch = (step - 1) // batches_per_epoch + 1
                batch_in_epoch = (step - 1) % batches_per_epoch + 1
                _append_jsonl(
                    self.output_dir / "train.jsonl",
                    {
                        "step": step,
                        "epoch": epoch,
                        "epoch_progress": step / batches_per_epoch,
                        "batch_in_epoch": batch_in_epoch,
                        "batches_per_epoch": batches_per_epoch,
                        "lr": float(self.optimizer.param_groups[0]["lr"]),
                        "grad/norm": grad_norm,
                        "amp/scale": float(self.scaler.get_scale()),
                        "amp/dtype": self.config.training.amp_dtype,
                        "optimizer/step": self.optimizer_step,
                        "optimizer/updated": optimizer_updated,
                        "optimizer/skipped": optimizer_skipped,
                        "scene_ids": [scene_id for scene_id in batch.scene_ids],
                        **_valid_ratios(batch.targets()),
                        **scalar_values,
                        **self.telemetry.snapshot(reset_window=True),
                        **_memory_stats(self.device),
                        **(
                            {
                                f"context_subset/{name}": value
                                for name, value in context_subset_result.audit_diagnostics.items()
                            }
                            if context_subset_result is not None
                            else {}
                        ),
                        **(
                            {"context_subset": context_subset_metadata}
                            if context_subset_metadata is not None
                            else {}
                        ),
                    },
                )
            if (
                step % self.config.training.checkpoint_every == 0
                or step == self.config.training.max_steps
            ):
                latest_checkpoint = self.save_checkpoint()
        return latest_checkpoint

    def _learning_rate(self, optimizer_step: int) -> float:
        if self.config.training.lr_schedule == "constant":
            return self.config.training.learning_rate
        return _scheduled_learning_rate(
            self.config.training.learning_rate,
            optimizer_step,
            self.total_optimizer_steps,
            self.config.training.warmup_steps,
            self.config.training.min_lr_ratio,
        )

    @torch.no_grad()
    def evaluate(self, checkpoint: str | Path | None = None) -> dict[str, float]:
        if checkpoint is not None:
            self.load_checkpoint(checkpoint, load_optimizer=False)
        self.model.eval()
        metric_records: list[dict[str, float]] = []
        scene_records: dict[str, list[dict[str, float]]] = {}
        evaluation_loader = DataLoader(
            self.dataset,
            batch_size=self.config.training.batch_size,
            shuffle=False,
            num_workers=self.config.training.num_workers,
            collate_fn=collate_scene_examples,
        )
        sample_count = 0
        for batch in evaluation_loader:
            batch = batch.to(self.device)
            with torch.autocast(
                device_type=self.device.type,
                dtype=self.amp_dtype,
                enabled=self.amp_enabled,
            ):
                output = self.model(
                    batch.context_rgb, batch.context_cameras, batch.target_cameras, batch.bounds
                )
            targets = batch.targets()
            for batch_index in range(batch.context_rgb.shape[0]):
                predictions_item = {
                    name: value[batch_index : batch_index + 1]
                    for name, value in output.predictions.items()
                }
                targets_item = {
                    name: value[batch_index : batch_index + 1] for name, value in targets.items()
                }
                metrics = compute_metrics(
                    predictions_item,
                    targets_item,
                    bounds=batch.bounds[batch_index : batch_index + 1],
                )
                scene_id = (
                    batch.scene_ids[batch_index]
                    if batch.scene_ids and batch.scene_ids[batch_index] is not None
                    else f"__sample_{sample_count:06d}"
                )
                metric_records.append(metrics)
                scene_records.setdefault(scene_id, []).append(metrics)
                sample_count += 1
        results = _legacy_window_macro(metric_records)
        per_scene_metrics = {
            scene_id: {
                "sample_count": len(records),
                "legacy_window_macro_metrics": _legacy_window_macro(records),
                "valid_count_weighted_metrics": _valid_count_weighted_metrics(records),
                "finite_window_counts": _finite_window_counts(records),
            }
            for scene_id, records in sorted(scene_records.items())
        }
        report = {
            "schema_version": "mcss.evaluation.v1",
            "checkpoint": _checkpoint_metadata(self.loaded_checkpoint_path, self.step),
            "resolved_config": asdict(self.config),
            "evaluation": {
                "sample_count": sample_count,
                "scene_count": len(scene_records),
            },
            "legacy_window_macro_metrics": results,
            "valid_count_weighted_metrics": _valid_count_weighted_metrics(metric_records),
            "per_scene_metrics": per_scene_metrics,
            "finite_window_counts": _finite_window_counts(metric_records),
            "aggregation_definitions": _aggregation_definitions(),
        }
        self.output_dir.mkdir(parents=True, exist_ok=True)
        _atomic_json(self.output_dir / "evaluation.json", results)
        _atomic_json(self.output_dir / "evaluation_report.json", report)
        return results

    def save_checkpoint(self) -> Path:
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        path = self._checkpoint_path(self.step)
        temporary = path.with_name(path.name + ".part")
        payload = {
            "step": self.step,
            "config": asdict(self.config),
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict(),
            "optimizer_step": self.optimizer_step,
            "telemetry": self.telemetry.state_dict(),
            "random_state": _random_state(),
        }
        if self.context_subset_generator is not None:
            payload["context_subset_generator_state"] = (
                self.context_subset_generator.get_state()
            )
        torch.save(payload, temporary)
        os.replace(temporary, path)
        self.loaded_checkpoint_path = path.resolve()
        return path

    def load_checkpoint(self, path: str | Path, *, load_optimizer: bool) -> None:
        checkpoint_path = Path(path).resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)
        payload = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(payload["model"])
        self.step = int(payload.get("step", 0))
        self.loaded_checkpoint_path = checkpoint_path
        if load_optimizer:
            self.optimizer.load_state_dict(payload["optimizer"])
            self.optimizer_step = int(
                payload.get(
                    "optimizer_step", self.step // self.config.training.gradient_accumulation
                )
            )
            if "scaler" in payload:
                self.scaler.load_state_dict(payload["scaler"])
            if "telemetry" in payload:
                self.telemetry.load_state_dict(payload["telemetry"])
            if "random_state" in payload:
                _restore_random_state(payload["random_state"])
            if (
                self.context_subset_generator is not None
                and "context_subset_generator_state" in payload
            ):
                self.context_subset_generator.set_state(
                    payload["context_subset_generator_state"].cpu()
                )

    def _checkpoint_path(self, step: int) -> Path:
        return self.checkpoint_dir / f"step_{step:06d}.pt"


def build_dataset(config: DatasetConfig, *, seed: int) -> Dataset[SceneExample]:
    if config.name == "synthetic":
        return SyntheticSceneDataset(
            length=config.length,
            image_size=config.image_size,
            context_views=config.context_views,
            target_views=config.target_views,
            seed=seed,
        )
    if config.root is None:
        raise ValueError("real datasets require a prepared manifest root")
    return ManifestSceneDataset(
        config.root,
        context_views=config.context_views,
        target_views=config.target_views,
        image_size=config.image_size,
        sample_stride=config.sample_stride,
        scene_ids=config.scene_ids,
        spatial_protocol=config.spatial_protocol,
        local_bounds_m=config.local_bounds_m,
    )


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _cycle(loader: DataLoader[SceneBatch]) -> Iterator[SceneBatch]:
    while True:
        yield from loader


def _state_regularization(
    state: SceneState,
    *,
    evidence_residual_weight: float = 0.0,
) -> Tensor:
    density = torch.nn.functional.softplus(state.density_logits)
    sparsity = density.mean() * 1e-4
    differences = [
        (density[..., 1:] - density[..., :-1]).abs().mean(),
        (density[:, :, :, 1:, :] - density[:, :, :, :-1, :]).abs().mean(),
        (density[:, :, 1:, :, :] - density[:, :, :-1, :, :]).abs().mean(),
    ]
    regularization = sparsity + 1e-5 * sum(differences)
    if evidence_residual_weight == 0.0:
        return regularization

    if state.transport_evidence is not None:
        transport = state.transport_evidence
        density_rewrite = (
            transport.surface_localization_support * transport.density_residual.abs()
        )
        if state.appearance is None:
            color_rewrite = (
                transport.appearance_confidence
                * transport.color_logit_residual.abs().mean(dim=1, keepdim=True)
            )
        else:
            appearance = state.appearance
            color_rewrite = (
                appearance.confidence
                * appearance.color_logit_residual.abs().mean(dim=1, keepdim=True)
            )
    elif state.dual_evidence is not None:
        dual = state.dual_evidence
        density_rewrite = (
            dual.surface_localization_support * dual.density_residual.abs()
        )
        if state.appearance is None:
            color_rewrite = (
                dual.appearance_confidence
                * dual.color_logit_residual.abs().mean(dim=1, keepdim=True)
            )
        else:
            appearance = state.appearance
            color_rewrite = (
                appearance.confidence
                * appearance.color_logit_residual.abs().mean(dim=1, keepdim=True)
            )
    elif state.evidence is not None:
        evidence = state.evidence
        density_rewrite = evidence.confidence * evidence.density_residual.abs()
        if state.appearance is None:
            color_rewrite = evidence.confidence * evidence.color_logit_residual.abs().mean(
                dim=1, keepdim=True
            )
        else:
            appearance = state.appearance
            color_rewrite = appearance.confidence * appearance.color_logit_residual.abs().mean(
                dim=1, keepdim=True
            )
    else:
        return regularization
    rewrite_penalty = density_rewrite.mean() + color_rewrite.mean()
    return regularization + evidence_residual_weight * rewrite_penalty


def _state_diagnostics(state: SceneState) -> dict[str, float]:
    if state.transport_evidence is not None:
        transport = state.transport_evidence
        diagnostics = {
            "state/surface_localization_support_mean": float(
                transport.surface_localization_support.detach().mean().item()
            ),
            "state/transported_appearance_confidence_mean": float(
                transport.transported_appearance_confidence.detach().mean().item()
            ),
            "state/appearance_confidence_mean": float(
                transport.appearance_confidence.detach().mean().item()
            ),
            "state/surface_peakness_mean": float(
                transport.surface_peakness.detach().mean().item()
            ),
            "state/transport_coverage_mean": float(
                transport.transport_coverage.detach().mean().item()
            ),
            "state/transport_coverage_nonzero_fraction": float(
                (transport.transport_coverage.detach() > 0).float().mean().item()
            ),
            "state/localization_unknown_mean": float(
                transport.localization_unknown_probability.detach().mean().item()
            ),
            "state/appearance_unknown_mean": float(
                transport.appearance_unknown_probability.detach().mean().item()
            ),
            "state/localization_completion_gate_mean": float(
                transport.localization_completion_gate.detach().mean().item()
            ),
            "state/appearance_completion_gate_mean": float(
                transport.appearance_completion_gate.detach().mean().item()
            ),
            "state/density_residual_abs": float(
                transport.density_residual.detach().abs().mean().item()
            ),
            "state/color_residual_abs": float(
                transport.color_logit_residual.detach().abs().mean().item()
            ),
            "state/surface_peakness_nonzero_fraction": float(
                (transport.surface_peakness.detach() > 0).float().mean().item()
            ),
            "state/localization_density_rewrite_mean": float(
                (
                    transport.surface_localization_support.detach()
                    * transport.density_residual.detach().abs()
                ).mean().item()
            ),
            "state/appearance_color_rewrite_mean": float(
                (
                    transport.appearance_confidence.detach()
                    * transport.color_logit_residual.detach().abs().mean(
                        dim=1, keepdim=True
                    )
                ).mean().item()
            ),
        }
        diagnostics.update(
            _decile_diagnostics(
                "state/surface_localization_support",
                transport.surface_localization_support,
            )
        )
        diagnostics.update(
            _decile_diagnostics(
                "state/transported_appearance_confidence",
                transport.transported_appearance_confidence,
            )
        )
        diagnostics.update(
            _decile_diagnostics("state/surface_peakness", transport.surface_peakness)
        )
        diagnostics.update(
            _decile_diagnostics("state/transport_coverage", transport.transport_coverage)
        )
    elif state.dual_evidence is not None:
        dual = state.dual_evidence
        diagnostics = {
            "state/surface_localization_support_mean": float(
                dual.surface_localization_support.detach().mean().item()
            ),
            "state/appearance_confidence_mean": float(
                dual.appearance_confidence.detach().mean().item()
            ),
            "state/surface_peakness_mean": float(
                dual.surface_peakness.detach().mean().item()
            ),
            "state/localization_unknown_mean": float(
                dual.localization_unknown_probability.detach().mean().item()
            ),
            "state/appearance_unknown_mean": float(
                dual.appearance_unknown_probability.detach().mean().item()
            ),
            "state/localization_completion_gate_mean": float(
                dual.localization_completion_gate.detach().mean().item()
            ),
            "state/appearance_completion_gate_mean": float(
                dual.appearance_completion_gate.detach().mean().item()
            ),
            "state/density_residual_abs": float(
                dual.density_residual.detach().abs().mean().item()
            ),
            "state/color_residual_abs": float(
                dual.color_logit_residual.detach().abs().mean().item()
            ),
            "state/surface_peakness_nonzero_fraction": float(
                (dual.surface_peakness.detach() > 0).float().mean().item()
            ),
            "state/localization_density_rewrite_mean": float(
                (
                    dual.surface_localization_support.detach()
                    * dual.density_residual.detach().abs()
                ).mean().item()
            ),
            "state/appearance_color_rewrite_mean": float(
                (
                    dual.appearance_confidence.detach()
                    * dual.color_logit_residual.detach().abs().mean(dim=1, keepdim=True)
                ).mean().item()
            ),
        }
        diagnostics.update(
            _decile_diagnostics(
                "state/surface_localization_support",
                dual.surface_localization_support,
            )
        )
        diagnostics.update(
            _decile_diagnostics(
                "state/appearance_confidence",
                dual.appearance_confidence,
            )
        )
        diagnostics.update(
            _decile_diagnostics("state/surface_peakness", dual.surface_peakness)
        )
    elif state.evidence is not None:
        evidence = state.evidence
        diagnostics = {
            "state/evidence_mean": float(evidence.confidence.detach().mean().item()),
            "state/unknown_mean": float(evidence.unknown_probability.detach().mean().item()),
            "state/completion_gate_mean": float(evidence.completion_gate.detach().mean().item()),
            "state/density_residual_abs": float(
                evidence.density_residual.detach().abs().mean().item()
            ),
            "state/color_residual_abs": float(
                evidence.color_logit_residual.detach().abs().mean().item()
            ),
        }
    else:
        return {}
    if state.appearance is not None:
        appearance = state.appearance
        if state.dual_evidence is None and state.transport_evidence is None:
            diagnostics.update({
                "state/appearance_evidence_mean": float(
                    appearance.confidence.detach().mean().item()
                ),
                "state/appearance_unknown_mean": float(
                    appearance.unknown_probability.detach().mean().item()
                ),
                "state/appearance_completion_gate_mean": float(
                    appearance.completion_gate.detach().mean().item()
                ),
                "state/appearance_color_residual_abs": float(
                    appearance.color_logit_residual.detach().abs().mean().item()
                ),
                "state/appearance_resolution_scale": (
                    appearance.spatial_shape[0] / state.spatial_shape[0]
                ),
            })
        else:
            diagnostics.update({
                "state/highres_appearance_confidence_mean": float(
                    appearance.confidence.detach().mean().item()
                ),
                "state/highres_appearance_unknown_mean": float(
                    appearance.unknown_probability.detach().mean().item()
                ),
                "state/highres_appearance_completion_gate_mean": float(
                    appearance.completion_gate.detach().mean().item()
                ),
                "state/highres_appearance_color_residual_abs": float(
                    appearance.color_logit_residual.detach().abs().mean().item()
                ),
                "state/appearance_resolution_scale": (
                    appearance.spatial_shape[0] / state.spatial_shape[0]
                ),
            })
    return diagnostics


def _decile_diagnostics(prefix: str, values: Tensor) -> dict[str, float]:
    probabilities = torch.arange(
        0.1,
        1.0,
        0.1,
        device=values.device,
        dtype=torch.float32,
    )
    quantiles = torch.quantile(values.detach().float(), probabilities)
    return {
        f"{prefix}_p{index * 10:02d}": float(value.item())
        for index, value in enumerate(quantiles, start=1)
    }


def _legacy_window_macro(records: list[dict[str, float]]) -> dict[str, float]:
    names = sorted({name for record in records for name in record})
    results: dict[str, float] = {}
    for name in names:
        values = [record[name] for record in records if name in record]
        if any(math.isinf(value) for value in values):
            results[name] = float("inf")
            continue
        finite = [value for value in values if math.isfinite(value)]
        if finite:
            results[name] = sum(finite) / len(finite)
    return results


def _finite_window_counts(records: list[dict[str, float]]) -> dict[str, int]:
    names = sorted({name for record in records for name in record})
    return {
        name: sum(math.isfinite(record[name]) for record in records if name in record)
        for name in names
    }


def _valid_count_weighted_metrics(records: list[dict[str, float]]) -> dict[str, float]:
    names = sorted({name for record in records for name in record})
    count_names = {
        name
        for name in names
        if name.endswith("/valid_pixels")
        or name.endswith("/valid_points")
        or name.endswith("/total_pixels")
    }
    results = {
        name: sum(
            record[name] for record in records if name in record and math.isfinite(record[name])
        )
        for name in sorted(count_names)
    }
    for name in names:
        if name in count_names or name in {"rgb/psnr", "support/coverage"}:
            continue
        weight_name = _metric_weight_name(name)
        if weight_name is None:
            continue
        weighted_values = [
            (record[name], record[weight_name])
            for record in records
            if name in record
            and weight_name in record
            and math.isfinite(record[name])
            and math.isfinite(record[weight_name])
            and record[weight_name] > 0
        ]
        total_weight = sum(weight for _, weight in weighted_values)
        if total_weight == 0:
            continue
        if name in {"depth/rmse", "point/rmse_m"}:
            value = math.sqrt(
                sum(metric**2 * weight for metric, weight in weighted_values) / total_weight
            )
        else:
            value = sum(metric * weight for metric, weight in weighted_values) / total_weight
        results[name] = value
    if "rgb/mse" in results:
        mse = results["rgb/mse"]
        results["rgb/psnr"] = float("inf") if mse == 0.0 else -10.0 * math.log10(mse)
    valid_support = results.get("support/valid_pixels")
    total_support = results.get("support/total_pixels")
    if valid_support is not None and total_support:
        results["support/coverage"] = valid_support / total_support
    return dict(sorted(results.items()))


def _metric_weight_name(name: str) -> str | None:
    prefix = name.split("/", maxsplit=1)[0]
    return {
        "rgb": "rgb/valid_pixels",
        "depth": "depth/valid_pixels",
        "normal": "normal/valid_pixels",
        "point": "point/valid_points",
        "visibility": "visibility/valid_pixels",
    }.get(prefix)


def _checkpoint_metadata(path: Path | None, step: int) -> dict[str, object]:
    if path is None:
        return {"absolute_path": None, "sha256": None, "step": step}
    return {
        "absolute_path": str(path.resolve()),
        "sha256": _sha256_file(path),
        "step": step,
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _aggregation_definitions() -> dict[str, object]:
    return {
        "legacy_window_macro": (
            "Arithmetic mean over finite per-sample window metrics; any infinite perfect-score "
            "window keeps that metric infinite for evaluation.json compatibility."
        ),
        "valid_count_weighted": (
            "Per-window metrics weighted by their valid pixel or point counts; count fields are "
            "summed across windows."
        ),
        "pooled_derivations": {
            "rgb/psnr": "Derived from the valid-count-pooled RGB MSE.",
            "depth/rmse": "Square root of pooled valid depth squared error.",
            "point/rmse_m": "Square root of pooled valid point squared error.",
            "support/coverage": "Total supported query pixels divided by total query pixels.",
        },
        "non_additive_metrics": (
            "SSIM, normal thresholds, Chamfer, F-scores, visibility IoU, and visibility F1 are "
            "computed within each window and then valid-count weighted. Point-cloud nearest "
            "neighbors are never matched across different scenes."
        ),
    }


def _append_jsonl(path: Path, record: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def _atomic_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _random_state() -> dict[str, object]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_random_state(state: dict[str, object]) -> None:
    random.setstate(state["python"])  # type: ignore[arg-type]
    np.random.set_state(state["numpy"])  # type: ignore[arg-type]
    torch_state = state["torch"]
    if not isinstance(torch_state, Tensor):
        raise TypeError("torch RNG state must be a tensor")
    torch.set_rng_state(torch_state.detach().cpu().to(dtype=torch.uint8))
    if state.get("cuda") is not None and torch.cuda.is_available():
        cuda_states = state["cuda"]
        if not isinstance(cuda_states, (list, tuple)) or not all(
            isinstance(cuda_state, Tensor) for cuda_state in cuda_states
        ):
            raise TypeError("CUDA RNG state must be a sequence of tensors")
        torch.cuda.set_rng_state_all(
            [cuda_state.detach().cpu().to(dtype=torch.uint8) for cuda_state in cuda_states]
        )


def _scheduled_learning_rate(
    base_learning_rate: float,
    optimizer_step: int,
    total_optimizer_steps: int,
    warmup_steps: int,
    min_lr_ratio: float,
) -> float:
    if base_learning_rate <= 0:
        raise ValueError("base_learning_rate must be positive")
    if total_optimizer_steps < 1 or not 1 <= optimizer_step <= total_optimizer_steps:
        raise ValueError("optimizer_step must be within the configured optimizer updates")
    if not 0 <= warmup_steps <= total_optimizer_steps:
        raise ValueError("warmup_steps must fit within total_optimizer_steps")
    if not 0.0 <= min_lr_ratio <= 1.0:
        raise ValueError("min_lr_ratio must be between zero and one")
    if warmup_steps and optimizer_step <= warmup_steps:
        return base_learning_rate * optimizer_step / warmup_steps
    decay_steps = total_optimizer_steps - warmup_steps
    if decay_steps == 0:
        return base_learning_rate
    progress = (optimizer_step - warmup_steps) / decay_steps
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return base_learning_rate * (min_lr_ratio + (1.0 - min_lr_ratio) * cosine)


def _float_mapping(value: object, name: str) -> dict[str, float]:
    if not isinstance(value, Mapping):
        raise TypeError(f"telemetry {name} must be a mapping")
    return {str(key): float(item) for key, item in value.items()}


def _valid_ratios(targets: Mapping[str, Tensor]) -> dict[str, float]:
    support = targets.get("support")
    support_valid = None if support is None else torch.isfinite(support) & (support > 0.5)
    ratios = {
        "valid/support": 1.0 if support_valid is None else float(support_valid.float().mean())
    }
    for name, target in targets.items():
        if name == "support":
            continue
        if name in {"rgb", "normal", "point"}:
            valid = torch.isfinite(target).all(dim=-3, keepdim=True)
        else:
            valid = torch.isfinite(target)
        if name == "depth":
            valid = valid & (target > 0)
        elif name == "normal":
            valid = valid & (target.square().sum(dim=-3, keepdim=True) > 1e-8)
        elif name == "point" and "visibility" in targets:
            valid = valid & (targets["visibility"] > 0.5)
        if support_valid is not None:
            valid = valid & support_valid
        ratios[f"valid/{name}"] = float(valid.float().mean().item())
    return ratios


def _memory_stats(device: torch.device) -> dict[str, float]:
    if device.type != "cuda":
        return {"memory/max_allocated_mb": 0.0, "memory/max_reserved_mb": 0.0}
    scale = 1024.0**2
    return {
        "memory/max_allocated_mb": torch.cuda.max_memory_allocated(device) / scale,
        "memory/max_reserved_mb": torch.cuda.max_memory_reserved(device) / scale,
    }
