"""Reproducible runner for diagnostic-only appearance capacity oracles."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from collections.abc import Mapping, Sequence
from pathlib import Path

import torch
from torch import Tensor

from mcss.appearance_oracle import AppearanceOracle, OracleField, OracleVariant
from mcss.config import load_config
from mcss.data.collate import collate_scene_examples
from mcss.engine import Trainer
from mcss.measurements import FixedMeasurementRenderer
from mcss.metrics import compute_metrics
from mcss.types import SceneBatch, SceneState, StateAppearance

NATIVE_VARIANTS: tuple[OracleVariant, ...] = (
    "native_shared",
    "highres_shared",
    "native_per_view",
)
TYPED_VARIANTS: tuple[OracleVariant, ...] = ("typed_shared", "typed_per_view")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Estimate the RGB capacity ceiling of fixed scene-state appearance fields."
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--field", choices=("native", "typed_appearance"), default="native")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.steps < 1:
        parser.error("--steps must be positive")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("--learning-rate must be positive and finite")

    output_dir: Path = args.output_dir
    report_path = output_dir / "oracle.json"
    if report_path.exists() and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite existing oracle report: {report_path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "oracle.jsonl"
    log_path.write_text("", encoding="utf-8")

    torch.manual_seed(args.seed)
    config = load_config(args.config)
    trainer = Trainer(config)
    trainer.load_checkpoint(args.checkpoint, load_optimizer=False)
    batch = collate_scene_examples([trainer.dataset[0]]).to(trainer.device)
    trainer.model.eval()
    with torch.no_grad():
        model_output = trainer.model(
            batch.context_rgb,
            batch.context_cameras,
            batch.target_cameras,
            batch.bounds,
        )
    field: OracleField = args.field
    reference_state = _strip_state(model_output.state, field=field)
    metric_targets = _rgb_targets(batch)
    baseline = _rgb_metrics(model_output.predictions["rgb"], metric_targets)
    renderer = FixedMeasurementRenderer(
        n_samples=config.model.n_samples,
        ray_chunk_size=config.model.ray_chunk_size,
    ).to(trainer.device)
    if sum(parameter.numel() for parameter in renderer.parameters()) != 0:
        raise RuntimeError("appearance oracle requires a parameter-free renderer")

    del model_output
    trainer.model.to("cpu")
    del trainer
    if reference_state.density_logits.device.type == "cuda":
        torch.cuda.empty_cache()

    variants_to_run = NATIVE_VARIANTS if field == "native" else TYPED_VARIANTS
    variants: dict[str, dict[str, object]] = {}
    for variant in variants_to_run:
        result = _run_variant(
            variant,
            reference_state,
            renderer,
            batch,
            field=field,
            steps=args.steps,
            learning_rate=args.learning_rate,
        )
        variants[variant] = result
        _append_jsonl(log_path, {"field": field, "variant": variant, **result})

    report = {
        "schema_version": "mcss.appearance_oracle.v1",
        "diagnostic_only": True,
        "target_labels_used": True,
        "field": field,
        "native_state_shape": list(reference_state.spatial_shape),
        "optimized_field_shape": list(reference_state.appearance_shape),
        "seed": args.seed,
        "source": {
            "config_path": str(args.config.resolve()),
            "config_sha256": _sha256(args.config),
            "checkpoint_path": str(args.checkpoint.resolve()),
            "checkpoint_sha256": _sha256(args.checkpoint),
            "scene_ids": list(batch.scene_ids),
        },
        "baseline": baseline,
        "selection": _selection_summary(variants, field=field),
        "variants": variants,
    }
    _atomic_json(report_path, report)
    print(json.dumps(report, sort_keys=True))
    return 0


def _run_variant(
    variant: OracleVariant,
    reference_state: SceneState,
    renderer: FixedMeasurementRenderer,
    batch: SceneBatch,
    *,
    field: OracleField,
    steps: int,
    learning_rate: float,
) -> dict[str, object]:
    oracle = AppearanceOracle(
        reference_state,
        target_views=batch.target_rgb.shape[1],
        variant=variant,
        field=field,
    ).to(batch.target_rgb.device)
    optimizer = torch.optim.Adam(oracle.parameters(), lr=learning_rate)
    targets = _rgb_targets(batch)
    started = time.perf_counter()
    trace: list[dict[str, float | int]] = []

    with torch.no_grad():
        initial_rgb = oracle.render_rgb(renderer, batch.target_cameras)
        initial = _rgb_metrics(initial_rgb, targets)
        best_loss = float(_masked_rgb_mse(initial_rgb, batch.target_rgb, batch.target_support))
        best_logits = oracle.color_logits.detach().clone()

    log_every = max(1, steps // 20)
    for step in range(1, steps + 1):
        optimizer.zero_grad(set_to_none=True)
        rgb = oracle.render_rgb(renderer, batch.target_cameras)
        loss = _masked_rgb_mse(rgb, batch.target_rgb, batch.target_support)
        if not torch.isfinite(loss):
            raise RuntimeError(f"{variant} produced non-finite RGB loss at step {step}")
        loss_value = float(loss.detach())
        if loss_value < best_loss:
            best_loss = loss_value
            best_logits = oracle.color_logits.detach().clone()
        loss.backward()
        gradient = oracle.color_logits.grad
        if gradient is None or not torch.isfinite(gradient).all():
            raise RuntimeError(f"{variant} produced invalid color gradients at step {step}")
        optimizer.step()
        if step == 1 or step == steps or step % log_every == 0:
            trace.append({"step": step, "rgb/mse_before_update": loss_value})

    with torch.no_grad():
        final_rgb = oracle.render_rgb(renderer, batch.target_cameras)
        final_loss = float(_masked_rgb_mse(final_rgb, batch.target_rgb, batch.target_support))
        final = _rgb_metrics(final_rgb, targets)
        if final_loss < best_loss:
            best_loss = final_loss
            best_logits = oracle.color_logits.detach().clone()
        oracle.color_logits.copy_(best_logits)
        best_rgb = oracle.render_rgb(renderer, batch.target_cameras)
        best = _rgb_metrics(best_rgb, targets)

    return {
        "steps": steps,
        "learning_rate": learning_rate,
        "parameter_count": oracle.parameter_count,
        "spatial_shape": list(oracle.spatial_shape),
        "runtime_seconds": time.perf_counter() - started,
        "initial": initial,
        "best": best,
        "final": final,
        "tail_improvement_fraction": _tail_improvement_fraction(trace),
        "trace": trace,
    }


def _rgb_targets(batch: SceneBatch) -> dict[str, Tensor]:
    targets = {"rgb": batch.target_rgb}
    if batch.target_support is not None:
        targets["support"] = batch.target_support
    return targets


def _rgb_metrics(prediction: Tensor, targets: Mapping[str, Tensor]) -> dict[str, float]:
    metrics = compute_metrics({"rgb": prediction}, targets)
    return {name: value for name, value in metrics.items() if name.startswith("rgb/")}


def _masked_rgb_mse(prediction: Tensor, target: Tensor, support: Tensor | None) -> Tensor:
    valid = torch.isfinite(target).all(dim=-3, keepdim=True)
    if support is not None:
        valid = valid & torch.isfinite(support) & (support > 0.5)
    mask = valid.expand_as(target).to(prediction.dtype)
    return ((prediction - target).square() * mask).sum() / mask.sum().clamp_min(1.0)


def _strip_state(state: SceneState, *, field: OracleField) -> SceneState:
    appearance = None
    if field == "typed_appearance":
        if state.appearance is None:
            raise ValueError("typed_appearance oracle requires SceneState.StateAppearance")
        source = state.appearance
        appearance = StateAppearance(
            source.confidence.detach().clone(),
            source.unknown_probability.detach().clone(),
            source.completion_gate.detach().clone(),
            source.provenance.detach().clone(),
            source.base_color.detach().clone(),
            source.color_logit_residual.detach().clone(),
        )
    return SceneState(
        state.density_logits.detach().clone(),
        state.color.detach().clone(),
        state.log_variance.detach().clone(),
        state.bounds.detach().clone(),
        appearance=appearance,
    )


def _tail_improvement_fraction(trace: list[dict[str, float | int]]) -> float:
    if len(trace) < 2:
        return 0.0
    tail_start = max(0, math.floor(0.8 * (len(trace) - 1)))
    start = float(trace[tail_start]["rgb/mse_before_update"])
    end = min(float(record["rgb/mse_before_update"]) for record in trace[tail_start:])
    return max(0.0, (start - end) / max(start, 1e-12))


def _selection_summary(
    variants: Mapping[str, Mapping[str, object]],
    *,
    field: OracleField,
) -> dict[str, object]:
    if field == "typed_appearance":
        shared = float(_best_metrics(variants["typed_shared"])["rgb/psnr"])
        per_view = float(_best_metrics(variants["typed_per_view"])["rgb/psnr"])
        per_view_gain = per_view - shared
        return {
            "provisional_branch": (
                "optimization_only" if per_view_gain < 0.25 else "directional_basis"
            ),
            "per_view_gain_db_over_shared": per_view_gain,
        }
    native = float(_best_metrics(variants["native_shared"])["rgb/psnr"])
    highres = float(_best_metrics(variants["highres_shared"])["rgb/psnr"])
    per_view = float(_best_metrics(variants["native_per_view"])["rgb/psnr"])
    highres_gain = highres - native
    per_view_gain = per_view - native
    if max(highres_gain, per_view_gain) < 0.25:
        branch = "optimization_only"
    elif highres_gain + 0.5 >= per_view_gain:
        branch = "highres_shared"
    else:
        branch = "directional_basis"
    return {
        "provisional_branch": branch,
        "highres_gain_db_over_native": highres_gain,
        "per_view_gain_db_over_native": per_view_gain,
        "tie_margin_db": 0.5,
    }


def _best_metrics(result: Mapping[str, object]) -> Mapping[str, object]:
    best = result["best"]
    if not isinstance(best, Mapping):
        raise TypeError("oracle best metrics must be a mapping")
    return best


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _append_jsonl(path: Path, payload: Mapping[str, object]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    temporary = path.with_name(path.name + ".part")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)
