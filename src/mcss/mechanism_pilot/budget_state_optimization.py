"""One frozen 16^3/64-sample trajectory, multiple predeclared budget readouts.

Uses the previous direct-state parameterization, renderer kernel and objective
unchanged. No query loader, carrier or writer is available to this module.
"""

from __future__ import annotations

import csv
import math
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from mcss.dynamic.types import clone_scene_state, hash_scene_state
from mcss.measurements import FixedMeasurementRenderer
from mcss.mechanism_pilot.direct_capacity_optimization import (
    DirectOptimizationConfig,
    DirectState,
    cache_rays,
    data_losses,
    full_objective,
    regularization,
    render_rays,
)
from mcss.mechanism_pilot.small_training import sha, write_json


def plateau_diagnostic(checkpoints, budget, *, trace, objective_threshold, gradient_threshold):
    """Frozen last10% window; median gradient change is descriptive, not convergence proof."""
    rows = [r for r in checkpoints if 0.9 * budget <= r["step"] <= budget]
    gradients = [
        r["gradient_norm_before_clip"] for r in trace if 0.9 * budget <= r["step"] <= budget
    ]
    result = {
        "budget": budget,
        "window_start_step": 0.9 * budget,
        "checkpoint_steps": [r["step"] for r in rows],
        "n_checkpoints": len(rows),
        "n_trace_gradients": len(gradients),
        "objective_threshold": objective_threshold,
        "gradient_threshold": gradient_threshold,
        "automatic_early_stop": False,
        "gradient_note": "sampled optimizer-batch gradients before clipping",
    }
    if len(rows) < 2 or len(gradients) < 4:
        return {**result, "status": "INSUFFICIENT_CHECKPOINTS_OR_TRACE", "plateau": False}
    improvement = (rows[0]["context_objective"] - rows[-1]["context_objective"]) / max(
        abs(rows[0]["context_objective"]), 1e-8
    )
    midpoint = len(gradients) // 2
    before, after = float(np.median(gradients[:midpoint])), float(np.median(gradients[midpoint:]))
    change = abs(after - before) / max(abs(before), 1e-8)
    return {
        **result,
        "status": "DIAGNOSTIC",
        "relative_objective_improvement": improvement,
        "gradient_median_first_half": before,
        "gradient_median_second_half": after,
        "relative_gradient_median_abs_change": change,
        "objective_plateau": abs(improvement) <= objective_threshold,
        "gradient_stable": change <= gradient_threshold,
        "plateau": abs(improvement) <= objective_threshold and change <= gradient_threshold,
    }


def _validate(budgets, config, engineering_fixture, secondary_sanity):
    if not isinstance(budgets, tuple) or not budgets or budgets != tuple(sorted(set(budgets))):
        raise ValueError("Predeclared budgets must be a nonempty increasing immutable tuple")
    if any(type(x) is not int or x <= 0 for x in budgets) or config.steps != budgets[-1]:
        raise ValueError("Maximum trajectory length must equal largest frozen budget")
    expected = DirectOptimizationConfig(steps=config.steps)
    if not engineering_fixture:
        expected_budgets = (10000,) if secondary_sanity else (1000, 3000, 10000)
        if budgets != expected_budgets:
            raise ValueError("Production budgets differ from the frozen main/secondary protocol")
        if config != expected:
            raise PermissionError("Budget study cannot change frozen optimizer/loss/ray sampling")
    if config.checkpoint_interval < 1 or config.batch_rays < 1:
        raise ValueError("Positive checkpoint interval and ray minibatch required")
    if any(b % config.checkpoint_interval for b in budgets):
        raise ValueError("Budgets must coincide with full-objective checkpoints")


def run_budget_trajectory(
    batch,
    bounds,
    output,
    *,
    seed,
    budgets=(1000, 3000, 10000),
    config=None,
    objective_threshold=0.001,
    gradient_threshold=0.10,
    engineering_fixture=False,
    secondary_sanity=False,
):
    config = config or DirectOptimizationConfig(steps=budgets[-1])
    if secondary_sanity and (
        batch.supervision != "CONTEXT_ONLY_RGB_ONLY" or batch.depth is not None
    ):
        raise PermissionError(
            "Secondary single-budget run requires explicit context RGB_ONLY supervision"
        )
    _validate(budgets, config, engineering_fixture, secondary_sanity)
    if any(not math.isfinite(x) or x <= 0 for x in (objective_threshold, gradient_threshold)):
        raise ValueError("Finite positive preregistered plateau thresholds required")
    output = Path(output)
    if output.exists():
        raise FileExistsError("Preserve existing trajectory; no implicit restart or resume")
    output.mkdir(parents=True)
    device = bounds.device
    model = DirectState(bounds, 16)
    renderer = FixedMeasurementRenderer(n_samples=64, ray_chunk_size=2048).to(device)
    cache = cache_rays(batch, bounds)
    generator = torch.Generator(device=device).manual_seed(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    write_json(
        output / "lock.json",
        {
            "budgets": list(budgets),
            "config": asdict(config),
            "seed": seed,
            "grid": 16,
            "samples": 64,
            "bounds": bounds.tolist(),
            "supervision_frame_ids": list(batch.frame_ids),
            "supervision": batch.supervision,
            "has_depth": batch.depth is not None,
            "parameter_count": sum(p.numel() for p in model.parameters()),
            "objective_threshold": objective_threshold,
            "gradient_threshold": gradient_threshold,
            "selection": (
                "minimum full supervision objective among checkpoints<=budget; earliest tie"
            ),
            "single_shared_trajectory": True,
            "query_selection": False,
            "new_carrier_trained": False,
            "dynamic_ttt_run": False,
            "engineering_fixture": engineering_fixture,
            "secondary_sanity": secondary_sanity,
        },
    )
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    checkpoints, outputs, traces, costs = [], [], [], []
    snapshots = {0, 100, 300, *budgets}
    best, best_step, best_state = math.inf, None, None

    def checkpoint(step, gradient):
        nonlocal best, best_step, best_state
        values = full_objective(model, renderer, cache, config)
        values.update(
            {
                "step": step,
                "gradient_norm_before_clip": gradient,
                "density_logits_norm": float(model.density.detach().norm()),
                "color_logits_norm": float(model.color_logits.detach().norm()),
                "color_norm": float(model.color_logits.detach().sigmoid().norm()),
            }
        )
        checkpoints.append(values)
        if values["context_objective"] < best:
            best, best_step = values["context_objective"], step
            best_state = clone_scene_state(model.state()).to(device="cpu")
        if step in snapshots:
            snap = output / "snapshots" / f"step_{step}.pt"
            snap.parent.mkdir(exist_ok=True)
            torch.save(clone_scene_state(model.state()).to(device="cpu"), snap)
        write_json(output / "full_objective_checkpoints.json", checkpoints)
        return values

    checkpoint(0, 0.0)
    with (output / "optimization_trace.csv").open("w", newline="") as handle:
        writer = None
        for step in range(1, budgets[-1] + 1):
            optimizer.zero_grad(set_to_none=True)
            indices = torch.randint(
                cache["rgb"].shape[1], (config.batch_rays,), generator=generator, device=device
            )
            state = model.state()
            pred = render_rays(renderer, state, cache, indices)
            rgb, depth, _ = data_losses(pred, cache, indices)
            reg = regularization(state, config)
            loss = config.rgb_weight * rgb + config.depth_weight * depth + reg
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite trajectory objective")
            loss.backward()
            gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            if not torch.isfinite(gradient):
                raise FloatingPointError("Nonfinite trajectory gradient")
            optimizer.step()
            if step == 1 or step % 10 == 0:
                row = {
                    "step": step,
                    "loss_rgb": float(rgb.detach()),
                    "loss_depth": float(depth.detach()) if batch.depth is not None else None,
                    "regularization": float(reg.detach()),
                    "sampled_objective": float(loss.detach()),
                    "opacity": float(pred["visibility"].detach().mean()),
                    "gradient_norm_before_clip": float(gradient),
                    "density_logits_norm": float(model.density.detach().norm()),
                    "color_logits_norm": float(model.color_logits.detach().norm()),
                    "color_norm": float(model.color_logits.detach().sigmoid().norm()),
                }
                if writer is None:
                    writer = csv.DictWriter(handle, fieldnames=list(row))
                    writer.writeheader()
                traces.append(row)
                writer.writerow(row)
                handle.flush()
            if step % config.checkpoint_interval == 0:
                values = checkpoint(step, float(gradient))
                if step in budgets:
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    elapsed = time.perf_counter() - started
                    fixed = clone_scene_state(model.state()).to(device="cpu")
                    plateau = plateau_diagnostic(
                        checkpoints,
                        step,
                        trace=traces,
                        objective_threshold=objective_threshold,
                        gradient_threshold=gradient_threshold,
                    )
                    costs.append(
                        {
                            "budget": step,
                            "seconds": elapsed,
                            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated()
                            if device.type == "cuda"
                            else None,
                            "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved()
                            if device.type == "cuda"
                            else None,
                        }
                    )
                    for rule, result_state, selected_step, selected_objective in [
                        ("FIXED_BUDGET", fixed, step, values["context_objective"]),
                        ("CONTEXT_SELECTED", best_state, best_step, best),
                    ]:
                        folder = output / f"budget_{step}" / rule
                        folder.mkdir(parents=True)
                        state_path = folder / "state.pt"
                        torch.save(result_state, state_path)
                        summary = {
                            "budget": step,
                            "rule": rule,
                            "selected_step": selected_step,
                            "selected_full_objective": selected_objective,
                            "state_hash": hash_scene_state(result_state),
                            "state_file_sha256": sha(state_path),
                            "grid": 16,
                            "samples": 64,
                            "shared_supervision_frame_ids": list(batch.frame_ids),
                            "has_depth": batch.depth is not None,
                            "context_at_budget": values,
                            "selected_context_metrics": next(
                                c for c in checkpoints if c["step"] == selected_step
                            ),
                            "cumulative_optimization_seconds": elapsed,
                            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated()
                            if device.type == "cuda"
                            else None,
                            "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved()
                            if device.type == "cuda"
                            else None,
                            "plateau_diagnostic": plateau,
                            "query_selection": False,
                        }
                        write_json(folder / "summary.json", summary)
                        outputs.append(
                            {
                                "budget": step,
                                "rule": rule,
                                "state_path": str(state_path.relative_to(output)),
                                **summary,
                            }
                        )
    report = {
        "status": "PASS",
        "budgets": list(budgets),
        "steps_completed": budgets[-1],
        "single_trajectory": True,
        "outputs": outputs,
        "full_objective_checkpoints": checkpoints,
        "trace": traces,
        "budget_costs": costs,
        "all_steps_finite": True,
        "seconds": time.perf_counter() - started,
        "query_selection": False,
        "new_carrier_trained": False,
        "dynamic_ttt_run": False,
    }
    write_json(output / "trajectory_summary.json", report)
    return report
