"""Matched context-only geometry attribution, using the unchanged renderer and state."""

from __future__ import annotations

import csv
import hashlib
import time
from dataclasses import asdict
from pathlib import Path

import torch
from torch import nn

from mcss.dynamic.types import clone_scene_state, hash_scene_state
from mcss.measurements import FixedMeasurementRenderer, _sample_volume
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

VARIANTS = ("S0", "S1", "S2", "S3")
LAMBDA_FREE = 0.1
LAMBDA_SURFACE = 0.1
EPSILON = 1e-8


def surface_tau(bounds):
    """Match physical spacing in the frozen regularizer and normal-grid kernel."""
    return 0.5 * torch.linalg.vector_norm((bounds.reshape(2, 3)[1] - bounds.reshape(2, 3)[0]) / 16)


def ray_geometry(renderer, state, cache, indices):
    """Same sample positions/alpha/transmittance as FixedMeasurementRenderer._render_chunk."""
    if renderer.n_samples != 64:
        raise ValueError("Exactly 64 samples required")
    origins, directions = cache["origins"][:, indices], cache["directions"][:, indices]
    near, far, hit = cache["near"][:, indices], cache["far"][:, indices], cache["hit"][:, indices]
    fractions = (
        torch.arange(renderer.n_samples, device=origins.device, dtype=origins.dtype) + 0.5
    ) / renderer.n_samples
    distances = near.unsqueeze(-1) + (far - near).unsqueeze(-1) * fractions
    points = origins.unsqueeze(-2) + directions.unsqueeze(-2) * distances.unsqueeze(-1)
    logits = _sample_volume(state.density_logits, points, state.bounds).squeeze(-1)
    density = torch.nn.functional.softplus(logits) * hit.unsqueeze(-1)
    deltas = ((far - near) / renderer.n_samples).unsqueeze(-1)
    alpha = 1.0 - torch.exp(-density * deltas)
    transmittance = torch.cumprod(
        torch.cat((torch.ones_like(alpha[..., :1]), 1.0 - alpha + 1e-10), dim=-1), dim=-1
    )[..., :-1]
    return distances, alpha, transmittance * alpha, hit


def geometry_losses(distances, alpha, weights, hit, depth, tau):
    """All valid rays normalize free-space; only valid hit rays with band normalize surface."""
    depth = depth.squeeze(-1)
    valid = torch.isfinite(depth) & (depth > 0)
    free = (distances < depth.unsqueeze(-1) - tau) & valid.unsqueeze(-1)
    band = ((distances - depth.unsqueeze(-1)).abs() <= tau) & valid.unsqueeze(-1)
    eligible = valid & hit & band.any(dim=-1)
    free_counts = free.sum(dim=-1)
    per_ray_free = (alpha * free).sum(dim=-1) / free_counts.clamp_min(1)
    loss_free = (per_ray_free * valid).sum() / valid.sum().clamp_min(1)
    mass = (weights * band).sum(dim=-1)
    loss_surface = (-torch.log(mass + EPSILON) * eligible).sum() / eligible.sum().clamp_min(1)
    counts = {
        "valid_depth_rays": int(valid.sum()),
        "valid_hit_rays": int((valid & hit).sum()),
        "surface_eligible_rays": int(eligible.sum()),
        "empty_band_hit_rays": int((valid & hit & ~band.any(dim=-1)).sum()),
        "valid_miss_rays": int((valid & ~hit).sum()),
        "free_empty_rays": int((valid & (free_counts == 0)).sum()),
        "free_samples": int(free.sum()),
        "surface_samples": int((band & hit.unsqueeze(-1)).sum()),
    }
    return loss_free, loss_surface, counts


def _geometry(renderer, state, cache, indices, tau):
    return geometry_losses(
        *ray_geometry(renderer, state, cache, indices), cache["depth"][:, indices], tau
    )


def _combined(base, free, surface, variant):
    if variant in ("S1", "S3"):
        base = base + LAMBDA_FREE * free
    if variant in ("S2", "S3"):
        base = base + LAMBDA_SURFACE * surface
    return base


@torch.no_grad()
def full_geometry_objective(model, renderer, cache, config, variant, tau):
    result = full_objective(model, renderer, cache, config)
    counts, free_sum, surface_sum = {}, 0.0, 0.0
    for start in range(0, cache["rgb"].shape[1], 2048):
        indices = torch.arange(
            start, min(start + 2048, cache["rgb"].shape[1]), device=model.bounds.device
        )
        free, surface, c = _geometry(renderer, model.state(), cache, indices, tau)
        free_sum += float(free) * c["valid_depth_rays"]
        surface_sum += float(surface) * c["surface_eligible_rays"]
        for key, value in c.items():
            counts[key] = counts.get(key, 0) + value
    result["render_objective"] = result["context_objective"]
    result["loss_free"] = free_sum / max(counts["valid_depth_rays"], 1)
    result["loss_surface"] = surface_sum / max(counts["surface_eligible_rays"], 1)
    result["context_objective"] = _combined(
        result["render_objective"], result["loss_free"], result["loss_surface"], variant
    )
    return {**result, **counts}


def optimize_geometric_state(
    batch, bounds, output, *, variant, seed, config=None, engineering_fixture=False
):
    """Fixed final-budget state. No query metric, weight, band, or selection arguments."""
    config = config or DirectOptimizationConfig(steps=10000)
    if batch.supervision != "CONTEXT_ONLY_RGBD" or batch.depth is None:
        raise PermissionError("Only context-loader RGBD supervision is permitted")
    if variant not in VARIANTS:
        raise ValueError("Unknown frozen supervision variant")
    expected = DirectOptimizationConfig(steps=10000)
    for field, value in asdict(expected).items():
        if engineering_fixture and field in ("steps", "batch_rays", "checkpoint_interval"):
            continue
        if getattr(config, field) != value:
            raise PermissionError(f"Frozen optimizer changed: {field}")
    if config.steps <= 0 or config.batch_rays <= 0 or config.checkpoint_interval <= 0:
        raise ValueError("Positive optimizer budget required")
    output = Path(output)
    if output.exists():
        raise FileExistsError("Refusing to overwrite geometry optimization")
    output.mkdir(parents=True)
    model = DirectState(bounds, 16)
    renderer = FixedMeasurementRenderer(n_samples=64, ray_chunk_size=2048).to(bounds.device)
    cache = cache_rays(batch, bounds)
    tau = surface_tau(bounds)
    generator = torch.Generator(device=bounds.device).manual_seed(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    initial_hash = hash_scene_state(model.state())
    lock = {
        "variant": variant,
        "seed": seed,
        "config": asdict(config),
        "grid": 16,
        "samples": 64,
        "bounds": bounds.tolist(),
        "tau_surface": float(tau),
        "lambda_free": LAMBDA_FREE,
        "lambda_surface": LAMBDA_SURFACE,
        "epsilon": EPSILON,
        "frame_ids": list(batch.frame_ids),
        "supervision": batch.supervision,
        "initial_state_hash": initial_hash,
        "selection": "FIXED_BUDGET",
        "engineering_fixture": engineering_fixture,
        "query_access": False,
    }
    write_json(output / "lock.json", lock)
    if bounds.device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    checkpoints = [
        {"step": 0, **full_geometry_objective(model, renderer, cache, config, variant, tau)}
    ]
    trace, stream = [], hashlib.sha256()
    for step in range(1, config.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        indices = torch.randint(
            cache["rgb"].shape[1], (config.batch_rays,), generator=generator, device=bounds.device
        )
        stream.update(indices.detach().cpu().numpy().astype("<i8", copy=False).tobytes())
        state = model.state()
        pred = render_rays(renderer, state, cache, indices)
        rgb, depth, _ = data_losses(pred, cache, indices)
        reg = regularization(state, config)
        base = config.rgb_weight * rgb + config.depth_weight * depth + reg
        # S0 keeps exactly the old autograd graph; geometry is diagnostic only.
        with torch.set_grad_enabled(variant != "S0"):
            free, surface, counts = _geometry(renderer, state, cache, indices, tau)
        loss = _combined(base, free, surface, variant)
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite objective")
        loss.backward()
        grad = nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
        if not torch.isfinite(grad):
            raise FloatingPointError("Nonfinite gradient")
        optimizer.step()
        if step == 1 or step % 10 == 0:
            trace.append(
                {
                    "step": step,
                    "loss_rgb": float(rgb.detach()),
                    "loss_depth": float(depth.detach()),
                    "regularization": float(reg.detach()),
                    "loss_free": float(free.detach()),
                    "loss_surface": float(surface.detach()),
                    "render_objective": float(base.detach()),
                    "context_objective": float(loss.detach()),
                    "grad_norm": float(grad),
                    "density_logits_norm": float(model.density.detach().norm()),
                    "color_logits_norm": float(model.color_logits.detach().norm()),
                    "opacity": float(pred["visibility"].detach().mean()),
                    **counts,
                }
            )
        if step % config.checkpoint_interval == 0 or step == config.steps:
            checkpoints.append(
                {
                    "step": step,
                    **full_geometry_objective(model, renderer, cache, config, variant, tau),
                }
            )
    if bounds.device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    state = clone_scene_state(model.state()).to(device="cpu")
    torch.save(state, output / "state.pt")
    summary = {
        **lock,
        "status": "PASS",
        "selected_step": config.steps,
        "steps_completed": config.steps,
        "state_hash": hash_scene_state(state),
        "state_file_sha256": sha(output / "state.pt"),
        "ray_stream_sha256": stream.hexdigest(),
        "ray_stream_encoding": "step order, little-endian int64 flattened sampled indices",
        "sampled_ray_count": config.steps * config.batch_rays,
        "final_context_metrics": checkpoints[-1],
        "all_steps_finite": True,
        "seconds": elapsed,
        "query_selection": False,
        "peak_cuda_allocated_bytes": (
            torch.cuda.max_memory_allocated() if bounds.device.type == "cuda" else None
        ),
        "peak_cuda_reserved_bytes": (
            torch.cuda.max_memory_reserved() if bounds.device.type == "cuda" else None
        ),
    }
    write_json(output / "summary.json", summary)
    write_json(output / "full_objective_checkpoints.json", checkpoints)
    write_json(output / "trace.json", trace)
    with (output / "optimization_trace.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(trace[0]))
        writer.writeheader()
        writer.writerows(trace)
    return state, summary
