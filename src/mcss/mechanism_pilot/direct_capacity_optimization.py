"""Direct typed-state optimization using the unchanged fixed renderer ray kernel.

No carrier, lifting, writer or query loader is imported. The caller must supply a
permitted observation batch through the capacity loader boundary.
"""

from __future__ import annotations

import csv
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn

from mcss.dynamic.types import clone_scene_state, hash_scene_state
from mcss.geometry import generate_rays, intersect_aabb
from mcss.measurements import FixedMeasurementRenderer, _state_normal_grid
from mcss.mechanism_pilot.small_training import write_json
from mcss.types import SceneState


@dataclass(frozen=True)
class DirectOptimizationConfig:
    steps: int = 1000
    learning_rate: float = 0.03
    batch_rays: int = 1024
    checkpoint_interval: int = 100
    rgb_weight: float = 1.0
    depth_weight: float = 1.0
    density_l2_weight: float = 1e-5
    tv_weight: float = 1e-4
    gradient_clip: float = 1.0
    selection: str = "BEST_FULL_SUPERVISION_OBJECTIVE_EARLIEST_TIE"


class DirectState(nn.Module):
    def __init__(self, bounds, grid):
        super().__init__()
        self.density = nn.Parameter(
            torch.full((1, 1, grid, grid, grid), -2.0, device=bounds.device)
        )
        self.color_logits = nn.Parameter(
            torch.zeros((1, 3, grid, grid, grid), device=bounds.device)
        )
        self.register_buffer("bounds", bounds.reshape(1, 2, 3).clone())
        self.register_buffer("log_variance", torch.full_like(self.density, -3.0))
        self._typed = SceneState(
            self.density, self.color_logits.sigmoid(), self.log_variance, self.bounds
        )

    def state(self):
        self._typed.color = self.color_logits.sigmoid()
        return self._typed


def cache_rays(batch, bounds):
    """Cache permitted supervision only; RGB_ONLY has no depth tensor to cache."""
    origins, directions = generate_rays(batch.cameras)
    origins, directions = origins.reshape(1, -1, 3), directions.reshape(1, -1, 3)
    near, far, hit = intersect_aabb(origins, directions, bounds.reshape(1, 2, 3))
    rgb = batch.rgb.permute(0, 1, 3, 4, 2).reshape(1, -1, 3)
    depth = None if batch.depth is None else batch.depth.reshape(1, -1, 1)
    return {
        "origins": origins,
        "directions": directions,
        "near": near,
        "far": far,
        "hit": hit,
        "rgb": rgb,
        "depth": depth,
    }


def render_rays(renderer, state, cache, indices):
    normal = _state_normal_grid(state)
    output, _ = renderer._render_chunk(
        state,
        normal,
        state.color,
        cache["origins"][:, indices],
        cache["directions"][:, indices],
        cache["near"][:, indices],
        cache["far"][:, indices],
        cache["hit"][:, indices],
        include_features=False,
    )
    return output


def regularization(state, config):
    density = torch.nn.functional.softplus(state.density_logits)
    penalty = config.density_l2_weight * density.square().mean()
    extent = state.bounds[0, 1] - state.bounds[0, 0]
    tv = density.new_zeros(())
    for field in (density, state.color):
        for axis, world_axis in [(-1, 0), (-2, 1), (-3, 2)]:
            spacing = extent[world_axis] / field.shape[axis]
            tv = tv + field.diff(dim=axis).abs().mean() / spacing
    return penalty + config.tv_weight * tv / 6


def data_losses(pred, cache, indices):
    rgb = (pred["rgb"] - cache["rgb"][:, indices]).square().mean()
    depth = rgb.new_zeros(())
    valid_count = 0
    if cache["depth"] is not None:
        gt = cache["depth"][:, indices]
        valid = torch.isfinite(gt) & (gt > 0)
        valid_count = int(valid.sum())
        if valid_count:
            depth = ((pred["depth"][valid] - gt[valid]).abs() / gt[valid]).mean()
    return rgb, depth, valid_count


@torch.no_grad()
def full_objective(model, renderer, cache, config):
    state = model.state()
    n = cache["rgb"].shape[1]
    sums = {"rgb": 0.0, "depth": 0.0, "valid": 0, "opacity": 0.0, "alpha_zero": 0, "alpha_one": 0}
    for start in range(0, n, 2048):
        idx = torch.arange(start, min(start + 2048, n), device=state.color.device)
        pred = render_rays(renderer, state, cache, idx)
        rgb, depth, valid = data_losses(pred, cache, idx)
        sums["rgb"] += float(rgb) * len(idx)
        sums["depth"] += float(depth) * valid
        sums["valid"] += valid
        alpha = pred["visibility"]
        sums["opacity"] += float(alpha.sum())
        sums["alpha_zero"] += int((alpha <= 1e-6).sum())
        sums["alpha_one"] += int((alpha >= 1 - 1e-6).sum())
    reg = float(regularization(state, config))
    rgb, depth = sums["rgb"] / n, sums["depth"] / max(sums["valid"], 1)
    return {
        "loss_rgb": rgb,
        "loss_depth": depth if cache["depth"] is not None else None,
        "regularization": reg,
        "opacity": sums["opacity"] / n,
        "alpha_zero_fraction": sums["alpha_zero"] / n,
        "alpha_one_fraction": sums["alpha_one"] / n,
        "context_objective": config.rgb_weight * rgb + config.depth_weight * depth + reg,
    }


def optimize_state(batch, bounds, grid, samples, output, *, seed, config=None):
    """One shared state per supplied supervision set. Never chooses using evaluation labels."""
    config = config or DirectOptimizationConfig()
    output = Path(output)
    if output.exists():
        raise FileExistsError("Refusing to overwrite a direct-state optimization")
    output.mkdir(parents=True)
    device = bounds.device
    model = DirectState(bounds, grid)
    renderer = FixedMeasurementRenderer(n_samples=samples, ray_chunk_size=2048).to(device)
    cache = cache_rays(batch, bounds)
    generator = torch.Generator(device=device).manual_seed(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    write_json(
        output / "lock.json",
        {
            "config": asdict(config),
            "seed": seed,
            "grid": grid,
            "renderer_samples": samples,
            "bounds": bounds.tolist(),
            "parameter_count": sum(p.numel() for p in model.parameters()),
            "supervision_frame_ids": list(batch.frame_ids),
            "has_depth": batch.depth is not None,
            "sampling": "uniform rays with replacement across all permitted full-resolution pixels",
            "renderer_kernel_unchanged": True,
            "carrier_loaded": False,
        },
    )
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    initial = full_objective(model, renderer, cache, config)
    best, best_step = initial["context_objective"], 0
    selected = {k: v.detach().clone() for k, v in model.state_dict().items()}
    checkpoints = [{"step": 0, **initial}]
    traces = []
    initial_hash = hash_scene_state(model.state())
    for step in range(1, config.steps + 1):
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
            raise FloatingPointError("Nonfinite optimization loss")
        loss.backward()
        grad = nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
        if not torch.isfinite(grad):
            raise FloatingPointError("Nonfinite state gradient")
        optimizer.step()
        if step == 1 or step % 10 == 0:
            traces.append(
                {
                    "step": step,
                    "loss_rgb": float(rgb.detach()),
                    "loss_depth": float(depth.detach()) if cache["depth"] is not None else None,
                    "regularization": float(reg.detach()),
                    "opacity": float(pred["visibility"].detach().mean()),
                    "grad_norm": float(grad),
                    "state_norm": float(
                        sum(p.detach().square().sum() for p in model.parameters()).sqrt()
                    ),
                    "context_metric": float(loss.detach()),
                    "kind": "sampled_training_objective",
                }
            )
        if step % config.checkpoint_interval == 0 or step == config.steps:
            values = full_objective(model, renderer, cache, config)
            checkpoints.append({"step": step, **values})
            if values["context_objective"] < best:
                best, best_step = values["context_objective"], step
                selected = {k: v.detach().clone() for k, v in model.state_dict().items()}
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    peak = torch.cuda.max_memory_allocated() if device.type == "cuda" else None
    peak_reserved = torch.cuda.max_memory_reserved() if device.type == "cuda" else None
    model.load_state_dict(selected)
    state = clone_scene_state(model.state())
    # Stored state contains detached tensors and never carries optimizer references.
    state = state.to(device="cpu")
    torch.save(state, output / "state.pt")
    torch.save(selected, output / "optimization_parameters.pt")
    selected_objective = full_objective(model, renderer, cache, config)
    changed = initial_hash != hash_scene_state(state)
    summary = {
        "selected_step": best_step,
        "selection_used_only_supervision_objective": True,
        "initial": initial,
        "selected": selected_objective,
        "full_objective_checkpoints": checkpoints,
        "state_hash": hash_scene_state(state),
        "changed": changed,
        "seconds": elapsed,
        "peak_cuda_allocated_bytes": peak,
        "peak_cuda_reserved_bytes": peak_reserved,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "nonzero_gradient_observed": any(r["grad_norm"] > 0 for r in traces),
        "has_depth": batch.depth is not None,
        "final_selected_at_budget_boundary": best_step == config.steps,
        "all_steps_finite": True,
        "shared_supervision_frame_ids": list(batch.frame_ids),
        "optimizer_limited_capacity_diagnostic": True,
    }
    write_json(output / "summary.json", summary)
    with (output / "optimization_trace.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(traces[0]))
        writer.writeheader()
        writer.writerows(traces)
    return state, summary
