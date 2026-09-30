"""V4: minimal non-learned plane-sweep geometry cue for the dense 16^3 learned carrier.

For every context view taken as reference, raw-RGB photometric variance against the other
context views is computed on 32 fronto-parallel inverse-depth planes (z-depth) spanning the
frozen TRAIN prior [near, far]; all views are area-downsampled to 32x40 and compared at that
resolution with correspondingly scaled intrinsics. A per-pixel softmax over depth
(temperature exp(log_tau), the only learned sweep parameter) gives a depth distribution. Every
voxel reads, per reference, a surface likelihood (D * P at its depth) and a free-space
likelihood (probability that the surface lies beyond it), averaged over valid references,
plus the valid-reference fraction. The three statistics enter the fusion through a
zero-initialized bias-free projection, so a sweep carrier starts exactly equal to the matched
dense carrier. Only context RGB, context cameras and the TRAIN prior are read.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn.functional as functional
from torch import nn

from mcss.dynamic.carrier import DynamicSceneCarrier, _CarrierComputation
from mcss.dynamic.config import CarrierConfig
from mcss.dynamic.lifting import lift_cached_observations
from mcss.dynamic.types import WriteTrace, hash_value
from mcss.geometry import project_world
from mcss.mechanism_pilot import geometry_carrier
from mcss.mechanism_pilot.dense_evidence_carrier import build_state
from mcss.mechanism_pilot.small_training import sha
from mcss.types import Cameras

EXPERIMENT = "EXP-3D-PLANE-SWEEP-CARRIER-V4"
GRID = (16, 16, 16)
TOKEN_COUNT = 4096
PLANES = 32
REF_SIZE = (32, 40)
LOG_TAU0 = math.log(0.01)
VARIANT_SPECS = {
    "C0": {"label": "DENSE_SURFACE", "sweep": False, "loss": "C1"},
    "C1": {"label": "SWEEP_SURFACE", "sweep": True, "loss": "C1"},
    "C2": {"label": "SWEEP_NO_SURFACE", "sweep": True, "loss": "C0"},
}
VARIANTS = tuple(VARIANT_SPECS)
PRIMARY_PAIR = ("C0", "C1")
SCHEMA = "mcss.plane_sweep_carrier.v4"

__all__ = [
    "EXPERIMENT",
    "PRIMARY_PAIR",
    "VARIANTS",
    "VARIANT_SPECS",
    "PlaneSweepCarrier",
    "build_state",
    "load_checkpoint",
    "make_carrier",
    "parameter_hash",
    "plane_sweep_statistics",
    "save_checkpoint",
    "training_loss",
]


def _scaled_camera(camera, size):
    """Intrinsics for an exact area downsample; integer pixel centers are kept consistent."""
    height, width = camera.image_size
    sy, sx = size[0] / height, size[1] / width
    intrinsics = camera.intrinsics.clone()
    intrinsics[0, 0] *= sx
    intrinsics[1, 1] *= sy
    intrinsics[0, 2] = sx * (intrinsics[0, 2] + 0.5) - 0.5
    intrinsics[1, 2] = sy * (intrinsics[1, 2] + 0.5) - 0.5
    return Cameras(intrinsics, camera.c2w, tuple(size))


def _sample_rgb(rgb, pixels):
    """Bilinear RGB at pixel coordinates [..., 2] (align_corners pixel-center convention)."""
    height, width = rgb.shape[-2:]
    x = pixels[..., 0] / max(width - 1, 1) * 2.0 - 1.0
    y = pixels[..., 1] / max(height - 1, 1) * 2.0 - 1.0
    grid = torch.nan_to_num(torch.stack((x, y), -1), nan=2.0, posinf=2.0, neginf=-2.0)
    flat = grid.reshape(1, -1, 1, 2)
    sampled = functional.grid_sample(
        rgb.unsqueeze(0), flat, mode="bilinear", padding_mode="zeros", align_corners=True
    )
    return sampled.reshape(3, *pixels.shape[:-1])


def _depth_distribution(observations, reference, near, far, log_tau, planes, size):
    """Per-pixel softmax over depth planes for one reference view; None if no source view."""
    ref = observations[reference]
    sources = [o for i, o in enumerate(observations) if i != reference]
    camera = _scaled_camera(ref.camera, size)
    device = ref.rgb.device
    with torch.no_grad():
        ref_rgb = functional.interpolate(
            ref.rgb.to(torch.float32).unsqueeze(0), size=size, mode="area"
        )[0]
        inverse = torch.linspace(1.0 / near, 1.0 / far, planes, device=device)
        depths = 1.0 / inverse
        v, u = torch.meshgrid(
            torch.arange(size[0], device=device, dtype=torch.float32),
            torch.arange(size[1], device=device, dtype=torch.float32),
            indexing="ij",
        )
        pixels = torch.stack((u, v, torch.ones_like(u)), -1)
        directions = pixels @ torch.linalg.inv(camera.intrinsics.to(torch.float32)).T
        local = directions.unsqueeze(0) * depths.view(-1, 1, 1, 1)
        c2w = camera.c2w.to(torch.float32)
        world = local @ c2w[:3, :3].T + c2w[:3, 3]
        total = ref_rgb.unsqueeze(1).expand(3, planes, *size).clone()
        squares = total.square()
        count = torch.ones(planes, *size, device=device)
        for source in sources:
            # Every view is compared at the same area-downsampled resolution (same blur).
            cam = _scaled_camera(source.camera, size).to(dtype=torch.float32)
            small = functional.interpolate(
                source.rgb.to(torch.float32).unsqueeze(0), size=size, mode="area"
            )[0]
            projected, _, inside = project_world(world.reshape(-1, 3), cam)
            colors = _sample_rgb(small, projected)
            valid = inside.reshape(planes, *size).to(torch.float32)
            colors = colors.reshape(3, planes, *size) * valid
            total = total + colors
            squares = squares + colors.square()
            count = count + valid
        mean = total / count
        variance = (squares / count - mean.square()).clamp_min(0).mean(0)
        valid_plane = count >= 2
        valid_pixel = valid_plane.any(0)
    logits = (-variance / torch.exp(log_tau)).masked_fill(~valid_plane, float("-inf"))
    logits = torch.where(valid_pixel.unsqueeze(0), logits, torch.zeros_like(logits))
    probability = torch.softmax(logits, dim=0) * valid_pixel.unsqueeze(0)
    cumulative = torch.cumsum(probability, dim=0)
    return camera, probability, cumulative, valid_pixel


def plane_sweep_statistics(
    observations, points, near, far, log_tau, *, planes=PLANES, size=REF_SIZE
):
    """[N, 3] = (surface likelihood, free-space likelihood, valid-reference fraction)."""
    observations = tuple(observations)
    result = points.new_zeros(points.shape[0], 3)
    if len(observations) < 2:
        return result
    points32 = points.to(torch.float32)
    surface = torch.zeros(points.shape[0], device=points.device)
    free = torch.zeros_like(surface)
    references = torch.zeros_like(surface)
    for reference in range(len(observations)):
        camera, probability, cumulative, valid_pixel = _depth_distribution(
            observations, reference, near, far, log_tau, planes, size
        )
        with torch.no_grad():
            pixels, depth, inside = project_world(points32, camera.to(dtype=torch.float32))
            in_range = inside & (depth >= near) & (depth <= far)
            index = (1.0 / depth.clamp_min(1e-6) - 1.0 / near) / (1.0 / far - 1.0 / near)
            coords = torch.stack(
                (
                    pixels[:, 0] / max(size[1] - 1, 1) * 2 - 1,
                    pixels[:, 1] / max(size[0] - 1, 1) * 2 - 1,
                    index * 2 - 1,
                ),
                -1,
            )
            coords = torch.nan_to_num(coords, nan=2.0, posinf=2.0, neginf=-2.0)
            grid = coords.view(1, -1, 1, 1, 3)
            pixel_ok = functional.grid_sample(
                valid_pixel.to(torch.float32).view(1, 1, 1, *size).expand(1, 1, 2, *size),
                grid,
                mode="bilinear",
                align_corners=True,
            ).view(-1)
            use = in_range & (pixel_ok >= 0.999)
        volume = torch.stack((probability, cumulative)).unsqueeze(0)
        sampled = functional.grid_sample(
            volume, grid, mode="bilinear", padding_mode="border", align_corners=True
        ).view(2, -1)
        weight = use.to(sampled.dtype)
        surface = surface + weight * planes * sampled[0]
        free = free + weight * (1.0 - sampled[1])
        references = references + weight
    denominator = references.clamp_min(1.0)
    result = torch.stack(
        (surface / denominator, free / denominator, references / len(observations)), -1
    )
    return result.to(points.dtype)


class PlaneSweepCarrier(DynamicSceneCarrier):
    """Dense carrier plus a zero-initialized plane-sweep bypass into fuse_up (49 parameters)."""

    def __init__(self, config: CarrierConfig, near: float, far: float) -> None:
        super().__init__(config)
        # Created without consuming the RNG, so shared parameters match the dense baseline.
        self.sweep_weight = nn.Parameter(torch.zeros(config.expansion_dim, 3))
        self.sweep_log_tau = nn.Parameter(torch.tensor(LOG_TAU0))
        if not 0 < near < far:
            raise ValueError("Frozen TRAIN prior requires 0 < near < far")
        self.sweep_near, self.sweep_far = float(near), float(far)

    def _compute(self, cache, fast):
        self._validate_fast(cache, fast)
        lifted = lift_cached_observations(
            cache,
            self._candidate_points,
            self._candidate_normalized_xyz,
            feature_dim=self.config.feature_dim,
        )
        statistics = lifted.observed_feature_statistics.to(dtype=self.fuse_up.weight.dtype)
        support_weights = lifted.support_weights.to(dtype=statistics.dtype)
        sweep = plane_sweep_statistics(
            [entry.observation for entry in cache.entries],
            self._candidate_points,
            self.sweep_near,
            self.sweep_far,
            self.sweep_log_tau,
        ).to(dtype=statistics.dtype)
        fuse_input = self.fuse_up(statistics) + functional.linear(sweep, self.sweep_weight)
        fuse_activation = functional.gelu(fuse_input)
        fused_hidden = functional.linear(
            fuse_activation,
            self.fuse_down.weight + fast.delta_fuse,
            self.fuse_down.bias,
        )
        fused_hidden = fused_hidden * support_weights.unsqueeze(-1)
        refined = self.refinement(self._scatter_candidates(fused_hidden))
        complete_input = self._gather_candidates(refined)
        complete_activation = functional.gelu(self.complete_up(complete_input))
        trace = WriteTrace(
            episode_id=cache.episode_id,
            cache_revision=cache.revision,
            fast_step=fast.step,
            fuse_activation=fuse_activation,
            complete_activation=complete_activation,
            observed_feature_statistics=statistics,
            support_weights=support_weights,
            candidate_ids=self._candidate_ids.clone(),
            branch_id=fast.branch_id,
            source_state_id=fast.state_id,
        )
        return _CarrierComputation(trace, refined)


def carrier_config():
    return CarrierConfig(grid_size=GRID, token_count=TOKEN_COUNT)


def make_carrier(variant, seed, device, near, far):
    """Identical per-seed draw of every shared parameter; the sweep bypass starts at zero."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if VARIANT_SPECS[variant]["sweep"]:
        carrier = PlaneSweepCarrier(carrier_config(), near, far)
    else:
        carrier = DynamicSceneCarrier(carrier_config())
    return carrier.to(device)


def parameter_hash(carrier):
    """Hash of the parameters shared by every variant (the sweep bypass is excluded)."""
    digest = hashlib.sha256()
    for name, value in carrier.named_parameters():
        if name.startswith("sweep_"):
            continue
        digest.update(name.encode())
        digest.update(value.detach().to("cpu", torch.float32).contiguous().numpy().tobytes())
    return digest.hexdigest()


def training_loss(state, local_cameras, rgb, depth, indices, variant):
    """Frozen V2 loss family: C0/C1 use the V2 surface recipe, C2 the V2 no-surface loss."""
    return geometry_carrier.training_loss(
        state, local_cameras, rgb, depth, indices, VARIANT_SPECS[variant]["loss"]
    )


def save_checkpoint(path, carrier, *, variant, seed, step, lock_sha256):
    sweep = None
    if isinstance(carrier, PlaneSweepCarrier):
        sweep = {"near": carrier.sweep_near, "far": carrier.sweep_far}
    payload = {
        "schema": SCHEMA,
        "config": asdict(carrier.config),
        "carrier": {k: v.detach().cpu() for k, v in carrier.state_dict().items()},
        "sweep": sweep,
        "variant": variant,
        "seed": seed,
        "step": step,
        "lock_sha256": lock_sha256,
        "test_time_input": "RGB+CAMERA",
        "writer_trained": False,
    }
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    torch.save(payload, path)
    restored, _ = load_checkpoint(path, "cpu")
    if hash_value(restored.state_dict()) != hash_value(payload["carrier"]):
        raise RuntimeError("Checkpoint restore mismatch")
    return sha(path)


def load_checkpoint(path, device):
    payload = torch.load(path, map_location=device, weights_only=True)
    if payload.get("schema") != SCHEMA or payload.get("writer_trained"):
        raise PermissionError("V4 static plane-sweep carrier schema required")
    config = CarrierConfig(**payload["config"])
    if tuple(config.grid_size) != GRID or config.token_count != TOKEN_COUNT:
        raise PermissionError("Only the frozen dense 16-grid carrier")
    if payload["sweep"] is None:
        model = DynamicSceneCarrier(config)
    else:
        model = PlaneSweepCarrier(config, payload["sweep"]["near"], payload["sweep"]["far"])
    model = model.to(device)
    model.load_state_dict(payload["carrier"], strict=True)
    return model, payload
