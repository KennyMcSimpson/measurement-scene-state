"""V5: measured context depth as the only new evidence for the dense 16^3 learned carrier.

RGB-D track: every context view also arrives with its measured metric ray distance, the way a
depth sensor would deliver it; query depth is never read before all DEV states are sealed.
For every voxel and every context view in which the voxel centre projects in front of the
camera and inside the image, with d its centre distance from the camera and D the measured
ray distance at the nearest pixel, the view gives a surface likelihood exp(-(d - D)^2 / 2s^2)
and a free-space likelihood sigmoid((D - d) / s), s = exp(log_sigma) * mean voxel edge. The
two are averaged over valid views, plus the valid-view fraction. log_sigma is the only learned
depth parameter. A single measured view is already informative, so the three statistics enter
after the >=2-view support gate through a zero-initialized bias-free 3->8 projection (25
parameters in total): a depth carrier starts exactly equal to the matched dense carrier.
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
from mcss.dynamic.types import OnlineObservation, WriteTrace, hash_value
from mcss.geometry import project_world
from mcss.mechanism_pilot import geometry_carrier
from mcss.mechanism_pilot.dense_evidence_carrier import build_state as dense_build_state
from mcss.mechanism_pilot.direct_capacity_contracts import ContextOnlyLoader
from mcss.mechanism_pilot.geometry_carrier_experiment import SceneData
from mcss.mechanism_pilot.small_training import sha
from mcss.types import Cameras

EXPERIMENT = "EXP-3D-RGBD-EVIDENCE-CARRIER-V5"
GRID = (16, 16, 16)
TOKEN_COUNT = 4096
LOG_SIGMA0 = math.log(0.5)
VARIANT_SPECS = {
    "C0": {"label": "DENSE_SURFACE", "depth": False, "loss": "C1"},
    "C1": {"label": "DEPTH_SURFACE", "depth": True, "loss": "C1"},
    "C2": {"label": "DEPTH_NO_SURFACE", "depth": True, "loss": "C0"},
}
VARIANTS = tuple(VARIANT_SPECS)
PRIMARY_PAIR = ("C0", "C1")
SCHEMA = "mcss.rgbd_evidence_carrier.v5"

__all__ = [
    "EXPERIMENT",
    "PRIMARY_PAIR",
    "VARIANTS",
    "VARIANT_SPECS",
    "RGBDContext",
    "RGBDEvidenceCarrier",
    "RGBDSceneData",
    "build_state",
    "depth_statistics",
    "load_checkpoint",
    "make_carrier",
    "parameter_hash",
    "save_checkpoint",
    "training_loss",
]


class RGBDContext(tuple):
    """Context observations, iterated exactly like the RGB tuple, plus measured ray distance."""

    def __new__(cls, observations, depths):
        observations = tuple(observations)
        if not observations or any(type(o) is not OnlineObservation for o in observations):
            raise TypeError("RGB-camera OnlineObservation values required")
        depths = torch.as_tensor(depths)
        if depths.ndim != 3 or depths.shape[0] != len(observations):
            raise ValueError("One [H, W] measured depth map per context view required")
        for observation, depth in zip(observations, depths, strict=True):
            if tuple(depth.shape) != tuple(observation.camera.image_size):
                raise ValueError("Depth map must match its camera image size")
        context = super().__new__(cls, observations)
        context.depths = depths
        context.frame_ids = tuple(o.frame_id for o in observations)
        return context


class RGBDSceneData(SceneData):
    """SceneData whose state-construction loader is the RGBD context-only loader.

    That loader holds only context frames, so query depth cannot enter construction; query
    targets remain behind the unchanged TRAIN-only target() boundary.
    """

    def __init__(self, record, manifest, root, device, access):
        super().__init__(record, manifest, root, device, access)
        self.loader = ContextOnlyLoader(
            record,
            manifest["image_size"],
            root,
            device,
            track="RGBD",
            capacity_scene_ids=self.allowed,
            holdout_scene_ids=self.forbidden,
            access_log=access,
        )

    def context(self, role):
        if role not in ("A", "B", "anchor"):
            raise PermissionError("Only context RGB/depth/camera")
        if role not in self._contexts:
            batch = self.loader.context(role)
            observations = [
                OnlineObservation(
                    batch.scene_id,
                    fid,
                    batch.rgb[0, j],
                    Cameras(
                        batch.cameras.intrinsics[0, j],
                        batch.cameras.c2w[0, j],
                        batch.cameras.image_size,
                    ),
                )
                for j, fid in enumerate(batch.frame_ids)
            ]
            self._contexts[role] = RGBDContext(observations, batch.depth[0, :, 0])
        return self._contexts[role]


def depth_statistics(observations, depths, points, log_sigma, voxel_edge):
    """[N, 3] = (surface likelihood, free-space likelihood, valid-view fraction)."""
    observations = tuple(observations)
    result = points.new_zeros(points.shape[0], 3)
    if not observations:
        return result
    points32 = points.to(torch.float32)
    sigma = torch.exp(log_sigma) * voxel_edge
    surface = torch.zeros(points.shape[0], device=points.device)
    free = torch.zeros_like(surface)
    count = torch.zeros_like(surface)
    for observation, depth in zip(observations, depths, strict=True):
        camera = observation.camera.to(dtype=torch.float32)
        height, width = camera.image_size
        with torch.no_grad():
            pixels, _, inside = project_world(points32, camera)
            finite = torch.isfinite(pixels).all(-1)
            safe = torch.where(finite.unsqueeze(-1), pixels, torch.zeros_like(pixels))
            u = safe[:, 0].round().clamp(0, width - 1).long()
            v = safe[:, 1].round().clamp(0, height - 1).long()
            measured = depth.to(device=points.device, dtype=torch.float32)[v, u]
            distance = (points32 - camera.c2w[:3, 3]).norm(dim=-1)
            valid = inside & finite & torch.isfinite(measured) & (measured > 0)
            measured = torch.where(valid, measured, distance)
        z = (distance - measured) / sigma
        weight = valid.to(z.dtype)
        surface = surface + weight * torch.exp(-0.5 * z.square())
        free = free + weight * torch.sigmoid(-z)
        count = count + weight
    denominator = count.clamp_min(1.0)
    result = torch.stack((surface / denominator, free / denominator, count / len(observations)), -1)
    return result.to(points.dtype)


class RGBDEvidenceCarrier(DynamicSceneCarrier):
    """Dense carrier plus a zero-initialized post-gate measured-depth bypass (25 parameters)."""

    def __init__(self, config: CarrierConfig) -> None:
        super().__init__(config)
        # Created without consuming the RNG, so shared parameters match the dense baseline.
        self.depth_weight = nn.Parameter(torch.zeros(config.hidden_dim, 3))
        self.depth_log_sigma = nn.Parameter(torch.tensor(LOG_SIGMA0))
        self._pending_depth = None

    def _compute(self, cache, fast):
        self._validate_fast(cache, fast)
        if self._pending_depth is None:
            raise PermissionError("RGB-D carrier states are built only through build_state")
        frame_ids, depths = self._pending_depth
        if tuple(entry.observation.frame_id for entry in cache.entries) != frame_ids:
            raise ValueError("Measured depth maps must align with the cached context views")
        lifted = lift_cached_observations(
            cache,
            self._candidate_points,
            self._candidate_normalized_xyz,
            feature_dim=self.config.feature_dim,
        )
        statistics = lifted.observed_feature_statistics.to(dtype=self.fuse_up.weight.dtype)
        support_weights = lifted.support_weights.to(dtype=statistics.dtype)
        fuse_activation = functional.gelu(self.fuse_up(statistics))
        fused_hidden = functional.linear(
            fuse_activation,
            self.fuse_down.weight + fast.delta_fuse,
            self.fuse_down.bias,
        )
        fused_hidden = fused_hidden * support_weights.unsqueeze(-1)
        grid = torch.tensor(GRID, device=self._bounds.device, dtype=self._bounds.dtype)
        edge = ((self._bounds[:, 1] - self._bounds[:, 0]) / grid).mean()
        measured = depth_statistics(
            [entry.observation for entry in cache.entries],
            depths,
            self._candidate_points,
            self.depth_log_sigma,
            edge.to(device=self._candidate_points.device, dtype=torch.float32),
        ).to(dtype=statistics.dtype)
        fused_hidden = fused_hidden + functional.linear(measured, self.depth_weight)
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


def build_state(carrier, context, bounds, episode_id):
    """Frozen V3 build_state; an RGB-D carrier additionally reads the context depth maps."""
    if not isinstance(context, RGBDContext):
        raise TypeError("V5 states are built from an RGBDContext")
    observations = tuple(context)
    if not isinstance(carrier, RGBDEvidenceCarrier):
        return dense_build_state(carrier, observations, bounds, episode_id)
    device = carrier._bounds.device
    carrier._pending_depth = (
        context.frame_ids,
        context.depths.to(device=device, dtype=torch.float32),
    )
    try:
        return dense_build_state(carrier, observations, bounds, episode_id)
    finally:
        carrier._pending_depth = None


def carrier_config():
    return CarrierConfig(grid_size=GRID, token_count=TOKEN_COUNT)


def make_carrier(variant, seed, device, near=None, far=None):
    """Identical per-seed draw of every shared parameter; the depth bypass starts at zero."""
    del near, far  # the depth bypass needs no depth range; bounds stay the frozen rule
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if VARIANT_SPECS[variant]["depth"]:
        carrier = RGBDEvidenceCarrier(carrier_config())
    else:
        carrier = DynamicSceneCarrier(carrier_config())
    return carrier.to(device)


def parameter_hash(carrier):
    """Hash of the parameters shared by every variant (the depth bypass is excluded)."""
    digest = hashlib.sha256()
    for name, value in carrier.named_parameters():
        if name.startswith("depth_"):
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
    payload = {
        "schema": SCHEMA,
        "config": asdict(carrier.config),
        "carrier": {k: v.detach().cpu() for k, v in carrier.state_dict().items()},
        "depth_bypass": isinstance(carrier, RGBDEvidenceCarrier),
        "variant": variant,
        "seed": seed,
        "step": step,
        "lock_sha256": lock_sha256,
        "test_time_input": "RGB+DEPTH+CAMERA",
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
        raise PermissionError("V5 static RGB-D evidence carrier schema required")
    config = CarrierConfig(**payload["config"])
    if tuple(config.grid_size) != GRID or config.token_count != TOKEN_COUNT:
        raise PermissionError("Only the frozen dense 16-grid carrier")
    has_depth = any(k.startswith("depth_") for k in payload["carrier"])
    if has_depth != bool(payload["depth_bypass"]):
        raise PermissionError("Checkpoint depth-bypass flag and tensors disagree")
    model = RGBDEvidenceCarrier(config) if has_depth else DynamicSceneCarrier(config)
    model = model.to(device)
    model.load_state_dict(payload["carrier"], strict=True)
    return model, payload
