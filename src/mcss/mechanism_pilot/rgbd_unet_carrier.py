"""V32: the V17 L carrier with a whole-volume 3D U-Net context branch (zero-initialized output).

PLAN_BEFORE_V32.md fixed V32 before any V32 training. The V17 L carrier completes its volume only
through two 3x3x3 convolutions (a 5^3-voxel receptive field); everything after them is per voxel.
U adds a residual 3D U-Net on the same scattered fused volume (32^3, 32 channels): per level one
3x3x3 convolution and a stride-2 3x3x3 convolution down to 4^3, where a 3x3x3 convolution plus a
global term (the whole-volume mean through a 1x1x1 convolution) reaches every voxel; trilinear
upsampling with skip concatenations and one 3x3x3 convolution per level back to 32^3; a 1x1x1
output convolution that starts at zero, so a new carrier computes exactly what the V17 L draw of
the same seed computes. The branch is drawn from its own seeded random stream; the global stream
after construction is V17 L's.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn.functional as functional
from torch import nn

from mcss.dynamic.carrier import _CarrierComputation
from mcss.dynamic.config import CarrierConfig
from mcss.dynamic.lifting import lift_cached_observations
from mcss.dynamic.types import WriteTrace, hash_value
from mcss.mechanism_pilot import rgbd_evidence_carrier as v5
from mcss.mechanism_pilot import rgbd_long_train_carrier as v17
from mcss.mechanism_pilot import rgbd_resolution_carrier as v8
from mcss.mechanism_pilot.small_training import sha

EXPERIMENT = "EXP-3D-RGBD-UNET-CONTEXT-V32"
SCHEMA = "mcss.rgbd_unet_carrier.v32"
UNET = {
    "channels": 32,
    "levels": [32, 16, 8, 4],
    "kernel": 3,
    "downsampling": "stride-2 3x3x3 convolution",
    "upsampling": "trilinear, align_corners=False, then skip concatenation and a 3x3x3 convolution",
    "global_term": "whole-volume mean at 4^3 through a 1x1x1 convolution",
    "activation": "SiLU",
    "output": "1x1x1 convolution, zero-initialized weight and bias",
    "seed_offset": 37,
}
UNET_SEED_OFFSET = UNET["seed_offset"]
EXTRA_SEED_OFFSET = v17.EXTRA_SEED_OFFSET
VARIANT_SPECS = v17.VARIANT_SPECS
VARIANTS = v17.VARIANTS
build_state = v17.build_state
carrier_config = v17.carrier_config
extra_count = v17.extra_count
parameter_hash = v17.parameter_hash
scene_data = v17.scene_data
train_records = v17.train_records
training_loss = v17.training_loss
with_extra_views = v17.with_extra_views

__all__ = [
    "EXPERIMENT",
    "EXTRA_SEED_OFFSET",
    "UNET",
    "VARIANTS",
    "VARIANT_SPECS",
    "ContextUNet",
    "UNetContextCarrier",
    "build_state",
    "carrier_config",
    "extra_count",
    "load_checkpoint",
    "make_carrier",
    "parameter_hash",
    "save_checkpoint",
    "scene_data",
    "train_records",
    "training_loss",
    "with_extra_views",
]


def _conv(inputs, outputs, stride=1):
    return nn.Conv3d(inputs, outputs, kernel_size=3, stride=stride, padding=1)


class ContextUNet(nn.Module):
    """32^3 -> 4^3 -> 32^3 with skips and a global term; the output convolution starts at zero."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        c = channels
        self.enc1, self.down1 = _conv(c, c), _conv(c, c, 2)
        self.enc2, self.down2 = _conv(c, c), _conv(c, c, 2)
        self.enc3, self.down3 = _conv(c, c), _conv(c, c, 2)
        self.bottom = _conv(c, c)
        self.glob = nn.Conv3d(c, c, kernel_size=1)
        self.up3, self.up2, self.up1 = _conv(2 * c, c), _conv(2 * c, c), _conv(2 * c, c)
        self.out = nn.Conv3d(c, c, kernel_size=1)
        self.zero_output()

    def zero_output(self):
        with torch.no_grad():
            self.out.weight.zero_()
            self.out.bias.zero_()

    def forward(self, volume):
        act = functional.silu

        def up(value, like):
            return functional.interpolate(
                value, size=like.shape[-3:], mode="trilinear", align_corners=False
            )

        e1 = act(self.enc1(volume))
        e2 = act(self.enc2(act(self.down1(e1))))
        e3 = act(self.enc3(act(self.down2(e2))))
        d3 = act(self.down3(e3))
        bottom = act(self.bottom(d3)) + act(self.glob(d3.mean(dim=(2, 3, 4), keepdim=True)))
        u3 = act(self.up3(torch.cat((up(bottom, e3), e3), 1)))
        u2 = act(self.up2(torch.cat((up(u3, e2), e2), 1)))
        u1 = act(self.up1(torch.cat((up(u2, e1), e1), 1)))
        return self.out(u1)


class UNetContextCarrier(v8.ResolutionRGBDCarrier):
    """V17 L carrier plus a residual whole-volume 3D U-Net next to its two-layer refinement."""

    def __init__(self, config: CarrierConfig) -> None:
        super().__init__(config)
        # Built inside a forked random stream: the shared parameters stay the V17 L draw.
        with torch.random.fork_rng(devices=[]):
            self.context_unet = ContextUNet(UNET["channels"])

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
        grid = torch.tensor(
            tuple(self.config.grid_size), device=self._bounds.device, dtype=self._bounds.dtype
        )
        edge = ((self._bounds[:, 1] - self._bounds[:, 0]) / grid).mean()
        measured = v5.depth_statistics(
            [entry.observation for entry in cache.entries],
            depths,
            self._candidate_points,
            self.depth_log_sigma,
            edge.to(device=self._candidate_points.device, dtype=torch.float32),
        ).to(dtype=statistics.dtype)
        fused_hidden = fused_hidden + functional.linear(measured, self.depth_weight)
        volume = self._scatter_candidates(fused_hidden)
        refined = self.refinement(volume) + self.context_unet(volume)
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


def make_carrier(variant, seed, device, near=None, far=None):
    """The V17 L draw of every shared parameter; the branch from its own seeded stream."""
    del near, far
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    carrier = UNetContextCarrier(carrier_config(variant))
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed + UNET_SEED_OFFSET)
        for layer in carrier.context_unet.modules():
            if isinstance(layer, nn.Conv3d):
                layer.reset_parameters()
    carrier.context_unet.zero_output()
    return carrier.to(device)


def save_checkpoint(path, carrier, *, variant, seed, step, lock_sha256):
    payload = {
        "schema": SCHEMA,
        "config": asdict(carrier.config),
        "carrier": {k: v.detach().cpu() for k, v in carrier.state_dict().items()},
        "depth_bypass": True,
        "unet": UNET,
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
        raise PermissionError("V32 U-Net context carrier schema required")
    if payload.get("unet") != UNET:
        raise PermissionError("The checkpoint was trained with another context branch")
    config = CarrierConfig(**payload["config"])
    v17._check_config(config)
    model = UNetContextCarrier(config).to(device)
    model.load_state_dict(payload["carrier"], strict=True)
    return model, payload
