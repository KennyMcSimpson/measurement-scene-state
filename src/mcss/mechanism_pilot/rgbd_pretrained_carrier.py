"""V31: the V17 L carrier with frozen DINOv2 ViT-S/14 (reg4) patch features through a zero bypass.

PLAN_BEFORE_V31.md fixed V31 before any V31 training. The V17 L carrier (width 32, 32^3, eight
learned channels per view from two 3x3 convolutions) gains PROJECTED = 24 channels per view: the
frozen DINOv2 patch tokens of the same RGB frame (ImageNet normalization, bilinear resize of the
128x160 frame to 252x322, i.e. 18x23 patches of 384 dimensions) through a learned 384->24
projection, bilinearly resized back to the frame. The lifted mean and variance of those 24
channels (48 statistics per voxel candidate) enter the fuse pre-activation through a bias-free
48->64 map that starts at zero, so a new carrier computes exactly what the V17 L draw of the same
seed computes. The projection comes from its own seeded generator; initialization, scene order,
rays and extra views keep V17 L's random streams. The backbone is never trained, never part of
the state_dict, and identified by the sha256 of its weights and code; its tokens depend only on
the frame and are cached per process and device.
"""

from __future__ import annotations

import hashlib
import math
import warnings
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

EXPERIMENT = "EXP-3D-RGBD-PRETRAINED-FEATURES-V31"
SCHEMA = "mcss.rgbd_pretrained_carrier.v31"
MODELS = Path("/home/xmum26_zonghan/zonghan_o_mirror/_models/dinov2")
BACKBONE = {
    "name": "dinov2_vits14_reg",
    "code_commit": "7764ea0f912e53c92e82eb78a2a1631e92725fc8",
    "code_zip_sha256": "04276715cddb29d45d05bff3a6fc132224dc27749b279ac98ad2ce4620e20d48",
    "code_tree_sha256": "117c2d9278ba282500e5d3924dae634672bd19cf892f3f8f4800fbd80b7b6af4",
    "weights": "dinov2_vits14_reg4_pretrain.pth",
    "weights_sha256": "f433177089a681826f849f194ece3bb48f4d63fb38d32fc837e3dc7a4e5641fb",
    "input_size": [252, 322],
    "patch_grid": [18, 23],
    "embed_dim": 384,
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225],
    "output": "x_norm_patchtokens",
}
PROJECTED = 24
PROJECTION_SEED_OFFSET = 31
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
_BACKBONES = {}
_TOKENS = {}

__all__ = [
    "BACKBONE",
    "EXPERIMENT",
    "EXTRA_SEED_OFFSET",
    "PROJECTED",
    "VARIANTS",
    "VARIANT_SPECS",
    "PretrainedFeatureCarrier",
    "backbone_files_present",
    "build_state",
    "carrier_config",
    "code_tree_sha256",
    "extra_count",
    "load_backbone",
    "load_checkpoint",
    "make_carrier",
    "parameter_hash",
    "patch_tokens",
    "save_checkpoint",
    "scene_data",
    "train_records",
    "training_loss",
    "with_extra_views",
]


def repository():
    return MODELS / f"dinov2-{BACKBONE['code_commit']}"


def backbone_files_present():
    return (MODELS / BACKBONE["weights"]).is_file() and (repository() / "hubconf.py").is_file()


def code_tree_sha256(repo):
    """sha256 over (relative path, file sha256) of hubconf.py and every dinov2/**/*.py."""
    repo = Path(repo)
    digest = hashlib.sha256()
    for path in sorted([repo / "hubconf.py", *repo.glob("dinov2/**/*.py")]):
        name = path.relative_to(repo).as_posix().encode()
        digest.update(name + b"\0" + sha(path).encode() + b"\n")
    return digest.hexdigest()


def load_backbone(device):
    """The frozen DINOv2 backbone on one device, loaded offline after every hash check."""
    key = str(torch.device(device))
    if key not in _BACKBONES:
        weights, repo = MODELS / BACKBONE["weights"], repository()
        if sha(weights) != BACKBONE["weights_sha256"]:
            raise PermissionError("DINOv2 weights differ from the registered file")
        if sha(MODELS / f"dinov2_{BACKBONE['code_commit']}.zip") != BACKBONE["code_zip_sha256"]:
            raise PermissionError("DINOv2 code archive differs from the registered snapshot")
        if code_tree_sha256(repo) != BACKBONE["code_tree_sha256"]:
            raise PermissionError("DINOv2 code differs from the registered snapshot")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)  # xFormers is optional
            model = torch.hub.load(str(repo), BACKBONE["name"], source="local", pretrained=False)
        model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
        model.requires_grad_(False).eval()
        _BACKBONES[key] = model.to(device)
    return _BACKBONES[key]


def patch_tokens(rgb):
    """Frozen [384, 18, 23] patch tokens of one RGB frame in [0, 1], cached per device and frame."""
    if rgb.ndim != 3 or rgb.shape[0] != 3:
        raise ValueError("One RGB frame [3,H,W] required")
    frame = rgb.detach().to(device="cpu", dtype=torch.float32).contiguous()
    key = (str(rgb.device), tuple(rgb.shape), hashlib.sha256(frame.numpy().tobytes()).hexdigest())
    tokens = _TOKENS.get(key)
    if tokens is None:
        model = load_backbone(rgb.device)
        mean = torch.tensor(BACKBONE["mean"], device=rgb.device).view(3, 1, 1)
        std = torch.tensor(BACKBONE["std"], device=rgb.device).view(3, 1, 1)
        image = ((rgb.detach().to(torch.float32) - mean) / std).unsqueeze(0)
        image = functional.interpolate(
            image, size=tuple(BACKBONE["input_size"]), mode="bilinear", align_corners=False
        )
        with torch.no_grad():
            patches = model.forward_features(image)[BACKBONE["output"]][0]
        tokens = patches.transpose(0, 1).reshape(BACKBONE["embed_dim"], *BACKBONE["patch_grid"])
        tokens = tokens.contiguous()
        _TOKENS[key] = tokens
    return tokens


class PretrainedFeatureCarrier(v8.ResolutionRGBDCarrier):
    """V17 L carrier plus frozen DINOv2 features through a zero-initialized fuse bypass."""

    def __init__(self, config: CarrierConfig) -> None:
        super().__init__(config)
        # Created without consuming the RNG, so the shared parameters are the V17 L draw.
        self.feature_projection = nn.Parameter(torch.zeros(PROJECTED, BACKBONE["embed_dim"]))
        self.feature_projection_bias = nn.Parameter(torch.zeros(PROJECTED))
        self.pretrained_weight = nn.Parameter(torch.zeros(config.expansion_dim, 2 * PROJECTED))

    def encode(self, observation):
        """The eight learned channels followed by the 24 projected backbone channels."""
        learned = super().encode(observation)
        tokens = patch_tokens(observation.rgb).to(dtype=learned.dtype)
        projected = torch.einsum("kc,chw->khw", self.feature_projection, tokens)
        projected = projected + self.feature_projection_bias.view(-1, 1, 1)
        projected = functional.interpolate(
            projected.unsqueeze(0),
            size=tuple(learned.shape[-2:]),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        return torch.cat((learned, projected), 0)

    def _compute(self, cache, fast):
        self._validate_fast(cache, fast)
        if self._pending_depth is None:
            raise PermissionError("RGB-D carrier states are built only through build_state")
        frame_ids, depths = self._pending_depth
        if tuple(entry.observation.frame_id for entry in cache.entries) != frame_ids:
            raise ValueError("Measured depth maps must align with the cached context views")
        width = self.config.feature_dim
        total = width + PROJECTED
        lifted = lift_cached_observations(
            cache,
            self._candidate_points,
            self._candidate_normalized_xyz,
            feature_dim=total,
        )
        full = lifted.observed_feature_statistics.to(dtype=self.fuse_up.weight.dtype)
        mean, variance, rest = full[:, :total], full[:, total : 2 * total], full[:, 2 * total :]
        # The V17 L statistics of the eight learned channels, in V17 L's order.
        statistics = torch.cat((mean[:, :width], variance[:, :width], rest), -1)
        pretrained = torch.cat((mean[:, width:], variance[:, width:]), -1)
        support_weights = lifted.support_weights.to(dtype=statistics.dtype)
        fuse_activation = functional.gelu(
            self.fuse_up(statistics) + functional.linear(pretrained, self.pretrained_weight)
        )
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


def make_carrier(variant, seed, device, near=None, far=None):
    """The V17 L draw of every shared parameter; the projection from its own seeded generator."""
    del near, far
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    carrier = PretrainedFeatureCarrier(carrier_config(variant))
    generator = torch.Generator().manual_seed(seed + PROJECTION_SEED_OFFSET)
    bound = 1.0 / math.sqrt(BACKBONE["embed_dim"])
    shape = carrier.feature_projection.shape
    with torch.no_grad():
        carrier.feature_projection.copy_((torch.rand(shape, generator=generator) * 2 - 1) * bound)
    return carrier.to(device)


def save_checkpoint(path, carrier, *, variant, seed, step, lock_sha256):
    payload = {
        "schema": SCHEMA,
        "config": asdict(carrier.config),
        "carrier": {k: v.detach().cpu() for k, v in carrier.state_dict().items()},
        "depth_bypass": True,
        "backbone": BACKBONE,
        "projected": PROJECTED,
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
        raise PermissionError("V31 pretrained-feature carrier schema required")
    if payload.get("backbone") != BACKBONE or payload.get("projected") != PROJECTED:
        raise PermissionError("The checkpoint was trained with another backbone or projection")
    config = CarrierConfig(**payload["config"])
    v17._check_config(config)
    model = PretrainedFeatureCarrier(config).to(device)
    model.load_state_dict(payload["carrier"], strict=True)
    return model, payload
