"""One-scene differentiable carrier with explicit fast writable down projections."""

from __future__ import annotations

from dataclasses import dataclass
from math import prod

import torch
import torch.nn.functional as functional
from torch import Tensor, nn

from mcss.dynamic.cache import ObservationCache
from mcss.dynamic.config import CarrierConfig
from mcss.dynamic.lifting import lift_cached_observations
from mcss.dynamic.types import FastWeights, OnlineObservation, WriteTrace
from mcss.types import SceneState


@dataclass(frozen=True)
class _CarrierComputation:
    trace: WriteTrace
    refined_volume: Tensor


class DynamicSceneCarrier(nn.Module):
    """Materialize one query-independent scene from an observed episode prefix."""

    def __init__(self, config: CarrierConfig) -> None:
        super().__init__()
        self.config = config
        feature_dim = config.feature_dim
        hidden_dim = config.hidden_dim
        expansion_dim = config.expansion_dim
        self.image_encoder = nn.Sequential(
            nn.Conv2d(3, feature_dim, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(feature_dim, feature_dim, kernel_size=3, padding=1),
            nn.SiLU(),
        )
        self.fuse_up = nn.Linear(config.statistics_dim, expansion_dim)
        self.fuse_down = nn.Linear(expansion_dim, hidden_dim)
        self.refinement = nn.Sequential(
            nn.Conv3d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv3d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.SiLU(),
        )
        self.complete_up = nn.Linear(hidden_dim, expansion_dim)
        self.complete_down = nn.Linear(expansion_dim, hidden_dim)
        self.density_head = nn.Conv3d(hidden_dim, 1, kernel_size=1)
        self.color_head = nn.Conv3d(hidden_dim, 3, kernel_size=1)
        self.log_variance_head = nn.Conv3d(hidden_dim, 1, kernel_size=1)
        self._register_candidate_geometry()
        nn.init.constant_(self.density_head.bias, -2.0)
        nn.init.constant_(self.log_variance_head.bias, -3.0)

    def encode(self, observation: OnlineObservation) -> Tensor:
        """Encode exactly one arrived RGB observation into its cacheable feature map."""

        parameter = self.image_encoder[0].weight
        if observation.rgb.device != parameter.device:
            raise ValueError("observation and carrier must share a device")
        encoded = self.image_encoder(observation.rgb.to(dtype=parameter.dtype).unsqueeze(0))
        return encoded.squeeze(0)

    def initial_fast(self, episode_id: str) -> FastWeights:
        """Create fresh, unregistered zero offsets for one episode."""

        if not episode_id:
            raise ValueError("episode_id must be nonempty")
        shape = self.fuse_down.weight.shape
        zero = self.fuse_down.weight.new_zeros(shape)
        return FastWeights(episode_id=episode_id, delta_fuse=zero, delta_complete=zero.clone())

    def trace(self, cache: ObservationCache, fast: FastWeights) -> WriteTrace:
        """Expose candidate activations from one unmodified fast-state version."""

        return self._compute(cache, fast).trace

    def materialize(self, cache: ObservationCache, fast: FastWeights) -> SceneState:
        """Build a typed state without accessing labels, depth, future frames, or queries."""

        computed = self._compute(cache, fast)
        trace = computed.trace
        complete_hidden = functional.linear(
            trace.complete_activation,
            self.complete_down.weight + fast.delta_complete,
            self.complete_down.bias,
        )
        complete_hidden = complete_hidden * trace.support_weights.unsqueeze(-1)
        hidden = computed.refined_volume + self._scatter_candidates(complete_hidden)
        density_logits = self.density_head(hidden)
        color = torch.sigmoid(self.color_head(hidden))
        log_variance = self.log_variance_head(hidden).clamp(-8.0, 4.0)
        bounds = self._bounds.expand(1, -1, -1).clone()
        return SceneState(density_logits, color, log_variance, bounds, features=hidden)

    def _compute(self, cache: ObservationCache, fast: FastWeights) -> _CarrierComputation:
        self._validate_fast(cache, fast)
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

    def _validate_fast(self, cache: ObservationCache, fast: FastWeights) -> None:
        if cache.episode_id != fast.episode_id:
            raise ValueError("fast weights episode_id does not match the observation cache")
        expected = self.fuse_down.weight.shape
        if fast.delta_fuse.shape != expected or fast.delta_complete.shape != expected:
            raise ValueError("fast weight shapes do not match carrier down projections")
        if fast.delta_fuse.device != self.fuse_down.weight.device:
            raise ValueError("fast weights and carrier must share a device")
        if (
            fast.delta_fuse.dtype != self.fuse_down.weight.dtype
            or fast.delta_complete.dtype != self.fuse_down.weight.dtype
        ):
            raise ValueError("fast weights and carrier must share a dtype")

    def _register_candidate_geometry(self) -> None:
        depth, height, width = self.config.grid_size
        voxel_count = prod(self.config.grid_size)
        if self.config.token_count == 1:
            candidate_ids = torch.tensor([(voxel_count - 1) // 2], dtype=torch.long)
        else:
            numerator = torch.arange(self.config.token_count, dtype=torch.long) * (voxel_count - 1)
            candidate_ids = torch.div(
                numerator,
                self.config.token_count - 1,
                rounding_mode="floor",
            )
        d = torch.div(candidate_ids, height * width, rounding_mode="floor")
        h = torch.div(candidate_ids, width, rounding_mode="floor").remainder(height)
        w = candidate_ids.remainder(width)
        fractions = torch.stack(
            (
                (w.to(torch.float32) + 0.5) / width,
                (h.to(torch.float32) + 0.5) / height,
                (d.to(torch.float32) + 0.5) / depth,
            ),
            dim=-1,
        )
        bounds = torch.tensor(self.config.local_bounds_m, dtype=torch.float32).unsqueeze(0)
        points = bounds[:, 0] + fractions * (bounds[:, 1] - bounds[:, 0])
        self.register_buffer("_bounds", bounds)
        self.register_buffer("_candidate_ids", candidate_ids)
        self.register_buffer("_candidate_points", points)
        self.register_buffer("_candidate_normalized_xyz", fractions * 2.0 - 1.0)

    def _scatter_candidates(self, values: Tensor) -> Tensor:
        if values.ndim != 2 or values.shape != (
            self.config.token_count,
            self.config.hidden_dim,
        ):
            raise ValueError("candidate values must have shape [token_count, hidden_dim]")
        volume = values.new_zeros(self.config.hidden_dim, prod(self.config.grid_size))
        volume[:, self._candidate_ids] = values.transpose(0, 1)
        return volume.reshape(1, self.config.hidden_dim, *self.config.grid_size)

    def _gather_candidates(self, volume: Tensor) -> Tensor:
        flattened = volume.reshape(1, self.config.hidden_dim, -1)
        return flattened[0, :, self._candidate_ids].transpose(0, 1)
