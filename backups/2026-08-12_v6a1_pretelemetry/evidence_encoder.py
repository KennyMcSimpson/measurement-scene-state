"""Cross-view evidence construction and bounded residual completion in a typed 3D state."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn.functional as functional
from torch import Tensor, nn

from mcss.geometry import make_voxel_centers, project_world
from mcss.model.encoder2d import PyramidImageEncoder
from mcss.model.state_encoder import Residual3DBlock
from mcss.types import Cameras, DualEvidence, SceneState, StateAppearance, StateEvidence


class EvidenceResidualStateEncoder(nn.Module):
    """Build an inspectable evidence state and add only a bounded, gated completion residual."""

    def __init__(
        self,
        *,
        voxel_resolution: int | Sequence[int],
        image_feature_dim: int,
        state_feature_dim: int,
        refinement_blocks: int,
        evidence_temperature: float,
        observed_residual_floor: float,
        completion_residual_scale: float,
        appearance_resolution_scale: int = 1,
    ) -> None:
        super().__init__()
        if isinstance(voxel_resolution, int):
            resolution = (voxel_resolution,) * 3
        else:
            resolution = tuple(voxel_resolution)
        if len(resolution) != 3 or min(resolution) < 4:
            raise ValueError("voxel_resolution must contain D, H, W values of at least four")
        if image_feature_dim < 4 or state_feature_dim < 4:
            raise ValueError("feature dimensions must be at least four")
        if refinement_blocks < 0:
            raise ValueError("refinement_blocks cannot be negative")
        if evidence_temperature <= 0:
            raise ValueError("evidence_temperature must be positive")
        if not 0.0 <= observed_residual_floor <= 1.0:
            raise ValueError("observed_residual_floor must be within [0, 1]")
        if completion_residual_scale <= 0:
            raise ValueError("completion_residual_scale must be positive")
        if appearance_resolution_scale not in {1, 2}:
            raise ValueError("appearance_resolution_scale must be one or two")

        self.voxel_resolution = resolution
        self.evidence_temperature = float(evidence_temperature)
        self.observed_residual_floor = float(observed_residual_floor)
        self.completion_residual_scale = float(completion_residual_scale)
        self.appearance_resolution_scale = appearance_resolution_scale
        self.image_encoder = PyramidImageEncoder(image_feature_dim)

        observation_channels = image_feature_dim + 3 + 3 + 1
        fused_channels = observation_channels * 2 + 3
        hidden_channels = max(32, state_feature_dim * 2)
        self.input_projection = nn.Sequential(
            nn.Conv3d(fused_channels, hidden_channels, 3, padding=1, bias=False),
            nn.GroupNorm(_group_count(hidden_channels), hidden_channels),
            nn.SiLU(inplace=True),
        )
        self.refinement = nn.Sequential(
            *(Residual3DBlock(hidden_channels) for _ in range(refinement_blocks))
        )
        self.density_residual_head = nn.Conv3d(hidden_channels, 1, 1)
        self.color_residual_head = nn.Conv3d(hidden_channels, 3, 1)
        self.variance_head = nn.Conv3d(hidden_channels, 1, 1)
        self.feature_head = nn.Conv3d(hidden_channels, state_feature_dim, 1)
        nn.init.zeros_(self.density_residual_head.weight)
        nn.init.zeros_(self.density_residual_head.bias)
        nn.init.zeros_(self.color_residual_head.weight)
        nn.init.zeros_(self.color_residual_head.bias)
        nn.init.constant_(self.variance_head.bias, -3.0)
        self.appearance_input_projection: nn.Module | None = None
        self.appearance_refinement: nn.Module | None = None
        self.appearance_residual_head: nn.Conv3d | None = None
        if appearance_resolution_scale == 2:
            appearance_hidden_channels = max(16, state_feature_dim)
            self.appearance_input_projection = nn.Sequential(
                nn.Conv3d(
                    hidden_channels + 5,
                    appearance_hidden_channels,
                    3,
                    padding=1,
                    bias=False,
                ),
                nn.GroupNorm(
                    _group_count(appearance_hidden_channels), appearance_hidden_channels
                ),
                nn.SiLU(inplace=True),
            )
            self.appearance_refinement = nn.Sequential(
                *(
                    Residual3DBlock(appearance_hidden_channels)
                    for _ in range(max(1, refinement_blocks // 2))
                )
            )
            self.appearance_residual_head = nn.Conv3d(appearance_hidden_channels, 3, 1)
            nn.init.zeros_(self.appearance_residual_head.weight)
            nn.init.zeros_(self.appearance_residual_head.bias)

    def forward(self, context_rgb: Tensor, cameras: Cameras, bounds: Tensor) -> SceneState:
        if context_rgb.ndim != 5 or context_rgb.shape[2] != 3:
            raise ValueError("context_rgb must have shape [B, V, 3, H, W]")
        batch_size, view_count, _, image_height, image_width = context_rgb.shape
        if cameras.leading_shape != (batch_size, view_count):
            raise ValueError("context cameras must align with context_rgb")
        if cameras.image_size != (image_height, image_width):
            raise ValueError("context camera image size must match context_rgb")
        if bounds.shape != (batch_size, 2, 3):
            raise ValueError("bounds must have shape [B, 2, 3]")

        encoded = self.image_encoder(
            context_rgb.reshape(batch_size * view_count, 3, image_height, image_width)
        )
        _, channels, feature_height, feature_width = encoded.shape
        encoded = encoded.reshape(batch_size, view_count, channels, feature_height, feature_width)

        points_per_view, normalized_grid, camera_depth, valid = _project_voxel_grid(
            bounds,
            cameras,
            self.voxel_resolution,
            image_height,
            image_width,
        )

        sampled_features = _sample_images(encoded, normalized_grid)
        sampled_rgb = _sample_images(context_rgb, normalized_grid)
        evidence_features = _fixed_rgb_pyramid(context_rgb, normalized_grid)
        confidence, provenance = _cross_view_evidence(
            evidence_features,
            sampled_rgb,
            valid,
            temperature=self.evidence_temperature,
        )
        unknown_probability = 1.0 - confidence

        camera_centers = cameras.c2w[..., :3, 3].unsqueeze(-2)
        view_direction = _normalize(points_per_view - camera_centers)
        scene_diagonal = torch.linalg.vector_norm(bounds[:, 1] - bounds[:, 0], dim=-1)
        normalized_depth = camera_depth / scene_diagonal[:, None, None].clamp_min(1e-6)
        observations = torch.cat(
            (sampled_features, sampled_rgb, view_direction, normalized_depth.unsqueeze(-1)), dim=-1
        )
        mean = (observations * provenance).sum(dim=1)
        variance = ((observations - mean[:, None]).square() * provenance).sum(dim=1)
        coverage = valid.to(observations.dtype).mean(dim=1, keepdim=False).unsqueeze(-1)
        fused = torch.cat((mean, variance, coverage, confidence, unknown_probability), dim=-1)
        fused_grid = _points_to_volume(fused, self.voxel_resolution)

        hidden = self.refinement(self.input_projection(fused_grid))
        density_residual = (
            torch.tanh(self.density_residual_head(hidden)) * self.completion_residual_scale
        )
        color_logit_residual = (
            torch.tanh(self.color_residual_head(hidden)) * self.completion_residual_scale
        )

        confidence_grid = _points_to_volume(confidence, self.voxel_resolution)
        unknown_grid = 1.0 - confidence_grid
        provenance_grid = provenance.squeeze(-1).reshape(
            batch_size, view_count, *self.voxel_resolution
        )
        base_color_points = _base_color_from_provenance(sampled_rgb, provenance)
        base_color = _points_to_volume(base_color_points, self.voxel_resolution).clamp(0.0, 1.0)
        base_density_logits = 4.0 * confidence_grid - 2.0
        completion_gate = self.observed_residual_floor + (
            1.0 - self.observed_residual_floor
        ) * unknown_grid
        density_logits = base_density_logits + completion_gate * density_residual
        base_color_logits = torch.logit(base_color.clamp(1e-4, 1.0 - 1e-4))
        color = torch.sigmoid(base_color_logits + completion_gate * color_logit_residual)
        log_variance = (self.variance_head(hidden) + 2.0 * unknown_grid).clamp(-8.0, 4.0)
        features = self.feature_head(hidden)
        evidence = StateEvidence(
            confidence=confidence_grid,
            unknown_probability=unknown_grid,
            completion_gate=completion_gate,
            provenance=provenance_grid,
            base_density_logits=base_density_logits,
            base_color=base_color,
            density_residual=density_residual,
            color_logit_residual=color_logit_residual,
        )
        appearance = None
        if self.appearance_resolution_scale == 2:
            appearance = self._build_highres_appearance(
                context_rgb,
                cameras,
                bounds,
                hidden,
            )
        return SceneState(
            density_logits,
            color,
            log_variance,
            bounds,
            features,
            evidence,
            appearance,
        )

    def _build_highres_appearance(
        self,
        context_rgb: Tensor,
        cameras: Cameras,
        bounds: Tensor,
        hidden: Tensor,
    ) -> StateAppearance:
        if (
            self.appearance_input_projection is None
            or self.appearance_refinement is None
            or self.appearance_residual_head is None
        ):
            raise RuntimeError("high-resolution appearance modules are not initialized")
        image_height, image_width = context_rgb.shape[-2:]
        resolution = tuple(
            axis * self.appearance_resolution_scale for axis in self.voxel_resolution
        )
        _, normalized_grid, _, valid = _project_voxel_grid(
            bounds,
            cameras,
            resolution,
            image_height,
            image_width,
        )
        sampled_rgb = _sample_images(context_rgb, normalized_grid)
        fixed_features = _fixed_rgb_pyramid(context_rgb, normalized_grid)
        confidence, provenance = _cross_view_evidence(
            fixed_features,
            sampled_rgb,
            valid,
            temperature=self.evidence_temperature,
        )
        confidence_grid = _points_to_volume(confidence, resolution)
        unknown_grid = 1.0 - confidence_grid
        completion_gate = self.observed_residual_floor + (
            1.0 - self.observed_residual_floor
        ) * unknown_grid
        provenance_grid = provenance.squeeze(-1).reshape(
            context_rgb.shape[0], context_rgb.shape[1], *resolution
        )
        base_color_points = _base_color_from_provenance(sampled_rgb, provenance)
        base_color = _points_to_volume(base_color_points, resolution).clamp(0.0, 1.0)

        lifted_hidden = functional.interpolate(
            hidden,
            size=resolution,
            mode="trilinear",
            align_corners=False,
        )
        appearance_inputs = torch.cat(
            (lifted_hidden, base_color, confidence_grid, unknown_grid), dim=1
        )
        appearance_hidden = self.appearance_refinement(
            self.appearance_input_projection(appearance_inputs)
        )
        residual = (
            torch.tanh(self.appearance_residual_head(appearance_hidden))
            * self.completion_residual_scale
        )
        return StateAppearance(
            confidence=confidence_grid,
            unknown_probability=unknown_grid,
            completion_gate=completion_gate,
            provenance=provenance_grid,
            base_color=base_color,
            color_logit_residual=residual,
        )


class DualEvidenceStateEncoder(EvidenceResidualStateEncoder):
    """Add a detached local ray-competition gate without changing V5 trainable parameters."""

    def __init__(
        self,
        *,
        voxel_resolution: int | Sequence[int],
        image_feature_dim: int,
        state_feature_dim: int,
        refinement_blocks: int,
        evidence_temperature: float,
        surface_peak_temperature: float,
        observed_residual_floor: float,
        completion_residual_scale: float,
        appearance_resolution_scale: int = 1,
        surface_peak_chunk_size: int = 4096,
    ) -> None:
        super().__init__(
            voxel_resolution=voxel_resolution,
            image_feature_dim=image_feature_dim,
            state_feature_dim=state_feature_dim,
            refinement_blocks=refinement_blocks,
            evidence_temperature=evidence_temperature,
            observed_residual_floor=observed_residual_floor,
            completion_residual_scale=completion_residual_scale,
            appearance_resolution_scale=appearance_resolution_scale,
        )
        if surface_peak_temperature <= 0:
            raise ValueError("surface_peak_temperature must be positive")
        if surface_peak_chunk_size < 1:
            raise ValueError("surface_peak_chunk_size must be positive")
        self.surface_peak_temperature = float(surface_peak_temperature)
        self.surface_peak_chunk_size = int(surface_peak_chunk_size)

    def forward(self, context_rgb: Tensor, cameras: Cameras, bounds: Tensor) -> SceneState:
        v5_state = super().forward(context_rgb, cameras, bounds)
        if v5_state.evidence is None:
            raise RuntimeError("dual evidence requires the V5 evidence construction")
        appearance_evidence = v5_state.evidence
        peakness = _surface_peakness(
            context_rgb,
            cameras,
            bounds,
            self.voxel_resolution,
            evidence_temperature=self.evidence_temperature,
            surface_peak_temperature=self.surface_peak_temperature,
            chunk_size=self.surface_peak_chunk_size,
        ).to(dtype=appearance_evidence.confidence.dtype)
        appearance_confidence = appearance_evidence.confidence
        localization_support = (appearance_confidence * peakness).clamp(0.0, 1.0)
        localization_unknown = 1.0 - localization_support
        appearance_unknown = 1.0 - appearance_confidence
        localization_gate = self.observed_residual_floor + (
            1.0 - self.observed_residual_floor
        ) * localization_unknown
        appearance_gate = appearance_evidence.completion_gate
        base_density_logits = 4.0 * localization_support - 2.0
        density_logits = (
            base_density_logits + localization_gate * appearance_evidence.density_residual
        )
        dual_evidence = DualEvidence(
            surface_localization_support=localization_support,
            appearance_confidence=appearance_confidence,
            surface_peakness=peakness,
            localization_unknown_probability=localization_unknown,
            appearance_unknown_probability=appearance_unknown,
            localization_completion_gate=localization_gate,
            appearance_completion_gate=appearance_gate,
            provenance=appearance_evidence.provenance,
            base_density_logits=base_density_logits,
            base_color=appearance_evidence.base_color,
            density_residual=appearance_evidence.density_residual,
            color_logit_residual=appearance_evidence.color_logit_residual,
        )
        return SceneState(
            density_logits=density_logits,
            color=v5_state.color,
            log_variance=v5_state.log_variance,
            bounds=v5_state.bounds,
            features=v5_state.features,
            evidence=None,
            appearance=v5_state.appearance,
            dual_evidence=dual_evidence,
        )


@torch.no_grad()
def _surface_peakness(
    context_rgb: Tensor,
    cameras: Cameras,
    bounds: Tensor,
    resolution: tuple[int, int, int],
    *,
    evidence_temperature: float,
    surface_peak_temperature: float,
    chunk_size: int,
) -> Tensor:
    """Return native-grid local ray-competition support in detached FP32."""

    if context_rgb.ndim != 5 or context_rgb.shape[2] != 3:
        raise ValueError("context_rgb must have shape [B, V, 3, H, W]")
    batch_size, view_count, _, image_height, image_width = context_rgb.shape
    if cameras.leading_shape != (batch_size, view_count):
        raise ValueError("context cameras must align with context_rgb")
    if bounds.shape != (batch_size, 2, 3):
        raise ValueError("bounds must have shape [B, 2, 3]")
    if evidence_temperature <= 0 or surface_peak_temperature <= 0:
        raise ValueError("evidence temperatures must be positive")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    if view_count < 2:
        return torch.zeros(
            batch_size,
            1,
            *resolution,
            device=context_rgb.device,
            dtype=torch.float32,
        )

    with torch.autocast(device_type=context_rgb.device.type, enabled=False):
        rgb = context_rgb.detach().float()
        bounds_fp32 = bounds.detach().float()
        cameras_fp32 = cameras.to(dtype=torch.float32)
        centers = make_voxel_centers(bounds_fp32, resolution).reshape(batch_size, -1, 3)
        point_count = centers.shape[1]
        center_points_per_view = centers[:, None].expand(
            batch_size, view_count, point_count, 3
        )
        center_pixels, _, center_valid = project_world(
            center_points_per_view, cameras_fp32
        )
        center_grid = _pixels_to_grid(center_pixels, image_height, image_width)
        reference_features = _fixed_rgb_pyramid(rgb, center_grid)
        reference_rgb = _sample_images(rgb, center_grid)

        depth, height, width = resolution
        axis_sizes = torch.tensor(
            [width, height, depth],
            device=bounds_fp32.device,
            dtype=bounds_fp32.dtype,
        )
        delta = ((bounds_fp32[:, 1] - bounds_fp32[:, 0]) / axis_sizes).amin(dim=-1)
        offsets = torch.tensor(
            [-2.0, -1.0, 0.0, 1.0, 2.0],
            device=bounds_fp32.device,
            dtype=bounds_fp32.dtype,
        )
        camera_centers = cameras_fp32.c2w[..., :3, 3]
        peakness_sum = torch.zeros(
            batch_size, point_count, device=bounds_fp32.device, dtype=torch.float32
        )
        reference_count = torch.zeros_like(peakness_sum)
        other_view_mask = ~torch.eye(
            view_count, device=bounds_fp32.device, dtype=torch.bool
        )

        for reference_index in range(view_count):
            for start in range(0, point_count, chunk_size):
                stop = min(start + chunk_size, point_count)
                center_chunk = centers[:, start:stop]
                direction = _normalize(
                    center_chunk - camera_centers[:, reference_index, None]
                )
                candidates = center_chunk[:, :, None] + (
                    direction[:, :, None]
                    * delta[:, None, None, None]
                    * offsets[None, None, :, None]
                )
                inside = (
                    (candidates >= bounds_fp32[:, None, None, 0])
                    & (candidates <= bounds_fp32[:, None, None, 1])
                ).all(dim=-1)
                chunk_points = candidates.reshape(batch_size, -1, 3)
                points_per_view = chunk_points[:, None].expand(
                    batch_size, view_count, chunk_points.shape[1], 3
                )
                pixels, _, projected_valid = project_world(
                    points_per_view, cameras_fp32
                )
                normalized_grid = _pixels_to_grid(pixels, image_height, image_width)
                sampled_features = _fixed_rgb_pyramid(rgb, normalized_grid).reshape(
                    batch_size, view_count, stop - start, 5, -1
                )
                sampled_rgb = _sample_images(rgb, normalized_grid).reshape(
                    batch_size, view_count, stop - start, 5, 3
                )
                projected_valid = projected_valid.reshape(
                    batch_size, view_count, stop - start, 5
                )
                observer_valid = projected_valid & inside[:, None]
                observer_valid &= other_view_mask[reference_index][None, :, None, None]
                reference_valid = projected_valid[:, reference_index] & inside
                candidate_valid = reference_valid & observer_valid.any(dim=1)

                reference_feature = reference_features[
                    :, reference_index, start:stop
                ][:, None, :, None]
                reference_color = reference_rgb[:, reference_index, start:stop][
                    :, None, :, None
                ]
                feature_agreement = (
                    functional.cosine_similarity(
                        reference_feature,
                        sampled_features,
                        dim=-1,
                        eps=1e-6,
                    )
                    + 1.0
                ) * 0.5
                rgb_error = (sampled_rgb - reference_color).square().mean(dim=-1)
                rgb_agreement = torch.exp(-rgb_error / evidence_temperature)
                pair_score = feature_agreement * rgb_agreement
                score = (
                    pair_score * observer_valid.to(pair_score.dtype)
                ).sum(dim=1) / (view_count - 1)
                peakness, contributes = _normalized_center_peakness(
                    score,
                    candidate_valid,
                    temperature=surface_peak_temperature,
                )
                peakness_sum[:, start:stop] += peakness
                reference_count[:, start:stop] += contributes.to(reference_count.dtype)

        peakness = peakness_sum / reference_count.clamp_min(1.0)
        peakness = torch.where(reference_count > 0, peakness, torch.zeros_like(peakness))
        return _points_to_volume(peakness.unsqueeze(-1), resolution).detach()


def _normalized_center_peakness(
    scores: Tensor,
    valid: Tensor,
    *,
    temperature: float,
    center_index: int = 2,
) -> tuple[Tensor, Tensor]:
    """Normalize center probability so a flat valid candidate profile maps to zero."""

    if scores.ndim < 1 or scores.shape != valid.shape:
        raise ValueError("scores and valid must have the same non-empty shape")
    if valid.dtype != torch.bool:
        raise ValueError("valid must be boolean")
    if not 0 <= center_index < scores.shape[-1]:
        raise ValueError("center_index must identify a candidate")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if not torch.isfinite(scores).all():
        raise ValueError("scores must be finite")

    candidate_count = valid.sum(dim=-1)
    contributes = valid[..., center_index] & (candidate_count >= 3)
    safe_valid = valid.clone()
    safe_valid[..., center_index] |= ~valid.any(dim=-1)
    logits = (scores.float() / temperature).masked_fill(~safe_valid, -torch.inf)
    center_probability = torch.softmax(logits, dim=-1)[..., center_index]
    count = candidate_count.to(center_probability.dtype)
    peakness = ((count * center_probability - 1.0) / (count - 1.0).clamp_min(1.0)).clamp(
        0.0, 1.0
    )
    return torch.where(contributes, peakness, torch.zeros_like(peakness)), contributes


def _cross_view_evidence(
    features: Tensor,
    rgb: Tensor,
    valid: Tensor,
    *,
    temperature: float,
) -> tuple[Tensor, Tensor]:
    """Return correspondence confidence `[B,N,1]` and view provenance `[B,V,N,1]`."""

    if features.ndim != 4 or rgb.ndim != 4 or valid.ndim != 3:
        raise ValueError("features, rgb, and valid must have [B,V,N,...] shapes")
    if features.shape[:3] != rgb.shape[:3] or features.shape[:3] != valid.shape:
        raise ValueError("features, rgb, and valid must align")
    if rgb.shape[-1] != 3:
        raise ValueError("rgb must have three channels")
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    # Evidence audits the context; it must not become a trainable escape route for completion.
    features = features.detach()
    rgb = rgb.detach()

    mask = valid.unsqueeze(-1).to(features.dtype)
    count = mask.sum(dim=1).clamp_min(1.0)
    normalized = functional.normalize(features, dim=-1, eps=1e-6) * mask
    feature_sum = normalized.sum(dim=1)
    pair_dot_sum = feature_sum.square().sum(dim=-1, keepdim=True) - mask.sum(dim=1)
    raw_count = mask.sum(dim=1)
    pair_count = (raw_count * (raw_count - 1.0)).clamp_min(1.0)
    pair_correlation = (pair_dot_sum / pair_count).clamp(-1.0, 1.0)
    feature_agreement = (pair_correlation + 1.0) * 0.5

    rgb_mean = (rgb * mask).sum(dim=1) / count
    rgb_variance = ((rgb - rgb_mean[:, None]).square() * mask).sum(dim=1) / count
    rgb_agreement = torch.exp(-rgb_variance.mean(dim=-1, keepdim=True) / temperature)
    view_count = features.shape[1]
    if view_count == 1:
        pair_support = torch.zeros_like(raw_count)
    else:
        pair_support = ((raw_count - 1.0) / (view_count - 1.0)).clamp(0.0, 1.0)
    confidence = (pair_support * feature_agreement * rgb_agreement).clamp(0.0, 1.0)

    consensus = functional.normalize(feature_sum, dim=-1, eps=1e-6)
    feature_score = (normalized * consensus[:, None]).sum(dim=-1, keepdim=True)
    rgb_score = -(rgb - rgb_mean[:, None]).square().mean(dim=-1, keepdim=True) / temperature
    score = feature_score / temperature + rgb_score
    score = score.masked_fill(~valid.unsqueeze(-1), -1e4)
    provenance = torch.softmax(score, dim=1) * valid.unsqueeze(-1).to(score.dtype)
    has_valid = valid.any(dim=1, keepdim=True).unsqueeze(-1)
    provenance = provenance * has_valid.to(provenance.dtype)
    provenance = provenance / provenance.sum(dim=1, keepdim=True).clamp_min(1e-8)
    return confidence, provenance


def _project_voxel_grid(
    bounds: Tensor,
    cameras: Cameras,
    resolution: tuple[int, int, int],
    image_height: int,
    image_width: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    centers = make_voxel_centers(bounds, resolution)
    point_count = math.prod(resolution)
    flat_centers = centers.reshape(bounds.shape[0], point_count, 3)
    points_per_view = flat_centers[:, None].expand(
        bounds.shape[0], cameras.leading_shape[1], point_count, 3
    )
    with torch.autocast(device_type=bounds.device.type, enabled=False):
        pixels, camera_depth, valid = project_world(
            points_per_view.float(), cameras.to(dtype=torch.float32)
        )
        normalized_grid = _pixels_to_grid(pixels, image_height, image_width)
    return points_per_view, normalized_grid, camera_depth, valid


def _base_color_from_provenance(sampled_rgb: Tensor, provenance: Tensor) -> Tensor:
    base_color = (sampled_rgb * provenance).sum(dim=1)
    has_provenance = provenance.sum(dim=1) > 0.0
    return torch.where(
        has_provenance.expand_as(base_color),
        base_color,
        torch.full_like(base_color, 0.5),
    )


def _sample_images(images: Tensor, normalized_grid: Tensor) -> Tensor:
    batch_size, view_count, channels, _, _ = images.shape
    point_count = normalized_grid.shape[-2]
    flattened_images = images.reshape(batch_size * view_count, channels, *images.shape[-2:])
    grid = normalized_grid.reshape(batch_size * view_count, point_count, 1, 2)
    sampled = functional.grid_sample(
        flattened_images, grid, mode="bilinear", padding_mode="zeros", align_corners=True
    )
    return sampled.reshape(batch_size, view_count, channels, point_count).permute(0, 1, 3, 2)


def _fixed_rgb_pyramid(images: Tensor, normalized_grid: Tensor) -> Tensor:
    batch_size, view_count, channels, height, width = images.shape
    flattened = images.reshape(batch_size * view_count, channels, height, width)
    levels = [flattened]
    for scale in (2, 4):
        if min(height, width) >= scale:
            levels.append(functional.avg_pool2d(flattened, kernel_size=scale, stride=scale))
    sampled = [
        _sample_images(
            level.reshape(batch_size, view_count, channels, *level.shape[-2:]),
            normalized_grid,
        )
        for level in levels
    ]
    return torch.cat(sampled, dim=-1)


def _pixels_to_grid(pixels: Tensor, height: int, width: int) -> Tensor:
    x = pixels[..., 0] / max(width - 1, 1) * 2.0 - 1.0
    y = pixels[..., 1] / max(height - 1, 1) * 2.0 - 1.0
    return torch.stack((x, y), dim=-1)


def _points_to_volume(values: Tensor, resolution: tuple[int, int, int]) -> Tensor:
    return values.reshape(values.shape[0], *resolution, values.shape[-1]).permute(0, 4, 1, 2, 3)


def _normalize(values: Tensor) -> Tensor:
    return values / torch.linalg.vector_norm(values, dim=-1, keepdim=True).clamp_min(1e-8)


def _group_count(channels: int) -> int:
    for groups in (8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1
