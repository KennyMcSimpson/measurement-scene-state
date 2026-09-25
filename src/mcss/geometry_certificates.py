"""Context-only correspondence proposals and fixed multi-view geometry certificates."""

from __future__ import annotations

import copy
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import torch
from torch import Tensor, nn

from mcss.types import Cameras


@dataclass(frozen=True)
class CertificateConfig:
    proposer: Literal["loftr_indoor_new"] = "loftr_indoor_new"
    confidence_threshold: float = 0.2
    reciprocal_threshold_px: float = 1.5
    cycle_threshold_px: float = 2.0
    epipolar_threshold_px: float = 1.5
    reprojection_threshold_px: float = 1.5
    verification_threshold_px: float = 2.0
    min_triangulation_angle_deg: float = 1.0
    grid_stride: int = 8
    device: str = "auto"
    model_cache_dir: str = "outputs/model_cache/torch"

    def __post_init__(self) -> None:
        if self.proposer != "loftr_indoor_new":
            raise ValueError("proposer must be loftr_indoor_new")
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be in [0, 1]")
        for name in (
            "reciprocal_threshold_px",
            "cycle_threshold_px",
            "epipolar_threshold_px",
            "reprojection_threshold_px",
            "verification_threshold_px",
            "min_triangulation_angle_deg",
        ):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if self.verification_threshold_px < self.reprojection_threshold_px:
            raise ValueError("verification threshold cannot be tighter than pair reprojection")
        if self.grid_stride < 1:
            raise ValueError("grid_stride must be positive")
        if not self.model_cache_dir:
            raise ValueError("model_cache_dir must be non-empty")


@dataclass(frozen=True)
class TriangulationResult:
    point: Tensor
    cheirality: bool
    reprojection_error_px: float
    triangulation_angle_deg: float


@dataclass(frozen=True)
class GeometryCertificate:
    source_view: int
    source_xy: tuple[float, float]
    matched_view: int
    matched_xy: tuple[float, float]
    verification_view: int
    verification_xy: tuple[float, float]
    point: tuple[float, float, float]
    descriptor_distance: float
    reciprocal_error_px: float
    cycle_error_px: float
    reprojection_error_px: float
    verification_error_px: float
    triangulation_angle_deg: float
    proposal_confidence: float = 1.0
    epipolar_error_px: float = 0.0


@dataclass(frozen=True)
class RawObservation:
    source_view: int
    source_xy: tuple[float, float]
    matched_view: int
    matched_xy: tuple[float, float]
    proposal_confidence: float
    point: tuple[float, float, float] | None
    cheirality: bool
    reprojection_error_px: float | None
    triangulation_angle_deg: float | None
    epipolar_error_px: float | None


@dataclass(frozen=True)
class CertificateBuildResult:
    image_size: tuple[int, int]
    context_views: int
    stage_counts: dict[str, int]
    certified_source_coverage: float
    source_grid_samples: int
    certificates: tuple[GeometryCertificate, ...]
    raw_observations: tuple[RawObservation, ...]
    pair_match_counts: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class _PairMatches:
    source: Tensor
    target: Tensor
    confidence: Tensor


_MATCHER_CACHE: dict[tuple[str, str, str], nn.Module] = {}


def fundamental_matrix(cameras: Cameras, source_view: int, target_view: int) -> Tensor:
    """Return the calibrated fundamental matrix mapping source points to target lines."""

    _validate_unbatched_cameras(cameras)
    _validate_view_index(cameras, source_view)
    _validate_view_index(cameras, target_view)
    relative = torch.linalg.inv(cameras.c2w[target_view]) @ cameras.c2w[source_view]
    rotation = relative[:3, :3]
    translation = relative[:3, 3]
    tx = torch.stack(
        (
            torch.stack((translation.new_zeros(()), -translation[2], translation[1])),
            torch.stack((translation[2], translation.new_zeros(()), -translation[0])),
            torch.stack((-translation[1], translation[0], translation.new_zeros(()))),
        )
    )
    essential = tx @ rotation
    matrix = (
        torch.linalg.inv(cameras.intrinsics[target_view]).transpose(-1, -2)
        @ essential
        @ torch.linalg.inv(cameras.intrinsics[source_view])
    )
    return matrix / torch.linalg.vector_norm(matrix).clamp_min(torch.finfo(matrix.dtype).eps)


def triangulate_dlt(
    source_xy: Tensor,
    target_xy: Tensor,
    cameras: Cameras,
    source_view: int,
    target_view: int,
) -> TriangulationResult:
    """Triangulate one calibrated correspondence with linear DLT."""

    if source_xy.shape != (2,) or target_xy.shape != (2,):
        raise ValueError("source_xy and target_xy must have shape [2]")
    points, cheirality, reprojection, angle, finite = _triangulate_batch(
        source_xy.unsqueeze(0), target_xy.unsqueeze(0), cameras, source_view, target_view
    )
    return TriangulationResult(
        point=points[0],
        cheirality=bool((cheirality & finite)[0].item()),
        reprojection_error_px=float(reprojection[0].item()),
        triangulation_angle_deg=float(angle[0].item()),
    )


def build_certificates(
    context_rgb: Tensor, context_cameras: Cameras, config: CertificateConfig
) -> CertificateBuildResult:
    """Build certificates without accepting any image target or geometry label."""

    if context_rgb.ndim != 4 or context_rgb.shape[1] != 3:
        raise ValueError("context_rgb must have shape [V, 3, H, W]")
    views, _, height, width = context_rgb.shape
    if views < 3:
        raise ValueError("at least three context views are required for certification")
    if context_cameras.leading_shape != (views,):
        raise ValueError("context cameras must have one view dimension matching context_rgb")
    if context_cameras.image_size != (height, width):
        raise ValueError("camera image_size must match context_rgb")
    if not torch.isfinite(context_rgb).all():
        raise ValueError("context_rgb must contain only finite values")

    device = _resolve_device(config.device)
    rgb = context_rgb.to(device=device, dtype=torch.float32)
    cameras = context_cameras.to(device=device, dtype=torch.float32)
    grayscale = (
        rgb[:, 0:1] * 0.2989 + rgb[:, 1:2] * 0.5870 + rgb[:, 2:3] * 0.1140
    ).clamp(0.0, 1.0)
    matcher = _get_matcher(config, device)
    pairs: dict[tuple[int, int], _PairMatches] = {}
    with torch.inference_mode():
        for source_view in range(views):
            for target_view in range(views):
                if source_view == target_view:
                    continue
                pairs[source_view, target_view] = _propose_pair(
                    matcher,
                    grayscale[source_view : source_view + 1],
                    grayscale[target_view : target_view + 1],
                    config.confidence_threshold,
                )

    stage_counts = {name: 0 for name in ("raw", "reciprocal", "cycle", "geometric", "certified")}
    certificates: list[GeometryCertificate] = []
    raw_observations: list[RawObservation] = []
    pair_match_counts: dict[str, int] = {}
    certified_cells: set[tuple[int, int, int]] = set()

    for source_view in range(views):
        for matched_view in range(views):
            if source_view == matched_view:
                continue
            pair = pairs[source_view, matched_view]
            pair_match_counts[f"{source_view}->{matched_view}"] = len(pair.source)
            stage_counts["raw"] += len(pair.source)
            if len(pair.source) == 0:
                continue

            raw_points, raw_cheirality, raw_reprojection, raw_angle, raw_finite = (
                _triangulate_batch(
                    pair.source, pair.target, cameras, source_view, matched_view
                )
            )
            raw_epipolar = _epipolar_distance(
                pair.source,
                pair.target,
                fundamental_matrix(cameras, source_view, matched_view),
            )
            for index in range(len(pair.source)):
                finite = bool(raw_finite[index].item())
                point = _point_tuple(raw_points[index]) if finite else None
                raw_observations.append(
                    RawObservation(
                        source_view=source_view,
                        source_xy=_xy_tuple(pair.source[index]),
                        matched_view=matched_view,
                        matched_xy=_xy_tuple(pair.target[index]),
                        proposal_confidence=float(pair.confidence[index].item()),
                        point=point,
                        cheirality=bool((raw_cheirality & raw_finite)[index].item()),
                        reprojection_error_px=(
                            float(raw_reprojection[index].item()) if finite else None
                        ),
                        triangulation_angle_deg=float(raw_angle[index].item()) if finite else None,
                        epipolar_error_px=float(raw_epipolar[index].item()) if finite else None,
                    )
                )

            reverse = pairs[matched_view, source_view]
            reverse_index, reverse_source_error = _nearest(pair.target, reverse.source)
            reverse_target = _safe_gather(reverse.target, reverse_index, pair.source)
            reverse_target_error = torch.linalg.vector_norm(reverse_target - pair.source, dim=-1)
            reciprocal_error = torch.maximum(reverse_source_error, reverse_target_error)
            reciprocal = reciprocal_error <= config.reciprocal_threshold_px
            stage_counts["reciprocal"] += int(reciprocal.sum().item())
            if not reciprocal.any():
                continue

            original_indices = torch.nonzero(reciprocal, as_tuple=False)[:, 0]
            source_xy = pair.source[original_indices]
            matched_xy = pair.target[original_indices]
            confidence = pair.confidence[original_indices]
            reciprocal_error = reciprocal_error[original_indices]

            cycle_error, verification_view, verification_xy = _best_cycle(
                source_xy,
                matched_xy,
                pairs,
                source_view,
                matched_view,
                views,
            )
            cycle = cycle_error <= config.cycle_threshold_px
            stage_counts["cycle"] += int(cycle.sum().item())
            if not cycle.any():
                continue

            source_xy = source_xy[cycle]
            matched_xy = matched_xy[cycle]
            confidence = confidence[cycle]
            reciprocal_error = reciprocal_error[cycle]
            cycle_error = cycle_error[cycle]
            verification_view = verification_view[cycle]
            verification_xy = verification_xy[cycle]

            points, cheirality, reprojection, angle, finite = _triangulate_batch(
                source_xy, matched_xy, cameras, source_view, matched_view
            )
            epipolar = _epipolar_distance(
                source_xy,
                matched_xy,
                fundamental_matrix(cameras, source_view, matched_view),
            )
            geometric = (
                finite
                & cheirality
                & (epipolar <= config.epipolar_threshold_px)
                & (reprojection <= config.reprojection_threshold_px)
                & (angle >= config.min_triangulation_angle_deg)
            )
            stage_counts["geometric"] += int(geometric.sum().item())
            if not geometric.any():
                continue

            source_xy = source_xy[geometric]
            matched_xy = matched_xy[geometric]
            confidence = confidence[geometric]
            reciprocal_error = reciprocal_error[geometric]
            cycle_error = cycle_error[geometric]
            verification_view = verification_view[geometric]
            verification_xy = verification_xy[geometric]
            points = points[geometric]
            reprojection = reprojection[geometric]
            angle = angle[geometric]
            epipolar = epipolar[geometric]

            verification_projection, verification_valid = _project_by_view(
                points, cameras, verification_view
            )
            verification_error = torch.linalg.vector_norm(
                verification_projection - verification_xy, dim=-1
            )
            certified = verification_valid & (
                verification_error <= config.verification_threshold_px
            )
            stage_counts["certified"] += int(certified.sum().item())

            for index in torch.nonzero(certified, as_tuple=False)[:, 0].tolist():
                source_cell_x = min(width - 1, max(0, int(source_xy[index, 0].item())))
                source_cell_y = min(height - 1, max(0, int(source_xy[index, 1].item())))
                certified_cells.add(
                    (
                        source_view,
                        source_cell_y // config.grid_stride,
                        source_cell_x // config.grid_stride,
                    )
                )
                certificates.append(
                    GeometryCertificate(
                        source_view=source_view,
                        source_xy=_xy_tuple(source_xy[index]),
                        matched_view=matched_view,
                        matched_xy=_xy_tuple(matched_xy[index]),
                        verification_view=int(verification_view[index].item()),
                        verification_xy=_xy_tuple(verification_xy[index]),
                        point=_point_tuple(points[index]),
                        descriptor_distance=float(1.0 - confidence[index].item()),
                        reciprocal_error_px=float(reciprocal_error[index].item()),
                        cycle_error_px=float(cycle_error[index].item()),
                        reprojection_error_px=float(reprojection[index].item()),
                        verification_error_px=float(verification_error[index].item()),
                        triangulation_angle_deg=float(angle[index].item()),
                        proposal_confidence=float(confidence[index].item()),
                        epipolar_error_px=float(epipolar[index].item()),
                    )
                )

    grid_height = math.ceil(height / config.grid_stride)
    grid_width = math.ceil(width / config.grid_stride)
    source_grid_samples = views * grid_height * grid_width
    coverage = len(certified_cells) / source_grid_samples
    certificates.sort(
        key=lambda item: (
            item.source_view,
            item.source_xy[1],
            item.source_xy[0],
            item.matched_view,
        )
    )
    raw_observations.sort(
        key=lambda item: (
            item.source_view,
            item.source_xy[1],
            item.source_xy[0],
            item.matched_view,
        )
    )
    return CertificateBuildResult(
        image_size=(height, width),
        context_views=views,
        stage_counts=stage_counts,
        certified_source_coverage=coverage,
        source_grid_samples=source_grid_samples,
        certificates=tuple(certificates),
        raw_observations=tuple(raw_observations),
        pair_match_counts=pair_match_counts,
    )


def _get_matcher(config: CertificateConfig, device: torch.device) -> nn.Module:
    cache_path = str(Path(config.model_cache_dir).resolve())
    key = (config.proposer, str(device), cache_path)
    if key not in _MATCHER_CACHE:
        from kornia.feature import LoFTR
        from kornia.feature.loftr.loftr import default_cfg

        Path(cache_path).mkdir(parents=True, exist_ok=True)
        torch.hub.set_dir(cache_path)
        matcher = LoFTR(pretrained="indoor_new", config=copy.deepcopy(default_cfg))
        matcher.requires_grad_(False).eval().to(device)
        _MATCHER_CACHE[key] = matcher
    return _MATCHER_CACHE[key]


def _propose_pair(
    matcher: nn.Module, image_source: Tensor, image_target: Tensor, confidence_threshold: float
) -> _PairMatches:
    output = matcher({"image0": image_source, "image1": image_target})
    source = output["keypoints0"].to(dtype=torch.float32)
    target = output["keypoints1"].to(dtype=torch.float32)
    confidence = output["confidence"].to(dtype=torch.float32)
    valid = (
        torch.isfinite(source).all(dim=-1)
        & torch.isfinite(target).all(dim=-1)
        & torch.isfinite(confidence)
        & (confidence >= confidence_threshold)
    )
    source = source[valid]
    target = target[valid]
    confidence = confidence[valid]
    if len(source):
        width = image_source.shape[-1]
        order = torch.argsort(source[:, 1] * (width + 1) + source[:, 0], stable=True)
        source = source[order]
        target = target[order]
        confidence = confidence[order]
    return _PairMatches(source, target, confidence)


def _best_cycle(
    source_xy: Tensor,
    matched_xy: Tensor,
    pairs: dict[tuple[int, int], _PairMatches],
    source_view: int,
    matched_view: int,
    views: int,
) -> tuple[Tensor, Tensor, Tensor]:
    count = len(source_xy)
    best_error = torch.full((count,), torch.inf, device=source_xy.device)
    best_view = torch.full((count,), -1, dtype=torch.long, device=source_xy.device)
    best_xy = torch.zeros_like(source_xy)
    for third_view in range(views):
        if third_view in {source_view, matched_view}:
            continue
        forward = pairs[matched_view, third_view]
        backward = pairs[third_view, source_view]
        forward_index, forward_error = _nearest(matched_xy, forward.source)
        third_xy = _safe_gather(forward.target, forward_index, matched_xy)
        backward_index, backward_error = _nearest(third_xy, backward.source)
        returned_xy = _safe_gather(backward.target, backward_index, source_xy)
        closure_error = torch.linalg.vector_norm(returned_xy - source_xy, dim=-1)
        cycle_error = torch.maximum(
            torch.maximum(forward_error, backward_error), closure_error
        )
        better = cycle_error < best_error
        best_error = torch.where(better, cycle_error, best_error)
        best_view = torch.where(
            better, torch.full_like(best_view, third_view), best_view
        )
        best_xy = torch.where(better.unsqueeze(-1), third_xy, best_xy)
    return best_error, best_view, best_xy


def _triangulate_batch(
    source_xy: Tensor,
    target_xy: Tensor,
    cameras: Cameras,
    source_view: int,
    target_view: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    _validate_unbatched_cameras(cameras)
    _validate_view_index(cameras, source_view)
    _validate_view_index(cameras, target_view)
    if source_xy.ndim != 2 or source_xy.shape[-1] != 2 or target_xy.shape != source_xy.shape:
        raise ValueError("source and target pixels must share shape [N, 2]")
    projection_source = (
        cameras.intrinsics[source_view] @ torch.linalg.inv(cameras.c2w[source_view])[:3]
    )
    projection_target = (
        cameras.intrinsics[target_view] @ torch.linalg.inv(cameras.c2w[target_view])[:3]
    )
    rows = torch.stack(
        (
            source_xy[:, 0, None] * projection_source[2] - projection_source[0],
            source_xy[:, 1, None] * projection_source[2] - projection_source[1],
            target_xy[:, 0, None] * projection_target[2] - projection_target[0],
            target_xy[:, 1, None] * projection_target[2] - projection_target[1],
        ),
        dim=1,
    )
    _, _, vh = torch.linalg.svd(rows)
    homogeneous = vh[:, -1]
    epsilon = torch.finfo(homogeneous.dtype).eps
    finite = torch.isfinite(homogeneous).all(dim=-1) & (homogeneous[:, 3].abs() > epsilon)
    safe_scale = torch.where(
        homogeneous[:, 3].abs() > epsilon, homogeneous[:, 3], torch.ones_like(homogeneous[:, 3])
    )
    points = homogeneous[:, :3] / safe_scale[:, None]
    finite = finite & torch.isfinite(points).all(dim=-1)

    projected_source, depth_source = _project_points(points, cameras, source_view)
    projected_target, depth_target = _project_points(points, cameras, target_view)
    reprojection = torch.maximum(
        torch.linalg.vector_norm(projected_source - source_xy, dim=-1),
        torch.linalg.vector_norm(projected_target - target_xy, dim=-1),
    )
    cheirality = (depth_source > epsilon) & (depth_target > epsilon)

    center_source = cameras.c2w[source_view, :3, 3]
    center_target = cameras.c2w[target_view, :3, 3]
    ray_source = points - center_source
    ray_target = points - center_target
    ray_source = ray_source / torch.linalg.vector_norm(ray_source, dim=-1, keepdim=True).clamp_min(
        epsilon
    )
    ray_target = ray_target / torch.linalg.vector_norm(ray_target, dim=-1, keepdim=True).clamp_min(
        epsilon
    )
    cosine = (ray_source * ray_target).sum(dim=-1).clamp(-1.0, 1.0)
    angle = torch.rad2deg(torch.acos(cosine))
    return points, cheirality, reprojection, angle, finite


def _project_points(points: Tensor, cameras: Cameras, view: int) -> tuple[Tensor, Tensor]:
    homogeneous = torch.cat((points, torch.ones_like(points[:, :1])), dim=-1)
    camera_points = homogeneous @ torch.linalg.inv(cameras.c2w[view])[:3].transpose(0, 1)
    projected = camera_points @ cameras.intrinsics[view].transpose(0, 1)
    epsilon = torch.finfo(points.dtype).eps
    safe_depth = torch.where(
        projected[:, 2].abs() > epsilon, projected[:, 2], torch.ones_like(projected[:, 2])
    )
    return projected[:, :2] / safe_depth[:, None], camera_points[:, 2]


def _project_by_view(points: Tensor, cameras: Cameras, views: Tensor) -> tuple[Tensor, Tensor]:
    pixels = torch.zeros_like(points[:, :2])
    valid = torch.zeros((len(points),), dtype=torch.bool, device=points.device)
    height, width = cameras.image_size
    for view in torch.unique(views).tolist():
        if view < 0:
            continue
        mask = views == view
        projected, depth = _project_points(points[mask], cameras, int(view))
        pixels[mask] = projected
        valid[mask] = (
            torch.isfinite(projected).all(dim=-1)
            & (depth > torch.finfo(depth.dtype).eps)
            & (projected[:, 0] >= 0)
            & (projected[:, 0] <= width - 1)
            & (projected[:, 1] >= 0)
            & (projected[:, 1] <= height - 1)
        )
    return pixels, valid


def _epipolar_distance(source_xy: Tensor, target_xy: Tensor, matrix: Tensor) -> Tensor:
    homogeneous_source = torch.cat((source_xy, torch.ones_like(source_xy[:, :1])), dim=-1)
    homogeneous_target = torch.cat((target_xy, torch.ones_like(target_xy[:, :1])), dim=-1)
    target_lines = homogeneous_source @ matrix.transpose(0, 1)
    numerator = (homogeneous_target * target_lines).sum(dim=-1).abs()
    denominator = torch.linalg.vector_norm(target_lines[:, :2], dim=-1).clamp_min(
        torch.finfo(source_xy.dtype).eps
    )
    return numerator / denominator


def _nearest(query: Tensor, candidates: Tensor) -> tuple[Tensor, Tensor]:
    if len(candidates) == 0:
        return (
            torch.zeros((len(query),), dtype=torch.long, device=query.device),
            torch.full((len(query),), torch.inf, device=query.device),
        )
    indices = []
    distances = []
    for start in range(0, len(query), 1024):
        matrix = torch.cdist(query[start : start + 1024], candidates)
        distance, index = matrix.min(dim=1)
        indices.append(index)
        distances.append(distance)
    return torch.cat(indices), torch.cat(distances)


def _safe_gather(values: Tensor, indices: Tensor, fallback: Tensor) -> Tensor:
    if len(values) == 0:
        return torch.full_like(fallback, torch.nan)
    return values[indices]


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def _validate_unbatched_cameras(cameras: Cameras) -> None:
    if len(cameras.leading_shape) != 1:
        raise ValueError("certificate geometry requires unbatched cameras with shape [V, ...]")


def _validate_view_index(cameras: Cameras, view: int) -> None:
    if not 0 <= view < cameras.leading_shape[0]:
        raise IndexError(f"camera view {view} is out of range")


def _xy_tuple(value: Tensor) -> tuple[float, float]:
    return float(value[0].item()), float(value[1].item())


def _point_tuple(value: Tensor) -> tuple[float, float, float]:
    return float(value[0].item()), float(value[1].item()), float(value[2].item())

