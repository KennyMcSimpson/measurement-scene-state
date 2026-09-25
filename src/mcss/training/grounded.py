"""Offline, query-supervised carrier objectives kept outside runtime modules."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch
from torch import Tensor

from mcss.dynamic.cache import ObservationCache
from mcss.dynamic.feedback import anchored_observation
from mcss.dynamic.types import OnlineObservation
from mcss.geometry import transform_cameras
from mcss.losses import MeasurementLoss
from mcss.measurements import FixedMeasurementRenderer
from mcss.types import Cameras, SceneState


@dataclass(frozen=True)
class GroundedPairResult:
    """Independent states and their shared-query offline objective."""

    loss: Tensor
    terms: Mapping[str, Tensor]
    first_state: SceneState
    second_state: SceneState
    first_anchor_c2w: Tensor
    second_anchor_c2w: Tensor


def build_grounded_state(
    carrier,
    context: Sequence[OnlineObservation],
    episode_id: str,
) -> tuple[SceneState, Tensor]:
    """Build an OFF state from one raw context under its first-camera anchor."""

    context = tuple(context)
    if not episode_id:
        raise ValueError("episode_id must be nonempty")
    _validate_context(context)

    anchor_c2w = context[0].camera.c2w
    cache = ObservationCache(episode_id=episode_id, scene_id=context[0].scene_id)
    fast = carrier.initial_fast(episode_id)
    for observation in context:
        arrived = anchored_observation(observation, anchor_c2w)
        cache = cache.append(arrived, carrier.encode(arrived))
    return carrier.materialize(cache, fast), anchor_c2w.clone()


def grounded_measurement_loss(
    state: SceneState,
    query_cameras: Cameras,
    query_rgb: Tensor,
    query_depth: Tensor,
    anchor_c2w: Tensor,
    renderer: FixedMeasurementRenderer,
    objective: MeasurementLoss | None = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Render a local state at raw query cameras and apply RGB/depth supervision.

    Query cameras arrive in the same raw world frame as their context.  They are rebased
    here, after state construction, so runtime scene construction never receives labels
    or query cameras.
    """

    local_query_cameras = _anchor_query_cameras(query_cameras, anchor_c2w)
    _validate_query_labels(state, local_query_cameras, query_rgb, query_depth)
    predictions = renderer(state, local_query_cameras, measurements=("rgb", "depth"))
    targets = {"rgb": query_rgb, "depth": query_depth}
    return (objective or MeasurementLoss({"rgb": 1.0, "depth": 1.0}))(predictions, targets)


def paired_context_measurement_loss(
    carrier,
    first_context: Sequence[OnlineObservation],
    second_context: Sequence[OnlineObservation],
    query_frame_ids: Sequence[int],
    query_cameras: Cameras,
    query_rgb: Tensor,
    query_depth: Tensor,
    renderer: FixedMeasurementRenderer,
    objective: MeasurementLoss | None = None,
) -> GroundedPairResult:
    """Train two independently built context states against one query set.

    The contexts share neither a cache nor fast weights.  Each query camera is converted
    independently using that context's anchor before fixed measurement rendering.
    """

    first_context = tuple(first_context)
    second_context = tuple(second_context)
    _validate_context_pair(first_context, second_context, query_frame_ids, query_cameras)
    scene_id = first_context[0].scene_id
    first_state, first_anchor_c2w = build_grounded_state(
        carrier, first_context, f"{scene_id}:grounded-context-a"
    )
    second_state, second_anchor_c2w = build_grounded_state(
        carrier, second_context, f"{scene_id}:grounded-context-b"
    )
    first_loss, first_terms = grounded_measurement_loss(
        first_state,
        query_cameras,
        query_rgb,
        query_depth,
        first_anchor_c2w,
        renderer,
        objective,
    )
    second_loss, second_terms = grounded_measurement_loss(
        second_state,
        query_cameras,
        query_rgb,
        query_depth,
        second_anchor_c2w,
        renderer,
        objective,
    )
    terms = {
        **{f"context_a/{name}": value for name, value in first_terms.items()},
        **{f"context_b/{name}": value for name, value in second_terms.items()},
    }
    return GroundedPairResult(
        loss=(first_loss + second_loss) * 0.5,
        terms=terms,
        first_state=first_state,
        second_state=second_state,
        first_anchor_c2w=first_anchor_c2w,
        second_anchor_c2w=second_anchor_c2w,
    )


def _anchor_query_cameras(query_cameras: Cameras, anchor_c2w: Tensor) -> Cameras:
    if anchor_c2w.shape != (4, 4):
        raise ValueError("anchor_c2w must describe one unbatched camera pose")
    if anchor_c2w.device != query_cameras.device or anchor_c2w.dtype != query_cameras.dtype:
        raise ValueError("anchor_c2w and query cameras must share device and dtype")
    return transform_cameras(query_cameras, torch.linalg.inv(anchor_c2w))


def _validate_context(context: tuple[OnlineObservation, ...]) -> None:
    if not context:
        raise ValueError("context must contain at least one observation")
    scene_id = context[0].scene_id
    image_size = context[0].camera.image_size
    device = context[0].rgb.device
    dtype = context[0].rgb.dtype
    previous_frame_id = -1
    for observation in context:
        if observation.scene_id != scene_id:
            raise ValueError("context cannot mix scene identities")
        if observation.camera.image_size != image_size:
            raise ValueError("context requires one image size")
        if observation.rgb.device != device or observation.rgb.dtype != dtype:
            raise ValueError("context requires one RGB device and dtype")
        if observation.frame_id <= previous_frame_id:
            raise ValueError("context frame IDs must be strictly increasing")
        previous_frame_id = observation.frame_id


def _validate_context_pair(
    first_context: tuple[OnlineObservation, ...],
    second_context: tuple[OnlineObservation, ...],
    query_frame_ids: Sequence[int],
    query_cameras: Cameras,
) -> None:
    _validate_context(first_context)
    _validate_context(second_context)
    if first_context[0].scene_id != second_context[0].scene_id:
        raise ValueError("paired contexts must belong to one scene")
    if first_context[0].camera.image_size != second_context[0].camera.image_size:
        raise ValueError("paired contexts must share an image size")
    if len(query_cameras.leading_shape) != 2:
        raise ValueError("query cameras must have shape [batch, view, ...]")
    if len(query_frame_ids) != query_cameras.leading_shape[1]:
        raise ValueError("query_frame_ids must match the query view count")
    if len(set(query_frame_ids)) != len(query_frame_ids) or any(
        not isinstance(frame_id, int) or frame_id < 0 for frame_id in query_frame_ids
    ):
        raise ValueError("query_frame_ids must be unique nonnegative integers")

    first_ids = {observation.frame_id for observation in first_context}
    second_ids = {observation.frame_id for observation in second_context}
    if first_ids & second_ids:
        raise ValueError("paired contexts must not overlap by frame ID")
    if (first_ids | second_ids) & set(query_frame_ids):
        raise ValueError("context and query frame IDs must not overlap")


def _validate_query_labels(
    state: SceneState,
    query_cameras: Cameras,
    query_rgb: Tensor,
    query_depth: Tensor,
) -> None:
    if len(query_cameras.leading_shape) != 2:
        raise ValueError("query cameras must have shape [batch, view, ...]")
    batch_size, view_count = query_cameras.leading_shape
    height, width = query_cameras.image_size
    if batch_size != state.density_logits.shape[0]:
        raise ValueError("query camera and state batch sizes must match")
    expected_rgb = (batch_size, view_count, 3, height, width)
    expected_depth = (batch_size, view_count, 1, height, width)
    if query_rgb.shape != expected_rgb or query_depth.shape != expected_depth:
        raise ValueError("query RGB/depth labels must match cameras exactly")
    if query_rgb.device != query_cameras.device or query_depth.device != query_cameras.device:
        raise ValueError("query labels and cameras must share a device")
    if query_rgb.dtype != query_cameras.dtype or query_depth.dtype != query_cameras.dtype:
        raise ValueError("query labels and cameras must share a dtype")
