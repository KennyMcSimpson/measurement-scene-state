"""Opt-in Phase A training adapter with a shared coordinate anchor.

This does not change the legacy trainer or qualify a carrier. Query labels belong to
training scenes only; independent evaluation must use the sealed evaluator instead.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor

from mcss.dynamic.types import OnlineObservation
from mcss.losses import MeasurementLoss
from mcss.measurements import FixedMeasurementRenderer
from mcss.training.grounded import (
    GroundedPairResult,
    build_grounded_state,
    grounded_measurement_loss,
)
from mcss.types import Cameras


def common_anchor_measurement_loss(
    carrier,
    first_context: Sequence[OnlineObservation],
    second_context: Sequence[OnlineObservation],
    query_frame_ids: Sequence[int],
    query_cameras: Cameras,
    query_rgb: Tensor,
    query_depth: Tensor,
    renderer: FixedMeasurementRenderer,
    *,
    split_id: str,
    objective: MeasurementLoss | None = None,
) -> GroundedPairResult:
    """Supervise two independently built OFF states in one anchor frame.

    Contexts have equal counts (at least two), share exactly their first observation,
    and have otherwise disjoint IDs. Queries are disjoint from both contexts and are
    used only after both states have been constructed. RGB and metric ray-distance
    depth are offline training targets, never carrier inputs. The caller must enforce
    physical-scene split provenance; ``split_id`` is a fail-closed contract, not proof
    of provenance. Fast offsets start at zero; no writer is invoked.

    Nonpositive/nonfinite depths are masked without changing the existing depth loss.
    Finite RGB supervision is required. Loss weights default to RGB=1, depth=1.
    """
    if split_id != "train":
        raise ValueError("This training adapter accepts only split_id='train'")
    a, b = tuple(first_context), tuple(second_context)
    if len(a) != len(b) or len(a) < 2:
        raise ValueError("Contexts require equal counts of at least two observations")
    first = a[0]
    for context in (a, b):
        ids = [o.frame_id for o in context]
        if ids != sorted(set(ids)):
            raise ValueError("Context frame IDs must be unique and strictly increasing")
        for o in context:
            if o.scene_id != first.scene_id:
                raise ValueError("Contexts must belong to one scene")
            if o.camera.image_size != first.camera.image_size:
                raise ValueError("Contexts must share an image size")
            if o.rgb.device != first.rgb.device or o.rgb.dtype != first.rgb.dtype:
                raise ValueError("Contexts must share RGB device and dtype")
    if a[0].frame_id != b[0].frame_id or not all(
        torch.equal(left, right)
        for left, right in (
            (a[0].rgb, b[0].rgb),
            (a[0].camera.c2w, b[0].camera.c2w),
            (a[0].camera.intrinsics, b[0].camera.intrinsics),
        )
    ):
        raise ValueError("Common anchor must have identical frame ID, RGB and camera")
    a_ids, b_ids = {o.frame_id for o in a}, {o.frame_id for o in b}
    if a_ids & b_ids != {first.frame_id}:
        raise ValueError("Contexts may overlap only at the common anchor")
    queries = tuple(query_frame_ids)
    if not queries or any(type(q) is not int or q < 0 for q in queries):
        raise ValueError("Query IDs must be nonempty nonnegative integers")
    if len(set(queries)) != len(queries) or set(queries) & (a_ids | b_ids):
        raise ValueError("Query IDs must be unique and disjoint from contexts")
    if query_cameras.leading_shape != (1, len(queries)):
        raise ValueError("Query cameras must have shape [1, query_count, ...]")
    if not isinstance(renderer, FixedMeasurementRenderer) or list(renderer.parameters()):
        raise ValueError("Training requires a parameter-free FixedMeasurementRenderer")
    if not torch.isfinite(query_rgb).all() or torch.any((query_rgb < 0) | (query_rgb > 1)):
        raise ValueError("Query RGB must be finite and in [0, 1]")
    depth_valid = torch.isfinite(query_depth) & (query_depth > 0)
    if not depth_valid.any():
        raise ValueError("Training query requires at least one valid metric depth")
    clean_depth = torch.where(depth_valid, query_depth, torch.zeros_like(query_depth))

    # Both state constructions finish before either query camera reaches the renderer.
    state_a, anchor_a = build_grounded_state(carrier, a, f"{first.scene_id}:common-anchor-a")
    state_b, anchor_b = build_grounded_state(carrier, b, f"{first.scene_id}:common-anchor-b")
    loss_a, terms_a = grounded_measurement_loss(
        state_a, query_cameras, query_rgb, clean_depth, anchor_a, renderer, objective
    )
    loss_b, terms_b = grounded_measurement_loss(
        state_b, query_cameras, query_rgb, clean_depth, anchor_b, renderer, objective
    )
    return GroundedPairResult(
        loss=(loss_a + loss_b) * 0.5,
        terms={
            **{f"context_a/{name}": value for name, value in terms_a.items()},
            **{f"context_b/{name}": value for name, value in terms_b.items()},
        },
        first_state=state_a,
        second_state=state_b,
        first_anchor_c2w=anchor_a,
        second_anchor_c2w=anchor_b,
    )
