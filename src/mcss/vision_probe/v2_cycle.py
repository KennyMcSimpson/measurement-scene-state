"""Source-annotated round-trip hypothesis; never accepts target annotations.

The two hard cosine nearest-neighbor searches are computed independently using
V1's frozen matching rule. Area occupancy and equal foreground-object weighting
match its source-mask convention. A good cycle is not proof of correct matching.
"""

from __future__ import annotations

import time

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

from mcss.vision_probe.opportunity_metrics import correspondence


class UnsupportedSourceAnnotation(ValueError):
    """UNSUPPORTED: the source annotation cannot supply the declared signal."""


@torch.no_grad()
def source_cycle(
    source_features: Tensor,
    target_features: Tensor,
    source_mask: np.ndarray | Tensor | None,
    source_id: str,
    target_id: str,
) -> dict:
    """Return JSON-safe cycle statistics and measured computation accounting.

    Features are HWC; production V2 uses true 32x32 grids. Original foreground
    IDs are retained even when their occupancy is less than one token. Void 255
    is not an object. Soft IoU uses product intersection (not min intersection).
    """
    if not isinstance(source_id, str) or not source_id.strip():
        raise ValueError("source_id must identify a frame")
    if not isinstance(target_id, str) or not target_id.strip():
        raise ValueError("target_id must identify a frame")
    if source_id == target_id:
        raise ValueError("Source and target must be different frames")
    if source_mask is None:
        raise UnsupportedSourceAnnotation("UNSUPPORTED: source annotation is missing")
    mask = (
        source_mask.detach().cpu().numpy()
        if isinstance(source_mask, Tensor)
        else np.asarray(source_mask)
    )
    if mask.ndim != 2 or not mask.size or mask.dtype.kind not in "biu":
        raise UnsupportedSourceAnnotation(
            "UNSUPPORTED: source mask must be a nonempty integer HW array"
        )
    if mask.min() < 0:
        raise UnsupportedSourceAnnotation("UNSUPPORTED: negative source object IDs")
    object_ids = [int(v) for v in np.unique(mask) if v not in (0, 255)]
    if not object_ids:
        raise UnsupportedSourceAnnotation("UNSUPPORTED: source contains no foreground objects")
    if isinstance(source_features, Tensor) and source_features.is_cuda:
        torch.cuda.synchronize(source_features.device)
    started = time.perf_counter()
    # The reverse search is not inversion of the first, and is never forced closed.
    target_to_source = correspondence(source_features, target_features).reshape(-1)
    source_to_target = correspondence(target_features, source_features).reshape(-1)
    if source_features.is_cuda:
        torch.cuda.synchronize(source_features.device)
    matching_seconds = time.perf_counter() - started
    per_object = []
    for object_id in object_ids:
        binary = torch.as_tensor(
            mask == object_id, dtype=torch.float32, device=source_features.device
        )[None, None]
        original = F.interpolate(binary, size=source_features.shape[:2], mode="area").reshape(-1)
        target_prediction = original[target_to_source]
        returned = target_prediction[source_to_target]
        intersection = (returned * original).sum()
        union = (returned + original - returned * original).sum()
        # Nonempty original source annotation has positive area occupancy.
        if float(union) <= 0:
            raise UnsupportedSourceAnnotation(
                "UNSUPPORTED: foreground lost during occupancy construction"
            )
        per_object.append(
            {
                "object_id": object_id,
                "soft_iou": float(intersection / union),
                "intersection": float(intersection),
                "union": float(union),
                "source_occupancy_sum": float(original.sum()),
                "returned_occupancy_sum": float(returned.sum()),
            }
        )
    if source_features.is_cuda:
        torch.cuda.synchronize(source_features.device)
    return {
        "L_cycle": 1.0 - float(np.mean([r["soft_iou"] for r in per_object])),
        "per_object": per_object,
        "source_id": source_id,
        "target_id": target_id,
        "source_grid": list(source_features.shape[:2]),
        "target_grid": list(target_features.shape[:2]),
        "matching": "independent_bidirectional_hard_cosine_lowest_flat_index_ties",
        "foreground_weighting": "equal_original_source_object_ids",
        "target_to_source_unique": int(target_to_source.unique().numel()),
        "source_to_target_unique": int(source_to_target.unique().numel()),
        "zero_source_tokens": int((source_features.norm(dim=-1) == 0).sum()),
        "zero_target_tokens": int((target_features.norm(dim=-1) == 0).sum()),
        "cost": {
            "correspondence_calls": 2,
            "cycle_calls": 1,
            "correspondence_seconds": matching_seconds,
            "total_seconds": time.perf_counter() - started,
            "timing": "wall time with CUDA synchronization when applicable",
        },
    }
