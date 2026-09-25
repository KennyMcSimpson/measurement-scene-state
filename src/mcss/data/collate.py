"""Strict collation for sparse-view scene examples."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor

from mcss.types import Cameras, SceneBatch, SceneExample


def collate_scene_examples(examples: Sequence[SceneExample]) -> SceneBatch:
    if not examples:
        raise ValueError("cannot collate an empty example sequence")
    context_size = examples[0].context_cameras.image_size
    target_size = examples[0].target_cameras.image_size
    if any(example.context_cameras.image_size != context_size for example in examples):
        raise ValueError("all context image sizes in a batch must match")
    if any(example.target_cameras.image_size != target_size for example in examples):
        raise ValueError("all target image sizes in a batch must match")
    return SceneBatch(
        context_rgb=torch.stack([example.context_rgb for example in examples]),
        context_cameras=Cameras(
            torch.stack([example.context_cameras.intrinsics for example in examples]),
            torch.stack([example.context_cameras.c2w for example in examples]),
            context_size,
        ),
        target_rgb=torch.stack([example.target_rgb for example in examples]),
        target_cameras=Cameras(
            torch.stack([example.target_cameras.intrinsics for example in examples]),
            torch.stack([example.target_cameras.c2w for example in examples]),
            target_size,
        ),
        bounds=torch.stack([example.bounds for example in examples]),
        target_depth=_stack_optional(examples, "target_depth"),
        target_normal=_stack_optional(examples, "target_normal"),
        target_point=_stack_optional(examples, "target_point"),
        target_visibility=_stack_optional(examples, "target_visibility"),
        target_support=_stack_optional(examples, "target_support"),
        scene_ids=tuple(example.scene_id for example in examples),
    )


def _stack_optional(examples: Sequence[SceneExample], name: str) -> Tensor | None:
    values = [getattr(example, name) for example in examples]
    if all(value is None for value in values):
        return None
    if any(value is None for value in values):
        raise ValueError(f"cannot mix missing and present values for {name}")
    return torch.stack(values)  # type: ignore[arg-type]
