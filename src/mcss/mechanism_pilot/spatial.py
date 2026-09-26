"""Explicit fixed geometry modes and observed-camera-only support diagnostics."""

from dataclasses import replace

import torch

from mcss.dynamic.config import CarrierConfig
from mcss.geometry import project_world, transform_cameras

SPATIAL_MODES = ("legacy-forward", "anchor-centered")


def carrier_config_for_spatial_mode(mode: str) -> CarrierConfig:
    if mode == "legacy-forward":
        return CarrierConfig()
    if mode == "anchor-centered":
        return replace(CarrierConfig(), local_bounds_m=((-6.0, -4.0, -6.0), (6.0, 4.0, 6.0)))
    raise ValueError(f"Unknown spatial mode: {mode}")


def observed_camera_support(candidate_points, observed_cameras):
    """Accept ONLY arrived cameras in temporal order; never query poses or labels.

    Support means two camera frusta contain a candidate, not surface visibility.
    All camera poses and candidate points are expressed in the first camera frame.
    """
    if not observed_cameras:
        raise ValueError("At least one observed camera is required")
    anchor_inverse = torch.linalg.inv(observed_cameras[0].c2w)
    masks = []
    prefixes = []
    for camera in observed_cameras:
        anchored = transform_cameras(camera, anchor_inverse)
        pixels, depth, inside = project_world(candidate_points, anchored)
        valid = inside & torch.isfinite(pixels).all(-1) & torch.isfinite(depth)
        masks.append(valid)
        counts = torch.stack(masks).sum(0)
        prefixes.append(int((counts >= 2).sum()))
    return {
        "visible_candidates_per_observation": [int(mask.sum()) for mask in masks],
        "supported_candidates_per_prefix": prefixes,
        "supported_candidates": prefixes[-1],
    }
