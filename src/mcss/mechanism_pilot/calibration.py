"""Exact Hypersim pixel-center rays from published per-scene camera metadata.

Uses official M_cam_from_uv and OpenGL-to-OpenCV conversion, not a FOV guess.
The output corresponds to half-pixel resampling at the requested resolution.
"""

import csv
from pathlib import Path

import numpy as np


def calibrated_intrinsics(row, image_size):
    height, width = image_size
    if min(height, width) < 2:
        raise ValueError("Image dimensions must be at least two")
    matrix = np.array([[float(row[f"M_cam_from_uv_{i}{j}"]) for j in range(3)] for i in range(3)])
    pixel_to_uv = np.array(
        [[2 / width, 0, 1 / width - 1], [0, -2 / height, 1 - 1 / height], [0, 0, 1]]
    )
    ray_matrix = np.diag([1, -1, -1]) @ matrix @ pixel_to_uv
    if not np.isfinite(ray_matrix).all() or abs(np.linalg.det(ray_matrix)) < 1e-15:
        raise ValueError("Invalid published ray matrix")
    intrinsics = np.linalg.inv(ray_matrix)
    if abs(intrinsics[2, 2]) < 1e-12:
        raise ValueError("Cannot normalize published camera")
    intrinsics /= intrinsics[2, 2]
    if intrinsics[0, 0] <= 0 or intrinsics[1, 1] <= 0:
        raise ValueError("Published camera has invalid focal signs")
    for x, y in [(0, 0), (width - 1, 0), (0, height - 1), (width - 1, height - 1)]:
        if (ray_matrix @ [x, y, 1])[2] <= 0:
            raise ValueError("Published rays are not front-facing")
    return intrinsics


def published_scene_metadata(path: str | Path):
    with Path(path).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    result = {}
    for row in rows:
        scene = row["scene_name"]
        if scene in result:
            raise ValueError("Duplicate calibration scene")
        scale = float(row["settings_units_info_meters_scale"])
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("Missing or invalid metric scale; no default allowed")
        result[scene] = row
    return result
