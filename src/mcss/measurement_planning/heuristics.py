"""Deployable selectors and features of MEASUREMENT_PLANNING_TASK_V1 (no candidate depth before a
request, no query RGB, depth or camera).

Every selector receives only a MeasurementEnv and the budget k and acquires exactly k candidate
depths through env.request_depth. Its scores use the role's measured context, the candidates' RGB
and cameras, and (for the greedy selectors) the depths it has already requested. Ties break toward
the earlier frame.
"""

from __future__ import annotations

import zlib

import numpy as np
import torch
from scipy import ndimage
from scipy.spatial import cKDTree

from mcss.geometry import generate_rays, project_world
from mcss.measurement_planning import geometry as g
from mcss.measurement_planning.information import MeasurementEnv

FILL_CAP_PX = 32.0
POSE_METRES_PER_RADIAN = 1.0
NOVELTY_RELATIVE = 0.02
RANDOM_SEED = 20261006
ROLE_CODE = {"A": 0, "B": 1}


# ------------------------------------------------ measured geometry seen by a selector
def camera_center(camera):
    return camera.c2w.to(torch.float64)[:3, 3].numpy()


def camera_rotation(camera):
    return camera.c2w.to(torch.float64)[:3, :3].numpy()


def rotation_angle(first, second):
    cos = (np.trace(first.T @ second) - 1.0) / 2.0
    return float(np.arccos(np.clip(cos, -1.0, 1.0)))


class MeasuredGeometry:
    """World points (with their RGB) of every frame measured so far: the context, then requested
    candidates."""

    def __init__(self, context):
        self.points, self.colors, self.cameras = {}, {}, {}
        self._tree = None
        self.add(context)

    def add(self, context):
        points = g.frame_points(context)
        for j, observation in enumerate(context):
            depth = torch.as_tensor(context.depths[j], dtype=torch.float64)
            valid = (torch.isfinite(depth) & (depth > 0)).numpy()
            rgb = observation.rgb.permute(1, 2, 0).numpy().astype(np.float64)
            self.points[observation.frame_id] = points[observation.frame_id]
            self.colors[observation.frame_id] = rgb[valid]
            self.cameras[observation.frame_id] = observation.camera
        self._tree = None

    def all_points(self):
        return torch.cat([self.points[f] for f in sorted(self.points)])

    def tree(self):
        if self._tree is None:
            self._tree = cKDTree(self.all_points().numpy())
        return self._tree

    def depth_at(self, camera):
        """Frozen z-buffer of all measured points at a camera (inf where none)."""
        return g.union([g.reproject(self.points[f], camera) for f in sorted(self.points)])

    def color_at(self, camera):
        """RGB of the nearest measured point per pixel (the frozen z-buffer's pixel rule), NaN where
        none."""
        camera64 = camera.to(dtype=torch.float64)
        height, width = camera64.image_size
        points = self.all_points()
        colors = np.concatenate([self.colors[f] for f in sorted(self.points)])
        pixels, _, inside = project_world(points, camera64)
        u = pixels[:, 0].round().long().clamp(0, width - 1)
        v = pixels[:, 1].round().long().clamp(0, height - 1)
        distance = (points - camera64.c2w[:3, 3]).norm(dim=-1)
        inside = inside.numpy()
        index = (v * width + u).numpy()[inside]
        order = np.lexsort((distance.numpy()[inside], index))
        index, first = np.unique(index[order], return_index=True)
        image = np.full((height * width, 3), np.nan)
        image[index] = colors[inside][order][first]
        return image.reshape(height, width, 3)


# ------------------------------------------------ deployable features of one candidate
def pose_features(camera, measured_cameras):
    center, rotation = camera_center(camera), camera_rotation(camera)
    translations = [float(np.linalg.norm(center - camera_center(m))) for m in measured_cameras]
    angles = [rotation_angle(rotation, camera_rotation(m)) for m in measured_cameras]
    combined = [t + POSE_METRES_PER_RADIAN * a for t, a in zip(translations, angles, strict=True)]
    return {
        "translation_m": min(translations),
        "rotation_deg": float(np.degrees(min(angles))),
        "pose_distance": min(combined),
    }


def coverage_features(measured, candidate):
    depth = measured.depth_at(candidate.camera)
    hit = np.isfinite(depth)
    distance = (
        ndimage.distance_transform_edt(~hit) if hit.any() else np.full(hit.shape, FILL_CAP_PX)
    )
    return depth, {
        "uncovered": float(1.0 - g.near_mask(hit).mean()),
        "fill_distance_px": float(np.minimum(distance, FILL_CAP_PX).mean()),
    }


def photometric_residual(measured, candidate):
    """Mean |RGB difference| to the measured colour where measured geometry lands, 1 where nothing
    does."""
    rgb = candidate.rgb.permute(1, 2, 0).numpy().astype(np.float64)
    reprojected = measured.color_at(candidate.camera)
    hit = np.isfinite(reprojected[..., 0])
    difference = np.where(hit, np.abs(np.nan_to_num(reprojected) - rgb).mean(axis=-1), 1.0)
    return float(difference.mean())


def novelty3d(measured, candidate, depth, constant):
    """Share of the candidate's predicted surface (frozen harmonic fill of the measured z-buffer)
    that lies farther than 2% of its depth from every measured point."""
    camera64 = candidate.camera.to(dtype=torch.float64)
    prediction, _ = g.harmonic(depth, g.ray_cosine(candidate.camera), constant)
    origins, rays = generate_rays(camera64)
    points = (origins + rays * torch.as_tensor(prediction)[..., None]).reshape(-1, 3).numpy()
    distance, _ = measured.tree().query(points, k=1)
    return float(np.mean(distance > NOVELTY_RELATIVE * prediction.reshape(-1)))


def candidate_features(measured, candidates, constant, pool):
    """Every deployable feature of every candidate given the measured frames (no candidate depth)"""
    rows = {}
    cameras = list(measured.cameras.values())
    for candidate in candidates:
        depth, coverage = coverage_features(measured, candidate)
        rows[candidate.frame_id] = {
            **pose_features(candidate.camera, cameras),
            **coverage,
            "photometric": photometric_residual(measured, candidate),
            "novelty3d": novelty3d(measured, candidate, depth, constant),
            "pool_index": pool.index(candidate.frame_id),
            "recency_rank": len(pool) - 1 - pool.index(candidate.frame_id),
        }
    return rows


# ---------------------------------------------------------------- selectors
def _require(env, k):
    if not isinstance(env, MeasurementEnv):
        raise TypeError("Selectors take a MeasurementEnv, never a teacher or a record")
    if k != env.budget:
        raise ValueError("A selector spends exactly the episode budget")


def _acquire(env, ids):
    for fid in ids:
        env.request_depth(fid)


def _top(scores, k):
    """k frame ids with the highest score; ties toward the earlier frame."""
    return [fid for fid, _ in sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:k]]


def select_uniform(env, k, **_):
    """Centred strata over the candidate order: index floor((i + 1/2) n / k)."""
    _require(env, k)
    pool = list(env.candidate_ids)
    _acquire(env, [pool[int(np.floor((i + 0.5) * len(pool) / k))] for i in range(k)])


def select_temporal_even(env, k, **_):
    """Candidates nearest to k evenly spaced frame indices over the candidates' span (centred)."""
    _require(env, k)
    pool = sorted(env.candidate_ids)
    low, high = pool[0], pool[-1]
    chosen = []
    for i in range(k):
        target = low + (i + 0.5) * (high - low) / k
        free = [f for f in pool if f not in chosen]
        chosen.append(min(free, key=lambda f: (abs(f - target), f)))
    _acquire(env, chosen)


def select_latest(env, k, **_):
    """The k most recent candidates."""
    _require(env, k)
    _acquire(env, sorted(env.candidate_ids)[len(env.candidate_ids) - k :])


def select_random(env, k, **_):
    """A frozen-seed draw per episode and budget."""
    _require(env, k)
    seed = [RANDOM_SEED, zlib.crc32(env.scene_id.encode()), ROLE_CODE[env.role], k]
    rng = np.random.default_rng(seed)
    _acquire(env, [int(f) for f in rng.choice(np.array(env.candidate_ids), size=k, replace=False)])


def select_pose_farthest(env, k, **_):
    """Greedy farthest camera from the measured or chosen cameras (metres + 1 m per radian)."""
    _require(env, k)
    measured = [o.camera for o in env.context()]
    candidates = {c.frame_id: c for c in env.candidates()}
    chosen = []
    for _ in range(k):
        scores = {
            fid: pose_features(c.camera, measured)["pose_distance"]
            for fid, c in candidates.items()
            if fid not in chosen
        }
        best = _top(scores, 1)[0]
        chosen.append(best)
        measured.append(candidates[best].camera)
    _acquire(env, chosen)


def _ranked(env, k, score):
    _require(env, k)
    measured = MeasuredGeometry(env.context())
    scores = {c.frame_id: score(measured, c) for c in env.candidates()}
    _acquire(env, _top(scores, k))


def _greedy(env, k, score):
    _require(env, k)
    measured = MeasuredGeometry(env.context())
    candidates = {c.frame_id: c for c in env.candidates()}
    for _ in range(k):
        scores = {
            fid: score(measured, c) for fid, c in candidates.items() if fid not in env.requested
        }
        measured.add(env.request_depth(_top(scores, 1)[0]))


def select_max_uncovered(env, k, **_):
    """Top-k share of candidate pixels farther than 8 px from the reprojected measured context."""
    _ranked(env, k, lambda m, c: coverage_features(m, c)[1]["uncovered"])


def select_max_novelty3d(env, k, constant, **_):
    """Top-k share of the predicted candidate surface away from every measured point."""
    _ranked(env, k, lambda m, c: novelty3d(m, c, m.depth_at(c.camera), constant))


def select_greedy_coverage(env, k, **_):
    """Greedy uncovered share, re-scored after every acquired depth."""
    _greedy(env, k, lambda m, c: coverage_features(m, c)[1]["uncovered"])


def select_greedy_fill_distance(env, k, **_):
    """Greedy mean distance to the nearest measured hit (capped at 32 px): an uncertainty proxy."""
    _greedy(env, k, lambda m, c: coverage_features(m, c)[1]["fill_distance_px"])


def select_greedy_photometric(env, k, **_):
    """Greedy photometric residual of the measured RGB-D reprojected into the candidate."""
    _greedy(env, k, photometric_residual)


DEPLOYABLE = {
    "UNIFORM": select_uniform,
    "TEMPORAL_EVEN": select_temporal_even,
    "LATEST": select_latest,
    "RANDOM_SEEDED": select_random,
    "POSE_FARTHEST": select_pose_farthest,
    "MAX_UNCOVERED": select_max_uncovered,
    "MAX_NOVELTY3D": select_max_novelty3d,
    "GREEDY_COVERAGE": select_greedy_coverage,
    "GREEDY_FILL_DISTANCE": select_greedy_fill_distance,
    "GREEDY_PHOTOMETRIC": select_greedy_photometric,
}
