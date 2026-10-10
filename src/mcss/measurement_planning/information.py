"""Information contract of MEASUREMENT_PLANNING_TASK_V1: what a deployable selector may see.

An episode is one record and one context role (A or B). A deployable selector receives a
MeasurementEnv built from a sanitized copy of the record that holds only that role's context frames
and the candidate frames, so query frames (RGB, depth, camera) and the other role's context frames
cannot be reached through it. Candidate depth is returned only by request_depth, within the budget,
and every read is logged with its purpose. The teacher (oracle) path is a separate object that is
never handed to a selector.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import torch

from mcss.dynamic.types import OnlineObservation
from mcss.mechanism_pilot.direct_capacity_contracts import _Media
from mcss.mechanism_pilot.rgbd_evidence_carrier import RGBDContext
from mcss.types import Cameras

ROLE_KEYS = {"A": "context_a", "B": "context_b"}
QUERY_KEYS = ("query", "primary_query", "secondary_query")
ROLE_FIELDS = (
    "context_a",
    "context_b",
    "query",
    "primary_query",
    "secondary_query",
    "stream",
    "far_query",
    "ignored",
)
CONTEXT_PURPOSE = "CONTEXT_ONLY_RGBD"
CANDIDATE_RGB_PURPOSE = "CANDIDATE_RGB_CAMERA"
CANDIDATE_DEPTH_PURPOSE = "CANDIDATE_DEPTH_REQUESTED"
TEACHER_PURPOSE = "ORACLE_TEACHER_ONLY"
QUERY_PURPOSE = "SEALED_QUERY_EVALUATION"
QUERY_CAMERA_PURPOSE = "QUERY_CAMERA_TEACHER_ONLY"


def query_ids(record):
    """Every frame any query role of the record names (primary, secondary and plain query lists)."""
    roles = record["roles"]
    return sorted(set().union(*(set(roles.get(k, [])) for k in QUERY_KEYS)))


def candidate_pool(record, min_query_gap=0):
    """Frames no role uses, in frame order; with min_query_gap > 0, frames whose index is closer
    than that to any query frame are left out. A candidate is never a context frame of either role
    nor a query."""
    roles = record["roles"]
    used = set().union(*(set(roles.get(k, [])) for k in ROLE_FIELDS))
    queries = query_ids(record)
    return [
        f["frame_id"]
        for f in record["frames"]
        if f["frame_id"] not in used
        and all(abs(f["frame_id"] - q) >= min_query_gap for q in queries)
    ]


def sanitized_record(record, role, pool):
    """The only record a selector's loader holds: the role's context and the candidate frames."""
    context = list(record["roles"][ROLE_KEYS[role]])
    forbidden = set(query_ids(record)) | set(record["roles"].get("far_query", []))
    if set(context) & set(pool) or set(pool) & forbidden or set(context) & forbidden:
        raise PermissionError("Candidates must be disjoint from context and query frames")
    keep = set(context) | set(pool)
    return {
        "scene_id": record["scene_id"],
        "frames": [deepcopy(f) for f in record["frames"] if f["frame_id"] in keep],
        "roles": {"context": context, "candidates": list(pool)},
    }


def single_camera(batch, j):
    return Cameras(
        batch.cameras.intrinsics[0, j], batch.cameras.c2w[0, j], batch.cameras.image_size
    )


def to_context(batch):
    """ObservationBatch with depth -> RGBDContext, exactly as the frozen context_of builds it."""
    observations = [
        OnlineObservation(batch.scene_id, fid, batch.rgb[0, j], single_camera(batch, j))
        for j, fid in enumerate(batch.frame_ids)
    ]
    return RGBDContext(observations, batch.depth[0, :, 0])


@dataclass(frozen=True)
class CandidateView:
    """What arrives with a candidate before any request: RGB [3, H, W] in [0, 1] and its camera."""

    frame_id: int
    rgb: torch.Tensor
    camera: Cameras


class BudgetExceededError(PermissionError):
    pass


class MeasurementEnv:
    """The deployable interface: measured context, candidate RGB and cameras, requested depth
    only."""

    def __init__(self, record, role, pool, budget, image_size, base, access_log):
        if not 0 <= budget <= len(pool):
            raise ValueError("Budget must lie within the candidate pool")
        self.scene_id, self.role = record["scene_id"], role
        self._record = sanitized_record(record, role, pool)
        self._media = _Media(self._record, image_size, base, "cpu", access_log)
        self.context_ids = tuple(self._record["roles"]["context"])
        self.candidate_ids = tuple(pool)
        self.budget = int(budget)
        self._requested = []
        self._context = None
        self._candidates = None

    def context(self):
        """The role's measured RGB-D context (fused before any decision)."""
        if self._context is None:
            batch = self._media.read(self.context_ids, depth_allowed=True, purpose=CONTEXT_PURPOSE)
            self._context = to_context(batch)
        return self._context

    def candidates(self):
        """Candidate RGB and cameras; no depth."""
        if self._candidates is None:
            batch = self._media.read(
                self.candidate_ids, depth_allowed=False, purpose=CANDIDATE_RGB_PURPOSE
            )
            self._candidates = tuple(
                CandidateView(fid, batch.rgb[0, j], single_camera(batch, j))
                for j, fid in enumerate(batch.frame_ids)
            )
        return self._candidates

    def request_depth(self, frame_id):
        """REQUEST_DEPTH: spends one unit of budget and returns that candidate as a one-frame
        RGBDContext."""
        if frame_id not in self.candidate_ids:
            raise PermissionError("Depth may be requested only for a candidate frame")
        if frame_id in self._requested:
            raise ValueError("Each candidate's depth is requested at most once")
        if len(self._requested) >= self.budget:
            raise BudgetExceededError("Depth budget exhausted")
        self._requested.append(frame_id)
        batch = self._media.read((frame_id,), depth_allowed=True, purpose=CANDIDATE_DEPTH_PURPOSE)
        return to_context(batch)

    @property
    def requested(self):
        return tuple(self._requested)

    @property
    def remaining(self):
        return self.budget - len(self._requested)


class TeacherAccess:
    """ORACLE / TEACHER ONLY: full access to one record for oracle utilities and scoring. Never a
    selector input."""

    def __init__(self, record, query_frames, image_size, base, access_log):
        self._record = deepcopy(record)
        self._media = _Media(self._record, image_size, base, "cpu", access_log)
        self.query_frames = tuple(query_frames)
        self.image_size = tuple(image_size)

    def frames(self, ids):
        """Measured RGB-D of context or candidate frames, for the oracle's exhaustive subset
        evaluation."""
        if set(ids) & set(self.query_frames) or set(ids) & set(query_ids(self._record)):
            raise PermissionError("Query frames are scored, never fused")
        return to_context(self._media.read(tuple(ids), depth_allowed=True, purpose=TEACHER_PURPOSE))

    def query_camera(self, frame_id):
        """A query camera for scoring (and the query-aware diagnostic only), as the frozen
        query_camera builds it."""
        if frame_id not in self.query_frames:
            raise PermissionError("Not a query frame of this episode")
        frame = self._media.frames[frame_id]
        self._media.access_log.append(
            {
                "scene_id": self._record["scene_id"],
                "frame_id": frame_id,
                "purpose": QUERY_CAMERA_PURPOSE,
                "channels": ["camera"],
            }
        )
        return Cameras(
            torch.tensor(frame["intrinsics"], dtype=torch.float32),
            torch.tensor(frame["c2w"], dtype=torch.float32),
            self.image_size,
        )

    def query_depth(self, frame_id):
        """Query ground truth as the arms evaluator scores it (np.load, squeeze, float64)."""
        if frame_id not in self.query_frames:
            raise PermissionError("Not a query frame of this episode")
        frame = self._media.frames[frame_id]
        self._media.access_log.append(
            {
                "scene_id": self._record["scene_id"],
                "frame_id": frame_id,
                "purpose": QUERY_PURPOSE,
                "channels": ["depth"],
            }
        )
        return np.load(self._media._path(frame, "depth")).squeeze().astype(np.float64)
