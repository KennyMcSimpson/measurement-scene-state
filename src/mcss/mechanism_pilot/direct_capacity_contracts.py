"""Direct-state capacity loader isolation; no carrier, encoder or writer dependency.

Guards cover these APIs, not arbitrary filesystem access. Context optimizers receive
only ContextOnlyLoader; evaluators and privileged oracle loaders are separate objects.
"""

from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from mcss.dynamic.types import clone_scene_state, hash_scene_state
from mcss.types import Cameras, SceneState

TRACKS = ("RGBD", "RGB_ONLY")
ORACLE_SCOPE = "QUERY_SUPERVISED_ORACLE"


@dataclass(frozen=True)
class ObservationBatch:
    scene_id: str
    frame_ids: tuple[int, ...]
    rgb: torch.Tensor
    cameras: Cameras
    depth: torch.Tensor | None
    supervision: str


def _scene_guard(record, capacity_scene_ids, holdout_scene_ids):
    allowed, forbidden = set(capacity_scene_ids), set(holdout_scene_ids)
    if not allowed or allowed & forbidden:
        raise PermissionError("Capacity allowlist must be nonempty and disjoint from holdout")
    if record["scene_id"] not in allowed or record["scene_id"] in forbidden:
        raise PermissionError("Final holdout and unknown scenes permanently forbidden")
    roles = record["roles"]
    queries = roles["primary_query"]
    a, b = roles["context_a"], roles["context_b"]
    if not a or len(a) != len(b) or a[0] != b[0] or set(a) & set(b) != {a[0]}:
        raise ValueError("Locked matched contexts with common anchor required")
    if not queries or set(queries) & (set(a) | set(b)):
        raise PermissionError("Query cannot enter context state optimization")
    all_ids = [f["frame_id"] for f in record["frames"]]
    if len(set(all_ids)) != len(all_ids) or not set(a + b + queries).issubset(all_ids):
        raise ValueError("Invalid or missing frame identities")
    if any(len(set(ids)) != len(ids) for ids in (a, b, queries)):
        raise ValueError("Duplicate context/query identities")


class _Media:
    def __init__(self, record, image_size, base, device, access_log):
        self.record = deepcopy(record)
        self.frames = {f["frame_id"]: f for f in self.record["frames"]}
        self.image_size, self.base, self.device = tuple(image_size), Path(base), device
        self.access_log = access_log if access_log is not None else []

    def _path(self, frame, kind):
        path = Path(frame[kind])
        return path if path.is_absolute() else self.base / path

    def read(self, ids, *, depth_allowed, purpose):
        rgbs, depths, intrinsics, poses = [], [], [], []
        for fid in ids:
            frame = self.frames[fid]
            self.access_log.append(
                {
                    "scene_id": self.record["scene_id"],
                    "frame_id": fid,
                    "purpose": purpose,
                    "channels": ["RGB", "camera"] + (["depth"] if depth_allowed else []),
                }
            )
            with Image.open(self._path(frame, "rgb")) as im:
                data = np.array(im.convert("RGB"), dtype=np.float32) / 255
            if data.shape != (*self.image_size, 3):
                raise ValueError("RGB dimensions mismatch")
            rgbs.append(torch.from_numpy(data).permute(2, 0, 1).to(self.device))
            intrinsics.append(
                torch.tensor(frame["intrinsics"], dtype=torch.float32, device=self.device)
            )
            poses.append(torch.tensor(frame["c2w"], dtype=torch.float32, device=self.device))
            if depth_allowed:
                data = np.load(self._path(frame, "depth")).squeeze()
                if data.shape != self.image_size:
                    raise ValueError("Depth dimensions mismatch")
                depth = torch.as_tensor(data, dtype=torch.float32, device=self.device)
                valid = torch.isfinite(depth) & (depth > 0)
                if not valid.any():
                    raise ValueError("No valid depth")
                depths.append(torch.where(valid, depth, 0)[None])
        return ObservationBatch(
            self.record["scene_id"],
            tuple(ids),
            torch.stack(rgbs)[None],
            Cameras(torch.stack(intrinsics)[None], torch.stack(poses)[None], self.image_size),
            torch.stack(depths)[None] if depth_allowed else None,
            purpose,
        )


class ContextOnlyLoader:
    """Optimizer-facing interface: context observations only; no query API can open media."""

    def __init__(
        self,
        record,
        image_size,
        base,
        device="cpu",
        *,
        track,
        capacity_scene_ids,
        holdout_scene_ids,
        access_log=None,
    ):
        _scene_guard(record, capacity_scene_ids, holdout_scene_ids)
        if track not in TRACKS:
            raise ValueError("Context-only loader requires RGBD or RGB_ONLY")
        context_ids = set(record["roles"]["context_a"] + record["roles"]["context_b"])
        private_record = deepcopy(record)
        private_record["frames"] = [
            f for f in private_record["frames"] if f["frame_id"] in context_ids
        ]
        if track == "RGB_ONLY":
            for frame in private_record["frames"]:
                frame.pop("depth", None)
        self.__media = _Media(private_record, image_size, base, device, access_log)
        self.__track = track
        self.__roles = {
            name: tuple(record["roles"][key])
            for name, key in [("A", "context_a"), ("B", "context_b")]
        }
        self.__roles["anchor"] = self.__roles["A"][:1]

    @property
    def track(self):
        return self.__track

    def context(self, role, *, track=None):
        if role not in self.__roles or (track is not None and track != self.__track):
            raise PermissionError("Context role/track cannot be changed or replaced by query")
        return self.__media.read(
            self.__roles[role],
            depth_allowed=self.__track == "RGBD",
            purpose=f"CONTEXT_ONLY_{self.__track}",
        )

    def context_depth(self, frame_id):
        if self.__track != "RGBD":
            raise PermissionError("RGB_ONLY optimizer cannot load context depth")
        if frame_id not in set(self.__roles["A"] + self.__roles["B"]):
            raise PermissionError("RGBD optimizer cannot load query depth")
        return self.__media.read((frame_id,), depth_allowed=True, purpose="CONTEXT_ONLY_RGBD").depth

    def query(self, *args, **kwargs):
        raise PermissionError("Query cameras/GT unavailable to context-only optimizer")

    query_camera = query
    query_depth = query
    query_rgb = query


class StateSealBarrier:
    """Every planned state must be sealed before any ordinary query camera/GT access."""

    def __init__(self, expected_state_ids):
        ids = tuple(expected_state_ids)
        if (
            not ids
            or len(set(ids)) != len(ids)
            or any(not isinstance(k, str) or not k for k in ids)
        ):
            raise ValueError("Nonempty unique planned state identities required")
        self.__expected = frozenset(ids)
        self.__states, self.__hashes, self.__scene_ids = {}, {}, {}
        self.events = []

    def seal(self, state_id, scene_id, state):
        if state_id not in self.__expected or state_id in self.__states:
            raise PermissionError("Unplanned or already sealed direct state")
        if not isinstance(state, SceneState):
            raise TypeError("Typed SceneState required")
        private = clone_scene_state(state)
        self.__states[state_id], self.__hashes[state_id] = private, hash_scene_state(private)
        self.__scene_ids[state_id] = scene_id
        self.events.append(
            {
                "event": "seal",
                "state_id": state_id,
                "scene_id": scene_id,
                "state_hash": self.__hashes[state_id],
            }
        )

    def assert_ready(self):
        if set(self.__states) != self.__expected:
            raise PermissionError("All planned direct states must be sealed before query access")
        if any(hash_scene_state(s) != self.__hashes[k] for k, s in self.__states.items()):
            raise PermissionError("Sealed direct state mutated")

    def state(self, state_id):
        self.assert_ready()
        return self.__states[state_id]

    def read(self, scene_id, query_id, loader):
        self.assert_ready()
        if scene_id not in self.__scene_ids.values():
            raise PermissionError("Query scene has no sealed state")
        self.events.append({"event": "query_camera_GT", "scene_id": scene_id, "query_id": query_id})
        value = loader()
        self.assert_ready()
        return value


class CapacityEvaluator:
    def __init__(
        self,
        record,
        image_size,
        base,
        device="cpu",
        *,
        barrier,
        capacity_scene_ids,
        holdout_scene_ids,
        access_log=None,
    ):
        _scene_guard(record, capacity_scene_ids, holdout_scene_ids)
        if not isinstance(barrier, StateSealBarrier):
            raise TypeError("StateSealBarrier required")
        self.__media = _Media(record, image_size, base, device, access_log)
        self.__barrier = barrier
        self.__queries = tuple(record["roles"]["primary_query"])
        self.__scene = record["scene_id"]

    def query(self, frame_id):
        if frame_id not in self.__queries:
            raise PermissionError("Evaluator query outside locked primary IDs")
        return self.__barrier.read(
            self.__scene,
            frame_id,
            lambda: self.__media.read(
                (frame_id,), depth_allowed=True, purpose="SEALED_QUERY_EVALUATION"
            ),
        )


class ExplicitQueryOracleLoader:
    """Privileged expressivity diagnostic; never supplied to context optimizer."""

    def __init__(
        self,
        record,
        image_size,
        base,
        device="cpu",
        *,
        scope,
        capacity_scene_ids,
        holdout_scene_ids,
        access_log=None,
    ):
        _scene_guard(record, capacity_scene_ids, holdout_scene_ids)
        if scope != ORACLE_SCOPE:
            raise PermissionError("Explicit QUERY_SUPERVISED_ORACLE designation required")
        self.__media = _Media(record, image_size, base, device, access_log)
        self.__queries = tuple(record["roles"]["primary_query"])
        self.scope = ORACLE_SCOPE

    def diagnostic_query_supervision(self):
        return self.__media.read(
            self.__queries,
            depth_allowed=True,
            purpose="DIAGNOSTIC_QUERY_SUPERVISED_ORACLE_NOT_CONTEXT_ONLY",
        )


@dataclass(frozen=True)
class FrozenOptimConfig:
    optimizer: str = "Adam"
    learning_rate: float = 0.03
    max_steps: int = 1000
    checkpoint_interval: int = 100
    selection: str = "best_context_objective"
    gradient_clip: float = 1.0
    seed: int = 20260927
    lambda_rgb: float = 1.0
    lambda_depth: float = 1.0
    lambda_smooth: float = 1e-4
    grid_size: int = 8
    renderer_samples: int = 64
    bounds_rule: str = "CURRENT_BOUNDS"
    scene_ids: tuple[str, ...] = ()
    query_ids: tuple[tuple[str, tuple[int, ...]], ...] = ()

    def __post_init__(self):
        if self.optimizer != "Adam" or self.selection != "best_context_objective":
            raise ValueError("Frozen Adam/context-only checkpoint selection required")
        if self.max_steps < 1 or self.checkpoint_interval < 1 or self.learning_rate <= 0:
            raise ValueError("Positive optimization budget and learning rate required")
        if self.grid_size not in (8, 12, 16, 32) or self.renderer_samples not in (64, 128):
            raise ValueError("Outside declared resolution/renderer diagnostic set")
        if not isinstance(self.scene_ids, tuple) or any(
            not isinstance(s, str) for s in self.scene_ids
        ):
            raise TypeError("Immutable scene IDs required")
        if not isinstance(self.query_ids, tuple) or any(
            not isinstance(p, tuple)
            or len(p) != 2
            or not isinstance(p[1], tuple)
            or any(type(i) is not int for i in p[1])
            for p in self.query_ids
        ):
            raise TypeError("Deeply immutable query IDs required")
        for value in (
            self.learning_rate,
            self.gradient_clip,
            self.lambda_rgb,
            self.lambda_depth,
            self.lambda_smooth,
        ):
            if not math.isfinite(value) or value < 0:
                raise ValueError("Finite nonnegative objective/optimizer constants required")


class ContextCheckpointSelector:
    """One objective over all fixed context views; no query score input exists."""

    def __init__(self, config, observed_ids):
        if not isinstance(config, FrozenOptimConfig):
            raise TypeError("FrozenOptimConfig required")
        self.config = config
        self.observed_ids = tuple(observed_ids)
        if not self.observed_ids or len(set(self.observed_ids)) != len(self.observed_ids):
            raise ValueError("Fixed nonempty context views required")
        self.best_objective, self.best_step, self.best_state = math.inf, None, None
        self.history = []

    def consider(self, step, *, context_objective, observed_ids, state):
        if tuple(observed_ids) != self.observed_ids:
            raise PermissionError("Checkpoint objective must cover all fixed context views")
        if (
            step < 0
            or step > self.config.max_steps
            or (step % self.config.checkpoint_interval != 0 and step != self.config.max_steps)
        ):
            raise PermissionError("Checkpoint outside frozen intervals")
        objective = float(context_objective)
        if not math.isfinite(objective):
            raise ValueError("Nonfinite context objective")
        if self.history and step <= self.history[-1]["step"]:
            raise ValueError("Checkpoint steps must advance")
        self.history.append({"step": step, "context_objective": objective})
        if objective < self.best_objective:
            self.best_objective, self.best_step = objective, step
            self.best_state = clone_scene_state(state)
            return True
        return False


def validate_resolution_sweep(configs, *, declared_grids=(8, 16, 32)):
    if tuple(c.grid_size for c in configs) != tuple(declared_grids):
        raise PermissionError("Resolution sweep differs from preregistered grids/order")
    baseline = asdict(configs[0])
    baseline.pop("grid_size")
    for config in configs:
        values = asdict(config)
        values.pop("grid_size")
        if values != baseline:
            raise PermissionError("Resolution sweep changes fields other than grid_size")
    return True


def validate_renderer_diagnostic(base, diagnostic):
    a, b = asdict(base), asdict(diagnostic)
    a.pop("renderer_samples")
    b.pop("renderer_samples")
    if a != b or (base.renderer_samples, diagnostic.renderer_samples) != (64, 128):
        raise PermissionError("Renderer diagnostic may change samples 64 to128 only")
    return True
