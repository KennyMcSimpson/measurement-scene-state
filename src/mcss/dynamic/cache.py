"""Append-only observed evidence; independent from scene fields and fast weights."""

from dataclasses import dataclass

import torch
from torch import Tensor

from mcss.dynamic.types import OnlineObservation
from mcss.types import Cameras


@dataclass(frozen=True)
class CachedObservation:
    observation: OnlineObservation
    features: Tensor


@dataclass(frozen=True)
class ObservationCache:
    episode_id: str
    scene_id: str
    entries: tuple[CachedObservation, ...] = ()

    @property
    def revision(self):
        return len(self.entries)

    def append(self, observation: OnlineObservation, features: Tensor):
        if observation.scene_id != self.scene_id:
            raise ValueError("Cannot mix scenes in an observation cache")
        if self.entries and observation.frame_id <= self.entries[-1].observation.frame_id:
            raise ValueError("Observations must arrive in strictly increasing frame order")
        if features.ndim != 3 or not torch.isfinite(features).all():
            raise ValueError("Features must be finite [F,H,W]")
        if features.device != observation.rgb.device:
            raise ValueError("Features and observation must share device")
        camera = observation.camera
        copied = OnlineObservation(
            observation.scene_id,
            observation.frame_id,
            observation.rgb.clone(),
            Cameras(camera.intrinsics.clone(), camera.c2w.clone(), camera.image_size),
        )
        entry = CachedObservation(copied, features.clone())
        return ObservationCache(self.episode_id, self.scene_id, (*self.entries, entry))
