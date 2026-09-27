"""Fail-closed pilot boundaries; these interfaces do not establish carrier qualification.

The evaluator ledger controls its loader API, not arbitrary filesystem access. Callers must
keep query providers exclusively in the evaluator. No query provider belongs in the runtime.
"""

from collections.abc import Mapping
from copy import deepcopy

import torch
from torch import nn

from mcss.dynamic.types import SealedScene, clone_scene_state, hash_scene_state, hash_value
from mcss.evaluation.sealed_queries import _validate_sealed
from mcss.geometry import generate_rays


def validate_scene_splits(records):
    """Validate audited physical identities, including repeated scans of the same site."""
    assignments, scans = {}, set()
    counts = {name: set() for name in ("train", "dev", "test")}
    for row in records:
        scene, physical, split = (row.get(k) for k in ("scene_id", "physical_scene_id", "split"))
        if not scene or not physical or row.get("physical_scene_identity_status") != "VERIFIED":
            raise ValueError("Verified physical scene identity is required")
        if split not in counts or scene in scans:
            raise ValueError("Invalid split or duplicate scan identity")
        if physical in assignments and assignments[physical] != split:
            raise ValueError("Physical scene leakage across splits")
        scans.add(scene)
        assignments[physical] = split
        counts[split].add(physical)
    return {split: len(ids) for split, ids in counts.items()}


class EvaluationLedger:
    """Require the complete predetermined candidate set before any query loader runs."""

    def __init__(self, expected_candidates):
        ids = tuple(expected_candidates)
        if (
            not ids
            or len(set(ids)) != len(ids)
            or any(not isinstance(x, str) or not x for x in ids)
        ):
            raise ValueError("Expected candidate IDs must be nonempty and unique")
        self.expected_candidates = frozenset(ids)
        self._sealed = {}
        self._fingerprints = {}
        self.events = []

    def seal(self, candidate_id, sealed):
        if candidate_id not in self.expected_candidates or candidate_id in self._sealed:
            raise ValueError("Unexpected or already sealed candidate")
        if not isinstance(sealed, SealedScene):
            raise TypeError("SealedScene required")
        _validate_sealed(sealed)
        if hash_scene_state(sealed.scene_state) != sealed.state_hash:
            raise ValueError("Candidate state hash mismatch")
        self._sealed[candidate_id] = sealed
        self._fingerprints[candidate_id] = hash_value(sealed)
        self.events.append(
            {"event": "seal", "candidate_id": candidate_id, "state_hash": sealed.state_hash}
        )

    def assert_ready(self):
        if set(self._sealed) != self.expected_candidates:
            raise PermissionError(
                "All expected candidate states must be sealed before query access"
            )
        for candidate, sealed in self._sealed.items():
            if hash_value(sealed) != self._fingerprints[candidate]:
                raise PermissionError("Sealed candidate or provenance mutated")

    def _read(self, query_id, loader, kind):
        self.assert_ready()
        if not isinstance(query_id, str) or not query_id:
            raise ValueError("Query identity required")
        self.events.append({"event": kind, "query_id": query_id, "status": "requested"})
        value = loader()
        self.assert_ready()
        self.events[-1]["status"] = "completed"
        return value

    def read_query_camera(self, query_id, loader):
        return self._read(query_id, loader, "query_camera")

    def read_query_ground_truth(self, query_id, loader):
        return self._read(query_id, loader, "query_ground_truth")


POLICY_KEYS = frozenset(
    {
        "arrived_rgb",
        "arrived_camera",
        "rgb_residual",
        "opacity",
        "coverage",
        "state_summary",
        "fast_summary",
        "history_summary",
    }
)
_FORBIDDEN = ("query", "depth", "future", "reward", "oracle", "ground_truth", "target", "label")


def validate_policy_inputs(inputs):
    """Reject unknown top-level channels and forbidden nested fields; never silently drop them.

    This is a structural boundary. The caller must bind arrived inputs to declared frame IDs;
    semantic relabeling of forbidden tensors cannot be detected from tensor values.
    """
    if not isinstance(inputs, Mapping) or set(inputs) - POLICY_KEYS:
        raise ValueError("Policy accepts only declared arrived-observation and summary channels")

    def inspect(value):
        if isinstance(value, Mapping):
            for key, item in value.items():
                if not isinstance(key, str) or any(word in key.lower() for word in _FORBIDDEN):
                    raise ValueError("Forbidden policy information")
                inspect(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                inspect(item)
        elif not isinstance(value, (torch.Tensor, int, float, str, bool, type(None))):
            # Cameras are explicitly admitted only for arrived_camera below.
            raise TypeError("Opaque policy payload is not permitted")

    from mcss.types import Cameras

    for key, value in inputs.items():
        if key == "arrived_camera" and type(value) is Cameras:
            continue
        inspect(value)
    return inputs


class ResidualReadoutControl(nn.Module):
    """Query-conditioned RGB/depth residual; detached private state, ordinary optimizer API.

    Train with model.train(), optimizer.zero_grad(), supervised_loss(model(state,cameras),
    train_labels).backward(), optimizer.step(). Only train-split labels are admissible.
    eval() does NOT imply trained or qualified. Training/evaluation provenance is external.
    Existing FreeDecoderControl replaces outputs rather than adding residuals, so is not reused.
    """

    def __init__(self, feature_dim, renderer):
        super().__init__()
        if list(renderer.parameters()):
            raise ValueError("Residual baseline requires parameter-free fixed renderer")
        self.renderer = renderer
        self.head = nn.Sequential(nn.Conv2d(feature_dim + 6, 16, 1), nn.SiLU(), nn.Conv2d(16, 4, 1))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)
        self.training_status = "UNTRAINED_UNEVALUATED"

    def forward(self, state, cameras):
        state_before, camera_before = hash_scene_state(state), hash_value(cameras)
        private_state, private_camera = clone_scene_state(state), deepcopy(cameras)
        with torch.no_grad():
            base = self.renderer(private_state, private_camera, {"rgb", "depth", "visibility"})
            features = self.renderer.render_features(private_state, private_camera)
            origins, directions = generate_rays(private_camera)
            rays = torch.cat((origins, directions), dim=-1).permute(0, 1, 4, 2, 3)
            combined = torch.cat((features, rays), dim=2)
        batch, views, channels, height, width = combined.shape
        residual = self.head(combined.reshape(batch * views, channels, height, width))
        residual = residual.reshape(batch, views, 4, height, width)
        result = dict(base)
        result["rgb"] = (base["rgb"] + residual[:, :, :3]).clamp(0, 1)
        result["depth"] = (base["depth"] + residual[:, :, 3:4]).clamp_min(0)
        if hash_scene_state(state) != state_before or hash_value(cameras) != camera_before:
            raise RuntimeError("Readout modified shared inputs")
        return result


def cross_scene_readout(
    renderer, replacement_state, query_cameras, source_scene_id, target_scene_id
):
    """Return renderer output dict from replacement state and numerically unchanged cameras.

    No registration/recentering is fitted. Input states must already use the preregistered
    coordinate convention. Independent identity verification belongs in the split audit.
    """
    if not source_scene_id or not target_scene_id or source_scene_id == target_scene_id:
        raise ValueError("Replacement requires different scenes")
    state_before, camera_before = hash_scene_state(replacement_state), hash_value(query_cameras)
    state, cameras = clone_scene_state(replacement_state), deepcopy(query_cameras)
    output = renderer(state, cameras, {"rgb", "depth", "visibility"})
    if (
        hash_value(cameras) != camera_before
        or hash_value(query_cameras) != camera_before
        or hash_scene_state(replacement_state) != state_before
    ):
        raise RuntimeError("Cross-scene readout changed query cameras or shared state")
    return output
