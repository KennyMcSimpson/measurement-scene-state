"""Fail-closed support-redesign roles, immutable methods and one-shot holdout access.

These guards enforce their loader APIs, not arbitrary filesystem reads. Oracle geometry
must remain a diagnostic and can never obtain a final-holdout session.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

from mcss.dynamic.types import SealedScene, hash_scene_state, hash_value
from mcss.evaluation.sealed_queries import _validate_sealed

DEPLOYABLE_METHODS = ("R0", "R1", "R2", "R3", "R12", "R123")
ORACLE_METHODS = ("ORACLE_VOLUME", "ORACLE_SUPPORT", "ORACLE_VOLUME_SUPPORT")
EXPOSED_SCENES = (
    "ai_004_001",
    "ai_006_001",
    "ai_007_002",
    "ai_008_001",
    "ai_009_001",
    "ai_010_001",
    "ai_011_001",
)
DEPLOYMENT_CHANNELS = (
    "arrived_rgb",
    "context_intrinsics",
    "context_poses",
    "frozen_image_features",
    "train_global_constants",
)
ORACLE_CHANNELS = DEPLOYMENT_CHANNELS + ("context_gt_depth", "query_gt_geometry")


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


@dataclass(frozen=True)
class SupportProtocolConfig:
    method: str
    scope: str = "DEPLOYABLE"
    input_channels: tuple[str, ...] = DEPLOYMENT_CHANNELS
    candidate_count: int = 128
    grid_size: tuple[int, int, int] = (8, 8, 8)
    renderer_samples: int = 64
    metric: str = "scene_macro_depth_absrel"
    hyperparameters: tuple[tuple[str, str | int | float | bool], ...] = ()

    def __post_init__(self):
        if self.method not in DEPLOYABLE_METHODS + ORACLE_METHODS:
            raise ValueError("Unknown support method")
        if self.scope not in ("DEPLOYABLE", "DIAGNOSTIC_ORACLE"):
            raise ValueError("Unknown method scope")
        if self.method in ORACLE_METHODS and self.scope != "DIAGNOSTIC_ORACLE":
            raise PermissionError("Oracle support cannot enter deployable configuration")
        if self.method in DEPLOYABLE_METHODS and self.scope != "DEPLOYABLE":
            raise ValueError("GT-free method must remain explicitly deployable")
        for name in ("input_channels", "grid_size", "hyperparameters"):
            if not isinstance(getattr(self, name), tuple):
                raise TypeError("Immutable tuple configuration required")
        allowed = ORACLE_CHANNELS if self.scope == "DIAGNOSTIC_ORACLE" else DEPLOYMENT_CHANNELS
        if set(self.input_channels) - set(allowed):
            raise PermissionError("Forbidden input channel for support method")
        if self.candidate_count != 128 or self.grid_size != (8, 8, 8):
            raise ValueError("Locked state and candidate budgets required")
        if self.renderer_samples != 64 or self.metric != "scene_macro_depth_absrel":
            raise ValueError("Locked renderer and primary metric required")
        for pair in self.hyperparameters:
            if not isinstance(pair, tuple) or len(pair) != 2 or not isinstance(pair[0], str):
                raise TypeError("Immutable named hyperparameter pairs required")
            if type(pair[1]) not in (str, int, float, bool):
                raise TypeError("Nested mutable hyperparameters prohibited")
        if len(dict(self.hyperparameters)) != len(self.hyperparameters):
            raise ValueError("Duplicate hyperparameter names")
        _digest(asdict(self))

    def assert_allowed_scene_role(self, role):
        if role == "FINAL_QUALIFICATION_HOLDOUT" and self.scope != "DEPLOYABLE":
            raise PermissionError("Oracle inference forbidden on final holdout")
        if role not in (
            "TRAIN",
            "REDESIGN_DEV",
            "REDESIGN_DEV_EXPOSED",
            "FINAL_QUALIFICATION_HOLDOUT",
        ):
            raise ValueError("Unknown scene role")


@dataclass(frozen=True)
class SceneRoleLock:
    train_scenes: tuple[str, ...]
    redesign_dev_scenes: tuple[str, ...]
    final_holdout_scenes: tuple[str, ...]
    exposed_scenes: tuple[str, ...] = EXPOSED_SCENES
    physical_identities: tuple[tuple[str, str], ...] = ()

    def __post_init__(self):
        groups = (
            self.train_scenes,
            self.redesign_dev_scenes,
            self.final_holdout_scenes,
            self.exposed_scenes,
        )
        if any(not isinstance(group, tuple) or not group for group in groups):
            raise TypeError("Nonempty immutable scene tuples required")
        if self.exposed_scenes != EXPOSED_SCENES:
            raise PermissionError("Old seven scenes must stay REDESIGN_DEV_EXPOSED")
        flat = tuple(s for group in groups for s in group)
        if len(set(flat)) != len(flat) or any(not isinstance(s, str) or not s for s in flat):
            raise PermissionError("Scene identity overlap or duplicate across locked roles")
        if not isinstance(self.physical_identities, tuple) or any(
            not isinstance(p, tuple) or len(p) != 2 for p in self.physical_identities
        ):
            raise TypeError("Immutable physical identity pairs required")
        ids = dict(self.physical_identities)
        if self.physical_identities:
            if set(ids) != set(flat) or len(ids) != len(self.physical_identities):
                raise PermissionError("Physical identity audit must cover each scene exactly once")
            if any(not isinstance(p, str) or not p for p in ids.values()):
                raise PermissionError("Unverified physical identity")
            other = {
                ids[s] for s in self.train_scenes + self.redesign_dev_scenes + self.exposed_scenes
            }
            held = [ids[s] for s in self.final_holdout_scenes]
            if set(held) & other or len(set(held)) != len(held):
                raise PermissionError("Physical environment leaks into final holdout")

    def assert_final_ready(self):
        if len(self.redesign_dev_scenes) < 8 or len(self.final_holdout_scenes) < 8:
            raise PermissionError("Final qualification requires >=8 dev and >=8 holdout scenes")
        if not self.physical_identities:
            raise PermissionError("Final qualification requires verified physical identity mapping")


@dataclass(frozen=True)
class MatchedTrainingConfig:
    support_method: str
    scene_ids: tuple[str, ...]
    seeds: tuple[int, ...]
    optimizer: str
    total_steps: int
    loss: str
    supervision: str
    parameter_count: int
    checkpoint_selection: str
    data_sha256: str
    optimizer_hyperparameters: tuple[tuple[str, float], ...] = ()

    def __post_init__(self):
        if self.support_method not in DEPLOYABLE_METHODS:
            raise PermissionError("Matched retraining requires deployable methods")
        if not isinstance(self.scene_ids, tuple) or not isinstance(self.seeds, tuple):
            raise TypeError("Immutable training scenes and seeds required")
        if not isinstance(self.optimizer_hyperparameters, tuple) or any(
            not isinstance(p, tuple) or len(p) != 2 or type(p[1]) not in (float, int)
            for p in self.optimizer_hyperparameters
        ):
            raise TypeError("Immutable optimizer hyperparameters required")
        if (
            not self.scene_ids
            or not self.seeds
            or self.total_steps <= 0
            or self.parameter_count <= 0
        ):
            raise ValueError("Nonempty matched training configuration required")


def assert_matched_training(current, redesigned):
    if not isinstance(current, MatchedTrainingConfig) or not isinstance(
        redesigned, MatchedTrainingConfig
    ):
        raise TypeError("MatchedTrainingConfig required")
    a, b = asdict(current), asdict(redesigned)
    a.pop("support_method")
    b.pop("support_method")
    differences = sorted(k for k in a if a[k] != b[k])
    if differences:
        raise PermissionError(f"Matched training differs beyond support_method: {differences}")
    return True


@dataclass(frozen=True)
class MethodLock:
    config: SupportProtocolConfig
    scene_roles: SceneRoleLock
    artifact_hashes: tuple[tuple[str, str, str], ...]
    dev_gate_path: str
    dev_gate_sha256: str
    digest: str


def seal_method_lock(config, scene_roles, artifact_paths, dev_gate_path):
    config.assert_allowed_scene_role("FINAL_QUALIFICATION_HOLDOUT")
    scene_roles.assert_final_ready()
    required = {"carrier_checkpoint", "support_config", "scene_role_lock", "inference_code"}
    if not required.issubset(artifact_paths):
        raise PermissionError("Incomplete method freeze artifacts")
    gate_path = str(Path(dev_gate_path).resolve())
    gate = json.loads(Path(gate_path).read_text())
    if gate.get("status") != "PASS":
        raise PermissionError("Development gate has not passed")
    artifacts = tuple(
        (k, str(Path(p).resolve()), _sha(p)) for k, p in sorted(artifact_paths.items())
    )
    if gate.get("method") != config.method or gate.get("config_digest") != _digest(asdict(config)):
        raise PermissionError("Development gate does not certify this frozen method configuration")
    if gate.get("artifact_hashes") != {name: digest for name, _, digest in artifacts}:
        raise PermissionError("Development gate artifacts differ from frozen method artifacts")
    if tuple(gate.get("dev_scene_ids", ())) != scene_roles.redesign_dev_scenes:
        raise PermissionError("Development gate scene identities differ from locked development")
    gate_hash = _sha(gate_path)
    digest = _digest(
        {
            "config": asdict(config),
            "roles": asdict(scene_roles),
            "artifacts": artifacts,
            "dev_gate": (gate_path, gate_hash),
        }
    )
    return MethodLock(config, scene_roles, artifacts, gate_path, gate_hash, digest)


class FinalHoldoutGate:
    """One atomic claim per experiment; interruptions do not silently reopen holdout."""

    def __init__(self, claim_path, method_lock=None):
        self.claim_path = Path(claim_path)
        self.method_lock = method_lock

    def _verify(self, config):
        lock = self.method_lock
        if not isinstance(lock, MethodLock):
            raise PermissionError("Final holdout cannot load before sealed method lock")
        if config != lock.config:
            raise PermissionError("Holdout inference cannot modify frozen configuration")
        config.assert_allowed_scene_role("FINAL_QUALIFICATION_HOLDOUT")
        rebuilt = seal_method_lock(
            config,
            lock.scene_roles,
            {name: path for name, path, _ in lock.artifact_hashes},
            lock.dev_gate_path,
        )
        if rebuilt != lock:
            raise PermissionError("Locked method artifact or development gate mutated")

    def claim(self, config, scene_ids):
        self._verify(config)
        lock = self.method_lock
        if tuple(scene_ids) != lock.scene_roles.final_holdout_scenes:
            raise PermissionError("Holdout scene identity allowlist changed")
        self.claim_path.parent.mkdir(parents=True, exist_ok=True)
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        try:
            descriptor = os.open(self.claim_path, flags, 0o600)
        except FileExistsError as error:
            raise PermissionError("Final holdout has already been claimed; no reopening") from error
        with os.fdopen(descriptor, "w") as handle:
            json.dump(
                {
                    "method_lock_digest": lock.digest,
                    "scene_ids": list(scene_ids),
                    "status": "CLAIMED_ONCE",
                },
                handle,
                sort_keys=True,
            )
            handle.flush()
            os.fsync(handle.fileno())
        return _HoldoutSession(self, config, _sha(self.claim_path))


class _HoldoutSession:
    def __init__(self, gate, config, claim_hash):
        self.gate, self.config, self.claim_hash = gate, config, claim_hash
        self.events = []
        self._model_scenes = set()
        self._queries = set()

    def _check(self, scene_id):
        self.gate._verify(self.config)
        if _sha(self.gate.claim_path) != self.claim_hash:
            raise PermissionError("Atomic holdout claim mutated")
        if scene_id not in self.gate.method_lock.scene_roles.final_holdout_scenes:
            raise PermissionError("Scene not in final holdout allowlist")

    def load_model_scene(self, scene_id, loader: Callable):
        self._check(scene_id)
        if scene_id in self._model_scenes:
            raise PermissionError("Holdout model scene cannot be opened twice")
        self._model_scenes.add(scene_id)
        self.events.append({"kind": "model_scene", "scene_id": scene_id})
        value = loader()
        self._check(scene_id)
        return value

    def read_query_ground_truth(self, scene_id, query_id, sealed_states, loader: Callable):
        self._check(scene_id)
        if scene_id not in self._model_scenes or not sealed_states:
            raise PermissionError("Query GT requires constructed and sealed states")
        fingerprints = []
        for state in sealed_states:
            if not isinstance(state, SealedScene) or state.scene_id != scene_id:
                raise PermissionError("Expected sealed states for target holdout scene")
            _validate_sealed(state)
            if hash_scene_state(state.scene_state) != state.state_hash:
                raise PermissionError("Sealed holdout state hash mismatch")
            fingerprints.append(hash_value(state))
        key = (scene_id, query_id)
        if key in self._queries:
            raise PermissionError("Holdout query GT cannot be reopened")
        self._queries.add(key)
        self.events.append({"kind": "query_GT", "scene_id": scene_id, "query_id": query_id})
        value = loader()
        self._check(scene_id)
        if fingerprints != [hash_value(s) for s in sealed_states]:
            raise PermissionError("Evaluator mutated sealed holdout states")
        return value


def old_exposed_manifest():
    return {
        "scene_ids": list(EXPOSED_SCENES),
        "role": "REDESIGN_DEV_EXPOSED",
        "final_qualification_allowed": False,
        "independent_confirmation_allowed": False,
    }


def validate_dev_scene_ids(roles, scene_ids):
    """Run before media loading; diagnostic query geometry exception never includes holdout."""
    if isinstance(roles, SceneRoleLock):
        train = set(roles.train_scenes)
        dev = set(roles.redesign_dev_scenes)
        holdout = set(roles.final_holdout_scenes)
    else:
        train = set(roles["TRAIN_SCENES"])
        dev = set(roles["REDESIGN_DEV_SCENES"])
        holdout = set(roles["FINAL_HOLDOUT_SCENES"])
    exposed = set(EXPOSED_SCENES)
    if holdout & (train | dev | exposed) or train & (dev | exposed) or dev & exposed:
        raise PermissionError("Scene roles overlap")
    ids = tuple(scene_ids)
    if not ids or len(ids) != len(set(ids)) or set(ids) - (dev | exposed):
        raise PermissionError(
            "Diagnostic/dev inference rejects holdout, training and unknown identities"
        )
    return True
