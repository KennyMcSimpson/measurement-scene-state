"""Locked static carrier experiment boundaries, dataset and checkpoint IO."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import torch

from mcss.dynamic.carrier import DynamicSceneCarrier
from mcss.dynamic.config import CarrierConfig
from mcss.dynamic.types import OnlineObservation, hash_value
from mcss.mechanism_pilot.direct_capacity_contracts import ContextOnlyLoader, _Media
from mcss.mechanism_pilot.optimization_bounds_contracts import (
    FrozenTrainingPrior,
    context_camera_bundle,
    frozen_gt_free_bounds,
)
from mcss.mechanism_pilot.small_training import sha
from mcss.types import Cameras

EXPERIMENT = "EXP-3D-GEOMETRY-AWARE-CARRIER-V2"
VARIANTS = ("C0", "C1")


def read(path):
    return json.loads(Path(path).read_text())


def validate_split(manifest):
    scenes = manifest["scenes"]
    ids = [r["scene_id"] for r in scenes]
    physical = [r["physical_scene_id"] for r in scenes]
    if len(ids) != len(set(ids)) or len(physical) != len(set(physical)):
        raise PermissionError("Duplicate physical scene / scene identity")
    protected = set(manifest.get("protected_scene_ids", []))
    if set(ids) & protected:
        raise PermissionError("Protected scene in train/dev")
    for r in scenes:
        if r["split"] not in ("TRAIN", "DEV", "FRESH_QUALIFICATION"):
            raise PermissionError("Unrecognized partition")
        if r["split"] == "FRESH_QUALIFICATION" and (
            r.get("historically_exposed", True) or not r.get("independence_verified", False)
        ):
            raise PermissionError("Exposed/unverified scene cannot be fresh")
        roles = r["roles"]
        a, b, q = (roles[k] for k in ("context_a", "context_b", "primary_query"))
        if len(a) != len(b) or len(a) < 2 or a[0] != b[0]:
            raise PermissionError("Matched common-anchor contexts required")
        if set(a) & set(b) != {a[0]} or set(q) & set(a + b) or len(q) < 2:
            raise PermissionError("Disjoint multiple queries required")


def verify_lock(root):
    root = Path(root)
    lock = read(root / "preregistration.json")
    if lock["status"] != "FROZEN_BEFORE_FORMAL_TRAINING":
        raise PermissionError("Formal preregistration absent")
    for section in ("input_sha256", "source_sha256"):
        for name, digest in lock[section].items():
            if sha(Path(name)) != digest:
                raise PermissionError(f"Frozen input/source changed: {name}")
    manifest = read(root / "scene_split.json")
    validate_split(manifest)
    return manifest, read(root / "training_contract.json")


def require_fresh_lock(root):
    root = Path(root)
    manifest, _ = verify_lock(root)
    if manifest.get("fresh_status") != "VERIFIED":
        raise PermissionError("BLOCKED_INDEPENDENCE_UNRESOLVED")
    lock_path = root / "fresh_qualification_lock.json"
    if not lock_path.exists():
        raise PermissionError("Frozen fresh qualification lock required")
    lock = read(lock_path)
    gate = read(root / "qualification_results.json")
    if gate["SURFACE_TRAINING_STATUS"] != "SUPPORTED" or gate["STATIC_DEV_STATUS"] != "SUPPORTED":
        raise PermissionError("Fresh qualification DEV gates not passed")
    if lock.get("primary_method") != "C1" or not lock.get("all_checkpoints_frozen"):
        raise PermissionError("Checkpoint/method freeze absent")
    for path, digest in lock["input_sha256"].items():
        if sha(Path(path)) != digest:
            raise PermissionError("Fresh lock changed")
    if (root / "fresh_qualification_opened.json").exists():
        raise PermissionError("Fresh qualification can run only once")
    return lock


def make_carrier(seed, device):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return DynamicSceneCarrier(CarrierConfig(grid_size=(16, 16, 16))).to(device)


def save_checkpoint(path, carrier, *, variant, seed, step, lock_sha256):
    payload = {
        "schema": "mcss.geometry_carrier.v2",
        "config": asdict(carrier.config),
        "carrier": {k: v.detach().cpu() for k, v in carrier.state_dict().items()},
        "variant": variant,
        "seed": seed,
        "step": step,
        "lock_sha256": lock_sha256,
        "test_time_input": "RGB+CAMERA",
        "writer_trained": False,
    }
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    torch.save(payload, path)
    restored, _ = load_checkpoint(path, "cpu")
    if hash_value(restored.state_dict()) != hash_value(payload["carrier"]):
        raise RuntimeError("Checkpoint restore mismatch")
    return sha(path)


def load_checkpoint(path, device):
    payload = torch.load(path, map_location=device, weights_only=True)
    if payload.get("schema") != "mcss.geometry_carrier.v2" or payload.get("writer_trained"):
        raise PermissionError("Static geometry carrier schema required")
    config = CarrierConfig(**payload["config"])
    if tuple(config.grid_size) != (16, 16, 16):
        raise PermissionError("Only frozen16 carrier")
    model = DynamicSceneCarrier(config).to(device)
    model.load_state_dict(payload["carrier"], strict=True)
    return model, payload


def observations(batch):
    if batch.depth is not None:
        raise PermissionError("Depth must never enter carrier input")
    return [
        OnlineObservation(
            batch.scene_id,
            fid,
            batch.rgb[0, j],
            Cameras(
                batch.cameras.intrinsics[0, j], batch.cameras.c2w[0, j], batch.cameras.image_size
            ),
        )
        for j, fid in enumerate(batch.frame_ids)
    ]


class SceneData:
    """Depth-free construction loader, disjoint offline training supervision boundary."""

    def __init__(self, record, manifest, root, device, access):
        self.record, self.device = record, device
        self.allowed = [r["scene_id"] for r in manifest["scenes"] if r["split"] == record["split"]]
        self.forbidden = list(
            set(manifest.get("protected_scene_ids", []))
            | {r["scene_id"] for r in manifest["scenes"] if r["split"] != record["split"]}
        )
        if record["split"] not in ("TRAIN", "DEV"):
            raise PermissionError("Fresh data unavailable to ordinary train/dev loader")
        self.loader = ContextOnlyLoader(
            record,
            manifest["image_size"],
            root,
            device,
            track="RGB_ONLY",
            capacity_scene_ids=self.allowed,
            holdout_scene_ids=self.forbidden,
            access_log=access,
        )
        self.media = _Media(record, manifest["image_size"], root, device, access)
        self._contexts = {}
        self._targets = {}
        prior_path = Path(root) / "train_depth_prior.json"
        prior = read(prior_path)
        prior = FrozenTrainingPrior(prior["near_m"], prior["far_m"], sha(prior_path))
        self.bounds = {
            role: frozen_gt_free_bounds(
                context_camera_bundle(record, role, manifest["image_size"]), prior
            ).to(device=device, dtype=torch.float32)
            for role in ("A", "B")
        }

    def context(self, role):
        if role not in ("A", "B", "anchor"):
            raise PermissionError("Only context RGB/camera")
        if role not in self._contexts:
            self._contexts[role] = observations(self.loader.context(role))
        return self._contexts[role]

    def target(self, slot):
        if self.record["split"] != "TRAIN":
            raise PermissionError("DEV/fresh target cannot enter optimizer")
        fid = self.record["roles"]["primary_query"][slot]
        if fid not in self._targets:
            self._targets[fid] = self.media.read(
                [fid], depth_allowed=True, purpose="TRAIN_OFFLINE_SUPERVISION"
            )
        return self._targets[fid]


def select_checkpoint(candidates):
    if any(r.get("partition") != "DEV" for r in candidates):
        raise PermissionError("Checkpoint selection requires DEV-only provenance")
    eligible = [r for r in candidates if r["eligible"]]
    if not eligible:
        return {"status": "NO_ELIGIBLE_CHECKPOINT", "selected_step": None, "checkpoint": None}
    best = min(r["query_absrel"] for r in eligible)
    selected = min(
        (r for r in eligible if r["query_absrel"] <= best + 1e-8), key=lambda r: r["step"]
    )
    return {
        "status": "SELECTED",
        "selected_step": selected["step"],
        "checkpoint": selected["checkpoint"],
        "checkpoint_sha256": selected["checkpoint_sha256"],
        "dev_query_absrel": selected["query_absrel"],
    }


def verify_data(root):
    for path, digest in read(Path(root) / "preregistration.json")["data_sha256"].items():
        if sha(Path(path)) != digest:
            raise PermissionError(f"Locked TRAIN/DEV data changed: {path}")
