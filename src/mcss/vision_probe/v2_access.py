"""Fail-closed V2 data independence and prediction/evaluation phase contracts.

This is a pipeline audit guard, not an operating-system security sandbox. Use its
context manager around processing to intercept ordinary Python file opens. Only
explicit annotation-read APIs authorize planned mask paths and record their use.
"""

from __future__ import annotations

import builtins
import hashlib
import io
import json
import os
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from PIL import Image


class IndependenceError(ValueError):
    """Independent confirmation must remain closed."""


class AccessDenied(PermissionError):
    """A protected identity or premature annotation access was requested."""


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _identity(row):
    return {str(row.get("sequence", "")), str(row.get("source_video_id", row.get("sequence", "")))}


def validate_independent_manifest(
    manifest: dict, historical_ids=(), historical_hashes=(), protected_ids=()
) -> dict:
    """Validate explicit provenance before any independent annotation is opened.

    Sequence/video identity and exact RGB hashes must be disjoint from historical
    exposure. Provenance uncertainty, missing hashes or authorization fail closed.
    Near-duplicate evidence should be attached to each row by the data auditor;
    its declared UNVERIFIED status never passes this function.
    """
    errors = []
    history, protected = set(historical_ids), set(protected_ids)
    old_hashes = {str(x).lower() for x in historical_hashes}
    if manifest.get("independence_status") != "VERIFIED":
        errors.append("manifest independence is UNVERIFIED")
    rows = manifest.get("sequences", [])
    if not rows:
        errors.append("no independent sequences")
    seen_ids, seen_hashes = set(), set()
    for row in rows:
        identities = _identity(row)
        sequence = row.get("sequence")
        if not sequence or not row.get("source_dataset"):
            errors.append(f"{sequence}: missing identity/source provenance")
        if not row.get("source_video_id") or row.get("source_video_identity_status") != "VERIFIED":
            errors.append(f"{sequence}: original video identity UNVERIFIED")
        if identities & history:
            errors.append(f"{sequence}: historical sequence/video overlap")
        if identities & protected:
            errors.append(f"{sequence}: protected sequence/video")
        if identities & seen_ids:
            errors.append(f"{sequence}: repeated sequence/video identity")
        seen_ids.update(identities)
        if row.get("independence_status") != "VERIFIED":
            errors.append(f"{sequence}: independence UNVERIFIED")
        if row.get("near_duplicate_status") != "VERIFIED":
            errors.append(f"{sequence}: near-duplicate independence UNVERIFIED")
        if row.get("authorized") is not True or not row.get("license"):
            errors.append(f"{sequence}: authorization/license unverified")
        frames = [row.get("source", {}), *row.get("targets", [])]
        if {t.get("target_slot") for t in row.get("targets", [])} != {1, 2}:
            errors.append(f"{sequence}: require target slots 1 and 2")
        frame_ids = []
        for frame in frames:
            frame_id = frame.get("frame_id")
            frame_ids.append(frame_id)
            digest = str(frame.get("image_sha256", "")).lower()
            if not frame_id or not frame.get("image_path") or not frame.get("mask_path"):
                errors.append(f"{sequence}: incomplete frame manifest")
            if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                errors.append(f"{sequence}: missing/invalid exact image hash")
            elif digest in old_hashes:
                errors.append(f"{sequence}: historical exact image hash overlap")
            if digest in seen_hashes:
                errors.append(f"{sequence}: duplicate image hash within confirmation")
            seen_hashes.add(digest)
        if len(set(frame_ids)) != len(frame_ids):
            errors.append(f"{sequence}: source/target frame identities are not distinct")
    audit = {
        "independence_status": "VERIFIED" if not errors else "UNVERIFIED",
        "n_sequences": len(rows),
        "errors": errors,
        "historical_id_count": len(history),
        "historical_hash_count": len(old_hashes),
        "confirmation_allowed": not errors,
    }
    if errors:
        raise IndependenceError("BLOCKED_NO_INDEPENDENT_DATA: " + "; ".join(errors))
    return audit


class PredictionAccessGuard:
    """Phased access to a fixed list of source-annotated frame pairs.

    Discovery only reads source masks and reuses sealed raw rewards externally.
    Confirmation requires independence validation and a complete, hash-verified
    prediction manifest before target-mask access. Protected IDs always win.
    """

    def __init__(
        self,
        manifest: dict,
        *,
        mode="confirmation",
        allowed_ids=(),
        historical_ids=(),
        historical_hashes=(),
        protected_ids=(),
        protected_paths=(),
    ):
        if mode not in ("confirmation", "discovery"):
            raise ValueError("Unknown access mode")
        # Deep-copy to prevent external mutation of the allowlist after validation.
        self.manifest = json.loads(json.dumps(manifest))
        self.mode = mode
        self.protected_ids = set(protected_ids)
        self.protected_paths = {Path(p).resolve() for p in protected_paths}
        self.source_log, self.target_log, self.denied_log = [], [], []
        self.phase = "prediction"
        self._authorized_path = None
        self._predictions_path = None
        self._predictions_hash = None
        self._prediction_artifacts = {}
        self._installed = False
        if mode == "confirmation":
            self.independence_audit = validate_independent_manifest(
                self.manifest, historical_ids, historical_hashes, protected_ids
            )
        else:
            ids = {r.get("sequence") for r in self.manifest.get("sequences", [])}
            if not ids or ids != set(allowed_ids):
                raise AccessDenied("Discovery manifest must exactly match approved discovery IDs")
            self.independence_audit = {
                "independence_status": "DISCOVERY_ONLY",
                "confirmation_allowed": False,
            }
        self._pairs = {}
        self._source_paths, self._target_paths = set(), set()
        for row in self.manifest.get("sequences", []):
            sequence = row["sequence"]
            if _identity(row) & self.protected_ids:
                raise AccessDenied("Protected sequence identity")
            source = row["source"]
            targets = row["targets"]
            if len(targets) != 2 or {t["target_slot"] for t in targets} != {1, 2}:
                raise ValueError("Every sequence needs exactly both target slots")
            self._source_paths.add(Path(source["mask_path"]).resolve())
            for target in targets:
                if source["frame_id"] == target["frame_id"]:
                    raise AccessDenied("Source and target must be different frames")
                key = (sequence, target["target_slot"])
                if key in self._pairs:
                    raise ValueError("Duplicate planned pair")
                self._pairs[key] = (source, target)
                self._target_paths.add(Path(target["mask_path"]).resolve())
                for frame in (source, target):
                    for name in ("image_path", "mask_path"):
                        self._check_protected(Path(frame[name]).resolve())
        if self._source_paths & self._target_paths:
            raise AccessDenied("Source/target annotation path aliasing is forbidden")
        if mode == "confirmation":
            for row in self.manifest["sequences"]:
                for frame in [row["source"], *row["targets"]]:
                    image_path = Path(frame["image_path"]).resolve()
                    if image_path in self._source_paths | self._target_paths:
                        raise AccessDenied("Image paths cannot alias annotation paths")
                    if _sha(image_path) != frame["image_sha256"].lower():
                        raise IndependenceError(
                            "Exact image bytes do not match independent manifest"
                        )

    def _check_protected(self, path):
        if set(path.parts) & self.protected_ids or any(
            path == p or p in path.parents for p in self.protected_paths
        ):
            self.denied_log.append({"path": str(path), "reason": "protected"})
            raise AccessDenied("Reserve/official validation/protected paths are forbidden")

    def _check_open(self, file):
        if isinstance(file, int):
            return
        path = Path(os.fsdecode(file)).resolve()
        self._check_protected(path)
        if path in self._source_paths | self._target_paths and path != self._authorized_path:
            self.denied_log.append({"path": str(path), "reason": "annotation outside guarded API"})
            raise AccessDenied("Annotation access requires the phased guarded API")

    def __enter__(self):
        if self._installed:
            raise RuntimeError("Guard is already installed")
        self._old_builtin, self._old_io = builtins.open, io.open

        def guarded_builtin(file, *args, **kwargs):
            self._check_open(file)
            return self._old_builtin(file, *args, **kwargs)

        def guarded_io(file, *args, **kwargs):
            self._check_open(file)
            return self._old_io(file, *args, **kwargs)

        builtins.open, io.open = guarded_builtin, guarded_io
        self._installed = True
        return self

    def __exit__(self, exc_type, exc, traceback):
        builtins.open, io.open = self._old_builtin, self._old_io
        self._installed = False

    @contextmanager
    def _permit(self, path):
        self._check_protected(path)
        previous = self._authorized_path
        self._authorized_path = path
        try:
            yield
        finally:
            self._authorized_path = previous

    def _pair(self, sequence, source_id, target_id):
        matches = [
            (source, target)
            for (seq, _), (source, target) in self._pairs.items()
            if seq == sequence
            and source["frame_id"] == source_id
            and target["frame_id"] == target_id
        ]
        if len(matches) != 1 or source_id == target_id:
            raise AccessDenied("Frame pair is not in the locked plan")
        return matches[0]

    def read_source_mask(self, sequence, source_id, target_id, path=None):
        if self.phase != "prediction":
            raise AccessDenied("Source signal reads belong to prediction phase")
        source, _ = self._pair(sequence, source_id, target_id)
        planned = Path(source["mask_path"]).resolve()
        if path is not None and Path(path).resolve() != planned:
            raise AccessDenied("Source mask path differs from locked plan")
        with self._permit(planned):
            with Image.open(planned) as image:
                value = np.asarray(image).copy()
            digest = _sha(planned)
        self.source_log.append(
            {
                "sequence": sequence,
                "source_id": source_id,
                "target_id": target_id,
                "path": str(planned),
                "sha256": digest,
                "phase": self.phase,
            }
        )
        return value

    def lock_predictions(self, path, expected_sha256):
        if self.mode != "confirmation" or self.phase != "prediction":
            raise AccessDenied("Prediction lock is only for independent confirmation")
        path = Path(path).resolve()
        self._check_protected(path)
        if path in self._source_paths | self._target_paths:
            raise AccessDenied("Prediction manifest cannot alias annotations")
        if _sha(path) != expected_sha256:
            raise AccessDenied("Prediction manifest hash mismatch")
        document = json.loads(path.read_text())
        if document.get("complete") is not True:
            raise AccessDenied("Prediction manifest is incomplete")
        found, artifacts = set(), {}
        for row in document.get("predictions", []):
            key = (row["sequence"], row["target_slot"])
            if key not in self._pairs or key in found:
                raise AccessDenied("Unexpected or duplicate prediction pair")
            source, target = self._pairs[key]
            if row["source_id"] != source["frame_id"] or row["target_id"] != target["frame_id"]:
                raise AccessDenied("Prediction frame mismatch")
            artifact = Path(row["artifact_path"])
            if not artifact.is_absolute():
                artifact = path.parent / artifact
            artifact = artifact.resolve()
            self._check_protected(artifact)
            if artifact in self._source_paths | self._target_paths:
                raise AccessDenied("Predictions cannot alias annotation files")
            if _sha(artifact) != row["artifact_sha256"]:
                raise AccessDenied("Prediction artifact hash mismatch")
            artifacts[str(artifact)] = row["artifact_sha256"]
            found.add(key)
        if found != set(self._pairs):
            raise AccessDenied("All planned predictions must be saved before target GT access")
        if self.target_log:
            raise AccessDenied("Target GT was already read")
        self._predictions_path, self._predictions_hash = path, expected_sha256
        self._prediction_artifacts = artifacts
        self.phase = "evaluation"
        return {
            "locked": True,
            "manifest_sha256": expected_sha256,
            "prediction_pairs": len(found),
            "target_reads_before_lock": 0,
        }

    def read_target_mask(self, sequence, source_id, target_id, path=None):
        if self.mode != "confirmation" or self.phase != "evaluation":
            raise AccessDenied("Target GT unavailable before complete prediction lock")
        if _sha(self._predictions_path) != self._predictions_hash:
            raise AccessDenied("Locked prediction manifest changed")
        for artifact, digest in self._prediction_artifacts.items():
            if _sha(artifact) != digest:
                raise AccessDenied("Locked prediction artifact changed")
        _, target = self._pair(sequence, source_id, target_id)
        planned = Path(target["mask_path"]).resolve()
        if path is not None and Path(path).resolve() != planned:
            raise AccessDenied("Target mask path differs from locked plan")
        with self._permit(planned):
            with Image.open(planned) as image:
                value = np.asarray(image).copy()
            digest = _sha(planned)
        self.target_log.append(
            {
                "sequence": sequence,
                "source_id": source_id,
                "target_id": target_id,
                "path": str(planned),
                "sha256": digest,
                "phase": self.phase,
                "prediction_manifest_sha256": self._predictions_hash,
            }
        )
        return value
