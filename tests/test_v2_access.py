import copy
import hashlib
import json

import numpy as np
import pytest
from PIL import Image

from mcss.vision_probe.v2_access import (
    AccessDenied,
    IndependenceError,
    PredictionAccessGuard,
    validate_independent_manifest,
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def manifest(tmp_path):
    frames = []
    for i in range(3):
        image = tmp_path / f"image{i}.png"
        mask = tmp_path / f"mask{i}.png"
        Image.fromarray(np.full((2, 2, 3), i, dtype=np.uint8)).save(image)
        Image.fromarray(np.full((2, 2), i + 1, dtype=np.uint8)).save(mask)
        frames.append(
            {
                "frame_id": str(i),
                "image_path": str(image),
                "mask_path": str(mask),
                "image_sha256": sha(image),
                "target_slot": i,
            }
        )
    return {
        "independence_status": "VERIFIED",
        "sequences": [
            {
                "sequence": "fresh",
                "source_dataset": "synthetic",
                "source_video_id": "video-fresh",
                "source_video_identity_status": "VERIFIED",
                "near_duplicate_status": "VERIFIED",
                "authorized": True,
                "license": "test fixture",
                "independence_status": "VERIFIED",
                "source": frames[0],
                "targets": frames[1:],
            }
        ],
    }


def prediction_lock(tmp_path):
    artifact = tmp_path / "predictions.json"
    artifact.write_text(json.dumps({"actions": ["OFF", "OFF"]}))
    path = tmp_path / "prediction_manifest.json"
    path.write_text(
        json.dumps(
            {
                "complete": True,
                "predictions": [
                    {
                        "sequence": "fresh",
                        "target_slot": i,
                        "source_id": "0",
                        "target_id": str(i),
                        "artifact_path": str(artifact),
                        "artifact_sha256": sha(artifact),
                    }
                    for i in (1, 2)
                ],
            }
        )
    )
    return path, sha(path)


def test_prediction_phase_denies_target_reads_and_direct_python_bypass(tmp_path):
    plan = manifest(tmp_path)
    guard = PredictionAccessGuard(plan)
    with guard:
        with pytest.raises(AccessDenied):
            guard.read_target_mask("fresh", "0", "1")
        with pytest.raises(AccessDenied):
            Image.open(plan["sequences"][0]["targets"][0]["mask_path"])
        with pytest.raises(AccessDenied):
            open(plan["sequences"][0]["source"]["mask_path"], "rb")
        source = guard.read_source_mask("fresh", "0", "1")
        assert (source == 1).all()
        assert guard.target_log == []
        assert len(guard.source_log) == 1
        path, digest = prediction_lock(tmp_path)
        guard.lock_predictions(path, digest)
        target = guard.read_target_mask("fresh", "0", "1")
        assert (target == 2).all()
        assert len(guard.target_log) == 1


def test_partial_or_tampered_prediction_manifest_cannot_unlock(tmp_path):
    guard = PredictionAccessGuard(manifest(tmp_path))
    path, digest = prediction_lock(tmp_path)
    document = json.loads(path.read_text())
    document["predictions"].pop()
    path.write_text(json.dumps(document))
    with pytest.raises(AccessDenied, match="hash mismatch"):
        guard.lock_predictions(path, digest)
    with pytest.raises(AccessDenied, match="All planned"):
        guard.lock_predictions(path, sha(path))
    assert guard.phase == "prediction"
    assert guard.target_log == []


def test_post_lock_artifact_mutation_denies_gt(tmp_path):
    guard = PredictionAccessGuard(manifest(tmp_path))
    path, digest = prediction_lock(tmp_path)
    guard.lock_predictions(path, digest)
    (tmp_path / "predictions.json").write_text("changed")
    with pytest.raises(AccessDenied, match="artifact changed"):
        guard.read_target_mask("fresh", "0", "1")
    assert guard.target_log == []


def test_historical_overlap_hash_overlap_uncertain_and_protected_reject(tmp_path):
    plan = manifest(tmp_path)
    with pytest.raises(IndependenceError, match="historical sequence"):
        validate_independent_manifest(plan, historical_ids={"video-fresh"})
    with pytest.raises(IndependenceError, match="exact image hash"):
        validate_independent_manifest(
            plan, historical_hashes={plan["sequences"][0]["source"]["image_sha256"]}
        )
    uncertain = copy.deepcopy(plan)
    uncertain["sequences"][0]["independence_status"] = "UNVERIFIED"
    with pytest.raises(IndependenceError, match="UNVERIFIED"):
        validate_independent_manifest(uncertain)
    with pytest.raises(IndependenceError, match="protected"):
        PredictionAccessGuard(plan, protected_ids={"fresh"})


def test_discovery_only_exact_ids_source_access_and_protected_paths(tmp_path):
    plan = manifest(tmp_path)
    with pytest.raises(AccessDenied, match="exactly match"):
        PredictionAccessGuard(plan, mode="discovery", allowed_ids={"other"})
    guard = PredictionAccessGuard(
        plan, mode="discovery", allowed_ids={"fresh"}, protected_ids={"reserve", "official-val"}
    )
    with guard:
        guard.read_source_mask("fresh", "0", "2")
        with pytest.raises(AccessDenied):
            guard.read_target_mask("fresh", "0", "2")
        with pytest.raises(AccessDenied):
            (tmp_path / "reserve" / "frame.png").read_bytes()
        with pytest.raises(AccessDenied):
            (tmp_path / "official-val" / "frame.png").read_bytes()
        with pytest.raises(AccessDenied, match="independent confirmation"):
            guard.lock_predictions(tmp_path / "anything", "x")
    assert guard.target_log == []


def test_same_frame_and_mask_aliasing_rejected(tmp_path):
    plan = manifest(tmp_path)
    plan["sequences"][0]["targets"][0]["frame_id"] = "0"
    with pytest.raises(AccessDenied, match="different frames"):
        PredictionAccessGuard(plan, mode="discovery", allowed_ids={"fresh"})
    plan["sequences"][0]["targets"][0]["frame_id"] = "1"
    plan["sequences"][0]["targets"][0]["mask_path"] = plan["sequences"][0]["source"]["mask_path"]
    with pytest.raises(AccessDenied, match="aliasing"):
        PredictionAccessGuard(plan, mode="discovery", allowed_ids={"fresh"})


def test_independence_binds_actual_image_bytes(tmp_path):
    plan = manifest(tmp_path)
    plan["sequences"][0]["source"]["image_sha256"] = "a" * 64
    with pytest.raises(IndependenceError, match="Exact image bytes"):
        PredictionAccessGuard(plan)


def test_unknown_video_or_missing_near_duplicate_evidence_fails_closed(tmp_path):
    for field in ("source_video_id", "source_video_identity_status", "near_duplicate_status"):
        plan = manifest(tmp_path)
        plan["sequences"][0].pop(field)
        with pytest.raises(IndependenceError, match="UNVERIFIED"):
            validate_independent_manifest(plan)
