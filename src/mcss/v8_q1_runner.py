"""Frozen two-phase runner for the V8 Q1 supplier-risk qualification."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import tempfile
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from mcss.data.manifest import SceneManifest, load_manifest
from mcss.geometry_supplier import GeometrySupplierRequest
from mcss.types import Cameras
from mcss.v8_geometry import (
    V8Proposal,
    depth_z_to_points_cam,
    ray_distance_to_depth_z,
    read_v8_proposal_cache,
    write_v8_proposal_cache,
)
from mcss.v8_risk_audit import (
    TYPE_MODEL_SUPPORTED,
    TYPE_NAMES,
    build_anchor_risk_fields,
    deterministic_permuted_risk,
    deterministic_source_derangement,
    evaluate_selective_risk,
    macro_mean,
    paired_bootstrap_mean_ci,
    read_v8_risk_cache,
    write_v8_risk_cache,
)
from mcss.v8_suppliers import EMVSNetSupplier, UniDepthV2SmallSupplier

Q1_SCHEMA_VERSION = "mcss.v8_q1.v2"
EXPECTED_WINDOW_COUNT = 32
RISK_NAMES = (
    "full",
    "emvsnet_native",
    "unidepth_native",
    "disagreement",
    "drop",
    "same_family",
    "full_no_drop",
    "full_no_disagreement",
    "random",
    "oracle",
)
NATIVE_RISK_NAMES = (
    "emvsnet_native",
    "unidepth_native",
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args(argv)
    config_path = args.config.resolve()
    config = _load_config(config_path)
    output = _path(config["output_dir"])
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty Q1 output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    record = _initial_record(config_path, config)
    _write_json(output / "q1_running.json", record)
    exit_code = 3
    try:
        _configure_determinism(int(config["seed"]))
        windows = _load_frozen_windows(_path(config["frozen_windows"]))
        q0 = _load_q0_gate(_path(config["q0_result"]))
        _check_q0_artifacts(q0, config)
        record["checks"]["q0_gate"] = {"status": "PASS"}
        record["checks"]["frozen_population"] = {
            "status": "PASS",
            "window_count": len(windows),
        }

        uni = UniDepthV2SmallSupplier(
            code_root=_path(config["unidepth_code_root"]),
            local_model_dir=_path(config["unidepth_bundle"]),
            device=str(config["device"]),
        )
        emv = EMVSNetSupplier(
            code_root=_path(config["emvsnet_code_root"]),
            checkpoint=_path(config["emvsnet_checkpoint"]),
            device=str(config["device"]),
            image_scale=0.25,
            depth_bins=64,
            max_h=512,
            max_w=640,
            evidential_method=str(config["evidential_method"]),
        )
        uni._load()
        emv._load()
        record["checks"]["model_load"] = {"status": "PASS"}

        phase_a = []
        for index, window in enumerate(windows):
            phase_a.append(
                _seal_window_proposals_and_risk(
                    output,
                    index,
                    window,
                    uni,
                    emv,
                    grid_stride=int(config["grid_stride"]),
                    cell_width_m=float(config["cell_width_m"]),
                )
            )
            record["progress"] = {
                "phase": "label_blind_proposal_sealing",
                "completed_windows": index + 1,
                "total_windows": len(windows),
            }
            _write_json(output / "q1_running.json", record)
            print(f"Q1 phase A: sealed window {index + 1}/{len(windows)}", flush=True)

        _require_all_phase_a_sealed(phase_a)
        record["checks"]["all_risk_caches_sealed_before_labels"] = {
            "status": "PASS",
            "window_count": len(phase_a),
        }
        labels_started_at = datetime.now(UTC).isoformat()
        record["research_boundary"]["labels_opened_after_all_risk_seals"] = True
        record["research_boundary"]["labels_started_at_utc"] = labels_started_at

        access_log: list[dict[str, Any]] = []
        windows_metrics = []
        for index, phase in enumerate(phase_a):
            windows_metrics.append(
                _evaluate_sealed_window(
                    output,
                    phase,
                    config,
                    access_log=access_log,
                    labels_started_at=labels_started_at,
                )
            )
            record["progress"] = {
                "phase": "postseal_context_depth_audit",
                "completed_windows": index + 1,
                "total_windows": len(windows),
            }
            _write_json(output / "q1_running.json", record)
            print(f"Q1 phase B: evaluated window {index + 1}/{len(windows)}", flush=True)

        _write_json(output / "access_log.json", access_log)
        summary = _aggregate(windows_metrics, config)
        decision = _decide(summary, config["decision"])
        record["windows"] = windows_metrics
        record["summary"] = summary
        record["decision"] = decision
        record["access_log"] = {
            "path": str(output / "access_log.json"),
            "entry_count": len(access_log),
            "all_after_risk_seals": all(
                bool(entry["opened_after_all_risk_seals"]) for entry in access_log
            ),
        }
        record["status"] = decision["status"]
        exit_code = 0 if record["status"] == "GO_V8_Q1_RISK_QUALIFICATION" else 2
    except Exception as error:
        record["status"] = "INCONCLUSIVE_V8_Q1_RUNTIME"
        record["error"] = {"type": type(error).__name__, "message": str(error)}
        exit_code = 3
    record["finished_seconds"] = time.time() - started
    _write_json(output / "q1_result.json", record)
    (output / "Q1_REPORT.md").write_text(_render_report(record), encoding="utf-8")
    try:
        (output / "q1_running.json").unlink()
    except FileNotFoundError:
        pass
    print(json.dumps({"status": record["status"], "output": str(output)}), flush=True)
    return exit_code


def _load_config(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "mcss.v8_q1_config.v2":
        raise ValueError("unsupported V8 Q1 config schema")
    required = {
        "seed",
        "frozen_windows",
        "q0_result",
        "output_dir",
        "device",
        "unidepth_code_root",
        "unidepth_bundle",
        "emvsnet_code_root",
        "emvsnet_checkpoint",
        "evidential_method",
        "grid_stride",
        "cell_width_m",
        "coverage_grid",
        "severe_error_thresholds_m",
        "bootstrap_replicates",
        "decision",
    }
    if not required.issubset(payload):
        raise ValueError(f"V8 Q1 config is missing {sorted(required - set(payload))}")
    if payload["evidential_method"] != "der":
        raise ValueError("V8 Q1 freezes the publisher DER inference contract")
    if int(payload["grid_stride"]) != 4 or float(payload["cell_width_m"]) != 0.25:
        raise ValueError("V8 Q1 freezes stride 4 and a 0.25 m physical risk scale")
    if list(payload["severe_error_thresholds_m"]) != [0.25, 0.5]:
        raise ValueError("V8 Q1 freezes severe-error thresholds at 0.25 m and 0.50 m")
    decision = payload["decision"]
    required_decision = {
        "minimum_windows",
        "minimum_comparison_coverage_macro",
        "minimum_relative_risk_gain",
        "minimum_ablation_degradation",
        "minimum_spearman_macro",
        "minimum_nonnegative_spearman_windows",
        "minimum_qjf_defined_windows",
        "minimum_type_ordering_defined_windows",
    }
    if not isinstance(decision, dict) or not required_decision.issubset(decision):
        missing = (
            required_decision - set(decision)
            if isinstance(decision, dict)
            else required_decision
        )
        raise ValueError(f"V8 Q1 decision config is missing {sorted(missing)}")
    if int(decision["minimum_windows"]) != EXPECTED_WINDOW_COUNT:
        raise ValueError("V8 Q1 freezes the population at 32 windows")
    for name in (
        "minimum_comparison_coverage_macro",
        "minimum_relative_risk_gain",
        "minimum_ablation_degradation",
    ):
        if not 0.0 <= float(decision[name]) <= 1.0:
            raise ValueError(f"{name} must be in [0, 1]")
    if not -1.0 <= float(decision["minimum_spearman_macro"]) <= 1.0:
        raise ValueError("minimum_spearman_macro must be in [-1, 1]")
    for name in (
        "minimum_nonnegative_spearman_windows",
        "minimum_qjf_defined_windows",
        "minimum_type_ordering_defined_windows",
    ):
        if not 1 <= int(decision[name]) <= EXPECTED_WINDOW_COUNT:
            raise ValueError(f"{name} must be in [1, 32]")
    return payload


def _load_frozen_windows(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    windows = payload.get("windows")
    if payload.get("partition") != "train" or not isinstance(windows, list) or len(windows) != 32:
        raise ValueError("Q1 requires exactly 32 frozen development-train windows")
    seen = set()
    for window in windows:
        if not isinstance(window, dict):
            raise ValueError("every frozen Q1 window must be a mapping")
        scene_id = window.get("scene_id")
        frame_ids = window.get("frame_ids")
        manifest = str(window.get("manifest", ""))
        if (
            not isinstance(scene_id, str)
            or not isinstance(frame_ids, list)
            or len(frame_ids) != 4
            or len(set(frame_ids)) != 4
        ):
            raise ValueError("every Q1 window requires one scene and four unique frames")
        if any(token in manifest.lower() for token in ("diagnostic_test", "final_holdout")):
            raise ValueError("Q1 cannot access diagnostic-test or final-holdout manifests")
        key = (scene_id, tuple(frame_ids))
        if key in seen:
            raise ValueError("Q1 frozen windows must be scene/window unique")
        seen.add(key)
    return windows


def _load_q0_gate(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "GO_TO_Q1_LABEL_BLIND_AUDIT":
        raise ValueError("Q0 must pass before Q1 can run")
    checks = payload.get("checks", {})
    required = {
        "input_contract",
        "model_load",
        "checkpoint_method_provenance",
        "all_proposals_sealed_before_context_depth",
        "repeat_stability",
        "source_drop_interventions",
        "peak_memory",
    }
    if not required.issubset(checks) or any(
        checks[name].get("status") != "PASS" for name in required
    ):
        raise ValueError("Q0 must pass every mechanical and provenance check")
    return payload


def _check_q0_artifacts(q0: dict[str, Any], config: dict[str, Any]) -> None:
    q0_config = q0.get("config", {})
    expected = {
        "unidepth_code_root": _path(config["unidepth_code_root"]),
        "unidepth_bundle": _path(config["unidepth_bundle"]),
        "emvsnet_code_root": _path(config["emvsnet_code_root"]),
        "emvsnet_checkpoint": _path(config["emvsnet_checkpoint"]),
    }
    for name, path in expected.items():
        if Path(str(q0_config.get(name, ""))).resolve() != path:
            raise ValueError(f"Q1 {name} does not match the passed Q0 configuration")
    if q0_config.get("evidential_method") != config["evidential_method"]:
        raise ValueError("Q1 evidential method does not match Q0")
    checkpoint_hash = q0["checks"]["checkpoint_method_provenance"].get("checkpoint_sha256")
    if checkpoint_hash != _sha256_file(expected["emvsnet_checkpoint"]):
        raise ValueError("Q1 EMVSNet checkpoint hash does not match Q0")


def _seal_window_proposals_and_risk(
    output: Path,
    index: int,
    window: dict[str, Any],
    uni: UniDepthV2SmallSupplier,
    emv: EMVSNetSupplier,
    *,
    grid_stride: int,
    cell_width_m: float,
) -> dict[str, Any]:
    window_id = f"window_{index:03d}"
    root = output / "phase_a" / window_id
    manifest = load_manifest(window["manifest"])
    if manifest.scene_id != window["scene_id"]:
        raise ValueError("frozen scene ID does not match manifest")
    request = _load_request(manifest, window["frame_ids"]).canonical()
    _check_request(request)
    input_hash = request.input_sha256

    full_uni = uni.infer(request)
    full_emv = emv.infer(request)
    _check_proposal(full_uni, request)
    _check_proposal(full_emv, request)
    uni_cache = write_v8_proposal_cache(
        root / "unidepth_full",
        full_uni,
        metadata={
            "phase": "label_blind",
            "window_id": window_id,
            "input_sha256": input_hash,
        },
    )
    emv_cache = write_v8_proposal_cache(
        root / "emvsnet_full",
        full_emv,
        metadata={
            "phase": "label_blind",
            "window_id": window_id,
            "input_sha256": input_hash,
        },
    )

    drops: list[V8Proposal] = []
    drop_caches = []
    for source_index in (1, 2, 3):
        keep = [value for value in range(4) if value != source_index]
        drop_request = _select_request(request, keep)
        proposal = emv.infer(drop_request)
        _check_proposal(proposal, drop_request)
        _check_drop_anchor_contract(request, drop_request, proposal)
        drops.append(proposal)
        drop_caches.append(
            write_v8_proposal_cache(
                root / f"emvsnet_drop_source_{source_index}",
                proposal,
                metadata={
                    "phase": "label_blind",
                    "window_id": window_id,
                    "source_index_dropped": source_index,
                    "input_sha256": drop_request.input_sha256,
                },
            )
        )

    derangement = deterministic_source_derangement(manifest.scene_id, request.frame_ids)
    shuffled_request = _shuffled_source_request(request, derangement)
    shuffled = emv.infer(shuffled_request)
    _check_proposal(shuffled, shuffled_request)
    shuffle_cache = write_v8_proposal_cache(
        root / "emvsnet_shuffled_sources",
        shuffled,
        metadata={
            "phase": "label_blind_negative_control",
            "window_id": window_id,
            "source_rgb_position_mapping": list(derangement),
            "camera_positions_unchanged": True,
            "input_sha256": shuffled_request.input_sha256,
        },
    )

    fields = build_anchor_risk_fields(
        emvsnet_points=full_emv.points_cam_m[0],
        unidepth_points=full_uni.points_cam_m[0],
        emvsnet_native_risk=full_emv.native_risk[0],
        unidepth_native_risk=full_uni.native_risk[0],
        emvsnet_valid=full_emv.valid_mask[0],
        unidepth_valid=full_uni.valid_mask[0],
        drop_points=[proposal.points_cam_m[0] for proposal in drops],
        drop_native_risks=[proposal.native_risk[0] for proposal in drops],
        drop_valids=[proposal.valid_mask[0] for proposal in drops],
        cell_width_m=cell_width_m,
    )
    shuffle_valid = full_emv.valid_mask[0] & shuffled.valid_mask[0]
    shuffle_response_raw = (
        np.linalg.norm(full_emv.points_cam_m[0] - shuffled.points_cam_m[0], axis=-1)
        / cell_width_m
    ).astype(np.float32)
    shuffle_response_raw[~shuffle_valid] = np.nan
    fields["shuffle_response_raw"] = shuffle_response_raw
    fields["shuffle_response_risk"] = np.minimum(shuffle_response_raw, 1.0).astype(np.float32)
    fields["shuffle_valid"] = shuffle_valid
    sampled = {name: _sample_grid(value, grid_stride) for name, value in fields.items()}
    sampled["random_risk"] = deterministic_permuted_risk(
        sampled["full_risk"],
        sampled["comparison_valid"],
        key=f"v8-q1-risk-permutation:{manifest.scene_id}:{window.get('start')}",
    )
    grid_y, grid_x = np.meshgrid(
        np.arange(0, manifest.image_size[0], grid_stride, dtype=np.int16),
        np.arange(0, manifest.image_size[1], grid_stride, dtype=np.int16),
        indexing="ij",
    )
    sampled["pixel_y"] = grid_y
    sampled["pixel_x"] = grid_x
    risk_cache = write_v8_risk_cache(
        root / "risk",
        sampled,
        metadata={
            "scene_id": manifest.scene_id,
            "window_id": window_id,
            "window_start": window.get("start"),
            "frame_ids": list(request.frame_ids),
            "anchor_frame_id": request.frame_ids[0],
            "grid_stride": grid_stride,
            "cell_width_m": cell_width_m,
            "risk_formula": "max(native_emv_rank,native_uni_rank,disagreement,source_drop)",
            "type_semantics": TYPE_NAMES,
            "geometry_action": "emvsnet_full_anchor_only_no_point_fusion",
            "source_derangement": list(derangement),
            "request_input_sha256": input_hash,
            "proposal_caches": {
                "unidepth_full": str(uni_cache),
                "emvsnet_full": str(emv_cache),
                "emvsnet_drops": [str(path) for path in drop_caches],
                "emvsnet_shuffled": str(shuffle_cache),
            },
        },
    )
    phase_record = {
        "window_id": window_id,
        "scene_id": manifest.scene_id,
        "frame_ids": list(request.frame_ids),
        "manifest": str(Path(window["manifest"]).resolve()),
        "risk_cache": str(risk_cache),
        "unidepth_cache": str(uni_cache),
        "emvsnet_cache": str(emv_cache),
        "drop_caches": [str(path) for path in drop_caches],
        "shuffle_cache": str(shuffle_cache),
        "proposal_phase_label_access": False,
        "risk_sealed_at_utc": _sealed_at(risk_cache),
    }
    _write_json(root / "phase_a.json", phase_record)
    return phase_record


def _evaluate_sealed_window(
    output: Path,
    phase: dict[str, Any],
    config: dict[str, Any],
    *,
    access_log: list[dict[str, Any]],
    labels_started_at: str,
) -> dict[str, Any]:
    fields, risk_metadata = read_v8_risk_cache(phase["risk_cache"])
    full_emv, _ = read_v8_proposal_cache(phase["emvsnet_cache"])
    full_uni, _ = read_v8_proposal_cache(phase["unidepth_cache"])
    shuffled, _ = read_v8_proposal_cache(phase["shuffle_cache"])
    manifest = load_manifest(phase["manifest"])
    anchor = {frame.frame_id: frame for frame in manifest.frames}[phase["frame_ids"][0]]
    if anchor.depth is None:
        raise ValueError("Q1 post-seal audit requires anchor context depth")
    depth_path = (manifest.root / anchor.depth).resolve()
    opened_at = datetime.now(UTC).isoformat()
    ray_distance = np.load(depth_path, allow_pickle=False).astype(np.float32)
    access_log.append(
        {
            "window_id": phase["window_id"],
            "scene_id": phase["scene_id"],
            "frame_id": anchor.frame_id,
            "path": str(depth_path),
            "opened_at_utc": opened_at,
            "opened_after_all_risk_seals": opened_at >= labels_started_at,
            "purpose": "postseal_development_context_geometry_audit",
        }
    )
    intrinsics = full_emv.intrinsics[0]
    depth_z = ray_distance_to_depth_z(ray_distance, intrinsics)
    gt_points = depth_z_to_points_cam(depth_z, intrinsics)
    stride = int(config["grid_stride"])
    gt_points_grid = _sample_grid(gt_points, stride)
    gt_valid = _sample_grid(np.isfinite(ray_distance) & (ray_distance > 0), stride)
    comparison_valid = fields["comparison_valid"].astype(np.bool_)
    label_valid = comparison_valid & gt_valid
    if not label_valid.any():
        raise ValueError("Q1 window has no label-valid comparison pixels")

    emv_error = np.linalg.norm(fields["action_points_m"] - gt_points_grid, axis=-1)
    uni_error = np.linalg.norm(fields["unidepth_points_m"] - gt_points_grid, axis=-1)
    shuffled_points = _sample_grid(shuffled.points_cam_m[0], stride)
    shuffle_error = np.linalg.norm(shuffled_points - gt_points_grid, axis=-1)
    risk_maps = {
        "full": fields["full_risk"],
        "emvsnet_native": fields["emvsnet_native_rank"],
        "unidepth_native": fields["unidepth_native_rank"],
        "disagreement": fields["disagreement_risk"],
        "drop": fields["drop_instability_risk"],
        "same_family": fields["same_family_risk"],
        "full_no_drop": fields["full_no_drop_risk"],
        "full_no_disagreement": fields["full_no_disagreement_risk"],
        "random": fields["random_risk"],
        "oracle": emv_error,
    }
    coverages = tuple(float(value) for value in config["coverage_grid"])
    metrics = {
        name: evaluate_selective_risk(risk, emv_error, label_valid, coverages=coverages)
        for name, risk in risk_maps.items()
    }
    severe = {
        str(threshold): {
            name: evaluate_selective_risk(
                risk,
                (emv_error > float(threshold)).astype(np.float32),
                label_valid,
                coverages=coverages,
            )
            for name, risk in risk_maps.items()
        }
        for threshold in config["severe_error_thresholds_m"]
    }

    type_code = fields["type_code"].astype(np.uint8)
    type_metrics = {}
    for code, name in TYPE_NAMES.items():
        mask = label_valid & (type_code == code)
        type_metrics[name] = _error_summary(emv_error, mask)

    supported = label_valid & (type_code == TYPE_MODEL_SUPPORTED)
    supported_count = int(supported.sum())
    native_matched = {}
    for name in NATIVE_RISK_NAMES:
        order = _stable_low_risk_order(risk_maps[name], label_valid)
        mask = np.zeros(label_valid.shape, dtype=np.bool_)
        if supported_count:
            mask.flat[order[:supported_count]] = True
        native_matched[name] = mask
    qjf = {}
    for threshold in config["severe_error_thresholds_m"]:
        threshold = float(threshold)
        quiet, shared = _quiet_joint_failure_masks(
            emv_error,
            uni_error,
            label_valid,
            supported,
            threshold=threshold,
        )
        qjf[str(threshold)] = {
            "quiet_joint_failure_count": int(quiet.sum()),
            "shared_error_count": int(shared.sum()),
            "supported_qjf_rate": _masked_mean(quiet, supported),
            "native_matched_qjf_rate": {
                name: _masked_mean(shared, native_matched[name]) for name in NATIVE_RISK_NAMES
            },
            "supported_emvsnet_severe_rate": _masked_mean(emv_error > threshold, supported),
            "native_matched_emvsnet_severe_rate": {
                name: _masked_mean(emv_error > threshold, native_matched[name])
                for name in NATIVE_RISK_NAMES
            },
            "shared_error_recall_top20_full": _high_risk_recall(
                fields["full_risk"], shared, label_valid, fraction=0.20
            ),
            "shared_error_recall_top20_native": {
                name: _high_risk_recall(risk_maps[name], shared, label_valid, fraction=0.20)
                for name in NATIVE_RISK_NAMES
            },
        }

    phase_b_root = output / "phase_b" / phase["window_id"]
    _write_postseal_cache(
        phase_b_root,
        {
            "gt_points_m": gt_points_grid.astype(np.float32),
            "label_valid": label_valid,
            "emvsnet_error_m": emv_error.astype(np.float32),
            "unidepth_error_m": uni_error.astype(np.float32),
            "shuffled_emvsnet_error_m": shuffle_error.astype(np.float32),
        },
        metadata={
            "scene_id": phase["scene_id"],
            "window_id": phase["window_id"],
            "source_risk_cache": phase["risk_cache"],
            "source_risk_metadata_sha256": _sha256_file(
                Path(phase["risk_cache"]) / "metadata.json"
            ),
            "risk_sealed_at_utc": risk_metadata["sealed_at_utc"],
            "label_opened_at_utc": opened_at,
        },
    )
    result = {
        "window_id": phase["window_id"],
        "scene_id": phase["scene_id"],
        "frame_ids": phase["frame_ids"],
        "grid_sample_count": int(label_valid.size),
        "comparison_valid_count": int(comparison_valid.sum()),
        "label_valid_count": int(label_valid.sum()),
        "comparison_coverage": float(comparison_valid.mean()),
        "label_valid_coverage": float(label_valid.mean()),
        "supported_count": supported_count,
        "all_drop_valid_count": int(
            (comparison_valid & fields["all_drop_valid"].astype(np.bool_)).sum()
        ),
        "all_drop_valid_coverage_of_comparison": float(
            (comparison_valid & fields["all_drop_valid"].astype(np.bool_)).sum()
            / comparison_valid.sum()
        ),
        "supported_coverage_of_comparison": (
            float(supported_count / label_valid.sum()) if label_valid.any() else 0.0
        ),
        "drop_instability_median": _masked_median(
            fields["drop_instability_raw"], comparison_valid
        ),
        "shuffle_response_median": _masked_median(
            fields["shuffle_response_raw"], fields["shuffle_valid"].astype(np.bool_)
        ),
        "emvsnet_error": _error_summary(emv_error, label_valid),
        "unidepth_error": _error_summary(uni_error, label_valid),
        "shuffled_emvsnet_error": _error_summary(
            shuffle_error, label_valid & fields["shuffle_valid"].astype(np.bool_)
        ),
        "metrics": metrics,
        "severe": severe,
        "type_metrics": type_metrics,
        "qjf": qjf,
        "postseal_cache": str(phase_b_root),
    }
    _write_json(phase_b_root / "window_metrics.json", result)
    return result


def _aggregate(windows: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    if len(windows) != EXPECTED_WINDOW_COUNT:
        raise ValueError("Q1 aggregation requires exactly 32 windows")
    metrics = {}
    for name in RISK_NAMES:
        records = [window["metrics"][name] for window in windows]
        metrics[name] = {
            "aurc_macro": float(np.mean([record["aurc"] for record in records])),
            "ause_macro": float(np.mean([record["ause"] for record in records])),
            "spearman_macro": macro_mean([record["spearman"] for record in records]),
            "nonnegative_spearman_windows": sum(
                record["spearman"] is not None and record["spearman"] >= 0
                for record in records
            ),
            "risk_curve_macro": np.mean(
                np.asarray([record["risk_curve"] for record in records]), axis=0
            ).tolist(),
            "oracle_curve_macro": np.mean(
                np.asarray([record["oracle_curve"] for record in records]), axis=0
            ).tolist(),
        }
    best_native = min(NATIVE_RISK_NAMES, key=lambda name: metrics[name]["aurc_macro"])
    metrics["best_native"] = {"name": best_native, **metrics[best_native]}

    thresholds = [str(float(value)) for value in config["severe_error_thresholds_m"]]
    qjf_thresholds = {}
    qjf_beats_native = True
    qjf_defined_windows = EXPECTED_WINDOW_COUNT
    paired_ci: dict[str, Any] = {}
    for threshold in thresholds:
        full_common = [
            window["qjf"][threshold]["supported_qjf_rate"] for window in windows
        ]
        native_common = [
            window["qjf"][threshold]["native_matched_qjf_rate"][best_native]
            for window in windows
        ]
        valid_pairs = [
            (full, native)
            for full, native in zip(full_common, native_common, strict=True)
            if full is not None and native is not None
        ]
        qjf_defined_windows = min(qjf_defined_windows, len(valid_pairs))
        differences = np.asarray([full - native for full, native in valid_pairs])
        ci = (
            paired_bootstrap_mean_ci(
                differences,
                replicates=int(config["bootstrap_replicates"]),
                seed=int(config["seed"]),
            )
            if differences.size
            else None
        )
        paired_ci[f"qjf_full_minus_native_{threshold}"] = ci
        full_macro = macro_mean(full_common)
        native_macro = macro_mean(native_common)
        threshold_pass = (
            full_macro is not None
            and native_macro is not None
            and full_macro < native_macro
            and ci is not None
            and ci["ci95"][1] < 0
        )
        qjf_beats_native &= threshold_pass
        qjf_thresholds[threshold] = {
            "supported_qjf_rate_macro": full_macro,
            "native_matched_qjf_rate_macro": native_macro,
            "best_native_name": best_native,
            "defined_windows": len(valid_pairs),
            "beats_native_with_ci": threshold_pass,
            "shared_error_recall_top20_full_macro": macro_mean(
                [
                    window["qjf"][threshold]["shared_error_recall_top20_full"]
                    for window in windows
                ]
            ),
            "shared_error_recall_top20_native_macro": macro_mean(
                [
                    window["qjf"][threshold]["shared_error_recall_top20_native"]
                    [best_native]
                    for window in windows
                ]
            ),
        }

    comparisons = {
        "full_minus_best_native_aurc": (
            "full",
            best_native,
            "aurc",
        ),
        "full_minus_best_native_ause": ("full", best_native, "ause"),
        "full_minus_same_family_aurc": ("full", "same_family", "aurc"),
        "full_minus_same_family_ause": ("full", "same_family", "ause"),
        "full_minus_random_aurc": ("full", "random", "aurc"),
        "full_minus_no_drop_aurc": ("full", "full_no_drop", "aurc"),
        "full_minus_no_disagreement_aurc": (
            "full",
            "full_no_disagreement",
            "aurc",
        ),
    }
    for key, (first, second, metric) in comparisons.items():
        differences = np.asarray(
            [
                window["metrics"][first][metric] - window["metrics"][second][metric]
                for window in windows
            ]
        )
        paired_ci[key] = paired_bootstrap_mean_ci(
            differences,
            replicates=int(config["bootstrap_replicates"]),
            seed=int(config["seed"]),
        )
    shuffle_minus_drop = np.asarray(
        [
            window["shuffle_response_median"] - window["drop_instability_median"]
            for window in windows
        ]
    )
    paired_ci["shuffle_minus_drop"] = paired_bootstrap_mean_ci(
        shuffle_minus_drop,
        replicates=int(config["bootstrap_replicates"]),
        seed=int(config["seed"]),
    )

    selective_severe_pass = True
    selective_severe = {}
    for threshold in thresholds:
        threshold_summary = {}
        for coverage in (0.20, 0.50):
            coverage_index = list(config["coverage_grid"]).index(coverage)
            full_values = np.asarray(
                [
                    window["severe"][threshold]["full"]["risk_curve"][coverage_index]
                    for window in windows
                ]
            )
            best_values = np.asarray(
                [
                    window["severe"][threshold][best_native]["risk_curve"][coverage_index]
                    for window in windows
                ]
            )
            random_values = np.asarray(
                [
                    window["severe"][threshold]["random"]["risk_curve"][coverage_index]
                    for window in windows
                ]
            )
            best_ci = paired_bootstrap_mean_ci(
                full_values - best_values,
                replicates=int(config["bootstrap_replicates"]),
                seed=int(config["seed"]),
            )
            random_ci = paired_bootstrap_mean_ci(
                full_values - random_values,
                replicates=int(config["bootstrap_replicates"]),
                seed=int(config["seed"]),
            )
            passed = _selective_severe_comparison_passes(
                full=float(np.mean(full_values)),
                best_native=float(np.mean(best_values)),
                random=float(np.mean(random_values)),
                best_ci=best_ci,
                random_ci=random_ci,
                relative_gain=float(config["decision"]["minimum_relative_risk_gain"]),
            )
            selective_severe_pass &= passed
            threshold_summary[str(coverage)] = {
                "full_macro": float(np.mean(full_values)),
                "best_native_macro": float(np.mean(best_values)),
                "random_macro": float(np.mean(random_values)),
                "full_minus_best_ci": best_ci,
                "full_minus_random_ci": random_ci,
                "passed": passed,
            }
        selective_severe[threshold] = threshold_summary

    supported_means = [
        window["type_metrics"]["model-supported"]["mean_m"] for window in windows
    ]
    nonsupported_means = []
    type_defined = 0
    for window in windows:
        other_counts = sum(
            window["type_metrics"][name]["count"] for name in ("uncertain", "conflict")
        )
        if window["type_metrics"]["model-supported"]["count"] and other_counts:
            weighted = sum(
                window["type_metrics"][name]["mean_m"]
                * window["type_metrics"][name]["count"]
                for name in ("uncertain", "conflict")
                if window["type_metrics"][name]["mean_m"] is not None
            ) / other_counts
            nonsupported_means.append(weighted)
            type_defined += 1
        else:
            nonsupported_means.append(None)
    type_pairs = [
        (supported, other)
        for supported, other in zip(supported_means, nonsupported_means, strict=True)
        if supported is not None and other is not None
    ]
    type_ci = (
        paired_bootstrap_mean_ci(
            np.asarray([supported - other for supported, other in type_pairs]),
            replicates=int(config["bootstrap_replicates"]),
            seed=int(config["seed"]),
        )
        if type_pairs
        else None
    )
    type_ordering_pass = type_ci is not None and type_ci["ci95"][1] < 0

    return {
        "window_count": len(windows),
        "comparison_coverage_macro": float(
            np.mean([window["comparison_coverage"] for window in windows])
        ),
        "label_valid_coverage_macro": float(
            np.mean([window["label_valid_coverage"] for window in windows])
        ),
        "supported_coverage_macro": float(
            np.mean([window["supported_coverage_of_comparison"] for window in windows])
        ),
        "supported_nonempty_windows": sum(window["supported_count"] > 0 for window in windows),
        "all_drop_valid_coverage_macro": float(
            np.mean([window["all_drop_valid_coverage_of_comparison"] for window in windows])
        ),
        "drop_instability_median_macro": float(
            np.mean([window["drop_instability_median"] for window in windows])
        ),
        "shuffle_response_median_macro": float(
            np.mean([window["shuffle_response_median"] for window in windows])
        ),
        "metrics": metrics,
        "paired_ci": paired_ci,
        "qjf": {
            "defined_windows": qjf_defined_windows,
            "beats_native_at_both_thresholds": qjf_beats_native,
            "thresholds": qjf_thresholds,
        },
        "selective_severe": selective_severe,
        "selective_severe_pass": selective_severe_pass,
        "type_ordering": {
            "defined_windows": type_defined,
            "supported_minus_other_ci": type_ci,
            "passed": type_ordering_pass,
        },
    }


def _decide(summary: dict[str, Any], decision_config: dict[str, Any]) -> dict[str, Any]:
    metrics = summary.get("metrics", {})
    full = metrics.get("full", {})
    best = metrics.get("best_native", {})
    same_family = metrics.get("same_family", {})
    random = metrics.get("random", {})
    no_drop = metrics.get("full_no_drop", {})
    no_disagreement = metrics.get("full_no_disagreement", {})
    paired = summary.get("paired_ci", {})

    def upper_below_zero(name: str) -> bool:
        value = paired.get(name)
        return (
            isinstance(value, dict)
            and value.get("ci95", [None, None])[1] is not None
            and value["ci95"][1] < 0
        )

    def lower_above_zero(name: str) -> bool:
        value = paired.get(name)
        return (
            isinstance(value, dict)
            and value.get("ci95", [None, None])[0] is not None
            and value["ci95"][0] > 0
        )

    relative_gain = float(decision_config["minimum_relative_risk_gain"])
    ablation_degradation = float(decision_config["minimum_ablation_degradation"])
    gates = {
        "complete_population": summary.get("window_count")
        == int(decision_config["minimum_windows"]),
        "comparison_coverage": summary.get("comparison_coverage_macro", 0.0)
        >= float(decision_config["minimum_comparison_coverage_macro"]),
        "source_drop_operational": summary.get("drop_instability_median_macro", 0.0) > 1e-4,
        "intervention_specificity": (
            summary.get("shuffle_response_median_macro", 0.0)
            > summary.get("drop_instability_median_macro", float("inf"))
            and lower_above_zero("shuffle_minus_drop")
        ),
        "beats_best_native_aurc": (
            full.get("aurc_macro", float("inf"))
            <= (1.0 - relative_gain) * best.get("aurc_macro", -float("inf"))
            and upper_below_zero("full_minus_best_native_aurc")
        ),
        "beats_best_native_ause": (
            full.get("ause_macro", float("inf"))
            <= (1.0 - relative_gain) * best.get("ause_macro", -float("inf"))
            and upper_below_zero("full_minus_best_native_ause")
        ),
        "beats_same_family": (
            full.get("aurc_macro", float("inf"))
            <= (1.0 - relative_gain) * same_family.get("aurc_macro", -float("inf"))
            and full.get("ause_macro", float("inf"))
            <= (1.0 - relative_gain) * same_family.get("ause_macro", -float("inf"))
            and upper_below_zero("full_minus_same_family_aurc")
            and upper_below_zero("full_minus_same_family_ause")
        ),
        "beats_random": (
            full.get("aurc_macro", float("inf"))
            <= (1.0 - relative_gain) * random.get("aurc_macro", -float("inf"))
            and upper_below_zero("full_minus_random_aurc")
        ),
        "component_necessity": (
            no_drop.get("aurc_macro", -float("inf"))
            >= (1.0 + ablation_degradation) * full.get("aurc_macro", float("inf"))
            and no_disagreement.get("aurc_macro", -float("inf"))
            >= (1.0 + ablation_degradation) * full.get("aurc_macro", float("inf"))
            and upper_below_zero("full_minus_no_drop_aurc")
            and upper_below_zero("full_minus_no_disagreement_aurc")
        ),
        "positive_risk_error_ordering": (
            full.get("spearman_macro") is not None
            and full["spearman_macro"]
            >= float(decision_config["minimum_spearman_macro"])
            and full.get("nonnegative_spearman_windows", 0)
            >= int(decision_config["minimum_nonnegative_spearman_windows"])
        ),
        "selective_severe_error": bool(summary.get("selective_severe_pass", False)),
        "quiet_joint_failure": (
            summary.get("qjf", {}).get("defined_windows", 0)
            >= int(decision_config["minimum_qjf_defined_windows"])
            and bool(summary.get("qjf", {}).get("beats_native_at_both_thresholds", False))
        ),
        "type_ordering": (
            summary.get("type_ordering", {}).get("defined_windows", 0)
            >= int(decision_config["minimum_type_ordering_defined_windows"])
            and bool(summary.get("type_ordering", {}).get("passed", False))
        ),
    }
    passed = all(gates.values())
    return {
        "status": "GO_V8_Q1_RISK_QUALIFICATION" if passed else "STOP_V8_Q1_RISK_STATE",
        "gates": gates,
        "failed_gates": [name for name, value in gates.items() if not value],
        "claim_ceiling": (
            "development_train_supplier_risk_qualification_only_no_state_or_cvpr_claim"
        ),
    }


def _initial_record(config_path: Path, config: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": Q1_SCHEMA_VERSION,
        "status": "RUNNING",
        "config_path": str(config_path),
        "config_sha256": _sha256_file(config_path),
        "config": config,
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "research_boundary": {
            "partition": "development-train",
            "proposal_phase_label_access": False,
            "labels_opened_after_all_risk_seals": False,
            "training": False,
            "target_rgb_access": False,
            "sim3_or_scale_fit": False,
            "supplier_writes_to_typed_state": False,
            "point_fusion": False,
            "geometry_action": "emvsnet_full_anchor_only",
            "pose_perturbation_in_primary_risk": False,
        },
        "checks": {},
    }


def _configure_determinism(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _load_request(manifest: SceneManifest, frame_ids: Sequence[int]) -> GeometrySupplierRequest:
    by_id = {frame.frame_id: frame for frame in manifest.frames}
    try:
        frames = [by_id[int(frame_id)] for frame_id in frame_ids]
    except KeyError as error:
        raise ValueError(f"frozen frame ID is absent from manifest: {error}") from error
    rgb = []
    intrinsics = []
    c2w = []
    for frame in frames:
        rgb.append(
            torch.from_numpy(
                np.asarray(
                    Image.open(manifest.root / frame.rgb).convert("RGB"), dtype=np.float32
                ).copy()
            ).permute(2, 0, 1)
            / 255.0
        )
        intrinsics.append(torch.tensor(frame.intrinsics, dtype=torch.float32))
        c2w.append(torch.tensor(frame.c2w, dtype=torch.float32))
    return GeometrySupplierRequest(
        tuple(frame.frame_id for frame in frames),
        torch.stack(rgb),
        Cameras(torch.stack(intrinsics), torch.stack(c2w), manifest.image_size),
    )


def _select_request(
    request: GeometrySupplierRequest, keep: Sequence[int]
) -> GeometrySupplierRequest:
    indices = torch.tensor(tuple(keep), dtype=torch.long)
    return GeometrySupplierRequest(
        tuple(request.frame_ids[index] for index in keep),
        request.rgb.index_select(0, indices),
        request.cameras.select_views(indices),
    )


def _shuffled_source_request(
    request: GeometrySupplierRequest, derangement: Sequence[int]
) -> GeometrySupplierRequest:
    if len(derangement) != 3:
        raise ValueError("source derangement must contain three positions")
    rgb = request.rgb.clone()
    rgb[1:] = request.rgb[torch.tensor(derangement, dtype=torch.long)]
    return GeometrySupplierRequest(request.frame_ids, rgb, request.cameras)


def _check_request(request: GeometrySupplierRequest) -> None:
    if request.rgb.shape != (4, 3, 128, 160):
        raise ValueError("Q1 requires four 128x160 context RGB views")


def _check_proposal(proposal: V8Proposal, request: GeometrySupplierRequest) -> None:
    canonical = request.canonical()
    if proposal.frame_ids != canonical.frame_ids or proposal.image_size != (128, 160):
        raise ValueError("proposal does not match canonical Q1 request")
    if not np.allclose(
        proposal.intrinsics,
        canonical.cameras.intrinsics.detach().cpu().numpy(),
        atol=1e-5,
        rtol=1e-5,
    ):
        raise ValueError("proposal intrinsics do not match Q1 request")
    if not np.allclose(
        proposal.c2w,
        canonical.cameras.c2w.detach().cpu().numpy(),
        atol=1e-5,
        rtol=1e-5,
    ):
        raise ValueError("proposal cameras do not match Q1 request")


def _check_drop_anchor_contract(
    full_request: GeometrySupplierRequest,
    drop_request: GeometrySupplierRequest,
    proposal: V8Proposal,
) -> None:
    full = full_request.canonical()
    drop = drop_request.canonical()
    if drop.frame_ids[0] != full.frame_ids[0] or proposal.frame_ids[0] != full.frame_ids[0]:
        raise ValueError("drop proposal anchor frame does not match the full proposal anchor")
    full_k = full.cameras.intrinsics[0].detach().cpu().numpy()
    full_c2w = full.cameras.c2w[0].detach().cpu().numpy()
    drop_k = drop.cameras.intrinsics[0].detach().cpu().numpy()
    drop_c2w = drop.cameras.c2w[0].detach().cpu().numpy()
    if not (
        np.array_equal(drop_k, full_k)
        and np.array_equal(drop_c2w, full_c2w)
        and np.array_equal(proposal.intrinsics[0], full_k)
        and np.array_equal(proposal.c2w[0], full_c2w)
    ):
        raise ValueError("drop proposal anchor camera must exactly match the full anchor camera")


def _sample_grid(value: np.ndarray, stride: int) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim < 2 or array.shape[:2] != (128, 160):
        raise ValueError("Q1 grid sampling requires leading shape 128x160")
    return array[::stride, ::stride].copy()


def _require_all_phase_a_sealed(phase_a: list[dict[str, Any]]) -> None:
    if len(phase_a) != EXPECTED_WINDOW_COUNT:
        raise ValueError("Q1 must seal all 32 windows before labels")
    for record in phase_a:
        read_v8_risk_cache(record["risk_cache"])
        for name in ("unidepth_cache", "emvsnet_cache", "shuffle_cache"):
            read_v8_proposal_cache(record[name])
        for path in record["drop_caches"]:
            read_v8_proposal_cache(path)


def _sealed_at(cache: Path) -> str:
    payload = json.loads((cache / "metadata.json").read_text(encoding="utf-8"))
    value = payload.get("sealed_at_utc")
    if not isinstance(value, str):
        raise ValueError("sealed cache is missing its timestamp")
    return value


def _stable_low_risk_order(risk: np.ndarray, valid: np.ndarray) -> np.ndarray:
    flat_indices = np.flatnonzero(valid & np.isfinite(risk))
    order = np.lexsort((flat_indices, risk.flat[flat_indices]))
    return flat_indices[order]


def _masked_mean(values: np.ndarray, mask: np.ndarray) -> float | None:
    selected = np.asarray(values)[np.asarray(mask, dtype=np.bool_)]
    return None if selected.size == 0 else float(np.mean(selected))


def _quiet_joint_failure_masks(
    emv_error: np.ndarray,
    uni_error: np.ndarray,
    label_valid: np.ndarray,
    supported: np.ndarray,
    *,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    shared = (
        np.asarray(label_valid, dtype=np.bool_)
        & (np.asarray(emv_error) > threshold)
        & (np.asarray(uni_error) > threshold)
    )
    quiet = shared & np.asarray(supported, dtype=np.bool_)
    return quiet, shared


def _selective_severe_comparison_passes(
    *,
    full: float,
    best_native: float,
    random: float,
    best_ci: dict[str, Any],
    random_ci: dict[str, Any],
    relative_gain: float,
) -> bool:
    return bool(
        full <= (1.0 - relative_gain) * best_native
        and full <= (1.0 - relative_gain) * random
        and best_ci["ci95"][1] < 0
        and random_ci["ci95"][1] < 0
    )


def _masked_median(values: np.ndarray, mask: np.ndarray) -> float:
    selected = np.asarray(values)[np.asarray(mask, dtype=np.bool_)]
    if selected.size == 0:
        raise ValueError("window metric requires at least one valid value")
    return float(np.median(selected))


def _error_summary(errors: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    selected = np.asarray(errors, dtype=np.float64)[np.asarray(mask, dtype=np.bool_)]
    if selected.size == 0:
        return {"count": 0, "mean_m": None, "median_m": None, "p90_m": None}
    return {
        "count": int(selected.size),
        "mean_m": float(np.mean(selected)),
        "median_m": float(np.median(selected)),
        "p90_m": float(np.quantile(selected, 0.90, method="linear")),
    }


def _high_risk_recall(
    risk: np.ndarray,
    event: np.ndarray,
    valid: np.ndarray,
    *,
    fraction: float,
) -> float | None:
    mask = np.asarray(valid, dtype=np.bool_) & np.isfinite(risk)
    event = np.asarray(event, dtype=np.bool_) & mask
    if event.sum() == 0:
        return None
    indices = np.flatnonzero(mask)
    order = np.lexsort((indices, -np.asarray(risk).flat[indices]))
    retained = max(1, int(np.ceil(fraction * indices.size)))
    selected = indices[order[:retained]]
    return float(np.asarray(event).flat[selected].sum() / event.sum())


def _write_postseal_cache(
    destination: Path,
    arrays: dict[str, np.ndarray],
    *,
    metadata: dict[str, Any],
) -> None:
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite post-seal cache: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=str(destination.parent))
    )
    try:
        arrays_path = temporary / "arrays.npz"
        np.savez_compressed(arrays_path, **arrays)
        payload = {
            "schema_version": "mcss.v8_q1_postseal.v1",
            "arrays_file": arrays_path.name,
            "arrays_sha256": _sha256_file(arrays_path),
            "metadata": metadata,
        }
        metadata_path = temporary / "metadata.json"
        metadata_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        (temporary / "SEAL.json").write_text(
            json.dumps(
                {
                    "metadata_sha256": _sha256_file(metadata_path),
                    "arrays_sha256": _sha256_file(arrays_path),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        shutil.move(str(temporary), str(destination))
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _render_report(record: dict[str, Any]) -> str:
    lines = [
        "# V8 Q1 Supplier-Risk Qualification",
        "",
        f"- Final status: `{record['status']}`",
        f"- Schema: `{record['schema_version']}`",
        "- Scope: 32 frozen development-train anchor views; all context-only proposals and risk "
        "fields were sealed before context depth opened.",
        "- Geometry action: EMVSNet full-view anchor proposal only. UniDepth and source-drop paths "
        "supply risk evidence; no point fusion, training, target RGB, or typed-state write.",
        "",
    ]
    if "error" in record:
        lines.extend(
            [
                "## Runtime Failure",
                "",
                f"`{record['error']['type']}: {record['error']['message']}`",
                "",
            ]
        )
    decision = record.get("decision")
    summary = record.get("summary")
    if isinstance(decision, dict) and isinstance(summary, dict):
        lines.extend(["## Decision Gates", ""])
        for name, passed in decision["gates"].items():
            lines.append(f"- `{name}`: {'PASS' if passed else 'FAIL'}")
        best = summary["metrics"]["best_native"]
        full = summary["metrics"]["full"]
        same = summary["metrics"]["same_family"]
        lines.extend(
            [
                "",
                "## Primary Metrics",
                "",
                f"- Comparison coverage: `{summary['comparison_coverage_macro']:.6f}`",
                f"- Model-supported coverage: `{summary['supported_coverage_macro']:.6f}`",
                "- AURC definition: empirical mean selective error over every retained count "
                "from 1/N through N/N.",
                f"- Full AURC / AUSE: `{full['aurc_macro']:.6f}` / `{full['ause_macro']:.6f}`",
                f"- Best native supplier ({best['name']}) AURC / AUSE: "
                f"`{best['aurc_macro']:.6f}` / `{best['ause_macro']:.6f}`",
                f"- Same-family AURC / AUSE: `{same['aurc_macro']:.6f}` / "
                f"`{same['ause_macro']:.6f}`",
                f"- Full risk-error Spearman: `{full['spearman_macro']}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Interpretation Boundary",
            "",
            "A GO would qualify a development-train risk-ranking mechanism only. It would not "
            "establish a measurement, correct geometry, state improvement, generalization, or "
            "CVPR-ready result. A STOP seals this exact risk rule; thresholds and windows must not "
            "be changed after labels were opened to rescue it.",
            "",
            "Formal Research Institute: not invoked. Evaluation Council: not convened. The two "
            "existing medium-intensity research consultants supplied framing and mechanism "
            "reviews; this report records executable evidence from the implemented audit.",
            "",
        ]
    )
    return "\n".join(lines)
