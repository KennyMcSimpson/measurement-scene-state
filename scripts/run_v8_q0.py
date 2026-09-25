"""Run the frozen V8 Q0 supplier qualification on one development-train window."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
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
    classify_dual_proposals,
    ray_distance_to_depth_z,
    write_v8_proposal_cache,
)
from mcss.v8_suppliers import (
    EMVSNET_DEPTH_FAR_M,
    EMVSNET_DEPTH_NEAR_M,
    EMVSNetSupplier,
    UniDepthV2SmallSupplier,
)

Q0_SCHEMA_VERSION = "mcss.v8_q0.v3"
REPEAT_TOLERANCE_M = 1e-5
PEAK_MEMORY_LIMIT_BYTES = 10 * 1024**3


def main() -> int:
    args = _parse_args()
    determinism = _configure_determinism()
    output = Path(args.output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty Q0 output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    record = _initial_record(args)
    record["runtime"]["determinism"] = determinism
    _write_json(output / "q0_running.json", record)
    try:
        window = _load_window(args.window_file)
        manifest = load_manifest(window["manifest"])
        frames = _select_frames(manifest, window["frame_ids"])
        request = _load_request(manifest, frames)
        record["window"] = {
            "scene_id": window["scene_id"],
            "frame_ids": list(request.frame_ids),
            "manifest": str(Path(window["manifest"]).resolve()),
            "image_size": list(request.cameras.image_size),
        }
        _check_input_contract(request)
        record["checks"]["input_contract"] = {"status": "PASS"}

        uni = UniDepthV2SmallSupplier(
            code_root=args.unidepth_code_root,
            local_model_dir=args.unidepth_bundle,
            device=args.device,
        )
        emv = EMVSNetSupplier(
            code_root=args.emvsnet_code_root,
            checkpoint=args.emvsnet_checkpoint,
            device=args.device,
            image_scale=0.25,
            depth_bins=64,
            max_h=512,
            max_w=640,
            evidential_method=args.evidential_method,
        )
        checkpoint_state = _check_model_loads(uni, emv)
        record["checks"]["model_load"] = {"status": "PASS"}
        provenance = _checkpoint_method_provenance(
            checkpoint_state,
            requested_method=args.evidential_method,
            checkpoint_path=args.emvsnet_checkpoint,
            provenance_path=args.emvsnet_provenance,
            model_card_path=args.emvsnet_model_card,
            parameter_names=[name for name, _parameter in emv._load().named_parameters()],
        )
        record["checks"]["checkpoint_method_provenance"] = provenance

        case_specs = [("four_view", request, [0, 1, 2, 3])]
        for drop in (1, 2, 3):
            keep = [index for index in range(4) if index != drop]
            case_specs.append((f"drop_source_{drop}", _select_request(request, keep), keep))

        # Phase A is completely label blind. Every proposal is sealed before any depth path opens.
        sealed_cases = []
        for name, case_request, keep in case_specs:
            sealed_cases.append(
                _run_and_seal_case(
                    output / "cases" / name,
                    name,
                    case_request,
                    keep,
                    uni,
                    emv,
                )
            )
        record["checks"]["all_proposals_sealed_before_context_depth"] = {"status": "PASS"}

        repeat = _check_repeat_stability(output, request, uni, emv)
        record["checks"]["repeat_stability"] = {
            "status": "PASS" if repeat["passed"] else "FAIL",
            **repeat,
        }
        if not repeat["passed"]:
            raise RuntimeError("STOP_EMVSNET_Q0: repeated supplier inference is not stable")

        full_emv = sealed_cases[0]["emvsnet_proposal"]
        interventions: dict[str, dict[str, Any]] = {}
        for case in sealed_cases[1:]:
            interventions[case["name"]] = _source_drop_intervention_delta(
                full_emv, case["emvsnet_proposal"]
            )
        _write_json(output / "source_drop_interventions.json", interventions)
        record["checks"]["source_drop_interventions"] = {
            "status": "PASS",
            "cases": interventions,
        }

        # Phase B may now open development context labels for descriptive checks only.
        access_log: list[dict[str, Any]] = []
        case_results = []
        for case in sealed_cases:
            context_depth = _load_context_depth(
                case["request"], args.window_file, access_log=access_log
            )
            intervention = interventions.get(case["name"])
            intervention_delta = (
                None if intervention is None else intervention["median_depth_delta_m"]
            )
            case_results.append(
                _finalize_case(
                    case,
                    context_depth,
                    intervention_delta_m=intervention_delta,
                )
            )
        record["cases"] = case_results
        _write_json(output / "access_log.json", access_log)
        record["access_log"] = {
            "path": str(output / "access_log.json"),
            "opened_after_all_proposal_seals": all(
                entry["opened_after_all_proposal_seals"] for entry in access_log
            ),
            "context_depth_file_count": len(access_log),
        }

        peak = (
            int(torch.cuda.max_memory_allocated())
            if torch.cuda.is_available() and args.device.startswith("cuda")
            else None
        )
        record["runtime"]["cuda_peak_memory_bytes"] = peak
        peak_status = "PASS" if peak is None or peak <= PEAK_MEMORY_LIMIT_BYTES else "FAIL"
        record["checks"]["peak_memory"] = {
            "status": peak_status,
            "bytes": peak,
            "limit_bytes": PEAK_MEMORY_LIMIT_BYTES,
        }
        if peak_status == "FAIL":
            raise RuntimeError("STOP_EMVSNET_Q0: peak CUDA memory exceeded 10 GiB")

        record["mechanical_qualification"] = {
            "status": "PASS",
            "meaning": "both suppliers completed frozen forward and protocol checks",
        }
        _require_checkpoint_method_provenance(provenance)
        record["status"] = "GO_TO_Q1_LABEL_BLIND_AUDIT"
    except Exception as error:
        record["status"] = "STOP_EMVSNET_Q0"
        record["error"] = {"type": type(error).__name__, "message": str(error)}
    record["finished_seconds"] = time.time() - started
    _write_json(output / "q0_result.json", record)
    (output / "Q0_REPORT.md").write_text(_render_report(record), encoding="utf-8")
    try:
        (output / "q0_running.json").unlink()
    except FileNotFoundError:
        pass
    print(json.dumps({"status": record["status"], "output": str(output)}, sort_keys=True))
    return 0 if record["status"] == "GO_TO_Q1_LABEL_BLIND_AUDIT" else 2


def _initial_record(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "schema_version": Q0_SCHEMA_VERSION,
        "status": "RUNNING",
        "research_boundary": {
            "partition": "development-train",
            "window": "window_000",
            "proposal_phase_target_access": False,
            "postseal_context_depth_audit": True,
            "training": False,
            "sim3_or_scale_fit": False,
            "threshold_selection_from_labels": False,
            "supplier_writes_to_typed_state": False,
        },
        "runtime": _runtime_info(),
        "config": {
            "window_file": str(Path(args.window_file).resolve()),
            "unidepth_code_root": str(Path(args.unidepth_code_root).resolve()),
            "emvsnet_code_root": str(Path(args.emvsnet_code_root).resolve()),
            "unidepth_bundle": str(Path(args.unidepth_bundle).resolve()),
            "emvsnet_checkpoint": str(Path(args.emvsnet_checkpoint).resolve()),
            "emvsnet_provenance": (
                None
                if args.emvsnet_provenance is None
                else str(Path(args.emvsnet_provenance).resolve())
            ),
            "emvsnet_model_card": (
                None
                if args.emvsnet_model_card is None
                else str(Path(args.emvsnet_model_card).resolve())
            ),
            "device": args.device,
            "image_size": [128, 160],
            "depth_bins": 64,
            "depth_range_m": [EMVSNET_DEPTH_NEAR_M, EMVSNET_DEPTH_FAR_M],
            "depth_range_provenance": "pre_registered_protocol_fixed_not_label_derived",
            "evidential_method": args.evidential_method,
        },
        "checks": {},
        "cases": [],
    }


def _configure_determinism() -> dict[str, Any]:
    """Freeze CUDA algorithm choices so repeatability is an executable contract."""

    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return {
        "torch_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--window-file", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--unidepth-code-root", required=True, type=Path)
    parser.add_argument("--unidepth-bundle", required=True, type=Path)
    parser.add_argument("--emvsnet-code-root", required=True, type=Path)
    parser.add_argument("--emvsnet-checkpoint", required=True, type=Path)
    parser.add_argument("--emvsnet-provenance", type=Path)
    parser.add_argument("--emvsnet-model-card", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--evidential-method", choices=("der", "sder"), default="der")
    return parser.parse_args()


def _load_window(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("partition") != "train" or len(payload.get("windows", [])) != 1:
        raise ValueError("Q0 requires exactly one frozen development-train window")
    window = payload["windows"][0]
    if window.get("scene_id") is None or len(window.get("frame_ids", [])) != 4:
        raise ValueError("Q0 window must contain exactly four frame IDs")
    if any(
        token in str(window.get("manifest", "")).lower()
        for token in ("diagnostic_test", "final_holdout")
    ):
        raise ValueError("Q0 cannot access diagnostic-test or final-holdout manifests")
    return window


def _select_frames(manifest: SceneManifest, frame_ids: list[int]) -> list[Any]:
    by_id = {frame.frame_id: frame for frame in manifest.frames}
    try:
        return [by_id[int(frame_id)] for frame_id in frame_ids]
    except KeyError as error:
        raise ValueError(f"frozen frame ID is absent from manifest: {error}") from error


def _load_request(manifest: SceneManifest, frames: list[Any]) -> GeometrySupplierRequest:
    rgb = []
    intrinsics = []
    c2w = []
    for frame in frames:
        rgb.append(
            torch.from_numpy(
                np.asarray(Image.open(manifest.root / frame.rgb).convert("RGB"), dtype=np.float32)
                / 255.0
            ).permute(2, 0, 1)
        )
        intrinsics.append(torch.tensor(frame.intrinsics, dtype=torch.float32))
        c2w.append(torch.tensor(frame.c2w, dtype=torch.float32))
    return GeometrySupplierRequest(
        tuple(frame.frame_id for frame in frames),
        torch.stack(rgb),
        Cameras(torch.stack(intrinsics), torch.stack(c2w), manifest.image_size),
    )


def _select_request(request: GeometrySupplierRequest, keep: list[int]) -> GeometrySupplierRequest:
    indices = torch.tensor(keep, dtype=torch.long)
    return GeometrySupplierRequest(
        tuple(request.frame_ids[index] for index in keep),
        request.rgb.index_select(0, indices),
        request.cameras.select_views(indices),
    )


def _check_input_contract(request: GeometrySupplierRequest) -> None:
    if request.rgb.shape[-2:] != (128, 160):
        raise ValueError("Q0 frozen input must be 128x160")
    if not torch.isfinite(request.rgb).all():
        raise ValueError("input RGB contains nonfinite values")


def _check_model_loads(
    uni: UniDepthV2SmallSupplier, emv: EMVSNetSupplier
) -> dict[str, Any]:
    uni._load()
    emv._load()
    state = torch.load(emv.checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(state, dict):
        raise ValueError("EMVSNet checkpoint payload must be a mapping")
    return state


def _checkpoint_method_provenance(
    checkpoint_state: dict[str, Any],
    *,
    requested_method: str,
    checkpoint_path: Path | None = None,
    provenance_path: Path | None = None,
    model_card_path: Path | None = None,
    parameter_names: list[str] | None = None,
) -> dict[str, Any]:
    recorded = checkpoint_state.get("evidential_method") or checkpoint_state.get("training_method")
    if recorded is not None and str(recorded).lower() != requested_method:
        return {
            "status": "FAIL",
            "reason": "training_method_mismatch",
            "requested_method": requested_method,
            "checkpoint_method": str(recorded),
        }
    if recorded is not None:
        return {
            "status": "PASS",
            "reason": "training_method_verified_inside_checkpoint",
            "requested_method": requested_method,
            "checkpoint_method": str(recorded),
            "checkpoint_internal_method": str(recorded),
            "evidence_kind": "checkpoint_internal_training_metadata",
            "claim_ceiling": "training_method_recorded_by_checkpoint",
        }

    base = {
        "requested_method": requested_method,
        "checkpoint_method": None,
        "checkpoint_internal_method": None,
        "checkpoint_keys": sorted(str(key) for key in checkpoint_state),
    }
    if checkpoint_path is None or provenance_path is None or model_card_path is None:
        return {"status": "FAIL", "reason": "training_method_unverified", **base}
    try:
        manifest = json.loads(Path(provenance_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return {
            "status": "FAIL",
            "reason": "publisher_provenance_unreadable",
            "detail": str(error),
            **base,
        }
    if manifest.get("schema_version") != "mcss.emvsnet_checkpoint_provenance.v1":
        return {"status": "FAIL", "reason": "publisher_provenance_schema_mismatch", **base}
    checkpoint = manifest.get("checkpoint")
    training = manifest.get("training")
    model_card = manifest.get("model_card")
    if not all(isinstance(value, dict) for value in (checkpoint, training, model_card)):
        return {"status": "FAIL", "reason": "publisher_provenance_incomplete", **base}
    assert isinstance(checkpoint, dict)
    assert isinstance(training, dict)
    assert isinstance(model_card, dict)
    actual_checkpoint = Path(checkpoint_path).resolve()
    actual_model_card = Path(model_card_path).resolve()
    if checkpoint.get("filename") != actual_checkpoint.name:
        return {"status": "FAIL", "reason": "checkpoint_filename_mismatch", **base}
    if checkpoint.get("sha256") != _sha256_file(actual_checkpoint):
        return {"status": "FAIL", "reason": "checkpoint_sha256_mismatch", **base}
    if model_card.get("sha256") != _sha256_file(actual_model_card):
        return {"status": "FAIL", "reason": "model_card_sha256_mismatch", **base}
    required_fragments = model_card.get("required_fragments")
    if not isinstance(required_fragments, list) or not required_fragments:
        return {"status": "FAIL", "reason": "model_card_fragments_missing", **base}
    model_card_text = actual_model_card.read_text(encoding="utf-8")
    if any(str(fragment) not in model_card_text for fragment in required_fragments):
        return {"status": "FAIL", "reason": "model_card_contract_mismatch", **base}
    released_method = str(training.get("evidential_method", "")).lower()
    evidence_kind = str(training.get("evidence_kind", ""))
    if released_method != requested_method:
        return {"status": "FAIL", "reason": "publisher_method_mismatch", **base}
    if evidence_kind != "publisher_release_inference_contract":
        return {"status": "FAIL", "reason": "publisher_evidence_kind_unsupported", **base}
    optimizer_signature = _optimizer_method_signature(checkpoint_state, parameter_names)
    warnings: list[dict[str, str]] = []
    if optimizer_signature.get("pattern") == "low_alpha_second_moment":
        warnings.append(
            {
                "code": "optimizer_alpha_moment_imbalance",
                "severity": "WARNING",
                "message": (
                    "Independent alpha channels have much smaller Adam second moments than peer "
                    "channels. This is an auditable training trace, but it cannot identify DER "
                    "versus SDER and does not override the hash-pinned publisher inference "
                    "contract."
                ),
            }
        )
    return {
        "status": "PASS",
        "reason": "inference_method_verified_by_publisher_release_contract",
        "requested_method": requested_method,
        "checkpoint_method": None,
        "checkpoint_internal_method": None,
        "publisher_inference_method": released_method,
        "checkpoint_keys": base["checkpoint_keys"],
        "evidence_kind": evidence_kind,
        "claim_ceiling": "publisher_prescribed_inference_method_not_checkpoint_training_log",
        "checkpoint_sha256": str(checkpoint["sha256"]),
        "model_card_sha256": str(model_card["sha256"]),
        "hub_model_id": str(checkpoint.get("hub_model_id", "")),
        "hub_revision": str(checkpoint.get("hub_revision", "")),
        "provenance_manifest": str(Path(provenance_path).resolve()),
        "model_card": str(actual_model_card),
        "optimizer_signature": optimizer_signature,
        "warnings": warnings,
    }


def _optimizer_method_signature(
    checkpoint_state: dict[str, Any], parameter_names: list[str] | None
) -> dict[str, Any]:
    """Audit alpha-channel Adam moments without inferring the training method.

    Small second moments are compatible with an unused channel, but also with saturation, branch
    weighting, loss scaling, or other training dynamics. The trace is therefore diagnostic only.
    """

    optimizer = checkpoint_state.get("optimizer")
    if not isinstance(optimizer, dict) or parameter_names is None:
        return {"status": "UNAVAILABLE", "reason": "optimizer_or_parameter_names_missing"}
    groups = optimizer.get("param_groups")
    states = optimizer.get("state")
    if not isinstance(groups, list) or not isinstance(states, dict):
        return {"status": "UNAVAILABLE", "reason": "optimizer_state_incomplete"}
    parameter_ids = [parameter_id for group in groups for parameter_id in group.get("params", [])]
    if len(parameter_ids) != len(parameter_names):
        return {"status": "UNAVAILABLE", "reason": "optimizer_parameter_order_unverified"}

    heads: list[dict[str, Any]] = []
    for name, parameter_id in zip(parameter_names, parameter_ids, strict=True):
        if not (
            name.startswith("evidential.classif")
            and name.endswith(".2.weight")
        ):
            continue
        state = states.get(parameter_id)
        if not isinstance(state, dict):
            continue
        second_moment = state.get("exp_avg_sq")
        if not isinstance(second_moment, torch.Tensor) or second_moment.ndim < 1:
            continue
        if second_moment.shape[0] != 4 or not torch.isfinite(second_moment).all():
            continue
        channel_sums = second_moment.detach().double().reshape(4, -1).sum(dim=1)
        other_reference = torch.median(channel_sums[[0, 1, 3]])
        heads.append(
            {
                "parameter": name,
                "second_moment_sums": {
                    label: float(channel_sums[index])
                    for index, label in enumerate(("gamma", "nu", "alpha", "beta"))
                },
                "other_channel_reference": float(other_reference),
                "alpha_to_other_ratio": (
                    None
                    if other_reference <= 0
                    else float(channel_sums[2] / other_reference)
                ),
            }
        )
    positive_references = [
        float(head["other_channel_reference"])
        for head in heads
        if float(head["other_channel_reference"]) > 0
    ]
    if not positive_references:
        return {"status": "UNAVAILABLE", "reason": "evidential_head_moments_missing"}
    active_floor = max(positive_references) * 1e-8
    active_heads = [
        head
        for head in heads
        if float(head["other_channel_reference"]) >= active_floor
    ]
    for head in heads:
        head["active"] = head in active_heads
    ratios = [float(head["alpha_to_other_ratio"]) for head in active_heads]
    low_alpha_pattern = len(active_heads) >= 2 and max(ratios) <= 1e-6
    return {
        "status": "AVAILABLE",
        "diagnostic_semantics": (
            "optimizer second moments only; records relative alpha-channel update scale but "
            "cannot identify the evidential training method"
        ),
        "active_head_rule": "other-channel reference >= 1e-8 of maximum head reference",
        "low_alpha_pattern_rule": "at least two active heads and max alpha/other ratio <= 1e-6",
        "active_head_count": len(active_heads),
        "pattern": (
            "low_alpha_second_moment" if low_alpha_pattern else "no_low_alpha_second_moment"
        ),
        "method_inference": "INDETERMINATE",
        "hard_gate_eligible": False,
        "heads": heads,
    }


def _require_checkpoint_method_provenance(provenance: dict[str, Any]) -> None:
    if provenance.get("status") != "PASS":
        raise RuntimeError(f"STOP_EMVSNET_Q0: {provenance.get('reason', 'unknown_provenance')}")


def _run_and_seal_case(
    case_dir: Path,
    name: str,
    request: GeometrySupplierRequest,
    keep: list[int],
    uni: UniDepthV2SmallSupplier,
    emv: EMVSNetSupplier,
) -> dict[str, Any]:
    case_dir.mkdir(parents=True, exist_ok=True)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    started = time.time()
    uni_proposal = uni.infer(request)
    _check_proposal(uni_proposal, request)
    uni_cache = write_v8_proposal_cache(
        case_dir / "unidepth", uni_proposal, metadata={"case": name, "source_indices": keep}
    )
    emv_proposal = emv.infer(request)
    _check_proposal(emv_proposal, request)
    emv_cache = write_v8_proposal_cache(
        case_dir / "emvsnet", emv_proposal, metadata={"case": name, "source_indices": keep}
    )
    seal_times = {
        "unidepth": _read_seal_time(uni_cache),
        "emvsnet": _read_seal_time(emv_cache),
    }
    return {
        "case_dir": case_dir,
        "name": name,
        "request": request,
        "source_indices": keep,
        "unidepth_proposal": uni_proposal,
        "emvsnet_proposal": emv_proposal,
        "unidepth_cache": uni_cache,
        "emvsnet_cache": emv_cache,
        "seal_times_utc": seal_times,
        "forward_seconds": time.time() - started,
        "cuda_peak_memory_bytes": (
            int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None
        ),
    }


def _finalize_case(
    case: dict[str, Any],
    context_depth: np.ndarray,
    *,
    intervention_delta_m: float | None,
) -> dict[str, Any]:
    uni = case["unidepth_proposal"]
    emv = case["emvsnet_proposal"]
    posthoc = _posthoc_depth_check(uni, emv, context_depth)
    status = classify_dual_proposals(
        uni,
        emv,
        disagreement_threshold_m=0.25,
        intervention_delta_m=intervention_delta_m,
    )
    metric = {
        "case": case["name"],
        "source_indices": case["source_indices"],
        "frame_ids": list(case["request"].frame_ids),
        "unidepth_cache": str(case["unidepth_cache"]),
        "emvsnet_cache": str(case["emvsnet_cache"]),
        "cache_seal_times_utc": case["seal_times_utc"],
        "dual_status": status,
        "posthoc_context_depth_check": posthoc,
        "finite_outputs": True,
        "forward_seconds": case["forward_seconds"],
        "cuda_peak_memory_bytes": case["cuda_peak_memory_bytes"],
    }
    _write_json(case["case_dir"] / "case.json", metric)
    return metric


def _check_proposal(proposal: V8Proposal, request: GeometrySupplierRequest) -> None:
    canonical = request.canonical()
    if proposal.frame_ids != canonical.frame_ids:
        raise ValueError("proposal frame IDs do not match canonical request")
    if proposal.image_size != canonical.cameras.image_size:
        raise ValueError("proposal spatial size does not match request")
    if not np.isfinite(proposal.depth_z_m).all() or not np.isfinite(proposal.points_cam_m).all():
        raise ValueError("proposal contains nonfinite values")
    valid = proposal.valid_mask
    if valid.any() and (proposal.depth_z_m[valid] <= 0).any():
        raise ValueError("proposal has nonpositive valid depth")
    expected_intrinsics = canonical.cameras.intrinsics.detach().cpu().numpy()
    expected_c2w = canonical.cameras.c2w.detach().cpu().numpy()
    if not np.allclose(proposal.intrinsics, expected_intrinsics, atol=1e-5, rtol=1e-5):
        raise ValueError("proposal intrinsics do not match canonical request")
    if not np.allclose(proposal.c2w, expected_c2w, atol=1e-5, rtol=1e-5):
        raise ValueError("proposal c2w does not match canonical request")
    if valid.any() and not np.allclose(
        proposal.points_cam_m[..., 2][valid],
        proposal.depth_z_m[valid],
        atol=1e-4,
        rtol=1e-4,
    ):
        raise ValueError("proposal point z coordinate does not match depth")


def _load_context_depth(
    request: GeometrySupplierRequest,
    window_file: Path,
    *,
    access_log: list[dict[str, Any]] | None = None,
) -> np.ndarray:
    payload = _load_window(window_file)
    manifest = load_manifest(payload["manifest"])
    by_id = {frame.frame_id: frame for frame in manifest.frames}
    depths = []
    for frame_id in request.frame_ids:
        frame = by_id[frame_id]
        path = (manifest.root / frame.depth).resolve()
        opened_at = datetime.now(UTC).isoformat()
        depths.append(np.load(path, allow_pickle=False).astype(np.float32))
        if access_log is not None:
            access_log.append(
                {
                    "frame_id": frame_id,
                    "path": str(path),
                    "opened_at_utc": opened_at,
                    "opened_after_all_proposal_seals": True,
                    "purpose": "development_context_posthoc_descriptive_only",
                }
            )
    return np.stack(depths)


def _posthoc_depth_check(
    uni: V8Proposal, emv: V8Proposal, context_ray_distance: np.ndarray
) -> dict[str, Any]:
    if context_ray_distance.shape != uni.depth_z_m.shape:
        raise ValueError("context ray distance shape does not match proposals")
    context_depth_z = np.stack(
        [
            ray_distance_to_depth_z(context_ray_distance[index], uni.intrinsics[index])
            for index in range(len(uni.frame_ids))
        ]
    )
    context_valid = np.isfinite(context_ray_distance) & (context_ray_distance > 0)
    uni_valid = uni.valid_mask & context_valid
    emv_valid = emv.valid_mask & context_valid
    return {
        "context_depth_opened_after_seal": True,
        "context_depth_semantics": "ray_distance_m_converted_to_opencv_z_m",
        "context_depth_shape": list(context_ray_distance.shape),
        "unidepth_overlap_fraction": float(uni_valid.mean()),
        "emvsnet_overlap_fraction": float(emv_valid.mean()),
        "unidepth_median_abs_error_m": _masked_median(
            np.abs(uni.depth_z_m - context_depth_z), uni_valid
        ),
        "emvsnet_median_abs_error_m": _masked_median(
            np.abs(emv.depth_z_m - context_depth_z), emv_valid
        ),
    }


def _check_repeat_stability(
    output: Path,
    request: GeometrySupplierRequest,
    uni: UniDepthV2SmallSupplier,
    emv: EMVSNetSupplier,
) -> dict[str, Any]:
    first_uni = uni.infer(request)
    second_uni = uni.infer(request)
    first_emv = emv.infer(request)
    second_emv = emv.infer(request)
    result = _repeat_stability_metrics(first_uni, second_uni, first_emv, second_emv)
    _write_json(output / "repeat_stability.json", result)
    return result


def _repeat_stability_metrics(
    first_uni: V8Proposal,
    second_uni: V8Proposal,
    first_emv: V8Proposal,
    second_emv: V8Proposal,
) -> dict[str, Any]:
    uni_mask = first_uni.valid_mask & second_uni.valid_mask
    emv_mask = first_emv.valid_mask & second_emv.valid_mask
    uni_delta = _masked_max(np.abs(first_uni.depth_z_m - second_uni.depth_z_m), uni_mask)
    emv_delta = _masked_max(np.abs(first_emv.depth_z_m - second_emv.depth_z_m), emv_mask)
    passed = (
        uni_delta is not None
        and emv_delta is not None
        and uni_delta <= REPEAT_TOLERANCE_M
        and emv_delta <= REPEAT_TOLERANCE_M
    )
    return {
        "max_unidepth_depth_delta_m": uni_delta,
        "max_emvsnet_depth_delta_m": emv_delta,
        "tolerance_m": REPEAT_TOLERANCE_M,
        "passed": passed,
    }


def _source_drop_intervention_delta(
    full: V8Proposal, dropped: V8Proposal
) -> dict[str, Any]:
    reference = dropped.frame_ids[0]
    try:
        full_index = full.frame_ids.index(reference)
        dropped_index = dropped.frame_ids.index(reference)
    except ValueError as error:
        raise ValueError("source-drop proposal does not share a reference frame") from error
    overlap = full.valid_mask[full_index] & dropped.valid_mask[dropped_index]
    delta = np.abs(full.depth_z_m[full_index] - dropped.depth_z_m[dropped_index])
    return {
        "reference_frame_id": reference,
        "overlap_fraction": float(overlap.mean()),
        "median_depth_delta_m": _masked_median(delta, overlap),
        "max_depth_delta_m": _masked_max(delta, overlap),
    }


def _read_seal_time(cache: Path) -> str:
    metadata = json.loads((cache / "metadata.json").read_text(encoding="utf-8"))
    value = metadata.get("sealed_at_utc")
    if not isinstance(value, str) or not value:
        raise ValueError("proposal cache does not record sealed_at_utc")
    return value


def _masked_median(values: np.ndarray, mask: np.ndarray) -> float | None:
    selected = values[mask]
    return float(np.median(selected)) if selected.size else None


def _masked_max(values: np.ndarray, mask: np.ndarray) -> float | None:
    selected = values[mask]
    return float(np.max(selected)) if selected.size else None


def _runtime_info() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _render_report(record: dict[str, Any]) -> str:
    lines = [
        "# V8 Q0 Supplier Qualification",
        "",
        f"- Final status: `{record['status']}`",
        f"- Schema: `{record['schema_version']}`",
        "- Scope: one frozen Hypersim development-train window; proposal generation is label "
        "blind, followed by a post-seal development context-depth audit; no training or "
        "typed-state write.",
        "",
    ]
    if "mechanical_qualification" in record:
        lines.extend(
            [
                "## Mechanical Qualification",
                "",
                f"- Status: `{record['mechanical_qualification']['status']}`",
                "- Meaning: both external suppliers ran and passed the executable protocol checks.",
                "",
            ]
        )
    if "error" in record:
        lines.extend(
            [
                "## Final Gate",
                "",
                f"`{record['error']['type']}: {record['error']['message']}`",
                "",
            ]
        )
    lines.extend(["## Checks", ""])
    for name, value in record.get("checks", {}).items():
        lines.append(f"- `{name}`: {value.get('status', 'recorded')}")
    warnings = [
        warning
        for value in record.get("checks", {}).values()
        if isinstance(value, dict)
        for warning in value.get("warnings", [])
        if isinstance(warning, dict)
    ]
    if warnings:
        lines.extend(["", "## Diagnostic Warnings", ""])
        for warning in warnings:
            severity = warning.get("severity", "WARNING")
            code = warning.get("code", "unspecified")
            message = warning.get("message", "")
            lines.append(f"- `[{severity}] {code}`: {message}")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "`model-supported` means supplier agreement only. It is not a measurement certificate, "
            "correctness proof, or typed scene state. Post-hoc context-depth values are "
            "descriptive and were opened only after all proposal caches were sealed.",
            "",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
