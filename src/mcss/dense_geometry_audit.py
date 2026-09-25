"""Frozen decision gates for the MapAnything supplier qualification audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch
import yaml
from PIL import Image

from mcss.data.manifest import SceneManifest, load_manifest
from mcss.dense_geometry_certificates import (
    DenseCertificateConfig,
    DenseGeometryCertificates,
    build_dense_certificates,
    read_dense_certificate_cache,
    write_dense_certificate_cache,
)
from mcss.geometry_supplier import (
    CodeDependency,
    GeometrySupplierRequest,
    GeometrySupplierResult,
    SupplierIdentity,
    SupplierRunConfig,
    read_supplier_cache,
    write_supplier_cache,
)
from mcss.mapanything_supplier import MapAnythingSupplier
from mcss.types import Cameras


class GeometrySupplier(Protocol):
    def infer(self, request: GeometrySupplierRequest) -> GeometrySupplierResult: ...


@dataclass(frozen=True)
class DenseGeometryAuditConfig:
    seed: int
    dataset_root: str
    frozen_windows: str
    output_dir: str
    context_views: int
    permutation_windows: int
    supplier_identity: SupplierIdentity
    run_config: SupplierRunConfig
    code_root: str
    dinov2_code_root: str
    device: str
    model_cache_dir: str
    certificate: DenseCertificateConfig
    decision: DenseAuditDecisionThresholds

    def __post_init__(self) -> None:
        if self.context_views < 2:
            raise ValueError("context_views must be at least two")
        if self.permutation_windows < 0:
            raise ValueError("permutation_windows must be nonnegative")
        for name in (
            "dataset_root",
            "frozen_windows",
            "output_dir",
            "code_root",
            "dinov2_code_root",
        ):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"{name} must be non-empty")
        if not self.device:
            raise ValueError("device must be non-empty")


@dataclass(frozen=True)
class DenseAuditDecisionThresholds:
    minimum_windows: int = 32
    minimum_certified_coverage: float = 0.15
    minimum_nonzero_window_fraction: float = 0.80
    maximum_median_error_m: float = 0.25
    maximum_p90_error_m: float = 0.50
    maximum_permutation_delta_m: float = 1e-4
    minimum_shuffled_error_ratio: float = 1.5
    minimum_shuffled_error_margin_m: float = 0.05
    maximum_shuffled_coverage_ratio: float = 0.5
    minimum_risk_error_spearman: float = 0.0

    def __post_init__(self) -> None:
        if self.minimum_windows < 1:
            raise ValueError("minimum_windows must be positive")
        fractions = (
            self.minimum_certified_coverage,
            self.minimum_nonzero_window_fraction,
            self.maximum_shuffled_coverage_ratio,
        )
        if any(value < 0 or value > 1 for value in fractions):
            raise ValueError("coverage thresholds must be in [0, 1]")


def decide_dense_audit(
    metrics: dict[str, Any], thresholds: DenseAuditDecisionThresholds
) -> dict[str, Any]:
    required = {
        "window_count",
        "certified_coverage_macro",
        "nonzero_window_fraction",
        "geometry_error_median_m",
        "geometry_error_p90_m",
        "matched_confidence_median_error_m",
        "risk_error_spearman",
        "permutation_max_depth_delta_m",
        "shuffled_certified_coverage_macro",
    }
    missing = sorted(name for name in required if metrics.get(name) is None)
    if missing or int(metrics.get("window_count", 0)) < thresholds.minimum_windows:
        return {
            "status": "INCONCLUSIVE",
            "gates": {},
            "missing": missing,
            "reason": "required metrics or frozen window count are incomplete",
        }
    certified_error = float(metrics["geometry_error_median_m"])
    certified_coverage = float(metrics["certified_coverage_macro"])
    shuffled_coverage = float(metrics["shuffled_certified_coverage_macro"])
    shuffle_degrades = shuffled_coverage <= (
        certified_coverage * thresholds.maximum_shuffled_coverage_ratio
    )
    if not shuffle_degrades:
        shuffled_error_value = metrics.get("shuffled_geometry_error_median_m")
        if shuffled_error_value is not None:
            shuffled_error = float(shuffled_error_value)
            shuffle_degrades = (
                shuffled_error >= certified_error * thresholds.minimum_shuffled_error_ratio
                and shuffled_error
                >= certified_error + thresholds.minimum_shuffled_error_margin_m
            )
    gates = {
        "certified_coverage": certified_coverage >= thresholds.minimum_certified_coverage,
        "nonzero_windows": float(metrics["nonzero_window_fraction"])
        >= thresholds.minimum_nonzero_window_fraction,
        "median_geometry_error": certified_error <= thresholds.maximum_median_error_m,
        "p90_geometry_error": float(metrics["geometry_error_p90_m"])
        <= thresholds.maximum_p90_error_m,
        "beats_confidence_at_matched_coverage": certified_error
        < float(metrics["matched_confidence_median_error_m"]),
        "risk_predicts_geometry_error": float(metrics["risk_error_spearman"])
        > thresholds.minimum_risk_error_spearman,
        "view_permutation_invariance": float(metrics["permutation_max_depth_delta_m"])
        <= thresholds.maximum_permutation_delta_m,
        "shuffled_control_degrades": shuffle_degrades,
    }
    return {
        "status": (
            "GO_TO_STATE_INTEGRATION" if all(gates.values()) else "STOP_MAPANYTHING_SUPPLIER"
        ),
        "gates": gates,
        "missing": [],
        "reason": "all frozen gates passed" if all(gates.values()) else "one or more gates failed",
    }


def validate_development_train_root(path: str | Path) -> Path:
    root = Path(path).resolve()
    if root.name.lower() != "train" or any(
        token in str(root).lower() for token in ("diagnostic_test", "final_holdout")
    ):
        raise ValueError(
            "dense geometry audit is restricted to the prepared development train root"
        )
    return root


def load_dense_audit_config(path: str | Path) -> DenseGeometryAuditConfig:
    config_path = Path(path)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("dense audit config root must be a mapping")
    allowed = {
        "seed",
        "dataset_root",
        "frozen_windows",
        "output_dir",
        "context_views",
        "permutation_windows",
        "supplier",
        "certificate",
        "decision",
    }
    _require_keys(raw, allowed, "dense audit")
    supplier_raw = dict(raw["supplier"])
    supplier_allowed = {
        "model_id",
        "model_revision",
        "code_repository",
        "code_revision",
        "dependencies",
        "code_root",
        "dinov2_code_root",
        "device",
        "model_cache_dir",
        "memory_efficient_inference",
        "minibatch_size",
        "use_amp",
        "amp_dtype",
        "apply_mask",
        "mask_edges",
        "apply_confidence_mask",
        "use_multiview_confidence",
    }
    _require_keys(supplier_raw, supplier_allowed, "supplier")
    dependencies = tuple(
        CodeDependency(
            name=str(value["name"]),
            repository=str(value["repository"]),
            revision=str(value["revision"]),
            source_sha256=str(value["source_sha256"]),
        )
        for value in supplier_raw.pop("dependencies")
    )
    identity = SupplierIdentity(
        model_id=str(supplier_raw.pop("model_id")),
        model_revision=str(supplier_raw.pop("model_revision")),
        code_repository=str(supplier_raw.pop("code_repository")),
        code_revision=str(supplier_raw.pop("code_revision")),
        dependencies=dependencies,
    )
    code_root = str(supplier_raw.pop("code_root"))
    dinov2_code_root = str(supplier_raw.pop("dinov2_code_root"))
    device = str(supplier_raw.pop("device"))
    model_cache_dir = str(supplier_raw.pop("model_cache_dir"))
    run_config = SupplierRunConfig(**supplier_raw)
    return DenseGeometryAuditConfig(
        seed=int(raw["seed"]),
        dataset_root=str(raw["dataset_root"]),
        frozen_windows=str(raw["frozen_windows"]),
        output_dir=str(raw["output_dir"]),
        context_views=int(raw["context_views"]),
        permutation_windows=int(raw["permutation_windows"]),
        supplier_identity=identity,
        run_config=run_config,
        code_root=code_root,
        dinov2_code_root=dinov2_code_root,
        device=device,
        model_cache_dir=model_cache_dir,
        certificate=DenseCertificateConfig(**dict(raw["certificate"])),
        decision=DenseAuditDecisionThresholds(**dict(raw["decision"])),
    )


def run_dense_audit(
    config: DenseGeometryAuditConfig,
    *,
    supplier: GeometrySupplier | None = None,
) -> dict[str, Any]:
    dataset_root = validate_development_train_root(config.dataset_root)
    output_dir = Path(config.output_dir).resolve()
    result_path = output_dir / "audit_result.json"
    if result_path.is_file():
        raise FileExistsError(f"completed audit already exists: {result_path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    windows = _load_frozen_windows(config, dataset_root)
    frozen_copy = {
        "partition": "train",
        "seed": config.seed,
        "source": str(Path(config.frozen_windows).resolve()),
        "windows": windows,
    }
    _atomic_json(output_dir / "frozen_windows.json", frozen_copy, overwrite=False)
    supplier = supplier or MapAnythingSupplier(
        identity=config.supplier_identity,
        run_config=config.run_config,
        code_root=config.code_root,
        dinov2_code_root=config.dinov2_code_root,
        backend=None,
        device=config.device,
        model_cache_dir=config.model_cache_dir,
    )

    started_at = _now()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    window_metrics = []
    access_records = []
    permutation_deltas = []
    shuffled_coverages = []
    shuffled_errors = []
    for index, window in enumerate(windows):
        window_dir = output_dir / "windows" / f"window_{index:03d}"
        metric_path = window_dir / "evaluation.json"
        if metric_path.is_file():
            metric = _read_json(metric_path)
            _validate_resumed_window(metric, window, window_dir)
        else:
            manifest = load_manifest(window["manifest"])
            frames = _frames_for_window(manifest, window)
            request = _load_request(manifest, frames)
            metric = materialize_and_evaluate_window(
                window_dir,
                request=request,
                supplier=supplier,
                identity=config.supplier_identity,
                run_config=config.run_config,
                certificate_config=config.certificate,
                depth_loader=lambda manifest=manifest, frames=frames: _load_depths(
                    manifest, frames
                ),
            )
            metric.update(
                {
                    "window_index": index,
                    "scene_id": window["scene_id"],
                    "start": window["start"],
                    "frame_ids": window["frame_ids"],
                }
            )
            _atomic_json(metric_path, metric, overwrite=False)
        window_metrics.append(metric)
        access_records.extend(
            {
                "window_index": index,
                "scene_id": metric["scene_id"],
                "path": path,
                "opened_at": metric["depth_opened_at"],
                "after_certificate_seal": metric["certificate_before_depth"],
            }
            for path in metric["accessed_depth_paths"]
        )

        supplier_result, _ = read_supplier_cache(window_dir / "supplier")
        if index < config.permutation_windows:
            manifest = load_manifest(window["manifest"])
            frames = _frames_for_window(manifest, window)
            request = _load_request(manifest, frames)
            permutation_deltas.append(
                _run_canonical_order_replay(
                    window_dir / "permutation_supplier",
                    request=request,
                    reference=supplier_result,
                    supplier=supplier,
                    identity=config.supplier_identity,
                    run_config=config.run_config,
                )
            )
        shuffled_coverages.append(float(metric["shuffled_certified_coverage"]))
        shuffled_errors.extend(
            float(value) for value in metric["shuffled_geometry_errors_m"]
        )

    metrics = _aggregate_dense_metrics(
        window_metrics,
        permutation_deltas=permutation_deltas,
        shuffled_coverages=shuffled_coverages,
        shuffled_errors=shuffled_errors,
    )
    decision = decide_dense_audit(metrics, config.decision)
    finished_at = _now()
    final_or_diagnostic_access = any(
        token in record["path"].lower()
        for record in access_records
        for token in ("diagnostic_test", "final_holdout")
    )
    certificate_before_depth = all(
        bool(metric["certificate_before_depth"]) for metric in window_metrics
    )
    result = {
        "schema_version": 1,
        "status": decision["status"],
        "started_at": started_at,
        "finished_at": finished_at,
        "partition": "train",
        "dataset_root": str(dataset_root),
        "frozen_windows": str(Path(config.frozen_windows).resolve()),
        "supplier_identity": config.supplier_identity.to_dict(),
        "supplier_run_config": asdict(config.run_config),
        "certificate_config": asdict(config.certificate),
        "decision_thresholds": asdict(config.decision),
        "decision": decision,
        "metrics": metrics,
        "certificate_before_depth": certificate_before_depth,
        "final_or_diagnostic_access": final_or_diagnostic_access,
        "cuda_peak_memory_bytes": (
            int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None
        ),
    }
    _atomic_json(output_dir / "access_log.json", {"records": access_records}, overwrite=True)
    _atomic_jsonl(output_dir / "window_metrics.jsonl", window_metrics)
    _atomic_json(result_path, result, overwrite=False)
    provenance = {
        "audit_result_sha256": _sha256_file(result_path),
        "model_files": [],
        "code_snapshots": [],
        "runtime": {
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "cuda_peak_memory_bytes": result["cuda_peak_memory_bytes"],
        },
        "disk": {},
        "training_data_disclosure": {
            "datasets": [],
            "hypersim_named_in_public_config": None,
            "limitation": "Training-data disclosure was not collected by this runner.",
        },
        "label_access": {
            "count": len(access_records),
            "all_after_certificate_seal": all(
                bool(record["after_certificate_seal"]) for record in access_records
            ),
            "final_or_diagnostic_access": final_or_diagnostic_access,
            "paths": [record["path"] for record in access_records],
        },
    }
    write_dense_audit_finalization(output_dir, provenance)
    return result


def write_dense_audit_finalization(
    output_dir: str | Path, provenance: dict[str, Any]
) -> None:
    """Write derived provenance/report/hash files without changing the scientific result."""

    root = Path(output_dir).resolve()
    result_path = root / "audit_result.json"
    if not result_path.is_file():
        raise FileNotFoundError(result_path)
    expected_result_hash = provenance.get("audit_result_sha256")
    actual_result_hash = _sha256_file(result_path)
    if expected_result_hash != actual_result_hash:
        raise ValueError("provenance audit_result_sha256 does not match the sealed result")
    result = _read_json(result_path)
    _atomic_json(root / "provenance.json", provenance, overwrite=True)
    (root / "REPORT.md").write_text(
        render_dense_report(result, provenance), encoding="utf-8"
    )
    artifact_hashes = {
        str(path.relative_to(root)).replace("\\", "/"): _sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "artifact_hashes.json"
    }
    _atomic_json(
        root / "artifact_hashes.json",
        {"sha256": artifact_hashes},
        overwrite=True,
    )


def render_dense_report(
    result: dict[str, Any], provenance: dict[str, Any] | None = None
) -> str:
    metrics = result["metrics"]
    gates = result["decision"].get("gates", {})
    provenance = provenance or {}
    model_lines = _report_model_files(provenance.get("model_files", []))
    snapshot_lines = _report_code_snapshots(provenance.get("code_snapshots", []))
    runtime_lines = _report_mapping(provenance.get("runtime", {}))
    disk_lines = _report_mapping(provenance.get("disk", {}))
    disclosure = provenance.get("training_data_disclosure", {})
    datasets = disclosure.get("datasets", [])
    dataset_lines = "\n".join(f"- {value}" for value in datasets) or "- not recorded"
    label_access = provenance.get("label_access", {})
    label_paths = label_access.get("paths", [])
    label_lines = "\n".join(f"- `{value}`" for value in label_paths) or "- none"
    gate_lines = "\n".join(
        f"- `{name}`: {'PASS' if passed else 'FAIL'}" for name, passed in gates.items()
    ) or "- none"
    return f"""# V7 MapAnything Supplier Qualification Audit

## Decision

- Status: `{result['status']}`
- Reason: {result['decision']['reason']}
- Frozen windows: `{metrics.get('window_count')}`
- Label isolation: `{'PASS' if result['certificate_before_depth'] else 'FAIL'}`
- Diagnostic/final access: `{result['final_or_diagnostic_access']}`

{gate_lines}

## Key metrics

- Certified coverage: `{_format(metrics.get('certified_coverage_macro'))}`
- Nonzero-window fraction: `{_format(metrics.get('nonzero_window_fraction'))}`
- Geometry median / P90: `{_format(metrics.get('geometry_error_median_m'))}` m /
  `{_format(metrics.get('geometry_error_p90_m'))}` m
- Same-coverage raw-confidence median:
  `{_format(metrics.get('matched_confidence_median_error_m'))}` m
- Risk-error Spearman: `{_format(metrics.get('risk_error_spearman'))}`
- Shuffled coverage / error median:
  `{_format(metrics.get('shuffled_certified_coverage_macro'))}` /
  `{_format(metrics.get('shuffled_geometry_error_median_m'))}` m
- View-permutation max depth delta:
  `{_format(metrics.get('permutation_max_depth_delta_m'))}` m

## Scope and claim boundary

MapAnything is a frozen external supplier and is not a project contribution. This audit does not
modify the typed state, renderer, or training loop and does not establish CVPR readiness. A GO
only authorizes a later write-protection experiment; a STOP seals this supplier configuration.

## Provenance

- Model: `{result['supplier_identity']['model_id']}` at
  `{result['supplier_identity']['model_revision']}`
- Code: `{result['supplier_identity']['code_repository']}` at
  `{result['supplier_identity']['code_revision']}`
- Code dependencies: `{result['supplier_identity']['dependencies']}`
- Model license: CC BY-NC 4.0; code license: Apache 2.0

### Model files

{model_lines}

### Code snapshots

{snapshot_lines}

### Runtime

{runtime_lines}

### Storage

{disk_lines}

### Public training-data disclosure

The pinned public reproduction configuration names these datasets:

{dataset_lines}

- Hypersim named in that public config: `{disclosure.get('hypersim_named_in_public_config')}`
- Limitation: {disclosure.get('limitation', 'not recorded')}

### Label access

- Count: `{label_access.get('count')}`
- All after certificate seal: `{label_access.get('all_after_certificate_seal')}`
- Diagnostic/final access: `{label_access.get('final_or_diagnostic_access')}`

{label_lines}

- Formal Research Institute: not invoked
- Evaluation Council: not convened
"""


def _report_model_files(values: Any) -> str:
    if not values:
        return "- not recorded"
    return "\n".join(
        f"- `{value.get('name')}`: `{value.get('bytes')}` bytes, "
        f"SHA-256 `{value.get('sha256')}`"
        for value in values
    )


def _report_code_snapshots(values: Any) -> str:
    if not values:
        return "- not recorded"
    return "\n".join(
        f"- `{value.get('name')}`: revision `{value.get('revision')}`, "
        f"source SHA-256 `{value.get('source_sha256')}`"
        for value in values
    )


def _report_mapping(value: Any) -> str:
    if not value:
        return "- not recorded"
    return "\n".join(f"- `{name}`: `{item}`" for name, item in value.items())


def materialize_and_evaluate_window(
    destination: str | Path,
    *,
    request: GeometrySupplierRequest,
    supplier: GeometrySupplier,
    identity: SupplierIdentity,
    run_config: SupplierRunConfig,
    certificate_config: DenseCertificateConfig,
    depth_loader: Callable[[], tuple[np.ndarray, list[Path]]],
) -> dict[str, Any]:
    """Seal label-free artifacts before invoking the post-hoc depth loader."""

    root = Path(destination).resolve()
    canonical_request = request.canonical()
    supplier_path = root / "supplier"
    certificate_path = root / "certificates"
    if root.exists():
        if not supplier_path.is_dir() or not certificate_path.is_dir():
            raise RuntimeError(f"audit window contains an incomplete sealed-cache pair: {root}")
        supplier_result, supplier_metadata = read_supplier_cache(supplier_path)
        certificates, certificate_metadata = read_dense_certificate_cache(certificate_path)
        _validate_sealed_window_inputs(
            canonical_request,
            supplier_result,
            supplier_metadata,
            certificates,
            certificate_metadata,
            identity,
            run_config,
            certificate_config,
        )
    else:
        root.mkdir(parents=True)
        supplier_result = supplier.infer(canonical_request).canonical()
        if supplier_result.frame_ids != canonical_request.frame_ids:
            raise ValueError("supplier result frame IDs do not match the canonical request")
        write_supplier_cache(
            supplier_path,
            supplier_result,
            identity,
            canonical_request.input_sha256,
            run_config,
        )
        supplier_metadata = _read_json(supplier_path / "metadata.json")
        certificates = build_dense_certificates(supplier_result, certificate_config)
        write_dense_certificate_cache(
            certificate_path,
            certificates,
            certificate_config,
            supplier_arrays_sha256=str(supplier_metadata["arrays_sha256"]),
        )
        certificate_metadata = _read_json(certificate_path / "metadata.json")
    certificates_sealed_at = _now()

    depths, depth_paths = depth_loader()
    depth_opened_at = _now()
    depths = np.asarray(depths, dtype=np.float32)
    expected_depth_shape = (
        len(canonical_request.frame_ids),
        *canonical_request.cameras.image_size,
    )
    if depths.shape != expected_depth_shape:
        raise ValueError(
            f"post-hoc depth shape {depths.shape} does not match request {expected_depth_shape}"
        )
    evaluation = evaluate_dense_certificates(
        certificates,
        supplier_result,
        canonical_request,
        depths,
    )
    shuffled_evaluation = _evaluate_camera_shuffled_control(
        supplier_result,
        canonical_request,
        depths,
        certificate_config,
    )
    evaluation.update(
        {
            "certificate_before_depth": certificates_sealed_at <= depth_opened_at,
            "certificates_sealed_at": certificates_sealed_at,
            "depth_opened_at": depth_opened_at,
            "accessed_depth_paths": [str(Path(path)) for path in depth_paths],
            "supplier_cache": str(supplier_path),
            "certificate_cache": str(certificate_path),
            "supplier_arrays_sha256": supplier_metadata["arrays_sha256"],
            "certificate_arrays_sha256": certificate_metadata["arrays_sha256"],
            "shuffled_certified_coverage": shuffled_evaluation[
                "certified_coverage"
            ],
            "shuffled_geometry_errors_m": shuffled_evaluation[
                "geometry_errors_m"
            ],
            "shuffled_geometry_error_median_m": shuffled_evaluation[
                "geometry_error_median_m"
            ],
        }
    )
    return evaluation


def evaluate_dense_certificates(
    certificates: DenseGeometryCertificates,
    supplier: GeometrySupplierResult,
    request: GeometrySupplierRequest,
    depths: np.ndarray,
) -> dict[str, Any]:
    """Evaluate sealed supplier geometry against source-view depth after the isolation gate."""

    request = request.canonical()
    supplier = supplier.canonical()
    certificates = certificates.canonical()
    if not (
        request.frame_ids == supplier.frame_ids == certificates.frame_ids
        and depths.shape[0] == len(request.frame_ids)
    ):
        raise ValueError("supplier, certificates, request, and depth must share frame IDs")
    if not np.allclose(
        supplier.c2w, request.cameras.c2w.detach().cpu().numpy(), atol=1e-5, rtol=1e-5
    ):
        raise ValueError("supplier cache must preserve the known request camera poses")

    truth_depth, truth_valid = _sample_truth_depth(
        certificates.pixel_xy,
        supplier.intrinsics,
        request.cameras.intrinsics.detach().cpu().numpy(),
        depths,
    )
    errors = np.abs(certificates.depth_along_ray - truth_depth)
    finite_truth = truth_valid & np.isfinite(errors) & (truth_depth > 0)
    certified_valid = certificates.certified & finite_truth
    raw_valid = certificates.raw_valid & finite_truth
    certified_errors = errors[certified_valid].astype(float).tolist()
    risks = certificates.risk[certified_valid].astype(float).tolist()
    raw_confidence_errors = list(
        zip(
            certificates.raw_confidence[raw_valid].astype(float).tolist(),
            errors[raw_valid].astype(float).tolist(),
            strict=True,
        )
    )
    matched_errors = [
        error
        for _, error in sorted(raw_confidence_errors, key=lambda item: (-item[0], item[1]))[
            : len(certified_errors)
        ]
    ]
    sample_count = int(certificates.certified.size)
    return {
        "grid_sample_count": sample_count,
        "raw_valid_count": int(certificates.raw_valid.sum()),
        "raw_valid_coverage": float(certificates.raw_valid.mean()),
        "certified_count": int(certificates.certified.sum()),
        "certified_coverage": float(certificates.certified.mean()),
        "evaluated_certified_count": len(certified_errors),
        "geometry_errors_m": certified_errors,
        "risks": risks,
        "raw_confidence_and_errors_m": raw_confidence_errors,
        "geometry_error_median_m": _quantile(certified_errors, 0.5),
        "geometry_error_p90_m": _quantile(certified_errors, 0.9),
        "matched_confidence_median_error_m": _quantile(matched_errors, 0.5),
        "risk_error_spearman": _spearman(risks, certified_errors),
        "mean_support_count": float(certificates.support_count.mean()),
        "mean_conflict_count": float(certificates.conflict_count.mean()),
        "mean_occlusion_count": float(certificates.occlusion_count.mean()),
    }


def _sample_truth_depth(
    processed_xy: np.ndarray,
    processed_intrinsics: np.ndarray,
    original_intrinsics: np.ndarray,
    depths: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    view_count, grid_height, grid_width, _ = processed_xy.shape
    sampled = np.zeros((view_count, grid_height, grid_width), dtype=np.float32)
    valid = np.zeros_like(sampled, dtype=np.bool_)
    height, width = depths.shape[-2:]
    for view in range(view_count):
        homogeneous = np.concatenate(
            (
                processed_xy[view],
                np.ones((grid_height, grid_width, 1), dtype=np.float32),
            ),
            axis=-1,
        )
        direction = homogeneous @ np.linalg.inv(processed_intrinsics[view]).T
        projected = direction @ original_intrinsics[view].T
        original_xy = projected[..., :2] / np.maximum(projected[..., 2:3], 1e-8)
        inside = (
            (projected[..., 2] > 1e-8)
            & (original_xy[..., 0] >= 0)
            & (original_xy[..., 0] <= width - 1)
            & (original_xy[..., 1] >= 0)
            & (original_xy[..., 1] <= height - 1)
        )
        x = np.clip(np.rint(original_xy[..., 0]).astype(np.int64), 0, width - 1)
        y = np.clip(np.rint(original_xy[..., 1]).astype(np.int64), 0, height - 1)
        sampled[view] = depths[view, y, x]
        valid[view] = inside & np.isfinite(sampled[view]) & (sampled[view] > 0)
    return sampled, valid


def _load_frozen_windows(
    config: DenseGeometryAuditConfig, dataset_root: Path
) -> list[dict[str, Any]]:
    value = _read_json(Path(config.frozen_windows).resolve())
    if value.get("partition") != "train" or int(value.get("seed", -1)) != config.seed:
        raise ValueError("frozen windows partition or seed does not match the audit config")
    raw_windows = value.get("windows")
    if not isinstance(raw_windows, list) or not raw_windows:
        raise ValueError("frozen windows must contain a non-empty windows list")
    windows = []
    for raw in raw_windows:
        if not isinstance(raw, dict):
            raise ValueError("every frozen window must be an object")
        scene_id = str(raw.get("scene_id", ""))
        start = int(raw.get("start", -1))
        frame_ids = [int(value) for value in raw.get("frame_ids", [])]
        if not scene_id or start < 0 or len(frame_ids) != config.context_views:
            raise ValueError("frozen window identity or context view count is invalid")
        manifest = (dataset_root / scene_id / "manifest.json").resolve()
        try:
            manifest.relative_to(dataset_root)
        except ValueError as error:
            raise ValueError("frozen window manifest escapes the development root") from error
        if not manifest.is_file():
            raise FileNotFoundError(manifest)
        loaded = load_manifest(manifest)
        frames = loaded.frames[start : start + config.context_views]
        if [frame.frame_id for frame in frames] != frame_ids:
            raise ValueError(f"frozen frame IDs do not match current manifest: {scene_id}")
        windows.append(
            {
                "scene_id": scene_id,
                "start": start,
                "frame_ids": frame_ids,
                "manifest": str(manifest),
            }
        )
    if len(windows) < config.decision.minimum_windows:
        raise ValueError("frozen window file contains fewer windows than the decision rule")
    if config.permutation_windows > len(windows):
        raise ValueError("permutation_windows exceeds the frozen window count")
    return windows


def _frames_for_window(manifest: SceneManifest, window: dict[str, Any]) -> tuple[Any, ...]:
    start = int(window["start"])
    count = len(window["frame_ids"])
    frames = tuple(manifest.frames[start : start + count])
    if [frame.frame_id for frame in frames] != window["frame_ids"]:
        raise ValueError("window frame IDs do not match manifest")
    return frames


def _load_request(
    manifest: SceneManifest, frames: Sequence[Any]
) -> GeometrySupplierRequest:
    if manifest.root is None:
        raise ValueError("manifest root is required")
    rgb_values = []
    intrinsics = []
    c2w = []
    for frame in frames:
        image = Image.open(manifest.root / frame.rgb).convert("RGB")
        if (image.height, image.width) != manifest.image_size:
            raise ValueError("prepared RGB size does not match manifest")
        array = np.asarray(image, dtype=np.float32) / 255.0
        rgb_values.append(torch.from_numpy(array).permute(2, 0, 1).contiguous())
        intrinsics.append(torch.tensor(frame.intrinsics, dtype=torch.float32))
        c2w.append(torch.tensor(frame.c2w, dtype=torch.float32))
    cameras = Cameras(torch.stack(intrinsics), torch.stack(c2w), manifest.image_size)
    return GeometrySupplierRequest(
        tuple(frame.frame_id for frame in frames), torch.stack(rgb_values), cameras
    )


def _load_depths(
    manifest: SceneManifest, frames: Sequence[Any]
) -> tuple[np.ndarray, list[Path]]:
    if manifest.root is None:
        raise ValueError("manifest root is required")
    paths = []
    values = []
    for frame in frames:
        if frame.depth is None:
            raise ValueError("post-hoc dense geometry audit requires context depth labels")
        path = (manifest.root / frame.depth).resolve()
        value = np.load(path, allow_pickle=False).astype(np.float32)
        if value.shape != manifest.image_size:
            raise ValueError(f"depth shape does not match manifest: {path}")
        paths.append(path)
        values.append(value)
    return np.stack(values), paths


def _run_canonical_order_replay(
    destination: Path,
    *,
    request: GeometrySupplierRequest,
    reference: GeometrySupplierResult,
    supplier: GeometrySupplier,
    identity: SupplierIdentity,
    run_config: SupplierRunConfig,
) -> float:
    indices = torch.arange(len(request.frame_ids) - 1, -1, -1, dtype=torch.long)
    reversed_request = GeometrySupplierRequest(
        tuple(reversed(request.frame_ids)),
        request.rgb.index_select(0, indices),
        request.cameras.select_views(indices),
    )
    if destination.exists():
        replay, metadata = read_supplier_cache(destination)
        if (
            metadata.get("identity") != identity.to_dict()
            or metadata.get("run_config") != asdict(run_config)
            or metadata.get("input_sha256") != reversed_request.input_sha256
        ):
            raise ValueError("canonical order replay cache does not match current inputs")
    else:
        replay = supplier.infer(reversed_request).canonical()
        write_supplier_cache(
            destination,
            replay,
            identity,
            reversed_request.input_sha256,
            run_config,
        )
    reference = reference.canonical()
    replay = replay.canonical()
    if reference.frame_ids != replay.frame_ids:
        return math.inf
    if reference.depth_along_ray.shape != replay.depth_along_ray.shape:
        return math.inf
    valid = reference.valid_mask & replay.valid_mask
    if not valid.any():
        return math.inf
    return float(
        np.max(np.abs(reference.depth_along_ray[valid] - replay.depth_along_ray[valid]))
    )


def _camera_shuffled_control(
    supplier: GeometrySupplierResult, config: DenseCertificateConfig
) -> DenseGeometryCertificates:
    if len(supplier.frame_ids) < 2:
        return build_dense_certificates(supplier, config)
    shuffled = GeometrySupplierResult(
        supplier.frame_ids,
        supplier.depth_along_ray,
        supplier.confidence,
        supplier.valid_mask,
        supplier.intrinsics,
        np.roll(supplier.c2w, shift=1, axis=0),
    )
    return build_dense_certificates(shuffled, config)


def _evaluate_camera_shuffled_control(
    supplier: GeometrySupplierResult,
    request: GeometrySupplierRequest,
    depths: np.ndarray,
    config: DenseCertificateConfig,
) -> dict[str, Any]:
    """Score a camera-shuffled certificate build against unshuffled known-camera GT."""

    shuffled_supplier = GeometrySupplierResult(
        supplier.frame_ids,
        supplier.depth_along_ray,
        supplier.confidence,
        supplier.valid_mask,
        supplier.intrinsics,
        np.roll(supplier.c2w, shift=1, axis=0),
    )
    certificates = build_dense_certificates(shuffled_supplier, config)
    truth_depth, truth_valid = _sample_truth_depth(
        certificates.pixel_xy,
        supplier.intrinsics,
        request.cameras.intrinsics.detach().cpu().numpy(),
        depths,
    )
    truth_world = _world_from_depth_grid(
        certificates.pixel_xy,
        truth_depth,
        supplier.intrinsics,
        request.cameras.c2w.detach().cpu().numpy(),
    )
    errors = np.linalg.norm(certificates.world_points - truth_world, axis=-1)
    valid = certificates.certified & truth_valid & np.isfinite(errors)
    values = errors[valid].astype(float).tolist()
    return {
        "certified_coverage": float(certificates.certified.mean()),
        "geometry_errors_m": values,
        "geometry_error_median_m": _quantile(values, 0.5),
    }


def _world_from_depth_grid(
    pixel_xy: np.ndarray,
    depth: np.ndarray,
    intrinsics: np.ndarray,
    c2w: np.ndarray,
) -> np.ndarray:
    world = np.zeros((*depth.shape, 3), dtype=np.float32)
    for view in range(depth.shape[0]):
        homogeneous = np.concatenate(
            (
                pixel_xy[view],
                np.ones((*pixel_xy.shape[1:3], 1), dtype=np.float32),
            ),
            axis=-1,
        )
        directions = homogeneous @ np.linalg.inv(intrinsics[view]).T
        directions /= np.maximum(np.linalg.norm(directions, axis=-1, keepdims=True), 1e-8)
        world_directions = directions @ c2w[view, :3, :3].T
        world[view] = c2w[view, :3, 3] + world_directions * depth[view][..., None]
    return world


def _aggregate_dense_metrics(
    windows: Sequence[dict[str, Any]],
    *,
    permutation_deltas: Sequence[float],
    shuffled_coverages: Sequence[float],
    shuffled_errors: Sequence[float],
) -> dict[str, Any]:
    geometry_errors = [
        float(value) for window in windows for value in window.get("geometry_errors_m", [])
    ]
    risks = [float(value) for window in windows for value in window.get("risks", [])]
    raw_confidence_errors = [
        (float(confidence), float(error))
        for window in windows
        for confidence, error in window.get("raw_confidence_and_errors_m", [])
    ]
    matched_errors = [
        error
        for _, error in sorted(
            raw_confidence_errors, key=lambda item: (-item[0], item[1])
        )[: len(geometry_errors)]
    ]
    return {
        "window_count": len(windows),
        "scene_count": len({str(window["scene_id"]) for window in windows}),
        "raw_valid_coverage_macro": _mean(
            [float(window["raw_valid_coverage"]) for window in windows]
        ),
        "certified_coverage_macro": _mean(
            [float(window["certified_coverage"]) for window in windows]
        ),
        "nonzero_window_fraction": _mean(
            [float(int(window["certified_count"]) > 0) for window in windows]
        ),
        "geometry_evaluated_count": len(geometry_errors),
        "geometry_error_median_m": _quantile(geometry_errors, 0.5),
        "geometry_error_p90_m": _quantile(geometry_errors, 0.9),
        "matched_confidence_count": len(matched_errors),
        "matched_confidence_median_error_m": _quantile(matched_errors, 0.5),
        "risk_error_spearman": _spearman(risks, geometry_errors),
        "risk_bin_median_errors_m": _risk_bins(risks, geometry_errors),
        "mean_support_count": _mean(
            [float(window["mean_support_count"]) for window in windows]
        ),
        "mean_conflict_count": _mean(
            [float(window["mean_conflict_count"]) for window in windows]
        ),
        "mean_occlusion_count": _mean(
            [float(window["mean_occlusion_count"]) for window in windows]
        ),
        "permutation_window_count": len(permutation_deltas),
        "permutation_max_depth_delta_m": (
            max(permutation_deltas) if permutation_deltas else None
        ),
        "shuffled_certified_coverage_macro": _mean(shuffled_coverages),
        "shuffled_geometry_error_median_m": _quantile(list(shuffled_errors), 0.5),
    }


def _quantile(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    return float(np.quantile(np.asarray(values, dtype=np.float64), probability))


def _spearman(first: list[float], second: list[float]) -> float | None:
    if len(first) != len(second) or len(first) < 2:
        return None
    first_array = np.asarray(first, dtype=np.float64)
    second_array = np.asarray(second, dtype=np.float64)
    first_rank = _average_ranks(first_array)
    second_rank = _average_ranks(second_array)
    if np.ptp(first_rank) == 0 or np.ptp(second_rank) == 0:
        return None
    correlation = float(np.corrcoef(first_rank, second_rank)[0, 1])
    return correlation if math.isfinite(correlation) else None


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranks


def _risk_bins(
    risks: Sequence[float], errors: Sequence[float], bins: int = 4
) -> list[float]:
    if len(risks) != len(errors) or not risks:
        return []
    order = np.argsort(np.asarray(risks), kind="stable")
    return [
        float(np.median(np.asarray(errors)[indices]))
        for indices in np.array_split(order, bins)
        if len(indices)
    ]


def _mean(values: Sequence[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _require_keys(value: dict[str, Any], allowed: set[str], name: str) -> None:
    unknown = sorted(set(value) - allowed)
    missing = sorted(allowed - set(value))
    if unknown:
        raise ValueError(f"unknown {name} config keys: {unknown}")
    if missing:
        raise ValueError(f"missing {name} config keys: {missing}")


def _validate_resumed_window(
    metric: dict[str, Any], window: dict[str, Any], root: Path
) -> None:
    if (
        metric.get("scene_id") != window["scene_id"]
        or metric.get("frame_ids") != window["frame_ids"]
        or not metric.get("certificate_before_depth")
    ):
        raise ValueError(f"resumed window identity or label-isolation record is invalid: {root}")
    read_supplier_cache(root / "supplier")
    read_dense_certificate_cache(root / "certificates")


def _validate_sealed_window_inputs(
    request: GeometrySupplierRequest,
    supplier: GeometrySupplierResult,
    supplier_metadata: dict[str, Any],
    certificates: DenseGeometryCertificates,
    certificate_metadata: dict[str, Any],
    identity: SupplierIdentity,
    run_config: SupplierRunConfig,
    certificate_config: DenseCertificateConfig,
) -> None:
    checks = {
        "frame IDs": supplier.frame_ids
        == certificates.frame_ids
        == request.frame_ids,
        "input hash": supplier_metadata.get("input_sha256") == request.input_sha256,
        "supplier identity": supplier_metadata.get("identity") == identity.to_dict(),
        "supplier run config": supplier_metadata.get("run_config")
        == asdict(run_config),
        "certificate config": certificate_metadata.get("config")
        == asdict(certificate_config),
        "supplier linkage": certificate_metadata.get("supplier_arrays_sha256")
        == supplier_metadata.get("arrays_sha256"),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"sealed audit window does not match current inputs: {failed}")


def _atomic_json(path: Path, value: Any, *, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing == value:
            return
        raise FileExistsError(f"refusing to overwrite artifact: {path}")
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _atomic_jsonl(path: Path, values: Sequence[dict[str, Any]]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        "".join(json.dumps(value, sort_keys=True) + "\n" for value in values),
        encoding="utf-8",
    )
    temporary.replace(path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _format(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _now() -> str:
    return datetime.now().astimezone().isoformat()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Frozen dense geometry audit YAML")
    args = parser.parse_args(argv)
    config = load_dense_audit_config(args.config)
    result = run_dense_audit(config)
    print(json.dumps({"status": result["status"], "metrics": result["metrics"]}, indent=2))
    return 0 if result["status"] != "INCONCLUSIVE" else 2
