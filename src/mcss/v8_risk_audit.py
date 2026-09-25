"""Label-blind V8 supplier-risk fields and window-level selective metrics."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

V8_RISK_CACHE_SCHEMA_VERSION = "mcss.v8_risk_cache.v2"
TYPE_UNKNOWN = np.uint8(0)
TYPE_MODEL_SUPPORTED = np.uint8(1)
TYPE_UNCERTAIN = np.uint8(2)
TYPE_CONFLICT = np.uint8(3)
TYPE_NAMES = {
    int(TYPE_UNKNOWN): "unknown",
    int(TYPE_MODEL_SUPPORTED): "model-supported",
    int(TYPE_UNCERTAIN): "uncertain",
    int(TYPE_CONFLICT): "conflict",
}


def midrank_percentile(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Map valid values to deterministic mid-rank percentiles in the open interval (0, 1)."""

    array = np.asarray(values, dtype=np.float64)
    mask = np.asarray(valid, dtype=np.bool_)
    if array.shape != mask.shape:
        raise ValueError("values and valid must share shape")
    mask = mask & np.isfinite(array)
    result = np.full(array.shape, np.nan, dtype=np.float32)
    selected = array[mask]
    if selected.size == 0:
        return result
    order = np.argsort(selected, kind="stable")
    sorted_values = selected[order]
    ranks = np.empty(selected.size, dtype=np.float64)
    start = 0
    while start < selected.size:
        end = start + 1
        while end < selected.size and sorted_values[end] == sorted_values[start]:
            end += 1
        midrank = (start + 1 + end) / 2.0
        ranks[order[start:end]] = midrank
        start = end
    result[mask] = ((ranks - 0.5) / selected.size).astype(np.float32)
    return result


def build_anchor_risk_fields(
    *,
    emvsnet_points: np.ndarray,
    unidepth_points: np.ndarray,
    emvsnet_native_risk: np.ndarray,
    unidepth_native_risk: np.ndarray,
    emvsnet_valid: np.ndarray,
    unidepth_valid: np.ndarray,
    drop_points: Sequence[np.ndarray],
    drop_native_risks: Sequence[np.ndarray],
    drop_valids: Sequence[np.ndarray],
    cell_width_m: float,
) -> dict[str, np.ndarray]:
    """Build fixed V8 risks for the shared anchor without fusing supplier geometry."""

    if not np.isfinite(cell_width_m) or cell_width_m <= 0:
        raise ValueError("cell_width_m must be positive and finite")
    if not (len(drop_points) == len(drop_native_risks) == len(drop_valids) == 3):
        raise ValueError("V8 Q1 requires exactly three source-drop proposals")

    emv_points = _points(emvsnet_points, "emvsnet_points")
    uni_points = _points(unidepth_points, "unidepth_points")
    if emv_points.shape != uni_points.shape:
        raise ValueError("supplier anchor points must share shape")
    shape = emv_points.shape[:-1]
    emv_risk = _map(emvsnet_native_risk, shape, "emvsnet_native_risk")
    uni_risk = _map(unidepth_native_risk, shape, "unidepth_native_risk")
    emv_valid = _valid(emvsnet_valid, shape, "emvsnet_valid")
    uni_valid = _valid(unidepth_valid, shape, "unidepth_valid")

    parsed_drop_points = [_points(value, "drop_points") for value in drop_points]
    parsed_drop_risks = [
        _map(value, shape, "drop_native_risks") for value in drop_native_risks
    ]
    parsed_drop_valids = [_valid(value, shape, "drop_valids") for value in drop_valids]
    if any(value.shape != emv_points.shape for value in parsed_drop_points):
        raise ValueError("source-drop points must match full EMVSNet anchor shape")

    comparison_valid = emv_valid & uni_valid
    all_drop_valid = np.logical_and.reduce(parsed_drop_valids)

    emv_rank = midrank_percentile(emv_risk, emv_valid)
    uni_rank = midrank_percentile(uni_risk, uni_valid)
    disagreement_raw = np.linalg.norm(emv_points - uni_points, axis=-1) / cell_width_m
    drop_deltas = np.stack(
        [np.linalg.norm(emv_points - value, axis=-1) / cell_width_m for value in parsed_drop_points]
    )
    drop_instability_raw = np.median(drop_deltas, axis=0)
    drop_instability_raw[~all_drop_valid] = 1.0
    disagreement_risk = np.minimum(disagreement_raw, 1.0).astype(np.float32)
    drop_instability_risk = np.minimum(drop_instability_raw, 1.0).astype(np.float32)

    drop_ranks = np.stack(
        [
            np.where(valid, midrank_percentile(risk, valid), 1.0)
            for risk, valid in zip(parsed_drop_risks, parsed_drop_valids, strict=True)
        ]
    )
    median_drop_points = np.median(np.stack(parsed_drop_points), axis=0)
    same_family_disagreement_raw = (
        np.linalg.norm(emv_points - median_drop_points, axis=-1) / cell_width_m
    )
    same_family_disagreement_risk = np.minimum(
        same_family_disagreement_raw, 1.0
    ).astype(np.float32)
    same_family_native_rank = np.nanmedian(drop_ranks, axis=0).astype(np.float32)
    same_family_disagreement_raw[~all_drop_valid] = 1.0
    same_family_disagreement_risk[~all_drop_valid] = 1.0
    same_family_native_rank[~all_drop_valid] = 1.0

    full_risk = np.maximum.reduce(
        [emv_rank, uni_rank, disagreement_risk, drop_instability_risk]
    ).astype(np.float32)
    same_family_risk = np.maximum.reduce(
        [emv_rank, same_family_native_rank, same_family_disagreement_risk]
    ).astype(np.float32)
    full_no_drop = np.maximum.reduce([emv_rank, uni_rank, disagreement_risk]).astype(
        np.float32
    )
    full_no_disagreement = np.maximum.reduce(
        [emv_rank, uni_rank, drop_instability_risk]
    ).astype(np.float32)

    type_code = np.full(shape, TYPE_UNKNOWN, dtype=np.uint8)
    conflict = comparison_valid & all_drop_valid & (disagreement_raw > 1.0)
    uncertain = (
        comparison_valid
        & ~conflict
        & (
            ~all_drop_valid
            | (drop_instability_raw > 1.0)
            | (emv_rank > 0.5)
            | (uni_rank > 0.5)
        )
    )
    supported = comparison_valid & ~conflict & ~uncertain
    type_code[supported] = TYPE_MODEL_SUPPORTED
    type_code[uncertain] = TYPE_UNCERTAIN
    type_code[conflict] = TYPE_CONFLICT

    invalid = ~comparison_valid
    for value in (
        full_risk,
        same_family_risk,
        full_no_drop,
        full_no_disagreement,
        disagreement_risk,
        drop_instability_risk,
        same_family_disagreement_risk,
    ):
        value[invalid] = np.nan

    return {
        "action_points_m": emv_points.astype(np.float32),
        "unidepth_points_m": uni_points.astype(np.float32),
        "comparison_valid": comparison_valid,
        "all_drop_valid": all_drop_valid,
        "type_code": type_code,
        "emvsnet_native_rank": emv_rank,
        "unidepth_native_rank": uni_rank,
        "disagreement_raw": disagreement_raw.astype(np.float32),
        "disagreement_risk": disagreement_risk,
        "drop_instability_raw": drop_instability_raw.astype(np.float32),
        "drop_instability_risk": drop_instability_risk,
        "same_family_native_rank": same_family_native_rank,
        "same_family_disagreement_raw": same_family_disagreement_raw.astype(np.float32),
        "same_family_disagreement_risk": same_family_disagreement_risk,
        "full_risk": full_risk,
        "same_family_risk": same_family_risk,
        "full_no_drop_risk": full_no_drop,
        "full_no_disagreement_risk": full_no_disagreement,
    }


def deterministic_permuted_risk(
    risk: np.ndarray, valid: np.ndarray, *, key: str
) -> np.ndarray:
    """Permute risk values within one window using a label-independent hash order."""

    values = np.asarray(risk, dtype=np.float32)
    mask = np.asarray(valid, dtype=np.bool_)
    if values.shape != mask.shape or not key:
        raise ValueError("risk, valid, and key are invalid")
    result = np.full(values.shape, np.nan, dtype=np.float32)
    flat_indices = np.flatnonzero(mask & np.isfinite(values))
    if flat_indices.size == 0:
        return result
    source = np.sort(values.flat[flat_indices], kind="stable")
    order = sorted(
        flat_indices.tolist(),
        key=lambda index: hashlib.sha256(f"{key}:{index}".encode()).digest(),
    )
    result.flat[np.asarray(order, dtype=np.int64)] = source
    return result


def deterministic_source_derangement(
    scene_id: str, frame_ids: Sequence[int]
) -> tuple[int, int, int]:
    """Choose one of the two derangements of three non-anchor source positions."""

    if not scene_id or len(frame_ids) != 4 or len(set(frame_ids)) != 4:
        raise ValueError("source derangement requires a scene and four unique frames")
    payload = f"{scene_id}:{','.join(str(value) for value in frame_ids)}:v8-q1-shuffle"
    selector = hashlib.sha256(payload.encode("utf-8")).digest()[0] & 1
    return (2, 3, 1) if selector == 0 else (3, 1, 2)


def evaluate_selective_risk(
    risk: np.ndarray,
    errors: np.ndarray,
    valid: np.ndarray,
    *,
    coverages: Sequence[float] = tuple(np.arange(0.05, 1.01, 0.05)),
) -> dict[str, Any]:
    """Compute a selective error curve within one window only."""

    score = np.asarray(risk, dtype=np.float64)
    error = np.asarray(errors, dtype=np.float64)
    mask = np.asarray(valid, dtype=np.bool_)
    if score.shape != error.shape or score.shape != mask.shape:
        raise ValueError("risk, errors, and valid must share shape")
    coverage = np.asarray(tuple(float(value) for value in coverages), dtype=np.float64)
    if (
        coverage.ndim != 1
        or coverage.size < 2
        or np.any(~np.isfinite(coverage))
        or np.any(coverage <= 0)
        or np.any(coverage > 1)
        or np.any(np.diff(coverage) <= 0)
        or coverage[-1] != 1.0
    ):
        raise ValueError("coverages must be strictly increasing in (0, 1] and end at 1")
    mask &= np.isfinite(score) & np.isfinite(error) & (error >= 0)
    scores = score[mask]
    errors_selected = error[mask]
    if scores.size == 0:
        raise ValueError("selective risk requires at least one valid sample")
    stable_index = np.arange(scores.size)
    risk_order = np.lexsort((stable_index, scores))
    oracle_order = np.lexsort((stable_index, errors_selected))
    risk_curve = []
    oracle_curve = []
    retained_counts = []
    for value in coverage:
        retained = min(scores.size, max(1, int(np.ceil(value * scores.size))))
        retained_counts.append(retained)
        risk_curve.append(float(np.mean(errors_selected[risk_order[:retained]])))
        oracle_curve.append(float(np.mean(errors_selected[oracle_order[:retained]])))
    risk_by_count = np.cumsum(errors_selected[risk_order]) / np.arange(1, scores.size + 1)
    oracle_by_count = np.cumsum(errors_selected[oracle_order]) / np.arange(1, scores.size + 1)
    raw_excess = risk_by_count - oracle_by_count
    aurc = float(np.mean(risk_by_count))
    ause = float(np.mean(np.maximum(raw_excess, 0.0)))
    return {
        "sample_count": int(scores.size),
        "coverages": coverage.tolist(),
        "retained_counts": retained_counts,
        "risk_curve": risk_curve,
        "oracle_curve": oracle_curve,
        "aurc": aurc,
        "ause": ause,
        "aurc_definition": "empirical_mean_over_all_retained_counts",
        "ause_definition": "empirical_mean_nonnegative_excess_over_all_retained_counts",
        "full_error": float(np.mean(errors_selected)),
        "spearman": _spearman(scores, errors_selected),
        "minimum_raw_excess": float(np.min(raw_excess)),
    }


def paired_bootstrap_mean_ci(
    differences: np.ndarray,
    *,
    replicates: int = 10_000,
    seed: int = 20_260_813,
) -> dict[str, Any]:
    """Bootstrap a paired window-level mean without resampling pixels."""

    values = np.asarray(differences, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("paired bootstrap requires a finite non-empty vector")
    if replicates < 1:
        raise ValueError("replicates must be positive")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.size, size=(replicates, values.size))
    means = values[indices].mean(axis=1)
    ci = np.quantile(means, [0.025, 0.975], method="linear")
    return {
        "sample_count": int(values.size),
        "replicates": int(replicates),
        "seed": int(seed),
        "mean": float(np.mean(values)),
        "ci95": [float(ci[0]), float(ci[1])],
    }


def write_v8_risk_cache(
    destination: str | Path,
    fields: dict[str, np.ndarray],
    *,
    metadata: dict[str, Any],
) -> Path:
    """Atomically seal label-free risk fields and refuse all overwrites."""

    target = Path(destination).resolve()
    if target.exists():
        raise FileExistsError(f"refusing to overwrite V8 risk cache: {target}")
    if not fields or not all(isinstance(name, str) and name for name in fields):
        raise ValueError("risk cache fields must be a non-empty string-keyed mapping")
    arrays = {name: np.asarray(value) for name, value in fields.items()}
    if any(value.dtype == object or value.size == 0 for value in arrays.values()):
        raise ValueError("risk cache arrays must be non-empty and non-object")
    metadata = dict(metadata)
    if metadata.get("label_access") not in {None, False}:
        raise ValueError("risk cache metadata must be label blind")
    metadata.pop("label_access", None)
    if _contains_forbidden_label_key(metadata):
        raise ValueError("risk cache metadata contains a forbidden label-derived key")

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=str(target.parent)))
    try:
        arrays_path = temporary / "arrays.npz"
        np.savez_compressed(arrays_path, **arrays)
        arrays_hash = _sha256_file(arrays_path)
        payload = {
            "schema_version": V8_RISK_CACHE_SCHEMA_VERSION,
            "sealed": True,
            "sealed_at_utc": datetime.now(UTC).isoformat(),
            "label_access": False,
            "arrays_file": arrays_path.name,
            "arrays_sha256": arrays_hash,
            "arrays": {
                name: {"shape": list(value.shape), "dtype": str(value.dtype)}
                for name, value in arrays.items()
            },
            "metadata": metadata,
        }
        metadata_path = temporary / "metadata.json"
        metadata_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        seal = {
            "schema_version": V8_RISK_CACHE_SCHEMA_VERSION,
            "metadata_sha256": _sha256_file(metadata_path),
            "arrays_sha256": arrays_hash,
        }
        (temporary / "SEAL.json").write_text(
            json.dumps(seal, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temporary, target)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return target


def read_v8_risk_cache(source: str | Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Read a V8 risk cache only after checking both metadata and array seals."""

    root = Path(source).resolve()
    metadata_path = root / "metadata.json"
    seal_path = root / "SEAL.json"
    if not metadata_path.is_file() or not seal_path.is_file():
        raise FileNotFoundError("V8 risk cache requires metadata.json and SEAL.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    if (
        metadata.get("schema_version") != V8_RISK_CACHE_SCHEMA_VERSION
        or metadata.get("sealed") is not True
        or metadata.get("label_access") is not False
    ):
        raise ValueError("V8 risk cache is not a sealed label-free cache")
    if seal.get("metadata_sha256") != _sha256_file(metadata_path):
        raise ValueError("V8 risk metadata seal does not match")
    arrays_path = root / str(metadata.get("arrays_file", ""))
    if (
        not arrays_path.is_file()
        or seal.get("arrays_sha256") != _sha256_file(arrays_path)
        or metadata.get("arrays_sha256") != _sha256_file(arrays_path)
    ):
        raise ValueError("V8 risk arrays seal does not match")
    with np.load(arrays_path, allow_pickle=False) as archive:
        fields = {name: archive[name].copy() for name in archive.files}
    actual_schema = {
        name: {"shape": list(value.shape), "dtype": str(value.dtype)}
        for name, value in fields.items()
    }
    if actual_schema != metadata.get("arrays"):
        raise ValueError("V8 risk array schema does not match metadata")
    return fields, metadata


def macro_mean(records: Iterable[float | None]) -> float | None:
    values = np.asarray([value for value in records if value is not None], dtype=np.float64)
    return None if values.size == 0 else float(np.mean(values))


def _points(value: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 3 or array.shape[-1] != 3 or not np.isfinite(array).all():
        raise ValueError(f"{name} must have finite shape [H, W, 3]")
    return array


def _map(value: np.ndarray, shape: tuple[int, int], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"{name} must have finite shape {shape}")
    return array


def _valid(value: np.ndarray, shape: tuple[int, int], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.bool_)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    return array


def _spearman(first: np.ndarray, second: np.ndarray) -> float | None:
    valid = np.ones(first.shape, dtype=np.bool_)
    first_rank = midrank_percentile(first, valid).astype(np.float64)
    second_rank = midrank_percentile(second, valid).astype(np.float64)
    first_centered = first_rank - np.mean(first_rank)
    second_centered = second_rank - np.mean(second_rank)
    denominator = np.linalg.norm(first_centered) * np.linalg.norm(second_centered)
    if denominator <= 0:
        return None
    return float(np.dot(first_centered, second_centered) / denominator)


def _contains_forbidden_label_key(value: Any) -> bool:
    forbidden = ("ground_truth", "target_depth", "context_depth", "error_map", "label_metric")
    if isinstance(value, dict):
        return any(
            any(token in str(key).lower() for token in forbidden)
            or _contains_forbidden_label_key(item)
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple)):
        return any(_contains_forbidden_label_key(item) for item in value)
    return False


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
