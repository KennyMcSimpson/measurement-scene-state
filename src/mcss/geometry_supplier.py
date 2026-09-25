"""Label-free geometry supplier contracts and structured atomic caches."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Self

import numpy as np
import torch
from torch import Tensor

from mcss.types import Cameras

SUPPLIER_CACHE_SCHEMA_VERSION = 2


@dataclass(frozen=True)
class CodeDependency:
    name: str
    repository: str
    revision: str
    source_sha256: str

    def __post_init__(self) -> None:
        for name in ("name", "repository", "revision"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        if not _is_sha256(self.source_sha256):
            raise ValueError("source_sha256 must be a lowercase SHA-256 digest")


@dataclass(frozen=True)
class SupplierIdentity:
    model_id: str
    model_revision: str
    code_repository: str
    code_revision: str
    dependencies: tuple[CodeDependency, ...] = ()

    def __post_init__(self) -> None:
        for name in ("model_id", "model_revision", "code_repository", "code_revision"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        if not isinstance(self.dependencies, tuple) or not all(
            isinstance(value, CodeDependency) for value in self.dependencies
        ):
            raise ValueError("dependencies must be a tuple of CodeDependency values")
        names = [value.name for value in self.dependencies]
        if len(names) != len(set(names)):
            raise ValueError("dependency names must be unique")

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "code_repository": self.code_repository,
            "code_revision": self.code_revision,
            "dependencies": [asdict(value) for value in self.dependencies],
        }


@dataclass(frozen=True)
class SupplierRunConfig:
    memory_efficient_inference: bool = True
    minibatch_size: int = 1
    use_amp: bool = True
    amp_dtype: str = "bf16"
    apply_mask: bool = True
    mask_edges: bool = True
    apply_confidence_mask: bool = False
    use_multiview_confidence: bool = False

    def __post_init__(self) -> None:
        if self.minibatch_size < 1:
            raise ValueError("minibatch_size must be positive")
        if self.amp_dtype not in {"bf16", "fp16", "fp32"}:
            raise ValueError("amp_dtype must be bf16, fp16, or fp32")

    def inference_kwargs(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class GeometrySupplierRequest:
    """The complete and deliberately label-free supplier input."""

    frame_ids: tuple[int, ...]
    rgb: Tensor
    cameras: Cameras

    def __post_init__(self) -> None:
        self.frame_ids = tuple(int(value) for value in self.frame_ids)
        if not self.frame_ids or any(value < 0 for value in self.frame_ids):
            raise ValueError("frame_ids must be non-empty nonnegative integers")
        if len(set(self.frame_ids)) != len(self.frame_ids):
            raise ValueError("frame_ids must be unique")
        if self.rgb.ndim != 4 or self.rgb.shape[1] != 3:
            raise ValueError("rgb must have shape [V, 3, H, W]")
        view_count = len(self.frame_ids)
        if self.rgb.shape[0] != view_count or self.cameras.leading_shape != (view_count,):
            raise ValueError("frame IDs, RGB, and cameras must share the view dimension")
        if tuple(self.rgb.shape[-2:]) != self.cameras.image_size:
            raise ValueError("RGB and cameras must share image size")
        if not torch.isfinite(self.rgb).all():
            raise ValueError("rgb must contain only finite values")
        if (self.rgb < 0).any() or (self.rgb > 1).any():
            raise ValueError("rgb must be normalized to [0, 1]")

    def canonical(self) -> Self:
        order = sorted(range(len(self.frame_ids)), key=self.frame_ids.__getitem__)
        if order == list(range(len(order))):
            return self
        indices = torch.tensor(order, dtype=torch.long, device=self.rgb.device)
        return type(self)(
            tuple(self.frame_ids[index] for index in order),
            self.rgb.index_select(0, indices),
            self.cameras.select_views(indices),
        )

    @property
    def input_sha256(self) -> str:
        canonical = self.canonical()
        digest = hashlib.sha256()
        digest.update(b"mcss-geometry-supplier-input-v1\0")
        digest.update(np.asarray(canonical.frame_ids, dtype="<i8").tobytes())
        for value in (
            canonical.rgb,
            canonical.cameras.intrinsics,
            canonical.cameras.c2w,
        ):
            _hash_tensor(digest, value)
        digest.update(np.asarray(canonical.cameras.image_size, dtype="<i8").tobytes())
        return digest.hexdigest()


@dataclass
class GeometrySupplierResult:
    frame_ids: tuple[int, ...]
    depth_along_ray: np.ndarray
    confidence: np.ndarray
    valid_mask: np.ndarray
    intrinsics: np.ndarray
    c2w: np.ndarray

    def __post_init__(self) -> None:
        self.frame_ids = tuple(int(value) for value in self.frame_ids)
        self.depth_along_ray = np.asarray(self.depth_along_ray, dtype=np.float32)
        self.confidence = np.asarray(self.confidence, dtype=np.float32)
        self.valid_mask = np.asarray(self.valid_mask, dtype=np.bool_)
        self.intrinsics = np.asarray(self.intrinsics, dtype=np.float32)
        self.c2w = np.asarray(self.c2w, dtype=np.float32)
        if not self.frame_ids or len(set(self.frame_ids)) != len(self.frame_ids):
            raise ValueError("result frame_ids must be non-empty and unique")
        view_count = len(self.frame_ids)
        reference_shape = self.depth_along_ray.shape
        if len(reference_shape) != 3 or reference_shape[0] != view_count:
            raise ValueError("depth_along_ray must have shape [V, H, W]")
        if self.confidence.shape != reference_shape or self.valid_mask.shape != reference_shape:
            raise ValueError("confidence and valid_mask shape must match depth_along_ray")
        if self.intrinsics.shape != (view_count, 3, 3):
            raise ValueError("intrinsics shape must be [V, 3, 3]")
        if self.c2w.shape != (view_count, 4, 4):
            raise ValueError("c2w shape must be [V, 4, 4]")
        finite = (
            np.isfinite(self.depth_along_ray).all()
            and np.isfinite(self.confidence).all()
            and np.isfinite(self.intrinsics).all()
            and np.isfinite(self.c2w).all()
        )
        if not finite:
            raise ValueError("supplier arrays must contain only finite values")
        if np.any(self.intrinsics[:, (0, 1), (0, 1)] <= 0):
            raise ValueError("supplier intrinsics require positive focal lengths")
        if not np.allclose(self.c2w[:, 3], np.asarray([0.0, 0.0, 0.0, 1.0]), atol=1e-5):
            raise ValueError("supplier c2w requires homogeneous final rows")

    @property
    def image_size(self) -> tuple[int, int]:
        return int(self.depth_along_ray.shape[1]), int(self.depth_along_ray.shape[2])

    def canonical(self) -> Self:
        order = np.argsort(np.asarray(self.frame_ids), kind="stable")
        if np.array_equal(order, np.arange(len(order))):
            return self
        return type(self)(
            tuple(self.frame_ids[int(index)] for index in order),
            self.depth_along_ray[order],
            self.confidence[order],
            self.valid_mask[order],
            self.intrinsics[order],
            self.c2w[order],
        )


def write_supplier_cache(
    destination: str | Path,
    result: GeometrySupplierResult,
    identity: SupplierIdentity,
    input_sha256: str,
    run_config: SupplierRunConfig,
) -> Path:
    """Write an overwrite-refusing JSON+NPZ cache through an atomic directory rename."""

    target = Path(destination).resolve()
    if target.exists():
        raise FileExistsError(f"refusing to overwrite supplier cache: {target}")
    if not _is_sha256(input_sha256):
        raise ValueError("input_sha256 must be a lowercase SHA-256 digest")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=str(target.parent))
    )
    canonical = result.canonical()
    try:
        arrays_path = temporary / "arrays.npz"
        np.savez_compressed(
            arrays_path,
            frame_ids=np.asarray(canonical.frame_ids, dtype=np.int64),
            depth_along_ray=canonical.depth_along_ray,
            confidence=canonical.confidence,
            valid_mask=canonical.valid_mask,
            intrinsics=canonical.intrinsics,
            c2w=canonical.c2w,
        )
        metadata = {
            "schema_version": SUPPLIER_CACHE_SCHEMA_VERSION,
            "identity": identity.to_dict(),
            "run_config": asdict(run_config),
            "input_sha256": input_sha256,
            "frame_ids": list(canonical.frame_ids),
            "arrays_file": arrays_path.name,
            "arrays_sha256": _sha256_file(arrays_path),
            "arrays": {
                name: {"shape": list(value.shape), "dtype": str(value.dtype)}
                for name, value in _result_arrays(canonical).items()
            },
        }
        (temporary / "metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temporary, target)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return target


def read_supplier_cache(
    source: str | Path,
) -> tuple[GeometrySupplierResult, dict[str, Any]]:
    root = Path(source).resolve()
    metadata_path = root / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(metadata_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if (
        not isinstance(metadata, dict)
        or metadata.get("schema_version") != SUPPLIER_CACHE_SCHEMA_VERSION
    ):
        raise ValueError("supplier cache metadata has an unsupported schema")
    arrays_path = root / str(metadata.get("arrays_file", ""))
    if not arrays_path.is_file():
        raise FileNotFoundError(arrays_path)
    if _sha256_file(arrays_path) != metadata.get("arrays_sha256"):
        raise ValueError("supplier cache arrays hash does not match metadata")
    with np.load(arrays_path, allow_pickle=False) as arrays:
        required = {
            "frame_ids",
            "depth_along_ray",
            "confidence",
            "valid_mask",
            "intrinsics",
            "c2w",
        }
        if set(arrays.files) != required:
            raise ValueError("supplier cache array keys do not match schema")
        frame_ids = tuple(int(value) for value in arrays["frame_ids"].tolist())
        result = GeometrySupplierResult(
            frame_ids,
            arrays["depth_along_ray"].copy(),
            arrays["confidence"].copy(),
            arrays["valid_mask"].copy(),
            arrays["intrinsics"].copy(),
            arrays["c2w"].copy(),
        )
    if list(frame_ids) != metadata.get("frame_ids"):
        raise ValueError("supplier cache metadata frame IDs do not match arrays")
    expected_schema = metadata.get("arrays")
    actual_schema = {
        name: {"shape": list(value.shape), "dtype": str(value.dtype)}
        for name, value in _result_arrays(result).items()
    }
    if expected_schema != actual_schema:
        raise ValueError("supplier cache metadata array schema does not match arrays")
    return result, metadata


def _result_arrays(result: GeometrySupplierResult) -> dict[str, np.ndarray]:
    return {
        "depth_along_ray": result.depth_along_ray,
        "confidence": result.confidence,
        "valid_mask": result.valid_mask,
        "intrinsics": result.intrinsics,
        "c2w": result.c2w,
    }


def _hash_tensor(digest: Any, value: Tensor) -> None:
    array = value.detach().cpu().contiguous().numpy()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
    digest.update(array.tobytes())


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)
