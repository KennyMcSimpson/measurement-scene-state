"""Pinned MapAnything adapter behind the project-owned supplier contract."""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch

from mcss.geometry_supplier import (
    CodeDependency,
    GeometrySupplierRequest,
    GeometrySupplierResult,
    SupplierIdentity,
    SupplierRunConfig,
)


class _Backend(Protocol):
    def preprocess(self, views: list[dict[str, object]]) -> list[dict[str, object]]: ...

    def infer(
        self, views: list[dict[str, object]], **kwargs: object
    ) -> list[dict[str, torch.Tensor]]: ...


class MapAnythingSupplier:
    def __init__(
        self,
        *,
        identity: SupplierIdentity,
        run_config: SupplierRunConfig,
        code_root: str | Path,
        dinov2_code_root: str | Path | None = None,
        backend: _Backend | None = None,
        device: str = "cuda",
        model_cache_dir: str | Path | None = None,
    ) -> None:
        self.identity = identity
        self.run_config = run_config
        self.code_root = Path(code_root).resolve()
        self.dinov2_code_root = (
            None if dinov2_code_root is None else Path(dinov2_code_root).resolve()
        )
        self.device = device
        self.model_cache_dir = None if model_cache_dir is None else Path(model_cache_dir).resolve()
        self._backend = backend

    def infer(self, request: GeometrySupplierRequest) -> GeometrySupplierResult:
        canonical = request.canonical()
        if self._backend is None:
            self._backend = self._load_backend()
        backend = self._backend
        raw_views = []
        for index, _frame_id in enumerate(canonical.frame_ids):
            raw_views.append(
                {
                    "img": canonical.rgb[index].permute(1, 2, 0).detach().cpu(),
                    "intrinsics": canonical.cameras.intrinsics[index].detach().cpu(),
                    "camera_poses": canonical.cameras.c2w[index].detach().cpu(),
                    "is_metric_scale": True,
                }
            )
        processed = backend.preprocess(raw_views)
        predictions = backend.infer(processed, **self.run_config.inference_kwargs())
        if len(predictions) != len(canonical.frame_ids) or len(processed) != len(predictions):
            raise ValueError("MapAnything output view count does not match request")

        depth = []
        confidence = []
        valid_mask = []
        intrinsics = []
        c2w = []
        for view, prediction in zip(processed, predictions, strict=True):
            depth.append(_prediction_array(prediction, "depth_along_ray", trailing=1))
            confidence.append(_prediction_array(prediction, "conf"))
            valid_mask.append(_prediction_array(prediction, "mask", trailing=1).astype(np.bool_))
            # Only known cameras propagated through official preprocessing are trusted.
            intrinsics.append(_view_matrix(view, "intrinsics", (3, 3)))
            c2w.append(_view_matrix(view, "camera_poses", (4, 4)))
        return GeometrySupplierResult(
            frame_ids=canonical.frame_ids,
            depth_along_ray=np.stack(depth),
            confidence=np.stack(confidence),
            valid_mask=np.stack(valid_mask),
            intrinsics=np.stack(intrinsics),
            c2w=np.stack(c2w),
        )

    def _load_backend(self) -> _Backend:
        if not self.code_root.is_dir():
            raise FileNotFoundError(self.code_root)
        revision = subprocess.run(
            ["git", "-C", str(self.code_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if revision != self.identity.code_revision:
            raise RuntimeError(
                f"MapAnything code revision mismatch: expected {self.identity.code_revision}, "
                f"found {revision}"
            )
        status = subprocess.run(
            ["git", "-C", str(self.code_root), "status", "--porcelain", "--untracked-files=all"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if status:
            raise RuntimeError("MapAnything code checkout is not clean")
        dependencies = {value.name: value for value in self.identity.dependencies}
        if set(dependencies) != {"dinov2"}:
            raise RuntimeError("MapAnything requires exactly one pinned dinov2 code dependency")
        if self.dinov2_code_root is None:
            raise RuntimeError("dinov2_code_root is required for a real MapAnything backend")
        verify_code_dependency(self.dinov2_code_root, dependencies["dinov2"])
        if str(self.code_root) not in sys.path:
            sys.path.insert(0, str(self.code_root))
        from mapanything.models import MapAnything
        from mapanything.utils.image import preprocess_inputs

        kwargs: dict[str, Any] = {"revision": self.identity.model_revision}
        if self.model_cache_dir is not None:
            kwargs["cache_dir"] = str(self.model_cache_dir)
        with local_only_dinov2_hub(self.dinov2_code_root):
            model = MapAnything.from_pretrained(self.identity.model_id, **kwargs)
        model = model.to(self.device).eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        return _RealBackend(model, preprocess_inputs)


def tracked_source_sha256(code_root: str | Path) -> str:
    """Hash the exact tracked source snapshot at HEAD, independent of checkout line endings."""

    root = Path(code_root).resolve()
    archive = subprocess.run(
        ["git", "-C", str(root), "archive", "--format=tar", "HEAD"],
        check=True,
        capture_output=True,
    ).stdout
    import hashlib

    return hashlib.sha256(archive).hexdigest()


def verify_code_dependency(
    code_root: str | Path, dependency: CodeDependency
) -> dict[str, str]:
    root = Path(code_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    revision = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if revision != dependency.revision:
        raise RuntimeError(
            f"{dependency.name} code revision mismatch: expected {dependency.revision}, "
            f"found {revision}"
        )
    status = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=all"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if status:
        raise RuntimeError(f"{dependency.name} code checkout is not clean")
    source_sha256 = tracked_source_sha256(root)
    if source_sha256 != dependency.source_sha256:
        raise RuntimeError(
            f"{dependency.name} source hash mismatch: expected {dependency.source_sha256}, "
            f"found {source_sha256}"
        )
    return {
        "name": dependency.name,
        "repository": dependency.repository,
        "revision": revision,
        "source_sha256": source_sha256,
        "code_root": str(root),
    }


@contextmanager
def local_only_dinov2_hub(code_root: str | Path) -> Iterator[None]:
    """Route UniCeption's hard-coded DINOv2 hub call to a verified local checkout."""

    root = Path(code_root).resolve()
    original_load = torch.hub.load
    original_load_state_dict_from_url = torch.hub.load_state_dict_from_url

    def load_local(repo_or_dir: object, model: str, *args: object, **kwargs: object) -> object:
        if repo_or_dir != "facebookresearch/dinov2":
            raise RuntimeError(f"unexpected torch.hub repository: {repo_or_dir}")
        if kwargs.pop("force_reload", False):
            raise RuntimeError("force_reload is forbidden for the pinned local DINOv2 checkout")
        if kwargs.pop("pretrained", True) is not False:
            raise RuntimeError("independent DINOv2 weights are forbidden for MapAnything loading")
        if "source" in kwargs:
            raise RuntimeError("caller-controlled torch.hub source is forbidden")
        return original_load(str(root), model, *args, pretrained=False, source="local", **kwargs)

    def reject_weight_download(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("DINOv2 weight download is forbidden during MapAnything loading")

    torch.hub.load = load_local  # type: ignore[assignment]
    torch.hub.load_state_dict_from_url = reject_weight_download  # type: ignore[assignment]
    try:
        yield
    finally:
        torch.hub.load = original_load
        torch.hub.load_state_dict_from_url = original_load_state_dict_from_url


class _RealBackend:
    def __init__(self, model: Any, preprocess: Any) -> None:
        self.model = model
        self._preprocess = preprocess

    def preprocess(self, views: list[dict[str, object]]) -> list[dict[str, object]]:
        return self._preprocess(views)

    def infer(
        self, views: list[dict[str, object]], **kwargs: object
    ) -> list[dict[str, torch.Tensor]]:
        return self.model.infer(views, **kwargs)


def _prediction_array(
    prediction: dict[str, torch.Tensor], key: str, trailing: int | None = None
) -> np.ndarray:
    if key not in prediction:
        raise ValueError(f"MapAnything prediction is missing {key}")
    value = prediction[key].detach().float().cpu().numpy()
    if value.shape[0] != 1:
        raise ValueError(f"MapAnything {key} must have batch size one")
    value = value[0]
    if trailing is not None:
        if value.ndim != 3 or value.shape[-1] != trailing:
            raise ValueError(f"MapAnything {key} has an unexpected shape")
        value = value[..., 0]
    elif value.ndim != 2:
        raise ValueError(f"MapAnything {key} has an unexpected shape")
    if not np.isfinite(value).all():
        raise ValueError(f"MapAnything {key} must contain only finite values")
    return value


def _view_matrix(view: dict[str, object], key: str, shape: tuple[int, int]) -> np.ndarray:
    if key not in view:
        raise ValueError(f"processed MapAnything view is missing known {key}")
    value = torch.as_tensor(view[key]).detach().float().cpu().numpy()
    if value.shape == (1, *shape):
        value = value[0]
    if value.shape != shape or not np.isfinite(value).all():
        raise ValueError(f"processed known {key} has an unexpected shape or value")
    return value
