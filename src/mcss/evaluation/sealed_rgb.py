"""Context-first RGB-only evaluation for an already sealed scene.

This module deliberately does not read a dataset manifest or any target files.  A
caller supplies lazy camera and RGB callbacks through :class:`RGBQuery`.  The
callbacks are reached only after the sealed scene and every query descriptor have
passed identity, overlap, and state-hash checks.

The default metric is exact full-image PSNR.  SSIM and LPIPS are intentionally
injected by the benchmark adapter: their implementation and preprocessing are
benchmark-owned, and this module must not silently substitute the masked
Hypersim metrics used elsewhere in the repository.
"""

from __future__ import annotations

import inspect
import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, fields
from numbers import Real
from typing import Any, Protocol, TypeAlias

import numpy as np
import torch
from torch import Tensor

from mcss.dynamic.types import SealedScene, clone_scene_state, hash_scene_state
from mcss.types import Cameras, SceneState

RGBValue: TypeAlias = Tensor | np.ndarray
RGBSupplier: TypeAlias = Callable[..., RGBValue]
CameraSupplier: TypeAlias = Callable[..., Cameras]
RGBRenderer: TypeAlias = Callable[..., Tensor | Mapping[str, Tensor]]
RGBMetricFn: TypeAlias = Callable[[Tensor, Tensor], Real | Tensor | np.generic]


class RGBMetricProtocol(Protocol):
    """Protocol accepted by :func:`evaluate_sealed_rgb` for metric objects."""

    name: str
    preprocessing: str
    full_resolution: bool

    def __call__(self, prediction: Tensor, target: Tensor) -> Real | Tensor | np.generic:
        ...


@dataclass(frozen=True)
class RGBMetric:
    """One benchmark-owned metric with explicit preprocessing provenance.

    ``compute`` receives cloned channel-first RGB tensors with shape ``[3, H, W]``.
    A preprocessing callback may convert the value representation (for example to
    ``[-1, 1]`` for an official LPIPS implementation), but it must preserve the
    full spatial resolution when ``full_resolution`` is true.
    """

    name: str
    compute: RGBMetricFn
    preprocessing: str = "identity: RGB float32 in [0, 1], full resolution"
    full_resolution: bool = True
    preprocess_prediction: Callable[[Tensor], Tensor] | None = None
    preprocess_target: Callable[[Tensor], Tensor] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name or self.name.strip() != self.name:
            raise ValueError("RGB metric name must be a non-empty string")
        if not callable(self.compute):
            raise TypeError("RGB metric compute must be callable")
        if not isinstance(self.preprocessing, str) or not self.preprocessing:
            raise ValueError("RGB metric preprocessing must be documented")
        if not isinstance(self.full_resolution, bool) or not self.full_resolution:
            raise ValueError("sealed RGB metrics must declare full_resolution=True")
        for name in ("preprocess_prediction", "preprocess_target"):
            callback = getattr(self, name)
            if callback is not None and not callable(callback):
                raise TypeError(f"{name} must be callable or None")

    def evaluate(self, prediction: Tensor, target: Tensor) -> float:
        prediction_input = prediction.detach().clone()
        target_input = target.detach().clone()
        if self.preprocess_prediction is not None:
            prediction_input = self.preprocess_prediction(prediction_input)
        if self.preprocess_target is not None:
            target_input = self.preprocess_target(target_input)
        if not isinstance(prediction_input, Tensor) or not isinstance(target_input, Tensor):
            raise TypeError(f"RGB metric {self.name!r} preprocessing must return tensors")
        if prediction_input.shape != prediction.shape or target_input.shape != target.shape:
            raise ValueError(
                f"RGB metric {self.name!r} preprocessing must preserve full resolution"
            )
        value = self.compute(prediction_input, target_input)
        if isinstance(value, Tensor):
            if value.numel() != 1:
                raise ValueError(f"RGB metric {self.name!r} must return a scalar")
            value = value.detach().cpu().item()
        elif isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(f"RGB metric {self.name!r} must return a real scalar")
        value = float(value)
        if math.isnan(value):
            raise ValueError(f"RGB metric {self.name!r} returned NaN")
        return value


@dataclass(frozen=True)
class RGBQuery:
    """Lazy query descriptor used by the sealed RGB evaluator.

    Camera suppliers return one world-space camera.  RGB suppliers return one
    channel-first ``[3, H, W]`` image, or an equivalent ``[H, W, 3]`` NumPy/PIL
    array representation.  Neither callback receives the mutable scene state.
    The optional ``image_size`` is a declaration checked against both results;
    when omitted it is inferred from the supplied camera before the RGB callback
    is invoked.
    """

    frame_id: int
    episode_id: str
    scene_id: str
    split_id: str
    query_vault_id: str
    camera_supplier: CameraSupplier
    rgb_supplier: RGBSupplier
    image_size: tuple[int, int] | None = None

    @classmethod
    def for_sealed(
        cls,
        sealed: SealedScene,
        frame_id: int,
        camera_supplier: CameraSupplier,
        rgb_supplier: RGBSupplier,
        image_size: tuple[int, int] | None = None,
    ) -> RGBQuery:
        """Construct a query with all identity fields copied from ``sealed``."""

        return cls(
            frame_id,
            sealed.episode_id,
            sealed.scene_id,
            sealed.split_id,
            sealed.query_vault_id,
            camera_supplier,
            rgb_supplier,
            image_size,
        )


# Names used by benchmark adapters can remain descriptive without duplicating the
# implementation.  Keep the aliases public for callers that prefer an explicit
# sealed-query name.
SealedRGBQuery = RGBQuery
RGBQuerySpec = RGBQuery
RGBQueryPreprocessor: TypeAlias = Callable[[RGBQuery], RGBQuery]


def psnr_full_image(prediction: Tensor, target: Tensor) -> float:
    """Compute the standard full-image RGB PSNR on ``[0, 1]`` values.

    No support mask, crop, resize, channel weighting, or error floor is applied.
    The zero-error case is represented by positive infinity, as in the official
    PSNR definition.
    """

    _validate_metric_image_pair(prediction, target)
    prediction = prediction.float().clamp(0.0, 1.0)
    target = target.float().clamp(0.0, 1.0)
    mse = float((prediction - target).square().mean().item())
    return float("inf") if mse == 0.0 else -10.0 * math.log10(mse)


# Short alias for metric registries and tests.
psnr = psnr_full_image


def official_psnr_metric() -> RGBMetric:
    """Return the tttLRM PSNR metric with its exact preprocessing declaration."""

    return RGBMetric(
        "psnr",
        psnr_full_image,
        preprocessing="tttLRM: clamp prediction and target to [0, 1]; full-resolution RGB MSE",
    )


def official_ssim_metric() -> RGBMetric:
    """Return the tttLRM SSIM metric, importing scikit-image lazily."""

    def compute(prediction: Tensor, target: Tensor) -> float:
        try:
            from skimage.metrics import structural_similarity
        except ImportError as error:
            raise RuntimeError(
                "official tttLRM SSIM requires scikit-image; install the evaluation extra"
            ) from error
        prediction_np = prediction.float().clamp(0.0, 1.0).cpu().numpy()
        target_np = target.float().clamp(0.0, 1.0).cpu().numpy()
        return float(
            structural_similarity(
                prediction_np,
                target_np,
                channel_axis=0,
                data_range=1.0,
            )
        )

    return RGBMetric(
        "ssim",
        compute,
        preprocessing=(
            "tttLRM: clamp prediction and target to [0, 1]; "
            "skimage.metrics.structural_similarity(channel_axis=0, data_range=1.0), "
            "default settings, full resolution"
        ),
    )


def official_lpips_metric() -> RGBMetric:
    """Return the tttLRM VGG-LPIPS metric with lazy model construction.

    The model is created on first use to keep importing this module free of weight
    downloads.  No resize or crop is performed here: the benchmark preprocessor
    must provide the declared 536x960 query images before evaluation.
    """

    model: Any = None

    def compute(prediction: Tensor, target: Tensor) -> float:
        nonlocal model
        try:
            import lpips
        except ImportError as error:
            raise RuntimeError(
                "official tttLRM LPIPS requires lpips; install the evaluation extra"
            ) from error
        if model is None:
            model = lpips.LPIPS(net="vgg")
            model.eval()
        model = model.to(device=prediction.device)
        with torch.no_grad():
            value = model(
                prediction.float().clamp(0.0, 1.0).unsqueeze(0),
                target.float().clamp(0.0, 1.0).unsqueeze(0),
                normalize=True,
            )
        return float(value.detach().mean().item())

    return RGBMetric(
        "lpips",
        compute,
        preprocessing=(
            "tttLRM: clamp prediction and target to [0, 1]; "
            "lpips.LPIPS(net='vgg') with normalize=True, full resolution"
        ),
    )


def official_tttlrm_metrics(*, include_lpips: bool = True) -> tuple[RGBMetric, ...]:
    """Build the official tttLRM metric set in reported order.

    ``include_lpips=False`` is an explicit caller choice for environments without
    the optional LPIPS dependency; the default retains LPIPS in the metric set and
    raises a clear error on use if the dependency or weights are unavailable.
    """

    metrics: list[RGBMetric] = [official_psnr_metric(), official_ssim_metric()]
    if include_lpips:
        metrics.append(official_lpips_metric())
    return tuple(metrics)


# Common adapter spellings retained as explicit aliases.
tttlrm_metrics = official_tttlrm_metrics
tttLRM_metrics = official_tttlrm_metrics
official_tttLRM_metrics = official_tttlrm_metrics


def evaluate_sealed_rgb(
    sealed: SealedScene,
    queries: Iterable[RGBQuery],
    renderer: RGBRenderer,
    *,
    metrics: Sequence[RGBMetric] | Mapping[str, RGBMetricFn | RGBMetric] | None = None,
    device: str | torch.device | None = None,
    query_preprocessor: RGBQueryPreprocessor | None = None,
) -> dict[str, Any]:
    """Evaluate one sealed scene's RGB queries without scene-level aggregation.

    The complete query descriptor list is validated before any camera or RGB
    supplier is called.  The sealed state is cloned for rendering, while the
    original state and anchor are hash-checked before and after every callback and
    render.  A target callback that mutates the sealed state is rejected and the
    original tensors are restored from the pre-evaluation snapshot before the
    exception is raised.

    ``renderer`` may accept ``(state, cameras)`` or the existing parameter-free
    renderer signature ``(state, cameras, {"rgb"})``.  It may return an RGB tensor
    directly or a mapping containing an ``"rgb"`` tensor.  Only that RGB output is
    consumed; depth and other modalities are neither requested nor fabricated.

    The return value contains per-image flat metric records.  Equal-scene pooling
    belongs to the benchmark aggregator and is intentionally absent here.
    """

    state_snapshot, seal_snapshot, anchor_snapshot = _validate_sealed(sealed)
    query_list = tuple(queries)
    _validate_queries(query_list, sealed)
    if query_preprocessor is not None and not callable(query_preprocessor):
        raise TypeError("query_preprocessor must be callable or None")
    metric_list = _normalise_metrics(metrics)
    metric_spec = {
        metric.name: {
            "preprocessing": metric.preprocessing,
            "full_resolution": metric.full_resolution,
        }
        for metric in metric_list
    }
    if len(metric_spec) != len(metric_list):
        raise ValueError("RGB metric names must be unique")

    target_device = (
        torch.device(device)
        if device is not None
        else sealed.scene_state.density_logits.device
    )
    render_state = clone_scene_state(sealed.scene_state).to(target_device)
    render_hash = hash_scene_state(render_state)
    anchor_inverse = torch.linalg.inv(
        anchor_snapshot.to(device=target_device, dtype=render_state.density_logits.dtype)
    )
    records: list[dict[str, Any]] = []
    try:
        if query_preprocessor is not None:
            query_list = tuple(query_preprocessor(query) for query in query_list)
            _assert_sealed_unchanged(sealed, seal_snapshot, anchor_snapshot)
            _validate_queries(query_list, sealed)
        with torch.no_grad():
            for query in query_list:
                supplied_camera = _invoke_supplier(query.camera_supplier, query)
                _assert_sealed_unchanged(sealed, seal_snapshot, anchor_snapshot)
                camera = _prepare_camera(supplied_camera, query, anchor_inverse, target_device)
                _assert_render_state_unchanged(render_state, render_hash)

                supplied_rgb = _invoke_supplier(query.rgb_supplier, query)
                _assert_sealed_unchanged(sealed, seal_snapshot, anchor_snapshot)
                target_rgb = _normalise_rgb(supplied_rgb, camera.image_size, target_device)

                rendered = _invoke_renderer(renderer, render_state, camera)
                _assert_sealed_unchanged(sealed, seal_snapshot, anchor_snapshot)
                _assert_render_state_unchanged(render_state, render_hash)
                prediction_rgb = _normalise_rgb(rendered, camera.image_size, target_device)
                if prediction_rgb.shape != target_rgb.shape:
                    raise ValueError(
                        f"RGB prediction/target shape mismatch for frame {query.frame_id}: "
                        f"{tuple(prediction_rgb.shape)} != {tuple(target_rgb.shape)}"
                    )
                metrics_for_image = {
                    metric.name: metric.evaluate(
                        prediction_rgb[0, 0], target_rgb[0, 0]
                    )
                    for metric in metric_list
                }
                records.append({"frame_id": query.frame_id, **metrics_for_image})

        _assert_sealed_unchanged(sealed, seal_snapshot, anchor_snapshot)
        _assert_render_state_unchanged(render_state, render_hash)
    except BaseException:
        _restore_sealed_scene(sealed, state_snapshot, anchor_snapshot, seal_snapshot)
        raise

    return {
        "episode_id": sealed.episode_id,
        "scene_id": sealed.scene_id,
        "split_id": sealed.split_id,
        "query_vault_id": sealed.query_vault_id,
        "state_hash": seal_snapshot["state_hash"],
        "metric_spec": metric_spec,
        "per_image": records,
        "per_query": records,
    }


# Descriptive compatibility aliases for adapters that name the operation in a
# different order.
evaluate_rgb_sealed = evaluate_sealed_rgb
evaluate_sealed_rgb_only = evaluate_sealed_rgb


def _normalise_metrics(
    metrics: Sequence[RGBMetric] | Mapping[str, RGBMetricFn | RGBMetric] | None,
) -> tuple[RGBMetric, ...]:
    if metrics is None:
        return (RGBMetric("psnr", psnr_full_image),)
    if isinstance(metrics, Mapping):
        values: list[RGBMetric] = []
        for name, value in metrics.items():
            if isinstance(value, RGBMetric):
                if value.name != name:
                    raise ValueError(
                        f"RGB metric mapping key {name!r} does not match metric name {value.name!r}"
                    )
                values.append(value)
            elif callable(value):
                values.append(RGBMetric(str(name), value))
            else:
                raise TypeError(f"RGB metric {name!r} must be callable or RGBMetric")
        if not values:
            raise ValueError("RGB metric mapping must not be empty")
        return tuple(values)
    values = tuple(metrics)
    if not values or not all(isinstance(metric, RGBMetric) for metric in values):
        raise TypeError("metrics must be a non-empty RGBMetric sequence")
    return values


def _validate_sealed(
    sealed: SealedScene,
) -> tuple[SceneState, dict[str, Any], Tensor]:
    if not isinstance(sealed, SealedScene):
        raise TypeError("evaluate_sealed_rgb requires a SealedScene")
    for name in (
        "episode_id",
        "scene_id",
        "split_id",
        "query_vault_id",
        "state_hash",
        "checkpoint_hash",
        "config_hash",
        "fast_state_hash",
    ):
        _require_identifier(getattr(sealed, name), name)
    if not sealed.observed_ids or any(
        type(item) is not int or item < 0 for item in sealed.observed_ids
    ):
        raise ValueError("sealed observed_ids must be non-empty non-negative integers")
    if len(sealed.observed_ids) != len(set(sealed.observed_ids)):
        raise ValueError("sealed observed_ids must be unique")
    anchor = sealed.anchor_c2w
    if not isinstance(anchor, Tensor) or anchor.shape != (4, 4) or not torch.isfinite(anchor).all():
        raise ValueError("sealed anchor_c2w must be a finite 4x4 transform")
    expected_row = torch.tensor([0.0, 0.0, 0.0, 1.0], device=anchor.device, dtype=anchor.dtype)
    if not torch.allclose(anchor[3], expected_row, atol=1e-5, rtol=1e-5):
        raise ValueError("sealed anchor_c2w must have a homogeneous final row")
    try:
        torch.linalg.inv(anchor)
    except torch.linalg.LinAlgError as error:
        raise ValueError("sealed anchor_c2w must be invertible") from error
    state_hash = hash_scene_state(sealed.scene_state)
    if state_hash != sealed.state_hash:
        raise ValueError("sealed scene state_hash does not match the current scene_state")
    state_snapshot = clone_scene_state(sealed.scene_state)
    anchor_snapshot = anchor.detach().clone()
    seal_snapshot = {
        "episode_id": sealed.episode_id,
        "scene_id": sealed.scene_id,
        "split_id": sealed.split_id,
        "query_vault_id": sealed.query_vault_id,
        "state_hash": sealed.state_hash,
        "checkpoint_hash": sealed.checkpoint_hash,
        "config_hash": sealed.config_hash,
        "fast_state_hash": sealed.fast_state_hash,
        "observed_ids": tuple(sealed.observed_ids),
        "scene_state_ref": sealed.scene_state,
        "anchor_ref": sealed.anchor_c2w,
    }
    return state_snapshot, seal_snapshot, anchor_snapshot


def _validate_queries(queries: Sequence[RGBQuery], sealed: SealedScene) -> None:
    if not queries:
        raise ValueError("sealed RGB evaluation requires at least one query")
    seen: set[int] = set()
    observed = set(sealed.observed_ids)
    for query in queries:
        if not isinstance(query, RGBQuery):
            raise TypeError("sealed RGB queries must be RGBQuery instances")
        if type(query.frame_id) is not int or query.frame_id < 0:
            raise ValueError("RGB query frame_id must be a non-negative integer")
        if query.frame_id in seen or query.frame_id in observed:
            raise ValueError(
                "RGB query frame_ids must be unique and disjoint from sealed observed_ids"
            )
        seen.add(query.frame_id)
        for name in ("episode_id", "scene_id", "split_id", "query_vault_id"):
            value = getattr(query, name)
            _require_identifier(value, f"query {name}")
            if value != getattr(sealed, name):
                raise ValueError(f"query {name} does not match sealed scene")
        if not callable(query.camera_supplier) or not callable(query.rgb_supplier):
            raise TypeError("RGB query camera_supplier and rgb_supplier must be callable")
        if query.image_size is not None:
            _validate_image_size(query.image_size)
    frame_ids = [query.frame_id for query in queries]
    if frame_ids != sorted(frame_ids):
        raise ValueError("RGB query frame_ids must be in ascending order")


def _prepare_camera(
    supplied: Cameras,
    query: RGBQuery,
    world_to_anchor: Tensor,
    device: torch.device,
) -> Cameras:
    if not isinstance(supplied, Cameras):
        raise TypeError("RGB camera_supplier must return Cameras")
    if supplied.leading_shape not in ((), (1,), (1, 1)):
        raise ValueError("RGB camera_supplier must return exactly one camera")
    if query.image_size is not None and supplied.image_size != query.image_size:
        raise ValueError("RGB camera image_size does not match query image_size")
    camera = supplied.to(device=device)
    intrinsics = camera.intrinsics
    c2w = camera.c2w
    if camera.leading_shape == ():
        intrinsics = intrinsics.unsqueeze(0).unsqueeze(0)
        c2w = c2w.unsqueeze(0).unsqueeze(0)
    elif camera.leading_shape == (1,):
        intrinsics = intrinsics.unsqueeze(0)
        c2w = c2w.unsqueeze(0)
    world_to_anchor = world_to_anchor.to(device=device, dtype=c2w.dtype)
    local_c2w = torch.matmul(world_to_anchor, c2w)
    anchored = Cameras(intrinsics, local_c2w, camera.image_size)
    if anchored.leading_shape != (1, 1):
        raise RuntimeError("anchored RGB camera must have shape [1, 1, 4, 4]")
    return anchored


def _normalise_rgb(
    value: Any,
    declared_size: tuple[int, int] | None,
    device: torch.device,
) -> Tensor:
    if isinstance(value, Mapping):
        if "rgb" not in value:
            raise ValueError("RGB renderer mapping must contain an 'rgb' output")
        value = value["rgb"]
    if isinstance(value, Tensor):
        tensor = value.detach().clone()
    elif isinstance(value, np.ndarray):
        tensor = torch.from_numpy(np.asarray(value).copy())
    else:
        try:
            from PIL import Image

            if isinstance(value, Image.Image):
                tensor = torch.from_numpy(np.asarray(value.convert("RGB")).copy())
            else:
                raise TypeError
        except ImportError as error:
            raise TypeError("RGB supplier must return a tensor or NumPy/PIL RGB image") from error
        except TypeError as error:
            raise TypeError("RGB supplier must return a tensor or NumPy/PIL RGB image") from error
    if tensor.ndim == 5 and tensor.shape[:2] == (1, 1):
        tensor = tensor[0, 0]
    elif tensor.ndim == 4 and tensor.shape[0] == 1:
        tensor = tensor[0]
    if tensor.ndim != 3:
        raise ValueError("RGB values must have shape [3, H, W] or [H, W, 3]")
    if tensor.shape[0] == 3:
        channel_first = tensor
    elif tensor.shape[-1] == 3:
        channel_first = tensor.permute(2, 0, 1)
    else:
        raise ValueError("RGB values require exactly three channels")
    if not channel_first.is_floating_point():
        channel_first = channel_first.float()
        if channel_first.max().item() > 1.0:
            channel_first = channel_first / 255.0
    channel_first = channel_first.to(device=device, dtype=torch.float32).contiguous()
    image_size = (int(channel_first.shape[-2]), int(channel_first.shape[-1]))
    if declared_size is not None and image_size != declared_size:
        raise ValueError(f"RGB image_size does not match declared image_size: {image_size}")
    if not torch.isfinite(channel_first).all():
        raise ValueError("RGB values must be finite")
    return channel_first.unsqueeze(0).unsqueeze(0)


def _invoke_supplier(supplier: Callable[..., Any], query: RGBQuery) -> Any:
    signature = inspect.signature(supplier)
    try:
        signature.bind()
    except TypeError:
        try:
            signature.bind(query)
        except TypeError as error:
            raise TypeError("RGB suppliers must accept zero arguments or one RGBQuery") from error
        return supplier(query)
    return supplier()


def _invoke_renderer(renderer: RGBRenderer, state: SceneState, camera: Cameras) -> Any:
    signature = inspect.signature(renderer)
    try:
        signature.bind(state, camera, {"rgb"})
    except TypeError:
        try:
            signature.bind(state, camera)
        except TypeError as error:
            raise TypeError(
                "RGB renderer must accept (state, cameras) or (state, cameras, measurements)"
            ) from error
        return renderer(state, camera)
    return renderer(state, camera, {"rgb"})


def _validate_metric_image_pair(prediction: Tensor, target: Tensor) -> None:
    if not isinstance(prediction, Tensor) or not isinstance(target, Tensor):
        raise TypeError("RGB metrics require tensor inputs")
    if prediction.ndim != 3 or target.ndim != 3 or prediction.shape[0] != 3 or target.shape[0] != 3:
        raise ValueError("RGB metrics require channel-first tensors with shape [3, H, W]")
    if prediction.shape != target.shape:
        raise ValueError("RGB metric prediction and target shapes must match")
    if not torch.isfinite(prediction).all() or not torch.isfinite(target).all():
        raise ValueError("RGB metric inputs must be finite")


def _assert_sealed_unchanged(
    sealed: SealedScene,
    snapshot: Mapping[str, Any],
    anchor_snapshot: Tensor,
) -> None:
    if sealed.scene_state is not snapshot["scene_state_ref"]:
        raise RuntimeError("sealed scene_state reference was replaced during RGB evaluation")
    if sealed.anchor_c2w is not snapshot["anchor_ref"]:
        raise RuntimeError("sealed anchor_c2w reference was replaced during RGB evaluation")
    if any(
        getattr(sealed, name) != value
        for name, value in snapshot.items()
        if name
        not in {"state_hash", "scene_state_ref", "anchor_ref"}
    ):
        raise RuntimeError("sealed scene identity was modified during RGB evaluation")
    if sealed.state_hash != snapshot["state_hash"] or hash_scene_state(
        sealed.scene_state
    ) != snapshot["state_hash"]:
        raise RuntimeError("sealed scene_state was modified during RGB evaluation")
    if not torch.equal(sealed.anchor_c2w, anchor_snapshot):
        raise RuntimeError("sealed anchor_c2w was modified during RGB evaluation")


def _restore_sealed_scene(
    sealed: SealedScene,
    state_snapshot: SceneState,
    anchor_snapshot: Tensor,
    seal_snapshot: Mapping[str, Any],
) -> None:
    original_state = seal_snapshot["scene_state_ref"]
    original_anchor = seal_snapshot["anchor_ref"]
    if sealed.scene_state is not original_state:
        object.__setattr__(sealed, "scene_state", original_state)
    if sealed.anchor_c2w is not original_anchor:
        object.__setattr__(sealed, "anchor_c2w", original_anchor)
    _restore_scene_state(original_state, state_snapshot)
    _restore_tensor(original_anchor, anchor_snapshot)
    for name in (
        "episode_id",
        "scene_id",
        "split_id",
        "query_vault_id",
        "state_hash",
        "checkpoint_hash",
        "config_hash",
        "fast_state_hash",
        "observed_ids",
    ):
        expected = seal_snapshot[name]
        if getattr(sealed, name) != expected:
            object.__setattr__(sealed, name, expected)


def _assert_render_state_unchanged(state: SceneState, expected_hash: str) -> None:
    if hash_scene_state(state) != expected_hash:
        raise RuntimeError("RGB renderer modified its private scene state")


def _restore_tensor(target: Tensor, snapshot: Tensor) -> None:
    if (
        target.shape == snapshot.shape
        and target.device == snapshot.device
        and target.dtype == snapshot.dtype
    ):
        target.copy_(snapshot)


def _restore_scene_state(target: SceneState, snapshot: SceneState) -> None:
    for field in fields(target):
        current = getattr(target, field.name)
        saved = getattr(snapshot, field.name)
        if isinstance(current, Tensor) and isinstance(saved, Tensor):
            if (
                current.shape == saved.shape
                and current.device == saved.device
                and current.dtype == saved.dtype
            ):
                current.copy_(saved)
            else:
                setattr(target, field.name, saved.detach().clone())
        elif current is not None and saved is not None and hasattr(current, "__dataclass_fields__"):
            _restore_dataclass(current, saved)
        elif current != saved:
            setattr(target, field.name, saved)


def _restore_dataclass(target: Any, snapshot: Any) -> None:
    for field in fields(target):
        current = getattr(target, field.name)
        saved = getattr(snapshot, field.name)
        if isinstance(current, Tensor) and isinstance(saved, Tensor):
            if (
                current.shape == saved.shape
                and current.device == saved.device
                and current.dtype == saved.dtype
            ):
                current.copy_(saved)
            else:
                setattr(target, field.name, saved.detach().clone())
        elif current != saved:
            setattr(target, field.name, saved)


def _validate_image_size(value: tuple[int, int]) -> tuple[int, int]:
    if (
        not isinstance(value, tuple)
        or len(value) != 2
        or any(type(item) is not int or item <= 0 for item in value)
    ):
        raise ValueError("RGB image_size must be a (height, width) tuple of positive integers")
    return value


def _require_identifier(value: Any, name: str) -> None:
    if not isinstance(value, str) or not value or any(char in value for char in "\\/"):
        raise ValueError(f"{name} must be a simple non-empty identifier")


__all__ = [
    "RGBMetric",
    "RGBQuery",
    "RGBQueryPreprocessor",
    "RGBQuerySpec",
    "SealedRGBQuery",
    "evaluate_rgb_sealed",
    "evaluate_sealed_rgb",
    "evaluate_sealed_rgb_only",
    "official_lpips_metric",
    "official_psnr_metric",
    "official_ssim_metric",
    "official_tttlrm_metrics",
    "official_tttLRM_metrics",
    "psnr",
    "psnr_full_image",
]
