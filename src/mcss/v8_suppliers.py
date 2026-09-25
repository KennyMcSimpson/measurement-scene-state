"""Adapters for the frozen V8 geometry-supplier qualification branch."""

from __future__ import annotations

import contextlib
import importlib
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from mcss.geometry_supplier import GeometrySupplierRequest
from mcss.types import Cameras
from mcss.v8_geometry import (
    V8Proposal,
    depth_z_to_points_cam,
    emvsnet_mm_to_m,
)

EMVSNET_DEPTH_NEAR_M = 0.10
EMVSNET_DEPTH_FAR_M = 20.0
EMVSNET_METRES_TO_CHECKPOINT_UNITS = 1000.0


def center_emvsnet_images(images: torch.Tensor, *, epsilon: float = 1e-8) -> torch.Tensor:
    """Apply the released EMVSNet per-image, per-channel center normalization."""

    if images.ndim != 4 or images.shape[1] != 3:
        raise ValueError("EMVSNet images must have shape [V, 3, H, W]")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    values = images.to(dtype=torch.float32)
    mean = values.mean(dim=(-2, -1), keepdim=True)
    variance = values.var(dim=(-2, -1), correction=0, keepdim=True)
    return (values - mean) / (torch.sqrt(variance) + epsilon)


def emvsnet_projection_coordinates(
    cameras: Cameras, *, translation_scale: float
) -> torch.Tensor:
    """Build EMVSNet P=K[R|t] matrices in an explicitly scaled metric space."""

    if not np.isfinite(translation_scale) or translation_scale <= 0:
        raise ValueError("translation_scale must be positive and finite")
    c2w = cameras.c2w.clone()
    c2w[..., :3, 3] *= float(translation_scale)
    w2c = torch.linalg.inv(c2w)
    projection = torch.zeros(
        (*cameras.leading_shape, 4, 4),
        dtype=cameras.dtype,
        device=cameras.device,
    )
    projection[..., 3, 3] = 1.0
    projection[..., :3, :4] = torch.matmul(
        cameras.intrinsics, w2c[..., :3, :4]
    )
    return projection


def emvsnet_epistemic_risk(
    evidential: torch.Tensor, *, method: str
) -> torch.Tensor:
    """Compute the released EMVSNet epistemic uncertainty as an uncalibrated rank score."""

    if evidential.ndim != 4 or evidential.shape[1] != 4:
        raise ValueError("EMVSNet evidential output must have shape [B, 4, H, W]")
    _gamma, nu, alpha, beta = torch.unbind(evidential, dim=1)
    epsilon = torch.finfo(evidential.dtype).eps
    if method == "der":
        aleatoric = beta / (alpha - 1.0 + epsilon)
        return aleatoric / (nu + epsilon)
    if method == "sder":
        return torch.rsqrt(nu + epsilon)
    raise ValueError("method must be der or sder")


class UniDepthV2SmallSupplier:
    """Run UniDepthV2-Small independently for every context view."""

    def __init__(
        self,
        *,
        code_root: str | Path,
        model_id: str = "lpiccinelli/unidepth-v2-vits14",
        revision: str | None = None,
        local_model_dir: str | Path | None = None,
        model_cache_dir: str | Path | None = None,
        device: str = "cuda",
    ) -> None:
        self.code_root = Path(code_root).resolve()
        self.model_id = model_id
        self.revision = revision
        self.local_model_dir = None if local_model_dir is None else Path(local_model_dir).resolve()
        self.model_cache_dir = None if model_cache_dir is None else Path(model_cache_dir).resolve()
        self.device = torch.device(device)
        self._model: Any | None = None

    def _load(self) -> Any:
        if self._model is not None:
            return self._model
        if not self.code_root.is_dir():
            raise FileNotFoundError(self.code_root)
        with _prepend_path(self.code_root):
            from unidepth.models import UniDepthV2

            kwargs: dict[str, Any] = {}
            model_source = self.model_id
            if self.local_model_dir is not None:
                if not (self.local_model_dir / "config.json").is_file() or not (
                    self.local_model_dir / "model.safetensors"
                ).is_file():
                    raise FileNotFoundError(
                        "local UniDepth bundle must contain config.json and model.safetensors"
                    )
                model_source = str(self.local_model_dir)
                kwargs["local_files_only"] = True
            elif self.revision is not None:
                kwargs["revision"] = self.revision
            if self.model_cache_dir is not None:
                # Hugging Face reads this environment variable during model resolution.
                import os

                os.environ.setdefault("HF_HOME", str(self.model_cache_dir))
            self._model = UniDepthV2.from_pretrained(model_source, **kwargs)
        self._model = self._model.to(self.device).eval()
        for parameter in self._model.parameters():
            parameter.requires_grad_(False)
        return self._model

    @torch.no_grad()
    def infer(self, request: GeometrySupplierRequest) -> V8Proposal:
        model = self._load()
        canonical = request.canonical()
        depths: list[np.ndarray] = []
        points: list[np.ndarray] = []
        confidences: list[np.ndarray] = []
        valid: list[np.ndarray] = []
        with torch.inference_mode():
            for index in range(len(canonical.frame_ids)):
                # UniDepth's public infer API expects uint8 RGB and performs its own normalization.
                rgb = (canonical.rgb[index].clamp(0, 1) * 255.0).round().to(torch.uint8)
                K = canonical.cameras.intrinsics[index].to(dtype=torch.float32)
                output = model.infer(rgb, K)
                depth = _squeeze_map(output, "depth")
                pts = _squeeze_points(output, "points")
                conf = _squeeze_map(output, "confidence")
                if depth.shape != tuple(canonical.rgb.shape[-2:]):
                    raise ValueError("UniDepth output spatial size does not match request")
                depths.append(depth)
                points.append(pts)
                confidences.append(conf)
                valid.append(np.isfinite(depth) & (depth > 0))
        return V8Proposal(
            supplier="unidepth-v2-small",
            frame_ids=canonical.frame_ids,
            depth_z_m=np.stack(depths),
            points_cam_m=np.stack(points),
            native_confidence=np.stack(confidences),
            native_risk=1.0 - np.clip(np.stack(confidences), 0.0, 1.0),
            valid_mask=np.stack(valid),
            intrinsics=canonical.cameras.intrinsics.detach().cpu().numpy(),
            c2w=canonical.cameras.c2w.detach().cpu().numpy(),
            uncertainty_provenance="unidepth_confidence_rank_only",
            uncertainty_formula="model_output_confidence_not_calibrated",
        )


class EMVSNetSupplier:
    """Run the pinned EMVSNet 64-bin model with an explicit mm->m boundary."""

    def __init__(
        self,
        *,
        code_root: str | Path,
        checkpoint: str | Path,
        device: str = "cuda",
        image_scale: float = 0.25,
        depth_bins: int = 64,
        max_h: int = 512,
        max_w: int = 640,
        evidential_method: str = "der",
    ) -> None:
        if depth_bins != 64:
            raise ValueError("Q0 freezes the 64-bin EMVSNet checkpoint")
        if evidential_method not in {"der", "sder"}:
            raise ValueError("evidential_method must be der or sder")
        self.code_root = Path(code_root).resolve()
        self.checkpoint = Path(checkpoint).resolve()
        self.device = torch.device(device)
        self.image_scale = float(image_scale)
        self.depth_bins = int(depth_bins)
        self.max_h = int(max_h)
        self.max_w = int(max_w)
        self.evidential_method = evidential_method
        self._model: Any | None = None

    def depth_hypotheses_m(self, *, device: torch.device | None = None) -> torch.Tensor:
        """Return the pre-registered, label-independent Q0 metric depth range."""

        return torch.linspace(
            EMVSNET_DEPTH_NEAR_M,
            EMVSNET_DEPTH_FAR_M,
            self.depth_bins,
            dtype=torch.float32,
            device=device or self.device,
        )

    def _load(self) -> Any:
        if self._model is not None:
            return self._model
        if not self.checkpoint.is_file():
            raise FileNotFoundError(self.checkpoint)
        with _prepend_path(self.code_root):
            module = importlib.import_module("models.drmvsnet")
            model = module.EMVSNet(
                disparity_level=self.depth_bins,
                image_scale=self.image_scale,
                max_h=self.max_h,
                max_w=self.max_w,
                return_depth=False,
                evidential_method=self.evidential_method,
            )
        state = torch.load(self.checkpoint, map_location="cpu", weights_only=False)
        if not isinstance(state, dict) or "model" not in state:
            raise ValueError("EMVSNet checkpoint must contain a model state_dict")
        state_dict = {str(k).removeprefix("module."): v for k, v in state["model"].items()}
        model.load_state_dict(state_dict, strict=True)
        self._model = model.to(self.device).eval()
        for parameter in self._model.parameters():
            parameter.requires_grad_(False)
        return self._model

    @torch.no_grad()
    def infer(self, request: GeometrySupplierRequest) -> V8Proposal:
        if len(request.frame_ids) < 2:
            raise ValueError("EMVSNet requires one reference and at least one source view")
        model = self._load()
        canonical = request.canonical()
        height, width = canonical.cameras.image_size
        expected_h = int(round(self.max_h * self.image_scale))
        expected_w = int(round(self.max_w * self.image_scale))
        if (height, width) != (expected_h, expected_w):
            raise ValueError(
                f"EMVSNet Q0 expects {(expected_h, expected_w)} input pixels, got {(height, width)}"
            )
        scaled_h, scaled_w = height, width
        images = torch.nn.functional.interpolate(
            canonical.rgb.to(self.device) * 255.0,
            size=(scaled_h, scaled_w),
            mode="bilinear",
            align_corners=True,
        )
        images = center_emvsnet_images(images)
        # EMVSNet's published DTU interface uses projection matrices P=K[R|t].  We construct
        # those matrices from the OpenCV c2w cameras and feed a metric-equivalent mm scene.
        cameras = canonical.cameras.to(self.device)
        projection = emvsnet_projection_coordinates(
            cameras, translation_scale=EMVSNET_METRES_TO_CHECKPOINT_UNITS
        ).unsqueeze(0)
        depth_values_m = self.depth_hypotheses_m(device=self.device)
        depth_values_mm = depth_values_m * EMVSNET_METRES_TO_CHECKPOINT_UNITS
        with torch.inference_mode():
            probability_volume, evidential, _branch_probabilities = model(
                images.unsqueeze(0), projection, depth_values_mm.unsqueeze(0)
            )
        # EMVSNet's train/validation path regresses depth from the primary probability volume.
        # The third output is the evidential branch ensemble and is not the reported depth head.
        depth_mm = torch.sum(
            probability_volume * depth_values_mm.view(1, -1, 1, 1), dim=1
        )
        depth_mm = torch.nn.functional.interpolate(
            depth_mm[:, None], size=(height, width), mode="bilinear", align_corners=True
        )[:, 0]
        risk = torch.nn.functional.interpolate(
            emvsnet_epistemic_risk(evidential, method=self.evidential_method)[:, None],
            size=(height, width),
            mode="bilinear",
            align_corners=True,
        )[:, 0]
        depth_m = emvsnet_mm_to_m(depth_mm[0].detach().cpu().numpy())
        # EMVSNet predicts one reference map. Replicate only as an explicit reference proposal;
        # source views remain unknown rather than being fabricated.
        depth_maps = np.zeros((len(canonical.frame_ids), height, width), dtype=np.float32)
        points_maps = np.zeros((len(canonical.frame_ids), height, width, 3), dtype=np.float32)
        conf_maps = np.zeros_like(depth_maps)
        risk_maps = np.zeros_like(depth_maps)
        valid_maps = np.zeros_like(depth_maps, dtype=np.bool_)
        depth_maps[0] = depth_m
        points_maps[0] = depth_z_to_points_cam(
            depth_m, canonical.cameras.intrinsics[0].detach().cpu().numpy()
        )
        risk_maps[0] = risk[0].detach().cpu().numpy()
        conf_maps[0] = 1.0 / (1.0 + risk_maps[0])
        valid_maps[0] = np.isfinite(depth_m) & (depth_m > 0)
        return V8Proposal(
            supplier="emvsnet-64",
            frame_ids=canonical.frame_ids,
            depth_z_m=depth_maps,
            points_cam_m=points_maps,
            native_confidence=conf_maps,
            native_risk=risk_maps,
            valid_mask=valid_maps,
            intrinsics=canonical.cameras.intrinsics.detach().cpu().numpy(),
            c2w=canonical.cameras.c2w.detach().cpu().numpy(),
            uncertainty_provenance="emvsnet_evidential_rank_only",
            uncertainty_formula=f"{self.evidential_method}_publisher_release_inference_formula",
        )


def _squeeze_map(output: dict[str, torch.Tensor], key: str) -> np.ndarray:
    if key not in output:
        raise ValueError(f"UniDepth output missing {key}")
    value = output[key].detach().float().cpu()
    while value.ndim > 2 and value.shape[0] == 1:
        value = value[0]
    if value.ndim == 3 and value.shape[0] == 1:
        value = value[0]
    if value.ndim != 2 or not torch.isfinite(value).all():
        raise ValueError(f"UniDepth {key} has an invalid shape or nonfinite values")
    return value.numpy()


def _squeeze_points(output: dict[str, torch.Tensor], key: str) -> np.ndarray:
    if key not in output:
        raise ValueError(f"UniDepth output missing {key}")
    value = output[key].detach().float().cpu()
    if value.ndim == 4 and value.shape[0] == 1:
        value = value[0]
    if value.ndim != 3 or value.shape[0] != 3:
        raise ValueError("UniDepth points must have shape [1,3,H,W] or [3,H,W]")
    if not torch.isfinite(value).all():
        raise ValueError("UniDepth points must be finite")
    return value.permute(1, 2, 0).numpy()


@contextlib.contextmanager
def _prepend_path(root: Path):
    root_text = str(root)
    old = list(sys.path)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    try:
        yield
    finally:
        sys.path[:] = old
