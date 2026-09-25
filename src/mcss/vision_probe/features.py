"""Frozen DINOv2 features and patch-level RGB observations for the 2D probe."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import TypedDict

import numpy as np
import torch
from PIL import Image
from torch import Tensor


class FeatureResult(TypedDict):
    """CPU tensors and image metadata produced for one image."""

    features: Tensor
    rgb: Tensor
    original_size: tuple[int, int]
    image_size: tuple[int, int]


class FrozenDinoExtractor:
    """Load a locally pinned DINOv2-S/14 checkpoint and extract patch features."""

    patch_size = 14
    feature_dim = 384

    def __init__(
        self,
        repo_root: Path,
        weights: Path,
        device: str | torch.device = "cuda",
        size: int = 224,
    ) -> None:
        self.repo_root = _existing_directory(repo_root, "DINOv2 repository")
        self.weights = _existing_file(weights, "DINOv2 checkpoint")
        self.device = _validate_device(device)
        if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
            raise ValueError("size must be a positive integer")
        if size % self.patch_size != 0:
            raise ValueError(f"size must be divisible by the {self.patch_size}-pixel patch size")
        self.size = size

        # pretrained=False is intentional: all weights must come from the explicit checkpoint.
        model = torch.hub.load(
            str(self.repo_root),
            "dinov2_vits14",
            source="local",
            pretrained=False,
        )
        state_dict = torch.load(self.weights, map_location="cpu", weights_only=True)
        if not isinstance(state_dict, Mapping):
            raise ValueError("DINOv2 checkpoint must be a raw state_dict mapping")
        model.load_state_dict(state_dict, strict=True)

        model_dim = getattr(model, "embed_dim", None)
        if model_dim is not None and int(model_dim) != self.feature_dim:
            raise ValueError(
                f"dinov2_vits14 must have feature dimension {self.feature_dim}, found {model_dim}"
            )
        model_patch_size = getattr(model, "patch_size", self.patch_size)
        if isinstance(model_patch_size, tuple):
            if model_patch_size != (self.patch_size, self.patch_size):
                raise ValueError("dinov2_vits14 must use 14x14 patches")
        elif model_patch_size != self.patch_size:
            raise ValueError("dinov2_vits14 must use 14x14 patches")

        self.model = model.to(self.device).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    def extract(self, path: Path) -> FeatureResult:
        """Extract frozen patch descriptors and the mean RGB of each image patch."""

        image_path = _existing_file(path, "image")
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            original_size = (int(image.width), int(image.height))
            resized = image.resize((self.size, self.size), resample=Image.Resampling.BICUBIC)
            rgb_image = torch.from_numpy(np.asarray(resized, dtype=np.float32)).div_(255.0)

        if rgb_image.shape != (self.size, self.size, 3):
            raise ValueError(f"resized RGB image has unexpected shape {tuple(rgb_image.shape)}")
        rgb = _patch_mean_rgb(rgb_image, self.patch_size)

        image_tensor = rgb_image.permute(2, 0, 1).unsqueeze(0)
        mean = torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32).view(1, 3, 1, 1)
        image_tensor = ((image_tensor - mean) / std).to(self.device)

        with torch.inference_mode():
            output = self.model.forward_features(image_tensor)
        if not isinstance(output, Mapping) or "x_norm_patchtokens" not in output:
            raise ValueError("DINOv2 forward_features must return x_norm_patchtokens")
        patch_tokens = output["x_norm_patchtokens"]
        if not isinstance(patch_tokens, Tensor):
            raise ValueError("DINOv2 x_norm_patchtokens must be a tensor")
        grid = self.size // self.patch_size
        expected_tokens = grid * grid
        if patch_tokens.ndim != 3 or patch_tokens.shape[0] != 1:
            raise ValueError(
                "DINOv2 x_norm_patchtokens must have shape [1, tokens, feature_dim]"
            )
        if patch_tokens.shape[1] != expected_tokens:
            raise ValueError(
                f"expected {expected_tokens} patch tokens for size {self.size}, "
                f"found {patch_tokens.shape[1]}"
            )
        if patch_tokens.shape[2] != self.feature_dim:
            raise ValueError(
                f"expected feature dimension {self.feature_dim}, found {patch_tokens.shape[2]}"
            )
        if not torch.isfinite(patch_tokens).all():
            raise ValueError("DINOv2 patch features must contain only finite values")

        features = (
            patch_tokens[0]
            .reshape(grid, grid, self.feature_dim)
            .detach()
            .to(device="cpu", dtype=torch.float32)
            .contiguous()
        )
        return {
            "features": features,
            "rgb": rgb.to(device="cpu", dtype=torch.float32).contiguous(),
            "original_size": original_size,
            "image_size": (self.size, self.size),
        }


def _existing_directory(value: Path, label: str) -> Path:
    try:
        path = Path(value).expanduser().resolve()
    except (TypeError, ValueError) as error:
        raise TypeError(f"{label} must be a filesystem path") from error
    if not path.is_dir():
        raise FileNotFoundError(f"{label} does not exist or is not a directory: {path}")
    return path


def _existing_file(value: Path, label: str) -> Path:
    try:
        path = Path(value).expanduser().resolve()
    except (TypeError, ValueError) as error:
        raise TypeError(f"{label} must be a filesystem path") from error
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist or is not a file: {path}")
    return path


def _validate_device(value: str | torch.device) -> torch.device:
    try:
        device = torch.device(value)
    except (TypeError, RuntimeError) as error:
        raise ValueError(f"invalid torch device: {value!r}") from error
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("device must be cpu or cuda")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA device requested but CUDA is unavailable")
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise ValueError(f"CUDA device index {device.index} is unavailable")
    return device


def _patch_mean_rgb(rgb_image: Tensor, patch_size: int) -> Tensor:
    """Average RGB over non-overlapping image patches, preserving H-W-C order."""

    height, width, channels = rgb_image.shape
    if channels != 3 or height % patch_size != 0 or width % patch_size != 0:
        raise ValueError("RGB image dimensions must be divisible by the patch size")
    patches = rgb_image.unfold(0, patch_size, patch_size).unfold(1, patch_size, patch_size)
    return patches.mean(dim=(-1, -2))
