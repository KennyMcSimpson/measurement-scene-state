from pathlib import Path

import pytest
import torch
from PIL import Image
from torch import nn

from mcss.vision_probe.features import FrozenDinoExtractor
from mcss.vision_probe.geometry import (
    homography_correspondence,
    mask_propagation_metrics,
)


class _StubBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed_dim = 384
        self.patch_size = 14
        self.weight = nn.Parameter(torch.ones(1))
        self.forward_inputs: list[torch.Tensor] = []
        self.loaded_state: object = None
        self.loaded_strict: bool | None = None

    def load_state_dict(self, state: object, strict: bool = True) -> None:
        self.loaded_state = state
        self.loaded_strict = strict

    def forward(self, _input: torch.Tensor) -> torch.Tensor:
        raise AssertionError("FrozenDinoExtractor must call forward_features directly")

    def forward_features(self, input_tensor: torch.Tensor) -> dict[str, torch.Tensor]:
        self.forward_inputs.append(input_tensor.detach().cpu())
        return {"x_norm_patchtokens": torch.zeros(1, 256, 384, device=input_tensor.device)}


def test_extractor_uses_full_resized_input_and_returns_cpu_patch_values(
    tmp_path: Path, monkeypatch
) -> None:
    repo_root = tmp_path / "dinov2"
    repo_root.mkdir()
    weights = tmp_path / "weights.pth"
    weights.write_bytes(b"stub")
    image_path = tmp_path / "image.png"
    Image.new("RGB", (9, 7), (255, 0, 128)).save(image_path)

    backbone = _StubBackbone()
    hub_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def load(*args: object, **kwargs: object) -> _StubBackbone:
        hub_calls.append((args, kwargs))
        return backbone

    state_dict = {"weight": torch.ones(1)}
    monkeypatch.setattr(torch.hub, "load", load)
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: state_dict)

    extractor = FrozenDinoExtractor(repo_root, weights, device="cpu")
    result = extractor.extract(image_path)

    assert hub_calls == [
        (
            (str(repo_root.resolve()), "dinov2_vits14"),
            {"source": "local", "pretrained": False},
        )
    ]
    assert backbone.loaded_state is state_dict
    assert backbone.loaded_strict is True
    assert len(backbone.forward_inputs) == 1
    assert tuple(backbone.forward_inputs[0].shape) == (1, 3, 224, 224)
    assert result["features"].shape == (16, 16, 384)
    assert result["rgb"].shape == (16, 16, 3)
    assert result["features"].device.type == "cpu"
    assert result["rgb"].device.type == "cpu"
    assert result["original_size"] == (9, 7)
    assert result["image_size"] == (224, 224)
    assert torch.allclose(result["rgb"][0, 0], torch.tensor([1.0, 0.0, 128.0 / 255.0]))
    assert not backbone.training
    assert all(not parameter.requires_grad for parameter in backbone.parameters())


def test_identity_homography_maps_all_token_centers_to_integer_coordinates() -> None:
    valid_indices, target_xy = homography_correspondence(
        torch.eye(3),
        src_original_size=(224, 224),
        dst_original_size=(224, 224),
    )

    assert torch.equal(valid_indices, torch.arange(256, dtype=torch.long))
    assert target_xy.shape == (256, 2)
    assert torch.equal(target_xy[0], torch.tensor([0.0, 0.0]))
    assert torch.equal(target_xy[-1], torch.tensor([15.0, 15.0]))


def test_mask_propagation_identity_and_disjoint_target_labels() -> None:
    features = torch.eye(4).reshape(2, 2, 4)
    source_labels = torch.tensor([[0, 1], [0, 1]], dtype=torch.int64)

    identity = mask_propagation_metrics(features, features, source_labels, source_labels)
    assert identity == {
        "mean_iou": 1.0,
        "foreground_iou": 1.0,
        "n_objects": 1,
        "accuracy": 1.0,
    }

    wrong_target = torch.tensor([[0, 0], [1, 1]], dtype=torch.int64)
    wrong = mask_propagation_metrics(features, features, source_labels, wrong_target)
    assert wrong["n_objects"] == 1
    assert wrong["mean_iou"] == pytest.approx(1.0 / 3.0)
    assert wrong["foreground_iou"] == pytest.approx(1.0 / 3.0)
    assert wrong["accuracy"] == 0.5


def test_mask_propagation_excludes_source_and_target_ignore_255() -> None:
    source_features = torch.eye(4).reshape(2, 2, 4)
    target_features = source_features.clone()
    target_features[0, 1] = source_features[0, 0]
    source_labels = torch.tensor([[255, 1], [0, 1]], dtype=torch.int64)
    target_labels = torch.tensor([[255, 1], [0, 1]], dtype=torch.int64)

    metrics = mask_propagation_metrics(
        source_features,
        target_features,
        source_labels,
        target_labels,
    )

    assert metrics == {
        "mean_iou": 1.0,
        "foreground_iou": 1.0,
        "n_objects": 1,
        "accuracy": 1.0,
    }
