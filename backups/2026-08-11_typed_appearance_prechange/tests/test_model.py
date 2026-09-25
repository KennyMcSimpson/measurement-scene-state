import inspect

import pytest
import torch

from mcss.geometry import look_at, make_intrinsics
from mcss.model.system import ModelConfig, build_model
from mcss.types import Cameras


def _model_inputs() -> tuple[torch.Tensor, Cameras, Cameras, torch.Tensor]:
    torch.manual_seed(3)
    context_rgb = torch.rand(1, 2, 3, 8, 8)
    intrinsics = make_intrinsics((8, 8), 60.0)
    context_poses = torch.stack(
        (
            look_at(torch.tensor([0.0, 0.0, -3.0]), torch.zeros(3)),
            look_at(torch.tensor([2.0, 0.0, -2.0]), torch.zeros(3)),
        )
    )
    target_pose = look_at(torch.tensor([-2.0, 0.0, -2.0]), torch.zeros(3))
    context_cameras = Cameras(
        intrinsics.expand(1, 2, 3, 3).clone(), context_poses.unsqueeze(0), (8, 8)
    )
    target_cameras = Cameras(intrinsics.view(1, 1, 3, 3), target_pose.view(1, 1, 4, 4), (8, 8))
    bounds = torch.tensor([[[-1.2, -1.2, -1.2], [1.2, 1.2, 1.2]]])
    return context_rgb, context_cameras, target_cameras, bounds


def _config(mode: str) -> ModelConfig:
    return ModelConfig(
        mode=mode,
        voxel_resolution=8,
        image_feature_dim=8,
        state_feature_dim=8,
        refinement_blocks=1,
        n_samples=8,
        ray_chunk_size=128,
    )


@pytest.mark.parametrize("mode", ["fixed", "learned_heads", "free_decoder"])
def test_all_model_modes_forward_and_backward(mode: str) -> None:
    model = build_model(_config(mode))
    inputs = _model_inputs()

    output = model(*inputs)

    assert output.state.spatial_shape == (8, 8, 8)
    assert output.predictions["depth"].shape == (1, 1, 1, 8, 8)
    loss = sum(value.mean() for value in output.predictions.values())
    loss.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.requires_grad]
    assert any(gradient is not None for gradient in gradients)
    assert all(torch.isfinite(gradient).all() for gradient in gradients if gradient is not None)


def test_main_forward_has_no_target_label_argument() -> None:
    model = build_model(_config("fixed"))
    signature = inspect.signature(model.forward)

    assert "target_rgb" not in signature.parameters
    assert "target_depth" not in signature.parameters
    with pytest.raises(TypeError):
        model(*_model_inputs(), target_rgb=torch.rand(1, 1, 3, 8, 8))


def test_model_forward_is_finite_under_cuda_autocast() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    device = torch.device("cuda")
    context_rgb = torch.rand(1, 2, 3, 8, 8)
    intrinsics = make_intrinsics((8, 8), 60.0)
    context_poses = torch.stack(
        (
            look_at(torch.tensor([0.0, 0.0, 0.0]), torch.tensor([0.0, 0.0, 1.0])),
            look_at(torch.tensor([1.0, 0.0, -1.0]), torch.zeros(3)),
        )
    )
    target_pose = look_at(torch.tensor([0.0, 0.0, 0.0]), torch.tensor([0.0, 0.0, 1.0]))
    context_cameras = Cameras(
        intrinsics.expand(1, 2, 3, 3).clone(), context_poses.unsqueeze(0), (8, 8)
    )
    target_cameras = Cameras(intrinsics.view(1, 1, 3, 3), target_pose.view(1, 1, 4, 4), (8, 8))
    bounds = torch.tensor([[[-2.0, -2.0, -2.0], [2.0, 2.0, 2.0]]])
    model = build_model(_config("fixed")).to(device)
    context_rgb = context_rgb.to(device)
    context_cameras = context_cameras.to(device)
    target_cameras = target_cameras.to(device)
    bounds = bounds.to(device)

    with torch.autocast(device_type="cuda", enabled=True):
        output = model(context_rgb, context_cameras, target_cameras, bounds)

    assert all(torch.isfinite(value).all() for value in output.predictions.values())
    assert torch.isfinite(output.state.density_logits).all()


def test_model_builds_anisotropic_scene_state() -> None:
    config = ModelConfig(
        mode="fixed",
        voxel_resolution=(6, 8, 10),
        image_feature_dim=8,
        state_feature_dim=8,
        refinement_blocks=0,
        n_samples=8,
        ray_chunk_size=128,
    )
    model = build_model(config)

    output = model(*_model_inputs())

    assert output.state.spatial_shape == (6, 8, 10)
