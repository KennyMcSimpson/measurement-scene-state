import torch

from mcss.geometry import look_at, make_intrinsics, make_voxel_centers
from mcss.losses import MeasurementLoss
from mcss.measurements import FixedMeasurementRenderer, _sample_volume, _state_normal_grid
from mcss.types import Cameras, SceneState


def _sphere_state(resolution: int = 12) -> SceneState:
    bounds = torch.tensor([[[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]]])
    points = make_voxel_centers(bounds, resolution)
    radius = torch.linalg.vector_norm(points, dim=-1)
    density = (12.0 * (0.65 - radius)).unsqueeze(1).requires_grad_()
    color = torch.zeros(1, 3, resolution, resolution, resolution)
    color[:, 0] = 0.8
    color[:, 1] = 0.2
    color[:, 2] = 0.1
    log_variance = torch.full_like(density, -3.0)
    return SceneState(density, color, log_variance, bounds)


def _cameras() -> Cameras:
    intrinsics = make_intrinsics((8, 8), 55.0).view(1, 1, 3, 3)
    c2w = look_at(torch.tensor([0.0, 0.0, -3.0]), torch.zeros(3)).view(1, 1, 4, 4)
    return Cameras(intrinsics, c2w, (8, 8))


def test_fixed_renderer_is_parameter_free_and_differentiable() -> None:
    renderer = FixedMeasurementRenderer(n_samples=24, ray_chunk_size=31)
    state = _sphere_state()

    outputs = renderer(state, _cameras())

    assert sum(parameter.numel() for parameter in renderer.parameters()) == 0
    assert set(outputs) == {"depth", "normal", "point", "rgb", "uncertainty", "visibility"}
    assert outputs["rgb"].shape == (1, 1, 3, 8, 8)
    assert outputs["depth"].shape == (1, 1, 1, 8, 8)
    assert all(torch.isfinite(value).all() for value in outputs.values())

    outputs["depth"].mean().backward()
    assert state.density_logits.grad is not None
    assert torch.isfinite(state.density_logits.grad).all()


def test_measurement_selection_and_query_permutation_are_exact() -> None:
    renderer = FixedMeasurementRenderer(n_samples=16, ray_chunk_size=64)
    state = _sphere_state(10)
    camera = _cameras()
    second_pose = look_at(torch.tensor([2.5, 0.0, -1.5]), torch.zeros(3))
    cameras = Cameras(
        camera.intrinsics.expand(1, 2, 3, 3).clone(),
        torch.stack((camera.c2w[0, 0], second_pose), dim=0).unsqueeze(0),
        camera.image_size,
    )

    reference = renderer(state, cameras, {"depth", "visibility"})
    permuted = renderer(state, cameras.select_views(torch.tensor([1, 0])), {"depth", "visibility"})

    assert set(reference) == {"depth", "visibility"}
    torch.testing.assert_close(reference["depth"][:, [1, 0]], permuted["depth"])
    torch.testing.assert_close(reference["visibility"][:, [1, 0]], permuted["visibility"])


def test_cell_centered_volume_samples_reproduce_voxel_values() -> None:
    bounds = torch.tensor([[[-3.0, -2.0, 1.0], [5.0, 4.0, 9.0]]])
    resolution = (4, 3, 5)
    values = torch.arange(4 * 3 * 5, dtype=torch.float32).reshape(1, 1, *resolution)
    centers = make_voxel_centers(bounds, resolution).reshape(1, -1, 1, 3)

    sampled = _sample_volume(values, centers, bounds)

    torch.testing.assert_close(sampled.reshape(-1), values.reshape(-1), atol=1e-5, rtol=1e-5)


def test_density_gradient_normal_uses_world_space_cell_spacing() -> None:
    bounds = torch.tensor([[[-4.0, -3.0, 1.0], [8.0, 4.0, 11.0]]])
    resolution = (5, 7, 9)
    centers = make_voxel_centers(bounds, resolution)
    density = 20.0 + 0.1 * (centers[..., 0] + 2.0 * centers[..., 1] + 3.0 * centers[..., 2])
    density_logits = torch.log(torch.expm1(density)).unsqueeze(1)
    state = SceneState(
        density_logits,
        torch.zeros(1, 3, *resolution),
        torch.zeros_like(density_logits),
        bounds,
    )

    normal = _state_normal_grid(state)[0, :, 2, 3, 4]

    expected = -torch.nn.functional.normalize(torch.tensor([1.0, 2.0, 3.0]), dim=0)
    torch.testing.assert_close(normal, expected, atol=1e-5, rtol=1e-5)


def test_nearly_flat_density_normal_backward_is_finite_under_bfloat16_cuda_amp() -> None:
    if not torch.cuda.is_available():
        return
    device = torch.device("cuda")
    resolution = 8
    bounds = torch.tensor([[[-6.0, -4.0, 0.05], [6.0, 4.0, 12.05]]], device=device)
    ramp = torch.linspace(-1e-3, 1e-3, resolution, device=device)
    density_parameter = torch.nn.Parameter(
        torch.full(
            (1, 1, resolution, resolution, resolution),
            -2.0,
            device=device,
        )
        + ramp.view(1, 1, 1, 1, resolution)
    )
    density = density_parameter.to(torch.bfloat16)
    state = SceneState(
        density,
        torch.zeros(1, 3, resolution, resolution, resolution, device=device, dtype=torch.bfloat16),
        torch.full_like(density, -3.0),
        bounds,
    )
    camera = Cameras(
        make_intrinsics((8, 8), 55.0, device=device).view(1, 1, 3, 3),
        torch.eye(4, device=device).view(1, 1, 4, 4),
        (8, 8),
    )
    renderer = FixedMeasurementRenderer(n_samples=8, ray_chunk_size=64).to(device)
    target = torch.tensor([0.0, 0.0, 1.0], device=device).view(1, 1, 3, 1, 1)
    target = target.expand(1, 1, 3, 8, 8)
    objective = MeasurementLoss({"normal": 1.0})
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
        normal = renderer(state, camera, {"normal"})["normal"]
        loss, _ = objective({"normal": normal}, {"normal": target})
    loss.backward()

    assert torch.isfinite(normal).all()
    assert density_parameter.grad is not None
    assert torch.isfinite(density_parameter.grad).all()
