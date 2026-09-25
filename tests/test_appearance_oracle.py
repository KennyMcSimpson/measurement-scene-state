import pytest
import torch

from mcss.appearance_oracle import AppearanceOracle
from mcss.geometry import look_at, make_intrinsics, make_voxel_centers
from mcss.measurements import FixedMeasurementRenderer
from mcss.types import Cameras, SceneState, StateAppearance


def _reference_state(resolution: int = 6) -> SceneState:
    bounds = torch.tensor([[[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]]])
    points = make_voxel_centers(bounds, resolution)
    radius = torch.linalg.vector_norm(points, dim=-1)
    density = (14.0 * (0.65 - radius)).unsqueeze(1)
    color = torch.full((1, 3, resolution, resolution, resolution), 0.35)
    variance = torch.full_like(density, -3.0)
    return SceneState(density, color, variance, bounds)


def _reference_state_with_typed_appearance(
    native_resolution: int = 4,
    appearance_resolution: int = 8,
) -> SceneState:
    state = _reference_state(native_resolution)
    scalar = torch.zeros(1, 1, appearance_resolution, appearance_resolution, appearance_resolution)
    color = torch.full(
        (1, 3, appearance_resolution, appearance_resolution, appearance_resolution),
        0.35,
    )
    appearance = StateAppearance(
        confidence=scalar,
        unknown_probability=torch.ones_like(scalar),
        completion_gate=torch.ones_like(scalar),
        provenance=torch.zeros(
            1,
            2,
            appearance_resolution,
            appearance_resolution,
            appearance_resolution,
        ),
        base_color=color,
        color_logit_residual=torch.zeros_like(color),
    )
    return SceneState(
        state.density_logits,
        state.color,
        state.log_variance,
        state.bounds,
        appearance=appearance,
    )


def _two_identical_cameras() -> Cameras:
    intrinsics = make_intrinsics((8, 8), 55.0).expand(1, 2, 3, 3).clone()
    pose = look_at(torch.tensor([0.0, 0.0, -3.0]), torch.zeros(3))
    poses = pose.view(1, 1, 4, 4).expand(1, 2, 4, 4).clone()
    return Cameras(intrinsics, poses, (8, 8))


def test_oracle_variants_have_explicit_parameter_shapes_and_detach_reference() -> None:
    state = _reference_state()
    original_color = state.color.clone()

    native = AppearanceOracle(state, target_views=2, variant="native_shared")
    highres = AppearanceOracle(state, target_views=2, variant="highres_shared")
    per_view = AppearanceOracle(state, target_views=2, variant="native_per_view")

    assert native.color_logits.shape == (1, 3, 6, 6, 6)
    assert highres.color_logits.shape == (1, 3, 12, 12, 12)
    assert per_view.color_logits.shape == (1, 2, 3, 6, 6, 6)
    assert native.spatial_shape == (6, 6, 6)
    assert highres.spatial_shape == (12, 12, 12)
    assert per_view.spatial_shape == (6, 6, 6)
    assert native.parameter_count == native.color_logits.numel()
    assert highres.parameter_count == highres.color_logits.numel()
    assert per_view.parameter_count == per_view.color_logits.numel()

    with torch.no_grad():
        native.color_logits.zero_()
    torch.testing.assert_close(state.color, original_color)
    assert not native.density_logits.requires_grad
    assert not highres.density_logits.requires_grad

    with pytest.raises(ValueError, match="unsupported appearance oracle variant"):
        AppearanceOracle(state, target_views=2, variant="unknown")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="target_views"):
        AppearanceOracle(state, target_views=0, variant="native_shared")


def test_per_view_oracle_routes_each_color_volume_to_its_camera() -> None:
    state = _reference_state()
    oracle = AppearanceOracle(state, target_views=2, variant="native_per_view")
    renderer = FixedMeasurementRenderer(n_samples=24, ray_chunk_size=128)
    cameras = _two_identical_cameras()

    red = torch.tensor([0.9, 0.1, 0.1]).view(1, 3, 1, 1, 1)
    green = torch.tensor([0.1, 0.9, 0.1]).view(1, 3, 1, 1, 1)
    with torch.no_grad():
        oracle.color_logits[:, 0].copy_(torch.logit(red).expand_as(oracle.color_logits[:, 0]))
        oracle.color_logits[:, 1].copy_(torch.logit(green).expand_as(oracle.color_logits[:, 1]))

    rgb = oracle.render_rgb(renderer, cameras)

    assert rgb.shape == (1, 2, 3, 8, 8)
    assert rgb[:, 0, 0].mean() > rgb[:, 0, 1].mean()
    assert rgb[:, 1, 1].mean() > rgb[:, 1, 0].mean()


@pytest.mark.parametrize("variant", ["native_shared", "highres_shared", "native_per_view"])
def test_oracle_variants_backpropagate_finite_color_gradients(variant: str) -> None:
    state = _reference_state()
    oracle = AppearanceOracle(state, target_views=2, variant=variant)  # type: ignore[arg-type]
    renderer = FixedMeasurementRenderer(n_samples=16, ray_chunk_size=128)

    rgb = oracle.render_rgb(renderer, _two_identical_cameras())
    loss = rgb.square().mean()
    loss.backward()

    assert oracle.color_logits.grad is not None
    assert torch.isfinite(oracle.color_logits.grad).all()
    assert oracle.color_logits.grad.abs().sum() > 0


def test_typed_appearance_oracle_keeps_native_geometry_and_optimizes_typed_color() -> None:
    state = _reference_state_with_typed_appearance()
    oracle = AppearanceOracle(
        state,
        target_views=2,
        variant="typed_shared",
        field="typed_appearance",
    )
    renderer = FixedMeasurementRenderer(n_samples=8, ray_chunk_size=128)
    cameras = _two_identical_cameras()

    assert oracle.spatial_shape == (8, 8, 8)
    torch.testing.assert_close(oracle.density_logits, state.density_logits)
    before_depth = renderer(oracle._state(torch.sigmoid(oracle.color_logits)), cameras, {"depth"})[
        "depth"
    ]
    rgb = oracle.render_rgb(renderer, cameras)
    rgb.mean().backward()

    assert oracle.color_logits.grad is not None
    assert torch.isfinite(oracle.color_logits.grad).all()
    with torch.no_grad():
        oracle.color_logits.add_(0.5)
    after_depth = renderer(oracle._state(torch.sigmoid(oracle.color_logits)), cameras, {"depth"})[
        "depth"
    ]
    torch.testing.assert_close(after_depth, before_depth)


def test_typed_appearance_oracle_requires_existing_typed_field() -> None:
    with pytest.raises(ValueError, match="StateAppearance"):
        AppearanceOracle(
            _reference_state(),
            target_views=2,
            variant="typed_shared",
            field="typed_appearance",
        )
