"""Read-only measurement feedback from the state before the current observation."""

import torch
import torch.nn.functional as functional
from torch import Tensor

from mcss.dynamic.types import OnlineObservation
from mcss.types import Cameras


def anchored_observation(observation: OnlineObservation, anchor_c2w: torch.Tensor):
    camera = observation.camera
    local = torch.linalg.solve(anchor_c2w, camera.c2w)
    return OnlineObservation(
        observation.scene_id,
        observation.frame_id,
        observation.rgb,
        Cameras(camera.intrinsics, local, camera.image_size),
    )


def batched_camera(camera: Cameras):
    if camera.leading_shape:
        raise ValueError("Expected one unbatched camera")
    return Cameras(camera.intrinsics[None, None], camera.c2w[None, None], camera.image_size)


def _camera_change(previous_camera, current_camera) -> tuple[float, ...]:
    if previous_camera is None:
        return (0.0,) * 6
    relative = torch.linalg.solve(previous_camera.c2w, current_camera.c2w)
    rotation = relative[:3, :3]
    rotation_vector = 0.5 * torch.stack(
        (
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        )
    )
    values = torch.cat((relative[:3, 3], rotation_vector))
    if not torch.isfinite(values).all():
        raise ValueError("camera change is nonfinite")
    return tuple(float(value) for value in values)


def _image_feature_stats(image_feature: Tensor | None) -> tuple[float, ...]:
    if image_feature is None:
        return (0.0,) * 4
    values = image_feature.detach().reshape(-1)
    if values.numel() == 0 or not torch.isfinite(values).all():
        raise ValueError("current image feature must contain finite values")
    return tuple(
        float(value)
        for value in (values.mean(), values.std(unbiased=False), values.min(), values.max())
    )


def preupdate_feedback(
    scene_state,
    observation,
    renderer,
    *,
    image_feature: Tensor | None = None,
    previous_camera=None,
):
    """Return old-state residuals and prefix summaries before absorbing RGB."""

    prediction = renderer(
        scene_state, batched_camera(observation.camera), measurements=("rgb", "visibility")
    )
    predicted_rgb = prediction["rgb"][0, 0]
    residual = observation.rgb - predicted_rgb
    mse = residual.square().mean()
    visibility = prediction["visibility"][0, 0, 0]
    valid = torch.isfinite(residual).all(dim=0)
    coverage = torch.isfinite(visibility) & (visibility > 0)
    residual_grid = functional.adaptive_avg_pool2d(
        residual.unsqueeze(0), (4, 4)
    )[0].reshape(-1)
    if not torch.isfinite(residual_grid).all():
        raise ValueError("RGB residual summary is nonfinite")
    return {
        "rgb_mse": float(mse),
        "opacity_mean": float(visibility.mean()),
        "image_feature_stats": _image_feature_stats(image_feature),
        "rgb_residual_4x4": tuple(float(value) for value in residual_grid),
        "coverage": float(coverage.float().mean()),
        "valid_count": int(valid.sum().item()),
        "camera_change": _camera_change(previous_camera, observation.camera),
    }
