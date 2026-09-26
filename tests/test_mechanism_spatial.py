import pytest
import torch

from mcss.dynamic.carrier import DynamicSceneCarrier
from mcss.dynamic.config import CarrierConfig
from mcss.geometry import make_intrinsics
from mcss.mechanism_pilot.spatial import (
    carrier_config_for_spatial_mode,
    observed_camera_support,
)
from mcss.types import Cameras


def test_backward_observations_recover_support_without_relaxing_two_views():
    k = make_intrinsics((32, 32), 100)
    front = Cameras(k, torch.eye(4), (32, 32))
    backward_pose = torch.diag(torch.tensor([-1.0, 1.0, -1.0, 1.0]))
    back = Cameras(k, backward_pose, (32, 32))
    old = DynamicSceneCarrier(carrier_config_for_spatial_mode("legacy-forward"))
    new = DynamicSceneCarrier(carrier_config_for_spatial_mode("anchor-centered"))
    assert old.config == CarrierConfig()
    assert torch.equal(old._candidate_ids, new._candidate_ids)
    assert torch.allclose(
        old._bounds[0, 1] - old._bounds[0, 0], new._bounds[0, 1] - new._bounds[0, 0]
    )
    assert (
        observed_camera_support(old._candidate_points, [front, back, back])["supported_candidates"]
        == 0
    )
    assert (
        observed_camera_support(new._candidate_points, [front, back, back])["supported_candidates"]
        > 0
    )
    assert observed_camera_support(new._candidate_points, [front])["supported_candidates"] == 0
    assert (
        observed_camera_support(new._candidate_points, [front, back])["supported_candidates"] == 0
    )
    with pytest.raises(ValueError):
        carrier_config_for_spatial_mode("auto-tune")
    with pytest.raises(ValueError):
        observed_camera_support(new._candidate_points, [])
