import torch

from mcss.dynamic.policy import (
    ACTION_ORDER,
    FEATURE_SCHEMA_VERSION,
    POLICY_FEATURE_DIM,
    LearnedActionPolicy,
    control_to_features,
)
from mcss.dynamic.types import Action, ControlInput


def _control(**updates):
    values = {
        "rgb_mse": 0.1,
        "opacity_mean": 0.2,
        "current_image_mean": 0.3,
        "density_mean": 0.4,
        "fast_norm": 0.5,
        "observed_count": 2,
        "previous_action": Action.OFF,
        "remaining_budget": 100.0,
        "image_feature_stats": (0.1, 0.2, -0.3, 0.4),
        "rgb_residual_4x4": tuple(float(index) / 100 for index in range(48)),
        "coverage": 0.7,
        "valid_count": 12,
        "camera_change": (0.1, 0.2, 0.0, 0.0, 0.0, 0.01),
        "fast_state_stats": (0.1, 0.2, 0.3, 0.04),
        "remaining_steps": 3,
    }
    values.update(updates)
    return ControlInput(**values)


def test_feature_schema_is_fixed_finite_and_prefix_only():
    control = _control()
    features = control_to_features(control)

    assert FEATURE_SCHEMA_VERSION == "mcss.dynamic.policy_features.v1"
    assert features.shape == (POLICY_FEATURE_DIM,)
    assert features.dtype == torch.float32
    assert torch.isfinite(features).all()
    assert control_to_features(control).equal(features)


def test_legacy_control_constructor_keeps_deterministic_optional_defaults():
    control = ControlInput(0.1, 0.2, 0.3, 0.4, 0.5, 2, Action.OFF, 100.0)
    features = control_to_features(control)

    assert features.shape == (POLICY_FEATURE_DIM,)
    assert torch.isfinite(features).all()


def test_learned_policy_masks_infeasible_actions_and_prefers_off_on_ties():
    policy = LearnedActionPolicy(hidden_dim=8, seed=3)
    with torch.no_grad():
        for parameter in policy.parameters():
            parameter.zero_()

    control = _control()
    assert policy.choose(control, ACTION_ORDER) is Action.OFF
    assert policy.choose(control, (Action.FUSE, Action.COMPLETE, Action.ALL)) is Action.FUSE
    assert policy.choose(control, (Action.ALL,)) is Action.ALL


def test_policy_rejects_nonfinite_features_and_negative_budget():
    policy = LearnedActionPolicy(hidden_dim=8, seed=3)
    with torch.no_grad():
        bad = _control(rgb_residual_4x4=tuple([float("nan")] + [0.0] * 47))
    try:
        control_to_features(bad)
    except ValueError as error:
        assert "finite" in str(error)
    else:
        raise AssertionError("nonfinite feature summary was accepted")

    try:
        _control(remaining_budget=-1.0)
    except ValueError as error:
        assert "remaining_budget" in str(error)
    else:
        raise AssertionError("negative remaining budget was accepted")

    assert policy.declared_work_units > 1
