from pathlib import Path

import torch

from mcss.config import load_config
from mcss.model.system import build_model


def test_frozen_v5_checkpoint_strictly_loads_into_current_model() -> None:
    project_root = Path(__file__).resolve().parents[1]
    baseline_root = project_root / "baselines" / "v5_typed_appearance_step4500"
    config = load_config(
        baseline_root / "config" / "hypersim_er_v5_appearance2x_overfit.yaml"
    )
    payload = torch.load(
        baseline_root / "checkpoint" / "step_004500.pt",
        map_location="cpu",
        weights_only=False,
    )
    model = build_model(config.model)

    assert len(payload["model"]) == len(model.state_dict()) == 92
    assert set(payload["model"]) == set(model.state_dict())
    result = model.load_state_dict(payload["model"], strict=True)

    assert result.missing_keys == []
    assert result.unexpected_keys == []
