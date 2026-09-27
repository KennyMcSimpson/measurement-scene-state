"""Geometry buffers must agree with the declared checkpoint configuration."""

from dataclasses import replace

import pytest
import torch

from mcss.dynamic.carrier import DynamicSceneCarrier
from mcss.dynamic.checkpoint import load_dynamic_checkpoint, save_dynamic_checkpoint
from mcss.dynamic.config import CarrierConfig, WriteConfig
from mcss.dynamic.write_rule import DirectWriteRule

CENTERED = ((-6.0, -4.0, -6.0), (6.0, 4.0, 6.0))


def _save(path, config=None):
    config = config or CarrierConfig()
    carrier = DynamicSceneCarrier(config)
    writer = DirectWriteRule(config, WriteConfig())
    save_dynamic_checkpoint(path, carrier, writer, phase="geometry-test", provenance={})
    return carrier, writer


@pytest.mark.parametrize("centered", [False, True])
@pytest.mark.parametrize("tokens", [1, 128])
def test_geometry_roundtrip_preserves_legacy_and_centered_bounds(tmp_path, centered, tokens):
    config = CarrierConfig(token_count=tokens)
    if centered:
        config = replace(config, local_bounds_m=CENTERED)
    path = tmp_path / "weights.pt"
    original, _ = _save(path, config)
    restored, _, metadata = load_dynamic_checkpoint(path)
    assert metadata["schema_version"] == "mcss.dynamic.v1"
    assert restored.config == config
    for name, value in original.state_dict().items():
        assert torch.equal(value, restored.state_dict()[name])


def test_geometry_rejects_changed_config_without_changed_buffers(tmp_path):
    path = tmp_path / "weights.pt"
    _save(path)
    payload = torch.load(path, weights_only=True)
    payload["carrier_config"]["local_bounds_m"] = CENTERED
    torch.save(payload, path)
    with pytest.raises(ValueError, match="geometry/config mismatch"):
        load_dynamic_checkpoint(path)


@pytest.mark.parametrize(
    "name", ["_bounds", "_candidate_points", "_candidate_ids", "_candidate_normalized_xyz"]
)
def test_geometry_rejects_tampered_buffer_on_load_and_save(tmp_path, name):
    path = tmp_path / "weights.pt"
    carrier, writer = _save(path)
    payload = torch.load(path, weights_only=True)
    payload["carrier_state_dict"][name].reshape(-1)[0] += 1
    torch.save(payload, path)
    with pytest.raises(ValueError, match="geometry/config mismatch"):
        load_dynamic_checkpoint(path)
    with torch.no_grad():
        getattr(carrier, name).reshape(-1)[0] += 1
    output = tmp_path / "invalid.pt"
    with pytest.raises(ValueError, match="geometry/config mismatch"):
        save_dynamic_checkpoint(output, carrier, writer, phase="test", provenance={})
    assert not output.exists()


def test_geometry_validation_does_not_consume_rng_during_save(tmp_path):
    carrier = DynamicSceneCarrier(CarrierConfig())
    writer = DirectWriteRule(carrier.config, WriteConfig())
    before = torch.get_rng_state().clone()
    save_dynamic_checkpoint(
        tmp_path / "weights.pt", carrier, writer, phase="rng-test", provenance={}
    )
    assert torch.equal(before, torch.get_rng_state())
