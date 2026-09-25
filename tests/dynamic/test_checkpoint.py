from hashlib import sha256

import pytest
import torch

from mcss.dynamic.carrier import DynamicSceneCarrier
from mcss.dynamic.checkpoint import load_dynamic_checkpoint, save_dynamic_checkpoint
from mcss.dynamic.config import CarrierConfig, WriteConfig
from mcss.dynamic.write_rule import DirectWriteRule


def _components() -> tuple[DynamicSceneCarrier, DirectWriteRule]:
    carrier_config = CarrierConfig(
        feature_dim=4,
        hidden_dim=4,
        expansion_dim=6,
        grid_size=(4, 4, 4),
        local_bounds_m=((-1.0, -1.0, 0.1), (1.0, 1.0, 3.0)),
        token_count=12,
    )
    write_config = WriteConfig(learning_rate=0.05, max_update_norm=0.2)
    return DynamicSceneCarrier(carrier_config), DirectWriteRule(carrier_config, write_config)


def test_dynamic_checkpoint_round_trip_is_cpu_strict_and_trainable(tmp_path) -> None:
    torch.manual_seed(21)
    carrier, write_rule = _components()
    path = tmp_path / "dynamic.pt"

    saved_hash = save_dynamic_checkpoint(
        path,
        carrier,
        write_rule,
        phase="grounded",
        provenance={"split": "train", "preprocessing": "rgb-only"},
    )
    with torch.no_grad():
        carrier.image_encoder[0].weight.add_(1.0)
        write_rule.targets["fuse"].weight.add_(1.0)
    restored_carrier, restored_rule, metadata = load_dynamic_checkpoint(path)
    payload = torch.load(path, map_location="cpu", weights_only=True)

    assert saved_hash == sha256(path.read_bytes()).hexdigest()
    assert set(payload) == {
        "schema_version",
        "carrier_config",
        "write_config",
        "carrier_state_dict",
        "write_rule_state_dict",
        "phase",
        "provenance",
    }
    assert all(value.device.type == "cpu" for value in payload["carrier_state_dict"].values())
    assert all(value.device.type == "cpu" for value in payload["write_rule_state_dict"].values())
    assert restored_carrier is not carrier
    assert restored_carrier.config == carrier.config
    assert restored_rule.carrier_config == write_rule.carrier_config
    assert restored_rule.write_config == write_rule.write_config
    assert metadata == {
        "schema_version": "mcss.dynamic.v1",
        "phase": "grounded",
        "provenance": {"split": "train", "preprocessing": "rgb-only"},
        "sha256": saved_hash,
    }
    assert all(parameter.requires_grad for parameter in restored_carrier.parameters())
    assert all(parameter.requires_grad for parameter in restored_rule.parameters())
    for name, value in payload["carrier_state_dict"].items():
        torch.testing.assert_close(value.cpu(), restored_carrier.state_dict()[name].cpu())
    for name, value in payload["write_rule_state_dict"].items():
        torch.testing.assert_close(value.cpu(), restored_rule.state_dict()[name].cpu())


def test_dynamic_checkpoint_normalizes_lists_and_rejects_v5_or_partial_state(tmp_path) -> None:
    carrier, write_rule = _components()
    path = tmp_path / "dynamic.pt"
    save_dynamic_checkpoint(path, carrier, write_rule, phase="write", provenance={"split": "train"})
    payload = torch.load(path, map_location="cpu", weights_only=True)
    payload["carrier_config"]["grid_size"] = list(payload["carrier_config"]["grid_size"])
    payload["carrier_config"]["local_bounds_m"] = [
        list(row) for row in payload["carrier_config"]["local_bounds_m"]
    ]
    torch.save(payload, path)
    loaded_carrier, _, _ = load_dynamic_checkpoint(path)
    assert loaded_carrier.config == carrier.config

    legacy_path = tmp_path / "v5.pt"
    torch.save({"schema_version": "mcss.v5", "model": {}, "optimizer": {}}, legacy_path)
    with pytest.raises(ValueError, match="V5"):
        load_dynamic_checkpoint(legacy_path)

    payload = torch.load(path, map_location="cpu", weights_only=True)
    payload["carrier_state_dict"].pop("fuse_down.weight")
    partial_path = tmp_path / "partial.pt"
    torch.save(payload, partial_path)
    with pytest.raises(RuntimeError, match="Missing key"):
        load_dynamic_checkpoint(partial_path)
