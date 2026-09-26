"""Strict, architecture-owned checkpoints for dynamic carrier training."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import asdict
from hashlib import sha256
from math import prod
from os import replace
from pathlib import Path

import torch

from mcss.dynamic.carrier import DynamicSceneCarrier
from mcss.dynamic.config import CarrierConfig, WriteConfig
from mcss.dynamic.write_rule import DirectWriteRule

SCHEMA_VERSION = "mcss.dynamic.v1"
_PAYLOAD_FIELDS = frozenset(
    {
        "schema_version",
        "carrier_config",
        "write_config",
        "carrier_state_dict",
        "write_rule_state_dict",
        "phase",
        "provenance",
    }
)


def save_dynamic_checkpoint(
    path: str | Path,
    carrier: DynamicSceneCarrier,
    write_rule: DirectWriteRule,
    *,
    phase: str,
    provenance: Mapping[str, object],
) -> str:
    """Persist only reusable slow state and return the final file SHA-256 hash."""

    _validate_components(carrier, write_rule)
    if not isinstance(phase, str) or not phase:
        raise ValueError("phase must be a nonempty string")
    if not isinstance(provenance, Mapping):
        raise ValueError("provenance must be a mapping")

    target = Path(path)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "carrier_config": asdict(carrier.config),
        "write_config": asdict(write_rule.write_config),
        "carrier_state_dict": _cpu_state_dict(carrier),
        "write_rule_state_dict": _cpu_state_dict(write_rule),
        "phase": phase,
        "provenance": deepcopy(dict(provenance)),
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f"{target.name}.part")
    torch.save(payload, temporary)
    replace(temporary, target)
    return _sha256_file(target)


def load_dynamic_checkpoint(
    path: str | Path,
    device: str | torch.device = "cpu",
) -> tuple[DynamicSceneCarrier, DirectWriteRule, dict[str, object]]:
    """Instantiate the exact dynamic architecture and restore both state dicts strictly."""

    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    payload = torch.load(checkpoint_path, map_location=device, weights_only=True)
    _validate_payload(payload)

    carrier_config = _carrier_config(payload["carrier_config"])
    write_config = _write_config(payload["write_config"])
    _validate_geometry(carrier_config, payload["carrier_state_dict"])
    carrier = DynamicSceneCarrier(carrier_config).to(device)
    write_rule = DirectWriteRule(carrier_config, write_config).to(device)
    carrier.load_state_dict(payload["carrier_state_dict"], strict=True)
    write_rule.load_state_dict(payload["write_rule_state_dict"], strict=True)
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "phase": payload["phase"],
        "provenance": deepcopy(payload["provenance"]),
        "sha256": _sha256_file(checkpoint_path),
    }
    return carrier, write_rule, metadata


def _validate_components(carrier: DynamicSceneCarrier, write_rule: DirectWriteRule) -> None:
    if not isinstance(carrier, DynamicSceneCarrier) or not isinstance(write_rule, DirectWriteRule):
        raise TypeError("dynamic checkpoints require DynamicSceneCarrier and DirectWriteRule")
    if carrier.config != write_rule.carrier_config:
        raise ValueError("carrier and write rule must use the same CarrierConfig")
    _validate_geometry(carrier.config, carrier.state_dict())


def _validate_geometry(config: CarrierConfig, state: Mapping[str, torch.Tensor]) -> None:
    """Reject stale geometry buffers without constructing a model or consuming RNG.

    Buffers are serialized alongside weights. Strict state-dict loading checks shapes,
    but otherwise permits old points/bounds to silently overwrite a new configuration.
    Rebuild the deterministic float32 geometry exactly as the carrier initializes it.
    """

    depth, height, width = config.grid_size
    voxel_count = prod(config.grid_size)
    if config.token_count == 1:
        ids = torch.tensor([(voxel_count - 1) // 2], dtype=torch.long)
    else:
        ids = torch.div(
            torch.arange(config.token_count, dtype=torch.long) * (voxel_count - 1),
            config.token_count - 1,
            rounding_mode="floor",
        )
    d = torch.div(ids, height * width, rounding_mode="floor")
    h = torch.div(ids, width, rounding_mode="floor").remainder(height)
    w = ids.remainder(width)
    fractions = torch.stack(
        (
            (w.to(torch.float32) + 0.5) / width,
            (h.to(torch.float32) + 0.5) / height,
            (d.to(torch.float32) + 0.5) / depth,
        ),
        dim=-1,
    )
    bounds = torch.tensor(config.local_bounds_m, dtype=torch.float32).unsqueeze(0)
    expected = {
        "_bounds": bounds,
        "_candidate_ids": ids,
        "_candidate_points": bounds[:, 0] + fractions * (bounds[:, 1] - bounds[:, 0]),
        "_candidate_normalized_xyz": fractions * 2.0 - 1.0,
    }
    for name, reference in expected.items():
        actual = state.get(name)
        if not isinstance(actual, torch.Tensor):
            raise ValueError(f"checkpoint geometry missing tensor {name}")
        if name == "_candidate_ids" and actual.dtype != torch.long:
            raise ValueError(f"checkpoint geometry dtype mismatch for {name}")
        if name != "_candidate_ids" and not actual.is_floating_point():
            raise ValueError(f"checkpoint geometry dtype mismatch for {name}")
        if actual.shape != reference.shape or not torch.equal(
            actual.detach().cpu(), reference.to(dtype=actual.dtype)
        ):
            raise ValueError(f"checkpoint geometry/config mismatch for {name}")


def _cpu_state_dict(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    state: dict[str, torch.Tensor] = {}
    for name, value in module.state_dict().items():
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"state_dict entry {name!r} must be a tensor")
        state[name] = value.detach().cpu().clone()
    return state


def _validate_payload(payload: object) -> None:
    if not isinstance(payload, dict):
        raise ValueError("dynamic checkpoint payload must be a dictionary")
    schema_version = payload.get("schema_version")
    if schema_version != SCHEMA_VERSION:
        if (
            schema_version is None
            or (isinstance(schema_version, str) and schema_version.startswith("mcss.v5"))
            or "model" in payload
            or "optimizer" in payload
        ):
            raise ValueError("legacy V5 checkpoints are not compatible with mcss.dynamic.v1")
        raise ValueError(f"expected dynamic checkpoint schema {SCHEMA_VERSION!r}")
    if set(payload) != _PAYLOAD_FIELDS:
        missing = sorted(_PAYLOAD_FIELDS - set(payload))
        unexpected = sorted(set(payload) - _PAYLOAD_FIELDS)
        raise ValueError(
            f"dynamic checkpoint fields mismatch; missing={missing}, unexpected={unexpected}"
        )
    if not isinstance(payload["phase"], str) or not payload["phase"]:
        raise ValueError("dynamic checkpoint phase must be nonempty")
    if not isinstance(payload["provenance"], dict):
        raise ValueError("dynamic checkpoint provenance must be a dictionary")
    for name in ("carrier_state_dict", "write_rule_state_dict"):
        if not isinstance(payload[name], dict) or not all(
            isinstance(value, torch.Tensor) for value in payload[name].values()
        ):
            raise ValueError(f"dynamic checkpoint {name} must be a tensor state dictionary")


def _carrier_config(value: object) -> CarrierConfig:
    if not isinstance(value, dict):
        raise ValueError("carrier_config must be a dictionary")
    normalized = _normalize_sequences(value)
    try:
        return CarrierConfig(**normalized)
    except (TypeError, ValueError) as error:
        raise ValueError("invalid dynamic carrier_config") from error


def _write_config(value: object) -> WriteConfig:
    if not isinstance(value, dict):
        raise ValueError("write_config must be a dictionary")
    normalized = _normalize_sequences(value)
    try:
        return WriteConfig(**normalized)
    except (TypeError, ValueError) as error:
        raise ValueError("invalid dynamic write_config") from error


def _normalize_sequences(value: object):
    if isinstance(value, dict):
        return {key: _normalize_sequences(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return tuple(_normalize_sequences(item) for item in value)
    return value


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
