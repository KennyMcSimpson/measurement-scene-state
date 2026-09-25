"""Independent, binding-checked checkpoints for the prefix action policy."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from hashlib import sha256
from os import replace
from pathlib import Path

import torch

from mcss.dynamic.policy import (
    ACTION_ORDER,
    FEATURE_SCHEMA_VERSION,
    POLICY_FEATURE_DIM,
    LearnedActionPolicy,
)

SCHEMA_VERSION = "mcss.dynamic.policy.v1"
_PAYLOAD_FIELDS = frozenset(
    {
        "schema_version",
        "policy_config",
        "normalization",
        "policy_state_dict",
        "binding",
        "provenance",
    }
)
_REQUIRED_BINDING_KEYS = frozenset(
    {
        "model_content_hash",
        "feature_schema_version",
        "render_protocol",
        "budget_protocol",
        "utility_protocol",
        "teacher_dataset_hash",
        "teacher_source_hashes",
    }
)


def save_policy_checkpoint(
    path: str | Path,
    policy: LearnedActionPolicy,
    *,
    binding: Mapping[str, object],
    provenance: Mapping[str, object] | None = None,
) -> str:
    """Save a policy-only checkpoint and return its final file SHA-256."""

    _validate_policy(policy)
    normalized_binding = _validate_binding(binding)
    if provenance is None:
        provenance = {}
    if not isinstance(provenance, Mapping):
        raise ValueError("provenance must be a mapping")
    if normalized_binding["feature_schema_version"] != FEATURE_SCHEMA_VERSION:
        raise ValueError("policy binding feature schema does not match runtime schema")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "policy_config": {
            "input_dim": policy.input_dim,
            "hidden_dim": policy.hidden_dim,
            "feature_schema_version": policy.feature_schema_version,
            "action_order": [str(action) for action in ACTION_ORDER],
            "seed": policy.seed,
            "target_mean": policy.target_mean,
            "target_scale": policy.target_scale,
        },
        "normalization": deepcopy(policy.normalization),
        "policy_state_dict": _cpu_state_dict(policy),
        "binding": normalized_binding,
        "provenance": deepcopy(dict(provenance)),
    }
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f"{target.name}.part")
    torch.save(payload, temporary)
    replace(temporary, target)
    return _sha256_file(target)


def load_policy_checkpoint(
    path: str | Path,
    *,
    expected_binding: Mapping[str, object] | None = None,
    device: str | torch.device = "cpu",
) -> tuple[LearnedActionPolicy, dict[str, object]]:
    """Load a policy and reject carrier/protocol binding mismatches."""

    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    payload = torch.load(checkpoint_path, map_location=device, weights_only=True)
    _validate_payload(payload)
    binding = _validate_binding(payload["binding"])
    if expected_binding is not None:
        expected = _validate_binding(expected_binding, allow_partial=True)
        mismatches = _binding_mismatches(expected, binding)
        if mismatches:
            raise ValueError(f"policy checkpoint binding mismatch: {sorted(mismatches)}")
    config = payload["policy_config"]
    normalization = payload["normalization"]
    policy = LearnedActionPolicy(
        hidden_dim=config["hidden_dim"],
        input_dim=config["input_dim"],
        normalization=normalization,
        seed=config.get("seed"),
        target_mean=config.get("target_mean", 0.0),
        target_scale=config.get("target_scale", 1.0),
    ).to(device)
    policy.load_state_dict(payload["policy_state_dict"], strict=True)
    policy.eval()
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "binding": deepcopy(binding),
        "provenance": deepcopy(payload["provenance"]),
        "policy_config": deepcopy(config),
        "normalization": deepcopy(normalization),
        "sha256": _sha256_file(checkpoint_path),
    }
    return policy, metadata


def _validate_policy(policy: object) -> None:
    if not isinstance(policy, LearnedActionPolicy):
        raise TypeError("policy checkpoints require LearnedActionPolicy")
    if policy.input_dim != POLICY_FEATURE_DIM:
        raise ValueError("policy input_dim does not match the runtime feature schema")
    if policy.feature_schema_version != FEATURE_SCHEMA_VERSION:
        raise ValueError("policy feature schema is unsupported")


def _validate_binding(
    binding: Mapping[str, object], *, allow_partial: bool = False
) -> dict[str, object]:
    if not isinstance(binding, Mapping):
        raise ValueError("policy binding must be a mapping")
    normalized = deepcopy(dict(binding))
    if not allow_partial:
        missing = _REQUIRED_BINDING_KEYS - normalized.keys()
        if missing:
            raise ValueError(f"policy binding is missing required keys: {sorted(missing)}")
    if "feature_schema_version" in normalized and not isinstance(
        normalized["feature_schema_version"], str
    ):
        raise ValueError("feature_schema_version must be a string")
    if "model_content_hash" in normalized and (
        not isinstance(normalized["model_content_hash"], str)
        or not normalized["model_content_hash"]
    ):
        raise ValueError("model_content_hash must be a nonempty string")
    if "teacher_source_hashes" in normalized and not isinstance(
        normalized["teacher_source_hashes"], (list, tuple, dict)
    ):
        raise ValueError("teacher_source_hashes must be serializable hashes")
    if "teacher_dataset_hash" in normalized and (
        not isinstance(normalized["teacher_dataset_hash"], str)
        or not normalized["teacher_dataset_hash"]
    ):
        raise ValueError("teacher_dataset_hash must be a nonempty string")
    if "teacher_source_hashes" in normalized:
        source_hashes = normalized["teacher_source_hashes"]
        if isinstance(source_hashes, Mapping) and any(
            not isinstance(key, str)
            or not isinstance(value, str)
            or not value
            for key, value in source_hashes.items()
        ):
            raise ValueError("teacher_source_hashes must map nonempty strings to hashes")
    return normalized


def _binding_mismatches(
    expected: Mapping[str, object], actual: Mapping[str, object], *, prefix: str = ""
) -> dict[str, tuple[object, object]]:
    """Compare an expected binding as a recursive subset of the saved binding.

    Runtime callers usually know only the fields they can validate before loading
    the policy, such as the carrier hash and budget ceiling.  Saved bindings may
    carry additional protocol detail (for example policy work units and teacher
    provenance), so nested mappings must preserve those extra fields while still
    rejecting missing or contradictory expected values.
    """

    mismatches: dict[str, tuple[object, object]] = {}
    for key, expected_value in expected.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if key not in actual:
            mismatches[path] = (expected_value, None)
            continue
        actual_value = actual[key]
        if isinstance(expected_value, Mapping):
            if not isinstance(actual_value, Mapping):
                mismatches[path] = (expected_value, actual_value)
                continue
            mismatches.update(
                _binding_mismatches(expected_value, actual_value, prefix=path)
            )
        elif actual_value != expected_value:
            mismatches[path] = (expected_value, actual_value)
    return mismatches


def _validate_payload(payload: object) -> None:
    if not isinstance(payload, dict):
        raise ValueError("policy checkpoint payload must be a dictionary")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"expected policy checkpoint schema {SCHEMA_VERSION!r}")
    if set(payload) != _PAYLOAD_FIELDS:
        missing = sorted(_PAYLOAD_FIELDS - set(payload))
        unexpected = sorted(set(payload) - _PAYLOAD_FIELDS)
        raise ValueError(
            f"policy checkpoint fields mismatch; missing={missing}, unexpected={unexpected}"
        )
    config = payload["policy_config"]
    if not isinstance(config, dict):
        raise ValueError("policy_config must be a dictionary")
    if config.get("feature_schema_version") != FEATURE_SCHEMA_VERSION:
        raise ValueError("policy checkpoint feature schema mismatch")
    if config.get("action_order") != [str(action) for action in ACTION_ORDER]:
        raise ValueError("policy checkpoint action order mismatch")
    if not isinstance(payload["normalization"], dict):
        raise ValueError("policy normalization must be a dictionary")
    for key in ("mean", "scale"):
        if key not in payload["normalization"]:
            raise ValueError(f"policy normalization missing {key}")
    if not isinstance(payload["policy_state_dict"], dict) or not all(
        isinstance(value, torch.Tensor) for value in payload["policy_state_dict"].values()
    ):
        raise ValueError("policy_state_dict must contain tensors")
    if not isinstance(payload["provenance"], dict):
        raise ValueError("policy provenance must be a dictionary")


def _cpu_state_dict(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in module.state_dict().items()
        if isinstance(value, torch.Tensor)
    }


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = ["SCHEMA_VERSION", "load_policy_checkpoint", "save_policy_checkpoint"]
