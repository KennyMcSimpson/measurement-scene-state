import hashlib

import pytest
import torch

from mcss.dynamic.policy import FEATURE_SCHEMA_VERSION, LearnedActionPolicy
from mcss.dynamic.policy_checkpoint import load_policy_checkpoint, save_policy_checkpoint


def _binding():
    return {
        "model_content_hash": "carrier-write-content",
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "render_protocol": {"renderer": "fixed", "n_samples": 8},
        "budget_protocol": {"max_units": 1000.0},
        "utility_protocol": {"horizon": 2, "metric": "depth_abs_rel"},
        "teacher_dataset_hash": "teacher-dataset",
        "teacher_source_hashes": {"rows.jsonl": "source-hash"},
    }


def test_policy_checkpoint_round_trip_and_expected_binding(tmp_path):
    torch.manual_seed(2)
    policy = LearnedActionPolicy(hidden_dim=7, seed=4, target_mean=0.2, target_scale=0.5)
    path = tmp_path / "policy.pt"
    saved_hash = save_policy_checkpoint(
        path,
        policy,
        binding=_binding(),
        provenance={"seed": 4},
    )

    restored, metadata = load_policy_checkpoint(
        path,
        expected_binding={
            "model_content_hash": "carrier-write-content",
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "budget_protocol": {"max_units": 1000.0},
        },
    )
    assert restored.hidden_dim == 7
    assert restored.target_mean == 0.2
    assert restored.target_scale == 0.5
    assert metadata["binding"] == _binding()
    assert metadata["provenance"] == {"seed": 4}
    assert saved_hash == hashlib.sha256(path.read_bytes()).hexdigest()
    for name, value in policy.state_dict().items():
        torch.testing.assert_close(value.cpu(), restored.state_dict()[name].cpu())


def test_policy_checkpoint_rejects_binding_mismatch(tmp_path):
    path = tmp_path / "policy.pt"
    save_policy_checkpoint(path, LearnedActionPolicy(hidden_dim=4), binding=_binding())

    with pytest.raises(ValueError, match="binding mismatch"):
        load_policy_checkpoint(
            path,
            expected_binding={"model_content_hash": "different"},
        )


def test_policy_checkpoint_rejects_missing_teacher_hash(tmp_path):
    path = tmp_path / "policy.pt"
    binding = _binding()
    binding["teacher_dataset_hash"] = ""

    with pytest.raises(ValueError, match="teacher_dataset_hash"):
        save_policy_checkpoint(path, LearnedActionPolicy(hidden_dim=4), binding=binding)


def test_policy_checkpoint_does_not_accept_dynamic_v1_payload(tmp_path):
    path = tmp_path / "policy.pt"
    torch.save({"schema_version": "mcss.dynamic.v1"}, path)

    with pytest.raises(ValueError, match="policy checkpoint schema"):
        load_policy_checkpoint(path)
