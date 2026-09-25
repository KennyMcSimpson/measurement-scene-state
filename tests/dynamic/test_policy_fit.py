import copy
import json

import pytest
import torch

from mcss.dynamic.policy import (
    FEATURE_NAMES,
    FEATURE_SCHEMA_VERSION,
    POLICY_FEATURE_DIM,
    LearnedActionPolicy,
)
from mcss.dynamic.policy_checkpoint import load_policy_checkpoint
from mcss.training.policy_fit import (
    ACTION_TEACHER_SCHEMA,
    fit_action_policy,
    load_teacher_dataset,
    main,
    parse_teacher_row,
    split_scene_ids,
)

_BINDING = {
    "model_content_hash": "carrier-abc",
    "feature_schema_version": FEATURE_SCHEMA_VERSION,
    "render_protocol": "render-v1",
    "budget_protocol": "budget-v1",
    "utility_protocol": "utility-v1",
    "teacher_dataset_hash": "dataset-abc",
    "teacher_source_hashes": {"teacher": "hash-abc"},
}


def _row(row_id: str, scene_id: str, value: float, *, binding=None) -> dict:
    features = [0.0] * POLICY_FEATURE_DIM
    features[0] = value
    features[1] = value * 0.5
    return {
        "schema_version": ACTION_TEACHER_SCHEMA,
        "row_id": row_id,
        "episode_id": f"episode-{scene_id}",
        "scene_id": scene_id,
        "split_id": "train",
        "prefix_step": 0,
        "prefix_frame_id": 1,
        "control_input": {},
        "features": features,
        "actions": ["OFF", "FUSE", "COMPLETE", "ALL"],
        "feasible_mask": [True, True, False, True],
        "losses": [1.0 - value, 1.0 + value, None, 1.0 - 0.5 * value],
        "advantages": [value, -value, None, 0.5 * value],
        "costs": [0.0, 2.0, None, 4.0],
        "horizon_used": 2,
        "provenance": dict(_BINDING if binding is None else binding),
    }


def test_scene_split_is_deterministic_and_has_no_row_scene_leak() -> None:
    scenes = ["scene-a", "scene-b", "scene-c", "scene-d", "scene-e"]
    train_one, valid_one = split_scene_ids(scenes, holdout_fraction=0.4, seed=17)
    train_two, valid_two = split_scene_ids(scenes, holdout_fraction=0.4, seed=17)

    assert (train_one, valid_one) == (train_two, valid_two)
    assert set(train_one).isdisjoint(valid_one)
    assert set(train_one) | set(valid_one) == set(scenes)


def test_masked_fit_is_finite_and_learns_continuous_advantages() -> None:
    rows = [_row(f"row-{index}", "scene-a", -1.0 + index * 0.25) for index in range(9)]
    initial = LearnedActionPolicy(hidden_dim=8, seed=19)
    initial_state = {name: value.detach().clone() for name, value in initial.state_dict().items()}

    result = fit_action_policy(
        rows,
        hidden_dim=8,
        epochs=80,
        learning_rate=0.02,
        holdout_fraction=0.0,
        seed=19,
        init_policy=initial,
    )

    assert torch.isfinite(result.policy.feature_mean).all()
    assert torch.isfinite(result.policy.feature_scale).all()
    assert torch.isfinite(torch.tensor(float(result.metrics["training_loss"])))
    assert result.metrics["heldout_row_count"] == 0
    changed = [
        not torch.equal(initial_state[name], value.detach().cpu())
        for name, value in result.policy.state_dict().items()
        if name.startswith("model.")
    ]
    assert any(changed)
    # The masked COMPLETE label is null and cannot create a nonfinite loss.
    assert result.target_scale >= 1e-6


def test_feature_normalization_uses_training_scenes_once() -> None:
    train_rows = [_row("train-a", "scene-train", 2.0), _row("train-b", "scene-train", 4.0)]
    heldout = _row("valid-a", "scene-valid", 100.0)
    result = fit_action_policy(
        [*train_rows, heldout],
        hidden_dim=6,
        epochs=1,
        holdout_fraction=0.0,
        validation_scene_ids=["scene-valid"],
        seed=3,
    )

    assert result.validation_scene_ids == ("scene-valid",)
    assert result.feature_mean[0].item() == pytest.approx(3.0)
    assert result.feature_scale[0].item() == pytest.approx(1.0)
    assert result.policy.target_mean == pytest.approx(result.target_mean)
    assert result.policy.target_scale == pytest.approx(result.target_scale)


def test_categorical_and_near_constant_features_use_identity_guards() -> None:
    action_indices = [
        FEATURE_NAMES.index(f"previous_action_{action.lower()}")
        for action in ("off", "fuse", "complete", "all")
    ]
    varying_fast_index = FEATURE_NAMES.index("fast_total_norm")
    near_constant_fast_index = FEATURE_NAMES.index("fast_max_abs")
    rows = []
    for index in range(4):
        row = _row(f"row-{index}", "scene-a", 0.1 + index * 0.1)
        features = row["features"]
        for action_index in action_indices:
            features[action_index] = 0.0
        features[action_indices[index]] = 1.0
        features[varying_fast_index] = float(index)
        features[near_constant_fast_index] = index * 5e-7
        rows.append(row)

    result = fit_action_policy(rows, hidden_dim=6, epochs=1, holdout_fraction=0)

    torch.testing.assert_close(
        result.feature_mean[action_indices], torch.zeros(len(action_indices))
    )
    torch.testing.assert_close(
        result.feature_scale[action_indices], torch.ones(len(action_indices))
    )
    assert result.feature_scale[near_constant_fast_index].item() == pytest.approx(1.0)
    assert result.feature_scale[varying_fast_index].item() > 1e-6

    changed = torch.stack([torch.tensor(row["features"]) for row in rows])
    changed[:, near_constant_fast_index] = 0.75
    normalized = (changed - result.feature_mean) / result.feature_scale
    assert normalized[:, near_constant_fast_index].abs().max().item() < 1.0
    assert normalized[:, action_indices].abs().max().item() <= 1.0


def test_conflicting_binding_is_rejected() -> None:
    first = _row("row-a", "scene-a", 0.1)
    second_binding = dict(_BINDING)
    second_binding["model_content_hash"] = "carrier-other"
    second = _row("row-b", "scene-b", 0.2, binding=second_binding)

    with pytest.raises(ValueError, match="binding mismatch"):
        fit_action_policy([first, second], holdout_fraction=0.0)


def test_conflicting_duplicate_row_id_is_rejected() -> None:
    first = _row("same-row", "scene-a", 0.1)
    second = copy.deepcopy(first)
    second["advantages"][0] = 0.2

    with pytest.raises(ValueError, match="conflicting duplicate"):
        fit_action_policy([first, second], holdout_fraction=0.0)


def test_future_top_level_fields_are_rejected_by_row_whitelist() -> None:
    row = _row("row-a", "scene-a", 0.1)
    row["future_label"] = 1.0

    with pytest.raises(ValueError, match="fields mismatch"):
        parse_teacher_row(row)


def test_row_json_round_trip_keeps_strict_schema() -> None:
    row = _row("row-a", "scene-a", 0.1)
    parsed = parse_teacher_row(json.loads(json.dumps(row)))
    assert parsed.row_id == "row-a"
    assert parsed.features.shape == (POLICY_FEATURE_DIM,)
    assert parsed.feasible_mask.tolist() == [True, True, False, True]


def test_sidecar_binding_is_carried_into_loaded_rows(tmp_path) -> None:
    row = _row("row-sidecar", "scene-a", 0.1)
    row["provenance"] = {}
    source = tmp_path / "teacher.jsonl"
    source.write_text(json.dumps(row) + "\n", encoding="utf-8")
    source.with_name("teacher.metadata.json").write_text(
        json.dumps(
            {
                "schema_version": "mcss.dynamic.action_teacher.v1",
                "row_schema_version": ACTION_TEACHER_SCHEMA,
                "binding": _BINDING,
            }
        ),
        encoding="utf-8",
    )

    dataset = load_teacher_dataset([source])
    assert dataset.rows[0].binding == _BINDING
    result = fit_action_policy(dataset.rows, epochs=1, holdout_fraction=0)
    assert result.binding["feature_schema_version"] == FEATURE_SCHEMA_VERSION


def test_loaded_rows_drop_expanded_provenance_after_fingerprint(tmp_path) -> None:
    row = _row("row-compact", "scene-a", 0.1)
    row["provenance"] = {
        **_BINDING,
        "source_data": {
            "manifest_sha256": [
                {"path": f"manifest-{index}.json", "sha256": f"hash-{index}"}
                for index in range(8)
            ]
        },
        "episode": {"episode_id": "episode-scene-a", "scene_id": "scene-a"},
    }
    expected = parse_teacher_row(row)
    source = tmp_path / "compact.jsonl"
    source.write_text(json.dumps(row) + "\n", encoding="utf-8")
    source.with_name("compact.metadata.json").write_text(
        json.dumps(
            {
                "schema_version": "mcss.dynamic.action_teacher.v1",
                "row_schema_version": ACTION_TEACHER_SCHEMA,
                "row_provenance_mode": "episode_compact_v1",
                "binding": _BINDING,
            }
        ),
        encoding="utf-8",
    )

    dataset = load_teacher_dataset([source])

    loaded = dataset.rows[0]
    assert loaded.fingerprint == expected.fingerprint
    assert loaded.binding == _BINDING
    assert loaded.provenance == _BINDING


def test_cli_writes_bound_checkpoint_and_refuses_second_output(tmp_path) -> None:
    source = tmp_path / "teacher.jsonl"
    source.write_text(
        "\n".join(
            json.dumps(_row(f"row-{index}", "scene-a", 0.1 + index * 0.01))
            for index in range(3)
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "policy.pt"
    assert main(
        [
            "--teacher",
            str(source),
            "--output",
            str(output),
            "--epochs",
            "3",
            "--holdout-fraction",
            "0",
            "--seed",
            "7",
        ]
    ) == 0
    policy, metadata = load_policy_checkpoint(output)
    assert policy.target_scale >= 1e-6
    assert metadata["binding"]["feature_schema_version"] == FEATURE_SCHEMA_VERSION
    with pytest.raises(FileExistsError, match="overwrite"):
        main(
            [
                "--teacher",
                str(source),
                "--output",
                str(output),
                "--epochs",
                "1",
                "--holdout-fraction",
                "0",
            ]
        )
