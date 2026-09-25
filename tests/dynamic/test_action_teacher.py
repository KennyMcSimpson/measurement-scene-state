import json

import pytest
import torch

import mcss.training.action_teacher as action_teacher
from mcss.dynamic.carrier import DynamicSceneCarrier
from mcss.dynamic.config import CarrierConfig, WriteConfig
from mcss.dynamic.policy import (
    ACTION_ORDER,
    POLICY_FEATURE_DIM,
    FixedPolicy,
    LearnedActionPolicy,
)
from mcss.dynamic.policy_checkpoint import save_policy_checkpoint
from mcss.dynamic.runner import StreamingRunner
from mcss.dynamic.types import Action, OnlineObservation, hash_fast_content, hash_scene_state
from mcss.dynamic.write_rule import DirectWriteRule
from mcss.geometry import make_intrinsics
from mcss.measurements import FixedMeasurementRenderer
from mcss.training.action_teacher import (
    DEFAULT_POLICY_WORK_UNITS,
    ROW_SCHEMA_VERSION,
    _atomic_json,
    _derive_episode_rollin_seed,
    _index_from_rows,
    _load_rollin,
    _read_rows,
    collect_episode_targets,
    make_binding,
)
from mcss.training.supervision import TrainingSupervision
from mcss.types import Cameras


def _config() -> CarrierConfig:
    return CarrierConfig(
        feature_dim=4,
        hidden_dim=4,
        expansion_dim=6,
        grid_size=(4, 4, 4),
        local_bounds_m=((-1.0, -1.0, 0.1), (1.0, 1.0, 3.0)),
        token_count=12,
    )


def _observation(frame_id: int) -> OnlineObservation:
    pose = torch.eye(4)
    pose[0, 3] = 0.05 * frame_id
    camera = Cameras(make_intrinsics((4, 4), 5.0), pose, (4, 4))
    return OnlineObservation(
        scene_id="scene-a",
        frame_id=frame_id,
        rgb=torch.full((3, 4, 4), 0.2 + 0.01 * frame_id),
        camera=camera,
    )


def _runner(*, max_units: float = 1e12) -> StreamingRunner:
    torch.manual_seed(21)
    config = _config()
    return StreamingRunner(
        DynamicSceneCarrier(config),
        DirectWriteRule(config, WriteConfig(learning_rate=0.1, max_update_norm=0.2)),
        FixedMeasurementRenderer(n_samples=4, ray_chunk_size=8),
        FixedPolicy(Action.OFF),
        max_units=max_units,
    )


def _supervision(depth: torch.Tensor | None = None) -> TrainingSupervision:
    query_camera = Cameras(
        make_intrinsics((4, 4), 5.0).view(1, 1, 3, 3),
        torch.eye(4).view(1, 1, 4, 4),
        (4, 4),
    )
    if depth is None:
        depth = torch.full((1, 1, 1, 4, 4), 2.0)
    return TrainingSupervision(
        episode_id="episode-a",
        scene_id="scene-a",
        query_vault_id="vault-a",
        frame_ids=(20,),
        query_cameras=query_camera,
        query_rgb=torch.zeros((1, 1, 3, 4, 4)),
        query_depth=depth,
    )


def _collect(
    runner: StreamingRunner,
    supervision: TrainingSupervision,
    *,
    horizon: int = 2,
    **kwargs,
):
    return collect_episode_targets(
        runner,
        [_observation(0)],
        [_observation(1), _observation(2), _observation(3)],
        supervision,
        episode_id="episode-a",
        scene_id="scene-a",
        query_vault_id="vault-a",
        horizon=horizon,
        **kwargs,
    )


def test_teacher_rows_have_fixed_actions_features_and_nullable_infeasible_values() -> None:
    rows = _collect(_runner(), _supervision())

    assert len(rows) == 3
    assert rows[0]["actions"] == [action.value for action in ACTION_ORDER]
    assert all(len(row["features"]) == POLICY_FEATURE_DIM == 75 for row in rows)
    assert all(all(value is not None for value in row["losses"]) for row in rows)

    limited = _runner(max_units=133869.0)
    limited_rows = _collect(limited, _supervision(), selected_prefix_steps=[0])
    assert limited_rows[0]["feasible_mask"] == [True, False, False, False]
    assert limited_rows[0]["losses"][1:] == [None, None, None]
    assert limited_rows[0]["advantages"][1:] == [None, None, None]
    assert limited_rows[0]["costs"][1:] == [None, None, None]


def test_teacher_direct_collection_charges_hidden32_rollin_and_selected_prefixes() -> None:
    runner = _runner()
    rows = _collect(runner, _supervision(), selected_prefix_steps=[1, 2])

    assert [row["prefix_step"] for row in rows] == [1, 2]
    assert runner.budget.policy_units == DEFAULT_POLICY_WORK_UNITS == 2639
    assert runner.budget.calls["policy"] == 3


def test_learned_rollin_budget_uses_delegate_work_units() -> None:
    runner = _runner()
    policy = LearnedActionPolicy(hidden_dim=8, seed=4)

    _collect(runner, _supervision(), rollin_policy=policy)

    assert runner.budget.policy_units == policy.declared_work_units == 719


def test_horizon_three_uses_off_for_each_continuation(monkeypatch: pytest.MonkeyPatch) -> None:
    suffixes: list[tuple[str, ...]] = []

    def fake_score(branch, utility, renderer_samples):
        if len(branch.history) >= 3:
            suffixes.append(tuple(record["action"] for record in branch.history[-3:]))
        return 0.0

    monkeypatch.setattr(action_teacher, "_query_absrel", fake_score)
    rows = _collect(
        _runner(),
        _supervision(),
        horizon=3,
        selected_prefix_steps=[0],
    )

    assert len(rows) == 1
    assert sorted(suffixes) == sorted(
        [
            (Action.OFF.value, Action.OFF.value, Action.OFF.value),
            (Action.FUSE.value, Action.OFF.value, Action.OFF.value),
            (Action.COMPLETE.value, Action.OFF.value, Action.OFF.value),
            (Action.ALL.value, Action.OFF.value, Action.OFF.value),
        ]
    )


def test_invalid_query_depth_values_are_masked() -> None:
    depth = torch.full((1, 1, 1, 4, 4), 2.0)
    depth[..., 0, 0] = float("nan")
    depth[..., 0, 1] = 0.0
    depth[..., 0, 2] = -1.0

    utility = action_teacher._build_query_utility(_supervision(depth))
    rows = _collect(_runner(), _supervision(depth), selected_prefix_steps=[0])

    assert utility.valid_mask[0, 0, 0, 0, 0:3].tolist() == [False, False, False]
    assert all(
        value is not None and bool(torch.isfinite(torch.tensor(value)))
        for value in rows[0]["losses"]
    )


def test_candidate_branches_do_not_mutate_the_rollin_runner() -> None:
    runner = _runner()
    observations = [_observation(0), _observation(1), _observation(2), _observation(3)]
    _collect(runner, _supervision())

    expected = _runner()
    expected.reset(
        observations[:1],
        episode_id="episode-a",
        scene_id="scene-a",
        split_id="train",
        query_vault_id="vault-a",
        stream_steps=3,
        query_count=1,
    )
    for observation in observations[1:]:
        expected.step(observation)

    assert hash_scene_state(runner.scene_state) == hash_scene_state(expected.scene_state)
    assert hash_fast_content(runner.fast) == hash_fast_content(expected.fast)
    assert [record["action"] for record in runner.history] == [
        Action.OFF.value,
        Action.OFF.value,
        Action.OFF.value,
    ]


def test_query_labels_do_not_change_prefix_features() -> None:
    first = _collect(_runner(), _supervision())
    second = _collect(_runner(), _supervision(torch.full((1, 1, 1, 4, 4), 3.0)))

    assert [row["features"] for row in first] == [row["features"] for row in second]


def test_make_binding_is_the_flat_seven_key_runtime_contract() -> None:
    binding = make_binding(
        model_content_hash="carrier-hash",
        renderer_samples=7,
        ray_chunk_size=13,
        max_units=123.5,
        horizon=2,
    )

    assert set(binding) == {
        "model_content_hash",
        "feature_schema_version",
        "render_protocol",
        "budget_protocol",
        "utility_protocol",
        "teacher_dataset_hash",
        "teacher_source_hashes",
    }
    assert binding["render_protocol"] == {
        "renderer": "FixedMeasurementRenderer",
        "renderer_samples": 7,
        "ray_chunk_size": 13,
    }
    assert binding["budget_protocol"] == {
        "max_units": 123.5,
        "policy_work_units": 2639,
    }
    assert binding["utility_protocol"] == {
        "horizon": 2,
        "horizon_window": "current_postwrite_plus_off_continuation",
        "continuation_action": "OFF",
        "utility": "mean_masked_depth_abs_rel",
        "query_mask": "finite_positive_depth_common_all_views",
        "query_weights": "uniform_valid_pixels",
    }
    assert binding["teacher_dataset_hash"] == ""
    assert binding["teacher_source_hashes"] == {}


def test_all_invalid_query_depth_is_rejected_before_branch_collection() -> None:
    depth = torch.full((1, 1, 1, 4, 4), float("nan"))
    with pytest.raises(ValueError, match="no valid depth"):
        _collect(_runner(), _supervision(depth))


def test_checkpoint_rollin_must_match_canonical_policy_work_units(tmp_path) -> None:
    policy = LearnedActionPolicy(hidden_dim=8, seed=4)
    binding = make_binding(
        model_content_hash="carrier-hash",
        renderer_samples=4,
        ray_chunk_size=8,
        max_units=1e12,
        horizon=2,
    )
    binding["teacher_dataset_hash"] = "dataset-hash"
    binding["teacher_source_hashes"] = {"rows": "rows-hash"}
    path = tmp_path / "noncanonical-rollin.pt"
    save_policy_checkpoint(path, policy, binding=binding)

    with pytest.raises(ValueError, match="canonical binding"):
        _load_rollin(
            path,
            device="cpu",
            expected_binding=binding,
            seed=0,
        )


def test_read_rows_streams_jsonl_and_preserves_blank_line_numbers(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "action_targets.jsonl"
    row = {
        "schema_version": ROW_SCHEMA_VERSION,
        "row_id": "episode-a:prefix:0",
        "episode_id": "episode-a",
    }
    path.write_text(json.dumps(row) + "\n\n", encoding="utf-8")

    def fail_read_text(*args, **kwargs):
        raise AssertionError("_read_rows must parse JSONL incrementally")

    monkeypatch.setattr(type(path), "read_text", fail_read_text)

    index = _read_rows(path)
    assert index["episode-a"].row_ids == frozenset({row["row_id"]})
    assert index["episode-a"].row_count == 1
    assert index["episode-a"].duplicate_row_count == 0


def test_read_rows_reports_stream_line_number_after_blank_line(tmp_path) -> None:
    path = tmp_path / "action_targets.jsonl"
    path.write_text("\nnot-json\n", encoding="utf-8")

    with pytest.raises(ValueError, match="invalid teacher JSONL line 2"):
        _read_rows(path)


def test_read_rows_retains_only_compact_identity_metadata(tmp_path) -> None:
    path = tmp_path / "action_targets.jsonl"
    rows = [
        {
            "schema_version": ROW_SCHEMA_VERSION,
            "row_id": f"episode-a:prefix:{index}",
            "episode_id": "episode-a",
            "features": ["large-payload" * 1000],
        }
        for index in range(2)
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    index = _read_rows(path)["episode-a"]

    assert index.row_ids == frozenset(row["row_id"] for row in rows)
    assert index.row_count == len(rows)
    assert not hasattr(index, "rows")
    assert "large-payload" not in repr(index)


def test_resume_index_round_trip_and_partial_overlap_are_exact(tmp_path) -> None:
    rows = [
        {
            "schema_version": ROW_SCHEMA_VERSION,
            "row_id": f"collection:episode-a:prefix:{index}",
            "episode_id": "episode-a",
            "features": [index],
        }
        for index in range(2)
    ]
    path = tmp_path / "action_targets.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    durable = _read_rows(path)["episode-a"]
    current = _index_from_rows(rows)

    assert durable.row_fingerprints == current.row_fingerprints
    assert durable.row_ids == current.row_ids
    assert durable.row_count == current.row_count == 2
    partial = _index_from_rows(rows[:1])
    assert partial.row_ids != current.row_ids

    duplicate_path = tmp_path / "duplicate.jsonl"
    duplicate_path.write_text(
        json.dumps(rows[0]) + "\n" + json.dumps(rows[0]) + "\n", encoding="utf-8"
    )
    duplicate = _read_rows(duplicate_path)["episode-a"]
    assert duplicate.row_ids == durable.row_ids - {rows[1]["row_id"]}
    assert duplicate.row_count == 2
    assert duplicate.duplicate_row_count == 1


def test_random_rollin_episode_seed_is_stable_and_episode_specific() -> None:
    first = _derive_episode_rollin_seed(17, "episode-a")
    second = _derive_episode_rollin_seed(17, "episode-b")

    assert first == _derive_episode_rollin_seed(17, "episode-a")
    assert first != second
    assert 0 <= first < 2**63


def test_atomic_json_retries_windows_replace_lock(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "progress.json"
    original_replace = action_teacher.os.replace
    attempts = 0

    def flaky_replace(source, destination):
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise PermissionError("transient Windows sharing violation")
        return original_replace(source, destination)

    monkeypatch.setattr(action_teacher.os, "replace", flaky_replace)
    _atomic_json(target, {"row_count": 2})

    assert attempts == 3
    assert json.loads(target.read_text(encoding="utf-8")) == {"row_count": 2}
    assert not list(tmp_path.glob("progress.json.*.part"))
