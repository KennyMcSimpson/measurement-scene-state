"""Support redesign boundaries without any real holdout loading."""

import hashlib
import json
from dataclasses import FrozenInstanceError, asdict, replace

import pytest

from mcss.mechanism_pilot.support_redesign_contracts import (
    EXPOSED_SCENES,
    FinalHoldoutGate,
    MatchedTrainingConfig,
    SceneRoleLock,
    SupportProtocolConfig,
    assert_matched_training,
    old_exposed_manifest,
    seal_method_lock,
    validate_dev_scene_ids,
)


def roles():
    train = ("train0", "train1", "train2")
    dev = tuple(f"dev{i}" for i in range(8))
    held = tuple(f"held{i}" for i in range(8))
    identities = tuple((sid, f"physical_{sid}") for sid in train + dev + held + EXPOSED_SCENES)
    return SceneRoleLock(train, dev, held, physical_identities=identities)


def fixture_lock(tmp_path):
    config, scene_roles = SupportProtocolConfig("R0"), roles()
    paths = {}
    for name in ["carrier_checkpoint", "support_config", "scene_role_lock", "inference_code"]:
        path = tmp_path / name
        path.write_text(name)
        paths[name] = path
    gate = tmp_path / "gate.json"
    config_digest = hashlib.sha256(
        json.dumps(asdict(config), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    gate.write_text(
        json.dumps(
            {
                "status": "PASS",
                "method": config.method,
                "config_digest": config_digest,
                "dev_scene_ids": list(scene_roles.redesign_dev_scenes),
                "artifact_hashes": {
                    name: hashlib.sha256(path.read_bytes()).hexdigest()
                    for name, path in paths.items()
                },
            }
        )
    )
    return config, scene_roles, paths, gate, seal_method_lock(config, scene_roles, paths, gate)


@pytest.mark.parametrize("name", ["ORACLE_VOLUME", "ORACLE_SUPPORT", "ORACLE_VOLUME_SUPPORT"])
def test_oracle_cannot_enter_deployment_or_holdout(name):
    with pytest.raises(PermissionError):
        SupportProtocolConfig(name)
    config = SupportProtocolConfig(
        name, scope="DIAGNOSTIC_ORACLE", input_channels=("context_gt_depth", "query_gt_geometry")
    )
    config.assert_allowed_scene_role("REDESIGN_DEV")
    with pytest.raises(PermissionError):
        config.assert_allowed_scene_role("FINAL_QUALIFICATION_HOLDOUT")


@pytest.mark.parametrize("channel", ["context_gt_depth", "query_gt", "future_rgb", "model_score"])
def test_gt_free_config_rejects_forbidden_input(channel):
    with pytest.raises(PermissionError):
        SupportProtocolConfig("R1", input_channels=(channel,))


def test_configuration_is_deeply_immutable():
    config = SupportProtocolConfig("R2", hyperparameters=(("threshold", 0.5),))
    with pytest.raises(FrozenInstanceError):
        config.method = "R1"
    with pytest.raises(TypeError):
        SupportProtocolConfig("R1", input_channels=["arrived_rgb"])
    with pytest.raises(TypeError):
        SupportProtocolConfig("R1", hyperparameters=(("nested", []),))


def test_holdout_denied_before_method_lock(tmp_path):
    gate = FinalHoldoutGate(tmp_path / "claim.json")
    with pytest.raises(PermissionError, match="before sealed"):
        gate.claim(SupportProtocolConfig("R0"), roles().final_holdout_scenes)
    assert not gate.claim_path.exists()


def test_failed_dev_gate_denied(tmp_path):
    config, scene_roles, paths, gate, _ = fixture_lock(tmp_path)
    gate.write_text(json.dumps({"status": "FAIL"}))
    with pytest.raises(PermissionError):
        seal_method_lock(config, scene_roles, paths, gate)


def test_atomic_once_claim_and_model_scene_allowlist(tmp_path):
    config, scene_roles, _, _, lock = fixture_lock(tmp_path)
    gate = FinalHoldoutGate(tmp_path / "claim.json", lock)
    session = gate.claim(config, scene_roles.final_holdout_scenes)
    with pytest.raises(PermissionError, match="already been claimed"):
        FinalHoldoutGate(gate.claim_path, lock).claim(config, scene_roles.final_holdout_scenes)
    calls = []
    assert session.load_model_scene("held0", lambda: calls.append("loaded")) is None
    with pytest.raises(PermissionError):
        session.load_model_scene("held0", lambda: calls.append("bad"))
    with pytest.raises(PermissionError):
        session.load_model_scene("train0", lambda: calls.append("bad"))
    with pytest.raises(PermissionError, match="sealed states"):
        session.read_query_ground_truth("held0", "q", [], lambda: calls.append("bad"))
    assert calls == ["loaded"]


def test_locked_config_and_artifacts_cannot_mutate(tmp_path):
    config, scene_roles, paths, _, lock = fixture_lock(tmp_path)
    gate = FinalHoldoutGate(tmp_path / "claim.json", lock)
    with pytest.raises(PermissionError, match="configuration"):
        gate.claim(replace(config, method="R1"), scene_roles.final_holdout_scenes)
    paths["carrier_checkpoint"].write_text("mutated")
    with pytest.raises(PermissionError, match="mutated|artifacts differ"):
        gate.claim(config, scene_roles.final_holdout_scenes)
    assert not gate.claim_path.exists()


def test_mid_inference_artifact_mutation_is_detected(tmp_path):
    config, scene_roles, paths, _, lock = fixture_lock(tmp_path)
    session = FinalHoldoutGate(tmp_path / "claim", lock).claim(
        config, scene_roles.final_holdout_scenes
    )
    with pytest.raises(PermissionError, match="mutated|artifacts differ"):
        session.load_model_scene("held0", lambda: paths["support_config"].write_text("mutated"))


def test_roles_disallow_identity_and_physical_environment_overlap():
    r = roles()
    with pytest.raises(PermissionError):
        replace(r, final_holdout_scenes=("train0",) + r.final_holdout_scenes[1:])
    pairs = tuple(
        (sid, "physical_train0" if sid == "held0" else physical)
        for sid, physical in r.physical_identities
    )
    with pytest.raises(PermissionError):
        replace(r, physical_identities=pairs)
    with pytest.raises(PermissionError):
        replace(r, exposed_scenes=EXPOSED_SCENES[:-1])
    assert old_exposed_manifest()["role"] == "REDESIGN_DEV_EXPOSED"
    assert old_exposed_manifest()["scene_ids"] == list(EXPOSED_SCENES)


def test_dev_guard_runs_before_loader_and_rejects_holdout():
    r = roles()
    d = {
        "TRAIN_SCENES": r.train_scenes,
        "REDESIGN_DEV_SCENES": r.redesign_dev_scenes,
        "FINAL_HOLDOUT_SCENES": r.final_holdout_scenes,
    }
    assert validate_dev_scene_ids(d, ("dev0", EXPOSED_SCENES[0]))
    for ids in [("held0",), ("train0",), ("unknown",), ("dev0", "dev0")]:
        with pytest.raises(PermissionError):
            validate_dev_scene_ids(d, ids)


@pytest.mark.parametrize(
    "field,value",
    [
        ("seeds", (2,)),
        ("total_steps", 200),
        ("loss", "different"),
        ("parameter_count", 900),
        ("scene_ids", ("other",)),
        ("data_sha256", "other"),
        ("optimizer_hyperparameters", (("lr", 0.02),)),
    ],
)
def test_matched_training_rejects_every_nonsupport_difference(field, value):
    config = MatchedTrainingConfig(
        "R0",
        ("train0",),
        (1,),
        "Adam",
        100,
        "RGB+AbsRel",
        "RGB+depth",
        308,
        "minval",
        "hash",
        (("lr", 0.001),),
    )
    redesigned = replace(config, support_method="R12")
    assert assert_matched_training(config, redesigned)
    with pytest.raises(PermissionError):
        assert_matched_training(config, replace(redesigned, **{field: value}))
    assert asdict(config)["support_method"] == "R0"
