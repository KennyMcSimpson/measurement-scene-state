"""Core B V5: EVAL-cohort episodes equal the DEV path, the seal boundary and the cohort rules."""

import copy
import dataclasses
import json

import pytest
import torch
from test_rgbd_stream_write import stream_fixture
from test_rgbd_stream_write_v3 import carrier

from mcss.mechanism_pilot import direct_capacity_contracts as contracts
from mcss.mechanism_pilot import rgbd_stream_ttt_eval_v5 as v5
from mcss.mechanism_pilot import rgbd_stream_ttt_evaluation_v4 as v4
from mcss.mechanism_pilot.direct_capacity_contracts import StateSealBarrier
from mcss.mechanism_pilot.rgbd_depth_bounds import DepthBoundsSceneData
from mcss.types import SceneState

SETTINGS = {"steps": 2, "learning_rate": 1e-3}


def tiny_state(value=0.5):
    return SceneState(
        torch.zeros(1, 1, 2, 2, 2),
        torch.full((1, 3, 2, 2, 2), value),
        torch.zeros(1, 1, 2, 2, 2),
        torch.tensor([[[-1.0, -1.0, 0.0], [1.0, 1.0, 3.0]]]),
    )


def test_cached_barrier_keeps_the_seal_and_verifies_once(monkeypatch):
    calls = []
    original = contracts.hash_scene_state

    def counted(state):
        calls.append(1)
        return original(state)

    monkeypatch.setattr(contracts, "hash_scene_state", counted)
    barrier = v5.CachedSealBarrier(("a", "b"))
    barrier.seal("a", "s0", tiny_state())
    with pytest.raises(PermissionError):
        barrier.state("a")
    barrier.seal("b", "s1", tiny_state(0.25))
    calls.clear()
    barrier.state("a")
    assert len(calls) == 2
    barrier.state("b")
    barrier.state("a")
    assert len(calls) == 2
    with pytest.raises(PermissionError, match="closed"):
        barrier.seal("a", "s0", tiny_state())
    barrier.release()
    assert len(calls) == 4
    field = dataclasses.fields(SceneState)[0].name
    getattr(barrier.state("a"), field).add_(1.0)
    with pytest.raises(PermissionError):
        barrier.release()


def test_cached_barrier_leaves_every_row_unchanged(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    manifest = stream_fixture(tmp_path)
    cohorts = cohorts_of(manifest, tmp_path)
    fast = v5.evaluate_ttt_cohorts(carrier(), SETTINGS, cohorts, tmp_path / "fast", 7, "cpu")

    class Uncached(v5.CachedSealBarrier):
        def assert_ready(self):
            StateSealBarrier.assert_ready(self)

    monkeypatch.setattr(v5, "CachedSealBarrier", Uncached)
    slow = v5.evaluate_ttt_cohorts(carrier(), SETTINGS, cohorts, tmp_path / "slow", 7, "cpu")
    assert len(fast) == len(slow)
    for a, b in zip(fast, slow, strict=True):
        assert {k: v for k, v in a.items() if k != "prediction_file_sha256"} == {
            k: v for k, v in b.items() if k != "prediction_file_sha256"
        }
    old = json.loads((tmp_path / "slow/state_hashes.json").read_text())
    new = json.loads((tmp_path / "fast/state_hashes.json").read_text())
    assert {k: v["state_hash"] for k, v in old.items()} == {
        k: v["state_hash"] for k, v in new.items()
    }


def cohorts_of(manifest, root):
    """The fixture's two DEV scenes as two one-scene EVAL cohorts (same frames and roles)."""
    dev = [r for r in manifest["scenes"] if r["split"] == "DEV"]
    cohorts = {}
    for name, record in zip(v5.COHORTS, dev, strict=True):
        cohort = copy.deepcopy(manifest)
        cohort["scenes"] = [{**copy.deepcopy(record), "split": name}]
        cohorts[name] = (cohort, root)
    return cohorts


def test_eval_contexts_and_bounds_equal_the_dev_scene_data(tmp_path):
    manifest = stream_fixture(tmp_path)
    for cohort, base in cohorts_of(manifest, tmp_path).values():
        record = cohort["scenes"][0]
        dev = next(r for r in manifest["scenes"] if r["scene_id"] == record["scene_id"])
        data = DepthBoundsSceneData(dev, manifest, tmp_path, "cpu", [])
        mine = v5.eval_context(record, cohort, base, "cpu", [])
        for role in ("A", "B"):
            theirs = data.context(role)
            assert [o.frame_id for o in mine[role]] == [o.frame_id for o in theirs]
            assert torch.equal(mine[role].depths, theirs.depths)
            for a, b in zip(mine[role], theirs, strict=True):
                assert torch.equal(a.rgb, b.rgb)
                assert torch.equal(a.camera.c2w, b.camera.c2w)
                assert torch.equal(a.camera.intrinsics, b.camera.intrinsics)
            assert torch.equal(v5.context_bounds(mine[role], "cpu"), data.bounds[role])


def test_cached_states_equal_core_b_v4_on_the_same_scenes(tmp_path):
    manifest = stream_fixture(tmp_path)
    model = carrier()
    v4.evaluate_ttt(model, SETTINGS, manifest, tmp_path, tmp_path / "v4", 7, "cpu")
    v5.evaluate_ttt_cohorts(
        model, SETTINGS, cohorts_of(manifest, tmp_path), tmp_path / "v5", 7, "cpu"
    )
    old = json.loads((tmp_path / "v4/state_hashes.json").read_text())
    new = json.loads((tmp_path / "v5/state_hashes.json").read_text())
    assert set(old) == set(new)
    for key, item in old.items():
        if item["policy"] in ("NO_STREAM", "OFF"):
            assert new[key]["state_hash"] == item["state_hash"]
            assert new[key]["bounds"] == item["bounds"]
            assert new[key]["stream_frame_ids"] == item["stream_frame_ids"]


def test_cohort_evaluation_seals_every_policy_before_query_depth(tmp_path):
    manifest = stream_fixture(tmp_path)
    out = tmp_path / "evaluation"
    rows = v5.evaluate_ttt_cohorts(
        carrier(), SETTINGS, cohorts_of(manifest, tmp_path), out, 7, "cpu"
    )
    assert len(rows) == 2 * 2 * 2 * 4
    assert {r["cohort"] for r in rows} == set(v5.COHORTS)
    access = json.loads((out / "GT_access.json").read_text())
    marker = next(i for i, e in enumerate(access) if e.get("event") == "ALL_STREAM_STATES_SEALED")
    early = [e for e in access[:marker] if "depth" in e.get("channels", [])]
    assert {e["purpose"] for e in early} == {"CONTEXT_ONLY_RGBD", "STREAM_RGBD"}
    assert all(e["frame_id"] not in (8, 9) for e in early)
    assert all(e.get("scene_id") != "train_unused" for e in access)
    meta = json.loads((out / "state_hashes.json").read_text())
    donors = {m["scene_id"]: m["donor_scene_id"] for m in meta.values() if "donor_scene_id" in m}
    assert donors == {"capacity0": "capacity1", "capacity1": "capacity0"}
    assert all(m["fast_step"] == 2 for m in meta.values() if m["policy"] == "TTT")
    with pytest.raises(FileExistsError):
        v5.evaluate_ttt_cohorts(carrier(), SETTINGS, cohorts_of(manifest, tmp_path), out, 7, "cpu")


def test_cohorts_must_be_labelled_disjoint_and_unprotected(tmp_path):
    manifest = stream_fixture(tmp_path)
    cohorts = cohorts_of(manifest, tmp_path)
    assert set(v5.cohort_records(cohorts)) == {"capacity0", "capacity1"}
    twice = copy.deepcopy(cohorts)
    twice["EVAL_V4"] = (copy.deepcopy(cohorts["EVAL_V3"][0]), tmp_path)
    twice["EVAL_V4"][0]["scenes"][0]["split"] = "EVAL_V4"
    protected = copy.deepcopy(cohorts)
    protected["EVAL_V3"][0]["protected_scene_ids"].append("capacity0")
    mislabelled = copy.deepcopy(cohorts)
    mislabelled["EVAL_V3"][0]["scenes"][0]["split"] = "DEV"
    for bad in (twice, protected, mislabelled):
        with pytest.raises(PermissionError):
            v5.cohort_records(bad)
