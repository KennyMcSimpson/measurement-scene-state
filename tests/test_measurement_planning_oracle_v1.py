"""Leakage and integrity tests of EXP-MEASUREMENT-PLANNING-ORACLE-V1 on a synthetic scene (no
project data)."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from mcss.measurement_planning import geometry as g
from mcss.measurement_planning import heuristics as h
from mcss.measurement_planning import oracle
from mcss.measurement_planning.information import (
    CANDIDATE_DEPTH_PURPOSE,
    BudgetExceededError,
    MeasurementEnv,
    TeacherAccess,
    candidate_pool,
    sanitized_record,
)

SIZE = (12, 16)
K = [[10.0, 0.0, 7.5], [0.0, 10.0, 5.5], [0.0, 0.0, 1.0]]
SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_measurement_planning_oracle_v1.py"


def _script():
    spec = importlib.util.spec_from_file_location("mp_oracle_v1_script", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _depth(x, scale=1.0):
    """Ray distance to the plane z = 3 m (scaled), with a small step so frames differ."""
    v, u = np.indices(SIZE)
    rays = np.sqrt(((u - 7.5) / 10) ** 2 + ((v - 5.5) / 10) ** 2 + 1)
    z = np.where(u > 8, 3.0, 2.5) * scale
    return (z * rays).astype(np.float32)


def _record(tmp_path, depth_scale=None):
    depth_scale = depth_scale or {}
    frames = []
    rng = np.random.default_rng(0)
    for fid in range(10):
        c2w = np.eye(4)
        c2w[0, 3] = 0.15 * fid
        c2w[1, 3] = 0.05 * (fid % 3)
        rgb = (rng.random((*SIZE, 3)) * 255).astype(np.uint8)
        Image.fromarray(rgb).save(tmp_path / f"{fid}.png")
        np.save(tmp_path / f"{fid}.npy", _depth(c2w[0, 3], depth_scale.get(fid, 1.0)))
        frames.append(
            {
                "frame_id": fid,
                "rgb": str(tmp_path / f"{fid}.png"),
                "depth": str(tmp_path / f"{fid}.npy"),
                "intrinsics": K,
                "c2w": c2w.tolist(),
            }
        )
    roles = {
        "anchor": 0,
        "context_a": [0, 1, 2],
        "context_b": [0, 3, 4],
        "primary_query": [8, 9],
        "query": [8, 9],
        "secondary_query": [],
        "stream": [],
    }
    return {"scene_id": "synthetic_001", "frames": frames, "roles": roles}


def _env(record, budget, log=None, role="A"):
    pool = candidate_pool(record)
    return MeasurementEnv(record, role, pool, budget, SIZE, Path("/"), [] if log is None else log)


def test_candidate_pool_excludes_context_and_queries(tmp_path):
    record = _record(tmp_path)
    assert candidate_pool(record) == [5, 6, 7]
    assert candidate_pool(record, 2) == [5, 6]
    with pytest.raises(PermissionError):
        sanitized_record(record, "A", [5, 8])


def test_candidate_depth_hidden_until_requested(tmp_path):
    log = []
    env = _env(_record(tmp_path), 1, log)
    views = env.candidates()
    assert all(not hasattr(v, "depth") for v in views)
    assert not [e for e in log if "depth" in e["channels"] and e["frame_id"] in (5, 6, 7)]
    env.request_depth(6)
    depth_reads = [e for e in log if e["purpose"] == CANDIDATE_DEPTH_PURPOSE]
    assert [e["frame_id"] for e in depth_reads] == [6]


def test_query_frames_and_other_role_unreachable(tmp_path):
    env = _env(_record(tmp_path), 0)
    assert set(env._media.frames) == {0, 1, 2, 5, 6, 7}
    with pytest.raises(KeyError):
        env._media.read((8,), depth_allowed=False, purpose="X")
    with pytest.raises(KeyError):
        env._media.read((3,), depth_allowed=False, purpose="X")


def test_budget_accounting_is_exact(tmp_path):
    env = _env(_record(tmp_path), 1)
    env.request_depth(5)
    with pytest.raises(BudgetExceededError):
        env.request_depth(6)
    env = _env(_record(tmp_path), 2)
    env.request_depth(5)
    with pytest.raises(ValueError):
        env.request_depth(5)
    with pytest.raises(PermissionError):
        env.request_depth(8)


@pytest.mark.parametrize("name", sorted(h.DEPLOYABLE))
@pytest.mark.parametrize("k", [1, 2, 3])
def test_every_selector_spends_exactly_k_legally(tmp_path, name, k):
    record, log = _record(tmp_path), []
    env = _env(record, k, log)
    h.DEPLOYABLE[name](env, k, constant=2.37)
    assert len(env.requested) == k == len(set(env.requested))
    audit = oracle.audit_log(log, {0, 1, 2}, {5, 6, 7}, k, name)
    assert audit["ok"], audit


def test_selectors_reject_teacher_access(tmp_path):
    record = _record(tmp_path)
    teacher = TeacherAccess(record, [8, 9], SIZE, Path("/"), [])
    with pytest.raises(TypeError):
        h.select_uniform(teacher, 1)


def test_selection_ignores_unrequested_candidate_depth(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    first = _record(tmp_path / "a")
    second = _record(tmp_path / "b", depth_scale={5: 2.0, 6: 0.5, 7: 3.0})
    for name, select in h.DEPLOYABLE.items():
        a, b = _env(first, 1), _env(second, 1)
        select(a, 1, constant=2.37)
        select(b, 1, constant=2.37)
        assert a.requested == b.requested, name
    fa = h.candidate_features(
        h.MeasuredGeometry(_env(first, 0).context()), _env(first, 0).candidates(), 2.37, [5, 6, 7]
    )
    fb = h.candidate_features(
        h.MeasuredGeometry(_env(second, 0).context()), _env(second, 0).candidates(), 2.37, [5, 6, 7]
    )
    assert json.dumps(fa, sort_keys=True) == json.dumps(fb, sort_keys=True)


def test_random_selector_uses_frozen_seed(tmp_path):
    record = _record(tmp_path)
    picks = []
    for _ in range(3):
        env = _env(record, 2)
        h.select_random(env, 2)
        picks.append(env.requested)
    assert picks[0] == picks[1] == picks[2]


def test_subset_union_equals_frozen_multi_frame_path(tmp_path):
    record = _record(tmp_path)
    teacher = TeacherAccess(record, [8, 9], SIZE, Path("/"), [])
    geo, inp = g.frozen("geometry"), g.frozen("inpainting")
    context = teacher.frames([0, 1, 2])
    extra = teacher.frames([5, 7])
    camera = teacher.query_camera(8)
    points = {**g.frame_points(context), **g.frame_points(extra)}
    from mcss.mechanism_pilot.rgbd_evidence_carrier import RGBDContext

    together = RGBDContext([*context, *extra], torch.cat((context.depths, extra.depths)))
    depth_frozen, hit_frozen = geo.reproject(geo.context_points(together), camera)
    depth_union = g.union([g.reproject(points[f], camera) for f in (0, 1, 2, 5, 7)])
    assert np.array_equal(depth_frozen, depth_union)
    cosine = inp.ray_cosine(camera)
    frozen_fill = inp.harmonic_fill(depth_frozen, hit_frozen, cosine, 2.37)
    ours, _ = g.harmonic(depth_union, cosine, 2.37)
    assert np.array_equal(frozen_fill, ours)
    base = g.union([g.reproject(points[f], camera) for f in (0, 1, 2)])
    k0_frozen, k0_hit = geo.reproject(geo.context_points(context), camera)
    assert np.array_equal(base, k0_frozen)
    assert np.array_equal(
        inp.harmonic_fill(k0_frozen, k0_hit, cosine, 2.37), g.harmonic(base, cosine, 2.37)[0]
    )


def test_teacher_refuses_to_fuse_queries(tmp_path):
    teacher = TeacherAccess(_record(tmp_path), [8, 9], SIZE, Path("/"), [])
    with pytest.raises(PermissionError):
        teacher.frames([5, 8])


def test_analysis_is_deterministic_and_oracle_dominates():
    script = _script()
    rng = np.random.default_rng(1)
    pool = [5, 6, 7, 9]
    meta = {
        "scene_id": "s",
        "pool": pool,
        "selections": {"UNIFORM": {str(k): pool[:k] for k in range(1, 5)}},
        "diagnostic_selections": {"QUERY_NEAREST": {str(k): pool[-k:] for k in range(1, 5)}},
        "table": {f: rng.random((16, 2)) for f in script.FIELDS},
    }
    first, second = script.episode_methods(meta), script.episode_methods(meta)
    assert json.dumps(first, sort_keys=True, default=str) == json.dumps(
        second, sort_keys=True, default=str
    )
    for k in range(1, 5):
        assert first[k]["ORACLE"]["all"] <= first[k]["UNIFORM"]["all"] + 1e-12
        assert first[k]["ORACLE"]["all"] <= first[k]["RANDOM_EXPECTED"]["all"] + 1e-12
        assert first[k]["WORST"]["all"] >= first[k]["RANDOM_EXPECTED"]["all"] - 1e-12


def test_protected_or_duplicate_scene_refused(tmp_path, monkeypatch):
    script = _script()
    manifest = {
        "image_size": list(SIZE),
        "protected_scene_ids": ["synthetic_001"],
        "scenes": [_record(tmp_path)],
    }
    path = tmp_path / "m.json"
    path.write_text(json.dumps(manifest))
    monkeypatch.setitem(script.MANIFESTS, "EVAL_V3", path)
    monkeypatch.setitem(script.MANIFESTS, "EVAL_V4", path)
    with pytest.raises(PermissionError):
        script.records("EVAL34")
