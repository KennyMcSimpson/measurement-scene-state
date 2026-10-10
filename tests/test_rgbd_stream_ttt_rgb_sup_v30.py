"""Core B V30: RGB-only stream frames as write supervision only (never cached); run and branch."""

import copy
import importlib.util
import json
import math
from pathlib import Path

import torch
from test_rgbd_stream_write import stream_fixture
from test_rgbd_stream_write_v3 import carrier, contexts

from mcss.dynamic.types import hash_scene_state
from mcss.mechanism_pilot import rgbd_stream_ttt_rgb_sup_v30 as v30
from mcss.mechanism_pilot import rgbd_stream_ttt_v7 as v7
from mcss.mechanism_pilot import rgbd_stream_ttt_v11 as v11
from mcss.mechanism_pilot.rgbd_evidence_carrier import RGBDContext


def script(name):
    path = Path(__file__).parents[1] / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_no_policy_caches_the_stream():
    assert v30.POLICY_SPECS["NO_STREAM"] == v7.POLICY_SPECS["NO_STREAM"]
    assert v30.POLICY_SPECS["CTX_TTT_100"] == v7.POLICY_SPECS["CTX_TTT_100"]
    assert v30.POLICY_SPECS["RGB_SUP_100"] == (False, "stream", 100)
    assert v30.POLICY_SPECS["CTX_RGB_SUP_100"] == (False, "all", 100)
    assert not any(spec[0] for spec in v30.POLICY_SPECS.values())
    assert v30.CONTROLS == {"CTX_RGB_SUP_100_WRONG_SCENE": "CTX_RGB_SUP_100"}


def rgb_only(stream):
    return RGBDContext(list(stream), torch.zeros_like(stream.depths))


def test_rgb_supervision_writes_from_uncached_rgb_only_frames():
    warm, stream, bounds, _ = contexts()
    stream = rgb_only(stream)
    model = carrier()
    specs = {
        "NO_STREAM": (False, None, 0),
        "CTX_TTT_100": (False, "context", 3),
        "RGB_SUP_100": (False, "stream", 3),
        "CTX_RGB_SUP_100": (False, "all", 3),
    }
    out = {p: v30.unroll_v30(model, warm, stream, bounds, p, "e", 7, specs) for p in specs}
    warm_ids = [o.frame_id for o in warm]
    for policy, (_, _, fast, episode, losses) in out.items():
        assert episode.source_frames == warm_ids
        assert all(math.isfinite(x) for x in losses)
        assert fast.step == specs[policy][2]
    hashes = {p: hash_scene_state(r[0]) for p, r in out.items()}
    assert len(set(hashes.values())) == 4
    again = v30.unroll_v30(model, warm, stream, bounds, "CTX_RGB_SUP_100", "e", 7, specs)
    assert hash_scene_state(again[0]) == hashes["CTX_RGB_SUP_100"]


def test_sealed_run_uses_rgb_only_streams_and_restores_core_b_v11(tmp_path):
    torch.set_num_threads(1)
    manifest = stream_fixture(tmp_path)
    dev = [r for r in manifest["scenes"] if r["split"] == "DEV"]
    cohorts = {}
    for name, record in zip(("EVAL_V3", "EVAL_V4"), dev, strict=True):
        cohort = copy.deepcopy(manifest)
        cohort["scenes"] = [{**copy.deepcopy(record), "split": name}]
        cohorts[name] = (cohort, tmp_path)
    expected = {name: name for name in cohorts}
    specs = {name: (s, sup, min(steps, 2)) for name, (s, sup, steps) in v30.POLICY_SPECS.items()}
    unroll, loader = v11.unroll_v11, v11.StreamRGBDLoader
    out = tmp_path / "evaluation"
    rows = v30.evaluate_v30(carrier(), cohorts, expected, out, 7, "cpu", specs)
    assert v11.unroll_v11 is unroll and v11.StreamRGBDLoader is loader
    assert len(rows) == 2 * 2 * 2 * (len(specs) + len(v30.CONTROLS))
    access = json.loads((out / "GT_access.json").read_text())
    marker = next(i for i, e in enumerate(access) if e.get("event") == "ALL_STREAM_STATES_SEALED")
    run = script("run_rgbd_stream_ttt_rgb_sup_v30.py")
    assert run.rgb_only_audit(access, marker) > 0
    meta = json.loads((out / "state_hashes.json").read_text())
    for m in meta.values():
        if m["policy"] == "CTX_RGB_SUP_100":
            assert m["cached_frame_ids"] == m["warmup_frame_ids"]
            assert len(m["supervised_frame_ids"]) == len(m["warmup_frame_ids"]) + 4


def test_branch_follows_the_two_primary_gates():
    run = script("run_rgbd_stream_ttt_rgb_sup_v30.py")
    assert run.branch("SUPPORTED", "SUPPORTED") == "A"
    assert run.branch("SUPPORTED", "PARTIAL") == "B"
    assert run.branch("NOT_ESTABLISHED", "SUPPORTED") == "C"
    assert run.branch("HARMFUL", "NOT_ESTABLISHED") == "D"
