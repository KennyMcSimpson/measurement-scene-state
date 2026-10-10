"""Core B V10: four registered carriers load frozen into Core B's class; the run and the branch."""

import copy
import importlib.util
import json
from pathlib import Path

import pytest
import torch
from test_rgbd_stream_write import stream_fixture
from test_rgbd_stream_write_v3 import carrier

from mcss.dynamic.types import hash_value
from mcss.mechanism_pilot import rgbd_long_train_carrier as v17
from mcss.mechanism_pilot import rgbd_long_width_carrier as v19
from mcss.mechanism_pilot import rgbd_stream_ttt_v7 as v7
from mcss.mechanism_pilot import rgbd_stream_ttt_v10 as v10
from mcss.mechanism_pilot import rgbd_stream_write_v3 as v3


def script(name):
    path = Path(__file__).parents[1] / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_policy_rows_are_core_b_v7s():
    assert all(v10.POLICY_SPECS[k] == v7.POLICY_SPECS[k] for k in v10.POLICY_SPECS)
    assert set(v10.POLICY_SPECS) == {"NO_STREAM", "OFF", "TTT_100"}
    assert v10.CONTROLS == {"TTT_100_WRONG_SCENE": "TTT_100"}
    widths = {(c[2], c[3]) for c in v10.CARRIERS.values()}
    assert widths == {(32, "TRAIN72"), (32, "TRAIN127"), (64, "TRAIN72"), (64, "TRAIN127")}


@pytest.mark.parametrize("variant", list(v10.CARRIERS))
def test_every_registered_carrier_loads_frozen_at_step_24000_only(tmp_path, variant):
    _, source, hidden, _ = v10.CARRIERS[variant]
    model = source.make_carrier("L", 7, "cpu")
    path = tmp_path / "step_024000.pt"
    source.save_checkpoint(path, model, variant="L", seed=7, step=24000, lock_sha256="x")
    loaded, payload = v10.load_stream_carrier(variant, path, "cpu")
    assert isinstance(loaded, v3.StreamResolutionCarrier) and not loaded.training
    assert loaded.config.hidden_dim == hidden and payload["step"] == 24000
    assert not any(p.requires_grad for p in loaded.parameters())
    assert hash_value(loaded.state_dict()) == hash_value(model.state_dict())
    early = tmp_path / "step_006000.pt"
    source.save_checkpoint(early, model, variant="L", seed=7, step=6000, lock_sha256="x")
    with pytest.raises(PermissionError):
        v10.load_stream_carrier(variant, early, "cpu")


def test_v17_l_loads_as_core_b_v3_loads_it_and_schemas_do_not_mix(tmp_path):
    path = tmp_path / "v17.pt"
    v17.save_checkpoint(
        path, v17.make_carrier("L", 7, "cpu"), variant="L", seed=7, step=24000, lock_sha256="x"
    )
    mine, _ = v10.load_stream_carrier("V17_L", path, "cpu")
    theirs, _ = v3.load_frozen_carrier(path, "cpu")
    assert hash_value(mine.state_dict()) == hash_value(theirs.state_dict())
    wide = tmp_path / "v19.pt"
    v19.save_checkpoint(
        wide, v19.make_carrier("L", 7, "cpu"), variant="L", seed=7, step=24000, lock_sha256="x"
    )
    for variant in ("V17_L", "V18_L", "V20_L"):
        with pytest.raises(PermissionError):
            v10.load_stream_carrier(variant, wide, "cpu")


def test_two_cohorts_are_sealed_before_query_depth(tmp_path):
    torch.set_num_threads(1)
    manifest = stream_fixture(tmp_path)
    dev = [r for r in manifest["scenes"] if r["split"] == "DEV"]
    cohorts = {}
    for name, record in zip(("EVAL_V3", "EVAL_V4"), dev, strict=True):
        cohort = copy.deepcopy(manifest)
        cohort["scenes"] = [{**copy.deepcopy(record), "split": name}]
        cohorts[name] = (cohort, tmp_path)
    expected = {name: name for name in cohorts}
    specs = {name: (s, sup, min(steps, 2)) for name, (s, sup, steps) in v10.POLICY_SPECS.items()}
    out = tmp_path / "evaluation"
    rows = v10.evaluate_v10(carrier(), cohorts, expected, out, 7, "cpu", specs)
    assert len(rows) == 2 * 2 * 2 * (len(specs) + len(v10.CONTROLS))
    access = json.loads((out / "GT_access.json").read_text())
    marker = next(i for i, e in enumerate(access) if e.get("event") == "ALL_STREAM_STATES_SEALED")
    early = [e for e in access[:marker] if "depth" in e.get("channels", [])]
    assert {e["purpose"] for e in early} == {"CONTEXT_ONLY_RGBD", "STREAM_RGBD"}
    assert all(e["frame_id"] not in (8, 9) for e in early)
    meta = json.loads((out / "state_hashes.json").read_text())
    donors = {m["scene_id"]: m["donor_scene_id"] for m in meta.values() if "donor_scene_id" in m}
    assert donors == {"capacity0": "capacity1", "capacity1": "capacity0"}


def rows_for(gains):
    rows = []
    for name, gain in gains.items():
        values = {
            "NO_STREAM": 0.30,
            "OFF": 0.24,
            "TTT_100": 0.24 - gain,
            "TTT_100_WRONG_SCENE": 0.25,
        }
        for seed in (1, 2):
            for i in range(8):
                for role in ("A", "B"):
                    for qid in (8, 9):
                        for policy, value in values.items():
                            rows.append(
                                {
                                    "carrier": name,
                                    "seed": seed,
                                    "cohort": "EVAL_V3",
                                    "scene_id": f"s{i}",
                                    "role": role,
                                    "query_id": qid,
                                    "policy": policy,
                                    "depth_absrel": value + 0.002 * ((i + qid + seed) % 3),
                                }
                            )
    return rows


def test_branches_count_the_three_new_carriers_and_factors_are_contrasts_of_gains():
    analyze = script("run_rgbd_stream_ttt_carrier_v10.py").analyze
    gains = {"V17_L": 0.02, "V18_L": 0.02, "V19_L": 0.03, "V20_L": 0.01}
    result = analyze(rows_for(gains), [1, 2])
    assert result["INTERPRETATION_BRANCH"] == "A"
    factors = result["factors_on_gain"]
    assert factors["WIDTH_EFFECT_ON_GAIN"]["mean"] == pytest.approx(0.0, abs=1e-9)
    assert factors["DATA_EFFECT_ON_GAIN"]["mean"] == pytest.approx(-0.01)
    assert factors["INTERACTION_ON_GAIN"]["mean"] == pytest.approx(-0.02)
    some = {**gains, "V19_L": 0.0}
    assert analyze(rows_for(some), [1, 2])["INTERPRETATION_BRANCH"] == "B"
    none = {"V17_L": 0.02, "V18_L": 0.0, "V19_L": 0.0, "V20_L": -0.01}
    assert analyze(rows_for(none), [1, 2])["INTERPRETATION_BRANCH"] == "C"
