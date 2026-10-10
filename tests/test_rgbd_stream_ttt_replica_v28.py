"""V28: the policy table (V7 rows plus TTT_1000), V27 hybrid code re-pointed, the branch."""

import importlib.util
from pathlib import Path

from mcss.mechanism_pilot import rgbd_stream_ttt_v7 as v7


def script(name):
    path = Path(__file__).parents[1] / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_policy_rows_are_core_b_v7s_plus_1000_stream_steps():
    v28 = script("run_rgbd_stream_ttt_replica_v28.py")
    shared = set(v28.POLICY_SPECS) - {"TTT_1000"}
    assert shared == {"NO_STREAM", "OFF", "OFF_CTX_TTT_100", "TTT_100"}
    assert all(v28.POLICY_SPECS[k] == v7.POLICY_SPECS[k] for k in shared)
    assert v28.POLICY_SPECS["TTT_1000"] == (True, "stream", 1000)
    assert v28.CONTROLS == {"TTT_100_WRONG_SCENE": "TTT_100"}


def test_v27_hybrid_code_is_pointed_at_this_round_only(tmp_path):
    v28 = script("run_rgbd_stream_ttt_replica_v28.py")
    hybrid = v28.hybrid_module(tmp_path)
    assert hybrid.ROUNDS == {"REPLICA": tmp_path}
    assert hybrid.EXPECTED_SPLIT == {"REPLICA": "REPLICA"}
    assert set(hybrid.POLICIES) <= set(v28.POLICY_SPECS)
    names = {name for pair in v28.HYBRID_CONTRASTS.values() for name in pair}
    assert "HFILL8_7_TTT_100" in names and "REPROJ7_HARMONIC" in names


def test_branch_follows_the_two_primary_gates():
    v28 = script("run_rgbd_stream_ttt_replica_v28.py")
    assert v28.branch("SUPPORTED", "SUPPORTED") == "A"
    assert v28.branch("SUPPORTED", "PARTIAL") == "B"
    assert v28.branch("NOT_ESTABLISHED", "SUPPORTED") == "C"
    assert v28.branch("PARTIAL", "NOT_ESTABLISHED") == "D"
