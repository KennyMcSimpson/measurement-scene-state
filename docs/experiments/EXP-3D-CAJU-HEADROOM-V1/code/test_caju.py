"""Unit and contract tests of EXP-3D-CAJU-HEADROOM-V1 (CPU). Run from the repository root with PYTHONPATH=src:
  python -m pytest -q -p no:cacheprovider outputs/EXP-3D-CAJU-HEADROOM-V1/code/test_caju.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))
import caju_logic as L  # noqa: E402
from mcss.mechanism_pilot.direct_capacity_statistics import describe  # noqa: E402
from mcss.mechanism_pilot.rgbd_stream_write import stream_frame_ids  # noqa: E402

MANIFEST = Path("outputs/EXP-3D-RGBD-ARKIT-DATA-V1/manifest_arkit_trainval.json")
ROOT = CODE.parent


def records():
    return json.loads(MANIFEST.read_text())["scenes"]


def frames(**per_frame):
    """{frame: {action: loss}} from f<frame>=(OFF, SHORT, FULL) losses."""
    return {int(k[1:]): dict(zip(L.ACTIONS, v, strict=True)) for k, v in per_frame.items()}


def test_episode_rule_on_every_arkit_record():
    for r in records():
        stream = stream_frame_ids(r, "A")
        assert stream == stream_frame_ids(r, "B")
        episodes = L.build_episodes(r, stream)
        assert [e["role"] for e in episodes] == ["A", "B"]
        assert episodes[0]["history"] == episodes[1]["history"] and episodes[0]["stream"] == episodes[1]["stream"]
        for e in episodes:
            inputs, targets = set(e["context"]) | set(e["stream"]), set(e["queries"]) | set(e["history"])
            assert not inputs & targets
            assert min(e["queries"]) > max(inputs)
            assert all(min(e["stream"]) < h < max(e["stream"]) for h in e["history"])
            assert e["place"] == r["physical_scene_id"]


def test_episode_rule_refuses_leaks():
    r = records()[0]
    with pytest.raises(ValueError):
        L.build_episodes(r, (5, 8, 10, 14))  # a query frame inside the stream
    late = json.loads(json.dumps(r))
    late["roles"]["primary_query"] = [12, 15]
    late["roles"]["query"] = [12, 15]
    with pytest.raises(ValueError):
        L.build_episodes(late, stream_frame_ids(late, "A"))  # frame 12 would precede an arrived frame
    with pytest.raises(ValueError):
        L.build_episodes(r, None)


def test_selection_is_seeded_disjoint_and_sized():
    train = [r["physical_scene_id"] for r in records() if r["split"] == "TRAIN"]
    a, b = L.select_places(train), L.select_places(list(reversed(train)))
    assert a == b
    assert len(a["CAJU_SEL"]) == 16 and len(a["CAJU_EVAL"]) == 34
    assert len(a["RESERVED_UNTOUCHED"]) == len(set(train)) - 50
    assert not set(a["CAJU_SEL"]) & set(a["CAJU_EVAL"])
    dev = {r["physical_scene_id"] for r in records() if r["split"] == "DEV"}
    assert not dev & (set(a["CAJU_SEL"]) | set(a["CAJU_EVAL"]))


def test_core_b_readout_matches_v3_definition():
    cb, f7 = L.core_b(np.array([[2.0, 1.0]]), np.array([[1.0, 0.25]]), np.array([[4.0, 9.0]]),
                      np.array([[0.5, 1.0]]))
    # pixel 0: ray = 2 / 1 / 0.5 = 4 (covered), CORE_B = sqrt(4 x 4); pixel 1: alpha < 0.5, F7 = R7 = 9
    assert np.allclose(f7, [[4.0, 9.0]]) and np.allclose(cb, [[4.0, 9.0]])
    floored, _ = L.core_b(np.array([[0.0]]), np.array([[0.9]]), np.array([[1.0]]), np.array([[1.0]]))
    assert np.allclose(floored, [[np.sqrt(1e-6)]])


def test_absrel_skips_invalid_ground_truth():
    gt, pred = np.array([1.0, 0.0, np.nan, 2.0]), np.array([1.5, 5.0, 5.0, 1.0])
    assert L.absrel(pred, gt) == pytest.approx(0.5)
    assert L.absrel(pred, gt, np.array([False, True, True, True])) == pytest.approx(0.5)
    assert L.absrel(pred, np.zeros(4)) is None


def test_utilities_best_action_and_ties():
    u = L.utilities(frames(f14=(0.10, 0.09, 0.08), f15=(0.10, 0.07, 0.12)))
    assert u["OFF"] == 0 and u["SHORT"] == pytest.approx(0.02) and u["FULL"] == pytest.approx(0.0)
    assert L.best_action(u) == "SHORT"
    assert L.best_action({"OFF": 0.0, "SHORT": 0.0, "FULL": 0.0}) == "OFF"
    assert L.cheapest_sufficient({"OFF": 0.0, "SHORT": 0.0009, "FULL": 0.001}, "FULL") == "SHORT"
    assert L.cheapest_sufficient({"OFF": -0.001, "SHORT": 0.0, "FULL": 0.002}, "FULL") == "FULL"


def test_cross_query_oracle_is_held_out():
    f = frames(f14=(0.10, 0.10, 0.05), f15=(0.10, 0.11, 0.13))
    value, cost, picks = L.episode_policies(f, "FULL")
    assert value["ORACLE"] == pytest.approx(max(L.utilities(f).values()))
    assert picks["XORACLE"] == ["FULL", "OFF"]
    assert value["XORACLE"] == pytest.approx(-0.015)  # FULL on frame 15 (-0.03) and OFF on frame 14 (0)
    assert cost["XORACLE"] == pytest.approx(50.0)
    single, _, _ = L.episode_policies(frames(f14=(0.10, 0.10, 0.05)), "FULL")
    assert single["XORACLE"] == single["FIX"]  # no held-out query: the fixed action


def test_in_sample_oracle_never_below_any_fixed_action():
    rng = np.random.default_rng(0)
    for _ in range(200):
        f = {q: dict(zip(L.ACTIONS, rng.uniform(0.05, 0.2, 3), strict=True)) for q in (14, 15)}
        value, _, _ = L.episode_policies(f, "FULL")
        assert all(value["ORACLE"] >= value[f"FIXED_{a}"] - 1e-12 for a in L.ACTIONS)


def test_place_tables_cross_role_and_place_oracle():
    episodes = {"s/A": {"place": "p", "frames": frames(f14=(0.1, 0.1, 0.05))},
                "s/B": {"place": "p", "frames": frames(f14=(0.1, 0.1, 0.15))}}
    values, costs, _ = L.policy_tables(episodes, "FULL")
    assert values["FIX"]["p"] == pytest.approx(0.0)
    assert values["XROLE"]["p"] == pytest.approx(-0.025)  # FULL chosen on A, OFF chosen on B, each scored on the other
    assert values["VORACLE"]["p"] == pytest.approx(0.0) and costs["VORACLE"]["p"] == 0.0


def test_select_fixed_prefers_the_cheaper_action_on_ties():
    assert L.select_fixed({"s/A": {"place": "p", "frames": frames(f14=(0.1, 0.1, 0.1))}})[0] == "OFF"


def test_pareto_frontier_and_fixed_mixtures():
    episodes = {f"s{i}/A": {"place": f"p{i}", "frames": frames(f14=(0.10, 0.09, 0.08 if i % 2 else 0.11),
                                                                 f15=(0.10, 0.09, 0.08 if i % 2 else 0.11))}
                for i in range(6)}
    front = L.frontier(L.pareto_points(episodes, "in_sample"))
    assert all(a["steps"] < b["steps"] and a["utility"] < b["utility"] for a, b in zip(front, front[1:]))
    assert front[-1]["utility"] == pytest.approx(0.015)
    fixed = [{"steps": 0.0, "utility": 0.0}, {"steps": 10.0, "utility": 0.01}, {"steps": 100.0, "utility": 0.005}]
    assert L.fixed_hull_value(fixed, 10) == pytest.approx(0.01)
    assert L.fixed_hull_value(fixed, 5) == pytest.approx(0.005)
    assert L.fixed_hull_value(fixed, 55) == pytest.approx(0.01)


def stats(mean, low, loso_min=0.001, top1=0.2):
    return {"mean": mean, "ci95": [low, mean + 0.001], "loso": {"a": loso_min, "b": mean},
            "positive_top1_contribution": top1}


def test_gate_branches():
    assert L.gate0(L.quality_checks(stats(0.002, 0.0005)), {"x": False}) == "QUALITY_HEADROOM_SUPPORTED"
    weak = L.quality_checks(stats(0.0008, 0.0001))
    assert not weak["mean_at_least_delta_min"]
    assert not L.quality_checks(stats(0.002, 0.0005, top1=0.6))["positive_top1_share_at_most_0_5"]
    assert not L.quality_checks(stats(0.002, 0.0005, loso_min=-1e-4))["every_leave_one_place_out_mean_positive"]
    loss = {"mean": 0.0001, "ci95": [-0.0001, 0.0004]}
    assert L.gate0(weak, L.compute_checks("FULL", 0.8, loss, {"OFF": 0.003, "SHORT": 0.002})) == \
        "COMPUTE_HEADROOM_SUPPORTED"
    assert L.gate0(weak, L.compute_checks("FULL", 0.8, loss, {"OFF": 0.003, "SHORT": 0.0002})) == \
        "NOT_ESTABLISHED_FIXED_RULE_SUFFICES"
    assert L.gate0(weak, L.compute_checks("FULL", 0.3, loss, {"OFF": 0.003, "SHORT": 0.002})) == "NOT_ESTABLISHED"
    assert L.gate0(weak, L.compute_checks("OFF", None, None, {})) == "NOT_ESTABLISHED"


def test_access_audit_flags_files_outside_the_arrived_frames(tmp_path):
    audit = L.AccessAudit()
    audit.begin([tmp_path / "rgb_0.png"])
    audit.note(tmp_path / "rgb_0.png")
    assert audit.violations() == []
    audit.note(str(tmp_path / "depth_14.npy"))
    assert audit.violations() == [str((tmp_path / "depth_14.npy").resolve())]


def test_root_lock_keeps_targets_out_of_the_state():
    path = ROOT / "frame_role_lock.json"
    if not path.is_file():
        pytest.skip("the root is not built yet")
    cohorts = json.loads(path.read_text())["cohorts"]
    for cohort in cohorts.values():
        for e in cohort["episodes"]:
            assert not (set(e["context"]) | set(e["stream"])) & (set(e["queries"]) | set(e["history"]))
    places = {n: {e["place"] for e in c["episodes"]} for n, c in cohorts.items()}
    assert not places["CAJU_SEL"] & places["CAJU_EVAL"]
    assert not places["CAJU_PILOT"] & (places["CAJU_SEL"] | places["CAJU_EVAL"])
    assert len(cohorts["CAJU_SEL"]["episodes"]) + len(cohorts["CAJU_EVAL"]["episodes"]) == 100


def test_evaluation_refuses_without_a_logged_selection(tmp_path):
    import caju_score

    with pytest.raises(PermissionError):
        caju_score.check_selection(tmp_path)
    (tmp_path / "fixed_action_selection.json").write_text("{}")
    (tmp_path / "audit").mkdir()
    (tmp_path / "audit" / "selection_log.txt").write_text("2026-10-14 0000\n")
    with pytest.raises(PermissionError):
        caju_score.check_selection(tmp_path)


def test_lock_refuses_after_any_output(tmp_path):
    import caju_lock

    (tmp_path / "replay").mkdir()
    with pytest.raises(SystemExit):
        caju_lock.guard(tmp_path)


def test_place_bootstrap_is_deterministic():
    values = {f"p{i}": i / 1000 for i in range(20)}
    a = describe(values, draws=L.DRAWS, seed=L.BOOT_SEED, positive_means_improvement=True)
    b = describe(values, draws=L.DRAWS, seed=L.BOOT_SEED, positive_means_improvement=True)
    assert a["ci95"] == b["ci95"] and a["n_scenes"] == 20
