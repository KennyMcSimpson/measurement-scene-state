"""EVAL-FAR frame rule: far query slots never reuse prepared frames and follow the locked order."""

import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "prepare_rgbd_eval_far_data.py"
spec = importlib.util.spec_from_file_location("prepare_rgbd_eval_far_data", SCRIPT)
far = importlib.util.module_from_spec(spec)
spec.loader.exec_module(far)


def test_slots_start_at_the_registered_positions_and_skip_prepared_frames():
    allowed = list(range(100))
    slots = far.far_slots(allowed, prepared=set(range(16)))
    assert slots == [[32, 33, 34, 35], [48, 49, 50, 51]]
    slots = far.far_slots(allowed, prepared=set(range(16)) | {32, 48})
    assert slots == [[33, 34, 35, 36], [49, 50, 51, 52]]


def test_slots_follow_the_sorted_official_list_and_run_out_on_short_trajectories():
    allowed = [5 * i for i in range(60)]  # sparse official frame ids, unsorted input
    slots = far.far_slots(list(reversed(allowed)), prepared={0, 5, 10})
    assert slots[0][0] == 160 and slots[1][0] == 240
    assert far.far_slots(list(range(40)), prepared=set(range(16)))[1] == []
