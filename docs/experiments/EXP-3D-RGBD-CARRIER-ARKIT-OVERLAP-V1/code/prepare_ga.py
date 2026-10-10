#!/usr/bin/env python3
"""EXP-3D-RGBD-CARRIER-ARKIT-OVERLAP-V1: freeze the GA training root (the frozen V17 L code, ARKit TRAIN, overlap
targets) before any GA training and before any EXP-3D-RGBD-DEPTHSPLAT-JEPA-ARKIT-OVERLAP-V1 result.

As in the V21 freeze, the frozen V17 prepare makes a V17-style root; then:
  - every ARKit TRAIN record of EXP-3D-RGBD-ARKIT-DATA-V1 gets primary_query = query = the frames in neither context
    whose context hit share (OVERLAP-V1 hit table, computed without target depth) is >= 0.05 for both roles
    (secondary_query empty); when there are more than four such frames, four evenly spaced ones by position in the
    sorted list (first and last included), so that the frozen extra-view rule always finds its four free frames
    (the first freeze, without this cap, left out the 40 records with the most overlap; it was moved aside to
    first_freeze_without_cap/ before any training); records with fewer than two such frames are left out;
  - these records are merged into the manifest, the frame-role lock (roles and extra-view frames recomputed by the
    frozen stream_frame_ids on the modified record) and the data hashes; the variant's train set, which the frozen
    code reads under the key TRAIN72, becomes these records only;
  - device cpu (AVX2, one thread, as V17); the V11-prefix control does not apply (other training data);
  - the root's own lock re-registers the rewritten inputs and the added data.
Run from the repository root under cv_host_run.py (records store zonghan_o paths), CUDA hidden:
  python prepare_ga.py PYTEST_LOG
"""

from __future__ import annotations

import datetime
import hashlib
import importlib.util
import json
import re
import shutil
import sys
from pathlib import Path

from mcss.mechanism_pilot.geometry_carrier_experiment import validate_split, verify_lock
from mcss.mechanism_pilot.rgbd_stream_write import stream_frame_ids

CARRIER = Path("outputs/EXP-3D-RGBD-CARRIER-ARKIT-OVERLAP-V1")
V17 = Path("outputs/EXP-3D-RGBD-LONG-TRAIN-V17")
V17_NAME = "EXP-3D-RGBD-LONG-TRAIN-V17"
V17_PREREGISTRATION_SHA256 = "36b8aa6e4e527025d3746aa7af5ffc060fbf7683fc1a26ff50a94e3c744c1d1f"
PILOT = Path("outputs/EXP-3D-RGBD-ARKIT-DATA-V1")
OVERLAP = Path("outputs/EXP-3D-RGBD-DEPTHSPLAT-JEPA-ARKIT-OVERLAP-V1")
MIN_HIT = 0.05
MAX_QUERIES = 4
FIRST_FREEZE = CARRIER / "first_freeze_without_cap"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def overlap_records():
    """ARKit TRAIN records with overlap targets; the hit table is the one OVERLAP-V1 locked."""
    table_path = OVERLAP / "training/hit_shares_TRAIN.json"
    if sha(table_path) != read(OVERLAP / "preregistration.json")["root_files"]["training/hit_shares_TRAIN.json"]:
        raise PermissionError("hit table differs from the one OVERLAP-V1 locked")
    table = read(table_path)
    manifest = read(PILOT / "manifest_arkit_trainval.json")
    kept, left_out = [], {}
    for r in manifest["scenes"]:
        if r["split"] != "TRAIN":
            continue
        roles = r["roles"]
        context = set(roles["context_a"]) | set(roles["context_b"])
        hits = table[r["scene_id"]]
        eligible = [f["frame_id"] for f in r["frames"] if f["frame_id"] not in context
                    and hits["A"][str(f["frame_id"])] >= MIN_HIT and hits["B"][str(f["frame_id"])] >= MIN_HIT]
        if len(eligible) < 2:
            left_out[r["scene_id"]] = f"{len(eligible)} overlap frames"
            continue
        if len(eligible) > MAX_QUERIES:
            positions = [round(i * (len(eligible) - 1) / (MAX_QUERIES - 1)) for i in range(MAX_QUERIES)]
            eligible = [eligible[i] for i in positions]
        record = dict(r, roles=dict(roles, primary_query=eligible, query=eligible, secondary_query=[]))
        extra = stream_frame_ids(record, "A")
        if extra is None or len(extra) != 4:
            left_out[r["scene_id"]] = "no four free frames for the extra-view rule"
            continue
        kept.append((record, list(extra)))
    return kept, left_out, sha(table_path)


def pytest_summary(log):
    text = Path(log).read_text()
    last = [line for line in text.splitlines() if line.strip()][-1]
    counts = {k: int(v) for v, k in re.findall(r"(\d+) (passed|failed|skipped|errors?|xfailed|xpassed)", last)}
    status = "PASS" if counts.get("passed", 0) > 0 and not counts.get("failed") and not counts.get("error") \
        and not counts.get("errors") else "FAIL"
    return {"status": status, "summary_line": last, "counts": counts, "log_path": str(log), "log_sha256": sha(log)}


def main():
    log = Path(sys.argv[1])
    root = (CARRIER / "GA" / V17_NAME).resolve()
    if root.exists():
        raise FileExistsError(root)
    if not (CARRIER / "PROTOCOL.md").is_file():
        raise PermissionError("the protocol must exist before the freeze")
    if (OVERLAP / "scores").exists() or (OVERLAP / "results").exists():
        raise PermissionError("an OVERLAP-V1 result exists: too late to freeze this design blind")
    if sha(V17 / "preregistration.json") != V17_PREREGISTRATION_SHA256:
        raise PermissionError("V17 is not the frozen V17")
    tests = pytest_summary(log)
    if tests["status"] != "PASS":
        raise PermissionError(f"full suite must PASS before the freeze: {tests['summary_line']}")
    kept, left_out, table_sha = overlap_records()
    root.mkdir(parents=True)
    for name in ("PLAN_BEFORE_V17_TRAINING.md", "PROTOCOL.md"):
        shutil.copyfile(V17 / name, root / name)
    module("v17_prepare", "scripts/prepare_rgbd_long_train_experiment.py").prepare(root)
    manifest = read(root / "scene_split.json")
    if {r["scene_id"] for r in manifest["scenes"]} & {r["scene_id"] for r, _ in kept}:
        raise PermissionError("ARKit ids collide with the manifest")
    manifest["scenes"] = manifest["scenes"] + [r for r, _ in kept]
    validate_split(manifest)
    write(root / "scene_split.json", manifest)
    roles = read(root / "frame_role_lock.json")
    for record, extra in kept:
        roles["roles"][record["scene_id"]] = record["roles"]
        roles["train_extra_view_frames"][record["scene_id"]] = extra
    roles["manifest_sha256"] = sha(root / "scene_split.json")
    roles["carrier_arkit_overlap"] = ("ARKit TRAIN records merged by the EXP-3D-RGBD-CARRIER-ARKIT-OVERLAP-V1 freeze; "
                                      f"queries = frames in neither context with hit share >= {MIN_HIT} for both roles")
    write(root / "frame_role_lock.json", roles)
    contract = read(root / "training_contract.json")
    contract["train_sets"]["TRAIN72"] = sorted(r["scene_id"] for r, _ in kept)
    contract["train_set_note"] = ("EXP-3D-RGBD-CARRIER-ARKIT-OVERLAP-V1 GA: the frozen V17 code reads the variant's train "
                                  "set under the key TRAIN72; here it holds only the ARKit TRAIN records with overlap "
                                  "targets")
    contract.update(device="cpu", carrier_arkit_overlap_arm="GA",
                    execution="three trainings (three seeds) concurrently on CPU (AVX2, one thread each) through "
                              "cv_host_run.py; evaluation by the arms evaluator after training")
    write(root / "training_contract.json", contract)
    hashes = read(root / "audit/data_hashes.json")
    added = {}
    for record, _ in kept:
        for frame in record["frames"]:
            for kind in ("rgb", "depth"):
                path = str(Path(frame[kind]))
                added[path] = hashes[path] = sha(path)
    write(root / "audit/data_hashes.json", hashes)
    write(root / "audit/v11_prefix_control.json", {
        "status": "PASS", "meaning": "NOT_APPLICABLE_OTHER_TRAINING_DATA",
        "note": "GA trains on ARKit records, so the V17 built-in control (training log lines 1-6000 equal V11 C1) "
                "cannot apply; this file only lets the frozen code run"})
    write(root / "tests.json", {**tests, "finished_utc": datetime.datetime.now(datetime.UTC).isoformat(),
                                "note": "full suite run with CUDA hidden immediately before the GA freeze"})
    lock = read(root / "preregistration.json")
    rewritten = ["training_contract.json", "scene_split.json", "frame_role_lock.json", "audit/data_hashes.json"]
    for name in rewritten:
        key = str((root / name).resolve())
        if key not in lock["input_sha256"]:
            raise KeyError(key)
        lock["input_sha256"][key] = sha(root / name)
    lock["data_sha256"].update(added)
    lock["carrier_arkit_overlap"] = {"arm": "GA", "rewritten": rewritten, "added_data": len(added),
                                     "train_records": len(kept), "left_out": left_out,
                                     "hit_table_sha256": table_sha, "min_hit": MIN_HIT}
    write(root / "preregistration.json", lock)
    manifest, config = verify_lock(root)
    query_counts = {}
    for record, _ in kept:
        n = len(record["roles"]["primary_query"])
        query_counts[n] = query_counts.get(n, 0) + 1
    write(CARRIER / "freeze.json", {
        "status": "FROZEN_BEFORE_GA_TRAINING",
        "frozen_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "protocol_sha256": sha(CARRIER / "PROTOCOL.md"),
        "ga_root": str(root), "ga_preregistration_sha256": sha(root / "preregistration.json"),
        "ga_training_contract_sha256": sha(root / "training_contract.json"),
        "train_records": len(kept), "left_out": left_out, "query_frames_per_record": query_counts,
        "hit_table_sha256": table_sha, "v17_preregistration_sha256": V17_PREREGISTRATION_SHA256,
        "tests": tests,
        "code_sha256": {p.name: sha(p) for p in sorted((CARRIER / "code").iterdir()) if p.is_file()},
        "overlap_v1_results_existed": False,
        "max_query_frames": MAX_QUERIES,
        "first_freeze": {"moved_to": str(FIRST_FREEZE), "trained": False,
                         "why_replaced": "without the four-frame cap the frozen extra-view rule left out the 40 "
                                         "records with the most overlap",
                         "freeze_sha256": sha(FIRST_FREEZE / "freeze.json")},
    })
    print("GA_FROZEN", root, "train records", len(config["train_sets"]["TRAIN72"]), "left out", len(left_out),
          "query frames per record", query_counts)


if __name__ == "__main__":
    main()
