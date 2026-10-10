#!/usr/bin/env python3
"""Freeze EXP-3D-RGBD-SEGMENT-DIVERSITY-V21: two GPU arms of the frozen V17 code (G1 and G5).

Both arms run the frozen V17 L training and EVAL code unchanged, each in its own V17-style root
outputs/EXP-3D-RGBD-SEGMENT-DIVERSITY-V21/<arm>/EXP-3D-RGBD-LONG-TRAIN-V17. The frozen V17 prepare
makes each root; then the V21 contract is applied and registered in the root's own lock:
  - both arms: device cuda on GPU 4 (launched through cv_gpu_run.sh), TF32 off. GPU training is
    not bit-reproducible, so there is no bit-identical control; the control file the frozen EVAL
    code requires says so (status PASS, meaning NOT_APPLICABLE_GPU);
  - G5: the SEG-V1 segment records are merged into the manifest, the frame-role lock and the data
    hashes, and the variant's train set, which the frozen code reads under the key TRAIN72, becomes
    TRAIN72 plus every valid SEG-V1 segment.
Then V21's freeze record (freeze.json) holds every hash, before any V21 training.
Run from the repository root under cv_host_run.py (records store zonghan_o paths).
"""

from __future__ import annotations

import datetime
import hashlib
import importlib.util
import json
import shutil
from pathlib import Path

from mcss.mechanism_pilot.geometry_carrier_experiment import verify_lock
from mcss.mechanism_pilot.rgbd_stream_write import stream_frame_ids

V21 = Path("outputs/EXP-3D-RGBD-SEGMENT-DIVERSITY-V21")
PLAN = V21 / "PLAN_BEFORE_V21_DATA.md"
PLAN_SHA256 = "f04e581e398ff1a47e14f3a0d46df2ebb0275a5d017b03ba3ce2fb26f74f2d24"
SEG = Path("outputs/EXP-3D-RGBD-SEGMENT-DATA-V1")
V17 = Path("outputs/EXP-3D-RGBD-LONG-TRAIN-V17")
V17_PREREGISTRATION_SHA256 = "36b8aa6e4e527025d3746aa7af5ffc060fbf7683fc1a26ff50a94e3c744c1d1f"
V17_NAME = "EXP-3D-RGBD-LONG-TRAIN-V17"
ARMS = ("G1", "G5")
GPU = "4"
SCRIPTS = (
    Path("scripts/prepare_rgbd_segment_diversity_v21.py"),
    Path("scripts/run_rgbd_segment_diversity_v21.sh"),
    Path("scripts/eval_rgbd_segment_diversity_v21.sh"),
    Path("scripts/analyze_rgbd_segment_diversity_v21.py"),
    Path("scripts/prepare_rgbd_segment_data.py"),
    Path.home() / "cv_tools/fp32_run.py",
    Path.home() / "cv_tools/cv_gpu_run.sh",
    Path.home() / "cv_tools/cv_host_run.py",
)


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


def relock(root, names, extra_data=None, note=None):
    """Re-register the rewritten inputs (and added data) in the root's own preregistration."""
    lock = read(root / "preregistration.json")
    for name in names:
        key = str((root / name).resolve())
        if key not in lock["input_sha256"]:
            raise KeyError(key)
        lock["input_sha256"][key] = sha(root / name)
    if extra_data:
        lock["data_sha256"].update(extra_data)
    lock["v21"] = note
    write(root / "preregistration.json", lock)


def seg_records():
    integrity = read(SEG / "preparation_integrity.json")
    manifest = read(SEG / "manifest_seg.json")
    if integrity["status"] != "PASS" or integrity["candidate_lock_sha256"] != sha(
        SEG / "candidate_lock.json"
    ):
        raise PermissionError("SEG-V1 must PASS against its frozen lock")
    if manifest["candidate_lock_sha256"] != sha(SEG / "candidate_lock.json"):
        raise PermissionError("SEG-V1 manifest is not from its lock")
    if read(SEG / "candidate_lock.json")["plan_sha256"] != PLAN_SHA256:
        raise PermissionError("SEG-V1 was not locked under the V21 plan")
    return manifest["scenes"]


def make_arm(arm, prepare_v17, stream_frame_ids, segments):
    root = (V21 / arm / V17_NAME).resolve()
    if root.exists():
        raise FileExistsError(root)
    root.mkdir(parents=True)
    for name in ("PLAN_BEFORE_V17_TRAINING.md", "PROTOCOL.md"):
        shutil.copyfile(V17 / name, root / name)
    prepare_v17(root)
    contract = read(root / "training_contract.json")
    rewritten = ["training_contract.json"]
    added = {}
    if arm == "G5":
        manifest = read(root / "scene_split.json")
        ids = {r["scene_id"] for r in manifest["scenes"]}
        if ids & {r["scene_id"] for r in segments}:
            raise PermissionError("Segment ids collide with the manifest")
        manifest["scenes"] = manifest["scenes"] + segments
        write(root / "scene_split.json", manifest)
        roles = read(root / "frame_role_lock.json")
        for record in segments:
            roles["roles"][record["scene_id"]] = record["roles"]
            extra = stream_frame_ids(record, "A")
            if extra is None or len(extra) != 4:
                raise PermissionError(f"Segment without 4 free frames: {record['scene_id']}")
            roles["train_extra_view_frames"][record["scene_id"]] = list(extra)
        roles["manifest_sha256"] = sha(root / "scene_split.json")
        roles["v21_segments"] = "SEG-V1 records merged by the V21 freeze"
        write(root / "frame_role_lock.json", roles)
        contract["train_sets"]["TRAIN72"] = sorted(
            contract["train_sets"]["TRAIN72"] + [r["scene_id"] for r in segments]
        )
        contract["train_set_note"] = (
            "V21 G5: the frozen V17 code reads the variant's train set under the key TRAIN72; "
            "here it holds TRAIN72 plus every valid SEG-V1 segment"
        )
        hashes = read(root / "audit/data_hashes.json")
        for record in segments:
            for frame in record["frames"]:
                for kind in ("rgb", "depth"):
                    path = str(Path(frame[kind]))
                    added[path] = hashes[path] = sha(path)
        write(root / "audit/data_hashes.json", hashes)
        rewritten += ["scene_split.json", "frame_role_lock.json", "audit/data_hashes.json"]
    contract.update(
        device="cuda",
        gpu=GPU,
        tf32="off (NVIDIA_TF32_OVERRIDE=0 and the fp32 launcher)",
        v21_arm=arm,
        execution="six trainings (two arms x three seeds) concurrently on GPU 4 through "
        "cv_gpu_run.sh; EVAL on CPU, AVX2",
    )
    write(root / "training_contract.json", contract)
    write(
        root / "audit/v11_prefix_control.json",
        {
            "status": "PASS",
            "meaning": "NOT_APPLICABLE_GPU",
            "note": "V21 GPU arm: GPU training is not bit-reproducible (HIT-REPRO-V1), so the V17 "
            "built-in control cannot apply; this file only lets the frozen V17 EVAL code run",
        },
    )
    write(root / "tests.json", read(V21 / "tests.json"))
    note = {
        "arm": arm,
        "plan_sha256": PLAN_SHA256,
        "rewritten": rewritten,
        "added_data": len(added),
    }
    relock(root, rewritten, added, note)
    return root


def main():
    if sha(PLAN) != PLAN_SHA256:
        raise PermissionError("The V21 plan changed")
    if sha(V17 / "preregistration.json") != V17_PREREGISTRATION_SHA256:
        raise PermissionError("V17 is not the frozen V17")
    if read(V21 / "tests.json")["status"] != "PASS":
        raise PermissionError("The full suite must PASS before the V21 freeze")
    if (V21 / "freeze.json").exists():
        raise FileExistsError("V21 is frozen once")
    segments = seg_records()
    v17_prepare = module("v17_prepare", "scripts/prepare_rgbd_long_train_experiment.py")
    roots = {arm: make_arm(arm, v17_prepare.prepare, stream_frame_ids, segments) for arm in ARMS}
    for arm, root in roots.items():
        manifest, config = verify_lock(root)
        names = set(config["train_sets"]["TRAIN72"])
        print(arm, "train records", len(names), "manifest records", len(manifest["scenes"]))
    write(
        V21 / "freeze.json",
        {
            "status": "FROZEN_BEFORE_V21_TRAINING",
            "frozen_utc": datetime.datetime.now(datetime.UTC).isoformat(),
            "plan_sha256": PLAN_SHA256,
            "seg_v1": {
                "candidate_lock_sha256": sha(SEG / "candidate_lock.json"),
                "manifest_sha256": sha(SEG / "manifest_seg.json"),
                "integrity_sha256": sha(SEG / "preparation_integrity.json"),
                "segments": len(segments),
                "scenes_with_segments": len({r["source_scene_id"] for r in segments}),
            },
            "arms": {
                arm: {
                    "root": str(root),
                    "preregistration_sha256": sha(root / "preregistration.json"),
                    "training_contract_sha256": sha(root / "training_contract.json"),
                }
                for arm, root in roots.items()
            },
            "v17_preregistration_sha256": V17_PREREGISTRATION_SHA256,
            "scripts_sha256": {str(p): sha(p) for p in SCRIPTS},
            "tests_sha256": sha(V21 / "tests.json"),
        },
    )
    print("V21_FROZEN", {a: str(r) for a, r in roots.items()}, "segments", len(segments))


if __name__ == "__main__":
    main()
