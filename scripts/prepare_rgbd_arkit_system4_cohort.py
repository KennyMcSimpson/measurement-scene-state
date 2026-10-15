#!/usr/bin/env python3
"""ARKitScenes fourth system cohort (EXP-3D-RGBD-ARKIT-SYSTEM4-DATA-V1): the never-prepared tail of the SYSTEM3 block.

EXP-3D-RGBD-ARKIT-SYSTEM3-DATA-V1 downloaded and camera-locked the 360 venues at positions 1000-1359 of the pilot's
seeded Training order and prepared them in order until 160 were eligible (block position 268). The 91 scans at block
positions 269-359 were downloaded and camera-locked but never prepared, and no image or depth byte of them was read.
This cohort prepares exactly those, in block order, with the base script's unchanged prepare_scan and SYSTEM3's
candidate lock (its locked source hashes are checked); every eligible scan is kept (no target). No download, no model
is run and no score is computed. Planned in EXP-3D-CAJU3D-SYSTEM4-HOLDOUT-V1 (plan locked before this script ran).
Stages (repository root, through cv_host_run.py, CUDA hidden):
  python prepare_rgbd_arkit_system4_cohort.py select | prepare
"""

from __future__ import annotations

import datetime
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path("outputs/EXP-3D-RGBD-ARKIT-SYSTEM4-DATA-V1").resolve()
SOURCE = Path("outputs/EXP-3D-RGBD-ARKIT-SYSTEM3-DATA-V1").resolve()
BASE = Path("scripts/prepare_rgbd_arkit_data.py")
SPLIT = "ARKIT_SYSTEM4"


def base():
    spec = importlib.util.spec_from_file_location("arkit_base", BASE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def select(b):
    ROOT.mkdir(parents=True, exist_ok=True)
    if (ROOT / "selection.json").exists():
        raise FileExistsError("selection exists; never re-select")
    integrity = json.loads((SOURCE / "preparation_integrity.json").read_text())
    if integrity["status"] != "PASS" or integrity["eligible"] != integrity["target"]:
        raise PermissionError("SYSTEM3 preparation did not stop at its target")
    block = json.loads((SOURCE / "selection.json").read_text())["block"]
    prepared = {r["scene_id"].removeprefix("arkit_")
                for r in json.loads((SOURCE / "manifest_arkit_system3.json").read_text())["scenes"]}
    failures = set(json.loads((SOURCE / "scene_failures.json").read_text()))
    done = [i for i, s in enumerate(block) if s["video_id"] in prepared | failures]
    first = max(done) + 1
    if done != list(range(first)):
        raise PermissionError("the processed SYSTEM3 scans are not a prefix of its block")
    tail = block[first:]
    locked = json.loads((SOURCE / "candidate_lock.json").read_text())["scans"]
    if any(s["video_id"] not in locked for s in tail):
        raise PermissionError("a tail scan has no camera lock")
    for s in tail:
        if (b.DATA / "prepared" / s["video_id"]).exists():
            raise PermissionError(f"{s['video_id']}: a prepared directory exists already")
    b.write(ROOT / "selection.json", {
        "rule": f"the never-prepared tail of EXP-3D-RGBD-ARKIT-SYSTEM3-DATA-V1's block: positions {first}.."
                f"{len(block) - 1} of that block (Training order positions {1000 + first}..1359), in block order",
        "first_block_position": first, "source_selection_sha256": sha(SOURCE / "selection.json"),
        "source_manifest_sha256": sha(SOURCE / "manifest_arkit_system3.json"),
        "source_failures_sha256": sha(SOURCE / "scene_failures.json"),
        "source_candidate_lock_sha256": sha(SOURCE / "candidate_lock.json"),
        "block": tail, "image_or_depth_read": False,
        "created_utc": datetime.datetime.now(datetime.UTC).isoformat()})
    print("SELECTED", len(tail), "first block position", first)


def prepare(b):
    if (ROOT / "manifest_arkit_system4.json").exists():
        raise FileExistsError("manifest exists; no overwrite")
    selection = json.loads((ROOT / "selection.json").read_text())
    if sha(SOURCE / "candidate_lock.json") != selection["source_candidate_lock_sha256"]:
        raise PermissionError("SYSTEM3 candidate lock changed since the selection")
    lock_ = json.loads((SOURCE / "candidate_lock.json").read_text())
    for path, digest in lock_["source_sha256"].items():
        if sha(path) != digest:
            raise PermissionError(f"locked source changed: {path}")
    rules = b.replica()
    roles_of, prior = rules.role_rule(), rules.frozen_prior()
    records, qualities, geometry, hashes, supports, failures = [], {}, {}, {}, {}, {}
    for scan in selection["block"]:
        video = scan["video_id"]
        record, support, quality, check, failure = b.prepare_scan(
            rules, roles_of, prior, video, lock_["scans"][video], b.DATA / "prepared" / video)
        qualities[video], geometry[video] = {str(k): v for k, v in quality.items()}, check
        if failure:
            failures[video] = {"fold": "Training", **failure}
            continue
        supports[video] = support
        for frame in record["frames"]:
            for kind in ("rgb", "depth"):
                hashes[frame[kind]] = sha(frame[kind])
        record["split"] = SPLIT
        records.append(record)
    b.write(ROOT / "manifest_arkit_system4.json", {
        "schema": "mcss.rgbd_arkit.data.v1", "image_size": list(b.TARGET_SIZE),
        "depth_semantics": "ray_distance_meters", "protected_scene_ids": [],
        "candidate_lock_sha256": sha(SOURCE / "candidate_lock.json"), "selection_sha256": sha(ROOT / "selection.json"),
        "scenes": records})
    for name, value in (("frame_quality.json", qualities), ("geometry_checks.json", geometry),
                        ("role_support.json", supports), ("scene_failures.json", failures),
                        ("hashes.json", hashes)):
        b.write(ROOT / name, value)
    b.write(ROOT / "preparation_integrity.json", {
        "status": "PASS" if records else "EMPTY", "eligible": len(records), "processed": len(selection["block"]),
        "failures": len(failures), "model_forward_run": False, "score_computed": False})
    print("PREPARED", len(records), "eligible;", len(failures), "failures")


if __name__ == "__main__":
    {"select": select, "prepare": prepare}[sys.argv[1]](base())
