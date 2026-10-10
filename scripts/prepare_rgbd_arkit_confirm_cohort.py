#!/usr/bin/env python3
"""ARKitScenes confirmation cohort (EXP-3D-RGBD-ARKIT-CONFIRM-DATA-V1): fresh real scans never used before.

Training-fold venues AFTER the pilot's 360 (EXP-3D-RGBD-ARKIT-DATA-V1 reserves the first 360 in its seeded order for
DEV/TRAIN), in the same seeded order (numpy default_rng 20261005: the Training permutation is the first draw); per
venue the scan with the smallest video id; a fixed block of BLOCK venues is downloaded and its camera metadata
locked (the base script's lock_scan, before any image or depth byte is read); the block is then prepared in order
with the base script's prepare_scan and rules until TARGET scans are eligible. No model is run and no score is
computed. Stages (repository root, through cv_host_run.py, CUDA hidden):
  python prepare_rgbd_arkit_confirm_cohort.py select | download | lock | prepare
"""

from __future__ import annotations

import csv
import datetime
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path("outputs/EXP-3D-RGBD-ARKIT-CONFIRM-DATA-V1").resolve()
PILOT = Path("outputs/EXP-3D-RGBD-ARKIT-DATA-V1/selection.json")
BASE = Path("scripts/prepare_rgbd_arkit_data.py")
START, BLOCK, TARGET = 360, 160, 80
SPLIT = "ARKIT_CONFIRM"


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
    rows = list(csv.DictReader(b.SPLITS_CSV.open()))
    venues = {}
    for row in rows:
        if row["visit_id"] in ("", "NA"):
            continue
        venues.setdefault((row["fold"], row["visit_id"]), []).append(row["video_id"])
    folds = {fold: sorted(v for f, v in venues if f == fold) for fold in b.VENUES}
    shared = sorted(set(folds["Training"]) & set(folds["Validation"]))
    folds = {fold: [v for v in visits if v not in shared] for fold, visits in folds.items()}
    rng = np.random.default_rng(b.SEED)
    order = [folds["Training"][int(i)] for i in rng.permutation(len(folds["Training"]))]
    pilot = json.loads(PILOT.read_text())["selected"]["Training"]
    if [s["visit_id"] for s in pilot] != order[:START]:
        raise PermissionError("recomputed Training order differs from the pilot selection")
    block = [{"visit_id": v, "video_id": min(venues[("Training", v)], key=int)} for v in order[START:START + BLOCK]]
    b.write(ROOT / "selection.json", {
        "rule": f"Training-fold venues at positions {START}..{START + BLOCK - 1} of the pilot's seeded order (numpy "
                "default_rng 20261005, Training permutation); the first 360 are reserved by the pilot; per venue the "
                "scan with the smallest video id",
        "pilot_selection_sha256": sha(PILOT), "splits_csv_sha256": sha(b.SPLITS_CSV), "prefix_check": "PASS",
        "target_eligible": TARGET, "block": block, "image_or_depth_read": False,
        "created_utc": datetime.datetime.now(datetime.UTC).isoformat()})
    print("SELECTED", len(block))


def download(b):
    record = {}
    for scan in json.loads((ROOT / "selection.json").read_text())["block"]:
        video = scan["video_id"]
        target = b.DATA / "zips" / "Training" / f"{video}.zip"
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.is_file():
            part = target.with_suffix(".part")
            subprocess.run(["curl", "-sS", "--fail", "--retry", "3", "-o", str(part),
                            f"{b.URL}/Training/{video}.zip"], check=True)
            part.rename(target)
        if not b.frames_dir("Training", video).is_dir():
            (b.DATA / "raw" / "Training").mkdir(parents=True, exist_ok=True)
            subprocess.run(["unzip", "-q", "-o", str(target), f"{video}/{video}_frames/*", "-d",
                            str(b.DATA / "raw" / "Training")], check=True)
        record[video] = {"bytes": target.stat().st_size, "sha256": sha(target)}
    b.write(ROOT / "downloads.json", {"url": b.URL, "scans": record,
                                      "total_bytes": sum(v["bytes"] for v in record.values())})
    print("DOWNLOADED", len(record))


def lock(b):
    if (ROOT / "candidate_lock.json").exists():
        raise FileExistsError("lock exists; never re-lock")
    rules = b.replica()
    scans, failures = {}, {}
    for scan in json.loads((ROOT / "selection.json").read_text())["block"]:
        video = scan["video_id"]
        try:
            scans[video] = {"fold": "Training", "visit_id": scan["visit_id"], **b.lock_scan(rules, "Training", video)}
        except ValueError as error:
            failures[video] = {"fold": "Training", "status": "INELIGIBLE_CAMERA_METADATA", "error": str(error)}
    ref = json.loads(Path("outputs/EXP-3D-RGBD-ARKIT-VAL-DATA-V1/candidate_lock.json").read_text())
    b.write(ROOT / "candidate_lock.json", {
        "rules": "identical to EXP-3D-RGBD-ARKIT-VAL-DATA-V1/candidate_lock.json",
        "frame_rule": ref["frame_rule"], "camera_model": ref["camera_model"], "validity_rule": ref["validity_rule"],
        "selection_sha256": sha(ROOT / "selection.json"), "downloads_sha256": sha(ROOT / "downloads.json"),
        "scans": scans, "failures": failures, "image_or_depth_read": False, "model_forward_run": False,
        "source_sha256": {str(p): sha(p) for p in (Path(__file__).resolve(), BASE.resolve(), b.REPLICA.resolve())},
        "created_utc": datetime.datetime.now(datetime.UTC).isoformat()})
    print("LOCKED", len(scans), "camera failures", len(failures))


def prepare(b):
    if (ROOT / "manifest_arkit_confirm.json").exists():
        raise FileExistsError("manifest exists; no overwrite")
    lock_ = json.loads((ROOT / "candidate_lock.json").read_text())
    for path, digest in lock_["source_sha256"].items():
        if sha(path) != digest:
            raise PermissionError(f"locked source changed: {path}")
    rules = b.replica()
    roles_of, prior = rules.role_rule(), rules.frozen_prior()
    records, qualities, geometry, failures, hashes, supports = [], {}, {}, dict(lock_["failures"]), {}, {}
    for scan in json.loads((ROOT / "selection.json").read_text())["block"]:
        video = scan["video_id"]
        if len(records) >= TARGET:
            break
        if video not in lock_["scans"]:
            continue
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
    b.write(ROOT / "manifest_arkit_confirm.json", {
        "schema": "mcss.rgbd_arkit.data.v1", "image_size": list(b.TARGET_SIZE), "depth_semantics": "ray_distance_meters",
        "protected_scene_ids": [], "candidate_lock_sha256": sha(ROOT / "candidate_lock.json"), "scenes": records})
    for name, value in (("frame_quality.json", qualities), ("geometry_checks.json", geometry),
                        ("role_support.json", supports), ("scene_failures.json", failures), ("hashes.json", hashes)):
        b.write(ROOT / name, value)
    status = "PASS" if len(records) >= TARGET else "SHORT"
    b.write(ROOT / "preparation_integrity.json", {"status": status, "eligible": len(records), "target": TARGET,
                                                  "failures": len(failures), "model_forward_run": False,
                                                  "score_computed": False})
    print(status, len(records), "eligible;", len(failures), "failures")


if __name__ == "__main__":
    {"select": select, "download": download, "lock": lock, "prepare": prepare}[sys.argv[1]](base())
