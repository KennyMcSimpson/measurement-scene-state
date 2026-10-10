#!/usr/bin/env python3
"""Extend the ARKitScenes Validation evaluation cohort (EXP-3D-RGBD-ARKIT-VAL-DATA-V1) along the locked venue order.

The base run of scripts/prepare_rgbd_arkit_data.py (--folds Validation) prepared the 70 selected Validation venues;
41 passed the locked validity rules (29 failed the source-geometry rule). This script continues the same seeded
order: it recomputes the selection rule (numpy default_rng 20261005, the Training permutation first, then the
Validation one; the first 70 must equal selection.json), takes every remaining Validation venue in that order (the
scan with the smallest video id per venue), downloads them, locks their camera metadata (the base script's
lock_scan, before any image or depth byte is read), then prepares them in order with the base script's
prepare_scan and the same rules until the cohort holds TARGET eligible scans (the 41 base scans first). No model is
run and no score is computed. Stages (repository root, through cv_host_run.py, CUDA hidden):
  python prepare_rgbd_arkit_val_cohort.py select | download | lock | prepare
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

ROOT = Path("outputs/EXP-3D-RGBD-ARKIT-VAL-DATA-V1").resolve()
BASE = Path("scripts/prepare_rgbd_arkit_data.py")
TARGET = 80


def base():
    spec = importlib.util.spec_from_file_location("arkit_base", BASE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def select(b):
    if (ROOT / "selection_extension.json").exists():
        raise FileExistsError("extension selection exists; never re-select")
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
    order = {}
    for fold in b.VENUES:  # same draws, same order as the base select stage
        order[fold] = [folds[fold][int(i)] for i in rng.permutation(len(folds[fold]))]
    selected = json.loads((ROOT / "selection.json").read_text())["selected"]
    for fold, count in b.VENUES.items():
        if [s["visit_id"] for s in selected[fold]] != order[fold][:count]:
            raise PermissionError(f"recomputed order differs from selection.json ({fold})")
    extension = [{"visit_id": v, "video_id": min(venues[("Validation", v)], key=int)}
                 for v in order["Validation"][b.VENUES["Validation"]:]]
    b.write(ROOT / "selection_extension.json", {
        "rule": "the base selection rule continued: Validation venues after the first 70 in the same seeded "
                "order (numpy default_rng 20261005: Training permutation, then Validation permutation); per venue "
                "the scan with the smallest video id",
        "base_selection_sha256": sha(ROOT / "selection.json"), "splits_csv_sha256": sha(b.SPLITS_CSV),
        "prefix_check": "PASS", "target_eligible": TARGET, "extension": extension,
        "image_or_depth_read": False, "created_utc": datetime.datetime.now(datetime.UTC).isoformat()})
    print("EXTENSION_SELECTED", len(extension))


def download(b):
    extension = json.loads((ROOT / "selection_extension.json").read_text())["extension"]
    record = {}
    for scan in extension:
        video = scan["video_id"]
        target = b.DATA / "zips" / "Validation" / f"{video}.zip"
        if not target.is_file():
            part = target.with_suffix(".part")
            subprocess.run(["curl", "-sS", "--fail", "--retry", "3", "-o", str(part),
                            f"{b.URL}/Validation/{video}.zip"], check=True)
            part.rename(target)
        if not b.frames_dir("Validation", video).is_dir():
            (b.DATA / "raw" / "Validation").mkdir(parents=True, exist_ok=True)
            subprocess.run(["unzip", "-q", "-o", str(target), f"{video}/{video}_frames/*", "-d",
                            str(b.DATA / "raw" / "Validation")], check=True)
        record[video] = {"bytes": target.stat().st_size, "sha256": sha(target)}
    b.write(ROOT / "downloads_extension.json", {"url": b.URL, "scans": record,
                                                "total_bytes": sum(v["bytes"] for v in record.values())})
    print("EXTENSION_DOWNLOADED", len(record))


def lock(b):
    if (ROOT / "candidate_lock_extension.json").exists():
        raise FileExistsError("extension lock exists; never re-lock")
    rules = b.replica()
    scans, failures = {}, {}
    for scan in json.loads((ROOT / "selection_extension.json").read_text())["extension"]:
        video = scan["video_id"]
        try:
            scans[video] = {"fold": "Validation", "visit_id": scan["visit_id"], **b.lock_scan(rules, "Validation", video)}
        except ValueError as error:
            failures[video] = {"fold": "Validation", "status": "INELIGIBLE_CAMERA_METADATA", "error": str(error)}
    base_lock = json.loads((ROOT / "candidate_lock.json").read_text())
    b.write(ROOT / "candidate_lock_extension.json", {
        "rules": "identical to candidate_lock.json (frame_rule, camera_model, validity_rule)",
        "base_candidate_lock_sha256": sha(ROOT / "candidate_lock.json"),
        "frame_rule": base_lock["frame_rule"], "camera_model": base_lock["camera_model"],
        "validity_rule": base_lock["validity_rule"],
        "selection_extension_sha256": sha(ROOT / "selection_extension.json"),
        "downloads_extension_sha256": sha(ROOT / "downloads_extension.json"),
        "scans": scans, "failures": failures, "image_or_depth_read": False, "model_forward_run": False,
        "source_sha256": {str(p): sha(p) for p in (Path(__file__).resolve(), BASE.resolve(), b.REPLICA.resolve())},
        "created_utc": datetime.datetime.now(datetime.UTC).isoformat()})
    print("EXTENSION_LOCKED", len(scans), "camera failures", len(failures))


def prepare(b):
    if (ROOT / "manifest_arkit_val80.json").exists():
        raise FileExistsError("cohort manifest exists; no overwrite")
    lock_ = json.loads((ROOT / "candidate_lock_extension.json").read_text())
    for path, digest in lock_["source_sha256"].items():
        if sha(path) != digest:
            raise PermissionError(f"locked source changed: {path}")
    base_manifest = json.loads((ROOT / "manifest_arkit_val.json").read_text())
    records = list(base_manifest["scenes"])
    rules = b.replica()
    roles_of, prior = rules.role_rule(), rules.frozen_prior()
    qualities, geometry, failures, hashes, supports = {}, {}, dict(lock_["failures"]), {}, {}
    for scan in json.loads((ROOT / "selection_extension.json").read_text())["extension"]:
        video = scan["video_id"]
        if len(records) >= TARGET:
            break
        if video not in lock_["scans"]:
            continue
        record, support, quality, check, failure = b.prepare_scan(
            rules, roles_of, prior, video, lock_["scans"][video], b.DATA / "prepared" / video)
        qualities[video], geometry[video] = {str(k): v for k, v in quality.items()}, check
        if failure:
            failures[video] = {"fold": "Validation", **failure}
            continue
        supports[video] = support
        for frame in record["frames"]:
            for kind in ("rgb", "depth"):
                hashes[frame[kind]] = sha(frame[kind])
        record["split"] = "ARKIT_VAL"
        records.append(record)
    b.write(ROOT / "manifest_arkit_val80.json", {
        "schema": "mcss.rgbd_arkit.data.v1", "image_size": base_manifest["image_size"],
        "depth_semantics": base_manifest["depth_semantics"], "protected_scene_ids": [],
        "candidate_lock_sha256": sha(ROOT / "candidate_lock.json"),
        "candidate_lock_extension_sha256": sha(ROOT / "candidate_lock_extension.json"),
        "base_manifest_sha256": sha(ROOT / "manifest_arkit_val.json"), "scenes": records})
    for name, value in (("frame_quality_extension.json", qualities), ("geometry_checks_extension.json", geometry),
                        ("role_support_extension.json", supports), ("scene_failures_extension.json", failures),
                        ("hashes_extension.json", hashes)):
        b.write(ROOT / name, value)
    status = "PASS" if len(records) >= TARGET else "SHORT"
    b.write(ROOT / "preparation_integrity_val80.json", {
        "status": status, "eligible": len(records), "target": TARGET, "base_eligible": len(base_manifest["scenes"]),
        "extension_failures": len(failures), "model_forward_run": False, "score_computed": False})
    print(status, len(records), "eligible;", len(failures), "extension failures")


if __name__ == "__main__":
    {"select": select, "download": download, "lock": lock, "prepare": prepare}[sys.argv[1]](base())
