#!/usr/bin/env python3
"""Select, download, lock and materialize the ARKitScenes 3dod cohorts (no model, no score).

Source: ARKitScenes (Baruch et al. 2021), the 3dod subset (low-resolution RGB, LiDAR depth,
per-frame intrinsics and camera poses), downloaded per scan from Apple's public URL. Its use is
non-commercial research only; the user approved the download on 2026-10-05; the media stay outside
the git repository (`zonghan_o_mirror/_data_arkitscenes`).

`--stage select` reads only the official 3dod split table: one scan per venue (visit), venues in a
seeded order, separately for the Training and Validation folds; scans without a visit id are
excluded. `--stage download` fetches the selected zips and keeps only their frame folders.
`--stage lock` is camera metadata only (trajectories, intrinsics, file names): it freezes each
scan's 16 frames (frame spacing matched to the TRAIN72 Hypersim camera rotation, as for Replica),
the crop to the Hypersim field of view, the depth conversion and the validity rules before any
image or depth byte is read. `--stage prepare` applies them, assigns the uses (DEV, TRAIN,
ARKIT_VAL) in the locked order and writes manifests in the format of the Hypersim cohorts; it
never runs a model or looks at a score. The rules and helpers are the Replica cohort's
(scripts/prepare_rgbd_replica_data.py), checked on the probe scan 40753679, which has no visit id
and can never be selected.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from mcss.mechanism_pilot.data_prep import resize_rgb_depth
from mcss.mechanism_pilot.unseen_data import frame_quality

EXPERIMENT = "EXP-3D-RGBD-ARKIT-DATA-V1"
DATA = Path("/home/xmum26_zonghan/zonghan_o_mirror/_data_arkitscenes")
ARKIT_REPO = Path("/home/xmum26_zonghan/cv_work/arkit/ARKitScenes")
ARKIT_COMMIT = "474aaa83d7b1c9080619d8485de2d7c885c01fc5"
SPLITS_CSV = ARKIT_REPO / "threedod/3dod_train_val_splits.csv"
URL = "https://docs-assets.developer.apple.com/ml-research/datasets/arkitscenes/v1/threedod"
SEED = 20261005
VENUES = {"Training": 360, "Validation": 70}
FOLDS = ("Training", "Validation")
USES = {"DEV": 8, "TRAIN": 288, "ARKIT_VAL": 50}
TRAIN_SETS = {"ARKIT72": 72, "ARKIT288": 288}
SOURCE_SIZE = (192, 256)
TARGET_SIZE = (128, 160)
# The TRAIN72 Hypersim intrinsics at 128x160 (every Hypersim record carries them).
HYPERSIM_FX, HYPERSIM_FY = 138.56405994296202, 147.80166393915948
FRAMES_PER_SCENE = 16
SPACING_CANDIDATES = (4, 5, 6, 8, 10, 12, 15, 20)
HYPERSIM_CONSECUTIVE_ROTATION_DEG = 15.4
INTRINSICS_TOLERANCE_PX = 0.5
POSE_TIME_TOLERANCE_S = 0.005
GEOMETRY_MEDIAN_ABSREL_MAX = 0.01
GEOMETRY_MIN_LANDING = 0.05
REPLICA = Path("scripts/prepare_rgbd_replica_data.py")
USE = (
    "ARKitScenes is a separate real-world domain (iPad RGB + LiDAR): no scene, venue or camera "
    "is shared with Hypersim. DEV and TRAIN come from the official Training fold, ARKIT_VAL from "
    "the Validation fold, one scan per venue, so the three are venue-disjoint. Its first user "
    "must be named in a plan written before any ARKitScenes result; the protected Hypersim final "
    "holdout is not opened."
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def read(path):
    return json.loads(Path(path).read_text())


def replica():
    spec = importlib.util.spec_from_file_location("arkit_replica_rules", REPLICA)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def frames_dir(fold, video):
    return DATA / "raw" / fold / video / f"{video}_frames"


# ---------------------------------------------------------------- select (split table only)
def select(root):
    root = Path(root).resolve()
    if root.name != EXPERIMENT:
        raise PermissionError("Exact experiment directory required")
    if (root / "selection.json").exists():
        raise FileExistsError("Selection exists; never re-select")
    rows = list(csv.DictReader(SPLITS_CSV.open()))
    venues = {}
    for row in rows:
        if row["visit_id"] in ("", "NA"):
            continue
        venues.setdefault((row["fold"], row["visit_id"]), []).append(row["video_id"])
    folds = {fold: sorted(v for f, v in venues if f == fold) for fold in VENUES}
    shared = sorted(set(folds["Training"]) & set(folds["Validation"]))
    folds = {fold: [v for v in visits if v not in shared] for fold, visits in folds.items()}
    rng = np.random.default_rng(SEED)
    selected = {}
    for fold, count in VENUES.items():
        order = rng.permutation(len(folds[fold]))[:count]
        picks = [folds[fold][int(i)] for i in order]
        selected[fold] = [
            {"visit_id": visit, "video_id": min(venues[(fold, visit)], key=int)} for visit in picks
        ]
    write(
        root / "selection.json",
        {
            "schema": "mcss.rgbd_arkit.selection.v1",
            "experiment": EXPERIMENT,
            "splits_csv": str(SPLITS_CSV),
            "splits_csv_sha256": sha(SPLITS_CSV),
            "arkit_repo_commit": ARKIT_COMMIT,
            "rule": "per fold, venues with a visit id in a seeded order (numpy default_rng "
            f"{SEED}, permutation), the first {VENUES}; per venue the scan with the smallest "
            "video id; scans without a visit id excluded; venues listed in both folds excluded",
            "excluded_shared_venues": shared,
            "selected": selected,
            "image_or_depth_read": False,
            "created_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        },
    )
    print("SELECTED", {k: len(v) for k, v in selected.items()}, flush=True)


# ---------------------------------------------------------------- download (zips -> frame folders)
def download(root, folds=FOLDS):
    root = Path(root).resolve()
    chosen = read(root / "selection.json")["selected"]
    record = {}
    for fold, scans in chosen.items():
        if fold not in folds:
            continue
        for scan in scans:
            video = scan["video_id"]
            target = DATA / "zips" / fold / f"{video}.zip"
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.is_file():
                part = target.with_suffix(".part")
                subprocess.run(
                    [
                        "curl",
                        "-sS",
                        "--fail",
                        "--retry",
                        "3",
                        "-o",
                        str(part),
                        f"{URL}/{fold}/{video}.zip",
                    ],
                    check=True,
                )
                part.rename(target)
            if not frames_dir(fold, video).is_dir():
                # unzip -d creates only the last path component (the 2026-10-05 exit status 2)
                (DATA / "raw" / fold).mkdir(parents=True, exist_ok=True)
                subprocess.run(
                    [
                        "unzip",
                        "-q",
                        "-o",
                        str(target),
                        f"{video}/{video}_frames/*",
                        "-d",
                        str(DATA / "raw" / fold),
                    ],
                    check=True,
                )
            record[video] = {"fold": fold, "bytes": target.stat().st_size, "sha256": sha(target)}
    total = sum(v["bytes"] for v in record.values())
    write(
        root / "downloads.json",
        {"url": URL, "folds": list(folds), "scans": record, "total_bytes": total},
    )
    print("DOWNLOADED", len(record), "scans", round(total / 1e9, 2), "GB", flush=True)


# ---------------------------------------------------------------- lock (camera metadata only)
def trajectory(path):
    stamps, poses = [], []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        tokens = line.split()
        rotation, _ = cv2.Rodrigues(np.array([float(x) for x in tokens[1:4]]))
        w2c = np.eye(4)
        w2c[:3, :3], w2c[:3, 3] = rotation, [float(x) for x in tokens[4:7]]
        stamps.append(float(tokens[0]))
        poses.append(np.linalg.inv(w2c))
    return np.array(stamps), np.array(poses)


def pincam(path):
    width, height, fx, fy, cx, cy = np.loadtxt(path)
    if (int(height), int(width)) != SOURCE_SIZE:
        raise ValueError("Unexpected frame size")
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])


def crop_box(source):
    """Crop to the Hypersim field of view, centred on the principal point."""
    width = round(TARGET_SIZE[1] * source[0, 0] / HYPERSIM_FX)
    height = round(TARGET_SIZE[0] * source[1, 1] / HYPERSIM_FY)
    x0 = round(source[0, 2] + 0.5 - width / 2)
    y0 = round(source[1, 2] + 0.5 - height / 2)
    if x0 < 0 or y0 < 0 or x0 + width > SOURCE_SIZE[1] or y0 + height > SOURCE_SIZE[0]:
        raise ValueError("Hypersim field of view does not fit the frame")
    return {"x0": x0, "y0": y0, "width": width, "height": height}


def target_intrinsics(source, box):
    """Crop, then resize with the pixel-centre convention of resize_rgb_depth (as for Replica)."""
    sy, sx = TARGET_SIZE[0] / box["height"], TARGET_SIZE[1] / box["width"]
    cx, cy = source[0, 2] - box["x0"], source[1, 2] - box["y0"]
    return np.array(
        [
            [source[0, 0] * sx, 0.0, (cx + 0.5) * sx - 0.5],
            [0.0, source[1, 1] * sy, (cy + 0.5) * sy - 0.5],
            [0.0, 0.0, 1.0],
        ]
    )


def lock_scan(rules, fold, video):
    base = frames_dir(fold, video)
    names = {}
    for kind, suffix in (
        ("lowres_wide", ".png"),
        ("lowres_depth", ".png"),
        ("lowres_wide_intrinsics", ".pincam"),
    ):
        names[kind] = {p.name[: -len(suffix)] for p in (base / kind).glob(f"*{suffix}")}
    common = sorted(set.intersection(*names.values()), key=lambda n: float(n.split("_")[1]))
    stamps, poses = trajectory(base / "lowres_wide.traj")
    usable = []
    for name in common:
        stamp = float(name.split("_")[1])
        index = int(np.argmin(np.abs(stamps - stamp)))
        if abs(stamps[index] - stamp) <= POSE_TIME_TOLERANCE_S:
            usable.append((name, poses[index]))
    table, best = {}, None
    for spacing in SPACING_CANDIDATES:
        chosen = usable[::spacing][:FRAMES_PER_SCENE]
        if len(chosen) < FRAMES_PER_SCENE:
            continue
        _, angles = rules.consecutive_motion(np.array([p for _, p in chosen]))
        table[spacing] = float(np.median(angles))
    if not table:
        raise ValueError("Too few frames for 16 at any candidate spacing")
    best = min(table, key=lambda s: (abs(table[s] - HYPERSIM_CONSECUTIVE_ROTATION_DEG), s))
    chosen = usable[::best][:FRAMES_PER_SCENE]
    sources = [pincam(base / "lowres_wide_intrinsics" / f"{name}.pincam") for name, _ in chosen]
    if max(float(np.max(np.abs(s - sources[0]))) for s in sources) > INTRINSICS_TOLERANCE_PX:
        raise ValueError("Intrinsics drift across the selected frames")
    box = crop_box(sources[0])
    return {
        "spacing": best,
        "spacing_table_median_rotation_deg": {str(k): v for k, v in table.items()},
        "frames": [
            {
                "frame_id": k,
                "source_name": name,
                "c2w_opencv": pose.tolist(),
                "source_intrinsics": source.tolist(),
                "target_intrinsics": target_intrinsics(source, box).tolist(),
            }
            for k, ((name, pose), source) in enumerate(zip(chosen, sources, strict=True))
        ],
        "crop": box,
    }


def lock(root, folds=FOLDS):
    root = Path(root).resolve()
    if (root / "candidate_lock.json").exists():
        raise FileExistsError("Candidate lock exists; never re-lock")
    selection = read(root / "selection.json")
    rules = replica()
    scans, failures = {}, {}
    for fold, chosen in selection["selected"].items():
        if fold not in folds:
            continue
        for scan in chosen:
            video = scan["video_id"]
            try:
                scans[video] = {
                    "fold": fold,
                    "visit_id": scan["visit_id"],
                    **lock_scan(rules, fold, video),
                }
            except ValueError as error:
                failures[video] = {
                    "fold": fold,
                    "status": "INELIGIBLE_CAMERA_METADATA",
                    "error": str(error),
                }
    write(
        root / "candidate_lock.json",
        {
            "schema": "mcss.rgbd_arkit.cohort.v1",
            "experiment": EXPERIMENT,
            "selection_sha256": sha(root / "selection.json"),
            "downloads_sha256": sha(root / "downloads.json"),
            "folds": list(folds),
            "license": "ARKitScenes: non-commercial research use only; download approved by the "
            "user on 2026-10-05; media outside the git repository",
            "frame_rule": {
                "frames_per_scene": FRAMES_PER_SCENE,
                "spacing_rule": f"per scan, among {SPACING_CANDIDATES} (frames of the ~10 fps "
                "3dod sequence, from its first usable frame), the spacing whose median "
                "consecutive camera rotation of the 16 frames is closest to the TRAIN72 Hypersim "
                f"first16 median ({HYPERSIM_CONSECUTIVE_ROTATION_DEG} deg); camera metadata only",
                "pose_rule": f"the trajectory row within {POSE_TIME_TOLERANCE_S} s of the frame",
            },
            "camera_model": {
                "pose_convention": "lowres_wide.traj rows are world-to-camera (axis-angle, "
                "translation in m) in the OpenCV convention; c2w = inverse; checked by the "
                "geometry rule (probe 40753679: median pair AbsRel 0.0047)",
                "crop": "to the TRAIN72 Hypersim field of view (fx 138.564, fy 147.802 at "
                "128x160), centred on the principal point; then resized to 128x160",
                "depth": "uint16 PNG / 1000 = planar z in metres, converted to ray distance at "
                "full resolution, then cropped and resized with the frozen resize_rgb_depth "
                "(bilinear RGB, validity-weighted bilinear depth)",
            },
            "validity_rule": {
                "frame": "frozen frame_quality on the cropped full-resolution frame (finite RGB, "
                "near-black <=.999, positive finite depth >=.95); camera max|R^T R - I|<=.001 and "
                "det>0",
                "geometry": "median over consecutive selected-frame pairs of the median relative "
                f"z error of reprojected full-resolution depth <= {GEOMETRY_MEDIAN_ABSREL_MAX}, "
                f"with median landing fraction >= {GEOMETRY_MIN_LANDING} (the Replica rule)",
                "roles": "frozen V2 camera_only_roles with the frozen TRAIN prior bounds",
                "uses": f"eligible scans in the selection order: Training fold -> the first "
                f"{USES['DEV']} DEV, the next {USES['TRAIN']} TRAIN; Validation fold -> the first "
                f"{USES['ARKIT_VAL']} ARKIT_VAL; train sets = the first 72 and all 288 TRAIN scans",
                "replacement": "none beyond the locked order",
            },
            "scans": scans,
            "failures": failures,
            "use": USE,
            "image_or_depth_read": False,
            "model_forward_run": False,
            "source_sha256": {
                str(p): sha(p) for p in (Path(__file__).resolve(), REPLICA.resolve(), SPLITS_CSV)
            },
            "created_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        },
    )
    print("LOCKED", len(scans), "scans", "camera failures", len(failures), flush=True)


# ---------------------------------------------------------------- prepare (media)
def prepare_scan(rules, roles_of, prior, video, scan, out):
    base = frames_dir(scan["fold"], video)
    box = scan["crop"]
    frames, quality, depths, poses, sources = [], {}, {}, {}, {}
    out.mkdir(parents=True, exist_ok=True)
    for entry in scan["frames"]:
        fid, name = entry["frame_id"], entry["source_name"]
        source = np.array(entry["source_intrinsics"])
        rgb = np.asarray(Image.open(base / "lowres_wide" / f"{name}.png").convert("RGB"))
        z = (
            np.asarray(Image.open(base / "lowres_depth" / f"{name}.png")).astype(np.float64)
            / 1000.0
        )
        if rgb.shape[:2] != SOURCE_SIZE or z.shape != SOURCE_SIZE:
            raise ValueError(f"{video}/{fid}: unexpected frame size")
        z = np.where(z > 0, z, np.nan)
        window = (
            slice(box["y0"], box["y0"] + box["height"]),
            slice(box["x0"], box["x0"] + box["width"]),
        )
        distance = rules.ray_distance(z, source)[window]
        rgb_crop = rgb[window]
        pose = np.array(entry["c2w_opencv"])
        rotation = pose[:3, :3]
        quality[fid] = frame_quality(rgb_crop, distance)
        orth = float(np.max(np.abs(rotation.T @ rotation - np.eye(3))))
        det = float(np.linalg.det(rotation))
        quality[fid].update(camera_orthogonality_error=orth, camera_determinant=det)
        quality[fid]["eligible"] = quality[fid]["eligible"] and orth <= 1e-3 and det > 0
        depths[fid], poses[fid], sources[fid] = z, pose, source
        rgb_small, depth_small = resize_rgb_depth(
            rgb_crop, np.nan_to_num(distance, nan=0.0), TARGET_SIZE
        )
        rgb_out, depth_out = out / f"{fid:04d}.png", out / f"{fid:04d}.npy"
        Image.fromarray(rgb_small).save(rgb_out)
        np.save(depth_out, depth_small.astype(np.float32))
        frames.append(
            {
                "frame_id": fid,
                "rgb": str(rgb_out),
                "depth": str(depth_out),
                "intrinsics": entry["target_intrinsics"],
                "c2w": entry["c2w_opencv"],
            }
        )
    errors, landings = [], []
    ids = [e["frame_id"] for e in scan["frames"]]
    for a, b in zip(ids[:-1], ids[1:], strict=True):
        error, landing = rules.reprojection_error(
            depths[a], poses[a], depths[b], poses[b], sources[a]
        )
        if error is not None:
            errors.append(error)
        landings.append(landing)
    check = {
        "median_pair_absrel": float(np.median(errors)) if errors else None,
        "median_landing_fraction": float(np.median(landings)),
        "pairs": len(ids) - 1,
    }
    check["pass"] = bool(
        errors
        and check["median_pair_absrel"] <= GEOMETRY_MEDIAN_ABSREL_MAX
        and check["median_landing_fraction"] >= GEOMETRY_MIN_LANDING
    )
    if not check["pass"]:
        return None, None, quality, check, {"status": "INELIGIBLE_SOURCE_GEOMETRY", **check}
    sid = f"arkit_{video}"
    try:
        roles, support = roles_of(sid, frames, quality, prior)
    except ValueError as error:
        failure = {"status": "INELIGIBLE_FRAME_ROLES", "error": str(error)}
        return None, None, quality, check, failure
    record = {
        "scene_id": sid,
        "physical_scene_id": f"arkit_visit_{scan['visit_id']}",
        "historically_exposed": False,
        "historical_exposure_basis": USE,
        "dataset": "ARKitScenes 3dod (lowres_wide, lowres_depth)",
        "roles": roles,
        "frames": frames,
    }
    return record, support, quality, check, None


def prepare(root, folds=FOLDS):
    root = Path(root).resolve()
    lock_ = read(root / "candidate_lock.json")
    if (root / "manifest_arkit_val.json").exists():
        raise FileExistsError("Completed manifests exist; no overwrite")
    for path, digest in lock_["source_sha256"].items():
        if sha(path) != digest:
            raise PermissionError(f"Locked source changed: {path}")
    rules = replica()
    roles_of, prior = rules.role_rule(), rules.frozen_prior()
    selection = read(root / "selection.json")["selected"]
    eligible = {"Training": [], "Validation": []}
    qualities, geometry, failures, hashes, supports = {}, {}, dict(lock_["failures"]), {}, {}
    for fold, chosen in selection.items():
        if fold not in folds:
            continue
        need = USES["DEV"] + USES["TRAIN"] if fold == "Training" else USES["ARKIT_VAL"]
        for scan in chosen:
            video = scan["video_id"]
            if video not in lock_["scans"] or len(eligible[fold]) >= need:
                continue
            record, support, quality, check, failure = prepare_scan(
                rules, roles_of, prior, video, lock_["scans"][video], DATA / "prepared" / video
            )
            qualities[video], geometry[video] = {str(k): v for k, v in quality.items()}, check
            if failure:
                failures[video] = {"fold": fold, **failure}
                continue
            supports[video] = support
            for frame in record["frames"]:
                for kind in ("rgb", "depth"):
                    hashes[frame[kind]] = sha(frame[kind])
            eligible[fold].append(record)
    train_fold, val_fold = eligible["Training"], eligible["Validation"]
    for record in train_fold[: USES["DEV"]]:
        record["split"] = "DEV"
    for record in train_fold[USES["DEV"] :]:
        record["split"] = "TRAIN"
    for record in val_fold:
        record["split"] = "ARKIT_VAL"
    train = [r["scene_id"] for r in train_fold[USES["DEV"] :]]
    common = {
        "image_size": list(TARGET_SIZE),
        "depth_semantics": "ray_distance_meters",
        "candidate_lock_sha256": sha(root / "candidate_lock.json"),
        "protected_scene_ids": [],
    }
    if "Training" in folds:
        write(
            root / "manifest_arkit_trainval.json",
            {
                "schema": "mcss.rgbd_arkit.data.v1",
                **common,
                "scenes": train_fold,
                "train_sets": {name: train[:count] for name, count in TRAIN_SETS.items()},
            },
        )
    if "Validation" in folds:
        write(
            root / "manifest_arkit_val.json",
            {"schema": "mcss.rgbd_arkit.data.v1", **common, "scenes": val_fold},
        )
    write(root / "frame_quality.json", qualities)
    write(root / "geometry_checks.json", geometry)
    write(root / "role_support.json", supports)
    write(root / "scene_failures.json", failures)
    write(root / "hashes.json", hashes)
    counts = {
        "DEV": len(train_fold[: USES["DEV"]]),
        "TRAIN": len(train),
        "ARKIT_VAL": len(val_fold),
    }
    use_folds = (("DEV", "Training"), ("TRAIN", "Training"), ("ARKIT_VAL", "Validation"))
    wanted = [k for k, fold in use_folds if fold in folds]
    status = (
        "PASS" if all(counts[k] >= USES[k] for k in wanted) else "BLOCKED_INSUFFICIENT_VALID_SCENES"
    )
    write(
        root / "preparation_integrity.json",
        {
            "status": status,
            "folds": list(folds),
            "counts": counts,
            "failures": len(failures),
            "model_forward_run": False,
            "score_computed": False,
            "candidate_lock_sha256": sha(root / "candidate_lock.json"),
        },
    )
    print(status, counts, "failures", len(failures), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("select", "download", "lock", "prepare"), required=True)
    parser.add_argument("--root", type=Path, default=Path("outputs") / EXPERIMENT)
    parser.add_argument(
        "--folds",
        nargs="+",
        choices=FOLDS,
        default=list(FOLDS),
        help="download / lock / prepare only these official folds (selection is unchanged)",
    )
    args = parser.parse_args()
    if args.stage == "select":
        select(args.root)
    else:
        stage = {"download": download, "lock": lock, "prepare": prepare}[args.stage]
        stage(args.root, tuple(args.folds))


if __name__ == "__main__":
    main()
