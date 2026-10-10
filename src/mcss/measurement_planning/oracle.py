"""Per-episode computation of EXP-MEASUREMENT-PLANNING-ORACLE-V1, in an importable module so that
worker processes can run it.

The teacher path evaluates every subset of the candidate pool with the frozen model-free backend;
the deployable path runs every selector through its own MeasurementEnv and audits what it read.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from mcss.measurement_planning import geometry as g
from mcss.measurement_planning import heuristics as h
from mcss.measurement_planning.information import (
    CANDIDATE_DEPTH_PURPOSE,
    CANDIDATE_RGB_PURPOSE,
    CONTEXT_PURPOSE,
    MeasurementEnv,
    TeacherAccess,
)

FIELDS = ("all", "near", "far", "coverage_all", "coverage_far", "g1_all", "g1_near", "g1_far")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def seal_arrays(seal_path, key):
    """A sealed arms-evaluator prediction file (REPROJ_HARMONIC, near, G1 seeds), hash-checked."""
    if seal_path is None or not Path(seal_path).exists():
        return None
    seal = json.loads(Path(seal_path).read_text())
    item = seal["predictions"].get(key)
    if item is None:
        return None
    if sha(item["path"]) != item["sha256"]:
        raise PermissionError("A sealed prediction changed")
    return dict(np.load(item["path"]))


def audit_log(log, context, pool, k, name):
    """A selector's reads: context RGB-D, candidate RGB/camera, and exactly k requested candidate
    depths."""
    problems = []
    depth_reads = [
        e for e in log if "depth" in e["channels"] and e["purpose"] == CANDIDATE_DEPTH_PURPOSE
    ]
    for e in log:
        fid, purpose = e["frame_id"], e["purpose"]
        if purpose == CONTEXT_PURPOSE and fid not in context:
            problems.append(f"context read of {fid}")
        elif purpose in (CANDIDATE_RGB_PURPOSE, CANDIDATE_DEPTH_PURPOSE) and fid not in pool:
            problems.append(f"candidate read of {fid}")
        elif purpose == CANDIDATE_RGB_PURPOSE and "depth" in e["channels"]:
            problems.append(f"depth with RGB of {fid}")
        elif purpose not in (CONTEXT_PURPOSE, CANDIDATE_RGB_PURPOSE, CANDIDATE_DEPTH_PURPOSE):
            problems.append(f"purpose {purpose}")
    if len(depth_reads) != k or len({e["frame_id"] for e in depth_reads}) != k:
        problems.append(f"{len(depth_reads)} candidate depth reads for budget {k}")
    return {"selector": name, "k": k, "ok": not problems, "problems": problems}


def run_episode(job):
    """Teacher table over every subset, deployable selections for K = 1 .. pool size, features."""
    torch.set_num_threads(1)
    constant, record, repo = job["constant"], job["record"], Path(job["repo"])
    pool, queries, context_ids = job["pool"], job["queries"], job["context"]
    image_size = tuple(job["image_size"])
    teacher_log = []
    teacher = TeacherAccess(record, queries, image_size, repo, teacher_log)
    points = {
        **g.frame_points(teacher.frames(context_ids)),
        **g.frame_points(teacher.frames(pool)),
    }
    n, subsets = len(pool), 1 << len(pool)
    table = {f: np.full((subsets, len(queries)), np.nan) for f in FIELDS}
    replay, cameras = {}, []
    for qi, qid in enumerate(queries):
        camera = teacher.query_camera(qid)
        cameras.append(camera)
        cosine = g.ray_cosine(camera)
        gt = teacher.query_depth(qid)
        valid = np.isfinite(gt) & (gt > 0)
        maps = {fid: g.reproject(points[fid], camera) for fid in [*context_ids, *pool]}
        base = g.union([maps[f] for f in context_ids])
        near0 = g.near_mask(np.isfinite(base))
        masks = {"all": valid, "near": valid & near0, "far": valid & ~near0}
        key = f"{job['seal_prefix']}/{job['scene_id']}/{job['role']}/{qid}"
        sealed = seal_arrays(job["seal_path"], key)
        g1 = None
        if sealed is not None:
            seeds = sorted(k for k in sealed if k.startswith("G1_"))
            g1 = np.mean([sealed[k] for k in seeds], axis=0) if seeds else None
        for bits in range(subsets):
            chosen = [pool[i] for i in range(n) if bits >> i & 1]
            depth = g.union([base, *[maps[f] for f in chosen]]) if chosen else base
            prediction, hit = g.harmonic(depth, cosine, constant)
            covered = g.near_mask(hit)
            for region, mask in masks.items():
                value = g.absrel(prediction, gt, mask)
                table[region][bits, qi] = np.nan if value is None else value
            table["coverage_all"][bits, qi] = float(covered[valid].mean())
            if masks["far"].any():
                table["coverage_far"][bits, qi] = float(covered[masks["far"]].mean())
            if g1 is not None:
                hybrid = np.where(covered, prediction, g1)
                for region, mask in masks.items():
                    value = g.absrel(hybrid, gt, mask)
                    table[f"g1_{region}"][bits, qi] = np.nan if value is None else value
            if bits == 0 and sealed is not None:
                replay[str(qid)] = {
                    "max_abs_diff_harmonic": float(
                        np.max(np.abs(prediction - sealed["REPROJ_HARMONIC"]))
                    ),
                    "near_equal": bool(np.array_equal(near0, sealed["near"].astype(bool))),
                }
    # deployable selections, budget curve K = 1 .. n
    selections, audits = defaultdict(dict), []
    for k in range(1, n + 1):
        for name, select in h.DEPLOYABLE.items():
            log = []
            env = MeasurementEnv(record, job["role"], pool, k, image_size, repo, log)
            select(env, k, constant=constant)
            if len(env.requested) != k:
                raise AssertionError(f"{name} acquired {len(env.requested)} != {k}")
            selections[name][str(k)] = list(env.requested)
            audits.append(audit_log(log, set(context_ids), set(pool), k, name))
    # query-aware diagnostic (teacher cameras; never deployable)
    candidate_cameras = {
        c.frame_id: c.camera
        for c in MeasurementEnv(record, job["role"], pool, 0, image_size, repo, []).candidates()
    }
    nearest = {
        fid: h.pose_features(camera, cameras)["pose_distance"]
        for fid, camera in candidate_cameras.items()
    }
    order = sorted(pool, key=lambda f: (nearest[f], f))
    diagnostic = {"QUERY_NEAREST": {str(k): order[:k] for k in range(1, n + 1)}}
    # deployable single-step features from the initial context
    env0 = MeasurementEnv(record, job["role"], pool, 0, image_size, repo, [])
    measured = h.MeasuredGeometry(env0.context())
    features = h.candidate_features(measured, env0.candidates(), constant, list(pool))
    teacher_audit = sorted({(e["purpose"], tuple(e["channels"])) for e in teacher_log})
    stem = f"{job['scene_id']}_{job['role']}"
    out = Path(job["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / f"{stem}.npz", **table)
    write(
        out / f"{stem}.json",
        {
            "scene_id": job["scene_id"],
            "role": job["role"],
            "pool": pool,
            "queries": queries,
            "context": context_ids,
            "selections": selections,
            "diagnostic_selections": diagnostic,
            "features": {str(k): v for k, v in features.items()},
            "replay_k0": replay,
            "selector_audits_ok": all(a["ok"] for a in audits),
            "selector_audit_failures": [a for a in audits if not a["ok"]],
            "teacher_reads": [[p, list(c)] for p, c in teacher_audit],
            "table_sha256": sha(out / f"{stem}.npz"),
        },
    )
    return stem
