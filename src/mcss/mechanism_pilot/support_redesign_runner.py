"""Frozen-carrier, static-only development interventions; final holdout is forbidden.

Oracle query geometry is an explicit preseal diagnostic exception. Evaluation labels
are read only after every state is sealed; GT-free states never consume that exception.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from mcss.dynamic.cache import ObservationCache
from mcss.dynamic.checkpoint import load_dynamic_checkpoint
from mcss.dynamic.feedback import anchored_observation, batched_camera
from mcss.dynamic.types import (
    OnlineObservation,
    SealedScene,
    hash_fast_content,
    hash_scene_state,
    hash_value,
)
from mcss.geometry import transform_cameras
from mcss.measurements import FixedMeasurementRenderer
from mcss.mechanism_pilot.contracts import EvaluationLedger
from mcss.mechanism_pilot.small_training import TrainScene, sha, write_json
from mcss.mechanism_pilot.statistics import measurement_metrics
from mcss.mechanism_pilot.support_redesign_contracts import validate_dev_scene_ids
from mcss.mechanism_pilot.support_redesign_geometry import (
    apply_plan,
    deployable_plan,
    enclose,
    nearest_distance,
    oracle_plan,
    surface_points,
    view_counts,
)

ORACLE_DESIGNS = ("R0", "ORACLE_VOLUME", "ORACLE_SUPPORT", "ORACLE_VOLUME_SUPPORT")
RADIUS = (1.5**2 + 1**2 + 1.5**2) ** 0.5 / 2


@torch.no_grad()
def run_development(root, dev_manifest, exposed_manifest, checkpoint, output, *, device="cuda"):
    root, output = Path(root), Path(output)
    if output.exists():
        raise FileExistsError("Refusing to overwrite locked development intervention")
    roles = json.loads((root / "scene_role_lock.json").read_text())
    manifests = {
        "exposed": Path(exposed_manifest),
        "new_dev": Path(dev_manifest),
    }
    parsed = {k: json.loads(v.read_text()) for k, v in manifests.items()}
    ids = [s["scene_id"] for m in parsed.values() for s in m["scenes"]]
    # No media, model loading or GT decode precedes this identity firewall.
    validate_dev_scene_ids(roles, ids)
    if len(parsed["new_dev"]["scenes"]) < 8:
        raise PermissionError("At least8 qualified new development scenes required")
    if set(s["scene_id"] for s in parsed["exposed"]["scenes"]) != set(
        roles["REDESIGN_DEV_EXPOSED"]
    ):
        raise ValueError("All seven exposed scenes must be retained")
    qualification = json.loads((root / "qualified_frame_role_lock.json").read_text())
    if qualification["initial_scene_role_lock_sha256"] != sha(
        root / "scene_role_lock.json"
    ) or qualification["manifest_hashes"]["dev_manifest.json"] != sha(Path(dev_manifest)):
        raise PermissionError("Manifest/scene roles differ from the pre-evaluation lock")
    output.mkdir(parents=True)
    scenes, cohorts, queries = {}, {}, {}
    input_hashes = {str(Path(checkpoint).resolve()): sha(Path(checkpoint))}
    for cohort, manifest in parsed.items():
        source = manifests[cohort]
        input_hashes[str(source.resolve())] = sha(source)
        for record in manifest["scenes"]:
            sid = record["scene_id"]
            scenes[sid] = TrainScene(record, manifest["image_size"], source.parent, device, [])
            cohorts[sid] = cohort
            q = record["roles"]["primary_query"]
            if len(q) < 2 or len(set(q)) != len(q):
                raise ValueError("At least two unique primary queries required")
            a, b = record["roles"]["context_a"], record["roles"]["context_b"]
            if len(a) != 3 or len(b) != 3 or a[0] != b[0] or set(a) & set(b) != {a[0]}:
                raise ValueError("Locked matched three-frame contexts required")
            if set(q) & (set(a) | set(b)):
                raise PermissionError("Query cannot enter arrived observations")
            queries[sid] = q
            for frame in record["frames"]:
                for field in ("rgb", "depth"):
                    path = scenes[sid].path(frame, field)
                    input_hashes[str(path.resolve())] = sha(path)
    for filename in (
        "scene_role_lock.json",
        "qualified_frame_role_lock.json",
        "preregistration.json",
        "oracle_adequacy_clarification.json",
    ):
        path = root / filename
        input_hashes[str(path.resolve())] = sha(path)
    source_files = [
        Path(__file__),
        Path(__file__).with_name("support_redesign_geometry.py"),
        Path(__file__).with_name("support_redesign_contracts.py"),
    ]
    source_files += list(Path(__file__).resolve().parents[1].joinpath("dynamic").glob("*.py"))
    source_hashes = {str(p.resolve()): sha(p) for p in source_files}
    carrier, _, meta = load_dynamic_checkpoint(checkpoint, device=device)
    carrier.eval().requires_grad_(False)
    expected = json.loads((root / "preregistration.json").read_text())["checkpoint_sha256"]
    assert meta["sha256"] == expected
    parameters_before = hash_value(dict(carrier.named_parameters()))
    renderer = FixedMeasurementRenderer(n_samples=64, ray_chunk_size=2048).to(device)
    original_bounds, original_ids = carrier._bounds[0].clone(), carrier._candidate_ids.clone()
    write_json(
        output / "lock.json",
        {
            "checkpoint": meta,
            "designs": ORACLE_DESIGNS,
            "scene_ids": ids,
            "cohorts": cohorts,
            "queries": queries,
            "input_sha256": input_hashes,
            "source_sha256": source_hashes,
            "candidate_budget": 128,
            "grid": [8, 8, 8],
            "renderer_samples": 64,
            "config": asdict(carrier.config),
            "oracle_is_deployable": False,
            "final_holdout_access": False,
            "dynamic_ttt_run": False,
            "created_unix": time.time(),
        },
    )
    access, states, plans, anchors, obs = [], {}, {}, {}, {}
    tick = time.perf_counter()
    for sid, scene in scenes.items():
        rr = scene.record["roles"]
        obs[sid] = {}
        for fid in sorted(set(rr["context_a"] + rr["context_b"])):
            access.append(
                {
                    "scene_id": sid,
                    "frame_id": fid,
                    "kind": "ARRIVED_RGB_CAMERA",
                    "stage": "construction",
                }
            )
            frame = scene.frames[fid]
            obs[sid][fid] = OnlineObservation(sid, fid, scene.rgb(frame), scene.camera(frame))
        anchors[sid] = obs[sid][rr["context_a"][0]].camera.c2w
    ledger = EvaluationLedger(
        f"{d}/{sid}/{role}" for d in ORACLE_DESIGNS for sid in ids for role in ("A", "B", "anchor")
    )
    # First seal all R0 states. Oracle label access never enters baseline construction.
    oracle_surfaces, query_bounds = {}, {}
    for design in ORACLE_DESIGNS:
        for sid, scene in scenes.items():
            rr, anchor = scene.record["roles"], anchors[sid]
            if design != "R0" and sid not in oracle_surfaces:
                surfaces = {}
                for fid in sorted(set(rr["context_a"] + rr["context_b"] + queries[sid])):
                    frame = scene.frames[fid]
                    access.append(
                        {
                            "scene_id": sid,
                            "frame_id": fid,
                            "kind": "ORACLE_GEOMETRY_GT_DEPTH_CAMERA",
                            "stage": "PRESEAL_DIAGNOSTIC_EXCEPTION",
                            "query_geometry": fid in queries[sid],
                            "deployable": False,
                        }
                    )
                    depth = (
                        torch.from_numpy(np.load(scene.path(frame, "depth")))
                        .float()
                        .to(device)
                        .squeeze()
                    )
                    surfaces[fid] = surface_points(scene.camera(frame), depth, anchor)
                oracle_surfaces[sid] = surfaces
                query_bounds[sid] = enclose(torch.cat([surfaces[i] for i in queries[sid]]))
            for role, frame_ids in [
                ("A", rr["context_a"]),
                ("B", rr["context_b"]),
                ("anchor", rr["context_a"][:1]),
            ]:
                cameras = [anchored_observation(obs[sid][i], anchor).camera for i in frame_ids]
                key = f"{design}/{sid}/{role}"
                if design == "R0":
                    plan = deployable_plan(design, original_bounds, original_ids, cameras)
                else:
                    plan = oracle_plan(
                        design,
                        original_bounds,
                        original_ids,
                        cameras,
                        query_bounds=query_bounds[sid],
                        context_surfaces=torch.cat([oracle_surfaces[sid][i] for i in frame_ids]),
                    )
                apply_plan(carrier, plan)
                episode = key.replace("/", "-")
                cache = ObservationCache(episode, sid)
                fast = carrier.initial_fast(episode)
                for fid in frame_ids:
                    arrived = anchored_observation(obs[sid][fid], anchor)
                    cache = cache.append(arrived, carrier.encode(arrived))
                state = carrier.materialize(cache, fast)
                sealed = SealedScene(
                    episode,
                    sid,
                    "dev",
                    "static-support-diagnostic",
                    state,
                    hash_scene_state(state),
                    tuple(frame_ids),
                    meta["sha256"],
                    hash_value({"design": design, "plan": plan}),
                    hash_fast_content(fast),
                    anchor.clone(),
                )
                states[key], plans[key] = sealed, plan
                ledger.seal(key, sealed)
                access.append(
                    {
                        "kind": "SEALED_STATE",
                        "state_key": key,
                        "oracle_geometry_used": design != "R0",
                    }
                )
                assert not fast.delta_fuse.any() and not fast.delta_complete.any()
    ledger.assert_ready()
    assert hash_value(dict(carrier.named_parameters())) == parameters_before
    torch.save(states, output / "sealed_states.pt")
    torch.save(plans, output / "geometry_plans.pt")
    write_json(
        output / "geometry_plans.json",
        {
            k: {
                "bounds": p.bounds.tolist(),
                "candidate_ids": p.candidate_ids.tolist(),
                "points": p.points.tolist(),
                "normalized_xyz": p.normalized_xyz.tolist(),
                "oracle": p.oracle,
            }
            for k, p in plans.items()
        },
    )
    rows, predictions = [], output / "predictions"
    predictions.mkdir()
    for sid, scene in scenes.items():
        cohort_ids = sorted(s for s in ids if cohorts[s] == cohorts[sid])
        donor = cohort_ids[(cohort_ids.index(sid) + 1) % len(cohort_ids)]
        for fid in queries[sid]:
            frame = scene.frames[fid]
            world_camera = ledger.read_query_camera(
                f"{sid}-{fid}", lambda frame=frame, scene=scene: scene.camera(frame)
            )
            camera = batched_camera(transform_cameras(world_camera, torch.linalg.inv(anchors[sid])))
            camera_hash = hash_value(camera)

            def truth_loader(sid=sid, scene=scene, frame=frame, fid=fid):
                access.append(
                    {
                        "scene_id": sid,
                        "frame_id": fid,
                        "kind": "QUERY_RGB_DEPTH_GT",
                        "stage": "POST_ALL_STATES_SEALED_EVALUATION",
                    }
                )
                return scene.rgb(frame), torch.from_numpy(
                    np.load(scene.path(frame, "depth"))
                ).float().to(device).squeeze()

            truth, depth = ledger.read_query_ground_truth(f"{sid}-{fid}", truth_loader)
            surfaces = surface_points(world_camera, depth, anchors[sid])
            for design in ORACLE_DESIGNS:
                for role in ("A", "B", "anchor", "prior", "wrong_scene"):
                    source_id = donor if role == "wrong_scene" else sid
                    source_role = "A" if role == "wrong_scene" else role
                    key = f"{design}/{source_id}/{source_role}"
                    t0 = time.perf_counter()
                    if role == "prior":
                        pred = {
                            "rgb": torch.full_like(truth[None, None], 0.5),
                            "depth": torch.full_like(depth[None, None, None], 5.0),
                            "visibility": torch.ones_like(depth[None, None, None]),
                        }
                        plan = plans[f"{design}/{sid}/A"]
                        counts = view_counts(
                            plan.points,
                            [
                                anchored_observation(o, anchors[sid]).camera
                                for o in obs[sid].values()
                            ],
                        )
                    else:
                        sealed, plan = states[key], plans[key]
                        pred = renderer(sealed.scene_state, camera, ("rgb", "depth", "visibility"))
                        assert hash_scene_state(sealed.scene_state) == sealed.state_hash
                        source_sid = donor if role == "wrong_scene" else sid
                        counts = view_counts(
                            plan.points,
                            [
                                anchored_observation(obs[source_sid][i], anchors[source_sid]).camera
                                for i in sealed.observed_ids
                            ],
                        )
                    if device == "cuda":
                        torch.cuda.synchronize()
                    seconds = time.perf_counter() - t0
                    assert camera_hash == hash_value(camera)
                    inside = ((surfaces >= plan.bounds[0]) & (surfaces <= plan.bounds[1])).all(-1)
                    supported = counts >= 2
                    dist = nearest_distance(surfaces, plan.points[supported])
                    scaled_radius = float(((plan.bounds[1] - plan.bounds[0]) / 8).norm() / 2)
                    row = {
                        "scene_id": sid,
                        "cohort": cohorts[sid],
                        "query_id": fid,
                        "design": design,
                        "method": role,
                        "query_camera_hash": camera_hash,
                        "donor_scene_id": donor if role == "wrong_scene" else None,
                        "seconds": seconds,
                        "candidate_count": 128,
                        "candidate_one_view": int((counts >= 1).sum()),
                        "candidate_two_view": int(supported.sum()),
                        "candidate_three_view": int((counts >= 3).sum()),
                        "two_view_candidate_fraction": float(supported.float().mean()),
                        "inside_fraction": float(inside.float().mean()),
                        "supported_surface_fraction": float((dist <= RADIUS).float().mean()),
                        "supported_surface_fraction_voxel_scaled": float(
                            (dist <= scaled_radius).float().mean()
                        ),
                        "fixed_surface_radius_m": RADIUS,
                        "voxel_scaled_radius_m": scaled_radius,
                        "volume_extent": (plan.bounds[1] - plan.bounds[0]).tolist(),
                        "bounds": plan.bounds.tolist(),
                        "prediction_hash": hash_value(pred),
                        **measurement_metrics(
                            pred["rgb"][0, 0],
                            truth,
                            pred["depth"][0, 0, 0],
                            depth,
                            pred["visibility"][0, 0, 0],
                        ),
                    }
                    rows.append(row)
                    path = predictions / f"{sid}_{fid}_{design}_{role}.npz"
                    np.savez_compressed(
                        path,
                        rgb=pred["rgb"][0, 0].cpu().numpy(),
                        depth=pred["depth"][0, 0, 0].cpu().numpy(),
                        opacity=pred["visibility"][0, 0, 0].cpu().numpy(),
                    )
    ledger.assert_ready()
    assert all(sha(Path(p)) == h for p, h in input_hashes.items())
    assert all(sha(Path(p)) == h for p, h in source_hashes.items())
    assert hash_value(dict(carrier.named_parameters())) == parameters_before
    write_json(output / "results.json", rows)
    write_json(output / "access_log.json", access)
    write_json(output / "sealed_query_log.json", ledger.events)
    write_json(
        output / "integrity.json",
        {
            "status": "PASS",
            "checkpoint_unchanged": True,
            "parameters_unchanged": True,
            "source_input_hashes_unchanged": True,
            "final_holdout_access": False,
            "writes": 0,
            "dynamic_ttt_run": False,
            "rows": len(rows),
            "n_scenes": len(ids),
            "seconds": time.perf_counter() - tick,
            "oracle_preseal_exception_explicit": True,
            "all_evaluation_labels_after_complete_seal": True,
        },
    )
    return rows
