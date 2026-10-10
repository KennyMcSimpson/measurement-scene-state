"""Sealed predictions of V17-style carrier arms built from SEVEN views: the role's three context frames followed by
its four Core B stream frames, through the training-time extra-view path (rgbd_viewcount_carrier.with_extra_views,
the path that trained every V17-style carrier on 3-7 views). The bounds stay the CONTEXT_DEPTH bounds of the three
context frames, as in training. Everything else is the arms evaluator's: its contract (arms, seeds, cohort), its
context loader, its arm checkpoints (hash-checked) and its renderer. Only context and stream depth are read; no
query depth is read. Output: ROOT/raw/eval7/COHORT/ (predictions with <ARM>_<seed> depth and OPACITY_<ARM>_<seed>,
prediction_seal.json, state hashes and the access log). Run through cv_host_run.py from the repository root
(CUDA hidden):  python eval_carrier_7view.py ROOT COHORT
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

from mcss.dynamic.types import hash_scene_state
from mcss.geometry import transform_cameras
from mcss.measurements import FixedMeasurementRenderer
from mcss.mechanism_pilot.rgbd_depth_bounds import context_depth_bounds
from mcss.mechanism_pilot.rgbd_stream_write import StreamRGBDLoader
from mcss.mechanism_pilot.rgbd_viewcount_carrier import with_extra_views
from mcss.mechanism_pilot.small_training import sha, write_json
from mcss.types import Cameras

ARMS = Path("scripts/evaluate_rgbd_carrier_arms.py")


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def main(root, cohort):
    arms_eval = module("arms_eval_7view", ARMS)
    config, cohorts = arms_eval.contract(root)
    entry, manifest, base = cohorts[cohort]
    out = root / "raw" / "eval7" / cohort
    if (out / "prediction_seal.json").exists():
        raise FileExistsError("7-view predictions are sealed once")
    frozen = arms_eval.frozen_v17_evaluation()
    v11 = frozen.host()
    geo = v11.geometry()
    models = arms_eval.arm_carriers(config)
    renderer = FixedMeasurementRenderer(n_samples=64, ray_chunk_size=2048)
    lock = json.loads((root / "frame_role_lock.json").read_text())["cohorts"][cohort]
    locked_streams = {(e["scene_id"], e["role"]): list(e["stream"]) for e in lock["episodes"]}
    ids = [r["scene_id"] for r in manifest["scenes"]]
    access, entries, states = [], {}, {}
    for record in manifest["scenes"]:
        sid = record["scene_id"]
        streams = StreamRGBDLoader(record, manifest["image_size"], base, "cpu", access, allowed_scene_ids=ids,
                                   holdout_scene_ids=manifest.get("protected_scene_ids", []))
        for role, context in v11.context_of(record, manifest, base, access).items():
            if list(streams.frame_ids(role)) != locked_streams[(sid, role)]:
                raise PermissionError(f"{sid}/{role}: stream differs from the frame-role lock")
            bounds = context_depth_bounds(context, context[0].camera.c2w)
            seven = with_extra_views(context, streams.stream(role), len(locked_streams[(sid, role)]))
            built = {}
            with torch.no_grad():
                for (arm, seed), carrier in models.items():
                    state, anchor = frozen.build_state(carrier, seven, bounds, f"{cohort.lower()}:{sid}:{role}:7view")
                    built[arm, seed] = (state, anchor)
                    states[f"{cohort}:{arm}:{seed}:{sid}/{role}:7view"] = hash_scene_state(state)
            for qid in record["roles"][entry["role"]]:
                access.append({"cohort": cohort, "scene_id": sid, "frame_id": qid, "purpose": "QUERY_CAMERA_ONLY"})
                camera = geo.query_camera(record, qid, manifest["image_size"])
                batched = Cameras(camera.intrinsics[None, None], camera.c2w[None, None], camera.image_size)
                arrays = {}
                with torch.no_grad():
                    for (arm, seed), (state, anchor) in built.items():
                        pred = renderer(state, transform_cameras(batched, torch.linalg.inv(anchor)), ("depth", "visibility"))
                        arrays[f"{arm}_{seed}"] = pred["depth"][0, 0, 0].numpy().astype(np.float64)
                        arrays[f"OPACITY_{arm}_{seed}"] = pred["visibility"][0, 0, 0].numpy()
                path = out / "predictions" / f"{sid}_{role}_{qid}.npz"
                path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(path, **arrays)
                entries[f"{cohort}/{sid}/{role}/{qid}"] = {"path": str(path), "sha256": sha(path)}
    if any("depth" in e.get("channels", []) and e.get("purpose") not in ("CONTEXT_ONLY_RGBD", "STREAM_RGBD")
           for e in access):
        raise AssertionError("only context and stream depth may be read before the seal")
    write_json(out / "state_hashes.json", states)
    write_json(out / "GT_access_before_seal.json", access)
    write_json(out / "prediction_seal.json", {
        "cohort": cohort, "arms": sorted(config["arms"]), "seeds": config["seeds"], "views": "3 context + 4 stream",
        "predictions": entries, "state_hashes_sha256": sha(out / "state_hashes.json"), "query_depth_read": False})
    print("SEVEN_VIEW_PREDICTIONS_SEALED", cohort, len(entries), flush=True)


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve(), sys.argv[2])
