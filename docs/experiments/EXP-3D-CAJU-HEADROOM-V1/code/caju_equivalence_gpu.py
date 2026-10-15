"""EXP-3D-CAJU-HEADROOM-V1 (pilot): independent reference runs for the replay's equivalence checks (GPU).

For the first N CAJU_PILOT episodes: the CACHE state encoded twice (OFF_REF, OFF_REF2: encoder determinism), and V1's
own ds_eval.ttt run from the CACHE state for 10 steps twice (SHORT_REF, SHORT_REF2) and for 100 steps twice
(FULL_REF, FULL_REF2), each rendered at the query and history cameras. The replay's SHORT (a snapshot of the FULL
trajectory) must match an independent 10-step run up to the run-to-run floor. File access is audited as in the
replay; no query or history image or depth is read.
Run with the DepthSplat environment from the DepthSplat repository root (PYTHONPATH=.):
  python caju_equivalence_gpu.py ROOT OUT [--episodes 4]
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import PIL.Image
import torch

import caju_replay as cr
from caju_replay import AUDIT, INJECTED, build, inject_measured_depth, local, set_scale_max, sha, v1

ARMS = (("SHORT_REF", 10), ("SHORT_REF2", 10), ("FULL_REF", 100), ("FULL_REF2", 100))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--episodes", type=int, default=4)
    args = parser.parse_args()
    root, out = args.root.resolve(), args.out.resolve()
    lock = json.loads((root / "frame_role_lock.json").read_text())["cohorts"]["CAJU_PILOT"]
    if sha(lock["manifest"]) != lock["manifest_sha256"]:
        raise PermissionError("manifest changed since the frame-role lock")
    records = {r["scene_id"]: r for r in json.loads(Path(lock["manifest"]).read_text())["scenes"]}
    episodes = lock["episodes"][: args.episodes]
    out.mkdir(parents=True, exist_ok=False)
    device = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(cr.SEED)
    set_scale_max(cr.SCALE_MAX)
    checkpoint = Path(cr.ds_predict.CHECKPOINT)
    encoder, decoder = build(device, checkpoint)
    inject_measured_depth(encoder)
    INJECTED["align"] = False
    np.load, PIL.Image.open = cr._audited_np_load, cr._audited_image_open
    started, predictions, logs = time.time(), {}, {}
    for e in episodes:
        frames = records[e["scene_id"]]["frames"]
        rows = {f["frame_id"]: f for f in frames}
        inputs, targets = e["context"] + e["stream"], e["queries"] + e["history"]
        AUDIT.begin([local(rows[i][k]) for i in inputs for k in ("rgb", "depth")])
        cams = v1.cameras(frames, targets, device)
        cached = v1.encode(encoder, frames, inputs, device)
        arrays = {"OFF_REF": v1.query_render(decoder, cached, cams),
                  "OFF_REF2": v1.query_render(decoder, v1.encode(encoder, frames, inputs, device), cams)}
        key = f"{e['scene_id']}/{e['role']}"
        logs[key] = {}
        for name, steps in ARMS:
            optimized, log = v1.ttt(decoder, cached, frames, inputs, steps, cr.MULT, device)
            arrays[name] = v1.query_render(decoder, optimized, cams)
            logs[key][name] = log
            del optimized
        for j, fid in enumerate(targets):
            data = {}
            for name, (depth_z, alpha) in arrays.items():
                data[f"{name}_depth_z"] = depth_z[j]
                data[f"{name}_alpha"] = alpha[j]
            path = out / f"{e['scene_id']}_{e['role']}_{fid}.npz"
            np.savez_compressed(path, **data)
            predictions[f"{key}/{fid}"] = {"path": str(path), "sha256": sha(path),
                                           "kind": "query" if fid in e["queries"] else "history"}
        if AUDIT.violations():
            raise PermissionError(f"{key}: files outside the arrived frames were opened")
        del cached
        print("episode", key, f"{time.time() - started:.0f}s", flush=True)
    np.load, PIL.Image.open = cr._np_load, cr._image_open
    info = {"cohort": "CAJU_PILOT", "episodes": len(episodes), "arms": ["OFF_REF", "OFF_REF2", *[a for a, _ in ARMS]],
            "checkpoint_sha256": sha(checkpoint), "scale_max": cr.SCALE_MAX, "ttt_mult": cr.MULT,
            "v1_ds_eval_sha256": sha(cr.V1_CODE / "ds_eval.py"), "query_depth_read": False,
            "query_image_read": False, "history_depth_read": False, "history_image_read": False,
            "access_violations": 0, "seconds": round(time.time() - started, 1),
            "max_memory_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2), "ttt_logs": logs,
            "predictions": predictions}
    (out / "equivalence_seal.json").write_text(json.dumps(info, indent=1))
    print("CAJU_EQUIVALENCE_DONE", len(episodes), "episodes", info["seconds"], "s", flush=True)


if __name__ == "__main__":
    main()
