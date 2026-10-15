"""EXP-3D-CAJU3D-KFUSE-V1: the reverse H-JEPA's prediction at the query cameras (GPU; no TTT, no ground truth).

Per episode, the frozen GeometricReverseJEPA of EXP-3D-CAJU3D-JEPAGATE-V1 encodes the 7 arrived views (RGB + measured
depth), warps their latent cells into each query camera (intrinsics and pose only) and predicts (z_hat, log P) there.
The target encoder is never run: nothing of a query frame but its camera enters. Saved per query (128 x 160, nearest
upsampling of the 64 x 80 cells): log_p (float32) and hit (uint8, a source cell landed in the query cell).
Every data file opened during an episode is audited (arrived frames only).
Run with the DepthSplat environment from the DepthSplat repository root (PYTHONPATH=.):
  python kf_query_jepa.py ROOT OUT --cohorts A B [--shard k/n] [--limit N]
"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
OUTPUTS = HERE.parents[1]
GATE_CODE = OUTPUTS / "EXP-3D-CAJU3D-JEPAGATE-V1" / "code"
for _path in (OUTPUTS / "EXP-3D-RGBD-DEPTHSPLAT-JEPA-TTT-V1" / "code", OUTPUTS / "EXP-3D-CAJU-HEADROOM-V1" / "code",
              GATE_CODE, HERE):
    sys.path.insert(0, str(_path))
import ds_eval as v1  # noqa: E402
import ds_predict  # noqa: E402
import hjepa as J  # noqa: E402
from caju_logic import AccessAudit  # noqa: E402
from ds_finetune import measured_z  # noqa: E402
from ds_predict import local, views  # noqa: E402
from kf_jepa_core import predict_at  # noqa: E402

MODEL = OUTPUTS / "EXP-3D-CAJU3D-JEPAGATE-V1" / "model" / "hjepa_final.pth"
AUDIT = AccessAudit()
_np_load, _image_open = np.load, PIL.Image.open


def _audited_np_load(file, *args, **kwargs):
    if isinstance(file, (str, Path)):
        AUDIT.note(file)
    return _np_load(file, *args, **kwargs)


def _audited_image_open(fp, *args, **kwargs):
    if isinstance(fp, (str, Path)):
        AUDIT.note(fp)
    return _image_open(fp, *args, **kwargs)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def camera(rows, ids, device):
    k = torch.tensor(np.array([rows[i]["intrinsics"] for i in ids], dtype=np.float32), device=device)
    c2w = torch.tensor(np.array([ds_predict.orthonormal(np.array(rows[i]["c2w"], dtype=np.float64)) for i in ids],
                                dtype=np.float32), device=device)
    return k, c2w


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--cohorts", nargs="+", required=True)
    parser.add_argument("--shard", default="0/1")
    parser.add_argument("--limit", type=int, default=0, help="rehearsal only: first N episodes")
    args = parser.parse_args()
    index, count = (int(x) for x in args.shard.split("/"))
    root = args.root.resolve()
    locks = json.loads((root / "frame_role_lock.json").read_text())["cohorts"]
    episodes, records = [], {}
    for cohort in args.cohorts:
        lock = locks[cohort]
        if sha(lock["manifest"]) != lock["manifest_sha256"]:
            raise PermissionError(f"{cohort}: manifest changed since the frame-role lock")
        for r in json.loads(Path(lock["manifest"]).read_text())["scenes"]:
            records[(cohort, r["scene_id"])] = r
        episodes += [dict(e, cohort=cohort) for e in lock["episodes"]]
    episodes = (episodes[: args.limit] if args.limit else episodes)[index::count]
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    device = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    jepa = J.GeometricReverseJEPA()
    jepa.load_state_dict(torch.load(MODEL, map_location="cpu", weights_only=False)["state_dict"])
    jepa = jepa.to(device).eval()
    np.load, PIL.Image.open = _audited_np_load, _audited_image_open
    started, predictions = time.time(), {}
    hit_share = []
    for n, e in enumerate(episodes):
        frames = records[(e["cohort"], e["scene_id"])]["frames"]
        rows = {f["frame_id"]: f for f in frames}
        inputs, queries = e["context"] + e["stream"], e["queries"]
        if set(inputs) & set(queries):
            raise PermissionError("a query frame would be a source")
        AUDIT.begin([local(rows[i][k]) for i in inputs for k in ("rgb", "depth")])
        rgb = views(frames, inputs, None, v1.NEAR, v1.FAR, device)["image"][0]
        planar = measured_z(frames, inputs).to(device)
        k, c2w = camera(rows, inputs, device)
        qk, qc2w = camera(rows, queries, device)
        _, log_p, hit = predict_at(jepa, rgb, planar * J.ray_factor(k, device=device), k, c2w, qk, qc2w)
        log_p = F.interpolate(log_p, size=(J.H, J.W), mode="nearest")[:, 0].cpu().numpy().astype(np.float32)
        hit = F.interpolate(hit, size=(J.H, J.W), mode="nearest")[:, 0].cpu().numpy().astype(np.uint8)
        for j, qid in enumerate(queries):
            path = out / f"{e['scene_id']}_{e['role']}_{qid}.npz"
            np.savez_compressed(path, log_p=log_p[j], hit=hit[j])
            predictions[f"{e['scene_id']}/{e['role']}/{qid}"] = {"path": str(path), "sha256": sha(path),
                                                                 "cohort": e["cohort"]}
            hit_share.append(float(hit[j].mean()))
        violations = AUDIT.violations()
        if violations:
            raise PermissionError(f"{e['scene_id']}/{e['role']}: files outside the arrived frames: {violations[:3]}")
        if n == 0 or (n + 1) % 25 == 0:
            print(f"episode {n + 1}/{len(episodes)} {time.time() - started:.0f}s", flush=True)
    np.load, PIL.Image.open = _np_load, _image_open
    info = {"cohorts": args.cohorts, "shard": args.shard, "limit": args.limit, "episodes": len(episodes),
            "queries": len(predictions), "jepa_model": str(MODEL), "jepa_model_sha256": sha(MODEL),
            "kf_query_jepa_sha256": sha(Path(__file__)), "kf_jepa_core_sha256": sha(HERE / "kf_jepa_core.py"),
            "hjepa_sha256": sha(GATE_CODE / "hjepa.py"),
            "query_media_read": False, "query_depth_read": False, "access_violations": 0,
            "mean_query_hit_share": float(np.mean(hit_share)) if hit_share else None,
            "seconds": round(time.time() - started, 1),
            "max_memory_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2), "predictions": predictions}
    (out / f"kf_jepa_seal_{index}of{count}.json").write_text(json.dumps(info, indent=1))
    print("KF_JEPA_DONE", len(episodes), "episodes", info["seconds"], "s", info["max_memory_gb"], "GB", flush=True)


if __name__ == "__main__":
    main()
