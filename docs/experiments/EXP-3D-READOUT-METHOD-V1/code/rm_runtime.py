"""EXP-3D-READOUT-METHOD-V1: runtime profile of every stage, on the first 5 ARKIT_SYSTEM4 episodes (2 queries each;
the first sample of every stage is a warm-up and is left out of its median). Arrived frames and query cameras only,
no ground truth.

  gpu   (DepthSplat environment, from the DepthSplat repository root, GPU 4, CUDA synchronized around each stage)
        per scene: DepthSplat 7-view encode with measured depth injected (the CACHE state); 10 and 100 Adam steps of
                   V1's ttt; Gaussian render at both query cameras
        per query: KF table weights (CPU, on the sealed R7 of that query from the cache); V2Net (GPU and CPU);
                   PixelMLP 8-150-150-150 (GPU); SysNet 14-ch (GPU)
  r7    (the project's .venv through cv_host_run.py from the measurement-scene-state root, CUDA hidden, one thread,
        exactly as the frozen geometry stage ran it)
        per scene: reading the 7 arrived RGB-D frames; unprojecting them (frame_points)
        per query: 7-frame 1-pixel z-buffer + harmonic fill + NEAR mask; checked against the sealed R7 in the cache
  merge writes RUNTIME_PROFILE.json from results/runtime_gpu.json and results/runtime_r7.json

  python rm_runtime.py gpu|r7|merge ROOT
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

MODE, ROOT = sys.argv[1], Path(sys.argv[2]).resolve()
OUT = ROOT.parent
S4 = OUT / "EXP-3D-CAJU3D-SYSTEM4-HOLDOUT-V1"
EPISODES = 5
CONSTANT = 2.3736329078674316


def episodes():
    lock = json.loads((S4 / "frame_role_lock.json").read_text())["cohorts"]["ARKIT_SYSTEM4"]
    records = {r["scene_id"]: r for r in json.loads(Path(lock["manifest"]).read_text())["scenes"]}
    return [(records[e["scene_id"]], e) for e in lock["episodes"][:EPISODES]]


def cache():
    arrays = np.load(ROOT / "cache" / "ARKIT_SYSTEM4.npz")
    rows = json.loads((ROOT / "cache" / "ARKIT_SYSTEM4.json").read_text())["queries"]
    return arrays, {r["key"]: i for i, r in enumerate(rows)}


def summary(samples):
    return {k: float(np.median(v[1:])) for k, v in samples.items()}


def gpu():
    import torch
    for p in ("EXP-3D-RGBD-DEPTHSPLAT-JEPA-TTT-V1/code", "EXP-3D-CAJU-HEADROOM-V1/code", "EXP-3D-CAJU3D-ENGINE-V1/code",
              "EXP-3D-CAJU3D-KFUSE-V1/code", "EXP-3D-CAJU3D-FUSENET-V1/code", "EXP-3D-CAJU3D-SYSTEM4-HOLDOUT-V1/code"):
        sys.path.insert(0, str(OUT / p))
    sys.path.insert(0, str(ROOT / "code"))
    import ds_eval as v1
    import ds_predict
    import rm_methods as M
    import s4_v2 as V2
    from ds_predict import INJECTED, build, inject_measured_depth
    from train_jepa import set_scale_max

    def timed(fn, sync=True):
        if sync:
            torch.cuda.synchronize()
        t = time.perf_counter()
        out = fn()
        if sync:
            torch.cuda.synchronize()
        return out, time.perf_counter() - t

    device = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    set_scale_max(0.1)
    encoder, decoder = build(device, Path(ds_predict.CHECKPOINT))
    inject_measured_depth(encoder)
    INJECTED["align"] = False
    tables = json.loads((OUT / "EXP-3D-CAJU3D-KFUSE-V1/fusion_tables.json").read_text())
    arrays, index = cache()
    v2, v2_cpu = V2.V2Net().to(device).eval(), V2.V2Net().eval()
    mlp, sysnet = M.PixelMLP((150, 150, 150)).to(device).eval(), M.SysNet(14).to(device).eval()
    scene = {k: [] for k in ("encode_s", "ttt10_s", "ttt100_s", "render_both_queries_s")}
    query = {k: [] for k in ("kf_table_cpu_s", "v2net_gpu_s", "v2net_cpu_s", "pixel_mlp_gpu_s", "sysnet_gpu_s")}
    log = []
    for rec, e in episodes():
        frames, inputs = rec["frames"], e["context"] + e["stream"]
        cams = v1.cameras(frames, e["queries"], device)
        cached, t_enc = timed(lambda: v1.encode(encoder, frames, inputs, device))
        _, t10 = timed(lambda: v1.ttt(decoder, cached, frames, inputs, 10, 1.0, device))
        out, t100 = timed(lambda: v1.ttt(decoder, cached, frames, inputs, 100, 1.0, device))
        state = out[0] if isinstance(out, tuple) else out
        (dz, al), t_r = timed(lambda: v1.query_render(decoder, state, cams))
        for k, v in zip(scene, (t_enc, t10, t100, t_r)):
            scene[k].append(round(v, 5))
        for j, qid in enumerate(e["queries"]):
            i = index[f"{e['scene_id']}/{e['role']}/{qid}"]
            r7, near7, cosine = arrays["r7"][i], arrays["near7"][i], arrays["cosine"][i]
            depth_z, alpha = np.asarray(dz[j], dtype=np.float32), np.asarray(al[j], dtype=np.float32)
            _, t_kf = timed(lambda: V2.kf_weight_map(depth_z, alpha, r7, near7, cosine, tables), sync=False)
            x8, x14 = torch.randn(1, 8, 128, 160, device=device), torch.randn(1, 14, 128, 160, device=device)
            with torch.no_grad():
                for _ in range(3):  # warm-up
                    v2(x8), mlp(x8), sysnet(x14)
                _, t_v2 = timed(lambda: v2(x8))
                _, t_v2c = timed(lambda: v2_cpu(x8.cpu()), sync=False)
                _, t_mlp = timed(lambda: mlp(x8))
                _, t_sys = timed(lambda: sysnet(x14))
            for k, v in zip(query, (t_kf, t_v2, t_v2c, t_mlp, t_sys)):
                query[k].append(round(v, 5))
        log.append(e["scene_id"] + "/" + e["role"])
        print(log[-1], {k: v[-1] for k, v in {**scene, **query}.items()}, flush=True)
    res = {"device": torch.cuda.get_device_name(0), "torch_threads_cpu": torch.get_num_threads(), "episodes": log,
           "per_scene_median_s": summary(scene), "per_query_median_s": summary(query),
           "per_scene_samples_s": scene, "per_query_samples_s": query}
    (ROOT / "results" / "runtime_gpu.json").write_text(json.dumps(res, indent=1) + "\n")
    print("RUNTIME_GPU_DONE", json.dumps(res["per_scene_median_s"]), json.dumps(res["per_query_median_s"]))


def r7():
    import torch
    torch.set_num_threads(1)
    from mcss.measurement_planning import geometry as g
    from mcss.measurement_planning.information import QUERY_CAMERA_PURPOSE, TEACHER_PURPOSE, TeacherAccess
    arrays, index = cache()
    scene = {k: [] for k in ("read_7_frames_s", "unproject_7_frames_s")}
    query = {k: [] for k in ("reproject_harmonic_near_s",)}
    log, purposes, max_diff = [], set(), 0.0
    for rec, e in episodes():
        access = []
        teacher = TeacherAccess(rec, e["queries"], (128, 160), Path.cwd(), access)
        t = time.perf_counter()
        ctx, stream = teacher.frames(e["context"]), teacher.frames(e["stream"])
        t_read = time.perf_counter() - t
        t = time.perf_counter()
        points = {**g.frame_points(ctx), **g.frame_points(stream)}
        t_points = time.perf_counter() - t
        scene["read_7_frames_s"].append(round(t_read, 5))
        scene["unproject_7_frames_s"].append(round(t_points, 5))
        for qid in e["queries"]:
            t = time.perf_counter()
            camera = teacher.query_camera(qid)
            cosine = g.ray_cosine(camera)
            r7_map, hit7 = g.harmonic(g.union([g.reproject(points[f], camera) for f in e["context"] + e["stream"]]),
                                      cosine, CONSTANT)
            g.near_mask(hit7)
            query["reproject_harmonic_near_s"].append(round(time.perf_counter() - t, 5))
            sealed = arrays["r7"][index[f"{e['scene_id']}/{e['role']}/{qid}"]].astype(np.float64)
            max_diff = max(max_diff, float(np.max(np.abs(np.asarray(r7_map, dtype=np.float64) - sealed) / sealed)))
        purposes |= {a["purpose"] for a in access}
        log.append(e["scene_id"] + "/" + e["role"])
        print(log[-1], scene["unproject_7_frames_s"][-1], query["reproject_harmonic_near_s"][-2:], flush=True)
    if purposes - {TEACHER_PURPOSE, QUERY_CAMERA_PURPOSE}:
        raise PermissionError(f"unexpected reads: {sorted(purposes)}")
    res = {"threads": 1, "episodes": log, "read_purposes": sorted(purposes),
           "max_relative_difference_vs_sealed_r7": max_diff,
           "per_scene_median_s": summary(scene), "per_query_median_s": summary(query),
           "per_scene_samples_s": scene, "per_query_samples_s": query}
    (ROOT / "results" / "runtime_r7.json").write_text(json.dumps(res, indent=1) + "\n")
    print("RUNTIME_R7_DONE", json.dumps(res["per_scene_median_s"]), json.dumps(res["per_query_median_s"]), max_diff)


def merge():
    gp = json.loads((ROOT / "results" / "runtime_gpu.json").read_text())
    rp = json.loads((ROOT / "results" / "runtime_r7.json").read_text())
    s = {**gp["per_scene_median_s"], **rp["per_scene_median_s"]}
    q = {**gp["per_query_median_s"], **rp["per_query_median_s"]}
    geometry = s["read_7_frames_s"] + s["unproject_7_frames_s"] + 2 * q["reproject_harmonic_near_s"]
    ff = s["encode_s"] + s["render_both_queries_s"]
    ttt = ff + s["ttt100_s"]
    kf = 2 * q["kf_table_cpu_s"]
    systems = {  # one scene with 2 queries, seconds; readout networks on the GPU. B0-B7 need the TTT state because the
        # KF table weights F7 / alpha of the TTT render on covered pixels; SYS_A needs no DepthSplat at all.
        "R7": geometry,
        "SYS_A_geometry_only_network": geometry + 2 * q["sysnet_gpu_s"],
        "CORE_A_feed_forward_plus_R7": ff + geometry,
        "SYS_B_feed_forward_plus_network": ff + geometry + 2 * q["sysnet_gpu_s"],
        "CORE_B_and_B1-B5_TTT100_plus_KF_table": ttt + geometry + kf,
        "B6_B6W_TTT100_plus_KF_table_plus_pixel_MLP": ttt + geometry + kf + 2 * q["pixel_mlp_gpu_s"],
        "B7_V2_TTT100_plus_KF_table_plus_V2Net": ttt + geometry + kf + 2 * q["v2net_gpu_s"],
        "SYS_C_TTT100_plus_network": ttt + geometry + 2 * q["sysnet_gpu_s"],
        "SYS_C_with_TTT10_not_evaluated": ff + s["ttt10_s"] + geometry + 2 * q["sysnet_gpu_s"]}
    out = {"device": gp["device"], "episodes": gp["episodes"],
           "note": "single process on GPU 4, which other processes may share; every median drops the first sample of "
                   "its stage (warm-up), i.e. per-scene medians over episodes 2-5 and per-query medians over queries "
                   "2-10; R7 timed on one CPU thread in the project environment, as the frozen geometry stage ran it "
                   "(16 such workers in parallel); network times use random inputs of the deployed shape",
           "per_scene_median_s": s, "per_query_median_s": q, "scene_with_2_queries_total_s": systems,
           "ttt100_share_of_SYS_C": s["ttt100_s"] / systems["SYS_C_TTT100_plus_network"],
           "SYS_C_over_SYS_B_time_ratio": systems["SYS_C_TTT100_plus_network"] / systems["SYS_B_feed_forward_plus_network"],
           "r7_check": {"max_relative_difference_vs_sealed_r7": rp["max_relative_difference_vs_sealed_r7"],
                        "read_purposes": rp["read_purposes"]},
           "sources": ["results/runtime_gpu.json", "results/runtime_r7.json"]}
    (ROOT / "RUNTIME_PROFILE.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


{"gpu": gpu, "r7": r7, "merge": merge}[MODE]()
