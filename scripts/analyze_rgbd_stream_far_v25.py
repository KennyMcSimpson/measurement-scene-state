#!/usr/bin/env python3
"""V25: Core B V8's sealed states rendered at EVAL-FAR's far queries, against model-free baselines.

  --stage freeze  hash-locks the plan, the protocol, Core B V8's and V17's records and the sources;
  --stage run     computes the 3- and 7-frame reprojection + harmonic baselines and renders every
                  sealed Core B V8 state at the far queries (no query depth is read), seals both,
                  then scores the carrier, the baselines and the hybrids against the far-query depth
                  and writes the analysis and the README.
Run from the repository root under cv_host_run.py with CUDA hidden. No state is built or updated.
"""

from __future__ import annotations

import argparse
import copy
import datetime
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from mcss.dynamic.types import hash_scene_state
from mcss.geometry import transform_cameras
from mcss.measurements import FixedMeasurementRenderer
from mcss.mechanism_pilot.direct_capacity_statistics import describe, paired
from mcss.mechanism_pilot.geometry_carrier_experiment import read
from mcss.mechanism_pilot.rgbd_evidence_carrier import RGBDContext
from mcss.mechanism_pilot.rgbd_stream_evaluation_v3 import preseal_depth_audit
from mcss.mechanism_pilot.rgbd_stream_ttt_eval_v5 import eval_context
from mcss.mechanism_pilot.rgbd_stream_ttt_v4 import StreamRGBDLoader, stream_frame_ids
from mcss.mechanism_pilot.small_training import sha, write_json
from mcss.types import Cameras

EXPERIMENT = "EXP-3D-RGBD-STREAM-FAR-V25"
COREB_V8 = Path("outputs/EXP-3D-RGBD-DYNAMIC-TTT-SOURCE-V8")
V17 = Path("outputs/EXP-3D-RGBD-LONG-TRAIN-V17")
CONCURRENT = (
    "EXP-3D-RGBD-DYNAMIC-TTT-DOSE-V9",
    "EXP-3D-RGBD-DYNAMIC-TTT-CARRIER-V10",
    "EXP-3D-RGBD-DYNAMIC-TTT-ONLINE-V11",
)
PLAN = "PLAN_BEFORE_V25.md"
PLAN_SHA256 = "f525b031610429ad04a112300373e88badce13e7677ca5892b1a3f6c85afd3a9"
COHORT = "EVAL_FAR"
SEEDS = (20260928, 20260929, 20260930)
POLICIES = ("NO_STREAM", "OFF", "OFF_CTX_TTT_100", "TTT", "TTT_100", "TTT_100_WRONG_SCENE")
CHECK_SEED, CHECK_POLICIES = 20260928, ("OFF", "TTT_100")
ABSTAIN_OPACITY = 0.5
CPU_CORES = set(range(32, 40))
THREADS = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "ATEN_CPU_CAPABILITY": "avx2",
    "MKL_CBWR": "AVX2",
    "DNNL_MAX_CPU_ISA": "AVX2",
    "CUDA_VISIBLE_DEVICES": "",
}
CONTRASTS = {
    "HYBRID_VS_MODEL_FREE_FAR": ("REPROJ7_HARMONIC", "HFILL8_7_TTT_100"),
    "TTT_GAIN_IN_HYBRID_FAR": ("HFILL8_7_OFF", "HFILL8_7_TTT_100"),
    "OFF_HYBRID_VS_MODEL_FREE": ("REPROJ7_HARMONIC", "HFILL8_7_OFF"),
    "CARRIER_TTT_100_VS_MODEL_FREE": ("REPROJ7_HARMONIC", "CARRIER_TTT_100"),
    "CARRIER_OFF_VS_MODEL_FREE": ("REPROJ7_HARMONIC", "CARRIER_OFF"),
    "CARRIER_TTT_GAIN": ("CARRIER_OFF", "CARRIER_TTT_100"),
    "CARRIER_TTT_SPECIFICITY": ("CARRIER_TTT_100_WRONG_SCENE", "CARRIER_TTT_100"),
    "CARRIER_CONTEXT_TTT_GAIN": ("CARRIER_OFF", "CARRIER_OFF_CTX_TTT_100"),
    "CARRIER_STREAM_CACHE": ("CARRIER_NO_STREAM", "CARRIER_OFF"),
    "CONTEXT_TTT_IN_HYBRID": ("HFILL8_7_OFF", "HFILL8_7_OFF_CTX_TTT_100"),
    "SPECIFICITY_IN_HYBRID": ("HFILL8_7_TTT_100_WRONG_SCENE", "HFILL8_7_TTT_100"),
    "TTT_GAIN_IN_AHFILL8": ("AHFILL8_7_OFF", "AHFILL8_7_TTT_100"),
    "AHFILL8_VS_MODEL_FREE": ("REPROJ7_HARMONIC", "AHFILL8_7_TTT_100"),
    "TTT_GAIN_IN_HYBRID_NORM": ("HFILL8_7N_OFF", "HFILL8_7N_TTT_100"),
    "HYBRID_NORM_VS_MODEL_FREE": ("REPROJ7_HARMONIC", "HFILL8_7N_TTT_100"),
    "STREAM_FRAMES_FOR_MODEL_FREE": ("REPROJ3_HARMONIC", "REPROJ7_HARMONIC"),
    "CORE_A_HYBRID": ("REPROJ3_HARMONIC", "HFILL8_3_NO_STREAM"),
}
PRIMARY = {
    "P1_HYBRID_VS_MODEL_FREE_FAR_STATUS": "HYBRID_VS_MODEL_FREE_FAR",
    "P2_TTT_GAIN_IN_HYBRID_FAR_STATUS": "TTT_GAIN_IN_HYBRID_FAR",
}
DRAWS, BOOT_SEED = 10000, 20260928
SCRIPTS = Path("scripts")


def module(name, filename):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / filename)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def v3_analysis():
    """Core B V3's scene aggregation and frozen V2 gate (as in Core B V4-V11 and V24)."""
    return module("coreb_v3_analysis_v25", "analyze_rgbd_stream_write_v3.py")


def resolve(path, base):
    path = Path(path)
    return path if path.is_absolute() else Path(base) / path


def hfill(near, harmonic, carrier):
    """HFILL8: the harmonic fill on NEAR pixels, the carrier everywhere else."""
    return np.where(near, harmonic, carrier)


def ahfill(near, opacity, harmonic, carrier, threshold=ABSTAIN_OPACITY):
    """AHFILL8: as HFILL8, and the harmonic fill also where the carrier's opacity is low."""
    return np.where(near | (opacity < threshold), harmonic, carrier)


def absrel(value, gt, mask):
    return float(np.mean(np.abs(value[mask] - gt[mask]) / gt[mask])) if mask.any() else None


def far_record(record):
    """The far queries as the record's primary queries; the near ones stay out of the stream.

    Without this, the stream rule (frames in no context or query role) would pick far frames.
    """
    derived = copy.deepcopy(record)
    roles = derived["roles"]
    roles["query"] = sorted(set(roles["primary_query"]) | set(roles.get("query", ())))
    roles["primary_query"] = list(roles["far_query"])
    return derived


def far_cohort():
    """(manifest, base, {scene: far record}) of EVAL-FAR, the manifest checked against V17's."""
    entry = read(V17 / "training_contract.json")["eval"]["cohorts"][COHORT]
    path = Path(entry["manifest"])
    if sha(path) != entry["manifest_sha256"]:
        raise PermissionError("EVAL-FAR manifest is not V17's")
    manifest = read(path)
    records = {r["scene_id"]: far_record(r) for r in manifest["scenes"] if r["split"] == COHORT}
    return manifest, path.parent, records


def check_streams(records):
    """Every far record's stream frames must be Core B V8's for that scene and role."""
    roles = read(COREB_V8 / "stream_roles.json")["episodes"]
    for sid, record in records.items():
        for role in ("A", "B"):
            ids = stream_frame_ids(record, role)
            if ids is None or list(ids) != roles[f"{sid}/{role}"]:
                raise PermissionError(f"{sid}/{role}: stream frames differ from Core B V8's")
            if set(ids) & set(record["roles"]["primary_query"]):
                raise PermissionError(f"{sid}/{role}: a far query entered the stream")
    return len(records) * 2


def seal_previous(root):
    """Every earlier experiment file, except this round's and the concurrently running ones."""
    files = {}
    for base in (Path("outputs"), Path("docs/experiments")):
        for path in sorted(base.rglob("*")):
            if path.is_file() and not {EXPERIMENT, *CONCURRENT} & set(path.parts):
                files[str(path)] = {"sha256": sha(path), "bytes": path.stat().st_size}
    write_json(
        root / "audit/previous_experiment_seal.json",
        {
            "created_utc": datetime.datetime.now(datetime.UTC).isoformat(),
            "excluded_concurrent": list(CONCURRENT),
            "files": files,
        },
    )
    return len(files)


def check_environment():
    for key, value in THREADS.items():
        if os.environ.get(key) != value:
            raise PermissionError(f"Environment {key} must be {value!r}")
    if not os.sched_getaffinity(0) <= CPU_CORES:
        raise PermissionError("V25 must be pinned to cores 32-39 (taskset)")


def freeze(root):
    root = Path(root).resolve()
    if root.name != EXPERIMENT or (root / "preregistration.json").exists():
        raise PermissionError("Exact, unfrozen V25 directory required")
    if not (root / "PROTOCOL.md").is_file() or sha(root / PLAN) != PLAN_SHA256:
        raise PermissionError("The V25 plan and protocol must be in place")
    check_environment()
    if read(COREB_V8 / "integrity.json")["status"] != "PASS":
        raise PermissionError("Core B V8 must be finalized")
    _, _, records = far_cohort()
    episodes = check_streams(records)
    contract = read(V17 / "training_contract.json")["eval"]
    write_json(
        root / "analysis_contract.json",
        {
            "experiment": EXPERIMENT,
            "states": str(COREB_V8),
            "cohort": COHORT,
            "scenes": len(records),
            "episodes": episodes,
            "policies": list(POLICIES),
            "seeds": list(SEEDS),
            "renderer": "FixedMeasurementRenderer(n_samples=64, ray_chunk_size=2048), CPU",
            "render_check": {"seed": CHECK_SEED, "policies": list(CHECK_POLICIES)},
            "reprojection_fill_constant_m": contract["reprojection_fill_constant_m"],
            "near_px": contract["near_px"],
            "abstain_opacity_threshold": ABSTAIN_OPACITY,
            "thread_environment": THREADS,
            "cpu_affinity": sorted(CPU_CORES),
            "device": "cpu",
        },
    )
    write_json(
        root / "decision_rules.json",
        {
            "aggregation": "per seed: scene mean of role means of far-query AbsRel; seeds equally "
            "weighted (model-free baselines take the same value in every seed)",
            "bootstrap": {"unit": "scene", "draws": DRAWS, "seed": BOOT_SEED},
            "gate": "frozen V2 primary-contrast checks (as Core B V2-V11 and V24)",
            "P1_HYBRID_VS_MODEL_FREE_FAR_STATUS": "gate on AbsRel(REPROJ7_HARMONIC) - "
            "AbsRel(HFILL8_7(TTT_100))",
            "P2_TTT_GAIN_IN_HYBRID_FAR_STATUS": "gate on AbsRel(HFILL8_7(OFF)) - "
            "AbsRel(HFILL8_7(TTT_100))",
            "interpretation": {
                "A": "both SUPPORTED",
                "B": "only P1 SUPPORTED",
                "C": "only P2 SUPPORTED",
                "D": "neither SUPPORTED",
            },
            "descriptive": [c for c in CONTRASTS if c not in PRIMARY.values()],
            "final_holdout": "never opened",
        },
    )
    sealed = seal_previous(root)
    names = ("analysis_contract.json", "decision_rules.json", "PROTOCOL.md", PLAN)
    entry = read(V17 / "training_contract.json")["eval"]["cohorts"][COHORT]
    inputs = [
        COREB_V8 / "integrity.json",
        COREB_V8 / "preregistration.json",
        COREB_V8 / "stream_roles.json",
        COREB_V8 / "raw/eval_ttt_query_results.json",
        V17 / "training_contract.json",
        V17 / "raw/eval" / COHORT / "prediction_seal.json",
        Path(entry["manifest"]),
        root / "audit/previous_experiment_seal.json",
    ]
    for seed in SEEDS:
        out = COREB_V8 / "raw/eval_ttt" / f"seed_{seed}"
        inputs += [out / "integrity.json", out / "state_hashes.json"]
    sources = sorted(Path("src/mcss").rglob("*.py"))
    sources += [
        SCRIPTS / "analyze_rgbd_stream_far_v25.py",
        SCRIPTS / "analyze_rgbd_stream_write_v3.py",
        SCRIPTS / "run_rgbd_geometric_baselines.py",
        SCRIPTS / "run_rgbd_inpaint_baselines.py",
        Path.home() / "cv_tools/cv_host_run.py",
    ]
    write_json(
        root / "preregistration.json",
        {
            "status": "FROZEN_BEFORE_ANALYSIS",
            "experiment": EXPERIMENT,
            "input_sha256": {
                **{str((root / n).resolve()): sha(root / n) for n in names},
                **{str(p.resolve()): sha(p) for p in inputs},
            },
            "source_sha256": {str(p.resolve()): sha(p) for p in sources},
        },
    )
    print("FROZEN_V25", len(records), "scenes", episodes, "episodes", sealed, "files sealed")


def verify_lock(root):
    lock = read(Path(root) / "preregistration.json")
    if lock["status"] != "FROZEN_BEFORE_ANALYSIS":
        raise PermissionError("Preregistration absent")
    for section in ("input_sha256", "source_sha256"):
        for name, digest in lock[section].items():
            if sha(Path(name)) != digest:
                raise PermissionError(f"Frozen input/source changed: {name}")
    return read(Path(root) / "analysis_contract.json")


def compute_baselines(root, config, manifest, base, records, access):
    """3- and 7-frame reprojection + harmonic fill for every far query; no query depth is read."""
    geo = module("v25_geometry", "run_rgbd_geometric_baselines.py")
    inp = module("v25_inpaint", "run_rgbd_inpaint_baselines.py")
    if inp.NEAR_PX != config["near_px"]:
        raise PermissionError("NEAR radius differs from V17's")
    out_dir = Path(root) / "raw/baselines"
    out_dir.mkdir(parents=True)
    ids = tuple(records)
    protected = tuple(manifest.get("protected_scene_ids", []))
    seal = {}
    for sid, record in sorted(records.items()):
        streams = StreamRGBDLoader(
            record,
            manifest["image_size"],
            base,
            "cpu",
            access,
            allowed_scene_ids=ids,
            holdout_scene_ids=protected,
        )
        for role, context in eval_context(record, manifest, base, "cpu", access).items():
            stream = streams.stream(role)
            frames = {
                "3": context,
                "7": RGBDContext([*context, *stream], torch.cat((context.depths, stream.depths))),
            }
            points = {k: geo.context_points(v) for k, v in frames.items()}
            for qid in record["roles"]["primary_query"]:
                camera = geo.query_camera(record, qid, manifest["image_size"])
                cosine = inp.ray_cosine(camera)
                arrays = {}
                for k, value in points.items():
                    depth, hit = geo.reproject(value, camera)
                    arrays[f"near{k}"] = inp.near_mask(hit)
                    arrays[f"harmonic{k}"] = inp.harmonic_fill(
                        depth, hit, cosine, config["reprojection_fill_constant_m"]
                    )
                if (arrays["near3"] & ~arrays["near7"]).any():
                    raise AssertionError("The 7-frame NEAR mask must contain the 3-frame one")
                path = out_dir / f"{sid}_{role}_{qid}.npz"
                np.savez_compressed(path, **arrays)
                seal[f"{sid}/{role}/{qid}"] = {"path": str(path), "sha256": sha(path)}
    return seal


def render_states(root, manifest, records):
    """Every sealed Core B V8 state at the far queries, plus the near-query render check."""
    geo = module("v25_geometry_render", "run_rgbd_geometric_baselines.py")
    renderer = FixedMeasurementRenderer(n_samples=64, ray_chunk_size=2048)
    v8_rows = read(COREB_V8 / "raw/eval_ttt_query_results.json")
    near_records = {
        r["scene_id"]: r
        for name, item in read(COREB_V8 / "cohort_lock.json").items()
        for r in read(Path(item["manifest"]))["scenes"]
    }
    seal, check = {}, []
    for seed in SEEDS:
        source = COREB_V8 / "raw/eval_ttt" / f"seed_{seed}"
        if sha(source / "states.pt") != read(source / "integrity.json")["states_file_sha256"]:
            raise PermissionError("Core B V8 sealed states changed")
        states = torch.load(source / "states.pt", map_location="cpu", weights_only=False)
        index = read(source / "state_hashes.json")
        out_dir = Path(root) / "raw/predictions" / f"seed_{seed}"
        out_dir.mkdir(parents=True)

        def render(state, record, qid, anchor):
            camera = geo.query_camera(record, qid, manifest["image_size"])
            cameras = Cameras(
                camera.intrinsics[None, None], camera.c2w[None, None], camera.image_size
            )
            local = transform_cameras(cameras, torch.linalg.inv(anchor))
            with torch.no_grad():
                pred = renderer(state, local, ("rgb", "depth", "visibility"))
            return pred["depth"][0, 0, 0].numpy(), pred["visibility"][0, 0, 0].numpy()

        for sid, record in sorted(records.items()):
            for role in ("A", "B"):
                for policy in POLICIES:
                    key = f"{sid}/{role}/{policy}"
                    state = states[key]
                    if hash_scene_state(state) != index[key]["state_hash"]:
                        raise AssertionError(f"State hash mismatch: {key}")
                    anchor = torch.tensor(index[key]["anchor_c2w"], dtype=torch.float32)
                    for qid in record["roles"]["primary_query"]:
                        depth, opacity = render(state, record, qid, anchor)
                        path = out_dir / f"{sid}_{role}_{qid}_{policy}.npz"
                        np.savez_compressed(path, depth=depth, opacity=opacity)
                        seal[f"{seed}/{sid}/{role}/{qid}/{policy}"] = {
                            "path": str(path),
                            "sha256": sha(path),
                        }
        if seed == CHECK_SEED:
            for row in v8_rows:
                if row["seed"] != seed or row["policy"] not in CHECK_POLICIES:
                    continue
                key = f"{row['scene_id']}/{row['role']}/{row['policy']}"
                anchor = torch.tensor(index[key]["anchor_c2w"], dtype=torch.float32)
                depth, _ = render(
                    states[key], near_records[row["scene_id"]], row["query_id"], anchor
                )
                sealed = source / row["prediction_path"]
                if sha(sealed) != row["prediction_file_sha256"]:
                    raise PermissionError("A Core B V8 prediction changed")
                check.append(float(np.max(np.abs(depth - np.load(sealed)["depth"]))))
        del states
    return seal, {
        "renders": len(check),
        "max_abs_depth_difference": max(check, default=None),
        "median_max_abs_depth_difference": float(np.median(check)) if check else None,
    }


def v17_agreement(baseline_seal, prediction_seal):
    """3-frame NEAR masks and harmonic fills must equal V17's EVAL-FAR seal; NO_STREAM vs T24000."""
    seal = read(V17 / "raw/eval" / COHORT / "prediction_seal.json")["predictions"]
    compared, worst, render = 0, 0.0, []
    for key, item in baseline_seal.items():
        found = seal.get(f"{COHORT}/{key}")
        if found is None:
            continue
        if sha(Path(found["path"])) != found["sha256"]:
            raise PermissionError("A V17 sealed prediction changed")
        theirs, mine = np.load(found["path"]), np.load(item["path"])
        if not np.array_equal(theirs["near"].astype(bool), mine["near3"]):
            raise AssertionError(f"3-frame NEAR mask differs from V17's: {key}")
        worst = max(worst, float(np.max(np.abs(theirs["REPROJ_HARMONIC"] - mine["harmonic3"]))))
        compared += 1
        sid, role, qid = key.split("/")
        for seed in SEEDS:
            ours = np.load(prediction_seal[f"{seed}/{sid}/{role}/{qid}/NO_STREAM"]["path"])
            render.append(float(np.max(np.abs(ours["depth"] - theirs[f"T24000_{seed}"]))))
    if worst > 1e-9:
        raise AssertionError(f"3-frame harmonic fill differs from V17's: {worst}")
    return {
        "compared": compared,
        "near_identical": compared,
        "harmonic_max_abs_diff": worst,
        "no_stream_vs_v17_t24000_max_abs_depth_difference": max(render, default=None),
        "no_stream_vs_v17_t24000_median_max_abs_depth_difference": (
            float(np.median(render)) if render else None
        ),
    }


def score(records, base, baseline_seal, prediction_seal):
    """Rows of every method on ALL valid far-query pixels and on FAR7, from sealed files only."""
    rows = {"ALL": [], "FAR7": []}
    shares = []
    for sid, record in sorted(records.items()):
        for role in ("A", "B"):
            for qid in record["roles"]["primary_query"]:
                item = baseline_seal[f"{sid}/{role}/{qid}"]
                if sha(Path(item["path"])) != item["sha256"]:
                    raise AssertionError("A sealed baseline changed")
                arrays = dict(np.load(item["path"]))
                frame = next(f for f in record["frames"] if f["frame_id"] == qid)
                gt = np.load(resolve(frame["depth"], base)).squeeze()
                valid = np.isfinite(gt) & (gt > 0)
                near3, near7 = arrays["near3"], arrays["near7"]
                far7 = valid & ~near7
                shares.append(
                    {
                        "near3_share": float((valid & near3).sum() / valid.sum()),
                        "near7_share": float((valid & near7).sum() / valid.sum()),
                    }
                )
                for seed in SEEDS:
                    methods = {
                        "REPROJ3_HARMONIC": arrays["harmonic3"],
                        "REPROJ7_HARMONIC": arrays["harmonic7"],
                    }
                    for policy in POLICIES:
                        entry = prediction_seal[f"{seed}/{sid}/{role}/{qid}/{policy}"]
                        if sha(Path(entry["path"])) != entry["sha256"]:
                            raise AssertionError("A sealed far-query render changed")
                        pred = np.load(entry["path"])
                        depth = pred["depth"].astype(np.float64)
                        opacity = pred["opacity"].astype(np.float64)
                        methods[f"CARRIER_{policy}"] = depth
                        if policy == "NO_STREAM":
                            methods["HFILL8_3_NO_STREAM"] = hfill(near3, arrays["harmonic3"], depth)
                            continue
                        harmonic = arrays["harmonic7"]
                        methods[f"HFILL8_7_{policy}"] = hfill(near7, harmonic, depth)
                        methods[f"AHFILL8_7_{policy}"] = ahfill(near7, opacity, harmonic, depth)
                        normalized = depth / np.maximum(opacity, 1e-3)
                        methods[f"HFILL8_7N_{policy}"] = hfill(near7, harmonic, normalized)
                    for name, value in methods.items():
                        common = {
                            "seed": seed,
                            "cohort": record["origin_split"],
                            "scene_id": sid,
                            "role": role,
                            "query_id": qid,
                            "policy": name,
                        }
                        rows["ALL"].append({**common, "depth_absrel": absrel(value, gt, valid)})
                        if far7.any():
                            rows["FAR7"].append({**common, "depth_absrel": absrel(value, gt, far7)})
    return rows, shares


def analyze(rows, seeds, primary=True):
    """Every contrast with Core B V3's aggregation and the frozen V2 gate; the plan's branch."""
    v3 = v3_analysis()
    policies = sorted({r["policy"] for r in rows})
    absolute = {
        p: describe(v3.scene_values(rows, p), draws=DRAWS, seed=BOOT_SEED)["mean"] for p in policies
    }
    contrasts, statuses = {}, {}
    for name, (a, b) in CONTRASTS.items():
        stats = paired(
            v3.scene_values(rows, a), v3.scene_values(rows, b), draws=DRAWS, seed=BOOT_SEED
        )
        status, _ = v3.gate(stats, True)
        contrasts[name] = {
            "mean": stats["mean"],
            "ci95": stats["ci95"],
            "positive_tied_negative": [stats["positive"], stats["tied"], stats["negative"]],
            "gate_descriptive": status,
        }
        for key, value in PRIMARY.items():
            if value == name:
                statuses[key] = status
    p1 = statuses["P1_HYBRID_VS_MODEL_FREE_FAR_STATUS"] == "SUPPORTED"
    p2 = statuses["P2_TTT_GAIN_IN_HYBRID_FAR_STATUS"] == "SUPPORTED"
    branch = {(True, True): "A", (True, False): "B", (False, True): "C", (False, False): "D"}[
        p1, p2
    ]
    per_seed = {
        str(seed): {
            name: float(np.mean(list(v3.scene_values(rows, a, seed=seed).values())))
            - float(np.mean(list(v3.scene_values(rows, b, seed=seed).values())))
            for name, (a, b) in CONTRASTS.items()
        }
        for seed in seeds
    }
    return {
        "absolute": absolute,
        "contrasts": contrasts,
        "statuses": statuses if primary else {},
        "INTERPRETATION_BRANCH": branch if primary else None,
        "per_seed_means": per_seed,
    }


def stat(value):
    lo, hi = value["ci95"]
    pos, tie, neg = value["positive_tied_negative"]
    return f"{value['mean']:+.4f} [{lo:+.4f}, {hi:+.4f}]，场景 {pos}/{tie}/{neg}"


def sci(value):
    return "—" if value is None else f"{value:.2e}"


def report(root, result):
    main, far = result["primary"], result["far7"]
    c, s = main["contrasts"], main["statuses"]
    share, v17, check = result["region_shares"], result["v17_agreement"], result["render_check"]
    lines = [
        f"# {EXPERIMENT}",
        "",
        "把 Core B V8 封存的状态在 EVAL-FAR 的远处 query 上渲染"
        "（79 个场景，每个场景 2 个远处帧），"
        "与只用同样 7 帧（3 个上下文帧与 4 个流帧）的无模型基线"
        "（重投影 + 调和插值，REPROJ7_HARMONIC）比较。"
        "混合方法 HFILL8_7(p) 在 7 帧 NEAR 像素上用调和插值，"
        "其余像素用策略 p 的 carrier 深度。"
        "只在 CPU 上渲染与计分，不构造新的状态。"
        "计划见 [PLAN_BEFORE_V25.md](PLAN_BEFORE_V25.md)。",
        "",
        "- **P1 HYBRID_VS_MODEL_FREE_FAR（REPROJ7_HARMONIC − HFILL8_7(TTT_100)）：** "
        f"{stat(c['HYBRID_VS_MODEL_FREE_FAR'])}，{s['P1_HYBRID_VS_MODEL_FREE_FAR_STATUS']}",
        "- **P2 TTT_GAIN_IN_HYBRID_FAR（HFILL8_7(OFF) − HFILL8_7(TTT_100)）：** "
        f"{stat(c['TTT_GAIN_IN_HYBRID_FAR'])}，{s['P2_TTT_GAIN_IN_HYBRID_FAR_STATUS']}",
        f"- **分支 {main['INTERPRETATION_BRANCH']}**",
        f"- **像素比例：** 3 帧 NEAR 占远处 query 有效像素的 {100 * share['near3_share']:.1f}%，"
        f"7 帧 NEAR 占 {100 * share['near7_share']:.1f}%。",
        "",
        "| 方法 | 全部像素 AbsRel | FAR7 AbsRel |",
        "|---|---|---|",
    ]
    for name in sorted(main["absolute"]):
        value = far["absolute"].get(name)
        far_text = f"{value:.4f}" if value is not None else "—"
        lines.append(f"| {name} | {main['absolute'][name]:.4f} | {far_text} |")
    lines += ["", "| 对比（正值：后者更好） | 全部像素 | FAR7 |", "|---|---|---|"]
    for name in CONTRASTS:
        lines.append(f"| {name} | {stat(c[name])} | {stat(far['contrasts'][name])} |")
    lines += [
        "",
        "- **各来源队列（P1 / P2）：** "
        + "；".join(
            f"{cohort} {stat(part['contrasts']['HYBRID_VS_MODEL_FREE_FAR'])} / "
            f"{stat(part['contrasts']['TTT_GAIN_IN_HYBRID_FAR'])}"
            for cohort, part in result["per_cohort"].items()
        ),
        "- **每个 seed（P1 / P2）：** "
        + "；".join(
            f"{k}: {v['HYBRID_VS_MODEL_FREE_FAR']:+.4f} / {v['TTT_GAIN_IN_HYBRID_FAR']:+.4f}"
            for k, v in main["per_seed_means"].items()
        ),
        "- **核对：** 3 帧 NEAR 掩码与调和插值与 V17 的 EVAL-FAR 封存比较 "
        f"{v17['compared']} 个，完全相同；"
        "NO_STREAM 的渲染与 V17 第 24000 步的封存预测，逐图最大深度差的中位数为 "
        f"{sci(v17['no_stream_vs_v17_t24000_median_max_abs_depth_difference'])}；"
        f"在近处 query 上重渲染 V8 的 {check['renders']} 个状态（CPU），"
        "与 V8 的 GPU 预测相比，"
        f"逐图最大深度差的中位数为 {sci(check['median_max_abs_depth_difference'])}，"
        f"最大为 {sci(check['max_abs_depth_difference'])}。",
        "",
        "说明：EVAL-FAR 的场景与 EVAL-V3/V4 相同，"
        "远处帧此前只在 Core A（V13、V17–V22）中用作 query；"
        "Core B 的状态从未在这些帧上计算过。受保护的 final holdout 未打开。",
    ]
    (Path(root) / "README.md").write_text("\n".join(lines) + "\n")


def run(root):
    root = Path(root).resolve()
    config = verify_lock(root)
    check_environment()
    torch.set_num_threads(1)
    manifest, base, records = far_cohort()
    check_streams(records)
    access = []
    baseline_seal = compute_baselines(root, config, manifest, base, records, access)
    prediction_seal, render_check = render_states(root, manifest, records)
    preseal_depth_audit(access, records)
    write_json(
        root / "raw/seal.json",
        {
            "query_depth_read": False,
            "baselines": baseline_seal,
            "predictions": prediction_seal,
            "created_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        },
    )
    write_json(root / "raw/input_access.json", access)
    agreement = v17_agreement(baseline_seal, prediction_seal)
    rows, shares = score(records, base, baseline_seal, prediction_seal)
    seeds = list(SEEDS)
    result = {
        "primary": analyze(rows["ALL"], seeds),
        "far7": analyze(rows["FAR7"], seeds, primary=False),
        "per_cohort": {
            name: analyze([r for r in rows["ALL"] if r["cohort"] == name], seeds, primary=False)
            for name in sorted({r["cohort"] for r in rows["ALL"]})
        },
        "region_shares": {k: float(np.mean([s[k] for s in shares])) for k in shares[0]},
        "v17_agreement": agreement,
        "render_check": render_check,
    }
    result["INTERPRETATION_BRANCH"] = result["primary"]["INTERPRETATION_BRANCH"]
    write_json(root / "v25_results.json", result)
    for name, values in rows.items():
        write_json(root / f"raw/{name.lower()}_rows.json", values)
    old = read(root / "audit/previous_experiment_seal.json")["files"]
    changed = [p for p, info in old.items() if sha(Path(p)) != info["sha256"]]
    if changed:
        raise AssertionError(f"Old experiment files changed: {changed[:3]}")
    write_json(
        root / "integrity.json",
        {
            "status": "PASS",
            "rows_all": len(rows["ALL"]),
            "rows_far7": len(rows["FAR7"]),
            "renders": len(prediction_seal),
            "old_files_unchanged": len(old),
            "states_built_or_updated": False,
            "final_holdout_touched": False,
            "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "finished_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        },
    )
    report(root, result)
    print(
        "V25_COMPLETE",
        result["INTERPRETATION_BRANCH"],
        json.dumps(result["primary"]["statuses"]),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--stage", choices=("freeze", "run"), required=True)
    args = parser.parse_args()
    if args.stage == "freeze":
        freeze(args.root)
    else:
        run(args.root)
    sys.exit(0)
