#!/usr/bin/env python3
"""V16: does the carrier's completion advantage survive on volume-disjoint scenes? (no training)

The frozen TRAIN72 carriers (V11 C1 primary, V12 C1 secondary) and the TRAIN72+EXT carriers of V14
and V15 (descriptive) render FRESH-V2, whose 10 volumes hold no TRAIN72 or DEV scene, with V13's
code path. The completion advantage A = FAR AbsRel(REPROJ_HARMONIC) - FAR AbsRel(V11 C1) on FRESH-V2
is compared with V13's sealed and scored EVAL-V3/V4 rows (volumes shared with TRAIN72), as fixed by
PROTOCOL.md. Stages: `lock`; `seal` (replay of two sealed EVAL-V3 scenes for every carrier family,
then every FRESH-V2 prediction hashed before any query depth is read); `score`; `analyze`.
"""
# ruff: noqa: E501 -- Chinese scientific report prose

from __future__ import annotations

import argparse
import datetime
import functools
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

from mcss.dynamic.types import hash_scene_state
from mcss.geometry import transform_cameras
from mcss.measurements import FixedMeasurementRenderer
from mcss.mechanism_pilot import rgbd_completion_carrier as v11_carrier
from mcss.mechanism_pilot import rgbd_data_scale_carrier as v14_carrier
from mcss.mechanism_pilot import rgbd_wide_scale_carrier as v15_carrier
from mcss.mechanism_pilot import rgbd_width_carrier as v12_carrier
from mcss.mechanism_pilot.readout_reanalysis import depth_metrics
from mcss.mechanism_pilot.rgbd_depth_bounds import context_depth_bounds
from mcss.types import Cameras

EXPERIMENT = "EXP-3D-RGBD-VOLUME-DISJOINT-V16"
FRESH = Path("outputs/EXP-3D-RGBD-FRESH-V2-DATA/manifest_fresh_v2.json")
EVAL_V3 = Path("outputs/EXP-3D-RGBD-EVAL-V3-DATA/manifest_eval_v3.json")
V11 = Path("outputs/EXP-3D-RGBD-COMPLETION-V11")
V12 = Path("outputs/EXP-3D-RGBD-WIDTH-V12")
V13 = Path("outputs/EXP-3D-RGBD-VOLUME-V13")
V14 = Path("outputs/EXP-3D-RGBD-DATA-SCALE-V14")
V15 = Path("outputs/EXP-3D-RGBD-WIDTH-DATA-V15")
INPAINT = Path("outputs/EXP-3D-RGBD-INPAINT-BASELINES-V1")
# family -> (carrier module, selected checkpoints, sealed EVAL-V3 replay: seal file, key prefix, array prefix)
FAMILIES = {
    "V11C1": (
        v11_carrier,
        V11 / "selected_checkpoints.json",
        (V11 / "raw/eval_v3/prediction_seal.json", "", "C1"),
    ),
    "V12C1": (
        v12_carrier,
        V12 / "selected_checkpoints.json",
        (V15 / "raw/eval/prediction_seal.json", "EVAL_V3/", "W64T72"),
    ),
    "V14C1": (
        v14_carrier,
        V14 / "selected_checkpoints.json",
        (V14 / "raw/eval/prediction_seal.json", "EVAL_V3/", "C1"),
    ),
    "V15C1": (
        v15_carrier,
        V15 / "selected_checkpoints.json",
        (V15 / "raw/eval/prediction_seal.json", "EVAL_V3/", "C1"),
    ),
}
TRAIN72_FAMILIES = ("V11C1", "V12C1")
THRESHOLD = 0.5
REGIONS = ("ALL", "NEAR", "FAR")
REPLAY_SCENES = 2
DRAWS, SEED = 10000, 20260928
SCRIPTS = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def read(path):
    return json.loads(Path(path).read_text())


def module(name, filename):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / filename)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


@functools.cache
def host():
    """V11's frozen EVAL-V3 module: context loader, aggregation, bootstrap and V2 gate."""
    return module("v16_v11_eval", "evaluate_rgbd_completion_eval_v3.py")


@functools.cache
def inpainting():
    """INPAINT-V1's frozen module: ray cosine, harmonic filling, NEAR mask, label rule."""
    return module("v16_inpaint", "run_rgbd_inpaint_baselines.py")


def inputs(root):
    paths = {
        "protocol": root / "PROTOCOL.md",
        "script": Path(__file__).resolve(),
        "v11_eval_script": SCRIPTS / "evaluate_rgbd_completion_eval_v3.py",
        "inpaint_script": SCRIPTS / "run_rgbd_inpaint_baselines.py",
        "geometric_script": SCRIPTS / "run_rgbd_geometric_baselines.py",
        "fresh_v2_manifest": FRESH,
        "v11_contract": V11 / "training_contract.json",
        "v13_query_rows": V13 / "raw/query_rows.json",
        "v14_contract": V14 / "training_contract.json",
    }
    for family, (_, selected, (seal, _, _)) in FAMILIES.items():
        paths[f"{family}_selected"] = selected
        paths[f"{family}_replay_seal"] = seal
    return paths


def lock(root):
    root = Path(root).resolve()
    if root.name != EXPERIMENT:
        raise PermissionError("Exact experiment directory required")
    if (root / "lock.json").exists():
        raise FileExistsError("Never re-lock")
    if not (root / "PROTOCOL.md").is_file():
        raise PermissionError("Protocol text required before locking")
    if (root / "raw").exists():
        raise PermissionError("No prediction may exist before the lock")
    if read(V15 / "integrity.json")["status"] != "PASS":
        raise PermissionError("V15 must be finalized with integrity PASS")
    write(
        root / "lock.json",
        {
            "status": "FROZEN_BEFORE_ANY_V16_PREDICTION",
            "experiment": EXPERIMENT,
            "created_utc": datetime.datetime.now(datetime.UTC).isoformat(),
            "primary": "FRESH-V2 A = FAR AbsRel(REPROJ_HARMONIC) - FAR AbsRel(V11 C1) (INPAINT-V1 label rule); P2 Delta = mean A(EVAL-V3/V4, V13 rows) - mean A(FRESH-V2), independent scene bootstraps",
            "input_sha256": {
                name: {"path": str(Path(p).resolve()), "sha256": sha(p)}
                for name, p in inputs(root).items()
            },
            "fresh_v2_use": "third use (V7 qualification, GB-V1 geometric baselines); mechanism only",
            "final_holdout_opened": False,
        },
    )
    print("LOCKED", EXPERIMENT, flush=True)


def verify(root):
    lock_ = read(root / "lock.json")
    for name, entry in lock_["input_sha256"].items():
        if sha(entry["path"]) != entry["sha256"]:
            raise PermissionError(f"Locked input changed: {name}")
    return lock_


def carriers():
    """{(family, seed): (model, carrier module)} with every selected checkpoint hash-checked."""
    seeds = read(V11 / "training_contract.json")["seeds"]
    models = {}
    for family, (carrier, selected_path, _) in FAMILIES.items():
        selected = read(selected_path)["C1"]
        for seed in seeds:
            item = selected[str(seed)]
            if sha(Path(item["path"])) != item["sha256"]:
                raise PermissionError(f"Selected {family} checkpoint changed")
            models[family, seed] = (carrier.load_checkpoint(item["path"], "cpu")[0].eval(), carrier)
    return seeds, models


@torch.no_grad()
def predict(record, manifest, base, models, constant, access, tag, states=None):
    """Every sealed array of every primary query of one scene (V13's code path, every family)."""
    stage, inp = host(), inpainting()
    geo = stage.geometry()
    renderer = FixedMeasurementRenderer(n_samples=64, ray_chunk_size=2048)
    out, sid = {}, record["scene_id"]
    for role, context in stage.context_of(record, manifest, base, access).items():
        anchor = context[0].camera.c2w
        bounds = context_depth_bounds(context, anchor)
        points = geo.context_points(context)
        built = {}
        for (family, seed), (model, carrier) in models.items():
            state, state_anchor = carrier.build_state(model, context, bounds, f"{tag}:{sid}:{role}")
            built[family, seed] = (state, state_anchor)
            if states is not None:
                states[f"{family}:{seed}:{sid}/{role}"] = hash_scene_state(state)
        for qid in record["roles"]["primary_query"]:
            access.append({"scene_id": sid, "frame_id": qid, "purpose": "QUERY_CAMERA_ONLY"})
            camera = geo.query_camera(record, qid, manifest["image_size"])
            depth, hit = geo.reproject(points, camera)
            reproj = geo.fill_nearest(depth, hit, constant) if hit.any() else None
            if reproj is None:
                reproj = np.full(camera.image_size, constant, dtype=np.float64)
            arrays = {
                "REPROJ_NN": reproj,
                "hit": hit,
                "near": inp.near_mask(hit),
                "REPROJ_HARMONIC": inp.harmonic_fill(depth, hit, inp.ray_cosine(camera), constant),
            }
            batched = Cameras(
                camera.intrinsics[None, None], camera.c2w[None, None], camera.image_size
            )
            for (family, seed), (state, state_anchor) in built.items():
                local = transform_cameras(batched, torch.linalg.inv(state_anchor))
                pred = renderer(state, local, ("depth", "visibility"))
                arrays[f"{family}_{seed}"] = pred["depth"][0, 0, 0].numpy().astype(np.float64)
                arrays[f"OPACITY_{family}_{seed}"] = pred["visibility"][0, 0, 0].numpy()
            out[role, qid] = arrays
    return out


def replay(models, constant):
    """Every family on the first EVAL-V3 scenes must reproduce its own sealed predictions."""
    manifest = read(EVAL_V3)
    seals = {family: read(seal)["predictions"] for family, (_, _, (seal, _, _)) in FAMILIES.items()}
    inpaint_seal = read(INPAINT / "raw/inpaint_seal.json")
    worst, compared = 0.0, 0
    for record in manifest["scenes"][:REPLAY_SCENES]:
        sid = record["scene_id"]
        mine = predict(record, manifest, EVAL_V3.parent, models, constant, [], "eval_v3")
        for (role, qid), arrays in mine.items():
            filled = dict(np.load(inpaint_seal[f"EVAL_V3/{sid}/{role}/{qid}"]["path"]))
            worst = max(
                worst, float(np.max(np.abs(arrays["REPROJ_HARMONIC"] - filled["REPROJ_HARMONIC"])))
            )
            for family, (_, _, (_, prefix, array)) in FAMILIES.items():
                sealed = dict(np.load(seals[family][f"{prefix}{sid}/{role}/{qid}"]["path"]))
                for seed in {s for (f, s) in models if f == family}:
                    worst = max(
                        worst,
                        float(
                            np.max(np.abs(arrays[f"{family}_{seed}"] - sealed[f"{array}_{seed}"]))
                        ),
                    )
                    compared += 1
    if worst > 1e-6:
        raise AssertionError(f"Replay of the sealed EVAL-V3 predictions failed: {worst}")
    return {"scenes": REPLAY_SCENES, "arrays_compared": compared, "max_abs_deviation": worst}


def seal(root):
    root = Path(root).resolve()
    verify(root)
    out = root / "raw"
    if (out / "prediction_seal.json").exists():
        raise FileExistsError("Predictions are sealed once")
    seeds, models = carriers()
    constant = read(V11 / "training_contract.json")["eval_v3"]["reprojection_fill_constant_m"]
    replayed = replay(models, constant)
    print("REPLAY_OK", json.dumps(replayed), flush=True)
    manifest = read(FRESH)
    if any(r["split"] != "FRESH_QUALIFICATION" for r in manifest["scenes"]):
        raise PermissionError("FRESH-V2 records only")
    access, entries, states = [], {}, {}
    for record in manifest["scenes"]:
        sid = record["scene_id"]
        scene = predict(
            record, manifest, FRESH.parent, models, constant, access, "fresh_v2", states
        )
        for (role, qid), arrays in scene.items():
            path = out / "predictions" / f"{sid}_{role}_{qid}.npz"
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, **arrays)
            entries[f"{sid}/{role}/{qid}"] = {"path": str(path), "sha256": sha(path)}
        print("SEALED_SCENE", sid, flush=True)
    if any(
        "depth" in e.get("channels", []) and e.get("purpose") != "CONTEXT_ONLY_RGBD" for e in access
    ):
        raise AssertionError("Only context depth may be read before the seal")
    write(out / "state_hashes.json", states)
    write(out / "GT_access_before_seal.json", access)
    write(
        out / "prediction_seal.json",
        {
            "predictions": entries,
            "state_hashes_sha256": sha(out / "state_hashes.json"),
            "query_depth_read": False,
            "reprojection_fill_constant_m": constant,
            "seeds": seeds,
            "replay_on_eval_v3": replayed,
        },
    )
    print("PREDICTIONS_SEALED", len(entries), flush=True)


def absrel(pred, gt, mask):
    return float(np.mean(np.abs(pred[mask] - gt[mask]) / gt[mask])) if mask.any() else None


def score(root):
    root = Path(root).resolve()
    seal_file = read(root / "raw" / "prediction_seal.json")
    constant, seeds = seal_file["reprojection_fill_constant_m"], seal_file["seeds"]
    rows, access = [], []
    for record in read(FRESH)["scenes"]:
        sid = record["scene_id"]
        frames = {f["frame_id"]: f for f in record["frames"]}
        for role in ("A", "B"):
            for qid in record["roles"]["primary_query"]:
                entry = seal_file["predictions"][f"{sid}/{role}/{qid}"]
                if sha(entry["path"]) != entry["sha256"]:
                    raise PermissionError("A sealed prediction changed")
                arrays = dict(np.load(entry["path"]))
                access.append(
                    {"scene_id": sid, "frame_id": qid, "purpose": "POST_SEAL_QUERY_DEPTH"}
                )
                gt = np.load(frames[qid]["depth"]).squeeze().astype(np.float64)
                valid = np.isfinite(gt) & (gt > 0)
                near = arrays["near"].astype(bool)
                masks = {"ALL": valid, "NEAR": valid & near, "FAR": valid & ~near}
                harmonic = arrays["REPROJ_HARMONIC"]
                predictions = {
                    ("CONSTANT", 0): np.full(gt.shape, constant, dtype=np.float64),
                    ("REPROJ_NN", 0): arrays["REPROJ_NN"],
                    ("REPROJ_HARMONIC", 0): harmonic,
                }
                for family in FAMILIES:
                    for seed in seeds:
                        carrier = arrays[f"{family}_{seed}"]
                        low = arrays[f"OPACITY_{family}_{seed}"].astype(np.float64) < THRESHOLD
                        predictions[family, seed] = carrier
                        predictions[f"HFILL8_{family}", seed] = np.where(near, harmonic, carrier)
                        predictions[f"AHFILL8_{family}", seed] = np.where(
                            near | low, harmonic, carrier
                        )
                for (method, seed), value in predictions.items():
                    row = {
                        "cohort": "FRESH_V2",
                        "scene_id": sid,
                        "role": role,
                        "query_id": qid,
                        "method": method,
                        "seed": seed,
                    }
                    for region, mask in masks.items():
                        row[f"absrel_{region}"] = absrel(value, gt, mask)
                    row["depth_absrel"] = depth_metrics(value, gt)["depth_absrel"]
                    row["far_fraction"] = float(masks["FAR"].sum() / max(valid.sum(), 1))
                    rows.append(row)
    write(root / "raw" / "query_rows.json", rows)
    write(root / "raw" / "GT_access_after_seal.json", access)
    print("SCORED", len(rows), flush=True)


def advantage(rows, carrier, baseline="REPROJ_HARMONIC"):
    """{scene: FAR AbsRel(baseline) - FAR AbsRel(carrier)} with the V11 aggregation."""
    stage = host()
    a, b = (
        stage.scene_values(rows, baseline, "absrel_FAR"),
        stage.scene_values(rows, carrier, "absrel_FAR"),
    )
    return {s: a[s] - b[s] for s in sorted(set(a) & set(b))}


def unpaired(first, second):
    """mean(first) - mean(second), independent scene bootstraps (10,000 draws, seed 20260928)."""
    rng = np.random.default_rng(SEED)
    x, y = np.array(list(first.values())), np.array(list(second.values()))
    draws = rng.choice(x, (DRAWS, len(x))).mean(1) - rng.choice(y, (DRAWS, len(y))).mean(1)
    return {
        "mean": float(x.mean() - y.mean()),
        "ci95": [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))],
        "n_first": len(x),
        "n_second": len(y),
    }


def analyze_rows(rows, eval_rows, ext_volumes):
    stage, label = host(), inpainting().geometry().label

    def values(method, region, subset=rows):
        return stage.scene_values(subset, method, f"absrel_{region}")

    methods = sorted({r["method"] for r in rows})
    contrasts = {}
    definitions = {
        "A_V11C1": ("REPROJ_HARMONIC", "FAR", "V11C1", "FAR"),
        "A_V12C1": ("REPROJ_HARMONIC", "FAR", "V12C1", "FAR"),
        "NN_MINUS_V11C1_FAR": ("REPROJ_NN", "FAR", "V11C1", "FAR"),
        "HARMONIC_MINUS_HFILL8_V11C1": ("REPROJ_HARMONIC", "ALL", "HFILL8_V11C1", "ALL"),
        "HARMONIC_MINUS_AHFILL8_V11C1": ("REPROJ_HARMONIC", "ALL", "AHFILL8_V11C1", "ALL"),
        "HARMONIC_MINUS_HFILL8_V12C1": ("REPROJ_HARMONIC", "ALL", "HFILL8_V12C1", "ALL"),
        "CONSTANT_MINUS_V11C1": ("CONSTANT", "ALL", "V11C1", "ALL"),
        "NN_MINUS_V11C1_ALL": ("REPROJ_NN", "ALL", "V11C1", "ALL"),
    }
    for name, (a, ra, b, rb) in definitions.items():
        stats = stage.contrast(values(a, ra), values(b, rb))
        contrasts[name] = {"gain": stats, "definition": f"AbsRel {ra}({a}) - AbsRel {rb}({b})"}
        if stats is not None:
            contrasts[name]["label"] = label(stats)
    a_fresh = advantage(rows, "V11C1")
    a_eval = advantage(eval_rows, "C1")
    delta = unpaired(a_eval, a_fresh)
    p1 = contrasts["A_V11C1"].get("label", "UNDEFINED")
    gap = delta["ci95"][0] > 0
    branch = {(True, False): "A", (True, True): "B", (False, True): "C", (False, False): "D"}[
        (p1 == "ABOVE", gap)
    ]
    seen = {r["scene_id"] for r in rows if r["scene_id"].split("_")[1] in ext_volumes}
    split = {}
    for family in ("V14C1", "V15C1", "V11C1"):
        adv = advantage(rows, family)
        split[family] = {
            name: {"n": len(v), "mean_A": float(np.mean(list(v.values()))) if v else None}
            for name, v in (
                ("volume_has_train_ext_scene", {s: x for s, x in adv.items() if s in seen}),
                (
                    "volume_unseen_by_every_training_set",
                    {s: x for s, x in adv.items() if s not in seen},
                ),
            )
        }
    return {
        "absolute_absrel": {
            m: {region: stage.summary(values(m, region)) for region in REGIONS} for m in methods
        },
        "n_scenes": len({r["scene_id"] for r in rows}),
        "far_pixel_fraction": float(
            np.mean([r["far_fraction"] for r in rows if r["method"] == "REPROJ_NN"])
        ),
        "contrasts": contrasts,
        "A_EVAL_V3_V4": {"mean": float(np.mean(list(a_eval.values()))), "n_scenes": len(a_eval)},
        "A_FRESH_V2": {"mean": float(np.mean(list(a_fresh.values()))), "n_scenes": len(a_fresh)},
        "FAMILIARITY_DELTA": delta,
        "P1_LABEL": p1,
        "FAMILIARITY_GAP": gap,
        "INTERPRETATION_BRANCH": branch,
        "train_ext_volume_split": split,
    }


def analyze(root):
    root = Path(root).resolve()
    rows_path = root / "raw" / "query_rows.json"
    eval_rows = [
        r
        for r in read(V13 / "raw/query_rows.json")
        if r["cohort"] in ("EVAL_V3_NEAR", "EVAL_V4_NEAR")
    ]
    config = read(V14 / "training_contract.json")
    ext = set(config["train_sets"]["TRAIN_EXT"]) - set(config["train_sets"]["TRAIN72"])
    result = analyze_rows(read(rows_path), eval_rows, {s.split("_")[1] for s in ext})
    result["replay_on_eval_v3"] = read(root / "raw" / "prediction_seal.json")["replay_on_eval_v3"]
    result["input_sha256"] = {
        str(rows_path): sha(rows_path),
        str(V13 / "raw/query_rows.json"): sha(V13 / "raw/query_rows.json"),
    }
    result["final_holdout_opened"] = False
    write(root / "volume_disjoint_results.json", result)
    report(root, result)
    return result


def fmt(stats):
    if not stats or stats.get("mean") is None:
        return "UNDEFINED"
    lo, hi = stats["ci95"]
    tail = (
        f"，场景 {stats['positive']}/{stats['tied']}/{stats['negative']}"
        if "positive" in stats
        else ""
    )
    return f"{stats['mean']:+.4f} [{lo:+.4f}, {hi:+.4f}]{tail}"


def report(root, result):
    absolute, contrasts = result["absolute_absrel"], result["contrasts"]

    def cell(method, region):
        value = (absolute.get(method, {}).get(region) or {}).get("mean")
        return "—" if value is None else f"{value:.4f}"

    lines = [
        f"# {EXPERIMENT}",
        "",
        f"不训练的机制检验：只用 TRAIN72 训练的 carrier，在与训练集 volume 完全不相交的 FRESH-V2（{result['n_scenes']} 个场景）上，相对调和插值补洞的补全优势是否还在，并与共享 volume 的 EVAL-V3/V4（V13 的封存结果）比较。协议见 [PROTOCOL.md](PROTOCOL.md)。",
        "",
        f"- **P1（FRESH-V2 上的补全优势 A = 调和插值 FAR − V11 C1 FAR）：** {fmt(contrasts['A_V11C1']['gain'])}，{result['P1_LABEL']}",
        f"- **P2（熟悉度差 Δ = A(EVAL-V3/V4) − A(FRESH-V2)）：** {fmt(result['FAMILIARITY_DELTA'])}；A(EVAL) = {result['A_EVAL_V3_V4']['mean']:+.4f}（{result['A_EVAL_V3_V4']['n_scenes']} 个场景），A(FRESH) = {result['A_FRESH_V2']['mean']:+.4f}",
        f"- **分支 {result['INTERPRETATION_BRANCH']}**；复现封存预测的最大差 {result['replay_on_eval_v3']['max_abs_deviation']}",
        "",
        f"FAR 像素占 {result['far_pixel_fraction']:.1%}。",
        "",
        "| FRESH-V2 AbsRel | ALL | NEAR | FAR |",
        "|---|---:|---:|---:|",
    ]
    for method in (
        "CONSTANT",
        "REPROJ_NN",
        "REPROJ_HARMONIC",
        "V11C1",
        "V12C1",
        "V14C1",
        "V15C1",
        "HFILL8_V11C1",
        "AHFILL8_V11C1",
        "HFILL8_V12C1",
    ):
        lines.append(
            f"| {method} | {cell(method, 'ALL')} | {cell(method, 'NEAR')} | {cell(method, 'FAR')} |"
        )
    lines += ["", "| 对比 | 定义 | 差值 [95% CI]，场景 好/平/差 | 标签 |", "|---|---|---|---|"]
    for name, entry in contrasts.items():
        lines.append(
            f"| {name} | {entry['definition']} | {fmt(entry['gain'])} | {entry.get('label', 'UNDEFINED')} |"
        )
    lines += [
        "",
        "**TRAIN72+EXT 的模型（描述性）：** 按 FRESH 场景所在 volume 是否有 TRAIN-EXT 场景分组的平均 A：",
    ]
    for family, groups in result["train_ext_volume_split"].items():
        text = "；".join(
            f"{name} {g['n']} 个场景 {g['mean_A']:+.4f}"
            if g["mean_A"] is not None
            else f"{name} 0 个场景"
            for name, g in groups.items()
        )
        lines.append(f"- {family}：{text}")
    lines += [
        "",
        "**限制：** FRESH-V2 是第三次使用，只作机制检验；场景只有 17 个、10 个 volume；两个队列的难度也可能不同；受保护的 final holdout 未打开。",
    ]
    (root / "README.md").write_text("\n".join(lines) + "\n")
    summary = [
        "RGBD VOLUME DISJOINT V16",
        f"P1_LABEL={result['P1_LABEL']}",
        f"A_FRESH_V2={result['A_FRESH_V2']['mean']:.6f}",
        f"A_EVAL_V3_V4={result['A_EVAL_V3_V4']['mean']:.6f}",
        f"FAMILIARITY_DELTA={result['FAMILIARITY_DELTA']['mean']:.6f} CI={result['FAMILIARITY_DELTA']['ci95']}",
        f"FAMILIARITY_GAP={str(result['FAMILIARITY_GAP']).lower()}",
        f"INTERPRETATION_BRANCH={result['INTERPRETATION_BRANCH']}",
        f"REPLAY_MAX_ABS_DEVIATION={result['replay_on_eval_v3']['max_abs_deviation']}",
        "FINAL_HOLDOUT_TOUCHED=false",
    ]
    (root / "terminal_summary.txt").write_text("\n".join(summary) + "\n")
    print("\n".join(summary), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("lock", "seal", "score", "analyze"), required=True)
    parser.add_argument("--root", type=Path, default=Path("outputs") / EXPERIMENT)
    args = parser.parse_args()
    torch.set_num_threads(1)
    {"lock": lock, "seal": seal, "score": score, "analyze": analyze}[args.stage](args.root)


if __name__ == "__main__":
    main()
