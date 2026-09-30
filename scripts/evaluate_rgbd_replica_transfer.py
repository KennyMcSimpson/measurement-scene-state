#!/usr/bin/env python3
"""Cross-dataset transfer to Replica of the frozen carriers (no training), used exactly once.

outputs/EXP-3D-RGBD-REPLICA-TRANSFER-V1/PLAN_BEFORE_REPLICA_USE.md (written before any model saw
Replica) fixes the design: the method M is V11 C1 inside HFILL8 (V12 C1 if V12 ran and reached
branch A on EVAL-V4); V11 C0 and the qualified Core A carrier V7 C1 are descriptive references.
Stages: `lock` (after V11, the EVAL-V4 replication and V12 if it runs) freezes the inputs and the
rule-selected method; `seal` builds every state from context frames only and seals every
prediction and geometric baseline before any query depth is read; `score`; `analyze`.
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
from mcss.mechanism_pilot import rgbd_evidence_carrier as v5
from mcss.mechanism_pilot.readout_reanalysis import depth_metrics
from mcss.mechanism_pilot.rgbd_depth_bounds import context_depth_bounds
from mcss.types import Cameras

EXPERIMENT = "EXP-3D-RGBD-REPLICA-TRANSFER-V1"
PLAN = "PLAN_BEFORE_REPLICA_USE.md"
PLAN_SHA256 = "a3578c547a8bb659bf688e92e38063fb8f4ea7e9f3fb1a004c5b9118be4cda93"
V7 = Path("outputs/EXP-3D-RGBD-DEPTH-BOUNDS-V7")
V11 = Path("outputs/EXP-3D-RGBD-COMPLETION-V11")
V12 = Path("outputs/EXP-3D-RGBD-WIDTH-V12")
REPLICATION = Path("outputs/EXP-3D-RGBD-REPLICATION-EVAL-V4")
REPLICA = Path("outputs/EXP-3D-RGBD-REPLICA-DATA-V1")
SCRIPTS = Path(__file__).resolve().parent
REGIONS = ("ALL", "NEAR", "FAR")


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
    return module("transfer_v11_eval", "evaluate_rgbd_completion_eval_v3.py")


@functools.cache
def inpainting():
    """INPAINT-V1's frozen module: ray cosine, harmonic/Telea filling, NEAR mask, label rule."""
    return module("transfer_inpaint", "run_rgbd_inpaint_baselines.py")


def method_rule(v11=V11, v12=V12):
    """The plan's selection: V12 C1 only if V12 ran (it must when COMPLETION_GAIN > 0) with branch A."""
    gain = float(read(v11 / "eval_v3_results.json")["COMPLETION_GAIN"]["mean"])
    if gain > 0:
        if read(v12 / "integrity.json")["status"] != "PASS":
            raise PermissionError("V12 must be finalized before Replica is used")
        branch = read(v12 / "eval_v4_results.json")["INTERPRETATION_BRANCH"]
        return ("V12C1" if branch == "A" else "V11C1"), {
            "v11_completion_gain": gain,
            "v12_branch": branch,
        }
    return "V11C1", {"v11_completion_gain": gain, "v12_branch": None}


def carrier_specs(method, v7=V7, v11=V11, v12=V12):
    """{name: (selected file, variant key, loader, state builder)} of every carrier rendered."""
    specs = {
        "V7C1": (v7 / "selected_checkpoints.json", "C1", v5.load_checkpoint, v5.build_state),
        "V11C0": (
            v11 / "selected_checkpoints.json",
            "C0",
            v11_carrier.load_checkpoint,
            v11_carrier.build_state,
        ),
        "V11C1": (
            v11 / "selected_checkpoints.json",
            "C1",
            v11_carrier.load_checkpoint,
            v11_carrier.build_state,
        ),
    }
    if method == "V12C1":
        from mcss.mechanism_pilot import rgbd_width_carrier as v12_module

        specs["V12C1"] = (
            v12 / "selected_checkpoints.json",
            "C1",
            v12_module.load_checkpoint,
            v12_module.build_state,
        )
    return specs


def inputs(root, method, replica=REPLICA, v7=V7, v11=V11, v12=V12, replication=REPLICATION):
    paths = {
        "protocol": root / "PROTOCOL.md",
        "plan": root / PLAN,
        "script": Path(__file__).resolve(),
        "v11_eval_script": SCRIPTS / "evaluate_rgbd_completion_eval_v3.py",
        "inpaint_script": SCRIPTS / "run_rgbd_inpaint_baselines.py",
        "geometric_script": SCRIPTS / "run_rgbd_geometric_baselines.py",
        "replica_manifest": replica / "manifest_replica.json",
        "replica_integrity": replica / "preparation_integrity.json",
        "replica_lock": replica / "candidate_lock.json",
        "v11_contract": v11 / "training_contract.json",
        "v11_eval_v3_results": v11 / "eval_v3_results.json",
        "replication_results": replication / "replication_results.json",
    }
    for name, (selected, _, _, _) in carrier_specs(method, v7, v11, v12).items():
        paths[f"{name}_selected"] = selected
    if method == "V12C1":
        paths["v12_eval_v4_results"] = v12 / "eval_v4_results.json"
    return paths


def lock(root, replica=REPLICA, v7=V7, v11=V11, v12=V12, replication=REPLICATION):
    root = Path(root).resolve()
    if root.name != EXPERIMENT:
        raise PermissionError("Exact experiment directory required")
    if (root / "lock.json").exists():
        raise FileExistsError("Never re-lock")
    if not (root / "PROTOCOL.md").is_file():
        raise PermissionError("Protocol text required before locking")
    if sha(root / PLAN) != PLAN_SHA256:
        raise PermissionError("The pre-Replica plan changed")
    if read(v11 / "integrity.json")["status"] != "PASS":
        raise PermissionError("V11 must be finalized")
    if not (replication / "replication_results.json").exists():
        raise PermissionError("The EVAL-V4 replication must be complete before Replica is used")
    if read(replica / "preparation_integrity.json")["status"] != "PASS":
        raise PermissionError("Replica data must be prepared with integrity PASS")
    if (root / "raw").exists():
        raise PermissionError("No prediction may exist before the lock")
    method, basis = method_rule(v11, v12)
    paths = inputs(root, method, replica, v7, v11, v12, replication)
    write(
        root / "lock.json",
        {
            "status": "FROZEN_BEFORE_ANY_REPLICA_PREDICTION",
            "experiment": EXPERIMENT,
            "created_utc": datetime.datetime.now(datetime.UTC).isoformat(),
            "method": method,
            "method_basis": basis,
            "carriers": sorted(carrier_specs(method, v7, v11, v12)),
            "primary": f"AbsRel(REPROJ_HARMONIC) - AbsRel(HFILL8 of {method}) on Replica (INPAINT-V1 label rule)",
            "input_sha256": {
                name: {"path": str(Path(p).resolve()), "sha256": sha(p)}
                for name, p in paths.items()
            },
            "replica_media_read_by_a_model_before_lock": False,
            "final_holdout_opened": False,
        },
    )
    print("LOCKED", EXPERIMENT, "method", method, flush=True)


def verify(root):
    lock_ = read(root / "lock.json")
    for name, entry in lock_["input_sha256"].items():
        if sha(entry["path"]) != entry["sha256"]:
            raise PermissionError(f"Locked input changed: {name}")
    return lock_


def load_carriers(method, seeds, v7=V7, v11=V11, v12=V12):
    models = {}
    for name, (selected_file, variant, loader, builder) in carrier_specs(
        method, v7, v11, v12
    ).items():
        selected = read(selected_file)[variant]
        for seed in seeds:
            item = selected[str(seed)]
            if sha(Path(item["path"])) != item["sha256"]:
                raise PermissionError(f"{name} selected checkpoint changed")
            model, _ = loader(item["path"], "cpu")
            models[name, seed] = (model.eval(), builder)
    return models


@torch.no_grad()
def replay(models, v7=V7, v11=V11, v12=V12):
    """Our construction/rendering on the first DEV scene must reproduce each source's sealed DEV depth."""
    sealed = {
        "V7C1": v7 / "raw/selected_dev" / "C1_{seed}",
        "V11C0": v11 / "raw/selected_dev" / "C0_{seed}",
        "V11C1": v11 / "raw/selected_dev" / "C1_{seed}",
        "V12C1": v12 / "raw/selected_dev" / "C1_{seed}",
    }
    manifest_path = v7 / "scene_split.json"
    manifest = read(manifest_path)
    record = min(
        (r for r in manifest["scenes"] if r["split"] == "DEV"), key=lambda r: r["scene_id"]
    )
    sid, stage = record["scene_id"], host()
    geo = stage.geometry()
    renderer = FixedMeasurementRenderer(n_samples=64, ray_chunk_size=2048)
    worst, compared = 0.0, 0
    for role, context in stage.context_of(record, manifest, manifest_path.parent, []).items():
        bounds = context_depth_bounds(context, context[0].camera.c2w)
        for (name, seed), (model, builder) in models.items():
            state, state_anchor = builder(model, context, bounds, f"replay:{sid}:{role}")
            for qid in record["roles"]["primary_query"]:
                camera = geo.query_camera(record, qid, manifest["image_size"])
                batched = Cameras(
                    camera.intrinsics[None, None], camera.c2w[None, None], camera.image_size
                )
                local = transform_cameras(batched, torch.linalg.inv(state_anchor))
                mine = renderer(state, local, ("depth",))["depth"][0, 0, 0].numpy()
                directory = Path(str(sealed[name]).format(seed=seed))
                theirs = np.load(directory / "predictions" / f"{sid}_{role}_{qid}_direct.npz")[
                    "depth"
                ]
                worst = max(worst, float(np.max(np.abs(mine.astype(np.float64) - theirs))))
                compared += 1
    if worst > 1e-4:
        raise AssertionError(f"DEV replay of the sealed carrier predictions failed: {worst} m")
    return {"dev_scene": sid, "arrays_compared": compared, "max_abs_depth_deviation_m": worst}


@torch.no_grad()
def seal(root, v7=V7, v11=V11, v12=V12):
    root = Path(root).resolve()
    lock_ = verify(root)
    out = root / "raw"
    if (out / "prediction_seal.json").exists():
        raise FileExistsError("Replica predictions are sealed once")
    contract = read(lock_["input_sha256"]["v11_contract"]["path"])
    seeds, constant = contract["seeds"], contract["eval_v3"]["reprojection_fill_constant_m"]
    manifest_path = Path(lock_["input_sha256"]["replica_manifest"]["path"])
    manifest = read(manifest_path)
    if any(r["split"] != "REPLICA" for r in manifest["scenes"]):
        raise PermissionError("Replica records only")
    models = load_carriers(lock_["method"], seeds, v7, v11, v12)
    replayed = replay(models, v7, v11, v12)
    stage, inp = host(), inpainting()
    geo = stage.geometry()
    renderer = FixedMeasurementRenderer(n_samples=64, ray_chunk_size=2048)
    access, entries, states = [], {}, {}
    for record in manifest["scenes"]:
        sid = record["scene_id"]
        for role, context in stage.context_of(
            record, manifest, manifest_path.parent, access
        ).items():
            anchor = context[0].camera.c2w
            bounds = context_depth_bounds(context, anchor)
            points = geo.context_points(context)
            built = {}
            for (name, seed), (model, builder) in models.items():
                state, state_anchor = builder(model, context, bounds, f"replica:{sid}:{role}")
                built[name, seed] = (state, state_anchor)
                states[f"{name}:{seed}:{sid}/{role}"] = hash_scene_state(state)
            for qid in record["roles"]["primary_query"]:
                access.append({"scene_id": sid, "frame_id": qid, "purpose": "QUERY_CAMERA_ONLY"})
                camera = geo.query_camera(record, qid, manifest["image_size"])
                depth, hit = geo.reproject(points, camera)
                reproj = geo.fill_nearest(depth, hit, constant) if hit.any() else None
                if reproj is None:
                    reproj = np.full(camera.image_size, constant, dtype=np.float64)
                cosine = inp.ray_cosine(camera)
                telea, fallback = inp.telea_fill(depth, hit, cosine, constant)
                arrays = {
                    "REPROJ_NN": reproj,
                    "hit": hit,
                    "near": inp.near_mask(hit),
                    "REPROJ_HARMONIC": inp.harmonic_fill(depth, hit, cosine, constant),
                    "REPROJ_TELEA": telea,
                    "telea_fallback": np.array(fallback),
                }
                batched = Cameras(
                    camera.intrinsics[None, None], camera.c2w[None, None], camera.image_size
                )
                for (name, seed), (state, state_anchor) in built.items():
                    local = transform_cameras(batched, torch.linalg.inv(state_anchor))
                    pred = renderer(state, local, ("depth",))["depth"][0, 0, 0].numpy()
                    arrays[f"{name}_{seed}"] = pred.astype(np.float64)
                path = out / "predictions" / f"{sid}_{role}_{qid}.npz"
                path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(path, **arrays)
                entries[f"{sid}/{role}/{qid}"] = {"path": str(path), "sha256": sha(path)}
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
            "near_px": inp.NEAR_PX,
            "reprojection_fill_constant_m": constant,
            "replay_on_dev": replayed,
        },
    )
    print("REPLICA_PREDICTIONS_SEALED", len(entries), json.dumps(replayed), flush=True)


def score(root):
    root = Path(root).resolve()
    lock_ = read(root / "lock.json")
    seal_file = read(root / "raw" / "prediction_seal.json")
    manifest = read(lock_["input_sha256"]["replica_manifest"]["path"])
    seeds = read(lock_["input_sha256"]["v11_contract"]["path"])["seeds"]
    constant = seal_file["reprojection_fill_constant_m"]
    rows, access = [], []
    for record in manifest["scenes"]:
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
                predictions = {
                    (name, 0): arrays[name]
                    for name in ("REPROJ_NN", "REPROJ_HARMONIC", "REPROJ_TELEA")
                }
                predictions["CONSTANT", 0] = np.full(gt.shape, constant, dtype=np.float64)
                for name in lock_["carriers"]:
                    for seed in seeds:
                        carrier = arrays[f"{name}_{seed}"]
                        predictions[name, seed] = carrier
                        predictions[f"HFILL8_{name}", seed] = np.where(
                            near, arrays["REPROJ_HARMONIC"], carrier
                        )
                        predictions[f"FILL8_{name}", seed] = np.where(
                            near, arrays["REPROJ_NN"], carrier
                        )
                for (method, seed), value in predictions.items():
                    row = {
                        "scene_id": sid,
                        "role": role,
                        "query_id": qid,
                        "method": method,
                        "seed": seed,
                    }
                    for region, mask in masks.items():
                        row[f"absrel_{region}"] = (
                            float(np.mean(np.abs(value[mask] - gt[mask]) / gt[mask]))
                            if mask.any()
                            else None
                        )
                    row["depth_absrel"] = depth_metrics(value, gt)["depth_absrel"]
                    row["far_fraction"] = float(masks["FAR"].sum() / max(valid.sum(), 1))
                    rows.append(row)
    write(root / "raw" / "query_rows.json", rows)
    write(root / "raw" / "GT_access_after_seal.json", access)
    print("REPLICA_SCORED", len(rows), flush=True)
    return rows


def contrasts(method):
    return {
        "HARMONIC_HYBRID_GAIN": ("REPROJ_HARMONIC", "ALL", f"HFILL8_{method}", "ALL"),
        "HARMONIC_FAR_GAIN": ("REPROJ_HARMONIC", "FAR", method, "FAR"),
        "CONSTANT_GAIN": ("CONSTANT", "ALL", method, "ALL"),
        "NN_HYBRID_GAIN": ("REPROJ_NN", "ALL", f"FILL8_{method}", "ALL"),
        "WIDTH_FAR_GAIN": ("V11C0", "FAR", method, "FAR"),
        "V7C1_CONSTANT_GAIN": ("CONSTANT", "ALL", "V7C1", "ALL"),
        "V7C1_REPROJ_GAIN": ("REPROJ_NN", "ALL", "V7C1", "ALL"),
        "NN_MINUS_HARMONIC_ALL": ("REPROJ_NN", "ALL", "REPROJ_HARMONIC", "ALL"),
        "TELEA_HYBRID_GAIN": ("REPROJ_TELEA", "ALL", f"HFILL8_{method}", "ALL"),
    }


def analyze_rows(rows, method):
    stage, label = host(), inpainting().geometry().label
    methods = sorted({r["method"] for r in rows})

    def values(name, region):
        return stage.scene_values(rows, name, f"absrel_{region}")

    block = {
        "absolute_absrel": {
            m: {region: stage.summary(values(m, region)) for region in REGIONS} for m in methods
        },
        "contrasts": {},
        "n_scenes": len({r["scene_id"] for r in rows}),
        "far_pixel_fraction": float(
            np.mean([r["far_fraction"] for r in rows if r["method"] == "REPROJ_NN"])
        ),
        "method": method,
    }
    for name, (a, ra, b, rb) in contrasts(method).items():
        stats = stage.contrast(values(a, ra), values(b, rb))
        entry = {"gain": stats, "definition": f"AbsRel {ra}({a}) - AbsRel {rb}({b})"}
        if stats is not None:
            entry["label"] = label(stats)
            entry["strict_gate"], entry["strict_checks"] = stage.gate(stats)
        block["contrasts"][name] = entry
    primary = block["contrasts"]["HARMONIC_HYBRID_GAIN"]
    block["PRIMARY_LABEL"] = primary.get("label", "UNDEFINED")
    block["PRIMARY_STRICT_GATE"] = primary.get("strict_gate", "UNDEFINED")
    block["INTERPRETATION_BRANCH"] = {"ABOVE": "A", "NOT_DISTINGUISHABLE": "B", "BELOW": "C"}.get(
        primary.get("label"), "UNDEFINED"
    )
    return block


def analyze(root):
    root = Path(root).resolve()
    lock_ = read(root / "lock.json")
    rows_path = root / "raw" / "query_rows.json"
    result = analyze_rows(read(rows_path), lock_["method"])
    result["input_sha256"] = {str(rows_path): sha(rows_path)}
    result["final_holdout_opened"] = False
    write(root / "transfer_results.json", result)
    report(root, result)
    return result


def fmt(stats):
    if not stats:
        return "UNDEFINED"
    lo, hi = stats["ci95"]
    return f"{stats['mean']:+.4f} [{lo:+.4f}, {hi:+.4f}]，场景 {stats['positive']}/{stats['tied']}/{stats['negative']}"


def report(root, result):
    method = result["method"]
    lines = [
        f"# {EXPERIMENT}",
        "",
        f"跨数据集迁移：冻结的 carrier 不经任何训练，直接用于 Replica（{result['n_scenes']} 个场景，FAR 像素占 {result['far_pixel_fraction']:.1%}）。被检验的方法按事先写定的规则选为 {method}（放在 HFILL8 组合中）。所有预测在读取 query 深度之前封存。",
        "",
        f"主对比：AbsRel(REPROJ_HARMONIC) − AbsRel(HFILL8 {method}) = {fmt(result['contrasts']['HARMONIC_HYBRID_GAIN']['gain'])}；标签 {result['PRIMARY_LABEL']}，严格 gate {result['PRIMARY_STRICT_GATE']}，分支 {result['INTERPRETATION_BRANCH']}。",
        "",
        "| 方法 | ALL | NEAR | FAR |",
        "|---|---:|---:|---:|",
    ]
    for name in (
        "CONSTANT",
        "REPROJ_NN",
        "REPROJ_HARMONIC",
        "V7C1",
        "V11C0",
        "V11C1",
        "V12C1",
        f"FILL8_{method}",
        f"HFILL8_{method}",
    ):
        cells = [result["absolute_absrel"].get(name, {}).get(r) for r in REGIONS]
        if any(cells):
            lines.append(
                f"| {name} | "
                + " | ".join("—" if not c else f"{c['mean']:.4f}" for c in cells)
                + " |"
            )
    lines += ["", "| 对比 | 定义 | 差值 [95% CI]，场景 好/平/差 | 标签 |", "|---|---|---|---|"]
    for name, entry in result["contrasts"].items():
        lines.append(
            f"| {name} | {entry['definition']} | {fmt(entry['gain'])} | {entry.get('label', 'UNDEFINED')} |"
        )
    lines += [
        "",
        "说明：Replica 与 Hypersim 没有任何共享的场景、资产或相机；只有 8 个场景，功效有限；这是它唯一一次被使用；受保护的 Hypersim final holdout 未打开。",
    ]
    (root / "README.md").write_text("\n".join(lines) + "\n")
    summary = [
        "RGBD REPLICA TRANSFER V1",
        f"METHOD={method}",
        f"PRIMARY_LABEL={result['PRIMARY_LABEL']}",
        f"PRIMARY_STRICT_GATE={result['PRIMARY_STRICT_GATE']}",
        f"INTERPRETATION_BRANCH={result['INTERPRETATION_BRANCH']}",
        f"N_SCENES={result['n_scenes']}",
    ]
    for name, entry in result["contrasts"].items():
        if entry["gain"]:
            summary.append(
                f"{name}={entry['gain']['mean']:.6f} CI={entry['gain']['ci95']} LABEL={entry['label']}"
            )
    summary.append("FINAL_HOLDOUT_TOUCHED=false")
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
