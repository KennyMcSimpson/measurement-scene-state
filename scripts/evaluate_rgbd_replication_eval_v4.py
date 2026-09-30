#!/usr/bin/env python3
"""Replication of V11 and INPAINT-V1 on the never-read EVAL-V4 cohort (no training).

The frozen V11 C0/C1 selected checkpoints, V11's state construction and rendering, V11's NEAR/FAR
regions and INPAINT-V1's inpainting are applied unchanged to EVAL-V4, as fixed by
outputs/EXP-3D-RGBD-REPLICATION-EVAL-V4/PLAN_BEFORE_V11_RESULTS.md (written before any V11
result). Stages: `lock` freezes this protocol and code; `seal` renders every prediction and
computes every geometric baseline from context frames and query cameras only, hashes them, and
replays the pipeline on two EVAL-V3 scenes against V11's and INPAINT-V1's sealed files; `score`
reads query depth only after the seal; `analyze` computes the preregistered contrasts.
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
from mcss.mechanism_pilot.readout_reanalysis import depth_metrics
from mcss.mechanism_pilot.rgbd_completion_carrier import PRIMARY_PAIR, build_state, load_checkpoint
from mcss.mechanism_pilot.rgbd_depth_bounds import context_depth_bounds
from mcss.types import Cameras

EXPERIMENT = "EXP-3D-RGBD-REPLICATION-EVAL-V4"
V11 = Path("outputs/EXP-3D-RGBD-COMPLETION-V11")
INPAINT = Path("outputs/EXP-3D-RGBD-INPAINT-BASELINES-V1")
EVAL_V4 = Path("outputs/EXP-3D-RGBD-EVAL-V4-DATA")
PLAN = "PLAN_BEFORE_V11_RESULTS.md"
PLAN_SHA256 = "b726d336a6b6377f49d75c991e228a21380092312d6ee5e9b6f57828f639fb1c"
SCRIPTS = Path(__file__).resolve().parent
REPLAY_SCENES = 2
REGIONS = ("ALL", "NEAR", "FAR")
CONTRASTS = {
    "HARMONIC_HYBRID_GAIN": ("REPROJ_HARMONIC", "ALL", "HFILL8_C1", "ALL"),
    "HARMONIC_HYBRID_GAIN_C0": ("REPROJ_HARMONIC", "ALL", "HFILL8_C0", "ALL"),
    "HARMONIC_FAR_GAIN_C1": ("REPROJ_HARMONIC", "FAR", "C1", "FAR"),
    "HARMONIC_FAR_GAIN_C0": ("REPROJ_HARMONIC", "FAR", "C0", "FAR"),
    "TELEA_HYBRID_GAIN": ("REPROJ_TELEA", "ALL", "TFILL8_C1", "ALL"),
    "TELEA_FAR_GAIN_C1": ("REPROJ_TELEA", "FAR", "C1", "FAR"),
    "NN_MINUS_HARMONIC_ALL": ("REPROJ_NN", "ALL", "REPROJ_HARMONIC", "ALL"),
    "NN_MINUS_HARMONIC_NEAR": ("REPROJ_NN", "NEAR", "REPROJ_HARMONIC", "NEAR"),
    "NN_MINUS_HARMONIC_FAR": ("REPROJ_NN", "FAR", "REPROJ_HARMONIC", "FAR"),
    "NN_MINUS_TELEA_ALL": ("REPROJ_NN", "ALL", "REPROJ_TELEA", "ALL"),
    "NN_HYBRID_GAIN_V11": ("REPROJ_NN", "ALL", "FILL8_C1", "ALL"),
    "HARMONIC_MINUS_NN_HYBRID": ("REPROJ_HARMONIC", "ALL", "FILL8_C1", "ALL"),
}


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
    """V11's frozen EVAL-V3 module: context loader, aggregation, bootstrap, gates."""
    return module("replication_v11_eval", "evaluate_rgbd_completion_eval_v3.py")


@functools.cache
def inpainting():
    """INPAINT-V1's frozen module: ray cosine, harmonic and Telea filling, NEAR mask, label rule."""
    return module("replication_inpaint", "run_rgbd_inpaint_baselines.py")


def inputs(root, v11=V11, inpaint=INPAINT, eval_v4=EVAL_V4):
    return {
        "protocol": root / "PROTOCOL.md",
        "plan": root / PLAN,
        "script": Path(__file__).resolve(),
        "v11_eval_script": SCRIPTS / "evaluate_rgbd_completion_eval_v3.py",
        "inpaint_script": SCRIPTS / "run_rgbd_inpaint_baselines.py",
        "geometric_script": SCRIPTS / "run_rgbd_geometric_baselines.py",
        "v11_contract": v11 / "training_contract.json",
        "inpaint_lock": inpaint / "lock.json",
        "eval_v4_manifest": eval_v4 / "manifest_eval_v4.json",
        "eval_v4_integrity": eval_v4 / "preparation_integrity.json",
        "eval_v4_candidate_lock": eval_v4 / "candidate_lock.json",
    }


# ---------------------------------------------------------------- stages
def lock(root, v11=V11, inpaint=INPAINT, eval_v4=EVAL_V4):
    root = Path(root).resolve()
    if root.name != EXPERIMENT:
        raise PermissionError("Exact experiment directory required")
    if (root / "lock.json").exists():
        raise FileExistsError("Never re-lock")
    if not (root / "PROTOCOL.md").is_file():
        raise PermissionError("Protocol text required before locking")
    if sha(root / PLAN) != PLAN_SHA256:
        raise PermissionError("The pre-V11 plan changed")
    if read(eval_v4 / "preparation_integrity.json")["status"] != "PASS":
        raise PermissionError("EVAL-V4 data must be prepared with integrity PASS")
    if (root / "raw").exists():
        raise PermissionError("No prediction may exist before the lock")
    paths = inputs(root, v11, inpaint, eval_v4)
    write(
        root / "lock.json",
        {
            "status": "FROZEN_BEFORE_ANY_EVAL_V4_PREDICTION",
            "experiment": EXPERIMENT,
            "created_utc": datetime.datetime.now(datetime.UTC).isoformat(),
            "primary": "AbsRel(REPROJ_HARMONIC) - AbsRel(HFILL8 of V11 C1) on EVAL-V4 (INPAINT-V1 label rule)",
            "secondary": "V11 COMPLETION_GAIN and HYBRID_GAIN on EVAL-V4 (frozen V2 gate)",
            "input_sha256": {
                name: {"path": str(Path(p).resolve()), "sha256": sha(p)}
                for name, p in paths.items()
            },
            "v11_results_existed_at_lock": (v11 / "eval_v3_results.json").exists(),
            "inpaint_results_existed_at_lock": (inpaint / "inpaint_results.json").exists(),
            "eval_v4_media_scored_before_lock": False,
            "final_holdout_opened": False,
        },
    )
    print("LOCKED", EXPERIMENT, flush=True)


def verify(root, v11=V11):
    lock_ = read(root / "lock.json")
    for name, entry in lock_["input_sha256"].items():
        if sha(entry["path"]) != entry["sha256"]:
            raise PermissionError(f"Locked input changed: {name}")
    if read(v11 / "integrity.json")["status"] != "PASS":
        raise PermissionError("V11 must be finalized with integrity PASS")
    return lock_


def carriers(v11=V11):
    config = read(v11 / "training_contract.json")
    selected = read(v11 / "selected_checkpoints.json")
    models = {}
    for variant in PRIMARY_PAIR:
        for seed in config["seeds"]:
            item = selected[variant][str(seed)]
            if sha(Path(item["path"])) != item["sha256"]:
                raise PermissionError("Selected V11 checkpoint changed")
            model, _ = load_checkpoint(item["path"], "cpu")
            models[variant, seed] = model.eval()
    return config, models


@torch.no_grad()
def predict(record, manifest, base, models, constant, access, tag, states=None):
    """Every sealed array of every primary query of one scene (V11's code path plus inpainting).

    `tag` only names the construction episode ("eval_v3" reproduces V11's own episode ids).
    """
    v11, inp, geo = host(), inpainting(), host().geometry()
    renderer = FixedMeasurementRenderer(n_samples=64, ray_chunk_size=2048)
    out = {}
    sid = record["scene_id"]
    for role, context in v11.context_of(record, manifest, base, access).items():
        anchor = context[0].camera.c2w
        bounds = context_depth_bounds(context, anchor)
        points = geo.context_points(context)
        built = {}
        for (variant, seed), carrier in models.items():
            state, state_anchor = build_state(carrier, context, bounds, f"{tag}:{sid}:{role}")
            built[variant, seed] = (state, state_anchor)
            if states is not None:
                states[f"{variant}:{seed}:{sid}/{role}"] = hash_scene_state(state)
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
            for (variant, seed), (state, state_anchor) in built.items():
                local = transform_cameras(batched, torch.linalg.inv(state_anchor))
                pred = renderer(state, local, ("depth",))["depth"][0, 0, 0].numpy()
                arrays[f"{variant}_{seed}"] = pred.astype(np.float64)
            out[role, qid] = arrays
    return out


def replay(models, constant, v11=V11, inpaint=INPAINT):
    """Our code path on the first EVAL-V3 scenes must reproduce V11's and INPAINT-V1's sealed files."""
    config = read(v11 / "training_contract.json")
    manifest_path = Path(config["eval_v3"]["manifest"])
    manifest = read(manifest_path)
    seal = read(v11 / "raw" / "eval_v3" / "prediction_seal.json")
    inpaint_seal = read(inpaint / "raw" / "inpaint_seal.json")
    worst, compared = 0.0, 0
    for record in manifest["scenes"][:REPLAY_SCENES]:
        sid = record["scene_id"]
        mine = predict(record, manifest, manifest_path.parent, models, constant, [], "eval_v3")
        for (role, qid), arrays in mine.items():
            sealed = dict(np.load(seal["predictions"][f"{sid}/{role}/{qid}"]["path"]))
            filled = dict(np.load(inpaint_seal[f"EVAL_V3/{sid}/{role}/{qid}"]["path"]))
            if not (
                np.array_equal(arrays["hit"], sealed["hit"])
                and np.array_equal(arrays["near"], sealed["near"])
            ):
                raise AssertionError(f"Replay hit/near mask differs for {sid}/{role}/{qid}")
            pairs = [(arrays["REPROJ_NN"], sealed["REPROJ_NN"])]
            pairs += [(arrays[k], sealed[k]) for k in sealed if k[:2] in ("C0", "C1")]
            pairs += [(arrays[k], filled[k]) for k in ("REPROJ_HARMONIC", "REPROJ_TELEA")]
            for a, b in pairs:
                worst = max(worst, float(np.max(np.abs(a - b))))
                compared += 1
    if worst > 1e-6:
        raise AssertionError(f"Replay of the sealed EVAL-V3 pipeline failed: {worst}")
    return {"scenes": REPLAY_SCENES, "arrays_compared": compared, "max_abs_deviation": worst}


def seal(root, v11=V11, inpaint=INPAINT, eval_v4=EVAL_V4):
    root = Path(root).resolve()
    lock_ = verify(root, v11)
    out = root / "raw"
    if (out / "prediction_seal.json").exists():
        raise FileExistsError("EVAL-V4 predictions are sealed once")
    config, models = carriers(v11)
    constant = config["eval_v3"]["reprojection_fill_constant_m"]
    if config["eval_v3"]["near_px"] != inpainting().NEAR_PX:
        raise PermissionError("NEAR definition must be V11's")
    manifest_path = Path(lock_["input_sha256"]["eval_v4_manifest"]["path"])
    manifest = read(manifest_path)
    if any(r["split"] != "EVAL_V4" for r in manifest["scenes"]):
        raise PermissionError("EVAL-V4 records only")
    replayed = replay(models, constant, v11, inpaint)
    access, entries, states = [], {}, {}
    for record in manifest["scenes"]:
        sid = record["scene_id"]
        scene = predict(
            record, manifest, manifest_path.parent, models, constant, access, "eval_v4", states
        )
        for (role, qid), arrays in scene.items():
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
            "near_px": config["eval_v3"]["near_px"],
            "reprojection_fill_constant_m": constant,
            "replay_on_eval_v3": replayed,
        },
    )
    print("EVAL_V4_PREDICTIONS_SEALED", len(entries), json.dumps(replayed), flush=True)


def score(root):
    root = Path(root).resolve()
    lock_ = read(root / "lock.json")
    seal_file = read(root / "raw" / "prediction_seal.json")
    manifest = read(lock_["input_sha256"]["eval_v4_manifest"]["path"])
    seeds = read(lock_["input_sha256"]["v11_contract"]["path"])["seeds"]
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
                    ("REPROJ_NN", 0): arrays["REPROJ_NN"],
                    ("REPROJ_HARMONIC", 0): arrays["REPROJ_HARMONIC"],
                    ("REPROJ_TELEA", 0): arrays["REPROJ_TELEA"],
                }
                for variant in PRIMARY_PAIR:
                    for seed in seeds:
                        carrier = arrays[f"{variant}_{seed}"]
                        predictions[variant, seed] = carrier
                        predictions[f"FILL8_{variant}", seed] = np.where(
                            near, arrays["REPROJ_NN"], carrier
                        )
                        predictions[f"HFILL8_{variant}", seed] = np.where(
                            near, arrays["REPROJ_HARMONIC"], carrier
                        )
                        predictions[f"TFILL8_{variant}", seed] = np.where(
                            near, arrays["REPROJ_TELEA"], carrier
                        )
                for (method, seed), value in predictions.items():
                    row = {
                        "cohort": "EVAL_V4",
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
    print("EVAL_V4_SCORED", len(rows), flush=True)
    return rows


def inpaint_block(rows):
    """INPAINT-V1's per-cohort analysis block (same contrasts, aggregation and labels)."""
    v11, label = host(), inpainting().geometry().label
    methods = sorted({r["method"] for r in rows})

    def values(method, region):
        return v11.scene_values(rows, method, f"absrel_{region}")

    block = {
        "absolute_absrel": {
            m: {region: v11.summary(values(m, region)) for region in REGIONS} for m in methods
        },
        "contrasts": {},
        "n_scenes": len({r["scene_id"] for r in rows}),
        "far_pixel_fraction": float(
            np.mean([r["far_fraction"] for r in rows if r["method"] == "REPROJ_NN"])
        ),
    }
    for name, (a, ra, b, rb) in CONTRASTS.items():
        stats = v11.contrast(values(a, ra), values(b, rb))
        entry = {"gain": stats, "definition": f"AbsRel {ra}({a}) - AbsRel {rb}({b})"}
        if stats is not None:
            entry["label"] = label(stats)
            entry["strict_gate"], entry["strict_checks"] = v11.gate(stats)
        block["contrasts"][name] = entry
    return block


def analyze(root):
    root = Path(root).resolve()
    rows_path = root / "raw" / "query_rows.json"
    rows = read(rows_path)
    block = inpaint_block(rows)
    v11_rows = [r for r in rows if r["method"] in {"REPROJ_NN", "C0", "C1", "FILL8_C0", "FILL8_C1"}]
    v11_gates = host().analyze_rows(v11_rows)
    primary = block["contrasts"]["HARMONIC_HYBRID_GAIN"]
    result = {
        "eval_v4": block,
        "v11_gates_on_eval_v4": v11_gates,
        "PRIMARY_LABEL": primary.get("label", "UNDEFINED"),
        "PRIMARY_STRICT_GATE": primary.get("strict_gate", "UNDEFINED"),
        "REPLICATION_BRANCH": {"ABOVE": "A", "NOT_DISTINGUISHABLE": "B", "BELOW": "C"}.get(
            primary.get("label"), "UNDEFINED"
        ),
        "V11_BRANCH_ON_EVAL_V4": v11_gates["INTERPRETATION_BRANCH"],
        "replay_on_eval_v3": read(root / "raw" / "prediction_seal.json")["replay_on_eval_v3"],
        "input_sha256": {str(rows_path): sha(rows_path)},
        "final_holdout_opened": False,
    }
    write(root / "replication_results.json", result)
    report(root, result)
    return result


def fmt(stats):
    if not stats:
        return "UNDEFINED"
    lo, hi = stats["ci95"]
    return f"{stats['mean']:+.4f} [{lo:+.4f}, {hi:+.4f}]，场景 {stats['positive']}/{stats['tied']}/{stats['negative']}"


def report(root, result):
    block, gates = result["eval_v4"], result["v11_gates_on_eval_v4"]
    lines = [
        f"# {EXPERIMENT}",
        "",
        f"在从未读取过的 EVAL-V4（{block['n_scenes']} 个场景，FAR 像素占 {block['far_pixel_fraction']:.1%}）上复现 V11 与 INPAINT-V1。V11 已选中的 checkpoint、区域定义、补洞方法与统计规则全部冻结，由 V11 出结果之前写定的计划规定。不训练。",
        "",
        f"- 主对比：AbsRel(REPROJ_HARMONIC) − AbsRel(HFILL8_C1) = {fmt(block['contrasts']['HARMONIC_HYBRID_GAIN']['gain'])}；标签 {result['PRIMARY_LABEL']}，严格 gate {result['PRIMARY_STRICT_GATE']}，复现分支 {result['REPLICATION_BRANCH']}。",
        f"- V11 的 COMPLETION_GAIN = {fmt(gates['COMPLETION_GAIN'])}（{gates['statuses']['COMPLETION_STATUS']}）；HYBRID_GAIN = {fmt(gates['HYBRID_GAIN'])}（{gates['statuses']['HYBRID_STATUS']}）；V11 分支 {result['V11_BRANCH_ON_EVAL_V4']}。",
        f"- 流程复现：在 EVAL-V3 的前 {result['replay_on_eval_v3']['scenes']} 个场景上重算的 {result['replay_on_eval_v3']['arrays_compared']} 个数组，与 V11 和 INPAINT-V1 的封存文件最大差 {result['replay_on_eval_v3']['max_abs_deviation']:.1e}。",
        "",
        "| 方法 | ALL | NEAR | FAR |",
        "|---|---:|---:|---:|",
    ]
    for method in (
        "REPROJ_NN",
        "REPROJ_HARMONIC",
        "REPROJ_TELEA",
        "C0",
        "C1",
        "FILL8_C1",
        "HFILL8_C0",
        "HFILL8_C1",
        "TFILL8_C1",
    ):
        cells = [block["absolute_absrel"].get(method, {}).get(r) for r in REGIONS]
        lines.append(
            f"| {method} | "
            + " | ".join("—" if not c else f"{c['mean']:.4f}" for c in cells)
            + " |"
        )
    lines += ["", "| 对比 | 定义 | 差值 [95% CI]，场景 好/平/差 | 标签 |", "|---|---|---|---|"]
    for name, entry in block["contrasts"].items():
        lines.append(
            f"| {name} | {entry['definition']} | {fmt(entry['gain'])} | {entry.get('label', 'UNDEFINED')} |"
        )
    lines += [
        "",
        "说明：EVAL-V4 与 EVAL-V3 一样是机制队列（按场景和源资产与训练集及 EVAL-V3 不相交，可能共享 volume），不是资格验证；受保护的 final holdout 未打开。",
    ]
    (root / "README.md").write_text("\n".join(lines) + "\n")
    summary = [
        "RGBD REPLICATION EVAL-V4",
        f"PRIMARY_LABEL={result['PRIMARY_LABEL']}",
        f"PRIMARY_STRICT_GATE={result['PRIMARY_STRICT_GATE']}",
        f"REPLICATION_BRANCH={result['REPLICATION_BRANCH']}",
        f"V11_BRANCH_ON_EVAL_V4={result['V11_BRANCH_ON_EVAL_V4']}",
        f"COMPLETION_STATUS={gates['statuses']['COMPLETION_STATUS']}",
        f"HYBRID_STATUS={gates['statuses']['HYBRID_STATUS']}",
        f"N_SCENES={block['n_scenes']}",
    ]
    for name, entry in block["contrasts"].items():
        if entry["gain"]:
            summary.append(
                f"{name}={entry['gain']['mean']:.6f} CI={entry['gain']['ci95']} LABEL={entry['label']}"
            )
    summary += [
        f"REPLAY_MAX_DEVIATION={result['replay_on_eval_v3']['max_abs_deviation']}",
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
