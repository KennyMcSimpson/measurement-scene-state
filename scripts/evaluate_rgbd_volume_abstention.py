#!/usr/bin/env python3
"""V13: abstain outside the state's volume (no training).

The frozen V11 C1 selected checkpoints render depth and opacity for queries far from the context
frames (EVAL-FAR) and for the near queries of EVAL-V3/V4. AHFILL8 uses REPROJ_HARMONIC on NEAR
pixels and on FAR pixels whose rendered opacity is below 0.5, the carrier elsewhere, as fixed by
PROTOCOL.md. Stages: `lock` freezes the protocol, code and inputs before any far query is rendered;
`seal --cohort` replays two sealed EVAL-V3 scenes, then renders and hashes every prediction and
geometric baseline of one cohort before any query depth is read; `score` needs every cohort's seal;
`analyze` computes the preregistered contrasts.
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
from mcss.geometry import generate_rays, intersect_aabb, transform_cameras
from mcss.measurements import FixedMeasurementRenderer
from mcss.mechanism_pilot import rgbd_completion_carrier as v11_carrier
from mcss.mechanism_pilot.readout_reanalysis import depth_metrics
from mcss.mechanism_pilot.rgbd_depth_bounds import context_depth_bounds
from mcss.types import Cameras

EXPERIMENT = "EXP-3D-RGBD-VOLUME-V13"
V11 = Path("outputs/EXP-3D-RGBD-COMPLETION-V11")
INPAINT = Path("outputs/EXP-3D-RGBD-INPAINT-BASELINES-V1")
EVAL_FAR = Path("outputs/EXP-3D-RGBD-EVAL-FAR-DATA")
EXPLORATORY = "audit/exploratory_opacity_fallback"
COHORTS = {
    "EVAL_FAR": (EVAL_FAR / "manifest_eval_far.json", "far_query", "EVAL_FAR"),
    "EVAL_V3_NEAR": (
        Path("outputs/EXP-3D-RGBD-EVAL-V3-DATA/manifest_eval_v3.json"),
        "primary_query",
        "EVAL_V3",
    ),
    "EVAL_V4_NEAR": (
        Path("outputs/EXP-3D-RGBD-EVAL-V4-DATA/manifest_eval_v4.json"),
        "primary_query",
        "EVAL_V4",
    ),
}
PRIMARY = "EVAL_FAR"
THRESHOLD = 0.5
DESCRIPTIVE_THRESHOLDS = (0.3, 0.7, 0.9)
REGIONS = ("ALL", "NEAR", "FAR")
REPLAY_SCENES = 2
METHODS = (
    "CONSTANT",
    "REPROJ_NN",
    "REPROJ_HARMONIC",
    "C1",
    "HFILL8",
    "AHFILL8",
    *(f"AHFILL8_t{t}" for t in DESCRIPTIVE_THRESHOLDS),
)
CONTRASTS = {
    "ABSTAIN_GAIN": ("HFILL8", "ALL", "AHFILL8", "ALL"),
    "HARMONIC_VS_AHFILL8": ("REPROJ_HARMONIC", "ALL", "AHFILL8", "ALL"),
    "HARMONIC_VS_HFILL8": ("REPROJ_HARMONIC", "ALL", "HFILL8", "ALL"),
    "NN_VS_AHFILL8": ("REPROJ_NN", "ALL", "AHFILL8", "ALL"),
    "CONSTANT_VS_C1": ("CONSTANT", "ALL", "C1", "ALL"),
    "NN_VS_C1": ("REPROJ_NN", "ALL", "C1", "ALL"),
    "HARMONIC_VS_C1_FAR": ("REPROJ_HARMONIC", "FAR", "C1", "FAR"),
    "HARMONIC_VS_AHFILL8_FAR": ("REPROJ_HARMONIC", "FAR", "AHFILL8", "FAR"),
    **{
        f"HARMONIC_VS_AHFILL8_t{t}": ("REPROJ_HARMONIC", "ALL", f"AHFILL8_t{t}", "ALL")
        for t in DESCRIPTIVE_THRESHOLDS
    },
}
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
    return module("v13_v11_eval", "evaluate_rgbd_completion_eval_v3.py")


@functools.cache
def inpainting():
    """INPAINT-V1's frozen module: ray cosine, harmonic filling, NEAR mask, label rule."""
    return module("v13_inpaint", "run_rgbd_inpaint_baselines.py")


def abstain(near, opacity, harmonic, carrier, threshold):
    """AHFILL8: harmonic on NEAR pixels and on FAR pixels the carrier renders with low opacity."""
    return np.where(near | (opacity < threshold), harmonic, carrier)


def inputs(root, v11=V11, inpaint=INPAINT):
    paths = {
        "protocol": root / "PROTOCOL.md",
        "script": Path(__file__).resolve(),
        "v11_eval_script": SCRIPTS / "evaluate_rgbd_completion_eval_v3.py",
        "inpaint_script": SCRIPTS / "run_rgbd_inpaint_baselines.py",
        "geometric_script": SCRIPTS / "run_rgbd_geometric_baselines.py",
        "v11_contract": v11 / "training_contract.json",
        "v11_selected": v11 / "selected_checkpoints.json",
        "v11_eval_v3_seal": v11 / "raw/eval_v3/prediction_seal.json",
        "inpaint_lock": inpaint / "lock.json",
        "eval_far_candidate_lock": EVAL_FAR / "candidate_lock.json",
        "eval_far_integrity": EVAL_FAR / "preparation_integrity.json",
    }
    for name, (manifest, _, _) in COHORTS.items():
        paths[f"{name}_manifest"] = manifest
    for cohort in ("EVAL_V3", "EVAL_V4", "REPLICA"):
        paths[f"exploratory_{cohort}"] = root / EXPLORATORY / f"fallback_{cohort}.json"
    return paths


# ---------------------------------------------------------------- stages
def lock(root, v11=V11, inpaint=INPAINT):
    root = Path(root).resolve()
    if root.name != EXPERIMENT:
        raise PermissionError("Exact experiment directory required")
    if (root / "lock.json").exists():
        raise FileExistsError("Never re-lock")
    if not (root / "PROTOCOL.md").is_file():
        raise PermissionError("Protocol text required before locking")
    if (root / "raw").exists():
        raise PermissionError("No prediction may exist before the lock")
    if read(EVAL_FAR / "preparation_integrity.json")["status"] != "PASS":
        raise PermissionError("EVAL-FAR data must be prepared with integrity PASS")
    paths = inputs(root, v11, inpaint)
    write(
        root / "lock.json",
        {
            "status": "FROZEN_BEFORE_ANY_FAR_QUERY_PREDICTION",
            "experiment": EXPERIMENT,
            "created_utc": datetime.datetime.now(datetime.UTC).isoformat(),
            "threshold": THRESHOLD,
            "primary": "EVAL_FAR: P1 AbsRel(HFILL8) - AbsRel(AHFILL8) (frozen V2 gate); P2 AbsRel(REPROJ_HARMONIC) - AbsRel(AHFILL8) (INPAINT-V1 label rule)",
            "input_sha256": {
                name: {"path": str(Path(p).resolve()), "sha256": sha(p)}
                for name, p in paths.items()
            },
            "far_query_rendered_before_lock": False,
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
    selected = read(v11 / "selected_checkpoints.json")["C1"]
    models = {}
    for seed in config["seeds"]:
        item = selected[str(seed)]
        if sha(Path(item["path"])) != item["sha256"]:
            raise PermissionError("Selected V11 C1 checkpoint changed")
        models[seed] = v11_carrier.load_checkpoint(item["path"], "cpu")[0].eval()
    return config, models


@torch.no_grad()
def predict(record, manifest, base, models, constant, role_name, access, tag, states=None):
    """Every sealed array of every query of one scene: carrier depth and opacity, box, geometry.

    `tag` only names the construction episode ("eval_v3" reproduces V11's own episode ids).
    """
    stage, inp = host(), inpainting()
    geo = stage.geometry()
    renderer = FixedMeasurementRenderer(n_samples=64, ray_chunk_size=2048)
    out, sid = {}, record["scene_id"]
    for role, context in stage.context_of(record, manifest, base, access).items():
        anchor = context[0].camera.c2w
        bounds = context_depth_bounds(context, anchor)
        points = geo.context_points(context)
        built = {}
        for seed, model in models.items():
            state, state_anchor = v11_carrier.build_state(
                model, context, bounds, f"{tag}:{sid}:{role}"
            )
            built[seed] = (state, state_anchor)
            if states is not None:
                states[f"C1:{seed}:{sid}/{role}"] = hash_scene_state(state)
        box_state, box_anchor = next(iter(built.values()))
        if not all(
            torch.equal(s.bounds, box_state.bounds) and torch.equal(a, box_anchor)
            for s, a in built.values()
        ):
            raise AssertionError("All seeds of one context must share the volume and the anchor")
        for qid in record["roles"][role_name]:
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
            box_local = transform_cameras(batched, torch.linalg.inv(box_anchor))
            origins, directions = generate_rays(box_local.to(dtype=torch.float64))
            t_near, t_exit, box_hit = intersect_aabb(
                origins[0, 0], directions[0, 0], box_state.bounds[0].double()
            )
            arrays |= {
                "BOX_NEAR": t_near.numpy(),
                "BOX_EXIT": t_exit.numpy(),
                "BOX_HIT": box_hit.numpy(),
            }
            for seed, (state, state_anchor) in built.items():
                local = transform_cameras(batched, torch.linalg.inv(state_anchor))
                pred = renderer(state, local, ("depth", "visibility"))
                arrays[f"C1_{seed}"] = pred["depth"][0, 0, 0].numpy().astype(np.float64)
                arrays[f"OPACITY_{seed}"] = pred["visibility"][0, 0, 0].numpy()
            out[role, qid] = arrays
    return out


def replay(models, constant, v11=V11, inpaint=INPAINT):
    """Our code path on the first EVAL-V3 scenes must reproduce V11's and INPAINT-V1's sealed files."""
    manifest_path, role_name, _ = COHORTS["EVAL_V3_NEAR"]
    manifest = read(manifest_path)
    seal_ = read(v11 / "raw/eval_v3/prediction_seal.json")["predictions"]
    inpaint_seal = read(inpaint / "raw/inpaint_seal.json")
    worst, compared = 0.0, 0
    for record in manifest["scenes"][:REPLAY_SCENES]:
        sid = record["scene_id"]
        mine = predict(
            record, manifest, manifest_path.parent, models, constant, role_name, [], "eval_v3"
        )
        for (role, qid), arrays in mine.items():
            sealed = dict(np.load(seal_[f"{sid}/{role}/{qid}"]["path"]))
            filled = dict(np.load(inpaint_seal[f"EVAL_V3/{sid}/{role}/{qid}"]["path"]))
            if not (
                np.array_equal(arrays["hit"], sealed["hit"])
                and np.array_equal(arrays["near"], sealed["near"])
            ):
                raise AssertionError(f"Replay hit/near mask differs for {sid}/{role}/{qid}")
            pairs = [
                (arrays["REPROJ_NN"], sealed["REPROJ_NN"]),
                (arrays["REPROJ_HARMONIC"], filled["REPROJ_HARMONIC"]),
            ]
            pairs += [(arrays[f"C1_{seed}"], sealed[f"C1_{seed}"]) for seed in models]
            for a, b in pairs:
                worst = max(worst, float(np.max(np.abs(a - b))))
                compared += 1
    if worst > 1e-6:
        raise AssertionError(f"Replay of the sealed EVAL-V3 pipeline failed: {worst}")
    return {"scenes": REPLAY_SCENES, "arrays_compared": compared, "max_abs_deviation": worst}


def seal(root, cohort, v11=V11, inpaint=INPAINT):
    root = Path(root).resolve()
    verify(root, v11)
    out = root / "raw" / cohort
    if (out / "prediction_seal.json").exists():
        raise FileExistsError("Each cohort is sealed once")
    if (root / "raw" / "query_rows.json").exists():
        raise PermissionError("No cohort may be sealed after scoring")
    config, models = carriers(v11)
    constant = config["eval_v3"]["reprojection_fill_constant_m"]
    if config["eval_v3"]["near_px"] != inpainting().NEAR_PX:
        raise PermissionError("NEAR definition must be V11's")
    manifest_path, role_name, split = COHORTS[cohort]
    manifest = read(manifest_path)
    if any(r["split"] != split for r in manifest["scenes"]):
        raise PermissionError(f"{cohort}: {split} records only")
    replayed = replay(models, constant, v11, inpaint)
    print("REPLAY_OK", cohort, json.dumps(replayed), flush=True)
    access, entries, states = [], {}, {}
    for number, record in enumerate(manifest["scenes"], 1):
        sid = record["scene_id"]
        scene = predict(
            record,
            manifest,
            manifest_path.parent,
            models,
            constant,
            role_name,
            access,
            cohort.lower(),
            states,
        )
        for (role, qid), arrays in scene.items():
            path = out / "predictions" / f"{sid}_{role}_{qid}.npz"
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, **arrays)
            entries[f"{sid}/{role}/{qid}"] = {"path": str(path), "sha256": sha(path)}
        print("SEALED_SCENE", cohort, number, len(manifest["scenes"]), sid, flush=True)
    if any(
        "depth" in e.get("channels", []) and e.get("purpose") != "CONTEXT_ONLY_RGBD" for e in access
    ):
        raise AssertionError("Only context depth may be read before the seal")
    write(out / "state_hashes.json", states)
    write(out / "GT_access_before_seal.json", access)
    write(
        out / "prediction_seal.json",
        {
            "cohort": cohort,
            "role": role_name,
            "predictions": entries,
            "state_hashes_sha256": sha(out / "state_hashes.json"),
            "query_depth_read": False,
            "near_px": config["eval_v3"]["near_px"],
            "reprojection_fill_constant_m": constant,
            "replay_on_eval_v3": replayed,
        },
    )
    print("PREDICTIONS_SEALED", cohort, len(entries), flush=True)


def absrel(pred, gt, mask):
    return float(np.mean(np.abs(pred[mask] - gt[mask]) / gt[mask])) if mask.any() else None


def score(root):
    root = Path(root).resolve()
    lock_ = verify(root)
    seals = {c: read(root / "raw" / c / "prediction_seal.json") for c in COHORTS}
    seeds = read(lock_["input_sha256"]["v11_contract"]["path"])["seeds"]
    rows, flags, access = [], [], []
    for cohort, (manifest_path, role_name, _) in COHORTS.items():
        seal_ = seals[cohort]
        constant = seal_["reprojection_fill_constant_m"]
        for record in read(manifest_path)["scenes"]:
            sid = record["scene_id"]
            frames = {f["frame_id"]: f for f in record["frames"]}
            for role in ("A", "B"):
                for qid in record["roles"][role_name]:
                    entry = seal_["predictions"][f"{sid}/{role}/{qid}"]
                    if sha(entry["path"]) != entry["sha256"]:
                        raise PermissionError("A sealed prediction changed")
                    arrays = dict(np.load(entry["path"]))
                    access.append(
                        {
                            "cohort": cohort,
                            "scene_id": sid,
                            "frame_id": qid,
                            "purpose": "POST_SEAL_QUERY_DEPTH",
                        }
                    )
                    gt = np.load(frames[qid]["depth"]).squeeze().astype(np.float64)
                    valid = np.isfinite(gt) & (gt > 0)
                    near = arrays["near"].astype(bool)
                    far = valid & ~near
                    harmonic = arrays["REPROJ_HARMONIC"]
                    inside = (
                        arrays["BOX_HIT"].astype(bool)
                        & (gt >= arrays["BOX_NEAR"])
                        & (gt <= arrays["BOX_EXIT"])
                    )
                    masks = {"ALL": valid, "NEAR": valid & near, "FAR": far}
                    predictions = {
                        ("CONSTANT", 0): np.full(gt.shape, constant, dtype=np.float64),
                        ("REPROJ_NN", 0): arrays["REPROJ_NN"],
                        ("REPROJ_HARMONIC", 0): harmonic,
                    }
                    for seed in seeds:
                        carrier = arrays[f"C1_{seed}"]
                        opacity = arrays[f"OPACITY_{seed}"].astype(np.float64)
                        predictions["C1", seed] = carrier
                        predictions["HFILL8", seed] = np.where(near, harmonic, carrier)
                        predictions["AHFILL8", seed] = abstain(
                            near, opacity, harmonic, carrier, THRESHOLD
                        )
                        for t in DESCRIPTIVE_THRESHOLDS:
                            predictions[f"AHFILL8_t{t}", seed] = abstain(
                                near, opacity, harmonic, carrier, t
                            )
                        flagged = far & (opacity < THRESHOLD)
                        flags.append(
                            {
                                "cohort": cohort,
                                "scene_id": sid,
                                "role": role,
                                "query_id": qid,
                                "seed": seed,
                                "surface_in_volume": float(inside[valid].mean()),
                                "far_fraction": float(far.sum() / max(valid.sum(), 1)),
                                "flagged_share_of_far": float(flagged.sum() / max(far.sum(), 1)),
                                "flagged": float(flagged.sum()),
                                "flagged_outside": float((flagged & ~inside).sum()),
                                "far_outside": float((far & ~inside).sum()),
                            }
                        )
                    for (method, seed), value in predictions.items():
                        row = {
                            "cohort": cohort,
                            "scene_id": sid,
                            "role": role,
                            "query_id": qid,
                            "method": method,
                            "seed": seed,
                        }
                        for region, mask in masks.items():
                            row[f"absrel_{region}"] = absrel(value, gt, mask)
                        row["depth_absrel"] = depth_metrics(value, gt)["depth_absrel"]
                        row["far_fraction"] = float(far.sum() / max(valid.sum(), 1))
                        rows.append(row)
    write(root / "raw" / "query_rows.json", rows)
    write(root / "raw" / "flag_rows.json", flags)
    write(root / "raw" / "GT_access_after_seal.json", access)
    print("SCORED", len(rows), flush=True)
    return rows


def cohort_block(rows, flags):
    stage, label = host(), inpainting().geometry().label

    def values(method, region):
        return stage.scene_values(rows, method, f"absrel_{region}")

    methods = sorted({r["method"] for r in rows})
    block = {
        "absolute_absrel": {
            m: {region: stage.summary(values(m, region)) for region in REGIONS} for m in methods
        },
        "n_scenes": len({r["scene_id"] for r in rows}),
        "contrasts": {},
    }
    for name, (a, ra, b, rb) in CONTRASTS.items():
        stats = stage.contrast(values(a, ra), values(b, rb))
        entry = {"gain": stats, "definition": f"AbsRel {ra}({a}) - AbsRel {rb}({b})"}
        if stats is not None:
            entry["label"] = label(stats)
            entry["gate"], entry["gate_checks"] = stage.gate(stats)
        block["contrasts"][name] = entry
    per_scene = {}
    for f in flags:
        per_scene.setdefault(f["scene_id"], []).append(f)

    def scene_mean(key):
        return float(np.mean([np.mean([f[key] for f in v]) for v in per_scene.values()]))

    pooled = {
        key: sum(f[key] for f in flags) for key in ("flagged", "flagged_outside", "far_outside")
    }
    block["coverage"] = {
        "surface_in_volume": scene_mean("surface_in_volume"),
        "far_fraction": scene_mean("far_fraction"),
        "flagged_share_of_far": scene_mean("flagged_share_of_far"),
        "precision_outside_volume": pooled["flagged_outside"] / max(pooled["flagged"], 1),
        "recall_outside_volume": pooled["flagged_outside"] / max(pooled["far_outside"], 1),
    }
    return block


def analyze_rows(rows, flags):
    result = {"cohorts": {}}
    for cohort in COHORTS:
        subset = [r for r in rows if r["cohort"] == cohort]
        if subset:
            result["cohorts"][cohort] = cohort_block(
                subset, [f for f in flags if f["cohort"] == cohort]
            )
    primary = result["cohorts"][PRIMARY]["contrasts"]
    p1 = primary["ABSTAIN_GAIN"].get("gate", "UNDEFINED")
    p2 = primary["HARMONIC_VS_AHFILL8"].get("label", "UNDEFINED")
    first, second = p1 == "SUPPORTED", p2 == "ABOVE"
    result |= {
        "ABSTAIN_STATUS": p1,
        "HARMONIC_LABEL": p2,
        "INTERPRETATION_BRANCH": "A" if first and second else "B" if first or second else "C",
    }
    return result


def analyze(root):
    root = Path(root).resolve()
    rows_path, flags_path = root / "raw" / "query_rows.json", root / "raw" / "flag_rows.json"
    result = analyze_rows(read(rows_path), read(flags_path))
    result["replay_on_eval_v3"] = {
        c: read(root / "raw" / c / "prediction_seal.json")["replay_on_eval_v3"] for c in COHORTS
    }
    result["input_sha256"] = {str(p): sha(p) for p in (rows_path, flags_path)}
    result["final_holdout_opened"] = False
    write(root / "volume_results.json", result)
    report(root, result)
    return result


def fmt(stats):
    if not stats:
        return "UNDEFINED"
    lo, hi = stats["ci95"]
    return f"{stats['mean']:+.4f} [{lo:+.4f}, {hi:+.4f}]，场景 {stats['positive']}/{stats['tied']}/{stats['negative']}"


def report(root, result):
    primary = result["cohorts"][PRIMARY]["contrasts"]
    worst = max(r["max_abs_deviation"] for r in result["replay_on_eval_v3"].values())
    lines = [
        f"# {EXPERIMENT}",
        "",
        "状态体积之外的弃权回退（不训练）：冻结的 V11 C1 在渲染不透明度 < 0.5 的 FAR 像素上弃权，改用调和插值补洞（AHFILL8）。主队列 EVAL-FAR 的 query 远离上下文帧，此前从未被任何模型读取。协议见 [PROTOCOL.md](PROTOCOL.md)。",
        "",
        f"- **P1 ABSTAIN_GAIN（HFILL8 − AHFILL8）：** {fmt(primary['ABSTAIN_GAIN']['gain'])}，{result['ABSTAIN_STATUS']}",
        f"- **P2 调和插值补洞 − AHFILL8：** {fmt(primary['HARMONIC_VS_AHFILL8']['gain'])}，{result['HARMONIC_LABEL']}",
        f"- **分支 {result['INTERPRETATION_BRANCH']}**；复现 V11 封存预测的最大差 {worst}",
    ]
    for cohort, block in result["cohorts"].items():
        cov = block["coverage"]
        lines += [
            "",
            f"## {cohort}（{block['n_scenes']} 个场景）",
            "",
            f"GT 表面落在体积内 {cov['surface_in_volume']:.1%}；FAR 像素占 {cov['far_fraction']:.1%}，其中被标记 {cov['flagged_share_of_far']:.1%}；标记对体积外像素的精度 {cov['precision_outside_volume']:.2f}、召回 {cov['recall_outside_volume']:.2f}。",
            "",
            "| 方法 | ALL | NEAR | FAR |",
            "|---|---:|---:|---:|",
        ]
        for method in METHODS:
            cells = [block["absolute_absrel"].get(method, {}).get(region) for region in REGIONS]
            lines.append(
                f"| {method} | "
                + " | ".join("—" if not c else f"{c['mean']:.4f}" for c in cells)
                + " |"
            )
        lines += [
            "",
            "| 对比 | 定义 | 差值 [95% CI]，场景 好/平/差 | 标签 | gate |",
            "|---|---|---|---|---|",
        ]
        for name, entry in block["contrasts"].items():
            lines.append(
                f"| {name} | {entry['definition']} | {fmt(entry['gain'])} | {entry.get('label', 'UNDEFINED')} | {entry.get('gate', 'UNDEFINED')} |"
            )
    lines += [
        "",
        "**限制：** EVAL-FAR 的场景已被用于近处 query，它是机制队列，不作资格验证；阈值 0.5 来自已用队列上的探索性分析；受保护的 final holdout 未打开。",
    ]
    (root / "README.md").write_text("\n".join(lines) + "\n")
    summary = [
        "RGBD VOLUME V13",
        f"ABSTAIN_STATUS={result['ABSTAIN_STATUS']}",
        f"HARMONIC_LABEL={result['HARMONIC_LABEL']}",
        f"INTERPRETATION_BRANCH={result['INTERPRETATION_BRANCH']}",
        f"REPLAY_MAX_ABS_DEVIATION={worst}",
    ]
    for cohort, block in result["cohorts"].items():
        for method in ("CONSTANT", "REPROJ_HARMONIC", "C1", "HFILL8", "AHFILL8"):
            value = block["absolute_absrel"].get(method, {}).get("ALL")
            if value:
                summary.append(f"{cohort}_{method}_ALL={value['mean']:.6f}")
        for name in ("ABSTAIN_GAIN", "HARMONIC_VS_AHFILL8", "CONSTANT_VS_C1"):
            entry = block["contrasts"][name]
            if entry["gain"]:
                summary.append(
                    f"{cohort}_{name}={entry['gain']['mean']:.6f} CI={entry['gain']['ci95']} LABEL={entry['label']} GATE={entry['gate']}"
                )
        summary.append(f"{cohort}_SURFACE_IN_VOLUME={block['coverage']['surface_in_volume']:.4f}")
    summary.append("FINAL_HOLDOUT_TOUCHED=false")
    (root / "terminal_summary.txt").write_text("\n".join(summary) + "\n")
    print("\n".join(summary), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("lock", "seal", "score", "analyze"), required=True)
    parser.add_argument("--cohort", choices=tuple(COHORTS))
    parser.add_argument("--root", type=Path, default=Path("outputs") / EXPERIMENT)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.stage == "seal":
        if args.cohort is None:
            parser.error("--stage seal needs --cohort")
        seal(args.root, args.cohort)
    else:
        {"lock": lock, "score": score, "analyze": analyze}[args.stage](args.root)


if __name__ == "__main__":
    main()
