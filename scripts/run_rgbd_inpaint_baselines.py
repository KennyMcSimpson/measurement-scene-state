#!/usr/bin/env python3
"""Classical geometric inpainting versus the learned completion of V11 (no training).

Every earlier comparison filled the holes of the context-depth reprojection with the nearest
filled pixel (REPROJ_NN), a weak completion baseline. This experiment keeps the same 1-px
z-buffer and replaces the hole filling by classical inpainting of inverse planar depth:
REPROJ_HARMONIC (primary; 4-neighbour Laplace equation, hits as Dirichlet data, zero-flux image
border; exact for a single plane) and REPROJ_TELEA (descriptive; OpenCV Telea, radius 5 px).
The learned side is V11's sealed EVAL-V3 predictions (C0 WIDTH8 and C1 WIDTH32) and its sealed
DEV predictions; nothing is retrained or re-rendered.

`--stage lock` freezes this protocol and code before any V11 result exists. `--stage run`
(after V11 finalizes) computes and seals every inpainting prediction from sealed inputs and
query camera metadata only, then reads query depth and scores every method with one function.
"""
# ruff: noqa: E501 -- Chinese scientific report prose

from __future__ import annotations

import argparse
import datetime
import hashlib
import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy import ndimage, sparse
from scipy.sparse.linalg import spsolve

from mcss.geometry import pixel_grid
from mcss.mechanism_pilot.readout_reanalysis import depth_metrics

EXPERIMENT = "EXP-3D-RGBD-INPAINT-BASELINES-V1"
V11 = Path("outputs/EXP-3D-RGBD-COMPLETION-V11")
GEOMETRIC = Path("outputs/EXP-3D-RGBD-GEOMETRIC-BASELINES-V1")
V7 = Path("outputs/EXP-3D-RGBD-DEPTH-BOUNDS-V7")
SCRIPTS = Path(__file__).resolve().parent
NEAR_PX = 8
TELEA_RADIUS = 5
TELEA_SCALE = 1000.0
VARIANTS = ("C0", "C1")
PRIMARY_CARRIER = "C1"
REGIONS = ("ALL", "NEAR", "FAR")
DRAWS, SEED = 10000, 20260928


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


def geometry():
    """The frozen geometric-baseline helpers (query camera, label rule)."""
    return module("inpaint_geometry", "run_rgbd_geometric_baselines.py")


def v11_analysis():
    """V11's frozen EVAL-V3 aggregation, bootstrap and gate (scene values, contrasts)."""
    return module("inpaint_v11_eval", "evaluate_rgbd_completion_eval_v3.py")


# ---------------------------------------------------------------- inpainting (no learned part)
def ray_cosine(camera):
    """|cos| between every pixel ray and the optical axis, [H, W] float64 (pixel_grid convention)."""
    camera = camera.to(dtype=torch.float64)
    height, width = camera.image_size
    pixels = pixel_grid(height, width, device=camera.device, dtype=torch.float64)
    directions = torch.einsum("ij,hwj->hwi", torch.linalg.inv(camera.intrinsics), pixels)
    directions = directions / directions.norm(dim=-1, keepdim=True)
    return directions[..., 2].abs().numpy()


def harmonic_fill(depth, hit, cosine, constant):
    """Hits kept; holes get the 4-neighbour Laplace solution in inverse planar depth.

    `depth` is ray distance, defined where `hit`; no hit at all gives the TRAIN constant.
    """
    if not hit.any():
        return np.full(hit.shape, constant, dtype=np.float64)
    height, width = hit.shape
    inverse = np.zeros(hit.shape, dtype=np.float64)
    inverse[hit] = 1.0 / (depth[hit] * cosine[hit])
    unknown = ~hit
    out = np.where(hit, depth, 0.0).astype(np.float64)
    n = int(unknown.sum())
    if n == 0:
        return out
    index = -np.ones(hit.shape, dtype=np.int64)
    index[unknown] = np.arange(n)
    ys, xs = np.nonzero(unknown)
    diagonal, rhs = np.zeros(n), np.zeros(n)
    rows, cols, values = [np.arange(n)], [np.arange(n)], []
    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        qy, qx = ys + dy, xs + dx
        inside = (qy >= 0) & (qy < height) & (qx >= 0) & (qx < width)
        p, qy, qx = index[ys[inside], xs[inside]], qy[inside], qx[inside]
        diagonal[p] += 1.0
        free = unknown[qy, qx]
        rows.append(p[free])
        cols.append(index[qy[free], qx[free]])
        values.append(-np.ones(int(free.sum())))
        np.add.at(rhs, p[~free], inverse[qy[~free], qx[~free]])
    matrix = sparse.csc_matrix(
        (np.concatenate([diagonal, *values]), (np.concatenate(rows), np.concatenate(cols))),
        shape=(n, n),
    )
    solution = spsolve(matrix, rhs)
    out[unknown] = 1.0 / (solution * cosine[unknown])
    return out


def telea_fill(depth, hit, cosine, constant):
    """OpenCV Telea on inverse planar depth in 1/km; a non-positive fill falls back to 1/constant.

    Returns (prediction, number of fallback pixels).
    """
    if not hit.any():
        return np.full(hit.shape, constant, dtype=np.float64), 0
    safe = np.where(hit, depth, 1.0)
    inverse = np.where(hit, TELEA_SCALE / (safe * cosine), 0.0).astype(np.float32)
    filled = cv2.inpaint(inverse, (~hit).astype(np.uint8), TELEA_RADIUS, cv2.INPAINT_TELEA)
    filled = filled.astype(np.float64) / TELEA_SCALE
    bad = ~hit & ~(np.isfinite(filled) & (filled > 0))
    filled[bad] = 1.0 / constant
    out = np.where(hit, depth, 0.0).astype(np.float64)
    out[~hit] = 1.0 / (filled[~hit] * cosine[~hit])
    return out, int(bad.sum())


def near_mask(hit):
    if not hit.any():
        return np.zeros(hit.shape, dtype=bool)
    return ndimage.distance_transform_edt(~hit) <= NEAR_PX


# ---------------------------------------------------------------- inputs
def inputs(v11=V11, geometric=GEOMETRIC, v7=V7):
    """Every frozen input path; the lock hashes them before V11 has any result."""
    contract = read(v11 / "training_contract.json")
    return {
        "v11_contract": v11 / "training_contract.json",
        "v11_protocol": v11 / "PROTOCOL.md",
        "v11_scene_split": v11 / "scene_split.json",
        "eval_v3_manifest": Path(contract["eval_v3"]["manifest"]),
        "dev_manifest": v7 / "scene_split.json",
        "dev_baseline_seal": geometric / "raw" / "baseline_seal.json",
        "geometric_script": SCRIPTS / "run_rgbd_geometric_baselines.py",
        "v11_eval_script": SCRIPTS / "evaluate_rgbd_completion_eval_v3.py",
    }


def dev_records(v11=V11, v7=V7):
    """DEV records; V11 must use exactly V7's DEV records (same frames and roles)."""
    mine = {
        r["scene_id"]: r for r in read(v11 / "scene_split.json")["scenes"] if r["split"] == "DEV"
    }
    base = read(v7 / "scene_split.json")
    theirs = {r["scene_id"]: r for r in base["scenes"] if r["split"] == "DEV"}
    if json.dumps(mine, sort_keys=True) != json.dumps(theirs, sort_keys=True):
        raise PermissionError("V11 DEV records differ from the V7 DEV records")
    return theirs, base["image_size"]


# ---------------------------------------------------------------- stages
def lock(root, v11=V11, geometric=GEOMETRIC, v7=V7):
    root = Path(root).resolve()
    if root.name != EXPERIMENT:
        raise PermissionError("Exact experiment directory required")
    if (root / "lock.json").exists():
        raise FileExistsError("Never re-lock")
    if not (root / "PROTOCOL.md").is_file():
        raise PermissionError("Protocol text required before locking")
    for name in ("eval_v3_results.json", "integrity.json", "terminal_summary.txt"):
        if (v11 / name).exists():
            raise PermissionError(f"V11 result {name} exists; this experiment must lock before it")
    if (v11 / "raw").exists():
        raise PermissionError("V11 evaluation outputs exist; lock before any V11 evaluation")
    contract = read(v11 / "training_contract.json")
    if contract["eval_v3"]["near_px"] != NEAR_PX:
        raise PermissionError("The FAR definition must equal V11's")
    if sha(contract["eval_v3"]["manifest"]) != contract["eval_v3"]["manifest_sha256"]:
        raise PermissionError("EVAL-V3 manifest changed")
    dev_records(v11, v7)
    paths = {
        **inputs(v11, geometric, v7),
        "protocol": root / "PROTOCOL.md",
        "script": Path(__file__).resolve(),
    }
    write(
        root / "lock.json",
        {
            "status": "FROZEN_BEFORE_ANY_V11_RESULT",
            "experiment": EXPERIMENT,
            "created_utc": datetime.datetime.now(datetime.UTC).isoformat(),
            "primary": "AbsRel(REPROJ_HARMONIC) - AbsRel(HFILL8 of V11 C1) on EVAL-V3, all valid pixels",
            "near_px": NEAR_PX,
            "telea": {"radius_px": TELEA_RADIUS, "inverse_depth_scale": TELEA_SCALE},
            "reprojection_fill_constant_m": contract["eval_v3"]["reprojection_fill_constant_m"],
            "input_sha256": {
                name: {"path": str(Path(p).resolve()), "sha256": sha(p)}
                for name, p in paths.items()
            },
            "v11_results_existed_at_lock": False,
            "query_gt_read_before_lock": False,
            "final_holdout_opened": False,
        },
    )
    print("LOCKED", EXPERIMENT, flush=True)


def verify(root, v11=V11):
    lock_ = read(root / "lock.json")
    for name, entry in lock_["input_sha256"].items():
        if sha(entry["path"]) != entry["sha256"]:
            raise PermissionError(f"Locked input changed: {name}")
    integrity = read(v11 / "integrity.json")
    if integrity["status"] != "PASS":
        raise PermissionError("V11 must be finalized with integrity PASS")
    return lock_


def episodes(v11=V11, geometric=GEOMETRIC, v7=V7):
    """(cohort, key, record, role, qid, image_size, loader) for every scored query.

    The loader returns (hit, raw ray distance on hits, {method: carrier or reference array})
    from sealed files only, checking every hash.
    """
    contract = read(v11 / "training_contract.json")
    manifest = read(contract["eval_v3"]["manifest"])
    seal = read(v11 / "raw" / "eval_v3" / "prediction_seal.json")
    if seal["near_px"] != NEAR_PX or seal.get("query_depth_read") is not False:
        raise PermissionError("V11 EVAL-V3 seal does not match this protocol")
    seeds = contract["seeds"]
    for record in manifest["scenes"]:
        sid = record["scene_id"]
        for role in ("A", "B"):
            for qid in record["roles"]["primary_query"]:
                entry = seal["predictions"][f"{sid}/{role}/{qid}"]

                def load(entry=entry):
                    if sha(entry["path"]) != entry["sha256"]:
                        raise PermissionError("A sealed V11 prediction changed")
                    arrays = dict(np.load(entry["path"]))
                    hit = arrays["hit"].astype(bool)
                    if not np.array_equal(near_mask(hit), arrays["near"].astype(bool)):
                        raise AssertionError(
                            "Sealed NEAR mask does not follow from the sealed hits"
                        )
                    carriers = {
                        f"{v}_{s}": arrays[f"{v}_{s}"].astype(np.float64)
                        for v in VARIANTS
                        for s in seeds
                    }
                    return (
                        hit,
                        np.where(hit, arrays["REPROJ_NN"], np.inf),
                        arrays["REPROJ_NN"],
                        carriers,
                    )

                yield (
                    "EVAL_V3",
                    f"EVAL_V3/{sid}/{role}/{qid}",
                    record,
                    role,
                    qid,
                    manifest["image_size"],
                    load,
                )
    records, image_size = dev_records(v11, v7)
    baseline_seal = read(geometric / "raw" / "baseline_seal.json")
    dev_rows = {}
    for variant in VARIANTS:
        for seed in seeds:
            directory = v11 / "raw" / "selected_dev" / f"{variant}_{seed}"
            for row in read(directory / "query_results.json"):
                if row["method"] == "direct":
                    dev_rows[variant, seed, row["scene_id"], row["role"], row["query_id"]] = (
                        directory / row["prediction_path"],
                        row["prediction_file_sha256"],
                    )
    for sid, record in sorted(records.items()):
        for role in ("A", "B"):
            for qid in record["roles"]["primary_query"]:
                entry = baseline_seal[f"DEV/{sid}/{role}/{qid}"]

                def load(entry=entry, sid=sid, role=role, qid=qid):
                    if sha(entry["path"]) != entry["sha256"]:
                        raise PermissionError("A sealed DEV reprojection changed")
                    arrays = dict(np.load(entry["path"]))
                    hit = arrays["hit"].astype(bool)
                    carriers = {}
                    for variant in VARIANTS:
                        for seed in seeds:
                            path, digest = dev_rows[variant, seed, sid, role, qid]
                            if sha(path) != digest:
                                raise PermissionError("A sealed V11 DEV prediction changed")
                            carriers[f"{variant}_{seed}"] = np.load(path)["depth"].astype(
                                np.float64
                            )
                    return (
                        hit,
                        np.where(hit, arrays["reproj"], np.inf),
                        arrays["REPROJ_NN"],
                        carriers,
                    )

                yield "DEV", f"DEV/{sid}/{role}/{qid}", record, role, qid, image_size, load


def compute(root, access, **paths):
    """Seal every inpainting prediction before any query depth is read."""
    geo = geometry()
    constant = read(root / "lock.json")["reprojection_fill_constant_m"]
    sealed, bad_total = {}, 0
    for cohort, key, record, _role, qid, image_size, load in episodes(**paths):
        hit, raw, _, _ = load()
        access.append({"key": key, "purpose": "QUERY_CAMERA_ONLY"})
        cosine = ray_cosine(geo.query_camera(record, qid, image_size))
        harmonic = harmonic_fill(raw, hit, cosine, constant)
        telea, bad = telea_fill(raw, hit, cosine, constant)
        bad_total += bad
        path = root / "raw" / "inpaint" / cohort / (key.split("/", 1)[1].replace("/", "_") + ".npz")
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, REPROJ_HARMONIC=harmonic, REPROJ_TELEA=telea)
        sealed[key] = {"path": str(path), "sha256": sha(path), "telea_fallback_pixels": bad}
    return sealed, bad_total


def absrel(prediction, gt, mask):
    return float(np.mean(np.abs(prediction[mask] - gt[mask]) / gt[mask])) if mask.any() else None


def score(root, sealed, access, **paths):
    rows = []
    for cohort, key, record, role, qid, _image_size, load in episodes(**paths):
        hit, _, reproj_nn, carriers = load()
        entry = sealed[key]
        if sha(entry["path"]) != entry["sha256"]:
            raise PermissionError("A sealed inpainting prediction changed")
        inpaint = dict(np.load(entry["path"]))
        frames = {f["frame_id"]: f for f in record["frames"]}
        access.append({"key": key, "purpose": "POST_SEAL_QUERY_DEPTH"})
        gt = np.load(frames[qid]["depth"]).squeeze().astype(np.float64)
        valid = np.isfinite(gt) & (gt > 0)
        near = near_mask(hit)
        masks = {"ALL": valid, "NEAR": valid & near, "FAR": valid & ~near}
        predictions = {
            ("REPROJ_NN", 0): reproj_nn,
            ("REPROJ_HARMONIC", 0): inpaint["REPROJ_HARMONIC"],
            ("REPROJ_TELEA", 0): inpaint["REPROJ_TELEA"],
        }
        for name, carrier in carriers.items():
            variant, seed = name.split("_")
            predictions[variant, int(seed)] = carrier
            predictions[f"FILL8_{variant}", int(seed)] = np.where(near, reproj_nn, carrier)
            predictions[f"HFILL8_{variant}", int(seed)] = np.where(
                near, inpaint["REPROJ_HARMONIC"], carrier
            )
            predictions[f"TFILL8_{variant}", int(seed)] = np.where(
                near, inpaint["REPROJ_TELEA"], carrier
            )
        for (method, seed), value in predictions.items():
            row = {"cohort": cohort, "scene_id": record["scene_id"], "role": role, "query_id": qid}
            row.update({"method": method, "seed": seed})
            for region, mask in masks.items():
                row[f"absrel_{region}"] = absrel(value, gt, mask)
            row["depth_absrel"] = depth_metrics(value, gt)["depth_absrel"]
            row["far_fraction"] = float(masks["FAR"].sum() / max(valid.sum(), 1))
            rows.append(row)
    return rows


def replay_v11(rows, v11=V11):
    """Our EVAL-V3 values for REPROJ_NN, C0, C1 and FILL8 must reproduce V11's scored rows."""
    theirs = {
        (r["scene_id"], r["role"], r["query_id"], r["method"], r["seed"]): r
        for r in read(v11 / "raw" / "eval_v3" / "query_rows.json")
    }
    worst, compared = 0.0, 0
    for r in rows:
        key = (r["scene_id"], r["role"], r["query_id"], r["method"], r["seed"])
        if r["cohort"] != "EVAL_V3" or key not in theirs:
            continue
        for field in ("absrel_ALL", "absrel_NEAR", "absrel_FAR", "depth_absrel"):
            a, b = r[field], theirs[key][field]
            if (a is None) != (b is None):
                raise AssertionError(f"Region presence differs from V11 for {key}")
            if a is not None:
                worst = max(worst, abs(a - b))
        compared += 1
    expected = sum(
        1 for key in theirs if key[3] in {"REPROJ_NN", "C0", "C1", "FILL8_C0", "FILL8_C1"}
    )
    if compared != expected or worst > 1e-9:
        raise AssertionError(f"V11 replay failed: compared {compared}/{expected}, worst {worst}")
    return {"compared_rows": compared, "max_abs_deviation": worst}


def analyze(rows):
    host = v11_analysis()
    label = geometry().label
    result = {"cohorts": {}}
    for cohort in ("EVAL_V3", "DEV"):
        subset = [r for r in rows if r["cohort"] == cohort]
        methods = sorted({r["method"] for r in subset})

        def values(method, region, subset=subset):
            return host.scene_values(subset, method, f"absrel_{region}")

        absolute = {
            m: {region: host.summary(values(m, region)) for region in REGIONS} for m in methods
        }
        contrasts = {
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
        block = {
            "absolute_absrel": absolute,
            "contrasts": {},
            "n_scenes": len({r["scene_id"] for r in subset}),
        }
        for name, (a, ra, b, rb) in contrasts.items():
            stats = host.contrast(values(a, ra), values(b, rb))
            entry = {"gain": stats, "definition": f"AbsRel {ra}({a}) - AbsRel {rb}({b})"}
            if stats is not None:
                entry["label"] = label(stats)
                entry["strict_gate"], entry["strict_checks"] = host.gate(stats)
            block["contrasts"][name] = entry
        block["far_pixel_fraction"] = float(
            np.mean([r["far_fraction"] for r in subset if r["method"] == "REPROJ_NN"])
        )
        result["cohorts"][cohort] = block
    primary = result["cohorts"]["EVAL_V3"]["contrasts"]["HARMONIC_HYBRID_GAIN"]
    result["PRIMARY_LABEL"] = primary.get("label", "UNDEFINED")
    result["PRIMARY_STRICT_GATE"] = primary.get("strict_gate", "UNDEFINED")
    result["INTERPRETATION_BRANCH"] = {"ABOVE": "A", "NOT_DISTINGUISHABLE": "B", "BELOW": "C"}.get(
        result["PRIMARY_LABEL"], "UNDEFINED"
    )
    return result


def run(root, v11=V11, geometric=GEOMETRIC, v7=V7):
    root = Path(root).resolve()
    verify(root, v11)
    if (root / "raw" / "inpaint").exists():
        raise FileExistsError("Inpainting predictions already computed; the run happens once")
    paths = {"v11": v11, "geometric": geometric, "v7": v7}
    access = []
    sealed, bad = compute(root, access, **paths)
    write(root / "raw" / "inpaint_seal.json", sealed)
    access.append({"event": "ALL_INPAINT_PREDICTIONS_SEALED", "count": len(sealed)})
    rows = score(root, sealed, access, **paths)
    write(root / "raw" / "scored_rows.json", rows)
    write(root / "audit" / "GT_access.json", access)
    result = analyze(rows)
    result["integrity"] = {
        "v11_replay": replay_v11(rows, v11),
        "inpaint_predictions_sealed_before_gt": True,
        "sealed_predictions": len(sealed),
        "telea_fallback_pixels": bad,
        "scored_rows": len(rows),
    }
    write(root / "inpaint_results.json", result)
    report(root, result)
    return result


def fmt(stats):
    if not stats:
        return "UNDEFINED"
    lo, hi = stats["ci95"]
    return f"{stats['mean']:+.4f} [{lo:+.4f}, {hi:+.4f}]，场景 {stats['positive']}/{stats['tied']}/{stats['negative']}"


def report(root, result):
    names = {
        "REPROJ_NN": "重投影 + 最近邻补洞（V11 的几何基线）",
        "REPROJ_HARMONIC": "重投影 + 逆深度调和插值补洞（本轮主基线）",
        "REPROJ_TELEA": "重投影 + Telea 补洞（描述性）",
        "C0": "V11 C0 carrier（hidden 8）",
        "C1": "V11 C1 carrier（hidden 32）",
        "FILL8_C1": "NEAR 用 REPROJ_NN，FAR 用 C1（V11 的组合）",
        "HFILL8_C1": "NEAR 用 REPROJ_HARMONIC，FAR 用 C1（本轮主方法）",
        "HFILL8_C0": "NEAR 用 REPROJ_HARMONIC，FAR 用 C0",
        "TFILL8_C1": "NEAR 用 REPROJ_TELEA，FAR 用 C1",
    }
    branch = result["INTERPRETATION_BRANCH"]
    lines = [
        f"# {EXPERIMENT}",
        "",
        "问题：在同样的测试时输入下，学到的补全（V11 已封存的预测）能否胜过经典的几何补洞？此前所有比较都用最近邻补洞（REPROJ_NN），这是较弱的补全基线。本轮不训练、不重新渲染任何模型。",
        "",
        f"主对比（EVAL-V3）：AbsRel(REPROJ_HARMONIC) − AbsRel(HFILL8_C1) = {fmt(result['cohorts']['EVAL_V3']['contrasts']['HARMONIC_HYBRID_GAIN']['gain'])}；标签 {result['PRIMARY_LABEL']}，严格 gate {result['PRIMARY_STRICT_GATE']}，分支 {branch}。",
    ]
    for cohort, block in result["cohorts"].items():
        lines += [
            "",
            f"## {cohort}（{block['n_scenes']} 个场景，FAR 像素占 {block['far_pixel_fraction']:.1%}）",
            "",
            "| 方法 | 说明 | ALL | NEAR | FAR |",
            "|---|---|---:|---:|---:|",
        ]
        for method, text in names.items():
            cells = []
            for region in REGIONS:
                value = block["absolute_absrel"].get(method, {}).get(region)
                cells.append("—" if not value else f"{value['mean']:.4f}")
            lines.append(f"| {method} | {text} | " + " | ".join(cells) + " |")
        lines += ["", "| 对比 | 定义 | 差值 [95% CI]，场景 好/平/差 | 标签 |", "|---|---|---|---|"]
        for name, entry in block["contrasts"].items():
            lines.append(
                f"| {name} | {entry['definition']} | {fmt(entry['gain'])} | {entry.get('label', 'UNDEFINED')} |"
            )
    lines += [
        "",
        {
            "A": "分支 A：即使与经典几何补洞相比，学到的补全在可部署的组合中仍然更好；补全优势不是弱基线造成的。",
            "B": "分支 B：学到的补全与经典几何补洞无法区分；此前相对 REPROJ_NN 的补全优势，至少一部分来自最近邻补洞太弱。",
            "C": "分支 C：经典几何补洞优于学到的补全；此前相对 REPROJ_NN 的补全优势来自弱基线。",
            "UNDEFINED": "主对比无法计算（没有共同场景）。",
        }[branch],
        "",
        "说明：EVAL-V3 是机制队列（按场景和源资产与训练集不相交，可能共享 volume），不是资格验证；DEV 只作描述；受保护的 final holdout 未打开。",
    ]
    (root / "README.md").write_text("\n".join(lines) + "\n")
    summary = [
        "RGBD INPAINT BASELINES V1",
        f"PRIMARY_LABEL={result['PRIMARY_LABEL']}",
        f"PRIMARY_STRICT_GATE={result['PRIMARY_STRICT_GATE']}",
        f"INTERPRETATION_BRANCH={branch}",
    ]
    for cohort, block in result["cohorts"].items():
        for method in names:
            value = block["absolute_absrel"].get(method, {}).get("ALL")
            if value:
                summary.append(f"{cohort}_{method}_ABSREL_ALL={value['mean']:.6f}")
        for name, entry in block["contrasts"].items():
            if entry["gain"]:
                summary.append(
                    f"{cohort}_{name}={entry['gain']['mean']:.6f} CI={entry['gain']['ci95']} LABEL={entry['label']}"
                )
    summary += [
        f"V11_REPLAY_MAX_DEVIATION={result['integrity']['v11_replay']['max_abs_deviation']}",
        "FINAL_HOLDOUT_TOUCHED=false",
    ]
    (root / "terminal_summary.txt").write_text("\n".join(summary) + "\n")
    print("\n".join(summary), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("lock", "run"), required=True)
    parser.add_argument("--root", type=Path, default=Path("outputs") / EXPERIMENT)
    args = parser.parse_args()
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    (lock if args.stage == "lock" else run)(args.root)


if __name__ == "__main__":
    main()
