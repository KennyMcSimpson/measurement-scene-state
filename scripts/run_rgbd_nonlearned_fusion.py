#!/usr/bin/env python3
"""Non-learned RGB-D fusion baseline: lock, TRAIN24 fit, sealed DEV/FRESH-V1 evaluation, report."""
# ruff: noqa: E501 -- Chinese report prose

from __future__ import annotations

import argparse
import datetime
import itertools
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from mcss.geometry import transform_cameras
from mcss.measurements import FixedMeasurementRenderer
from mcss.mechanism_pilot.direct_capacity_contracts import CapacityEvaluator, StateSealBarrier
from mcss.mechanism_pilot.direct_capacity_evaluation import _metrics
from mcss.mechanism_pilot.direct_capacity_statistics import TIE, paired
from mcss.mechanism_pilot.geometry_carrier_statistics import _evidence, _macro
from mcss.mechanism_pilot.rgbd_evidence_carrier import RGBDSceneData
from mcss.mechanism_pilot.rgbd_fresh_evaluation import FreshRGBDSceneData
from mcss.mechanism_pilot.rgbd_nonlearned_fusion import GRID_SEARCH, fused_state
from mcss.mechanism_pilot.small_training import sha, write_json

EXPERIMENT = "EXP-3D-RGBD-NONLEARNED-FUSION-V1"
V5 = Path("outputs/EXP-3D-RGBD-EVIDENCE-CARRIER-V5")
FRESH = Path("outputs/EXP-3D-RGBD-FRESH-QUALIFICATION-V1")
INPUTS = (
    V5 / "scene_split.json",
    V5 / "train_depth_prior.json",
    V5 / "raw/dev_query_results.json",
    V5 / "raw/dev_reference_rows.json",
    FRESH / "scene_split.json",
    FRESH / "train_depth_prior.json",
    FRESH / "raw/fresh_query_results.json",
    FRESH / "raw/fresh_reference_rows.json",
)
SOURCES = (
    Path("src/mcss/mechanism_pilot/rgbd_nonlearned_fusion.py"),
    Path("src/mcss/mechanism_pilot/rgbd_evidence_carrier.py"),
    Path("src/mcss/mechanism_pilot/rgbd_fresh_evaluation.py"),
    Path("src/mcss/measurements.py"),
    Path(__file__),
)


def lock(root):
    root.mkdir(parents=True, exist_ok=False)
    write_json(
        root / "preregistration.json",
        {
            "status": "FROZEN_BEFORE_FIT_AND_EVALUATION",
            "experiment": EXPERIMENT,
            "created_utc": datetime.datetime.now(datetime.UTC).isoformat(),
            "grid_search": GRID_SEARCH,
            "objective": "mean GT-valid query AbsRel (RAW) over TRAIN24 primary queries, roles A/B; "
            "ties -> first combination in itertools.product order of k, a, b, c",
            "evaluation": "DEV (8) and FRESH-V1 (5): every state sealed before any query GT; "
            "same renderer, bounds, anchoring and metric code as V5",
            "comparisons": {
                "LEARNED_MINUS_NLF": "AbsRel(NLF) - AbsRel(V5 C1, seed-mean); positive = learned better",
                "NLF_MINUS_CONSTANT": "AbsRel(REF_TRAIN_ABSREL_OPTIMAL) - AbsRel(NLF); positive = NLF better",
                "RGB_ONLY_MINUS_NLF": "AbsRel(NLF) - AbsRel(V5 C0); positive = RGB-only carrier better",
            },
            "labels": "ABOVE (frozen V2 evidence rule) / BELOW (mirror) / NOT_DISTINGUISHABLE; "
            "all scenes and NO_HIT-excluded scenes",
            "bootstrap": {"unit": "scene", "draws": 10000, "seed": 20260929},
            "role": "descriptive control; changes no frozen status",
            "device": "cpu",
            "input_sha256": {str(p): sha(p) for p in INPUTS},
            "source_sha256": {str(p): sha(p) for p in SOURCES},
        },
    )
    print("LOCKED", root)


def verify(root):
    lock_ = json.loads((root / "preregistration.json").read_text())
    if lock_["status"] != "FROZEN_BEFORE_FIT_AND_EVALUATION":
        raise PermissionError("Frozen lock required")
    for group in ("input_sha256", "source_sha256"):
        for name, digest in lock_[group].items():
            if sha(Path(name)) != digest:
                raise PermissionError(f"Frozen file changed: {name}")
    return lock_


def fit(manifest):
    renderer = FixedMeasurementRenderer(n_samples=64, ray_chunk_size=2048)
    records = sorted(
        (r for r in manifest["scenes"] if r["split"] == "TRAIN"), key=lambda r: r["scene_id"]
    )
    access = []
    scenes = [RGBDSceneData(r, manifest, V5, "cpu", access) for r in records]
    grid = list(itertools.product(*(GRID_SEARCH[k] for k in ("k", "a", "b", "c"))))
    results = []
    for k, a, b, c in grid:
        params = {"k": k, "a": a, "b": b, "c": c}
        values = []
        for scene in scenes:
            for role in ("A", "B"):
                state, anchor = fused_state(scene.context(role), scene.bounds[role], params)
                for slot in range(len(scene.record["roles"]["primary_query"])):
                    target = scene.target(slot)
                    cameras = transform_cameras(target.cameras, torch.linalg.inv(anchor))
                    with torch.no_grad():
                        pred = renderer(state, cameras, ("rgb", "depth", "visibility"))
                    values.append(_metrics(pred, target.rgb, target.depth)["depth_absrel"])
        results.append(
            {**params, "train24_query_absrel": float(np.mean(values)), "views": len(values)}
        )
        print("FIT", params, round(results[-1]["train24_query_absrel"], 4), flush=True)
    best = min(results, key=lambda r: r["train24_query_absrel"])
    return {k: best[k] for k in ("k", "a", "b", "c")}, results


def evaluate(manifest, base, split, scene_cls, params, access):
    records = {r["scene_id"]: r for r in manifest["scenes"] if r["split"] == split}
    forbidden = set(manifest.get("protected_scene_ids", [])) | {
        r["scene_id"] for r in manifest["scenes"] if r["split"] != split
    }
    keys = [f"{sid}/{role}" for sid in records for role in ("A", "B")]
    barrier = StateSealBarrier(keys)
    anchors = {}
    for sid, record in records.items():
        data = scene_cls(record, manifest, base, "cpu", access)
        for role in ("A", "B"):
            state, anchor = fused_state(data.context(role), data.bounds[role], params)
            barrier.seal(f"{sid}/{role}", sid, state)
            anchors[sid, role] = anchor
    barrier.assert_ready()
    access.append({"event": f"ALL_{split}_NLF_STATES_SEALED", "state_count": len(keys)})
    renderer = FixedMeasurementRenderer(n_samples=64, ray_chunk_size=2048)
    rows = []
    for sid, record in records.items():
        evaluator = CapacityEvaluator(
            record,
            manifest["image_size"],
            base,
            "cpu",
            barrier=barrier,
            capacity_scene_ids=tuple(records),
            holdout_scene_ids=tuple(forbidden),
            access_log=access,
        )
        for role in ("A", "B"):
            state = barrier.state(f"{sid}/{role}")
            for qid in record["roles"]["primary_query"]:
                batch = evaluator.query(qid)
                cameras = transform_cameras(batch.cameras, torch.linalg.inv(anchors[sid, role]))
                with torch.no_grad():
                    pred = renderer(state, cameras, ("rgb", "depth", "visibility"))
                metrics = _metrics(pred, batch.rgb, batch.depth)
                rows.append(
                    {
                        "scene_id": sid,
                        "role": role,
                        "query_id": qid,
                        "seed": 0,
                        "variant": "NLF",
                        "method": "direct",
                        "depth_absrel": metrics["depth_absrel"],
                        "depth_delta1": metrics["depth_delta1"],
                        "opacity": metrics["opacity"],
                    }
                )
    return rows


def label(stats):
    x = np.asarray(list(stats["per_scene"].values()))
    if _evidence(stats):
        return "ABOVE"
    if stats["mean"] < 0 and stats["ci95"][1] < 0 and np.mean(x <= TIE) >= 0.75:
        return "BELOW"
    return "NOT_DISTINGUISHABLE"


def compare(nlf_rows, carrier_rows, reference_rows, exclude=()):
    keep = lambda rows: [r for r in rows if r["scene_id"] not in exclude]  # noqa: E731
    nlf = _macro(keep(nlf_rows), "depth_absrel")
    c1 = _macro(
        keep([r for r in carrier_rows if r["variant"] == "C1" and r["method"] == "direct"]),
        "depth_absrel",
    )
    c0 = _macro(
        keep([r for r in carrier_rows if r["variant"] == "C0" and r["method"] == "direct"]),
        "depth_absrel",
    )
    const = _macro(
        keep(
            [
                r
                for r in reference_rows
                if r["reference"] == "REF_TRAIN_ABSREL_OPTIMAL" and r["readout"] == "RAW"
            ]
        ),
        "depth_absrel",
    )
    args = {"draws": 10000, "seed": 20260929}
    out = {
        "means": {
            k: float(np.mean(list(v.values())))
            for k, v in (("NLF", nlf), ("C1", c1), ("C0", c0), ("constant", const))
        },
        "per_scene": {
            s: {"NLF": nlf[s], "C1": c1[s], "C0": c0[s], "constant": const[s]} for s in sorted(nlf)
        },
    }
    for name, stats in (
        ("LEARNED_MINUS_NLF", paired(nlf, c1, **args)),
        ("NLF_MINUS_CONSTANT", paired(const, nlf, **args)),
        ("RGB_ONLY_MINUS_NLF", paired(nlf, c0, **args)),
    ):
        out[name] = {"stats": stats, "label": label(stats)}
    return out


def no_hit(carrier_rows):
    """Scenes whose every direct V5 prediction had zero opacity (rays outside the bounds)."""
    groups = defaultdict(list)
    for r in carrier_rows:
        if r["method"] == "direct":
            groups[r["scene_id"]].append(r["opacity"] == 0)
    return sorted(s for s, flags in groups.items() if all(flags))


def run(root):
    lock_ = verify(root)
    v5_manifest = json.loads((V5 / "scene_split.json").read_text())
    torch.set_num_threads(1)
    params, grid = fit(v5_manifest)
    write_json(root / "fit_results.json", {"selected": params, "grid": grid})
    access = []
    dev_rows = evaluate(v5_manifest, V5, "DEV", RGBDSceneData, params, access)
    fresh_manifest = json.loads((FRESH / "scene_split.json").read_text())
    fresh_rows = evaluate(
        fresh_manifest, FRESH, "FRESH_QUALIFICATION", FreshRGBDSceneData, params, access
    )
    write_json(root / "audit/gt_access.json", access)
    write_json(root / "raw/nlf_dev_rows.json", dev_rows)
    write_json(root / "raw/nlf_fresh_rows.json", fresh_rows)
    v5_rows = json.loads((V5 / "raw/dev_query_results.json").read_text())
    fresh_carrier = json.loads((FRESH / "raw/fresh_query_results.json").read_text())
    dev_ref = json.loads((V5 / "raw/dev_reference_rows.json").read_text())
    fresh_ref = json.loads((FRESH / "raw/fresh_reference_rows.json").read_text())
    results = {
        "selected_params": params,
        "DEV": compare(dev_rows, v5_rows, dev_ref),
        "DEV_hit_scenes": compare(dev_rows, v5_rows, dev_ref, exclude=no_hit(v5_rows)),
        "FRESH_V1": compare(fresh_rows, fresh_carrier, fresh_ref),
        "no_hit_scenes": {"DEV": no_hit(v5_rows), "FRESH_V1": no_hit(fresh_carrier)},
        "lock_sha256": sha(root / "preregistration.json"),
    }
    write_json(root / "results.json", results)
    lines = [
        f"# {EXPERIMENT}",
        "",
        "不经学习的 RGB-D 融合基线：上下文深度直接体素化进同一个 16³ 网格、同一 bounds，用同一个固定 renderer 读出；只有 4 个标量在 TRAIN24 上拟合，"
        f"选中 {params}。所有状态在读取 query GT 之前封存。",
        "",
    ]
    for name in ("DEV", "DEV_hit_scenes", "FRESH_V1"):
        r = results[name]
        m = r["means"]
        lines += [
            f"## {name}",
            "",
            f"平均 AbsRel：NLF {m['NLF']:.4f}，V5 C1（学习，测得深度）{m['C1']:.4f}，V5 C0（学习，只用 RGB）{m['C0']:.4f}，常数 {m['constant']:.4f}。",
            "",
        ]
        for key in ("LEARNED_MINUS_NLF", "NLF_MINUS_CONSTANT", "RGB_ONLY_MINUS_NLF"):
            s = r[key]["stats"]
            lines.append(
                f"- {key} = {s['mean']:+.4f}，95% CI [{s['ci95'][0]:+.4f}, {s['ci95'][1]:+.4f}]，{r[key]['label']}"
            )
        lines.append("")
    (root / "README.md").write_text("\n".join(lines) + "\n")
    print(
        json.dumps(
            {k: results[k]["means"] for k in ("DEV", "DEV_hit_scenes", "FRESH_V1")}, indent=1
        )
    )
    print("NLF_COMPLETE", lock_["experiment"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("lock", "run"), required=True)
    parser.add_argument("--root", type=Path, default=Path("outputs") / EXPERIMENT)
    args = parser.parse_args()
    if args.stage == "lock":
        lock(args.root)
    else:
        (args.root / "raw").mkdir(exist_ok=True)
        (args.root / "audit").mkdir(exist_ok=True)
        run(args.root)
