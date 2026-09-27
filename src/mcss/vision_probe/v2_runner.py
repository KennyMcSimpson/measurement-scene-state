"""Stage-separated V2 discovery fitting, visible prediction, and sealed evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from mcss.vision_probe.adaptation import FastState, Readout, masks, materialize, proposals
from mcss.vision_probe.experiment import _cached_extract, _module_hash
from mcss.vision_probe.features import FrozenDinoExtractor
from mcss.vision_probe.opportunity_selectors import (
    enrich_candidates,
    select_candidates,
    visible_features,
)
from mcss.vision_probe.v2_access import PredictionAccessGuard
from mcss.vision_probe.v2_cycle import source_cycle
from mcss.vision_probe.v2_selectors import TRAJECTORIES, fit_v2, select_v2

ROOT = Path(__file__).resolve().parents[3]
V1_REPORT = ROOT / "docs/experiments/EXP-2D-20260926-opportunity-selector-v1"
V1_WORK = ROOT / "outputs/EXP-2D-20260926-opportunity-selector-v1"
DATA = ROOT / "data/vision_2d_probe/davis2017_trainval_480p/DAVIS"
WEIGHTS = ROOT / "data/vision_2d_probe/weights/dinov2_vits14_pretrain.pth"
METHODS = (
    "OFF",
    "discovery_fixed16",
    "v1_rgb_selector",
    "v1_visible_selector",
    "GateOnly",
    "CycleGate",
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write_rows(path, rows):
    Path(path).write_text(
        "".join(json.dumps(r, sort_keys=True, allow_nan=False) + "\n" for r in rows)
    )


def _context(paths):
    paths = read(paths) if not isinstance(paths, dict) else paths
    report, work = Path(paths["report"]), Path(paths["work"])
    report.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    config = read(report / "config.json")
    if config.get("eta", 5.0) != 5.0 or config.get("max_increment_norm", 0.15) != 0.15:
        raise ValueError("V1 update formula and fixed hyperparameters must remain unchanged")
    torch.set_num_threads(config.get("torch_num_threads", 4))
    torch.manual_seed(config.get("seed", 20260926))
    return report, work, config


def _v1_artifact():
    return read(V1_REPORT / "selectors.json")["by_resolution"]["448"]


def _readout():
    provenance = read(V1_REPORT / "readout_448_provenance.json")
    if sha(V1_WORK / "readout_448.pt") != provenance["sha256"]:
        raise ValueError("Frozen V1 readout hash mismatch")
    return Readout(**torch.load(V1_WORK / "readout_448.pt", map_location="cpu", weights_only=True))


def _candidate_states(target, readout):
    z = readout.project(target.features)
    result = []
    start = time.perf_counter()
    for trajectory in TRAJECTORIES:
        state = FastState.zero(64)
        for step, action in enumerate(trajectory):
            support, _ = masks(*z.shape[:2], step)
            state = proposals(z, target.rgb, readout, state, support, 5.0, 0.15)[action]
        adapted = readout.restore(target.features, z, materialize(z, state))
        result.append((state, adapted))
    return (
        z,
        result,
        {
            "proposal_calls": 32,
            "increment_calculations": 64,
            "candidate_materializations": 16,
            "candidate_seconds": time.perf_counter() - start,
        },
    )


def _discovery_manifest(raw):
    sequences = []
    for sequence in sorted({r["sequence"] for r in raw}):
        records = [r for r in raw if r["sequence"] == sequence]
        first = records[0]

        def frame(name, digest, sequence=sequence):
            return {
                "frame_id": name,
                "image_path": str(DATA / "JPEGImages/480p" / sequence / name),
                "mask_path": str(DATA / "Annotations/480p" / sequence / (Path(name).stem + ".png")),
                "image_sha256": digest,
            }

        source = frame(first["source_frame"], first["source_image_sha256"])
        targets = []
        for slot in (1, 2):
            item = next(r for r in records if r["target_slot"] == slot)
            targets.append(
                {**frame(item["target_frame"], item["target_image_sha256"]), "target_slot": slot}
            )
        sequences.append({"sequence": sequence, "source": source, "targets": targets})
    return {"sequences": sequences}


def _cached_only(frame, cache_index):
    digest = frame["image_sha256"]
    candidates = cache_index.get(digest, [])
    if len(candidates) != 1:
        raise ValueError(f"Expected unique frozen V1 cached feature: {digest}")
    tensor_path = candidates[0]
    payload = torch.load(tensor_path, map_location="cpu", weights_only=True)
    if tuple(payload["features"].shape) != (32, 32, 384):
        raise ValueError("Frozen cache is not real 448 features")
    return SimpleNamespace(**payload)


def discovery(paths):
    report, work, config = _context(paths)
    out = report / "discovery_rows.jsonl"
    if out.exists():
        raise FileExistsError("V2 discovery rows already exist")
    raw = [
        r
        for r in read_rows(V1_REPORT / "discovery_raw.jsonl")
        if r["resolution"] == 448 and r["split"] == "discovery"
    ]
    artifact = _v1_artifact()
    base = artifact["feature_names"]
    manifest = _discovery_manifest(raw)
    splits = read(V1_REPORT / "split_manifest.json")
    if {s["sequence"] for s in manifest["sequences"]} != set(splits["discovery"]):
        raise ValueError("Discovery identity mismatch")
    index = {}
    cache_hashes = {}
    weight_hash, extractor_hash = sha(WEIGHTS), _module_hash()
    for metadata in (V1_WORK / "cache/448").glob("*.json"):
        info = read(metadata)
        if info["weight_sha256"] != weight_hash or info["model_source_sha256"] != extractor_hash:
            continue
        if info["size"] == 448:
            tensor = metadata.with_suffix(".pt")
            index.setdefault(info["input_sha256"], []).append(tensor)
    readout = _readout()
    guard = PredictionAccessGuard(
        manifest,
        mode="discovery",
        allowed_ids=splits["discovery"],
        protected_ids=splits["reserve"] + splits["official_val_sealed"] + splits["validation"],
    )
    rows, costs = [], []
    try:
        with guard:
            for seq in manifest["sequences"]:
                source = _cached_only(seq["source"], index)
                for target_frame in seq["targets"]:
                    target = _cached_only(target_frame, index)
                    for frame in (seq["source"], target_frame):
                        path = index[frame["image_sha256"]][0]
                        cache_hashes[str(path)] = sha(path)
                    mask = guard.read_source_mask(
                        seq["sequence"], seq["source"]["frame_id"], target_frame["frame_id"]
                    )
                    _, candidates, cost = _candidate_states(target, readout)
                    cycles = [
                        source_cycle(
                            source.features,
                            a,
                            mask,
                            seq["source"]["frame_id"],
                            target_frame["frame_id"],
                        )
                        for _, a in candidates
                    ]
                    pair = [
                        r
                        for r in raw
                        if r["sequence"] == seq["sequence"]
                        and r["target_slot"] == target_frame["target_slot"]
                    ]
                    lookup = {tuple(r["trajectory"]): r for r in pair}
                    if set(lookup) != set(TRAJECTORIES) or len(pair) != 16:
                        raise ValueError("RAW_INCOMPLETE discovery candidates")
                    for i, trajectory in enumerate(TRAJECTORIES):
                        old = lookup[trajectory]
                        if set(old["visible"]) != set(base):
                            raise ValueError("V1 visible schema differs from artifact")
                        rows.append(
                            {
                                "split": "discovery",
                                "sequence": seq["sequence"],
                                "target_slot": target_frame["target_slot"],
                                "trajectory": list(trajectory),
                                "J": old["J"],
                                "visible": {
                                    **old["visible"],
                                    "f_cycle": cycles[0]["L_cycle"] - cycles[i]["L_cycle"],
                                },
                                "L_cycle": cycles[i]["L_cycle"],
                                "cycle": cycles[i],
                            }
                        )
                    costs.append(
                        {
                            "sequence": seq["sequence"],
                            "target_slot": target_frame["target_slot"],
                            **cost,
                            "cycle_seconds": sum(c["cost"]["total_seconds"] for c in cycles),
                            "correspondence_seconds": sum(
                                c["cost"]["correspondence_seconds"] for c in cycles
                            ),
                            "correspondence_calls": 32,
                            "cycle_calls": 16,
                            "backbone_calls": 0,
                        }
                    )
                    print(f"discovery {seq['sequence']}/{target_frame['target_slot']}", flush=True)
        write_rows(out, rows)
        write(
            report / "discovery_feature_provenance.json",
            {
                "v1_raw_sha256": sha(V1_REPORT / "discovery_raw.jsonl"),
                "v1_readout_sha256": sha(V1_WORK / "readout_448.pt"),
                "cache_hashes": cache_hashes,
                "base_feature_names": base,
                "new_feature": "f_cycle",
                "no_target_gt_reads": not guard.target_log,
            },
        )
        write(report / "discovery_cost.json", costs)
    finally:
        write(
            report / "discovery_access.json",
            {
                "source_reads": guard.source_log,
                "target_reads": guard.target_log,
                "denied": guard.denied_log,
            },
        )


def fit(paths):
    report, work, config = _context(paths)
    if (report / "selectors.json").exists():
        raise FileExistsError("V2 selectors already fitted")
    rows = read_rows(report / "discovery_rows.jsonl")
    result = fit_v2(rows, _v1_artifact()["feature_names"], config)
    write(report / "selectors.json", result["selectors"])
    write(report / "nested_cv_results.json", result["nested_cv_results"])
    # V1 discovery-fixed16 is also discovery-only and kept unchanged.
    write(
        report / "selector_lock.json",
        {
            "selectors_sha256": sha(report / "selectors.json"),
            "config_sha256": sha(report / "config.json"),
            "discovery_sha256": sha(report / "discovery_rows.jsonl"),
            "v1_selectors_sha256": sha(V1_REPORT / "selectors.json"),
            "readout_sha256": sha(V1_WORK / "readout_448.pt"),
            "weights_sha256": sha(WEIGHTS),
            "fit_split": "discovery",
            "primary_method": "CycleGate",
            "discovery_fixed16": _v1_artifact()["discovery_best_fixed"],
        },
    )

    save_discovery_decisions(report)


def save_discovery_decisions(report):
    """Replay final frozen policies on discovery features, explicitly not OOF."""
    report = Path(report)
    rows = read_rows(report / "discovery_rows.jsonl")
    selectors = read(report / "selectors.json")
    v1 = _v1_artifact()
    result = []
    for sequence, slot in sorted({(r["sequence"], r["target_slot"]) for r in rows}):
        group = [r for r in rows if r["sequence"] == sequence and r["target_slot"] == slot]
        lookup = {tuple(r["trajectory"]): r for r in group}
        group = [lookup[t] for t in TRAJECTORIES]
        features = [r["visible"] for r in group]
        paths = [list(t) for t in TRAJECTORIES]
        base = [{n: f[n] for n in v1["feature_names"]} for f in features]
        old = select_candidates(base, paths, v1)
        choices = {
            "OFF": 0,
            "discovery_fixed16": paths.index(v1["discovery_best_fixed"]),
            "v1_rgb_selector": old["rgb_selector"],
            "v1_visible_selector": old["visible_selector"],
            **{n: select_v2(features, paths, a) for n, a in selectors.items()},
        }
        result.append(
            {
                "sequence": sequence,
                "target_slot": slot,
                "choices": choices,
                "trajectories": paths,
                "scope": "discovery in-sample final policies, NOT outer OOF",
                "selected_J": {n: group[i]["J"] for n, i in choices.items()},
            }
        )
    write_rows(report / "discovery_decisions.jsonl", result)


def _verify_selectors(report):
    lock = read(report / "selector_lock.json")
    checks = {
        "selectors_sha256": sha(report / "selectors.json"),
        "config_sha256": sha(report / "config.json"),
        "discovery_sha256": sha(report / "discovery_rows.jsonl"),
        "v1_selectors_sha256": sha(V1_REPORT / "selectors.json"),
        "readout_sha256": sha(V1_WORK / "readout_448.pt"),
        "weights_sha256": sha(WEIGHTS),
    }
    if any(lock.get(k) != v for k, v in checks.items()):
        raise ValueError("Selector lock mismatch")
    return lock


def _source_hashes():
    names = (
        "adaptation",
        "experiment",
        "features",
        "geometry",
        "opportunity_metrics",
        "opportunity_selectors",
        "v2_selectors",
        "v2_cycle",
        "v2_access",
        "v2_runner",
        "v2_dataset",
    )
    files = [ROOT / f"src/mcss/vision_probe/{name}.py" for name in names]
    files.extend((ROOT / "third_party/dinov2").rglob("*.py"))
    files.append(ROOT / "scripts/run_vision_2d_v2.py")
    return {str(p): sha(p) for p in sorted(files) if p.exists()}


def _guard(manifest):
    splits = read(V1_REPORT / "split_manifest.json")
    old = read_rows(V1_REPORT / "raw_results.jsonl")
    hashes = {r[k] for r in old for k in ("source_image_sha256", "target_image_sha256")}
    ids = [s for key in ("fit", "discovery", "validation", "reserve") for s in splits[key]]
    return PredictionAccessGuard(
        manifest,
        historical_ids=ids,
        historical_hashes=hashes,
        protected_ids=splits["reserve"] + splits["official_val_sealed"],
        protected_paths=[DATA],
    )


def _prediction_checks(report, manifest_path, document):
    lock = read(report / "prediction_lock.json")
    checks = {
        "selector_lock_sha256": sha(report / "selector_lock.json"),
        "config_sha256": sha(report / "config.json"),
        "split_sha256": sha(manifest_path),
        "source_hashes": _source_hashes(),
    }
    if any(lock.get(k) != v for k, v in checks.items()):
        raise ValueError("Frozen prediction/evaluation code or artifact changed")
    manifest = read(manifest_path)
    expected = {
        (r["sequence"], t["target_slot"]) for r in manifest["sequences"] for t in r["targets"]
    }
    actual = {(r["sequence"], r["target_slot"]) for r in document["predictions"]}
    if (
        expected != actual
        or len(actual) != len(document["predictions"])
        or not document.get("complete")
    ):
        raise ValueError("Incomplete prediction pair coverage")
    if document.get("prediction_lock_sha256") != sha(report / "prediction_lock.json"):
        raise ValueError("Prediction manifest lock binding mismatch")
    for row in document["predictions"]:
        if set(row["choices"]) != set(METHODS) or any(
            type(i) is not int or not 0 <= i < 16 for i in row["choices"].values()
        ):
            raise ValueError("Missing or invalid deployment decisions")
        if row.get("trajectories") != [list(t) for t in TRAJECTORIES]:
            raise ValueError("Missing candidate trajectories")
        if row["selector_lock_sha256"] != lock["selector_lock_sha256"]:
            raise ValueError("Per-pair selector binding mismatch")
        path = Path(row["artifact_path"])
        if sha(path) != row["artifact_sha256"]:
            raise ValueError("Prediction tensor hash mismatch")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload["adapted"].shape != (16, 32, 32, 384) or payload["source_features"].shape != (
            32,
            32,
            384,
        ):
            raise ValueError("Prediction tensor candidate schema mismatch")
        if payload["choices"] != row["choices"] or payload["trajectories"] != row["trajectories"]:
            raise ValueError("Tensor/manifest decision mismatch")
    _verify_selectors(report)
    return lock


def freeze_implementation(paths):
    report, _, _ = _context(paths)
    _verify_selectors(report)
    value = {
        "source_hashes": _source_hashes(),
        "selector_lock_sha256": sha(report / "selector_lock.json"),
        "config_sha256": sha(report / "config.json"),
    }
    path = report / "implementation_lock.json"
    if path.exists() and read(path) != value:
        raise ValueError("Implementation snapshot differs; preserve and audit before replacing")
    write(path, value)
    return value


class _CountedExtractor:
    def __init__(self, extractor):
        self.extractor = extractor
        self.calls = 0

    def extract(self, path):
        self.calls += 1
        return self.extractor.extract(path)


def predict(paths, manifest_path):
    report, work, config = _context(paths)
    if (report / "prediction_lock.json").exists():
        raise FileExistsError(
            "Formal prediction already started; audit failures rather than overwrite"
        )
    selector_lock = _verify_selectors(report)
    freeze_implementation(paths)
    manifest = read(manifest_path)
    guard = _guard(manifest)
    selectors = read(report / "selectors.json")
    v1 = _v1_artifact()
    readout = _readout()
    lock = {
        "selector_lock_sha256": sha(report / "selector_lock.json"),
        "config_sha256": sha(report / "config.json"),
        "split_sha256": sha(manifest_path),
        "source_hashes": _source_hashes(),
        "expected_pairs": 2 * len(manifest["sequences"]),
        "target_gt_reads_before_prediction": 0,
    }
    write(report / "prediction_lock.json", lock)
    device = config.get("device", "cuda")
    extractor = _CountedExtractor(
        FrozenDinoExtractor(ROOT / "third_party/dinov2", WEIGHTS, device, 448)
    )
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    records, costs = [], []
    try:
        with guard:
            for seq in manifest["sequences"]:
                started = time.perf_counter()
                source_count_before = extractor.calls
                source = _cached_extract(
                    extractor,
                    Path(seq["source"]["image_path"]),
                    work / "cache448",
                    sha(WEIGHTS),
                    _module_hash(),
                    448,
                )
                source_seconds = time.perf_counter() - started
                source_calls = extractor.calls - source_count_before
                for frame in seq["targets"]:
                    started = time.perf_counter()
                    target_count_before = extractor.calls
                    target = _cached_extract(
                        extractor,
                        Path(frame["image_path"]),
                        work / "cache448",
                        sha(WEIGHTS),
                        _module_hash(),
                        448,
                    )
                    backbone_seconds = time.perf_counter() - started
                    target_calls = extractor.calls - target_count_before
                    mask = guard.read_source_mask(
                        seq["sequence"], seq["source"]["frame_id"], frame["frame_id"]
                    )
                    z, candidates, cost = _candidate_states(target, readout)
                    start = time.perf_counter()
                    base = enrich_candidates(
                        [
                            visible_features(
                                source.features, target.features, a, target.rgb, readout, s, z
                            )
                            for s, a in candidates
                        ],
                        [a for _, a in candidates],
                    )
                    visible_seconds = time.perf_counter() - start
                    cycles = [
                        source_cycle(
                            source.features, a, mask, seq["source"]["frame_id"], frame["frame_id"]
                        )
                        for _, a in candidates
                    ]
                    features = [
                        {**b, "f_cycle": cycles[0]["L_cycle"] - cycles[i]["L_cycle"]}
                        for i, b in enumerate(base)
                    ]
                    start = time.perf_counter()
                    paths_list = [list(t) for t in TRAJECTORIES]
                    v1choices = select_candidates(base, paths_list, v1)
                    choices = {
                        "OFF": 0,
                        "discovery_fixed16": paths_list.index(selector_lock["discovery_fixed16"]),
                        "v1_rgb_selector": v1choices["rgb_selector"],
                        "v1_visible_selector": v1choices["visible_selector"],
                        **{
                            name: select_v2(features, paths_list, artifact)
                            for name, artifact in selectors.items()
                        },
                    }
                    selector_seconds = time.perf_counter() - start
                    tensor_path = (
                        work / "predictions" / f"{seq['sequence']}_{frame['target_slot']}.pt"
                    ).resolve()
                    tensor_path.parent.mkdir(exist_ok=True)
                    torch.save(
                        {
                            "source_features": source.features,
                            "source_mask": torch.from_numpy(mask.astype(np.int64)),
                            "adapted": torch.stack([a for _, a in candidates]),
                            "choices": choices,
                            "trajectories": paths_list,
                        },
                        tensor_path,
                    )
                    records.append(
                        {
                            "sequence": seq["sequence"],
                            "target_slot": frame["target_slot"],
                            "source_id": seq["source"]["frame_id"],
                            "target_id": frame["frame_id"],
                            "artifact_path": str(tensor_path),
                            "artifact_sha256": sha(tensor_path),
                            "choices": choices,
                            "trajectories": paths_list,
                            "visible": features,
                            "L_cycle": [c["L_cycle"] for c in cycles],
                            "selector_lock_sha256": lock["selector_lock_sha256"],
                        }
                    )
                    costs.append(
                        {
                            "sequence": seq["sequence"],
                            "target_slot": frame["target_slot"],
                            **cost,
                            "source_extraction_seconds_shared": source_seconds
                            if frame["target_slot"] == 1
                            else 0.0,
                            "target_extraction_seconds": backbone_seconds,
                            "backbone_source_calls_shared": source_calls
                            if frame["target_slot"] == 1
                            else 0,
                            "backbone_target_calls": target_calls,
                            "visible_seconds": visible_seconds,
                            "visible_readout_calls": 16,
                            "visible_matching_matrices": 16,
                            "cycle_calls": 16,
                            "cycle_correspondence_calls": 32,
                            "cycle_seconds": sum(c["cost"]["total_seconds"] for c in cycles),
                            "cycle_correspondence_seconds": sum(
                                c["cost"]["correspondence_seconds"] for c in cycles
                            ),
                            "selector_seconds": selector_seconds,
                            "selector_calls": 4,
                            "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated()
                            if torch.cuda.is_available()
                            else 0,
                        }
                    )
                    print(f"predict {seq['sequence']}/{frame['target_slot']}", flush=True)
            document = {
                "complete": True,
                "predictions": records,
                "prediction_lock_sha256": sha(report / "prediction_lock.json"),
            }
            write(report / "predictions_manifest.json", document)
            _prediction_checks(report, manifest_path, document)
            receipt = guard.lock_predictions(
                report / "predictions_manifest.json", sha(report / "predictions_manifest.json")
            )
            write(report / "predictions_lock_receipt.json", receipt)
            write(
                report / "cost_analysis.json",
                {
                    "pairs": costs,
                    "scope": (
                        "All candidates trialed even when OFF selected; CPU adaptation/matching; "
                        "GPU backbone; no matched-budget efficiency claim"
                    ),
                },
            )
    finally:
        write(
            report / "prediction_access.json",
            {
                "source_reads": guard.source_log,
                "target_reads": guard.target_log,
                "denied": guard.denied_log,
            },
        )


def evaluate(paths, manifest_path):
    report, work, config = _context(paths)
    if (report / "raw_results.jsonl").exists() or (report / "evaluation_started.json").exists():
        raise FileExistsError(
            "Evaluation already started; explicit failure/exposure audit required"
        )
    document = read(report / "predictions_manifest.json")
    _prediction_checks(report, manifest_path, document)
    guard = _guard(read(manifest_path))
    receipt = read(report / "predictions_lock_receipt.json")
    if receipt["manifest_sha256"] != sha(report / "predictions_manifest.json"):
        raise ValueError("Prediction manifest no longer matches sealed receipt")
    guard.lock_predictions(report / "predictions_manifest.json", receipt["manifest_sha256"])
    # Target retrieval is imported only after every prediction and lock has passed validation.
    from mcss.vision_probe.opportunity_metrics import evaluate_pair
    from mcss.vision_probe.v2_dataset import fetch_target_masks

    write(
        report / "evaluation_started.json",
        {
            "utc": datetime.now(UTC).isoformat(),
            "predictions_manifest_sha256": receipt["manifest_sha256"],
            "warning": "Any failure preserves exposure history; no unaudited restart",
        },
    )
    rows = []
    try:
        fetch_receipt = fetch_target_masks(
            read(manifest_path),
            report / "predictions_manifest.json",
            receipt["manifest_sha256"],
            guard=guard,
        )
        write(report / "target_fetch_receipt.json", fetch_receipt)
        with guard:
            for pair in document["predictions"]:
                payload = torch.load(pair["artifact_path"], map_location="cpu", weights_only=True)
                mask = guard.read_target_mask(
                    pair["sequence"], pair["source_id"], pair["target_id"]
                )
                for i, trajectory in enumerate(TRAJECTORIES):
                    metric = evaluate_pair(
                        payload["source_features"],
                        payload["adapted"][i],
                        payload["source_mask"].numpy(),
                        mask,
                    )
                    rows.append(
                        {
                            "resolution": 448,
                            "split": "independent",
                            "sequence": pair["sequence"],
                            "target_slot": pair["target_slot"],
                            "source_frame": pair["source_id"],
                            "target_frame": pair["target_id"],
                            "trajectory": list(trajectory),
                            "visible": pair["visible"][i],
                            "L_cycle": pair["L_cycle"][i],
                            "selected_methods": [
                                name for name, index in pair["choices"].items() if index == i
                            ],
                            **metric,
                        }
                    )
                print(f"evaluate {pair['sequence']}/{pair['target_slot']}", flush=True)
        write_rows(report / "raw_results.jsonl", rows)
    finally:
        write_rows(work / "evaluation_partial_rows.jsonl", rows)
        write(
            report / "evaluation_access.json",
            {
                "source_reads": guard.source_log,
                "target_reads": guard.target_log,
                "denied": guard.denied_log,
            },
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("discovery", "fit", "predict", "evaluate"))
    parser.add_argument("--paths", type=Path, default=Path("/tmp/mcss_v2_paths.json"))
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    if args.stage in ("predict", "evaluate"):
        if args.manifest is None:
            parser.error("--manifest required for confirmation stages")
        globals()[args.stage](args.paths, args.manifest)
    else:
        globals()[args.stage](args.paths)


if __name__ == "__main__":
    main()
