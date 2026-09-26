"""Locked, stage-separated opportunity study; no validation-driven fitting."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from mcss.vision_probe.adaptation import (
    ACTIONS,
    FastState,
    Readout,
    fit_readout,
    masks,
    materialize,
    proposals,
)
from mcss.vision_probe.experiment import _cached_extract, _module_hash
from mcss.vision_probe.features import FrozenDinoExtractor

ROOT = Path(__file__).resolve().parents[3]
REPORT = ROOT / "docs/experiments/EXP-2D-20260926-opportunity-selector-v1"
WORK = ROOT / "outputs/EXP-2D-20260926-opportunity-selector-v1"
DATA = ROOT / "data/vision_2d_probe/davis2017_trainval_480p/DAVIS"
WEIGHTS = ROOT / "data/vision_2d_probe/weights/dinov2_vits14_pretrain.pth"
TRAJECTORIES = list(itertools.product(ACTIONS, repeat=2))
CONFIG = {
    "schema": "opportunity-selector-v1",
    "resolutions": [224, 448],
    "primary_resolution": 448,
    "primary_metric": "J",
    "seed": 20260926,
    "bootstrap_seed": 20260926,
    "bootstrap_samples": 10000,
    "eta": 5.0,
    "eta_rule": "fixed from historical discovery; no new eta search",
    "latent_width": 64,
    "rgb_ridge": 0.1,
    "max_increment_norm": 0.15,
    "boundary_tolerance_diagonal": 0.008,
    "probability_threshold": 0.5,
    "size_bucket_thresholds": [0.005, 0.02, 0.1],
    "ridge_alphas": [0.1, 1.0, 10.0, 100.0],
    "abstention_thresholds": [0.0, 0.001, 0.005, 0.01, 0.02],
    "rgb_thresholds": [0.0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0],
    "rank_relative_tolerance": 1e-5,
    "rank_absolute_tolerance": 1e-8,
    "opportunity_strong": {
        "minimum_positive_sequences": 4,
        "top1_max": 0.5,
        "sequence_ci_lower_gt": 0,
        "loso_min_gt": 0,
        "non_tiny_gain_gt": 0,
    },
    "opportunity_moderate": {"minimum_positive_sequences": 2, "top1_max_exclusive": 0.8},
    "identifiability_strong": {"both_ci_lower_gt": 0, "harmful_among_writes_max": 0.25},
    "reserve_policy": "remains sealed; no confirmation in this v1",
    "validation_history": "same 16 historical internal validation sequences; not a fresh holdout",
    "aggregation": "objects within pair, both target pairs within sequence, equal sequence mean",
    "tie_rule": "OFF,A,B,ALL Cartesian order, earliest exact maximum",
    "correspondence": "cosine hard nearest source token, no target masks",
    "pixel_readout": "source object area occupancy, nearest transfer, bilinear upsample, >=0.5",
    "F_scope": "approximate binary boundary F, secondary; not official DAVIS J&F",
    "stages": [
        "fit",
        "discovery_opportunity",
        "discovery_selectors",
        "lock",
        "validation",
        "report",
    ],
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def read_json(path: Path):
    return json.loads(path.read_text())


def source_hashes() -> dict[str, str]:
    files = [
        p
        for p in (ROOT / "src/mcss/vision_probe").glob("*.py")
        if p.name != "opportunity_reporting.py"
    ]
    files += list((ROOT / "third_party/dinov2").rglob("*.py"))
    files += [
        ROOT / "scripts/run_opportunity_probe.py",
        ROOT / "scripts/audit_opportunity_probe.py",
        ROOT / "pyproject.toml",
        ROOT / "uv.lock",
    ]
    return {str(p.relative_to(ROOT)): sha(p) for p in sorted(files) if p.exists()}


def validate_splits(splits: dict) -> None:
    groups = [
        set(splits[k]) for k in ("fit", "discovery", "validation", "reserve", "official_val_sealed")
    ]
    if any(
        len(g) != len(splits[k])
        for g, k in zip(
            groups,
            ("fit", "discovery", "validation", "reserve", "official_val_sealed"),
            strict=True,
        )
    ):
        raise ValueError("Duplicate split identity")
    if any(a & b for a, b in itertools.combinations(groups, 2)):
        raise ValueError("Split overlap")
    if list(map(len, groups)) != [24, 12, 16, 8, 30]:
        raise ValueError("Frozen split counts changed")


class MediaGuard:
    """Fail closed for any image/mask open or directory listing outside allowed identities."""

    def __init__(self, data: Path, splits: dict):
        self.data = data.resolve()
        self.allowed: set[str] = set()
        self.events: list[dict] = []
        self.splits = splits

    def allow(self, split: str) -> None:
        if split not in ("fit", "discovery", "validation"):
            raise ValueError("Sealed split cannot be enabled")
        self.allowed = set(self.splits[split])
        self.events.append({"event": "allow", "split": split, "utc": datetime.now(UTC).isoformat()})

    def check_path(self, path: str | Path) -> None:
        try:
            rel = Path(path).resolve().relative_to(self.data)
        except (ValueError, TypeError):
            return
        parts = rel.parts
        if parts and parts[0] in ("JPEGImages", "Annotations"):
            if len(parts) < 3 or parts[2] not in self.allowed:
                raise PermissionError(f"Sealed/unauthorized media access: {rel}")

    def hook(self, event: str, args: tuple) -> None:
        if (
            event in ("open", "os.listdir", "os.scandir")
            and args
            and isinstance(args[0], (str, bytes, Path))
        ):
            path = args[0].decode() if isinstance(args[0], bytes) else args[0]
            self.check_path(path)
            try:
                rel = Path(path).resolve().relative_to(self.data)
            except ValueError:
                return
            if rel.parts and rel.parts[0] in ("JPEGImages", "Annotations"):
                self.events.append({"event": event, "path": str(rel)})


def frames_for(name: str) -> list[Path]:
    frames = sorted((DATA / "JPEGImages/480p" / name).glob("*.jpg"))
    if len(frames) < 3:
        raise ValueError(f"Insufficient frames: {name}")
    return [frames[0], frames[len(frames) // 3], frames[2 * len(frames) // 3]]


def prepare() -> None:
    if REPORT.exists():
        raise FileExistsError("Refusing to overwrite experiment directory")
    # Use original explicit identity protocol, not a scan of sealed media.
    splits = read_json(ROOT / "docs/experiments/EXP-2D-20260921-corrected/protocol.json")["splits"]
    validate_splits(splits)
    train = set((DATA / "ImageSets/2017/train.txt").read_text().split())
    if train != set().union(
        *(set(splits[k]) for k in ("fit", "discovery", "validation", "reserve"))
    ):
        raise ValueError("Official train identities changed")
    REPORT.mkdir(parents=True)
    WORK.mkdir(parents=True, exist_ok=True)
    write_json(REPORT / "config.json", CONFIG)
    write_json(REPORT / "split_manifest.json", splits)
    env = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "weight_sha256": sha(WEIGHTS),
        "created_utc": datetime.now(UTC).isoformat(),
        "submodules": subprocess.check_output(["git", "submodule", "status"], cwd=ROOT, text=True),
        "packages": subprocess.check_output(
            [str(Path.home() / ".local/bin/uv"), "pip", "freeze", "--python", sys.executable],
            cwd=ROOT,
            text=True,
        ),
    }
    write_json(REPORT / "environment.json", env)
    (REPORT / "git_commit.txt").write_text(
        subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True)
        + "Working changes are fingerprinted in validation_lock.json; base commit is n"
        "ot experiment code.\n"
    )
    (REPORT / "STATUS.md").write_text("PHASE=PREREGISTERED\nFINAL_INTEGRITY=PENDING\n")
    (REPORT / "README.md").write_text(
        "# Opportunity + visible selector v1\n"
        "\n"
        "Pre-registered before new validation evaluation. Primary: pixel-space objec"
        "t J at448;224 control. Preserve 64-dimensional latent and all16 two-step tr"
        "ajectories. No reserve or official validation access.\n"
        "\n"
        "Protocol details are in config.json. Equal sequence aggregation; 10,000 seq"
        "uence bootstrap draws. Historical internal validation identities are reused"
        ", so this is not a fresh final holdout. No parameter changes after validati"
        "on starts.\n"
    )
    (REPORT / "commands.sh").write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "cd /home/zonghan/measurement-scene-state\n"
        "export OMP_NUM_THREADS=4\n"
        "PY=.venv/bin/python\n"
        "$PY scripts/run_opportunity_probe.py prepare\n"
        "$PY scripts/run_opportunity_probe.py discovery\n"
        "$PY scripts/run_opportunity_probe.py lock\n"
        "$PY scripts/run_opportunity_probe.py validation\n"
        "$PY scripts/run_opportunity_probe.py report\n"
    )
    write_json(
        REPORT / "preregistration.json",
        {
            "utc": datetime.now(UTC).isoformat(),
            "config_sha256": sha(REPORT / "config.json"),
            "split_sha256": sha(REPORT / "split_manifest.json"),
            "source_hashes_at_registration": source_hashes(),
        },
    )


def rank_record(delta: torch.Tensor, decoder: torch.Tensor, step: int, module: str) -> dict:
    matrix = delta.double()
    singular = torch.linalg.svdvals(matrix)
    tol = max(1e-8, float(singular[0]) * 1e-5)
    basis = torch.linalg.svd(decoder.double(), full_matrices=False).U
    residual = matrix - basis @ (basis.T @ matrix)
    return {
        "step": step,
        "module": module,
        "singular_values": singular.tolist(),
        "numerical_rank": int((singular > tol).sum()),
        "tolerance": tol,
        "span_residual": float(residual.norm() / matrix.norm().clamp_min(1e-15)),
        "norm": float(matrix.norm()),
        "span": "column span of frozen RGB decoder (64x3)",
    }


def candidate_states(target, readout: Readout, eta: float):
    z = readout.project(target.features)
    result = []
    for trajectory in TRAJECTORIES:
        state = FastState.zero(64)
        audits = []
        for step, action in enumerate(trajectory):
            support, _ = masks(*z.shape[:2], step)
            old = state
            state = proposals(z, target.rgb, readout, old, support, eta)[action]
            for module in ("a", "b"):
                audits.append(
                    rank_record(
                        getattr(state, module) - getattr(old, module),
                        readout.decoder,
                        step,
                        module.upper(),
                    )
                )
        adapted = readout.restore(target.features, z, materialize(z, state))
        result.append((state, adapted, audits))
    return z, result


def extraction(size: int, split: str, splits: dict, guard: MediaGuard):
    guard.allow(split)
    extractor = FrozenDinoExtractor(ROOT / "third_party/dinov2", WEIGHTS, "cuda", size)
    features = {}
    for name in splits[split]:
        paths = frames_for(name)
        features[name] = [
            (
                path,
                _cached_extract(
                    extractor, path, WORK / f"cache/{size}", sha(WEIGHTS), _module_hash(), size
                ),
            )
            for path in paths
        ]
        print(f"extract resolution={size} split={split} sequence={name}", flush=True)
    del extractor
    torch.cuda.empty_cache()
    return features


def fit_resolution(size: int, splits: dict, guard: MediaGuard) -> Readout:
    features = extraction(size, "fit", splits, guard)
    items = [item for frames in features.values() for _, item in frames]
    readout = fit_readout(
        torch.cat([i.features for i in items]), torch.cat([i.rgb for i in items]), 64, 0.1
    )
    torch.save(vars(readout), WORK / f"readout_{size}.pt")
    write_json(
        REPORT / f"readout_{size}_provenance.json",
        {
            "fit_sequences": splits["fit"],
            "mask_access": False,
            "resolution": size,
            "latent_width": 64,
            "sha256": sha(WORK / f"readout_{size}.pt"),
        },
    )
    return readout


def run_pairs(
    size: int, split: str, splits: dict, guard: MediaGuard, readout: Readout, artifact=None
):
    from mcss.vision_probe.opportunity_metrics import evaluate_pair
    from mcss.vision_probe.opportunity_selectors import (
        enrich_candidates,
        select_candidates,
        visible_features,
    )

    features = extraction(size, split, splits, guard)
    records = []
    for name, frames in features.items():
        source_path, source = frames[0]
        for slot, (target_path, target) in enumerate(frames[1:], 1):
            z, candidates = candidate_states(target, readout, CONFIG["eta"])
            visible = [
                visible_features(
                    source.features, target.features, adapted, target.rgb, readout, state, z
                )
                for state, adapted, _ in candidates
            ]
            visible = enrich_candidates(visible, [a for _, a, _ in candidates])
            choices = (
                {}
                if artifact is None
                else select_candidates(visible, [list(t) for t in TRAJECTORIES], artifact)
            )
            # Commit deployable choices BEFORE opening any target GT.
            with (REPORT / f"{split}_decisions.jsonl").open("a") as handle:
                handle.write(
                    json.dumps(
                        {
                            "resolution": size,
                            "sequence": name,
                            "target_slot": slot,
                            "choices": choices,
                            "before_target_gt": True,
                        }
                    )
                    + "\n"
                )
            source_mask_path = DATA / "Annotations/480p" / name / (source_path.stem + ".png")
            target_mask_path = DATA / "Annotations/480p" / name / (target_path.stem + ".png")
            with Image.open(source_mask_path) as im:
                source_mask = np.asarray(im).copy()
            with Image.open(target_mask_path) as im:
                target_mask = np.asarray(im).copy()
            for index, (trajectory, (_, adapted, audit), visible_row) in enumerate(
                zip(TRAJECTORIES, candidates, visible, strict=True)
            ):
                metric = evaluate_pair(source.features, adapted, source_mask, target_mask)
                row = {
                    "resolution": size,
                    "split": split,
                    "sequence": name,
                    "target_slot": slot,
                    "source_frame": source_path.name,
                    "source_image_sha256": source.input_sha256,
                    "target_image_sha256": target.input_sha256,
                    "source_mask_sha256": sha(source_mask_path),
                    "target_mask_sha256": sha(target_mask_path),
                    "target_frame": target_path.name,
                    "trajectory": list(trajectory),
                    "visible": visible_row,
                    "rank_audit": audit,
                    "selected_methods": [m for m, i in choices.items() if i == index],
                    **metric,
                }
                records.append(row)
            print(
                f"evaluate resolution={size} split={split} sequence={name} slot={slot}", flush=True
            )
    return records


def save_rows(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in rows)
    )


def load_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def discovery(splits: dict, guard: MediaGuard):
    if (REPORT / "discovery_raw.jsonl").exists():
        raise FileExistsError("Discovery already completed; do not overwrite")
    rows = []
    for size in CONFIG["resolutions"]:
        readout = fit_resolution(size, splits, guard)
        rows.extend(run_pairs(size, "discovery", splits, guard, readout))
        save_rows(WORK / "discovery_progress.jsonl", rows)
    save_rows(REPORT / "discovery_raw.jsonl", rows)
    from mcss.vision_probe.opportunity_analysis import opportunity_summary

    write_json(REPORT / "discovery_opportunity.json", opportunity_summary(rows))
    (REPORT / "STATUS.md").write_text(
        "PHASE=DISCOVERY_OPPORTUNITY_COMPLETE\nFINAL_INTEGRITY=PENDING\n"
    )


def lock_selectors(splits: dict):
    from mcss.vision_probe.opportunity_selectors import fit_selectors, select_candidates

    if (REPORT / "validation_lock.json").exists():
        raise FileExistsError("Selectors already locked")
    if not (REPORT / "discovery_opportunity.json").exists():
        raise RuntimeError("Opportunity analysis must precede selector fitting")
    rows = load_rows(REPORT / "discovery_raw.jsonl")
    artifact = fit_selectors(rows, CONFIG)
    write_json(REPORT / "selectors.json", artifact)
    for size in CONFIG["resolutions"]:
        for name in splits["discovery"]:
            for slot in (1, 2):
                pair = [
                    r
                    for r in rows
                    if r["resolution"] == size
                    and r["sequence"] == name
                    and r["target_slot"] == slot
                ]
                choices = select_candidates(
                    [r["visible"] for r in pair],
                    [r["trajectory"] for r in pair],
                    artifact["by_resolution"][str(size)],
                )
                for index, row in enumerate(pair):
                    row["selected_methods"] = [m for m, i in choices.items() if i == index]
    save_rows(REPORT / "raw_results.jsonl", rows)
    lock = {
        "utc": datetime.now(UTC).isoformat(),
        "source_hashes": source_hashes(),
        "config_sha256": sha(REPORT / "config.json"),
        "split_sha256": sha(REPORT / "split_manifest.json"),
        "selectors_sha256": sha(REPORT / "selectors.json"),
        "weights_sha256": sha(WEIGHTS),
        "discovery_sha256": sha(REPORT / "discovery_raw.jsonl"),
        "readout_sha256": {str(s): sha(WORK / f"readout_{s}.pt") for s in CONFIG["resolutions"]},
        "fit_split": "discovery",
        "validation_started": False,
    }
    write_json(REPORT / "validation_lock.json", lock)
    (REPORT / "STATUS.md").write_text("PHASE=LOCKED_BEFORE_VALIDATION\nFINAL_INTEGRITY=PENDING\n")


def verify_lock() -> dict:
    lock = read_json(REPORT / "validation_lock.json")
    checks = {
        "source_hashes": source_hashes(),
        "config_sha256": sha(REPORT / "config.json"),
        "split_sha256": sha(REPORT / "split_manifest.json"),
        "selectors_sha256": sha(REPORT / "selectors.json"),
        "weights_sha256": sha(WEIGHTS),
        "discovery_sha256": sha(REPORT / "discovery_raw.jsonl"),
        "readout_sha256": {str(s): sha(WORK / f"readout_{s}.pt") for s in CONFIG["resolutions"]},
    }
    for key, value in checks.items():
        if lock[key] != value:
            raise RuntimeError(f"Validation lock mismatch: {key}")
    if lock["fit_split"] != "discovery":
        raise ValueError("Selectors not fitted on discovery")
    return lock


def validation(splits: dict, guard: MediaGuard):
    verify_lock()
    marker = REPORT / "validation_started.json"
    if marker.exists():
        raise RuntimeError("Validation already started; explicit bug audit required for any rerun")
    write_json(
        marker,
        {"utc": datetime.now(UTC).isoformat(), "lock_sha256": sha(REPORT / "validation_lock.json")},
    )
    artifact = read_json(REPORT / "selectors.json")
    rows = load_rows(REPORT / "raw_results.jsonl")
    for size in CONFIG["resolutions"]:
        readout = Readout(**torch.load(WORK / f"readout_{size}.pt", weights_only=True))
        new = run_pairs(
            size, "validation", splits, guard, readout, artifact["by_resolution"][str(size)]
        )
        rows.extend(new)
        save_rows(REPORT / "raw_results.jsonl", rows)
    verify_lock()
    write_json(
        REPORT / "validation_complete.json",
        {
            "utc": datetime.now(UTC).isoformat(),
            "raw_sha256": sha(REPORT / "raw_results.jsonl"),
            "n_rows": len(rows),
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "discovery", "lock", "validation", "report"])
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(CONFIG["seed"])
    np.random.seed(CONFIG["seed"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.stage == "prepare":
        prepare()
        return
    if read_json(REPORT / "config.json") != CONFIG:
        raise RuntimeError("Runtime config differs from preregistration")
    splits = read_json(REPORT / "split_manifest.json")
    validate_splits(splits)
    guard = MediaGuard(DATA, splits)
    sys.addaudithook(guard.hook)
    try:
        if args.stage == "discovery":
            discovery(splits, guard)
        elif args.stage == "lock":
            lock_selectors(splits)
        elif args.stage == "validation":
            validation(splits, guard)
        else:
            verify_lock()
            from mcss.vision_probe.opportunity_analysis import analyze

            analyze(
                REPORT / "raw_results.jsonl", REPORT, CONFIG, read_json(REPORT / "selectors.json")
            )
    finally:
        write_json(REPORT / f"access_audit_{args.stage}.json", guard.events)


if __name__ == "__main__":
    main()
