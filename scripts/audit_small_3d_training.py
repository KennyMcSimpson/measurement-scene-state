"""Independent post-run audit of the locked short training pilot."""

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import torch

from mcss.dynamic.checkpoint import load_dynamic_checkpoint


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def difference(a, b):
    squared = 0.0
    changed = []
    for name, old in a.items():
        new = b[name]
        assert torch.isfinite(new).all(), name
        if not torch.equal(old, new):
            changed.append(name)
        squared += float((new.double() - old.double()).square().sum())
    return {"l2": math.sqrt(squared), "changed_tensors": changed}


def audit(training, output, *, expected_a=100, expected_b=30):
    lock = json.loads((training / "lock.json").read_text())
    summary = json.loads((training / "summary.json").read_text())
    history = [json.loads(s) for s in (training / "training.jsonl").read_text().splitlines()]
    assert expected_a > 0 and expected_b > 0
    assert lock["config"]["stage_a_steps"] == expected_a
    assert lock["config"]["stage_b_steps"] == expected_b
    expected = {"phase_a": expected_a, "phase_b": expected_b}
    assert Counter(r["phase"] for r in history) == expected
    for phase, count in expected.items():
        assert [r["step"] for r in history if r["phase"] == phase] == list(range(count))
    assert all(
        math.isfinite(r["loss"]) and math.isfinite(r["gradient_norm_before_clip"]) for r in history
    )
    assert all(sha(p) == h for p, h in lock["data_sha256"].items())
    assert all(sha(p) == h for p, h in lock["source_sha256"].items())
    models = {}
    for label in ["initial", "phase_a_final", "phase_b_final"]:
        carrier, writer, meta = load_dynamic_checkpoint(training / f"{label}.pt")
        assert meta["sha256"] == summary["checkpoint_sha256"][label]
        assert meta["provenance"]["lock_sha256"] == sha(training / "lock.json")
        models[label] = (carrier.state_dict(), writer.state_dict())
    changes = {}
    for before, after in [("initial", "phase_a_final"), ("phase_a_final", "phase_b_final")]:
        changes[before + "->" + after] = {
            name: difference(models[before][i], models[after][i])
            for i, name in enumerate(["carrier", "writer"])
        }
    assert changes["initial->phase_a_final"]["writer"]["l2"] == 0
    assert changes["initial->phase_a_final"]["carrier"]["l2"] > 0
    assert changes["phase_a_final->phase_b_final"]["writer"]["l2"] > 0
    rows = json.loads((training / "train_evaluation.json").read_text())
    keys = [(r["checkpoint"], r["scene_id"], r["query_id"]) for r in rows]
    assert len(keys) == len(set(keys)) == 36
    assert all(r["split"] == "train" and r["action"] == "OFF" for r in rows)
    metrics = [
        "loss",
        "rgb_psnr",
        "rgb_mse",
        "rgb_ssim",
        "depth_absrel",
        "depth_rmse",
        "depth_delta1",
        "opacity",
        "coverage",
    ]

    def complete_mean(values):
        # Preserve undefined/+infinite PSNR markers; never drop inconvenient rows.
        return None if any(v is None for v in values) else sum(values) / len(values)

    means = {}
    per_scene = {}
    for label in models:
        selected = [r for r in rows if r["checkpoint"] == label]
        scenes = sorted({r["scene_id"] for r in selected})
        assert len(scenes) == 3
        per_scene[label] = {
            s: {m: complete_mean([r[m] for r in selected if r["scene_id"] == s]) for m in metrics}
            for s in scenes
        }
        means[label] = {m: complete_mean([per_scene[label][s][m] for s in scenes]) for m in metrics}
    counts = defaultdict(Counter)
    for row in history:
        counts[row["phase"]][row["scene_id"]] += 1
    result = {
        "status": "PASS",
        "scope": "TRAIN_ONLY_ENGINEERING_NOT_INDEPENDENT_EVALUATION",
        "steps": {"A": expected_a, "B": expected_b},
        "scene_step_counts": dict(counts),
        "state_dict_changes": changes,
        "train_OFF_scene_macro": means,
        "per_scene": per_scene,
        "write_benefit_evaluated": False,
        "psnr_note": (
            "Null macro when any query PSNR is infinite/undefined; "
            "perfect flags retained, no row dropped."
        ),
        "perfect_rgb_queries": [
            {k: r[k] for k in ["checkpoint", "scene_id", "query_id", "coverage", "depth_absrel"]}
            for r in rows
            if r["rgb_psnr_perfect"]
        ],
        "writer_nonzero_gradient_steps": sum(
            r["writer_gradient_norm_after_clip"] > 0 for r in history if r["phase"] == "phase_b"
        ),
        "training_loss_first_last": {
            p: {
                "first": next(r["loss"] for r in history if r["phase"] == p),
                "last": next(r["loss"] for r in reversed(history) if r["phase"] == p),
            }
            for p in ["phase_a", "phase_b"]
        },
        "loss_first_last_note": "different scenes/queries; do not interpret as matched gain",
        "source_data_hashes_verified": True,
        "strict_checkpoint_loads": 3,
        "checkpoint_restore": summary["restore_checks"],
        "elapsed_seconds": summary["elapsed_seconds"],
        "controller_trained": False,
        "carrier_qualification": "NOT_EVALUATED",
    }
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--training", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--expected-a", type=int, default=100)
    p.add_argument("--expected-b", type=int, default=30)
    a = p.parse_args()
    audit(a.training, a.output, expected_a=a.expected_a, expected_b=a.expected_b)
