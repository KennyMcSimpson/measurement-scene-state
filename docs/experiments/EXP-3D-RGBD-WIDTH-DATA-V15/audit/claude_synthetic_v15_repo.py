"""Engineering-only CPU pipeline for V15 (width x data): train -> evaluate -> diagnose ->
reference -> analyze -> audit -> EVAL seal/score/analyze -> report (-> finalize inside a git
repo with --finalize).

Synthetic 8x8 media from the evaluation test fixture; no real TRAIN/DEV/EVAL data. Fake V12 C1
checkpoints (width 64) stand in for the 2x2 cell; the fake "sealed V14" predictions are produced
after selection by the V13 code path from V15 C0's own selected weights, so the C0 consistency
check runs for real.
"""

import copy
import importlib.util
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path.cwd() / "tests"))
from test_geometry_carrier_evaluation import fixture  # noqa: E402

from mcss.mechanism_pilot import rgbd_width_carrier as v12  # noqa: E402
from mcss.mechanism_pilot.readout_reanalysis import (  # noqa: E402
    PRIMARY_REFERENCE,
    READOUTS,
    REFERENCES,
    reference_values,
)
from mcss.mechanism_pilot.rgbd_wide_scale_carrier import load_checkpoint  # noqa: E402
from mcss.mechanism_pilot.small_training import sha, write_json  # noqa: E402

FINALIZE = "--finalize" in sys.argv


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


base = Path("/tmp/claude-rgbd-synthetic-v15")
docs_base = Path("/tmp/claude-rgbd-synthetic-v15-docs")
for p in (base, docs_base):
    if p.exists():
        shutil.rmtree(p)
root = (base / "EXP-3D-RGBD-WIDTH-DATA-V15").resolve()
docs = (docs_base / "EXP-3D-RGBD-WIDTH-DATA-V15").resolve()
root.mkdir(parents=True)
manifest, _ = fixture(root)
records = manifest["scenes"][:2]
rng = np.random.default_rng(0)
for record in records:
    for f in record["frames"]:
        Image.fromarray(np.full((8, 8, 3), 26, np.uint8)).save(f["rgb"])
        np.save(f["depth"], rng.uniform(1.5, 4.0, (8, 8)).astype(np.float32))
        f["intrinsics"] = [[8.0, 0.0, 4.0], [0.0, 8.0, 4.0], [0.0, 0.0, 1.0]]
for record in records:
    source = next(f for f in record["frames"] if f["frame_id"] == 4)
    for fid in (5, 6, 7, 10):
        frame = copy.deepcopy(source)
        frame["frame_id"] = fid
        for key, suffix in (("rgb", ".png"), ("depth", ".npy")):
            target = Path(source[key]).with_name(f"{record['scene_id']}_{fid}{suffix}")
            shutil.copyfile(source[key], target)
            frame[key] = str(target)
        pose = np.array(frame["c2w"])
        pose[0, 3] += 0.01 * fid
        frame["c2w"] = pose.tolist()
        record["frames"].append(frame)
    record["frames"].sort(key=lambda f: f["frame_id"])
manifest["scenes"] = []
manifest["image_size"] = [8, 8]
for partition, n in [("TRAIN", 102), ("DEV", 8)]:
    for i in range(n):
        r = copy.deepcopy(records[i % 2])
        r["scene_id"] = r["physical_scene_id"] = f"ENGINEERING_ONLY_{partition}_{i:03d}"
        r["split"] = partition
        manifest["scenes"].append(r)
manifest["fresh_status"] = "BLOCKED_INDEPENDENCE_UNRESOLVED"
train = [f"ENGINEERING_ONLY_TRAIN_{i:03d}" for i in range(102)]
config = {
    "variants": ["C0", "C1"],
    "base_variant": "V11 C1",
    "base_extra": "uniform_0_4",
    "primary_pair": ["C0", "C1"],
    "seeds": [20260928, 20260929],
    "steps": 12,
    "train_sets": {"TRAIN24": train[:24], "TRAIN72": train[:72], "TRAIN_EXT": train},
    "checkpoint_steps": [6, 12],
    "batch_rays": 16,
    "num_threads": 1,
    "learning_rate": 0.001,
    "gradient_clip": 1.0,
}
# Fake V12 C1 selected checkpoints (width 64, the 2x2 cell), written with the frozen V12 saver.
control_dir = base / "v12_control"
control_dir.mkdir()
control = {"C1": {}}
for seed in config["seeds"]:
    path = control_dir / f"C1_{seed}.pt"
    v12.save_checkpoint(path, v12.make_carrier("C1", seed, "cpu"), variant="C1", seed=seed, step=2, lock_sha256="v12")
    control["C1"][str(seed)] = {"path": str(path), "sha256": sha(path), "step": 2}
write_json(control_dir / "selected_checkpoints.json", control)
fake_v14 = base / "fake_v14" / "prediction_seal.json"
roles = {"EVAL_V3": "primary_query", "EVAL_V4": "primary_query", "EVAL_FAR": "far_query"}
cohorts = {}
for name, role in roles.items():
    directory = base / f"EXP-3D-RGBD-{name}-DATA"
    directory.mkdir(parents=True)
    eval_manifest = {"image_size": [8, 8], "scenes": [], "protected_scene_ids": []}
    for i in range(4):
        r = copy.deepcopy(records[i % 2])
        r["scene_id"] = r["physical_scene_id"] = f"ENGINEERING_ONLY_{name}_{i:02d}"
        r["split"] = name
        r["roles"]["far_query"] = list(r["roles"]["primary_query"])
        eval_manifest["scenes"].append(r)
    write_json(directory / "manifest.json", eval_manifest)
    cohorts[name] = {
        "manifest": str(directory / "manifest.json"),
        "manifest_sha256": sha(directory / "manifest.json"),
        "scenes": 4,
        "role": role,
        "v14_seal": str(fake_v14),
    }
config["eval"] = {
    "cohorts": cohorts,
    "primary": ["EVAL_V3", "EVAL_V4"],
    "descriptive": ["EVAL_FAR"],
    "reprojection_fill_constant_m": 2.37,
    "near_px": 8,
    "abstain_opacity_threshold": 0.5,
    "c0_reproduction_tolerance": 1e-6,
    "width64_train72": {
        "selected": str(control_dir / "selected_checkpoints.json"),
        "selected_sha256": sha(control_dir / "selected_checkpoints.json"),
    },
    "last_step": 12,
}
write_json(root / "scene_split.json", manifest)
write_json(root / "training_contract.json", config)
train_depths = [
    np.load(f["depth"])
    for r in manifest["scenes"]
    if r["split"] == "TRAIN"
    for f in r["frames"]
    if f["frame_id"] in r["roles"]["primary_query"]
]
write_json(
    root / "decision_rules.json",
    {
        "GEOMETRY_FREE_REFERENCE": {
            "values": reference_values(train_depths),
            "references": list(REFERENCES),
            "primary_reference": PRIMARY_REFERENCE,
            "readouts": list(READOUTS),
            "bootstrap": {"unit": "scene", "draws": 200, "seed": 20260928},
        }
    },
)
data = {f[k]: sha(Path(f[k])) for r in manifest["scenes"] for f in r["frames"] for k in ("rgb", "depth")}
paths = [
    root / n
    for n in ("scene_split.json", "training_contract.json", "train_depth_prior.json", "decision_rules.json")
]
sources = sorted(Path("src/mcss/mechanism_pilot").glob("*.py")) + sorted(Path("scripts").glob("*rgbd_wide_scale*.py"))
write_json(
    root / "preregistration.json",
    {
        "status": "FROZEN_BEFORE_FORMAL_TRAINING",
        "input_sha256": {str(p): sha(p) for p in paths},
        "source_sha256": {str(p.resolve()): sha(p) for p in sources},
        "data_sha256": data,
        "engineering_only": True,
    },
)
train_script = module("synthetic_train", "scripts/train_rgbd_wide_scale_carrier.py")
evaluate = module("synthetic_evaluate", "scripts/evaluate_rgbd_wide_scale_carrier.py")
reference = module("synthetic_reference", "scripts/reference_rgbd_wide_scale_carrier.py")
analyze = module("synthetic_analyze", "scripts/analyze_rgbd_wide_scale_carrier.py")
audit = module("synthetic_audit", "scripts/audit_rgbd_wide_scale_statistics.py")
report = module("synthetic_report", "scripts/report_rgbd_wide_scale_carrier.py")
started = time.perf_counter()
for variant in config["variants"]:
    for seed in config["seeds"]:
        train_script.train(root, variant, seed, "cpu")
streams = {
    seed: {v: json.loads((root / f"checkpoints/{v}_{seed}/summary.json").read_text())["data_ray_stream_sha256"] for v in ("C0", "C1")}
    for seed in config["seeds"]
}
evaluate.evaluate(root, "cpu")
module("synthetic_diagnose", "scripts/diagnose_rgbd_wide_scale_states.py").diagnose(root)
ref = reference.run(root)
analyze.run(root)
audit.audit(root)
curve_states = sorted(str(p) for p in (root / "checkpoints").rglob("states.pt"))
selected_states = sorted(str(p) for p in (root / "raw/selected_dev").rglob("states.pt"))
assert not curve_states, curve_states
assert len(selected_states) == 4, selected_states
# Fake "sealed V14" predictions: the V13 code path with V15 C0's own selected weights.
v13 = module("synthetic_v13", "scripts/evaluate_rgbd_volume_abstention.py")
selected = json.loads((root / "selected_checkpoints.json").read_text())
models = {seed: load_checkpoint(selected["C0"][str(seed)]["path"], "cpu")[0].eval() for seed in config["seeds"]}
predictions = {}
for name, entry in cohorts.items():
    eval_manifest = json.loads(Path(entry["manifest"]).read_text())
    for record in eval_manifest["scenes"]:
        scene = v13.predict(record, eval_manifest, Path(entry["manifest"]).parent, models, 2.37, entry["role"], [], name.lower())
        for (role, qid), arrays in scene.items():
            path = fake_v14.parent / "predictions" / name / f"{record['scene_id']}_{role}_{qid}.npz"
            path.parent.mkdir(parents=True, exist_ok=True)
            kept = {k: v for k, v in arrays.items() if k.startswith("C1_")}
            kept |= {k.replace("C1_", "C0_"): v + 0.01 for k, v in kept.items()}
            np.savez_compressed(path, **kept)
            predictions[f"{name}/{record['scene_id']}/{role}/{qid}"] = {"path": str(path), "sha256": sha(path)}
write_json(fake_v14, {"predictions": predictions, "query_depth_read": False, "consistency": {"c0_minus_v11_c1_max_abs": 0.0}})
stage = module("synthetic_eval", "scripts/evaluate_rgbd_wide_scale_eval.py")
torch.set_num_threads(1)
stage.seal(root)
stage.score(root)
eval_full = stage.analyze(root)
write_json(root / "tests.json", {"status": "PASS", "synthetic_only": True})
if FINALIZE:
    write_json(root / "audit/previous_experiment_seal.json", {"files": {}})
    module("synthetic_finalize", "scripts/finalize_rgbd_wide_scale_carrier.py").finish(root, docs)
    fields = None
else:
    fields = report.run(root)
access = json.loads((root / "raw/selected_dev/C1_20260928/GT_access.json").read_text())
marker = next(i for i, e in enumerate(access) if e.get("event") == "ALL_DEV_STATES_SEALED")
methods = sorted({r["method"] for r in json.loads((root / "raw/eval/query_rows.json").read_text())})
result = {
    "status": "PASS",
    "synthetic_only": True,
    "real_DEV_read": False,
    "finalized": FINALIZE,
    "seconds": time.perf_counter() - started,
    "data_stream_shared_by_widths": {str(k): v["C0"] == v["C1"] for k, v in streams.items()},
    "curve_state_files": len(curve_states),
    "selected_state_files": len(selected_states),
    "query_rows": len(json.loads((root / "raw/dev_query_results.json").read_text())),
    "evaluated_variants": json.loads((root / "evaluated_variants.json").read_text())["variants"],
    "preseal_depth_reads": sum("depth" in e.get("channels", []) for e in access[:marker]),
    "reference_status": ref["summary"]["CARRIER_REFERENCE_STATUS"],
    "eval_consistency": eval_full["consistency"],
    "eval_statuses": eval_full["statuses"],
    "eval_branch": eval_full["INTERPRETATION_BRANCH"],
    "eval_methods": methods,
    "eval_cohorts": {k: v["n_scenes"] for k, v in eval_full["cohorts"].items()},
    "interaction_defined": eval_full["primary"]["contrasts"]["INTERACTION"]["gain"] is not None,
    "report_branch": fields["INTERPRETATION_BRANCH"] if fields else None,
}
assert all(result["data_stream_shared_by_widths"].values())
print(json.dumps(result, indent=1, default=str))
print("SYNTHETIC_RGBD_V15_PIPELINE_PASS", flush=True)
if FINALIZE:
    out = Path("outputs/EXP-3D-RGBD-WIDTH-DATA-V15/audit/synthetic_full_pipeline_rgbd_wide_scale_v15.json")
    result["postrun_validation"] = json.loads((root / "audit/postrun_validation.json").read_text())
    out.parent.mkdir(parents=True, exist_ok=True)
    write_json(out, result)
    print("WROTE", out)
