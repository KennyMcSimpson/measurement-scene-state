import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


replication = load("replication", "evaluate_rgbd_replication_eval_v4.py")
inpaint = load("inpaint_for_replication_test", "run_rgbd_inpaint_baselines.py")


def rows_for(cohort, scenes, rng, sign=1):
    rows = []
    base = {
        "REPROJ_NN": 0.30,
        "REPROJ_HARMONIC": 0.28,
        "REPROJ_TELEA": 0.29,
        "C0": 0.33,
        "C1": 0.32,
    }
    for s in range(scenes):
        for role in ("A", "B"):
            for q in (1, 2):
                common = {"cohort": cohort, "scene_id": f"s{s}", "role": role, "query_id": q}
                common["far_fraction"] = 0.2
                for method, value in base.items():
                    for seed in (0,) if method.startswith("REPROJ") else (1, 2):
                        noise = rng.normal(0, 0.003)
                        rows.append(
                            {
                                **common,
                                "method": method,
                                "seed": seed,
                                "depth_absrel": value + noise,
                                "absrel_ALL": value + noise,
                                "absrel_NEAR": value - noise,
                                "absrel_FAR": value + 2 * noise,
                            }
                        )
                for method in ("FILL8_C0", "FILL8_C1", "HFILL8_C0", "HFILL8_C1", "TFILL8_C1"):
                    for seed in (1, 2):
                        value = 0.28 - sign * (0.02 + abs(rng.normal(0, 0.003)))
                        rows.append(
                            {
                                **common,
                                "method": method,
                                "seed": seed,
                                "depth_absrel": value,
                                "absrel_ALL": value,
                                "absrel_NEAR": value,
                                "absrel_FAR": value,
                            }
                        )
    return rows


def test_contrasts_and_block_equal_the_frozen_inpaint_v1_analysis():
    rng = np.random.default_rng(3)
    rows = rows_for("EVAL_V3", 12, rng) + rows_for("DEV", 4, rng)
    frozen = inpaint.analyze(rows)["cohorts"]["EVAL_V3"]
    mine = replication.inpaint_block([r for r in rows if r["cohort"] == "EVAL_V3"])
    assert set(mine["contrasts"]) == set(frozen["contrasts"])
    for name, entry in frozen["contrasts"].items():
        assert entry["definition"] == mine["contrasts"][name]["definition"]
        assert entry["gain"] == mine["contrasts"][name]["gain"]
        assert entry.get("label") == mine["contrasts"][name].get("label")
    assert frozen["absolute_absrel"] == mine["absolute_absrel"]
    assert frozen["n_scenes"] == mine["n_scenes"] == 12


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def test_lock_requires_the_frozen_plan_and_a_prepared_cohort(tmp_path, monkeypatch):
    root = tmp_path / replication.EXPERIMENT
    root.mkdir()
    (root / "PROTOCOL.md").write_text("protocol")
    (root / replication.PLAN).write_text("a different plan")
    v11, inp, eval_v4 = tmp_path / "v11", tmp_path / "inpaint", tmp_path / "eval_v4"
    for directory in (v11, inp, eval_v4):
        directory.mkdir()
    (v11 / "training_contract.json").write_text("{}")
    (inp / "lock.json").write_text("{}")
    for name in ("manifest_eval_v4.json", "candidate_lock.json"):
        (eval_v4 / name).write_text("{}")
    (eval_v4 / "preparation_integrity.json").write_text(json.dumps({"status": "PASS"}))
    with pytest.raises(PermissionError):
        replication.lock(root, v11, inp, eval_v4)
    monkeypatch.setattr(replication, "PLAN_SHA256", sha(root / replication.PLAN))
    (eval_v4 / "preparation_integrity.json").write_text(json.dumps({"status": "BLOCKED"}))
    with pytest.raises(PermissionError):
        replication.lock(root, v11, inp, eval_v4)
    (eval_v4 / "preparation_integrity.json").write_text(json.dumps({"status": "PASS"}))
    replication.lock(root, v11, inp, eval_v4)
    lock = json.loads((root / "lock.json").read_text())
    assert lock["v11_results_existed_at_lock"] is False and lock["final_holdout_opened"] is False
    assert set(lock["input_sha256"]) >= {"plan", "script", "eval_v4_manifest", "inpaint_lock"}
    with pytest.raises(FileExistsError):
        replication.lock(root, v11, inp, eval_v4)


def test_score_and_analyze_on_a_synthetic_seal(tmp_path):
    rng = np.random.default_rng(5)
    root = tmp_path / replication.EXPERIMENT
    (root / "raw" / "predictions").mkdir(parents=True)
    seeds = [1, 2]
    contract = tmp_path / "contract.json"
    contract.write_text(json.dumps({"seeds": seeds}))
    scenes, entries = [], {}
    for s in range(6):
        sid = f"ai_{s:03d}_001"
        frames = []
        for fid in (10, 11):
            gt = rng.uniform(1, 5, (1, 32, 40)).astype(np.float32)
            np.save(tmp_path / f"{sid}_{fid}.npy", gt)
            frames.append({"frame_id": fid, "depth": str(tmp_path / f"{sid}_{fid}.npy")})
            for role in ("A", "B"):
                hit = rng.random((32, 40)) < 0.5
                hit[5:25, 10:30] = False
                arrays = {
                    "hit": hit,
                    "near": inpaint.near_mask(hit),
                    "REPROJ_NN": rng.uniform(1, 5, (32, 40)),
                    "REPROJ_HARMONIC": rng.uniform(1, 5, (32, 40)),
                    "REPROJ_TELEA": rng.uniform(1, 5, (32, 40)),
                }
                for variant in ("C0", "C1"):
                    for seed in seeds:
                        arrays[f"{variant}_{seed}"] = rng.uniform(1, 5, (32, 40))
                path = root / "raw" / "predictions" / f"{sid}_{role}_{fid}.npz"
                np.savez_compressed(path, **arrays)
                entries[f"{sid}/{role}/{fid}"] = {"path": str(path), "sha256": sha(path)}
        scenes.append(
            {
                "scene_id": sid,
                "split": "EVAL_V4",
                "frames": frames,
                "roles": {"primary_query": [10, 11]},
            }
        )
    manifest = tmp_path / "manifest_eval_v4.json"
    manifest.write_text(json.dumps({"scenes": scenes, "image_size": [32, 40]}))
    lock = {
        "input_sha256": {
            "eval_v4_manifest": {"path": str(manifest), "sha256": sha(manifest)},
            "v11_contract": {"path": str(contract), "sha256": sha(contract)},
        }
    }
    (root / "lock.json").write_text(json.dumps(lock))
    replay = {"scenes": 2, "arrays_compared": 10, "max_abs_deviation": 0.0}
    (root / "raw" / "prediction_seal.json").write_text(
        json.dumps({"predictions": entries, "replay_on_eval_v3": replay})
    )
    rows = replication.score(root)
    per_query = 3 + 2 * len(seeds) * 4
    assert len(rows) == 6 * 2 * 2 * per_query
    result = replication.analyze(root)
    assert result["eval_v4"]["n_scenes"] == 6
    assert result["REPLICATION_BRANCH"] in {"A", "B", "C"}
    assert result["V11_BRANCH_ON_EVAL_V4"] in {"A", "B", "C"}
    assert (root / "README.md").exists() and (root / "terminal_summary.txt").exists()
    entry = next(iter(entries.values()))
    Path(entry["path"]).write_bytes(b"tampered")
    with pytest.raises(PermissionError):
        replication.score(root)
