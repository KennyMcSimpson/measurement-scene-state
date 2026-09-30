import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from mcss.types import Cameras

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_rgbd_inpaint_baselines.py"
spec = importlib.util.spec_from_file_location("inpaint", SCRIPT)
inpaint = importlib.util.module_from_spec(spec)
spec.loader.exec_module(inpaint)


def camera():
    intrinsics = torch.tensor([[140.0, 0, 79.5], [0, 140.0, 63.5], [0, 0, 1]])
    return Cameras(intrinsics, torch.eye(4), (128, 160))


def plane_distance(cosine):
    v, u = np.mgrid[0:128, 0:160].astype(np.float64)
    rays = np.stack(((u - 79.5) / 140.0, (v - 63.5) / 140.0, np.ones_like(u)), -1)
    return 3.0 / (rays @ np.array([0.2, -0.3, 1.0])) / cosine


def holes(seed=0):
    hit = np.random.default_rng(seed).random((128, 160)) < 0.3
    hit[40:90, 50:120] = False
    return hit


def test_harmonic_fill_is_exact_for_a_plane_with_interior_holes():
    cosine = inpaint.ray_cosine(camera())
    distance = plane_distance(cosine)
    hit = holes()
    hit[0, :] = hit[-1, :] = hit[:, 0] = hit[:, -1] = True
    filled = inpaint.harmonic_fill(np.where(hit, distance, np.inf), hit, cosine, 2.0)
    assert np.abs(filled - distance).max() / distance.min() < 1e-9
    assert np.array_equal(filled[hit], distance[hit])


def test_harmonic_fill_edge_cases_and_maximum_principle():
    cosine = inpaint.ray_cosine(camera())
    none = np.zeros((128, 160), dtype=bool)
    assert np.array_equal(
        inpaint.harmonic_fill(np.full(none.shape, np.inf), none, cosine, 2.5),
        np.full(none.shape, 2.5),
    )
    full = np.ones_like(none)
    depth = np.random.default_rng(1).uniform(1, 5, none.shape)
    assert np.array_equal(inpaint.harmonic_fill(depth, full, cosine, 2.5), depth)
    hit = holes(2)
    filled = inpaint.harmonic_fill(np.where(hit, depth, np.inf), hit, cosine, 2.5)
    inverse = 1.0 / (filled * cosine)
    known = inverse[hit]
    assert np.isfinite(filled).all() and (filled > 0).all()
    assert inverse[~hit].min() >= known.min() - 1e-12 and inverse[~hit].max() <= known.max() + 1e-12


def test_telea_fill_keeps_hits_and_stays_positive():
    cosine = inpaint.ray_cosine(camera())
    distance = plane_distance(cosine)
    hit = holes(3)
    filled, bad = inpaint.telea_fill(np.where(hit, distance, np.inf), hit, cosine, 2.0)
    assert bad == 0 and np.array_equal(filled[hit], distance[hit])
    assert np.isfinite(filled).all() and (filled > 0).all()
    assert np.mean(np.abs(filled - distance) / distance) < 0.05
    constant, count = inpaint.telea_fill(distance, np.zeros_like(hit), cosine, 2.0)
    assert count == 0 and np.array_equal(constant, np.full(hit.shape, 2.0))


def fake_inputs(tmp_path):
    manifest = tmp_path / "eval_v3.json"
    manifest.write_text(json.dumps({"scenes": [], "image_size": [128, 160]}))
    v11, v7, geometric = tmp_path / "v11", tmp_path / "v7", tmp_path / "geo"
    for directory in (v11, v7, geometric / "raw"):
        directory.mkdir(parents=True)
    contract = {
        "eval_v3": {
            "manifest": str(manifest),
            "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
            "near_px": 8,
            "reprojection_fill_constant_m": 2.3736,
        },
        "seeds": [1],
    }
    (v11 / "training_contract.json").write_text(json.dumps(contract))
    (v11 / "PROTOCOL.md").write_text("v11")
    dev = [{"scene_id": "d1", "split": "DEV", "frames": [], "roles": {"primary_query": []}}]
    (v11 / "scene_split.json").write_text(json.dumps({"scenes": dev}))
    (v7 / "scene_split.json").write_text(json.dumps({"scenes": dev, "image_size": [128, 160]}))
    (geometric / "raw" / "baseline_seal.json").write_text("{}")
    return v11, v7, geometric


def test_lock_happens_once_and_only_before_any_v11_evaluation(tmp_path):
    v11, v7, geometric = fake_inputs(tmp_path)
    root = tmp_path / inpaint.EXPERIMENT
    root.mkdir()
    with pytest.raises(PermissionError):
        inpaint.lock(root, v11, geometric, v7)
    (root / "PROTOCOL.md").write_text("protocol")
    inpaint.lock(root, v11, geometric, v7)
    lock = json.loads((root / "lock.json").read_text())
    assert lock["v11_results_existed_at_lock"] is False and lock["near_px"] == 8
    with pytest.raises(FileExistsError):
        inpaint.lock(root, v11, geometric, v7)
    other = tmp_path / "second" / inpaint.EXPERIMENT
    other.mkdir(parents=True)
    (other / "PROTOCOL.md").write_text("protocol")
    (v11 / "raw").mkdir()
    with pytest.raises(PermissionError):
        inpaint.lock(other, v11, geometric, v7)


def synthetic_rows(sign):
    rows, rng = [], np.random.default_rng(4)
    base = {
        "REPROJ_NN": 0.30,
        "REPROJ_HARMONIC": 0.28,
        "REPROJ_TELEA": 0.29,
        "C0": 0.33,
        "C1": 0.32,
    }
    for cohort, scenes in (("EVAL_V3", 20), ("DEV", 8)):
        for s in range(scenes):
            for role in ("A", "B"):
                for q in (1, 2):
                    common = {
                        "cohort": cohort,
                        "scene_id": f"s{s}",
                        "role": role,
                        "query_id": q,
                        "far_fraction": 0.2,
                    }
                    for method, value in base.items():
                        seeds = (0,) if method.startswith("REPROJ") else (1, 2)
                        for seed in seeds:
                            noise = rng.normal(0, 0.002)
                            rows.append(
                                {
                                    **common,
                                    "method": method,
                                    "seed": seed,
                                    "depth_absrel": value,
                                    "absrel_ALL": value + noise,
                                    "absrel_NEAR": value,
                                    "absrel_FAR": value + noise,
                                }
                            )
                    for method in ("FILL8_C0", "FILL8_C1", "HFILL8_C0", "HFILL8_C1", "TFILL8_C1"):
                        for seed in (1, 2):
                            value = 0.28 - sign * (0.02 + abs(rng.normal(0, 0.002)))
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


@pytest.mark.parametrize(("sign", "label", "branch"), [(1, "ABOVE", "A"), (-1, "BELOW", "C")])
def test_analysis_labels_the_primary_contrast(sign, label, branch):
    result = inpaint.analyze(synthetic_rows(sign))
    assert result["PRIMARY_LABEL"] == label and result["INTERPRETATION_BRANCH"] == branch
    primary = result["cohorts"]["EVAL_V3"]["contrasts"]["HARMONIC_HYBRID_GAIN"]
    assert primary["gain"]["n_scenes"] == 20 and (primary["gain"]["mean"] > 0) == (sign > 0)
    assert result["cohorts"]["DEV"]["n_scenes"] == 8


def test_replay_of_v11_rows_must_match(tmp_path):
    rows = [
        r
        for r in synthetic_rows(1)
        if r["cohort"] == "EVAL_V3"
        and r["method"] in {"REPROJ_NN", "C0", "C1", "FILL8_C0", "FILL8_C1"}
    ]
    v11 = tmp_path / "v11"
    (v11 / "raw" / "eval_v3").mkdir(parents=True)
    theirs = [{k: v for k, v in r.items() if k != "cohort"} for r in rows]
    (v11 / "raw" / "eval_v3" / "query_rows.json").write_text(json.dumps(theirs))
    assert inpaint.replay_v11(rows, v11)["max_abs_deviation"] == 0.0
    theirs[0]["absrel_FAR"] += 1e-6
    (v11 / "raw" / "eval_v3" / "query_rows.json").write_text(json.dumps(theirs))
    with pytest.raises(AssertionError):
        inpaint.replay_v11(rows, v11)
