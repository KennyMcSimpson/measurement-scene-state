"""Tests of EXP-3D-CAJU3D-KFUSE-V1 (CPU, repository root, PYTHONPATH=src)."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

CODE = Path(__file__).resolve().parent
OUTPUTS = CODE.parents[1]
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(OUTPUTS / "EXP-3D-CAJU-HEADROOM-V1" / "code"))
import kf_logic as K  # noqa: E402
from caju_logic import core_b  # noqa: E402


def query(rng, n=400, place="p0", episode="e0", gt=None):
    r7 = rng.uniform(0.5, 5.0, n)
    f7 = r7 * np.exp(rng.normal(0, 0.2, n))
    alpha = rng.uniform(0.0, 1.0, n)
    f7 = np.where(alpha >= 0.5, f7, r7)
    return {"place": place, "episode": episode, "key": f"{episode}/{rng.integers(1e9)}", "r7": r7, "f7": f7,
            "alpha": alpha, "near": rng.uniform(size=n) < 0.7, "log_p": rng.normal(-1, 0.5, n),
            "gt": r7 * np.exp(rng.normal(0, 0.1, n)) if gt is None else gt}


def test_core_b_arm_is_core_b():
    rng = np.random.default_rng(0)
    depth_z, alpha = rng.uniform(0.3, 4, (8, 10)), rng.uniform(0, 1, (8, 10))
    r7, cosine = rng.uniform(0.5, 5, (8, 10)), rng.uniform(0.8, 1, (8, 10))
    cb, f7 = core_b(depth_z, alpha, r7, cosine)
    q = {"r7": np.maximum(r7.ravel(), 1e-6), "f7": np.maximum(f7.ravel(), 1e-6), "alpha": alpha.ravel(),
         "near": np.ones(80, bool), "log_p": np.zeros(80)}
    assert np.allclose(K.readout(q, "CORE_B", None), cb.ravel(), rtol=1e-12)


def test_uncovered_pixels_keep_r7_and_weight_one_is_f7():
    rng = np.random.default_rng(1)
    q = query(rng)
    table = {"edges": {"DIS": [0.1] * 7, "ALPHA": [0.6, 0.7, 0.8], "LOGP": [-1.5, -1, -0.5]},
             "weights": {a: [20] * K.n_bins(v) for a, v in K.ARMS.items()}}
    for arm in K.ARMS:
        out = K.readout(q, arm, table)
        cov = q["alpha"] >= 0.5
        assert np.allclose(out[~cov], q["r7"][~cov]) and np.allclose(out[cov], q["f7"][cov])


def test_place_weights_reproduce_the_place_mean():
    rng = np.random.default_rng(2)
    qs = [query(rng, n=50 + 10 * i, place=f"p{i % 3}", episode=f"e{i % 5}") for i in range(10)]
    pw = K.place_weights(qs)
    values = [rng.uniform(size=q["gt"].size) for q in qs]
    weighted = sum(pw[i] * v.sum() for i, v in enumerate(values))
    places = {}
    for q, v in zip(qs, values):
        places.setdefault(q["place"], {}).setdefault(q["episode"], []).append(v.mean())
    expected = np.mean([np.mean([np.mean(x) for x in eps.values()]) for eps in places.values()])
    assert weighted == pytest.approx(expected, rel=1e-12)


def test_fit_recovers_the_better_source_per_bin():
    rng = np.random.default_rng(3)
    qs = []
    for i in range(6):
        q = query(rng, n=3000, place=f"p{i}", episode=f"e{i}")
        cov = q["alpha"] >= 0.5
        dis = np.abs(np.log(q["f7"]) - np.log(q["r7"]))
        q["gt"] = np.where(cov & (dis > np.median(dis[cov])), q["f7"], q["r7"])  # large DIS: F7 is right
        qs.append(q)
    covered = [K.covered_features(q)[1] for q in qs]
    edges = {n: K.quantile_edges(np.concatenate([f[n] for f in covered]), b).tolist() for n, b in K.FEATURE_BINS.items()}
    j = K.fit_weights(K.accumulate(qs, edges, {"KF_DIS": ("DIS",)})["KF_DIS"])
    w = K.WEIGHTS[j]
    assert np.all(w[:3] == 0.0) and np.all(w[-3:] == 1.0)


def test_bin_index_range_and_exclusion():
    rng = np.random.default_rng(4)
    q = query(rng)
    _, feats = K.covered_features(q)
    edges = {"DIS": list(np.linspace(0.01, 0.3, 7)), "ALPHA": [0.6, 0.7, 0.8], "LOGP": [-1.5, -1, -0.5]}
    for arm, names in K.ARMS.items():
        idx = K.bin_index(feats, names, edges)
        assert idx.min() >= 0 and idx.max() < K.n_bins(names)
    loss = {"a": np.array([[1.0, 0.0], [0.0, 1.0]]), "b": np.array([[0.0, 5.0], [5.0, 0.0]])}
    assert list(K.fit_weights(loss)) == [0, 1] and list(K.fit_weights(loss, exclude="b")) == [1, 0]


def test_branches():
    assert K.branch({"status": "SUPPORTED"}, {"mean": 0.001}) == "JEPA_KALMAN_SUPPORTED"
    assert K.branch({"status": "SUPPORTED"}, {"mean": -0.001}) == "JEPA_KALMAN_PARTIAL"
    assert K.branch({"status": "PARTIAL"}, {"mean": 0.001}) == "JEPA_KALMAN_PARTIAL"
    assert K.branch({"status": "HARMFUL"}, {"mean": 0.1}) == "JEPA_KALMAN_HARMFUL"
    assert K.branch({"status": "NOT_ESTABLISHED"}, {"mean": 0.1}) == "JEPA_KALMAN_NOT_ESTABLISHED"


def test_predict_at_equals_forward_without_target_media():
    from kf_jepa_core import predict_at
    import hjepa as J  # noqa: E402  (path set by kf_jepa_core)
    torch.manual_seed(0)
    model = J.GeometricReverseJEPA().eval()
    s, t = 3, 2
    rgb = torch.rand(s, 3, J.H, J.W)
    k = torch.tensor([[[140.0, 0, 80], [0, 140.0, 64], [0, 0, 1]]]).repeat(s + t, 1, 1)
    c2w = torch.eye(4).repeat(s + t, 1, 1)
    c2w[:, 0, 3] = torch.linspace(0, 0.4, s + t)
    ray = (2.0 + torch.rand(s, J.H, J.W)) * J.ray_factor(k[:s])
    z_hat, log_p, hit = predict_at(model, rgb, ray, k[:s], c2w[:s], k[s:], c2w[s:])
    for i in range(t):
        with torch.no_grad():
            ref = model(rgb[None], ray[None], k[:s][None], c2w[:s][None], torch.rand(1, 3, J.H, J.W),
                        torch.rand(1, J.H, J.W), k[s + i][None], c2w[s + i][None])
        assert torch.allclose(z_hat[i], ref["z_hat"][0], atol=1e-5)
        assert torch.allclose(log_p[i], ref["log_p"][0], atol=1e-5)
        assert torch.equal(hit[i], ref["hit"][0])


def test_root_is_disjoint_from_jepa_training():
    root = CODE.parent
    if not (root / "frame_role_lock.json").is_file():
        pytest.skip("the root is not built yet")
    lock = json.loads((root / "frame_role_lock.json").read_text())
    places = {n: {e["place"] for e in c["episodes"]} for n, c in lock["cohorts"].items()}
    names = list(places)
    assert all(not (places[a] & places[b]) for i, a in enumerate(names) for b in names[i + 1:])
    assert lock["jepa_training_data"]["overlap_with_cohorts"] == 0
