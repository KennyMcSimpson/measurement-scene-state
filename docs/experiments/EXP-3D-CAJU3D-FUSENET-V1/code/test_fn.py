"""Tests of EXP-3D-CAJU3D-FUSENET-V1 (CPU, repository root, PYTHONPATH=src)."""

import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

CODE = Path(__file__).resolve().parent
OUTPUTS = CODE.parents[1]
sys.path.insert(0, str(CODE))
import fn_data as D  # noqa: E402
import fn_model as M  # noqa: E402
import fn_score as FS  # noqa: E402
import fn_train as T  # noqa: E402


def random_base(rng, b=2, h=16, w=20):
    depth_z = rng.uniform(0.3, 4, (b, h, w))
    alpha = rng.uniform(0, 1, (b, h, w))
    r7 = rng.uniform(0.5, 5, (b, h, w))
    near = rng.uniform(size=(b, h, w)) < 0.7
    cosine = rng.uniform(0.8, 1, (b, h, w))
    base = np.stack([D.base_image(depth_z[i], alpha[i], r7[i], near[i], cosine[i]) for i in range(b)])
    return base, depth_z, alpha, r7, cosine


def test_base_channels():
    rng = np.random.default_rng(0)
    base, depth_z, alpha, r7, cosine = random_base(rng, b=1)
    b = base[0]
    cov = alpha[0] >= 0.5
    assert np.array_equal(b[6] > 0.5, cov)
    assert np.allclose(b[1][~cov], b[0][~cov])
    assert np.allclose(np.exp(b[1][cov]), depth_z[0][cov] / alpha[0][cov] / cosine[0][cov], rtol=1e-5)
    assert np.allclose(b[2], np.abs(b[1] - b[0]), atol=1e-6)


def test_untrained_fusenet_is_core_b():
    rng = np.random.default_rng(1)
    base, *_ = random_base(rng)
    base_t = torch.tensor(base)
    for use_jepa in (False, True):
        model = M.FuseNet(use_jepa)
        logit, r = model(base_t, torch.randn(2, M.JEPA_CH, 8, 10) if use_jepa else None)
        out = M.log_prediction(base_t, logit, r)
        expected = torch.where(base_t[:, 6] > 0.5, 0.5 * (base_t[:, 0] + base_t[:, 1]), base_t[:, 0])
        assert torch.allclose(out, expected, atol=1e-6)


def test_absrel_ignores_invalid_pixels():
    log_pred = torch.zeros(1, 2, 2)
    gt = torch.tensor([[[1.0, 2.0], [0.0, 0.0]]])
    assert torch.allclose(M.absrel_per_query(log_pred, gt), torch.tensor([0.25]))


def test_query_weights_reproduce_place_mean():
    rng = np.random.default_rng(2)
    qs = [{"place": f"p{i % 3}", "episode": f"e{i % 5}"} for i in range(11)]
    w = D.query_weights(qs)
    v = rng.uniform(size=len(qs))
    places = {}
    for q, x in zip(qs, v):
        places.setdefault(q["place"], {}).setdefault(q["episode"], []).append(x)
    expected = np.mean([np.mean([np.mean(e) for e in p.values()]) for p in places.values()])
    assert float((w * v).sum()) == pytest.approx(expected, rel=1e-12)


def test_split_is_deterministic_and_disjoint():
    places = {f"place_{i:03d}" for i in range(160)}
    train, val = T.split_places(places)
    assert len(train) == 128 and len(val) == 32 and not set(train) & set(val)
    assert (train, val) == T.split_places(set(places))


def test_jepa_path_shapes_and_learning():
    torch.manual_seed(0)
    rng = np.random.default_rng(3)
    base, *_ = random_base(rng, b=2, h=128, w=160)
    base_t = torch.tensor(base)
    jepa = torch.randn(2, M.JEPA_CH, 64, 80)
    gt = torch.exp(base_t[:, 1]) * 1.1
    model = M.FuseNet(True)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    first = None
    for _ in range(5):
        logit, r = model(base_t, jepa)
        assert logit.shape == (2, 128, 160) and r.shape == (2, 128, 160)
        loss = M.absrel_per_query(M.log_prediction(base_t, logit, r), gt).mean()
        first = float(loss) if first is None else first
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert float(loss) < first


def test_branches():
    assert FS.branch({"status": "SUPPORTED"}, {"mean": 0.001}) == "JEPA_FUSION_SUPPORTED"
    assert FS.branch({"status": "SUPPORTED"}, {"mean": -0.001}) == "JEPA_FUSION_PARTIAL"
    assert FS.branch({"status": "PARTIAL"}, {"mean": 0.1}) == "JEPA_FUSION_PARTIAL"
    assert FS.branch({"status": "HARMFUL"}, {"mean": 0.1}) == "JEPA_FUSION_HARMFUL"
    assert FS.branch({"status": "NOT_ESTABLISHED"}, {"mean": 0.1}) == "JEPA_FUSION_NOT_ESTABLISHED"


def test_root_files_are_byte_copies_of_kfuse():
    root = CODE.parent
    if not (root / "frame_role_lock.json").is_file():
        pytest.skip("the root is not built yet")
    for name in ("frame_role_lock.json", "reused_seals.json"):
        a = hashlib.sha256((root / name).read_bytes()).hexdigest()
        b = hashlib.sha256((OUTPUTS / "EXP-3D-CAJU3D-KFUSE-V1" / name).read_bytes()).hexdigest()
        assert a == b
