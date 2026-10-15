"""Unit tests of the geometry-aligned reverse JEPA (CPU)."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hjepa as J  # noqa: E402

K = torch.tensor([[100.0, 0.0, 79.5], [0.0, 100.0, 63.5], [0.0, 0.0, 1.0]])


def plane(depth_planar, s=1):
    """Ray depth of a fronto-parallel plane at a planar depth, for s views of the camera K: [1, s, H, W]."""
    rf = J.ray_factor(K.view(1, 1, 3, 3))
    return (depth_planar * rf).expand(1, s, J.H, J.W).clone()


def column_features(c=4):
    j = torch.arange(J.W2, dtype=torch.float32).view(1, 1, 1, 1, J.W2).expand(1, 1, c, J.H2, J.W2)
    return j.clone()


def warp(feat, ray, src_c2w, tgt_c2w):
    s = feat.shape[1]
    k = K.view(1, 1, 3, 3).expand(1, s, 3, 3)
    planar = ray / J.ray_factor(k)
    pts, valid = J.cell_points(planar, k, src_c2w)
    return J.warp_to_target(feat, pts, valid, K.view(1, 3, 3), tgt_c2w)


def test_identity_warp_returns_the_source_features():
    feat = column_features()
    eye = torch.eye(4).view(1, 1, 4, 4)
    warped, hit, logd = warp(feat, plane(2.0), eye, torch.eye(4).view(1, 4, 4))
    assert float(hit.min()) == 1.0
    assert torch.allclose(warped, feat[:, 0])
    assert torch.allclose(logd, torch.full_like(logd, torch.log(torch.tensor(2.0))), atol=1e-5)


def test_translation_shifts_by_the_parallax():
    feat = column_features()
    eye = torch.eye(4).view(1, 1, 4, 4)
    target = torch.eye(4).view(1, 4, 4).clone()
    target[0, 0, 3] = 0.04  # camera moved 4 cm to +x: points at 2 m move 2 px (one latent cell) to the left
    warped, hit, _ = warp(feat, plane(2.0), eye, target)
    assert float(hit[0, 0, :, -1].max()) == 0.0  # the rightmost cells see nothing of the source
    assert torch.allclose(warped[0, :, 10:50, 5:60], feat[0, 0, :, 10:50, 6:61])


def test_zbuffer_keeps_the_nearest_surface():
    near, far = torch.ones(1, 1, 4, J.H2, J.W2), torch.full((1, 1, 4, J.H2, J.W2), 2.0)
    feat = torch.cat([far, near], dim=1)
    ray = torch.cat([plane(4.0), plane(2.0)], dim=1)
    eye = torch.eye(4).view(1, 1, 4, 4).expand(1, 2, 4, 4)
    warped, hit, logd = warp(feat, ray, eye, torch.eye(4).view(1, 4, 4))
    assert torch.allclose(warped, torch.ones_like(warped))
    assert torch.allclose(logd, torch.full_like(logd, torch.log(torch.tensor(2.0))), atol=1e-5)


def test_invalid_depth_is_not_warped():
    feat = column_features()
    ray = plane(2.0)
    ray[..., :, :80] = 0.0
    warped, hit, _ = warp(feat, ray, torch.eye(4).view(1, 1, 4, 4), torch.eye(4).view(1, 4, 4))
    assert float(hit[0, 0, :, :40].max()) == 0.0 and float(hit[0, 0, :, 40:].min()) == 1.0


def test_pluecker_principal_ray():
    rays = J.pluecker(K.view(1, 3, 3), torch.eye(4).view(1, 4, 4))
    assert rays.shape == (1, 6, J.H2, J.W2)
    centre = rays[0, :3, 31, 39]
    assert float(centre[2]) == pytest.approx(1.0, abs=1e-3)
    assert float(rays[0, 3:].abs().max()) == 0.0  # camera at the origin: zero moments


def test_model_forward_loss_backward_and_ema():
    torch.manual_seed(0)
    model = J.GeometricReverseJEPA(latent=16, hidden=32)
    s = 2
    src_rgb, tgt_rgb = torch.rand(1, s, 3, J.H, J.W), torch.rand(1, 3, J.H, J.W)
    src_ray, tgt_ray = plane(2.0, s), plane(2.0)[:, 0]
    k = K.view(1, 1, 3, 3).expand(1, s, 3, 3)
    c2w = torch.eye(4).view(1, 1, 4, 4).expand(1, s, 4, 4)
    out = model(src_rgb, src_ray, k, c2w, tgt_rgb, tgt_ray, K.view(1, 3, 3), torch.eye(4).view(1, 4, 4))
    assert out["z_hat"].shape == out["z"].shape == (1, 16, J.H2, J.W2) and out["log_p"].shape == (1, 1, J.H2, J.W2)
    loss, parts = J.jepa_loss(out)
    before = [p.clone() for p in model.target_encoder.parameters()]
    loss.backward()
    assert all(p.grad is None for p in model.target_encoder.parameters())
    assert any(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in model.encoder.parameters())
    with torch.no_grad():
        for p in model.encoder.parameters():
            p.add_(0.1)
    model.update_target()
    assert any(not torch.equal(a, b) for a, b in zip(before, model.target_encoder.parameters()))
    assert set(parts) == {"nll", "var", "mse"}
