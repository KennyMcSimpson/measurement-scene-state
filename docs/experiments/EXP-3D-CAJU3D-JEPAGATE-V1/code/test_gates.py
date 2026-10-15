"""Unit tests of the inconsistency scores and the matched-budget gate (CPU)."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gates as G  # noqa: E402
import hjepa as J  # noqa: E402

K = torch.tensor([[100.0, 0.0, 79.5], [0.0, 100.0, 63.5], [0.0, 0.0, 1.0]])


def two_views(shift=0.04):
    planar = torch.full((2, J.H, J.W), 2.0)
    c2w = torch.eye(4).repeat(2, 1, 1)
    c2w[1, 0, 3] = shift
    return planar, K.repeat(2, 1, 1), c2w


def test_consistent_views_score_zero_and_artifacts_score_high():
    planar, k, c2w = two_views()
    s = G.depth_disagreement(planar, k, c2w)
    assert float(s.max()) < 1e-5  # one plane seen from two places agrees everywhere it overlaps
    planar[1, 40:60, 60:90] = 1.0  # a spurious near blob in view 1 only
    s = G.depth_disagreement(planar, k, c2w)
    assert float(s[1, 45:55, 65:85].min()) > 0.9  # view 1's blob contradicts view 0's plane (|1 - 2| / 1)
    assert float(s[0, :, :40].max()) < 1e-5  # far from the blob, view 0 is still consistent


def test_invalid_pixels_score_zero():
    planar, k, c2w = two_views()
    planar[0, :, :20] = 0.0
    s = G.depth_disagreement(planar, k, c2w)
    assert float(s[0, :, :20].abs().max()) == 0.0


def test_matched_gate_drops_the_top_share_of_valid_pixels():
    score = torch.arange(100, dtype=torch.float32).view(10, 10)
    valid = torch.ones(10, 10, dtype=torch.bool)
    valid[0] = False  # the 10 lowest scores are invalid
    gate = G.matched_gate(score, valid, 0.1)
    assert int((gate == 0).sum()) == 10 + 9  # 10 invalid + 9 dropped (10 % of 90)
    assert float(gate[9, 1:].min()) == 0.0 and float(gate[1:9].min()) == 1.0
    assert float(gate[0].max()) == 0.0


def test_jepa_innovation_shapes():
    torch.manual_seed(0)
    model = J.GeometricReverseJEPA(latent=16, hidden=32).eval()
    v = 3
    rgb = torch.rand(v, 3, J.H, J.W)
    ray = torch.full((v, J.H, J.W), 2.0)
    nu = G.jepa_innovation(model, rgb, ray, K.repeat(v, 1, 1), torch.eye(4).repeat(v, 1, 1))
    assert nu.shape == (v, J.H, J.W) and bool(torch.isfinite(nu).all()) and float(nu.min()) >= 0
