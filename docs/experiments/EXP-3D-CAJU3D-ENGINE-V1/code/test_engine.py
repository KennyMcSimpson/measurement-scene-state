"""Unit and contract tests of EXP-3D-CAJU3D-ENGINE-V1 (CPU). Run from the repository root with PYTHONPATH=src:
  python -m pytest -q -p no:cacheprovider outputs/EXP-3D-CAJU3D-ENGINE-V1/code/test_engine.py
"""

import json
import sys
from pathlib import Path

import pytest
import torch

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))
import engine_core as core  # noqa: E402
import engine_logic as L  # noqa: E402

ROOT = CODE.parent
K = torch.tensor([[100.0, 0.0, 79.5], [0.0, 100.0, 63.5], [0.0, 0.0, 1.0]]).view(1, 3, 3)


def test_incidence_weight_frontal_plane_is_kept():
    z = torch.full((1, 128, 160), 2.0)
    w, cos, jump = core.incidence_weight(z, K, **{k: L.INC[k] for k in ("cos_lo", "cos_hi")},
                                         edge_max=L.INC["edge_max"])
    assert float(cos[0, 64, 80]) == pytest.approx(1.0, abs=1e-4)
    assert float(w[0, :-1, :-1].min()) == pytest.approx(1.0)
    assert float(w[0, -1].max()) == 0.0 and float(w[0, :, -1].max()) == 0.0  # no forward neighbour


def test_incidence_weight_drops_grazing_planes_edges_and_holes():
    rows = torch.arange(128, dtype=torch.float64).view(128, 1).expand(128, 160)
    # a floor 1 m below the camera (Y = 1): Z = fy / (v - cy) below the horizon, nothing above it
    floor = torch.where(rows > 63.5, 100.0 / (rows - 63.5), torch.zeros_like(rows)).float().view(1, 128, 160)
    w, cos, _ = core.incidence_weight(floor, K, L.INC["cos_lo"], L.INC["cos_hi"], L.INC["edge_max"])
    assert float(cos[0, 66, 80]) < L.INC["cos_lo"] and float(w[0, 66, 80]) == 0.0  # near the horizon: edge-on
    assert float(cos[0, 120, 80]) > L.INC["cos_hi"] and float(w[0, 120, 80]) == 1.0  # steep view down: kept
    step = torch.full((1, 128, 160), 2.0)
    step[:, :, 80:] = 3.0
    w, _, jump = core.incidence_weight(step, K, L.INC["cos_lo"], L.INC["cos_hi"], L.INC["edge_max"])
    assert float(jump[0, 64, 79]) > L.INC["edge_max"] and float(w[0, 64, 79]) == 0.0
    hole = torch.full((1, 128, 160), 2.0)
    hole[:, 60, 60] = 0.0
    w, _, _ = core.incidence_weight(hole, K, L.INC["cos_lo"], L.INC["cos_hi"], L.INC["edge_max"])
    assert float(w[0, 60, 60]) == 0.0 and float(w[0, 60, 59]) == 0.0 and float(w[0, 59, 60]) == 0.0


def test_norm_terms_normalize_and_mask_without_gradient_through_the_mask():
    z = torch.full((1, 4, 4), 2.0)
    z[0, 0, 0] = 0.0
    alpha = torch.full((1, 4, 4), 0.8, requires_grad=True)
    with torch.no_grad():
        alpha[0, 1] = 0.3  # an uncovered row: no loss there
    true = (z * alpha).detach()  # depth_z = alpha * z  -> normalized depth = z exactly
    depth = true.clone().requires_grad_(True)
    color = torch.full((1, 3, 4, 4), 0.4) * alpha.detach().unsqueeze(1)
    rgb = torch.full((1, 3, 4, 4), 0.4)
    d, c, cov = core.norm_terms(depth, color, alpha, z, rgb, torch.ones_like(z), L.ALPHA_MIN)
    assert float(d) == pytest.approx(0.0, abs=1e-6) and float(c) == pytest.approx(0.0, abs=1e-6)
    assert float(cov) == pytest.approx(11 / 15)  # 15 valid pixels, 4 of them uncovered
    depth2 = (true * 1.1).requires_grad_(True)  # 10 % too far everywhere
    d2, _, _ = core.norm_terms(depth2, color, alpha, z, rgb, torch.ones_like(z), L.ALPHA_MIN)
    assert float(d2) == pytest.approx(0.1, abs=1e-5)
    d2.backward()
    assert float(depth2.grad[0, 1].abs().sum()) == 0.0  # uncovered pixels carry no gradient
    assert float(depth2.grad[0, 2].abs().sum()) > 0.0


def test_weight_restricts_the_depth_term():
    z = torch.full((1, 2, 2), 1.0)
    alpha = torch.ones(1, 2, 2)
    depth = torch.tensor([[[1.0, 1.0], [1.5, 1.0]]])
    weight = torch.ones(1, 2, 2)
    weight[0, 1, 0] = 0.0
    d, _, _ = core.norm_terms(depth, torch.zeros(1, 3, 2, 2), alpha, z, torch.zeros(1, 3, 2, 2), weight, 0.5)
    assert float(d) == pytest.approx(0.0)


def test_anchor_and_cone_projection():
    d_means = torch.zeros(1, 5, 3, requires_grad=True)
    d_logs = torch.zeros(1, 5, 3, requires_grad=True)
    d_logit = torch.zeros(1, 5, requires_grad=True)
    conf = torch.ones(1, 5)
    assert float(core.anchor_loss(d_means, d_logs, d_logit, conf)) == 0.0
    grads = [torch.ones(1, 5, 3), torch.zeros(1, 5, 3), torch.zeros(1, 5)]
    anchor = [-torch.ones(1, 5, 3), torch.zeros(1, 5, 3), torch.zeros(1, 5)]
    out, projected = core.cone_project(grads, anchor)
    assert projected and abs(sum(float((g * a).sum()) for g, a in zip(out, anchor))) < 1e-6
    same, projected = core.cone_project(grads, [torch.ones(1, 5, 3), torch.zeros(1, 5, 3), torch.zeros(1, 5)])
    assert not projected and all(torch.equal(a, b) for a, b in zip(same, grads))
    zero, projected = core.cone_project(grads, [torch.zeros(1, 5, 3), torch.zeros(1, 5, 3), torch.zeros(1, 5)])
    assert not projected  # the anchor gradient is zero at the first step


def test_selection_rule_and_stop():
    means = {k: 0.15 for k in L.CANDIDATES}
    means.update({L.COMPARATOR: 0.150, "JN1_NORM": 0.148, "ZS_NORM": 0.148})
    assert L.select_candidate(means) == ("ZS_NORM", False)  # tie broken by the CANDIDATES order
    means.update({"JN1_NORM": 0.151, "ZS_NORM": 0.151, L.COMPARATOR: 0.149})
    best, stop = L.select_candidate(means)
    assert stop
    assert L.arm_of("JN1_NORM_DEC_INC") == ("JN1", "NORM_DEC_INC")
    with pytest.raises(ValueError):
        L.arm_of("XX_NORM")


def test_branches():
    assert L.branch("SUPPORTED", {"mean": 0.001}) == "CAJU3D_ENGINE_SUPPORTED"
    assert L.branch("SUPPORTED", {"mean": -0.001}) == "CAJU3D_ENGINE_PARTIAL"
    assert L.branch("PARTIAL", {"mean": 0.001}) == "CAJU3D_ENGINE_PARTIAL"
    assert L.branch("HARMFUL", {"mean": -0.001}) == "CAJU3D_ENGINE_HARMFUL"
    assert L.branch("NOT_ESTABLISHED", {"mean": 0.001}) == "CAJU3D_ENGINE_NOT_ESTABLISHED"


def test_dev_scan_selection_is_seeded():
    ids = [f"arkit_{i}" for i in range(80)]
    a, b = L.select_dev_scans(ids, 0), L.select_dev_scans(list(reversed(ids)), 0)
    assert a == b and len(a) == L.DEV_PER_COHORT
    assert L.select_dev_scans(ids, 1) != a


def test_root_lock_cohorts():
    path = ROOT / "frame_role_lock.json"
    if not path.is_file():
        pytest.skip("the root is not built yet")
    cohorts = json.loads(path.read_text())["cohorts"]
    places = {n: {e["place"] for e in c["episodes"]} for n, c in cohorts.items()}
    assert len(places["DEV_SYSTEM"]) == len(places["DEV_SYSTEM2"]) == L.DEV_PER_COHORT
    assert len(cohorts["CONF_SYSTEM3"]["episodes"]) == 320 and len(cohorts["CONF_VAL"]["episodes"]) == 160
    names = list(places)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            assert not places[a] & places[b]
    for c in cohorts.values():
        for e in c["episodes"]:
            assert not (set(e["context"]) | set(e["stream"])) & set(e["queries"])


def test_lock_refuses_after_any_output(tmp_path):
    import engine_lock

    (tmp_path / "engine").mkdir()
    with pytest.raises(SystemExit):
        engine_lock.guard(tmp_path)
