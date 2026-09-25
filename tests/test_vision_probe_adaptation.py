import torch

from mcss.vision_probe.adaptation import (
    FastState,
    Readout,
    masks,
    materialize,
    proposals,
    residual_selector,
)


def fixture():
    rng = torch.Generator().manual_seed(41)
    z = torch.randn(8, 8, 6, generator=rng)
    decoder = torch.randn(6, 3, generator=rng) * 0.1
    readout = Readout(torch.zeros(6), torch.eye(6), torch.ones(6), decoder, torch.zeros(3))
    rgb = z @ decoder + 0.1
    return z, rgb, readout


def test_off_preserves_content_and_proposals_are_branch_private():
    z, rgb, r = fixture()
    old = FastState.zero(6)
    support, _ = masks(8, 8, 0)
    before = materialize(z, old).clone()
    c = proposals(z, rgb, r, old, support, 0.4)
    assert torch.equal(materialize(z, c["OFF"]), before)
    assert torch.equal(materialize(z, old), before)
    assert c["A"].a.norm() > 0
    assert torch.equal(c["ALL"].a, c["A"].a)
    assert torch.equal(c["ALL"].b, c["B"].b)
    c["OFF"].a.fill_(17)
    assert old.a.count_nonzero() == 0
    assert not torch.equal(c["A"].a, c["OFF"].a)


def test_update_uses_only_support_targets_and_selection_has_no_geometry_input():
    z, rgb, r = fixture()
    support, probe = masks(8, 8, 0)
    changed = rgb.clone()
    changed[~support] += 100
    old = FastState.zero(6)
    first = proposals(z, rgb, r, old, support, 0.4)
    second = proposals(z, changed, r, old, support, 0.4)
    for key in first:
        assert torch.equal(first[key].a, second[key].a)
        assert torch.equal(first[key].b, second[key].b)
    assert not (support & probe).any()
    assert residual_selector(z, rgb, r, first, probe) in first


def test_zero_residual_produces_zero_write():
    z, _, r = fixture()
    rgb = r.predict(z)
    support, _ = masks(8, 8, 0)
    for state in proposals(z, rgb, r, FastState.zero(6), support, 0.4).values():
        assert state.a.count_nonzero() == 0
        assert state.b.count_nonzero() == 0


def test_second_step_all_is_simultaneous_not_sequential():
    z, rgb, readout = fixture()
    support0, _ = masks(8, 8, 0)
    support1, _ = masks(8, 8, 1)
    old = proposals(z, rgb, readout, FastState.zero(6), support0, 2.0)["ALL"]
    snapshot = old.fork()
    candidates = proposals(z, rgb, readout, old, support1, 2.0)
    sequential_b = proposals(z, rgb, readout, candidates["A"], support1, 2.0)["B"]
    assert old.a.norm() > 0 and old.b.norm() > 0
    assert torch.equal(old.a, snapshot.a) and torch.equal(old.b, snapshot.b)
    assert torch.equal(candidates["ALL"].a, candidates["A"].a)
    assert torch.equal(candidates["ALL"].b, candidates["B"].b)
    assert not torch.allclose(candidates["ALL"].b, sequential_b.b)
