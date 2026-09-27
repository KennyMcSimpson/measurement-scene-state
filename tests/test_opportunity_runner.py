import inspect
import json

import pytest
import torch
from PIL import Image
from torch import nn

from mcss.vision_probe import opportunity_runner as runner
from mcss.vision_probe.adaptation import Readout
from mcss.vision_probe.experiment import FrameFeatures
from mcss.vision_probe.features import FrozenDinoExtractor
from mcss.vision_probe.opportunity_selectors import select_candidates


@pytest.mark.parametrize("size,grid", [(224, 16), (448, 32)])
def test_true_input_resolution_not_interpolation(tmp_path, monkeypatch, size, grid):
    class Backbone(nn.Module):
        embed_dim = 384
        patch_size = 14

        def load_state_dict(self, state, strict=True):
            assert strict

        def forward_features(self, x):
            assert x.shape == (1, 3, size, size)
            return {"x_norm_patchtokens": torch.ones(1, (x.shape[-1] // 14) ** 2, 384)}

    weights = tmp_path / "weights.pt"
    torch.save({}, weights)
    image = tmp_path / "image.png"
    Image.new("RGB", (51, 37)).save(image)
    monkeypatch.setattr(torch.hub, "load", lambda *a, **kw: Backbone())
    result = FrozenDinoExtractor(tmp_path, weights, "cpu", size).extract(image)
    assert result["features"].shape == (grid, grid, 384)
    assert result["rgb"].shape == (grid, grid, 3)


def test_candidate_isolation_off_and_rank():
    torch.manual_seed(2)
    readout = Readout(
        torch.zeros(64), torch.eye(64), torch.ones(64), torch.randn(64, 3) * 0.1, torch.zeros(3)
    )
    features = torch.randn(4, 4, 64)
    target = FrameFeatures(features, torch.rand(4, 4, 3), (56, 56), (56, 56), "fake")
    _, states = runner.candidate_states(target, readout, 5.0)
    state, adapted, audits = states[0]
    assert state.a.count_nonzero() == state.b.count_nonzero() == 0
    assert torch.equal(adapted, features)
    assert all(a["numerical_rank"] <= 3 for _, _, aa in states for a in aa)
    states[1][0].a.fill_(123)
    assert states[0][0].a.count_nonzero() == 0
    assert not torch.equal(states[1][0].a, states[2][0].a)


def test_guard_denies_official_val_reserve_and_unopened_internal_validation(tmp_path):
    splits = {
        "fit": ["fit"],
        "discovery": ["disc"],
        "validation": ["val"],
        "reserve": ["reserve"],
        "official_val_sealed": ["official"],
    }
    guard = runner.MediaGuard(tmp_path, splits)
    guard.allow("discovery")
    guard.check_path(tmp_path / "JPEGImages/480p/disc/00000.jpg")
    for sequence in ("official", "reserve", "val", "fit"):
        with pytest.raises(PermissionError):
            guard.check_path(tmp_path / "Annotations/480p" / sequence / "00000.png")
    with pytest.raises(ValueError):
        guard.allow("reserve")


def test_split_identity_overlap_rejected():
    path = runner.ROOT / "docs/experiments/EXP-2D-20260921-corrected/protocol.json"
    splits = json.loads(path.read_text())["splits"]
    runner.validate_splits(splits)
    splits["validation"][0] = splits["discovery"][0]
    with pytest.raises(ValueError):
        runner.validate_splits(splits)


def test_selector_signature_no_target_answers():
    assert set(inspect.signature(select_candidates).parameters) == {
        "features",
        "trajectories",
        "artifact",
    }


def test_validation_lock_detects_selector_changes(tmp_path, monkeypatch):
    report = tmp_path / "report"
    work = tmp_path / "work"
    report.mkdir()
    work.mkdir()
    monkeypatch.setattr(runner, "REPORT", report)
    monkeypatch.setattr(runner, "WORK", work)
    weights = tmp_path / "weights.pt"
    weights.write_bytes(b"self-contained weight artifact for lock hashing")
    monkeypatch.setattr(runner, "WEIGHTS", weights)
    monkeypatch.setattr(runner, "source_hashes", lambda: {"runner": "sourcehash"})
    for name in ("config.json", "split_manifest.json", "selectors.json", "discovery_raw.jsonl"):
        (report / name).write_text("{}")
    for size in runner.CONFIG["resolutions"]:
        (work / f"readout_{size}.pt").write_text("readout")
    lock = {
        "source_hashes": runner.source_hashes(),
        "config_sha256": runner.sha(report / "config.json"),
        "split_sha256": runner.sha(report / "split_manifest.json"),
        "selectors_sha256": runner.sha(report / "selectors.json"),
        "weights_sha256": runner.sha(runner.WEIGHTS),
        "discovery_sha256": runner.sha(report / "discovery_raw.jsonl"),
        "readout_sha256": {
            str(s): runner.sha(work / f"readout_{s}.pt") for s in runner.CONFIG["resolutions"]
        },
        "fit_split": "discovery",
    }
    runner.write_json(report / "validation_lock.json", lock)
    runner.verify_lock()
    (report / "selectors.json").write_text('{"threshold":999}')
    with pytest.raises(RuntimeError, match="selectors_sha256"):
        runner.verify_lock()
