"""V2 runner candidate isolation and complete-prediction lock contracts."""

from types import SimpleNamespace

import pytest
import torch

from mcss.vision_probe.adaptation import Readout
from mcss.vision_probe.v2_runner import (
    METHODS,
    TRAJECTORIES,
    _candidate_states,
    _prediction_checks,
    sha,
    write,
)


def test_off_isolation_and_reproducible_candidate_rollouts():
    torch.manual_seed(12)
    target = SimpleNamespace(features=torch.randn(2, 2, 64), rgb=torch.rand(2, 2, 3))
    readout = Readout(
        torch.zeros(64), torch.eye(64), torch.ones(64), torch.randn(64, 3), torch.zeros(3)
    )
    original = target.features.clone()
    _, first, cost = _candidate_states(target, readout)
    _, second, _ = _candidate_states(target, readout)
    assert torch.equal(first[0][1], original)
    assert not first[0][0].a.any() and not first[0][0].b.any()
    assert torch.equal(target.features, original)
    for (left, a), (right, b) in zip(first, second, strict=True):
        assert torch.equal(left.a, right.a) and torch.equal(left.b, right.b)
        assert torch.equal(a, b)
    first[1][0].a.fill_(99)
    assert not first[0][0].a.any()
    assert not torch.equal(first[1][0].a, first[2][0].a)
    assert cost["proposal_calls"] == 32 and cost["increment_calculations"] == 64


def setup_lock(tmp_path, monkeypatch):
    import mcss.vision_probe.v2_runner as module

    manifest = tmp_path / "manifest.json"
    write(
        manifest,
        {"sequences": [{"sequence": "new", "targets": [{"target_slot": 1}, {"target_slot": 2}]}]},
    )
    write(tmp_path / "selector_lock.json", {})
    write(tmp_path / "config.json", {})
    lock = {
        "selector_lock_sha256": sha(tmp_path / "selector_lock.json"),
        "config_sha256": sha(tmp_path / "config.json"),
        "split_sha256": sha(manifest),
        "source_hashes": {},
    }
    write(tmp_path / "prediction_lock.json", lock)
    monkeypatch.setattr(module, "_source_hashes", lambda: {})
    monkeypatch.setattr(module, "_verify_selectors", lambda p: {})
    document = {
        "complete": True,
        "prediction_lock_sha256": sha(tmp_path / "prediction_lock.json"),
        "predictions": [
            {
                "sequence": "new",
                "target_slot": i,
                "choices": {m: 0 for m in METHODS},
                "trajectories": [list(t) for t in TRAJECTORIES],
                "selector_lock_sha256": lock["selector_lock_sha256"],
            }
            for i in (1, 2)
        ],
    }
    return manifest, document


def test_prediction_gate_rejects_missing_pair(tmp_path, monkeypatch):
    manifest, document = setup_lock(tmp_path, monkeypatch)
    document["predictions"].pop()
    with pytest.raises(ValueError, match="pair coverage"):
        _prediction_checks(tmp_path, manifest, document)


def test_prediction_gate_rejects_missing_method_or_candidate(tmp_path, monkeypatch):
    manifest, document = setup_lock(tmp_path, monkeypatch)
    del document["predictions"][0]["choices"]["CycleGate"]
    with pytest.raises(ValueError, match="deployment decisions"):
        _prediction_checks(tmp_path, manifest, document)
    document["predictions"][0]["choices"]["CycleGate"] = 0
    document["predictions"][0]["trajectories"].pop()
    with pytest.raises(ValueError, match="candidate trajectories"):
        _prediction_checks(tmp_path, manifest, document)


def test_prediction_gate_rejects_changed_source(tmp_path, monkeypatch):
    import mcss.vision_probe.v2_runner as module

    manifest, document = setup_lock(tmp_path, monkeypatch)
    monkeypatch.setattr(module, "_source_hashes", lambda: {"changed.py": "different"})
    with pytest.raises(ValueError, match="changed"):
        _prediction_checks(tmp_path, manifest, document)


def test_complete_synthetic_prediction_artifacts_verify(tmp_path, monkeypatch):
    manifest, document = setup_lock(tmp_path, monkeypatch)
    for row in document["predictions"]:
        path = tmp_path / f"tensor_{row['target_slot']}.pt"
        torch.save(
            {
                "source_features": torch.zeros(1).expand(32, 32, 384),
                "adapted": torch.zeros(1).expand(16, 32, 32, 384),
                "choices": row["choices"],
                "trajectories": row["trajectories"],
            },
            path,
        )
        row["artifact_path"] = str(path)
        row["artifact_sha256"] = sha(path)
    assert _prediction_checks(tmp_path, manifest, document)["source_hashes"] == {}
    payload = torch.load(document["predictions"][0]["artifact_path"], weights_only=True)
    payload["adapted"] = payload["adapted"][:15]
    torch.save(payload, document["predictions"][0]["artifact_path"])
    document["predictions"][0]["artifact_sha256"] = sha(document["predictions"][0]["artifact_path"])
    with pytest.raises(ValueError, match="candidate schema"):
        _prediction_checks(tmp_path, manifest, document)
