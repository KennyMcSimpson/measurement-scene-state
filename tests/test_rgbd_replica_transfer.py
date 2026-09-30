import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
spec = importlib.util.spec_from_file_location(
    "transfer", SCRIPTS / "evaluate_rgbd_replica_transfer.py"
)
transfer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(transfer)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fake_v11_v12(tmp_path, gain, v12_branch=None):
    v11, v12 = tmp_path / "v11", tmp_path / "v12"
    v11.mkdir(parents=True, exist_ok=True)
    v12.mkdir(parents=True, exist_ok=True)
    (v11 / "eval_v3_results.json").write_text(json.dumps({"COMPLETION_GAIN": {"mean": gain}}))
    if v12_branch is not None:
        (v12 / "integrity.json").write_text(json.dumps({"status": "PASS"}))
        (v12 / "eval_v4_results.json").write_text(json.dumps({"INTERPRETATION_BRANCH": v12_branch}))
    return v11, v12


def test_the_method_follows_the_plan_rule(tmp_path):
    assert transfer.method_rule(*fake_v11_v12(tmp_path / "a", -0.01))[0] == "V11C1"
    assert transfer.method_rule(*fake_v11_v12(tmp_path / "b", 0.02, "A"))[0] == "V12C1"
    assert transfer.method_rule(*fake_v11_v12(tmp_path / "c", 0.02, "B"))[0] == "V11C1"
    with pytest.raises(FileNotFoundError):
        transfer.method_rule(*fake_v11_v12(tmp_path / "d", 0.02))


def test_lock_requires_the_frozen_plan_the_replication_and_prepared_data(tmp_path, monkeypatch):
    root = tmp_path / transfer.EXPERIMENT
    root.mkdir()
    (root / "PROTOCOL.md").write_text("protocol")
    (root / transfer.PLAN).write_text("plan")
    v11, v12 = fake_v11_v12(tmp_path, -0.01)
    (v11 / "integrity.json").write_text(json.dumps({"status": "PASS"}))
    replication, replica = tmp_path / "replication", tmp_path / "replica"
    replication.mkdir()
    replica.mkdir()
    (replica / "preparation_integrity.json").write_text(json.dumps({"status": "PASS"}))
    with pytest.raises(PermissionError):
        transfer.lock(root, replica, tmp_path / "v7", v11, v12, replication)
    monkeypatch.setattr(transfer, "PLAN_SHA256", sha(root / transfer.PLAN))
    with pytest.raises(PermissionError):
        transfer.lock(root, replica, tmp_path / "v7", v11, v12, replication)


def rows(sign, scenes=8):
    rng = np.random.default_rng(1)
    out = []
    base = {"REPROJ_NN": 0.30, "REPROJ_HARMONIC": 0.28, "REPROJ_TELEA": 0.29, "CONSTANT": 0.45}
    carriers = {"V7C1": 0.33, "V11C0": 0.34, "V11C1": 0.32}
    for s in range(scenes):
        for role in ("A", "B"):
            for q in (700, 750):
                common = {
                    "scene_id": f"replica_{s}",
                    "role": role,
                    "query_id": q,
                    "far_fraction": 0.3,
                }
                for method, value in base.items():
                    noise = rng.normal(0, 0.002)
                    out.append(
                        {
                            **common,
                            "method": method,
                            "seed": 0,
                            "depth_absrel": value,
                            "absrel_ALL": value + noise,
                            "absrel_NEAR": value,
                            "absrel_FAR": value + noise,
                        }
                    )
                for method, value in carriers.items():
                    for seed in (1, 2):
                        far = 0.40 - sign * 0.05 if method == "V11C1" else 0.42
                        out.append(
                            {
                                **common,
                                "method": method,
                                "seed": seed,
                                "depth_absrel": value,
                                "absrel_ALL": value,
                                "absrel_NEAR": value,
                                "absrel_FAR": far,
                            }
                        )
                        for prefix in ("HFILL8_", "FILL8_"):
                            hybrid = 0.28 - sign * (0.02 + abs(rng.normal(0, 0.002)))
                            out.append(
                                {
                                    **common,
                                    "method": prefix + method,
                                    "seed": seed,
                                    "depth_absrel": hybrid,
                                    "absrel_ALL": hybrid,
                                    "absrel_NEAR": hybrid,
                                    "absrel_FAR": hybrid,
                                }
                            )
    return out


@pytest.mark.parametrize(("sign", "label", "branch"), [(1, "ABOVE", "A"), (-1, "BELOW", "C")])
def test_analysis_sets_the_primary_label(sign, label, branch):
    result = transfer.analyze_rows(rows(sign), "V11C1")
    assert result["PRIMARY_LABEL"] == label and result["INTERPRETATION_BRANCH"] == branch
    assert result["n_scenes"] == 8
    assert all(entry["gain"] is not None for entry in result["contrasts"].values())
    assert result["contrasts"]["CONSTANT_GAIN"]["label"] == "ABOVE"


def test_every_carrier_loads_with_its_own_module_and_path(tmp_path):
    """Regression: the V11 path parameter once shadowed the V11 module in carrier_specs."""
    from mcss.mechanism_pilot import rgbd_completion_carrier as v11_carrier
    from mcss.mechanism_pilot import rgbd_evidence_carrier as v5

    v7, v11, v12 = tmp_path / "v7", tmp_path / "v11", tmp_path / "v12"
    for directory in (v7, v11):
        directory.mkdir()
    selected = {"v7": {"C1": {}}, "v11": {"C0": {}, "C1": {}}}
    seed = 3
    for variant in ("C0", "C1"):
        path = v11 / f"{variant}.pt"
        carrier = v11_carrier.make_carrier(variant, seed, "cpu")
        v11_carrier.save_checkpoint(
            path, carrier, variant=variant, seed=seed, step=1, lock_sha256="x"
        )
        selected["v11"][variant][str(seed)] = {"path": str(path), "sha256": sha(path), "step": 1}
    path = v7 / "C1.pt"
    v5.save_checkpoint(
        path, v5.make_carrier("C1", seed, "cpu"), variant="C1", seed=seed, step=1, lock_sha256="x"
    )
    selected["v7"]["C1"][str(seed)] = {"path": str(path), "sha256": sha(path), "step": 1}
    (v7 / "selected_checkpoints.json").write_text(json.dumps(selected["v7"]))
    (v11 / "selected_checkpoints.json").write_text(json.dumps(selected["v11"]))
    specs = transfer.carrier_specs("V11C1", v7, v11, v12)
    assert set(specs) == {"V7C1", "V11C0", "V11C1"}
    models = transfer.load_carriers("V11C1", [seed], v7, v11, v12)
    assert set(models) == {("V7C1", seed), ("V11C0", seed), ("V11C1", seed)}
    assert models["V11C1", seed][0].config.hidden_dim == 32
    assert models["V11C0", seed][0].config.hidden_dim == 8
    assert models["V11C1", seed][1] is v11_carrier.build_state
    assert models["V7C1", seed][1] is v5.build_state
