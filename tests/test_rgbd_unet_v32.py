"""V32: a whole-volume 3D U-Net context branch on the V17 L carrier; scripts and branch."""

import importlib.util
from pathlib import Path

import pytest
import torch
from test_rgbd_stream_write_v3 import contexts

from mcss.dynamic.types import hash_scene_state
from mcss.mechanism_pilot import rgbd_long_train_carrier as v17
from mcss.mechanism_pilot import rgbd_unet_carrier as v32

SEED = 20260928
SCRIPTS = Path(__file__).parents[1] / "scripts"


def script(name):
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), SCRIPTS / name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_shared_parameters_are_the_v17_draw_and_the_global_rng_is_untouched():
    base = v17.make_carrier("L", SEED, "cpu")
    after_base = torch.get_rng_state()
    mine = v32.make_carrier("L", SEED, "cpu")
    assert torch.equal(torch.get_rng_state(), after_base)
    state = mine.state_dict()
    assert all(k.startswith("context_unet.") for k in set(state) - set(base.state_dict()))
    assert all(torch.equal(state[k], v) for k, v in base.state_dict().items())
    assert sum(p.numel() for p in mine.context_unet.parameters()) == 361856
    assert not mine.context_unet.out.weight.any() and not mine.context_unet.out.bias.any()
    first = mine.context_unet.enc1.weight
    assert torch.equal(v32.make_carrier("L", SEED, "cpu").context_unet.enc1.weight, first)
    assert not torch.equal(v32.make_carrier("L", SEED + 1, "cpu").context_unet.enc1.weight, first)


def test_zero_output_builds_the_v17_l_state_and_a_live_branch_changes_it():
    torch.set_num_threads(1)
    warm, _, bounds, _ = contexts()
    base = v17.make_carrier("L", SEED, "cpu")
    mine = v32.make_carrier("L", SEED, "cpu")
    with torch.no_grad():
        expected, anchor = v17.build_state(base, warm, bounds, "e")
        state, mine_anchor = v32.build_state(mine, warm, bounds, "e")
    assert hash_scene_state(state) == hash_scene_state(expected)
    assert torch.equal(anchor, mine_anchor)
    with torch.no_grad():
        mine.context_unet.out.weight.normal_(0.0, 0.1, generator=torch.Generator().manual_seed(1))
        live, _ = v32.build_state(mine, warm, bounds, "e")
    assert hash_scene_state(live) != hash_scene_state(expected)


def test_every_output_voxel_sees_the_whole_volume():
    unet = v32.make_carrier("L", SEED, "cpu").context_unet
    with torch.no_grad():
        unet.out.weight.normal_(0.0, 0.1, generator=torch.Generator().manual_seed(2))
    generator = torch.Generator().manual_seed(3)
    volume = torch.randn(1, 32, 32, 32, 32, generator=generator).requires_grad_()
    unet(volume)[0, :, 31, 31, 31].sum().backward()
    assert volume.grad[0, :, 0, 0, 0].abs().sum() > 0


def test_checkpoint_roundtrip_and_schema_guards(tmp_path):
    carrier = v32.make_carrier("L", SEED, "cpu")
    with torch.no_grad():
        carrier.context_unet.out.weight.add_(0.01)
    path = tmp_path / "step_000100.pt"
    digest = v32.save_checkpoint(path, carrier, variant="L", seed=SEED, step=100, lock_sha256="x")
    assert len(digest) == 64
    restored, payload = v32.load_checkpoint(path, "cpu")
    assert isinstance(restored, v32.UNetContextCarrier) and payload["unet"] == v32.UNET
    assert all(
        torch.equal(a, b)
        for a, b in zip(restored.state_dict().values(), carrier.state_dict().values(), strict=True)
    )
    with pytest.raises(PermissionError):
        v17.load_checkpoint(path, "cpu")
    old = tmp_path / "v17.pt"
    v17.save_checkpoint(
        old, v17.make_carrier("L", SEED, "cpu"), variant="L", seed=SEED, step=1, lock_sha256="x"
    )
    with pytest.raises(PermissionError):
        v32.load_checkpoint(old, "cpu")
    payload["unet"] = {**payload["unet"], "channels": 64}
    torch.save(payload, tmp_path / "other_branch.pt")
    with pytest.raises(PermissionError):
        v32.load_checkpoint(tmp_path / "other_branch.pt", "cpu")


def test_derived_scripts_change_only_the_carrier_module():
    train = (SCRIPTS / "train_rgbd_unet_v32.py").read_text()
    frozen_train = (SCRIPTS / "train_rgbd_long_train_carrier.py").read_text()
    start, end = "def train(", '        "dynamic_ttt_run": False,\n'
    assert (
        train[train.index(start) : train.index(end)]
        == frozen_train[frozen_train.index(start) : frozen_train.index(end)]
    )
    assert "rgbd_unet_carrier import (" in train and "rgbd_long_train_carrier" not in train
    evaluate = (SCRIPTS / "evaluate_rgbd_unet_eval_v32.py").read_text()
    frozen_evaluate = (SCRIPTS / "evaluate_rgbd_long_train_eval.py").read_text()
    line = "build_state, load_checkpoint\n"
    assert "from mcss.mechanism_pilot.rgbd_unet_carrier import " + line in evaluate
    assert evaluate.split(line, 1)[1] == frozen_evaluate.split(line, 1)[1]


def test_branch_needs_both_the_context_gain_and_the_hybrid_label():
    analysis = script("analyze_rgbd_unet_v32.py")
    assert analysis.ARMS == ("G1", "U")
    assert analysis.branch("SUPPORTED", "ABOVE") == "A"
    assert analysis.branch("SUPPORTED", "NOT_DISTINGUISHABLE") == "B"
    assert analysis.branch("PARTIAL", "ABOVE") == "B"
    assert analysis.branch("NOT_ESTABLISHED", "BELOW") == "C"
