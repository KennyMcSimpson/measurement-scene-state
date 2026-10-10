"""V31: frozen DINOv2 features through a zero bypass on the V17 L carrier; scripts and branch."""

import importlib.util
from pathlib import Path

import pytest
import torch
from test_rgbd_stream_write_v3 import contexts

from mcss.dynamic.types import hash_scene_state
from mcss.mechanism_pilot import rgbd_long_train_carrier as v17
from mcss.mechanism_pilot import rgbd_pretrained_carrier as v31

SEED = 20260928
SCRIPTS = Path(__file__).parents[1] / "scripts"
needs_backbone = pytest.mark.skipif(
    not v31.backbone_files_present(), reason="the DINOv2 snapshot exists only on the training host"
)


def script(name):
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), SCRIPTS / name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_shared_parameters_are_the_v17_draw_and_the_global_rng_is_untouched():
    base = v17.make_carrier("L", SEED, "cpu")
    after_base = torch.get_rng_state()
    mine = v31.make_carrier("L", SEED, "cpu")
    assert torch.equal(torch.get_rng_state(), after_base)
    state = mine.state_dict()
    extra = set(state) - set(base.state_dict())
    assert extra == {"feature_projection", "feature_projection_bias", "pretrained_weight"}
    assert all(torch.equal(state[k], v) for k, v in base.state_dict().items())
    assert sum(state[k].numel() for k in extra) == 384 * 24 + 24 + 48 * 64 == 12312
    assert not mine.pretrained_weight.any() and not mine.feature_projection_bias.any()
    assert torch.equal(
        v31.make_carrier("L", SEED, "cpu").feature_projection, state["feature_projection"]
    )
    assert not torch.equal(
        v31.make_carrier("L", SEED + 1, "cpu").feature_projection, state["feature_projection"]
    )
    bound = 1 / 384**0.5
    assert mine.feature_projection.abs().max() <= bound
    assert mine.feature_projection.std() > bound / 4


def test_checkpoint_roundtrip_and_schema_and_backbone_guards(tmp_path):
    carrier = v31.make_carrier("L", SEED, "cpu")
    path = tmp_path / "step_000100.pt"
    digest = v31.save_checkpoint(path, carrier, variant="L", seed=SEED, step=100, lock_sha256="x")
    assert len(digest) == 64
    restored, payload = v31.load_checkpoint(path, "cpu")
    assert isinstance(restored, v31.PretrainedFeatureCarrier)
    assert payload["backbone"] == v31.BACKBONE and payload["projected"] == v31.PROJECTED == 24
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
        v31.load_checkpoint(old, "cpu")
    payload["backbone"] = {**payload["backbone"], "weights_sha256": "0" * 64}
    torch.save(payload, tmp_path / "other_backbone.pt")
    with pytest.raises(PermissionError):
        v31.load_checkpoint(tmp_path / "other_backbone.pt", "cpu")


@needs_backbone
def test_backbone_is_registered_frozen_cached_and_outside_the_state_dict():
    assert v31.code_tree_sha256(v31.repository()) == v31.BACKBONE["code_tree_sha256"]
    model = v31.load_backbone("cpu")
    assert v31.load_backbone("cpu") is model and not model.training
    assert not any(p.requires_grad for p in model.parameters())
    warm, _, _, _ = contexts()
    tokens = v31.patch_tokens(warm[0].rgb)
    assert tokens.shape == (384, 18, 23) and not tokens.requires_grad
    assert torch.isfinite(tokens).all()
    assert v31.patch_tokens(warm[0].rgb.clone()) is tokens
    carrier = v31.make_carrier("L", SEED, "cpu")
    assert len(carrier.state_dict()) == len(v17.make_carrier("L", SEED, "cpu").state_dict()) + 3
    features = carrier.encode(warm[0])
    assert features.shape == (32, *warm[0].rgb.shape[-2:])
    assert torch.equal(features[:8], v17.make_carrier("L", SEED, "cpu").encode(warm[0]))
    features[8:].square().sum().backward()
    assert carrier.feature_projection.grad.abs().sum() > 0


@needs_backbone
def test_zero_bypass_builds_the_v17_l_state_and_a_live_bypass_changes_it():
    torch.set_num_threads(1)
    warm, _, bounds, _ = contexts()
    base = v17.make_carrier("L", SEED, "cpu")
    mine = v31.make_carrier("L", SEED, "cpu")
    with torch.no_grad():
        expected, anchor = v17.build_state(base, warm, bounds, "e")
        state, mine_anchor = v31.build_state(mine, warm, bounds, "e")
    assert hash_scene_state(state) == hash_scene_state(expected)
    assert torch.equal(anchor, mine_anchor)
    with torch.no_grad():
        mine.pretrained_weight.normal_(0.0, 0.1, generator=torch.Generator().manual_seed(1))
        live, _ = v31.build_state(mine, warm, bounds, "e")
    assert hash_scene_state(live) != hash_scene_state(expected)


def test_derived_scripts_change_only_the_carrier_module():
    train = (SCRIPTS / "train_rgbd_pretrained_v31.py").read_text()
    frozen_train = (SCRIPTS / "train_rgbd_long_train_carrier.py").read_text()
    start, end = "def train(", '        "dynamic_ttt_run": False,\n'
    assert (
        train[train.index(start) : train.index(end)]
        == frozen_train[frozen_train.index(start) : frozen_train.index(end)]
    )
    assert "rgbd_pretrained_carrier import (" in train and "rgbd_long_train_carrier" not in train
    evaluate = (SCRIPTS / "evaluate_rgbd_pretrained_eval_v31.py").read_text()
    frozen_evaluate = (SCRIPTS / "evaluate_rgbd_long_train_eval.py").read_text()
    line = "build_state, load_checkpoint\n"
    assert "from mcss.mechanism_pilot.rgbd_pretrained_carrier import " + line in evaluate
    assert evaluate.split(line, 1)[1] == frozen_evaluate.split(line, 1)[1]


def test_branch_needs_both_the_feature_gain_and_the_hybrid_label():
    analysis = script("analyze_rgbd_pretrained_v31.py")
    assert analysis.ARMS == ("G1", "P")
    assert analysis.branch("SUPPORTED", "ABOVE") == "A"
    assert analysis.branch("SUPPORTED", "NOT_DISTINGUISHABLE") == "B"
    assert analysis.branch("PARTIAL", "ABOVE") == "B"
    assert analysis.branch("NOT_ESTABLISHED", "BELOW") == "C"
