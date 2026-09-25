"""Actual adapter/runner/evaluator composition on synthetic local images."""

import numpy as np
import torch
from PIL import Image

from mcss.data import dl3dv_benchmark as adapter
from mcss.dynamic.carrier import DynamicSceneCarrier
from mcss.dynamic.config import CarrierConfig, WriteConfig
from mcss.dynamic.runner import StreamingRunner
from mcss.dynamic.types import hash_value
from mcss.dynamic.write_rule import DirectWriteRule
from mcss.evaluation.dl3dv_report import _evaluate_scene
from mcss.evaluation.sealed_rgb import official_psnr_metric, official_ssim_metric


def test_real_dl3dv_adapter_streams_before_opening_targets(tmp_path, monkeypatch):
    torch.set_num_threads(2)
    torch.manual_seed(42)
    frames = []
    root = tmp_path / "synthetic_scene"
    folder = root / "nerfstudio/images_4"
    folder.mkdir(parents=True)
    for index in range(20):
        pose = np.eye(4)
        pose[0, 3] = index * .02
        filename = f"frame_{index:05d}.png"
        pixels = np.full((8, 10, 3), 50 + index, dtype=np.uint8)
        Image.fromarray(pixels).save(folder / filename)
        frames.append(adapter.Dl3dvFrame(
            "synthetic_scene", index, f"images_4/{filename}",
            tuple(tuple(float(x) for x in row) for row in pose),
            (8., 8., 4.5, 3.5), (0., 0., 0., 0.), (8, 10),
        ))
    inputs = tuple(index for index in range(20) if index % 8)[:16]
    scene = adapter.Dl3dvSceneMetadata(
        "synthetic_scene", root, tuple(frames), inputs, (0, 8, 16),
        adapter.Dl3dvProtocol("full", 16), target_size=(8, 10),
    )
    opened = []
    has_sealed = False
    original_preprocess = adapter.preprocess_dl3dv_rgb
    original_seal = StreamingRunner.seal

    def spy_preprocess(path, frame, **kwargs):
        if frame.frame_id in scene.query_indices:
            assert has_sealed, "target pixels opened before real runner seal"
        opened.append(frame.frame_id)
        return original_preprocess(path, frame, **kwargs)

    def spy_seal(runner):
        nonlocal has_sealed
        sealed = original_seal(runner)
        has_sealed = True
        return sealed

    monkeypatch.setattr(adapter, "preprocess_dl3dv_rgb", spy_preprocess)
    monkeypatch.setattr(StreamingRunner, "seal", spy_seal)
    config = CarrierConfig(
        feature_dim=4, hidden_dim=4, expansion_dim=8, grid_size=(4, 4, 4), token_count=16,
    )
    carrier = DynamicSceneCarrier(config)
    write = DirectWriteRule(config, WriteConfig())
    model_hash = hash_value({"carrier": carrier.state_dict(), "write_rule": write.state_dict()})
    report = _evaluate_scene(
        scene, base_carrier=carrier, base_write_rule=write, expected_state_hash=model_hash,
        learned_policy=None, policy_name="ALL", residual_threshold=.03,
        device=torch.device("cpu"), max_units=1e12,
        metrics=(official_psnr_metric(), official_ssim_metric()),
        source_factory=adapter.Dl3dvOnlineObservationSource,
    )
    assert report["status"] == "ok", report.get("failure")
    assert report["observed_ids"] == list(inputs)
    assert report["actions"] == ["ALL"] * 12
    assert report["target_frame_ids"] == [0, 8, 16]
    assert opened == list(inputs) + [0, 8, 16]
    assert report["valid_counts"] == {"psnr": 3, "ssim": 3}
