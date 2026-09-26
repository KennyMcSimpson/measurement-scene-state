"""One synthetic gradient pass; no optimizer step, dataset access, or checkpoint."""

import argparse
import json
import time
from pathlib import Path

import torch

from mcss.dynamic.carrier import DynamicSceneCarrier
from mcss.dynamic.config import CarrierConfig
from mcss.dynamic.feedback import batched_camera
from mcss.dynamic.types import OnlineObservation, hash_value
from mcss.measurements import FixedMeasurementRenderer
from mcss.mechanism_pilot.synthetic import fixture
from mcss.mechanism_pilot.training import common_anchor_measurement_loss


def run(output):
    if output.exists():
        raise FileExistsError("Refusing overwrite")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(20260927)
    carrier = DynamicSceneCarrier(CarrierConfig()).to(device)
    renderer = FixedMeasurementRenderer(n_samples=64, ray_chunk_size=2048).to(device)
    before = hash_value(carrier.state_dict())
    observations = []
    for frame in range(5):
        camera, rgb, _, _ = fixture(0, frame, (128, 160))
        observations.append(
            OnlineObservation("synthetic-probe", frame, rgb.to(device), camera.to(device))
        )
    camera, rgb, depth, _ = fixture(0, 8, (128, 160))
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    result = common_anchor_measurement_loss(
        carrier,
        observations[:3],
        [observations[0], observations[3], observations[4]],
        (8,),
        batched_camera(camera.to(device)),
        rgb[None, None].to(device),
        depth[None, None, None].to(device),
        renderer,
        split_id="train",
    )
    result.loss.backward()
    if device.type == "cuda":
        torch.cuda.synchronize()
    gradients = {
        name: float(p.grad.norm()) for name, p in carrier.named_parameters() if p.grad is not None
    }
    assert gradients and all(
        torch.isfinite(p.grad).all() for p in carrier.parameters() if p.grad is not None
    )
    assert hash_value(carrier.state_dict()) == before
    report = {
        "scope": "SYNTHETIC_RESOURCE_PROBE_NOT_TRAINING_RESULT",
        "device": str(device),
        "image_size": [128, 160],
        "contexts": [3, 3],
        "queries": 1,
        "renderer_samples": 64,
        "forward_backward_seconds": time.perf_counter() - start,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated()
        if device.type == "cuda"
        else None,
        "peak_reserved_bytes": torch.cuda.max_memory_reserved() if device.type == "cuda" else None,
        "finite_gradients": True,
        "parameter_values_unchanged": True,
        "optimizer_steps": 0,
        "gradient_norms": gradients,
        "scope_limit": "Static 3+3 contexts only; not Phase B unroll or full cohort throughput",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    run(p.parse_args().output)
