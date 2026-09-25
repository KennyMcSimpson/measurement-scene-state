"""Run minimal CUDA forward/backward checks for every model mode and state architecture."""

from __future__ import annotations

import json

import torch

from mcss.geometry import look_at, make_intrinsics
from mcss.model.system import ModelConfig, build_model
from mcss.types import Cameras


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available in this PyTorch environment")

    device = torch.device("cuda")
    torch.manual_seed(3)
    context_rgb = torch.rand(1, 3, 3, 8, 8, device=device)
    intrinsics = make_intrinsics((8, 8), 60.0).to(device)
    context_poses = torch.stack(
        (
            look_at(torch.tensor([0.0, 0.0, -3.0]), torch.zeros(3)),
            look_at(torch.tensor([2.0, 0.0, -2.0]), torch.zeros(3)),
            look_at(torch.tensor([-2.0, 0.0, -2.0]), torch.zeros(3)),
        )
    ).to(device)
    target_pose = look_at(torch.tensor([-2.0, 0.0, -2.0]), torch.zeros(3)).to(device)
    context_cameras = Cameras(
        intrinsics.expand(1, 3, 3, 3).clone(),
        context_poses.unsqueeze(0),
        (8, 8),
    )
    target_cameras = Cameras(
        intrinsics.view(1, 1, 3, 3),
        target_pose.view(1, 1, 4, 4),
        (8, 8),
    )
    bounds = torch.tensor(
        [[[-1.2, -1.2, -1.2], [1.2, 1.2, 1.2]]],
        device=device,
    )

    results: dict[str, object] = {}
    cases = [
        (state_architecture, mode, 1)
        for state_architecture in (
            "legacy",
            "evidence_residual",
            "dual_evidence",
            "dual_evidence_transport",
        )
        for mode in ("fixed", "learned_heads", "free_decoder")
    ]
    cases.append(("evidence_residual", "fixed", 2))
    cases.append(("dual_evidence", "fixed", 2))
    cases.append(("dual_evidence_transport", "fixed", 2))
    for state_architecture, mode, appearance_scale in cases:
        name = f"{state_architecture}/{mode}/appearance{appearance_scale}x"
        torch.cuda.reset_peak_memory_stats(device)
        model = build_model(
            ModelConfig(
                mode=mode,
                state_architecture=state_architecture,
                voxel_resolution=8,
                image_feature_dim=8,
                state_feature_dim=8,
                refinement_blocks=1,
                appearance_resolution_scale=appearance_scale,
                n_samples=8,
                ray_chunk_size=128,
            )
        ).to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = model(context_rgb, context_cameras, target_cameras, bounds)
        loss = sum(value.float().mean() for value in output.predictions.values())
        loss.backward()
        gradients = [parameter.grad for parameter in model.parameters() if parameter.requires_grad]
        if not torch.isfinite(loss):
            raise RuntimeError(f"{name} produced a non-finite loss")
        if not gradients or not any(gradient is not None for gradient in gradients):
            raise RuntimeError(f"{name} produced no trainable gradients")
        if not all(
            torch.isfinite(gradient).all() for gradient in gradients if gradient is not None
        ):
            raise RuntimeError(f"{name} produced non-finite gradients")
        if state_architecture == "evidence_residual" and output.state.evidence is None:
            raise RuntimeError(f"{name} did not expose evidence fields")
        if state_architecture == "dual_evidence" and output.state.dual_evidence is None:
            raise RuntimeError(f"{name} did not expose dual-evidence fields")
        if (
            state_architecture == "dual_evidence_transport"
            and output.state.transport_evidence is None
        ):
            raise RuntimeError(f"{name} did not expose transport-evidence fields")
        if state_architecture != "dual_evidence" and output.state.dual_evidence is not None:
            raise RuntimeError(f"{name} unexpectedly exposed dual-evidence fields")
        if state_architecture != "evidence_residual" and output.state.evidence is not None:
            raise RuntimeError(f"{name} unexpectedly exposed evidence fields")
        if (
            state_architecture != "dual_evidence_transport"
            and output.state.transport_evidence is not None
        ):
            raise RuntimeError(f"{name} unexpectedly exposed transport-evidence fields")
        if sum(parameter.numel() for parameter in model.renderer.parameters()) != 0:
            raise RuntimeError(f"{name} renderer unexpectedly has parameters")
        if appearance_scale == 2 and output.state.appearance is None:
            raise RuntimeError(f"{name} did not expose high-resolution appearance")
        if appearance_scale == 1 and output.state.appearance is not None:
            raise RuntimeError(f"{name} unexpectedly exposed high-resolution appearance")
        results[name] = {
            "loss": float(loss.detach()),
            "measurements": sorted(output.predictions),
            "peak_memory_mb": round(torch.cuda.max_memory_allocated() / 1024**2, 2),
        }
        del model, output, loss, gradients
        torch.cuda.empty_cache()

    torch.cuda.synchronize()
    results["cuda"] = {
        "device": torch.cuda.get_device_name(0),
    }
    print(json.dumps(results, sort_keys=True))


if __name__ == "__main__":
    main()
