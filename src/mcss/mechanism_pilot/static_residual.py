"""Frozen-carrier residual control; training-side labels only, no dynamic writes."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from mcss.dynamic.checkpoint import load_dynamic_checkpoint
from mcss.dynamic.types import hash_scene_state, hash_value
from mcss.geometry import generate_rays, transform_cameras
from mcss.measurements import FixedMeasurementRenderer
from mcss.mechanism_pilot.contracts import ResidualReadoutControl
from mcss.mechanism_pilot.small_training import AccessLog, TrainScene, sha, write_json
from mcss.training.grounded import build_grounded_state
from mcss.types import SceneState

SCHEMA = "mcss.static_residual.v1"
ZERO_STATE_DEFINITION = (
    "same bounds; density_logits=-100; color/features/log_variance=0; no evidence payload"
)


@dataclass(frozen=True)
class ResidualTrainingConfig:
    seed: int = 20260927
    steps: int = 3000
    learning_rate: float = 1e-3
    gradient_clip: float = 1.0
    validation_interval: int = 250
    renderer_samples: int = 64
    ray_chunk_size: int = 2048


def validate_training_manifest(manifest, training_scenes):
    """Identity firewall runs before loading any RGB/depth from manifest."""
    rows = manifest["scenes"]
    actual = [row["scene_id"] for row in rows]
    if len(actual) != len(set(actual)) or set(actual) != set(training_scenes):
        raise PermissionError(
            "Residual training scene identities must equal checkpoint train allowlist"
        )
    for row in rows:
        if row.get("split") != "train":
            raise PermissionError("Residual trainer rejects unseen/dev/test scene records")
        if row["roles"]["query"] != [12, 13, 14, 15]:
            raise ValueError("Locked residual train/validation query split changed")
        context = set(row["roles"]["context_a"]) | set(row["roles"]["context_b"])
        if context.intersection([8, 9, 12, 13, 14, 15]):
            raise PermissionError("Query entered state construction")


def zero_state_like(state):
    """DIAGNOSTIC ONLY: empty near-transparent volume, identical coordinate bounds."""
    return SceneState(
        torch.full_like(state.density_logits, -100),
        torch.zeros_like(state.color),
        torch.zeros_like(state.log_variance),
        state.bounds.detach().clone(),
        features=None if state.features is None else torch.zeros_like(state.features),
    )


def cache_readout(model, state, cameras):
    """Cache detached immutable fixed measurement inputs; no GT accepted."""
    state_hash, camera_hash = hash_scene_state(state), hash_value(cameras)
    with torch.no_grad():
        base = model.renderer(state, cameras, {"rgb", "depth", "visibility"})
        features = model.renderer.render_features(state, cameras)
        origins, directions = generate_rays(cameras)
        rays = torch.cat((origins, directions), dim=-1).permute(0, 1, 4, 2, 3)
        combined = torch.cat((features, rays), dim=2).detach()
    if state_hash != hash_scene_state(state) or camera_hash != hash_value(cameras):
        raise RuntimeError("Cache construction mutated shared inputs")
    return {
        "base": base,
        "combined": combined,
        "state_hash": state_hash,
        "camera_hash": camera_hash,
    }


def cached_forward(model, cache):
    combined, base = cache["combined"], cache["base"]
    batch, views, channels, height, width = combined.shape
    residual = model.head(combined.reshape(batch * views, channels, height, width))
    residual = residual.reshape(batch, views, 4, height, width)
    result = dict(base)
    result["rgb"] = (base["rgb"] + residual[:, :, :3]).clamp(0, 1)
    result["depth"] = (base["depth"] + residual[:, :, 3:4]).clamp_min(0)
    return result


def supervised_loss(prediction, rgb, depth):
    valid = torch.isfinite(depth) & (depth > 0)
    if not valid.any():
        raise ValueError("No valid train depth")
    rgb_mse = (prediction["rgb"] - rgb).square().mean()
    absrel = ((prediction["depth"][valid] - depth[valid]).abs() / depth[valid]).mean()
    return rgb_mse + absrel, {
        "rgb_mse": float(rgb_mse.detach()),
        "depth_absrel": float(absrel.detach()),
    }


def load_residual_checkpoint(path, *, carrier_sha256, device="cpu"):
    payload = torch.load(path, map_location=device, weights_only=True)
    if payload.get("schema") != SCHEMA or payload["carrier_sha256"] != carrier_sha256:
        raise ValueError("Residual schema/carrier checkpoint mismatch")
    config = ResidualTrainingConfig(**payload["config"])
    renderer = FixedMeasurementRenderer(
        n_samples=config.renderer_samples, ray_chunk_size=config.ray_chunk_size
    ).to(device)
    model = ResidualReadoutControl(payload["feature_dim"], renderer).to(device)
    model.head.load_state_dict(payload["head_state_dict"], strict=True)
    model.eval().requires_grad_(False)
    model.training_status = "TRAINED_FROZEN_TRAIN_SIDE_SELECTED"
    return model, payload


def train_residual(manifest_path, carrier_path, output_dir, *, device="cuda", config=None):
    config = config or ResidualTrainingConfig()
    if min(config.steps, config.validation_interval) <= 0:
        raise ValueError("Positive training and validation budgets required")
    manifest_path, carrier_path, output = map(Path, (manifest_path, carrier_path, output_dir))
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Preserve previous residual results")
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(manifest_path.read_text())
    carrier, _, metadata = load_dynamic_checkpoint(carrier_path, device=device)
    training_scenes = metadata["provenance"]["training_scenes"]
    validate_training_manifest(manifest, training_scenes)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    carrier.eval().requires_grad_(False)
    renderer = FixedMeasurementRenderer(
        n_samples=config.renderer_samples, ray_chunk_size=config.ray_chunk_size
    ).to(device)
    model = ResidualReadoutControl(carrier.config.feature_dim, renderer).to(device)
    carrier_before = hash_value(carrier.state_dict())
    log = AccessLog(output / "label_access.jsonl")
    write_json(
        output / "lock.json",
        {
            "config": asdict(config),
            "carrier_sha256": sha(carrier_path),
            "manifest_sha256": sha(manifest_path),
            "training_scenes": training_scenes,
            "train_query_ids": [12, 13, 14],
            "validation_query_ids": [15],
            "validation_caveat": (
                "15 was seen by frozen carrier; held out only from residual optimizer"
            ),
            "primary_query_ids_never_loaded": [8, 9],
            "zero_state_definition": ZERO_STATE_DEFINITION,
            "selection": "lowest equal scene/context mean RGB MSE + depth AbsRel on query 15",
            "architecture": "existing ResidualReadoutControl unchanged",
        },
    )
    train, validation, cached_checks = [], [], []
    start = time.perf_counter()
    for row in manifest["scenes"]:
        scene = TrainScene(row, manifest["image_size"], manifest_path.parent, device, log)
        for role in ("context_a", "context_b"):
            with torch.no_grad():
                state, anchor = build_grounded_state(
                    carrier, scene.observations(role), f"residual:{row['scene_id']}:{role}"
                )
            for slot in range(4):
                frame_id, camera, rgb, depth = scene.query(
                    slot, "residual_validation" if slot == 3 else "residual_optimizer_training"
                )
                local = transform_cameras(camera, torch.linalg.inv(anchor))
                cache = cache_readout(model, state, local)
                with torch.no_grad():
                    direct, cached = model(state, local), cached_forward(model, cache)
                exact = all(torch.equal(direct[k], cached[k]) for k in direct)
                if not exact:
                    raise RuntimeError("Cached readout differs from original forward")
                item = {
                    "scene_id": row["scene_id"],
                    "context": role,
                    "query_id": frame_id,
                    "cache": cache,
                    "rgb": rgb,
                    "depth": depth,
                    "state": state,
                    "camera": local,
                }
                (validation if slot == 3 else train).append(item)
                cached_checks.append(
                    {
                        "scene_id": row["scene_id"],
                        "context": role,
                        "query_id": frame_id,
                        "initial_exact": exact,
                    }
                )
    cache_seconds = time.perf_counter() - start
    optimizer = torch.optim.Adam(model.head.parameters(), lr=config.learning_rate)
    best, best_step = float("inf"), None
    curves = []
    optimizer_start = time.perf_counter()

    def evaluate(step):
        nonlocal best, best_step
        with torch.no_grad():
            losses = [
                float(supervised_loss(cached_forward(model, r["cache"]), r["rgb"], r["depth"])[0])
                for r in validation
            ]
            training_loss = np.mean(
                [
                    float(
                        supervised_loss(cached_forward(model, r["cache"]), r["rgb"], r["depth"])[0]
                    )
                    for r in train
                ]
            )
        val_loss = float(np.mean(losses))
        curves.append(
            {
                "step": step,
                "train_mean_loss": float(training_loss),
                "validation_mean_loss": val_loss,
                "validation_per_item": losses,
            }
        )
        if val_loss < best:
            best, best_step = val_loss, step
            torch.save(
                {
                    "schema": SCHEMA,
                    "head_state_dict": model.head.state_dict(),
                    "feature_dim": carrier.config.feature_dim,
                    "config": asdict(config),
                    "carrier_sha256": sha(carrier_path),
                    "manifest_sha256": sha(manifest_path),
                    "training_scenes": training_scenes,
                    "selected_step": step,
                    "validation_loss": val_loss,
                },
                output / "best.pt",
            )
        write_json(output / "validation_curve.json", curves)

    evaluate(0)
    with (output / "training.jsonl").open("w") as handle:
        for step in range(config.steps):
            row = train[step % len(train)]
            optimizer.zero_grad(set_to_none=True)
            loss, terms = supervised_loss(
                cached_forward(model, row["cache"]), row["rgb"], row["depth"]
            )
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.head.parameters(), config.gradient_clip)
            if not torch.isfinite(norm) or not torch.isfinite(loss):
                raise RuntimeError("Nonfinite residual optimization")
            optimizer.step()
            handle.write(
                json.dumps(
                    {
                        "step": step + 1,
                        "scene_id": row["scene_id"],
                        "context": row["context"],
                        "query_id": row["query_id"],
                        "loss": float(loss.detach()),
                        "gradient_norm": float(norm),
                        **terms,
                    }
                )
                + "\n"
            )
            if (step + 1) % config.validation_interval == 0 or step + 1 == config.steps:
                evaluate(step + 1)
    optimizer_seconds = time.perf_counter() - optimizer_start
    model, _ = load_residual_checkpoint(
        output / "best.pt", carrier_sha256=sha(carrier_path), device=device
    )
    with torch.no_grad():
        for row in train + validation:
            direct, cached = model(row["state"], row["camera"]), cached_forward(model, row["cache"])
            if not all(torch.equal(direct[k], cached[k]) for k in direct):
                raise RuntimeError("Trained checkpoint cache/direct mismatch")
    if (
        hash_value(carrier.state_dict()) != carrier_before
        or sha(carrier_path) != metadata["sha256"]
    ):
        raise RuntimeError("Frozen carrier modified")
    parameters = sum(p.numel() for p in model.head.parameters())
    height, width = manifest["image_size"]
    sync = (lambda: torch.cuda.synchronize()) if str(device).startswith("cuda") else lambda: None
    row = train[0]
    timings = {}
    with torch.no_grad():
        for name, call in (
            ("head_only", lambda: cached_forward(model, row["cache"])),
            (
                "fixed",
                lambda: renderer(row["state"], row["camera"], {"rgb", "depth", "visibility"}),
            ),
            ("residual_full", lambda: model(row["state"], row["camera"])),
        ):
            for _ in range(3):
                call()
            sync()
            clock = time.perf_counter()
            for _ in range(20):
                call()
            sync()
            timings[name + "_seconds_per_query"] = (time.perf_counter() - clock) / 20
    recent = curves[-min(3, len(curves)) :]
    improvement = (recent[0]["train_mean_loss"] - recent[-1]["train_mean_loss"]) / max(
        recent[0]["train_mean_loss"], 1e-12
    )
    summary = {
        "status": "TRAINED_FROZEN",
        "selected_step": best_step,
        "validation_loss": best,
        "steps_completed": config.steps,
        "parameters": parameters,
        "head_linear_flops_per_query": 2
        * height
        * width
        * ((carrier.config.feature_dim + 6) * 16 + 16 * 4),
        "flops_definition": "multiply+add=2; excludes SiLU/bias/clamp/renderer",
        "cache_seconds": cache_seconds,
        "optimizer_seconds": optimizer_seconds,
        **timings,
        "training_convergence": "NOT_ESTABLISHED"
        if improvement > 0.01
        else "RECENT_TRAIN_LOSS_PLATEAU",
        "recent_relative_train_loss_improvement": improvement,
        "convergence_note": (
            "fixed budget; plateau is not proof of global optimum or sufficient capacity"
        ),
        "carrier_unchanged": True,
        "cache_exact_initial_and_selected": True,
        "checkpoint_sha256": sha(output / "best.pt"),
        "dynamic_ttt_run": False,
        "zero_state_definition": ZERO_STATE_DEFINITION,
    }
    write_json(output / "cache_equivalence.json", cached_checks)
    write_json(output / "summary.json", summary)
    return summary
