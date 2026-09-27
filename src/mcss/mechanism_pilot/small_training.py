"""Train-only short engineering pilot; does not qualify a scientific carrier."""

from __future__ import annotations

import hashlib
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from mcss.dynamic.carrier import DynamicSceneCarrier
from mcss.dynamic.checkpoint import load_dynamic_checkpoint, save_dynamic_checkpoint
from mcss.dynamic.config import CarrierConfig, WriteConfig
from mcss.dynamic.types import Action, OnlineObservation
from mcss.dynamic.write_rule import DirectWriteRule
from mcss.geometry import transform_cameras
from mcss.measurements import FixedMeasurementRenderer
from mcss.mechanism_pilot.statistics import measurement_metrics
from mcss.mechanism_pilot.training import common_anchor_measurement_loss
from mcss.training.grounded import build_grounded_state, grounded_measurement_loss
from mcss.training.write_unroll import unroll_observed_episode
from mcss.types import Cameras

STATUS = "SHORT_ENGINEERING_TRAINING_NO_SCIENTIFIC_QUALIFICATION"


@dataclass(frozen=True)
class SmallTrainingConfig:
    seed: int = 20260927
    stage_a_steps: int = 100
    stage_b_steps: int = 30
    stage_a_lr: float = 1e-3
    stage_b_lr: float = 3e-4
    renderer_samples: int = 64
    ray_chunk_size: int = 2048
    gradient_clip: float = 1.0


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def schedule(step: int, scene_count: int) -> tuple[int, int, Action]:
    """Every scene cycles its own queries and actions, without scene/action confounding."""
    visit = step // scene_count
    return step % scene_count, visit % 4, (Action.FUSE, Action.COMPLETE, Action.ALL)[visit % 3]


def validate_manifest(manifest: dict, *, engineering_fixture=False) -> None:
    if manifest.get("schema") != "mcss.small_training.data.v1":
        raise ValueError("Unexpected data schema")
    scenes = manifest["scenes"]
    if not engineering_fixture:
        if [s["scene_id"] for s in scenes] != ["ai_001_001", "ai_002_001", "ai_003_001"]:
            raise ValueError("Scenes differ from the preregistered whitelist")
        if manifest.get("image_size") != [128, 160]:
            raise ValueError("Production training requires 128x160 images")
        if manifest.get("depth_semantics") != "ray_distance_meters":
            raise ValueError("Production depth must be metric ray distance")
    if len(scenes) != 3 or len({s["scene_id"] for s in scenes}) != 3:
        raise ValueError("Pilot requires exactly three distinct train scenes")
    for scene in scenes:
        if scene.get("split") != "train":
            raise ValueError("Only train scenes may enter this trainer")
        frames = scene["frames"]
        ids = [f["frame_id"] for f in frames]
        if len(ids) != 16 or ids != sorted(set(ids)):
            raise ValueError("Exactly sixteen ascending unique frames are required")
        expected = {
            "context_a": [ids[i] for i in (0, 1, 2)],
            "context_b": [ids[i] for i in (0, 3, 4)],
            "stream": [ids[i] for i in (5, 6)],
            "query": [ids[i] for i in (12, 13, 14, 15)],
        }
        if any(scene["roles"].get(k) != v for k, v in expected.items()):
            raise ValueError("Frame roles differ from the locked pilot definition")


class AccessLog(list):
    """Durable event log survives training errors and interrupted runs."""

    def __init__(self, path):
        super().__init__()
        self.path = path

    def append(self, value):
        super().append(value)
        with self.path.open("a") as handle:
            handle.write(json.dumps(value) + "\n")


class TrainScene:
    def __init__(self, record, image_size, base: Path, device, access_log):
        self.record, self.image_size, self.base = record, tuple(image_size), base
        self.device, self.access_log = device, access_log
        self.frames = {f["frame_id"]: f for f in record["frames"]}

    def path(self, frame, key):
        path = Path(frame[key])
        return path if path.is_absolute() else self.base / path

    def log(self, frame_id, kind, purpose):
        self.access_log.append(
            {
                "scene_id": self.record["scene_id"],
                "split": "train",
                "frame_id": frame_id,
                "kind": kind,
                "purpose": purpose,
            }
        )

    def camera(self, frame):
        return Cameras(
            torch.tensor(frame["intrinsics"], dtype=torch.float32, device=self.device),
            torch.tensor(frame["c2w"], dtype=torch.float32, device=self.device),
            self.image_size,
        )

    def rgb(self, frame):
        with Image.open(self.path(frame, "rgb")) as im:
            data = np.array(im.convert("RGB"), dtype=np.float32) / 255
        if data.shape != (*self.image_size, 3):
            raise ValueError("Unexpected RGB dimensions")
        return torch.from_numpy(data).permute(2, 0, 1).to(self.device)

    def observations(self, role):
        if role not in ("context_a", "context_b", "stream"):
            raise ValueError("Query cameras/labels cannot construct scene state")
        result = []
        for frame_id in self.record["roles"][role]:
            frame = self.frames[frame_id]
            self.log(frame_id, "rgb_camera", "observed_state_construction")
            result.append(
                OnlineObservation(
                    self.record["scene_id"], frame_id, self.rgb(frame), self.camera(frame)
                )
            )
        return result

    def query(self, slot, purpose):
        frame_id = self.record["roles"]["query"][slot]
        frame = self.frames[frame_id]
        self.log(frame_id, "rgb_camera_depth", purpose)
        camera = self.camera(frame)
        cameras = Cameras(camera.intrinsics[None, None], camera.c2w[None, None], self.image_size)
        depth = torch.from_numpy(np.load(self.path(frame, "depth"))).float().to(self.device)
        depth = depth.squeeze()
        if tuple(depth.shape) != self.image_size:
            raise ValueError("Unexpected ray-distance depth dimensions")
        valid = torch.isfinite(depth) & (depth > 0)
        if not valid.any():
            raise ValueError("Training query has no valid depth")
        depth = torch.where(valid, depth, torch.zeros_like(depth))
        return frame_id, cameras, self.rgb(frame)[None, None], depth[None, None, None]


def evaluate_train(carrier, scenes, renderer, label):
    rows = []
    with torch.no_grad():
        for scene in scenes:
            # State construction finishes before any evaluation query is loaded.
            state, anchor = build_grounded_state(
                carrier, scene.observations("context_a"), f"{scene.record['scene_id']}:train-eval"
            )
            for slot in range(4):
                frame_id, camera, rgb, depth = scene.query(slot, "train_only_post_state_evaluation")
                local = transform_cameras(camera, torch.linalg.inv(anchor))
                prediction = renderer(state, local, measurements=("rgb", "depth", "visibility"))
                loss, terms = grounded_measurement_loss(state, camera, rgb, depth, anchor, renderer)
                rows.append(
                    {
                        "checkpoint": label,
                        "scene_id": scene.record["scene_id"],
                        "query_id": frame_id,
                        "split": "train",
                        "action": "OFF",
                        "loss": float(loss),
                        "terms": {k: float(v) for k, v in terms.items()},
                        **measurement_metrics(
                            prediction["rgb"],
                            rgb,
                            prediction["depth"],
                            depth,
                            prediction["visibility"],
                        ),
                    }
                )
    return rows


def run_small_training(
    manifest_path,
    output_dir,
    *,
    device="cuda",
    config=None,
    engineering_fixture=False,
    carrier_config=None,
):
    config = config or SmallTrainingConfig()
    if min(config.stage_a_steps, config.stage_b_steps) < 1:
        raise ValueError("Both training phases require at least one step")
    manifest_path, output = Path(manifest_path).resolve(), Path(output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Refusing to overwrite a training run")
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(manifest_path.read_text())
    validate_manifest(manifest, engineering_fixture=engineering_fixture)
    if not engineering_fixture:
        data_root = manifest_path.parent
        if sha(data_root / "data_lock.json") != manifest["data_lock_sha256"]:
            raise ValueError("Data lock hash changed")
        expected_hashes = json.loads((data_root / "hashes.json").read_text())
        for relative, expected in expected_hashes.items():
            if sha(data_root / relative) != expected:
                raise ValueError(f"Prepared data hash changed: {relative}")
        for scene in manifest["scenes"]:
            for frame in scene["frames"]:
                for key in ("rgb", "depth"):
                    path = Path(frame[key])
                    path = path if path.is_absolute() else data_root / path
                    relative = str(path.resolve().relative_to(data_root))
                    if relative not in expected_hashes:
                        raise ValueError("Manifest references an unlocked media file")
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    access_log = AccessLog(output / "label_access.jsonl")
    scenes = [
        TrainScene(s, manifest["image_size"], manifest_path.parent, device, access_log)
        for s in manifest["scenes"]
    ]
    carrier = DynamicSceneCarrier(carrier_config or CarrierConfig()).to(device)
    writer = DirectWriteRule(carrier.config, WriteConfig()).to(device)
    renderer = FixedMeasurementRenderer(
        n_samples=config.renderer_samples, ray_chunk_size=config.ray_chunk_size
    ).to(device)
    data_hashes = {str(manifest_path): sha(manifest_path)}
    for scene in scenes:
        for frame in scene.frames.values():
            for key in ("rgb", "depth"):
                path = scene.path(frame, key)
                data_hashes[str(path)] = sha(path)
    source_root = Path(__file__).resolve().parents[1]
    source_hashes = {str(p): sha(p) for p in sorted(source_root.rglob("*.py"))}
    lock = {
        "status": STATUS,
        "config": asdict(config),
        "carrier": asdict(carrier.config),
        "write": asdict(writer.write_config),
        "data_sha256": data_hashes,
        "source_sha256": source_hashes,
        "scene_ids": [s.record["scene_id"] for s in scenes],
        "image_size": manifest["image_size"],
        "test_scene_count": 0,
        "stage_b_schedule": (
            "scene=step%3;query=(step//3)%4;action=[FUSE,COMPLETE,ALL][(step//3)%3]"
        ),
    }
    write_json(output / "lock.json", lock)
    provenance = {
        "status": STATUS,
        "lock_sha256": sha(output / "lock.json"),
        "training_scenes": lock["scene_ids"],
        "training_scene_count": 3,
        "config": asdict(config),
        "query_ids": {s.record["scene_id"]: s.record["roles"]["query"] for s in scenes},
    }
    checkpoint_hashes, restores, evaluations = {}, {}, []
    start = time.monotonic()

    def checkpoint(label):
        path = output / f"{label}.pt"
        checkpoint_hashes[label] = save_dynamic_checkpoint(
            path, carrier, writer, phase=label, provenance=provenance
        )
        restored, restored_writer, _ = load_dynamic_checkpoint(path, device)
        exact = all(
            torch.equal(v, restored.state_dict()[k]) for k, v in carrier.state_dict().items()
        )
        exact &= all(
            torch.equal(v, restored_writer.state_dict()[k]) for k, v in writer.state_dict().items()
        )
        with torch.no_grad():
            context = scenes[0].observations("context_a")
            original, _ = build_grounded_state(carrier, context, "restore-check")
            reloaded, _ = build_grounded_state(restored, context, "restore-check")
            same_state = all(
                torch.equal(getattr(original, k), getattr(reloaded, k))
                for k in ("density_logits", "color", "log_variance", "features")
            )
        if not exact or not same_state:
            raise RuntimeError("Checkpoint restore is not exact")
        with torch.no_grad():
            stream = scenes[0].observations("stream")
            replay_a = unroll_observed_episode(
                carrier, writer, context, stream, (Action.ALL, Action.ALL), "restore-all"
            )
            replay_b = unroll_observed_episode(
                restored, restored_writer, context, stream, (Action.ALL, Action.ALL), "restore-all"
            )
            all_fast_exact = all(
                torch.equal(getattr(replay_a[1], k), getattr(replay_b[1], k))
                for k in ("delta_fuse", "delta_complete")
            )
            all_state_exact = all(
                torch.equal(getattr(replay_a[0], k), getattr(replay_b[0], k))
                for k in ("density_logits", "color", "log_variance", "features")
            )
            all_fast_nonzero = bool(
                replay_a[1].delta_fuse.abs().sum() > 0
                and replay_a[1].delta_complete.abs().sum() > 0
            )
        if not (all_fast_exact and all_state_exact and all_fast_nonzero):
            raise RuntimeError("ALL writer replay failed checkpoint roundtrip")
        restores[label] = {
            "all_two_step_fast_exact": all_fast_exact,
            "all_two_step_state_exact": all_state_exact,
            "all_two_step_fast_nonzero": all_fast_nonzero,
            "state_dict_exact": exact,
            "materialized_state_exact": same_state,
        }
        evaluations.extend(evaluate_train(carrier, scenes, renderer, label))
        write_json(output / "train_evaluation.json", evaluations)
        write_json(output / "checkpoint_restore.json", restores)

    checkpoint("initial")
    with (output / "training.jsonl").open("w", buffering=1) as log:
        for phase, steps, lr in (
            ("phase_a", config.stage_a_steps, config.stage_a_lr),
            ("phase_b", config.stage_b_steps, config.stage_b_lr),
        ):
            parameters = list(carrier.parameters()) + (
                list(writer.parameters()) if phase == "phase_b" else []
            )
            optimizer = torch.optim.Adam(parameters, lr=lr)
            for step in range(steps):
                tick = time.monotonic()
                index, slot, action = schedule(step, len(scenes))
                scene = scenes[index]
                optimizer.zero_grad(set_to_none=True)
                a = scene.observations("context_a")
                if phase == "phase_a":
                    b = scene.observations("context_b")
                    qid, camera, rgb, depth = scene.query(slot, "offline_training_supervision")
                    result = common_anchor_measurement_loss(
                        carrier, a, b, [qid], camera, rgb, depth, renderer, split_id="train"
                    )
                    loss, terms = result.loss, result.terms
                else:
                    state, _, _, anchor = unroll_observed_episode(
                        carrier,
                        writer,
                        a,
                        scene.observations("stream"),
                        (action, action),
                        f"{phase}:{step}",
                    )
                    _, camera, rgb, depth = scene.query(slot, "offline_training_supervision")
                    loss, terms = grounded_measurement_loss(
                        state, camera, rgb, depth, anchor, renderer
                    )
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite training loss")
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(
                    parameters, config.gradient_clip, error_if_nonfinite=True
                )
                writer_gradient = (
                    sum(
                        float(p.grad.square().sum())
                        for p in writer.parameters()
                        if p.grad is not None
                    )
                    ** 0.5
                )
                optimizer.step()
                record = {
                    "phase": phase,
                    "step": step,
                    "scene_id": scene.record["scene_id"],
                    "query_slot": slot,
                    "action": action.value if phase == "phase_b" else "OFF",
                    "loss": float(loss.detach()),
                    "gradient_norm_before_clip": float(norm),
                    "writer_gradient_norm_after_clip": writer_gradient,
                    "seconds": time.monotonic() - tick,
                    "terms": {k: float(v.detach()) for k, v in terms.items()},
                }
                log.write(json.dumps(record, allow_nan=False) + "\n")
                print(json.dumps(record), flush=True)
            torch.save(
                {
                    "phase": phase,
                    "completed_steps": steps,
                    "optimizer": optimizer.state_dict(),
                    "torch_rng": torch.get_rng_state(),
                    "python_rng": random.getstate(),
                    "numpy_rng": np.random.get_state(),
                    "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                    "config": asdict(config),
                },
                output / f"{phase}_training_state.pt",
            )
            checkpoint(f"{phase}_final")
    write_json(output / "label_access.json", access_log)
    summary = {
        "status": STATUS,
        "elapsed_seconds": time.monotonic() - start,
        "checkpoint_sha256": checkpoint_hashes,
        "restore_checks": restores,
        "n_train_scenes": 3,
        "n_dev_scenes": 0,
        "n_test_scenes": 0,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated()
        if str(device).startswith("cuda")
        else None,
        "source_unchanged": all(sha(Path(p)) == h for p, h in source_hashes.items()),
        "data_unchanged": all(sha(Path(p)) == h for p, h in data_hashes.items()),
        "resume_supported": False,
        "controller_trained": False,
    }
    write_json(output / "summary.json", summary)
    return summary
