"""Frozen real-data runner for the label-blind geometry certificate audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as functional
import yaml
from PIL import Image
from torch import Tensor

from mcss.data.manifest import SceneManifest, load_manifest
from mcss.geometry import transform_cameras
from mcss.geometry_certificates import (
    CertificateBuildResult,
    CertificateConfig,
    GeometryCertificate,
    RawObservation,
    build_certificates,
    triangulate_dlt,
)
from mcss.types import Cameras


@dataclass(frozen=True)
class AuditDecisionThresholds:
    minimum_windows: int = 32
    minimum_certified_coverage: float = 0.15
    minimum_nonzero_window_fraction: float = 0.80
    maximum_median_error_m: float = 0.25
    maximum_p90_error_m: float = 0.50
    maximum_permutation_delta_m: float = 1e-4
    minimum_shuffled_error_ratio: float = 1.5
    minimum_shuffled_error_margin_m: float = 0.05
    minimum_risk_error_spearman: float = 0.0

    def __post_init__(self) -> None:
        if self.minimum_windows < 1:
            raise ValueError("minimum_windows must be positive")
        if not 0.0 <= self.minimum_certified_coverage <= 1.0:
            raise ValueError("minimum_certified_coverage must be in [0, 1]")
        if not 0.0 <= self.minimum_nonzero_window_fraction <= 1.0:
            raise ValueError("minimum_nonzero_window_fraction must be in [0, 1]")
        if min(
            self.maximum_median_error_m,
            self.maximum_p90_error_m,
            self.maximum_permutation_delta_m,
            self.minimum_shuffled_error_ratio,
            self.minimum_shuffled_error_margin_m,
        ) < 0:
            raise ValueError("decision thresholds must be nonnegative")
        if not -1.0 <= self.minimum_risk_error_spearman <= 1.0:
            raise ValueError("minimum_risk_error_spearman must be in [-1, 1]")


@dataclass(frozen=True)
class GeometryAuditConfig:
    seed: int
    dataset_root: str
    output_dir: str
    context_views: int
    image_size: tuple[int, int]
    window_count: int
    window_stride: int
    permutation_windows: int
    local_bounds_m: tuple[tuple[float, float, float], tuple[float, float, float]]
    voxel_resolution: tuple[int, int, int]
    certificate: CertificateConfig
    decision: AuditDecisionThresholds

    def __post_init__(self) -> None:
        if self.context_views < 3:
            raise ValueError("context_views must be at least three")
        if min(*self.image_size, self.window_count, self.window_stride) < 1:
            raise ValueError("image/window values must be positive")
        if not 0 <= self.permutation_windows <= self.window_count:
            raise ValueError("permutation_windows must be in [0, window_count]")
        if min(self.voxel_resolution) < 1:
            raise ValueError("voxel_resolution values must be positive")
        lower, upper = self.local_bounds_m
        if any(low >= high for low, high in zip(lower, upper, strict=True)):
            raise ValueError("local_bounds_m must have increasing bounds")


def select_frozen_windows(
    entries: Sequence[tuple[str, int]], *, count: int, seed: int
) -> list[tuple[str, int]]:
    """Select scene-balanced windows by stable hashes, independent of input ordering."""

    if count < 1:
        raise ValueError("count must be positive")
    unique = set(entries)
    if len(unique) < count:
        raise ValueError(f"requested {count} windows but only {len(unique)} are available")
    by_scene: dict[str, list[int]] = {}
    for scene_id, start in unique:
        if not scene_id or start < 0:
            raise ValueError("window entries require non-empty scene IDs and nonnegative starts")
        by_scene.setdefault(scene_id, []).append(start)
    scene_order = sorted(by_scene, key=lambda scene: _stable_hash_int(seed, scene))
    starts = {
        scene: sorted(values, key=lambda start: _stable_hash_int(seed, scene, str(start)))
        for scene, values in by_scene.items()
    }
    selected: list[tuple[str, int]] = []
    round_index = 0
    while len(selected) < count:
        added = False
        for scene in scene_order:
            if round_index < len(starts[scene]):
                selected.append((scene, starts[scene][round_index]))
                added = True
                if len(selected) == count:
                    break
        if not added:
            raise RuntimeError("window selection exhausted entries unexpectedly")
        round_index += 1
    return selected


def decide_audit(
    metrics: dict[str, Any], thresholds: AuditDecisionThresholds
) -> dict[str, Any]:
    """Apply the pre-registered all-gates decision without threshold tuning."""

    required = {
        "window_count",
        "certified_coverage_macro",
        "nonzero_window_fraction",
        "geometry_error_median_m",
        "geometry_error_p90_m",
        "matched_confidence_median_error_m",
        "permutation_max_point_delta_m",
        "shuffled_median_error_m",
        "risk_error_spearman",
    }
    missing = sorted(key for key in required if metrics.get(key) is None)
    if missing or int(metrics.get("window_count", 0)) < thresholds.minimum_windows:
        return {
            "status": "INCONCLUSIVE",
            "gates": {},
            "missing": missing,
            "reason": "required metrics or frozen window count are incomplete",
        }
    certified_error = float(metrics["geometry_error_median_m"])
    shuffled_error = float(metrics["shuffled_median_error_m"])
    gates = {
        "certified_coverage": float(metrics["certified_coverage_macro"])
        >= thresholds.minimum_certified_coverage,
        "nonzero_windows": float(metrics["nonzero_window_fraction"])
        >= thresholds.minimum_nonzero_window_fraction,
        "median_geometry_error": certified_error <= thresholds.maximum_median_error_m,
        "p90_geometry_error": float(metrics["geometry_error_p90_m"])
        <= thresholds.maximum_p90_error_m,
        "beats_confidence_at_matched_coverage": certified_error
        < float(metrics["matched_confidence_median_error_m"]),
        "view_permutation_invariance": float(metrics["permutation_max_point_delta_m"])
        <= thresholds.maximum_permutation_delta_m,
        "shuffled_correspondence_degrades": (
            shuffled_error
            >= certified_error * thresholds.minimum_shuffled_error_ratio
            and shuffled_error
            >= certified_error + thresholds.minimum_shuffled_error_margin_m
        ),
        "risk_predicts_geometry_error": float(metrics["risk_error_spearman"])
        > thresholds.minimum_risk_error_spearman,
    }
    return {
        "status": (
            "GO_TO_STATE_INTEGRATION" if all(gates.values()) else "STOP_CERTIFICATE_FRONTEND"
        ),
        "gates": gates,
        "missing": [],
        "reason": "all frozen gates passed" if all(gates.values()) else "one or more gates failed",
    }


def render_markdown_report(
    decision: dict[str, Any],
    metrics: dict[str, Any],
    *,
    config_path: Path,
    artifact_paths: Sequence[Path],
    started_at: str,
    finished_at: str,
) -> str:
    """Render the Chinese-first terminal audit report, including absent formal roles."""

    status = decision.get("status", "INCONCLUSIVE")
    gates = decision.get("gates", {})
    artifact_lines = "\n".join(f"- `{path}`" for path in artifact_paths) or "- none recorded"
    gate_lines = (
        "\n".join(
            f"- `{name}`: {'PASS' if passed else 'FAIL'}" for name, passed in gates.items()
        )
        or "- none recorded"
    )
    institute_roles = "\n".join(
        f"- {role}: not invoked"
        for role in (
            "research director",
            "framing architect",
            "domain scientist",
            "evidence curator",
            "model scientist",
            "experiment statistician",
            "software executor",
            "independent validator",
            "venue editor",
        )
    )
    return f"""# 几何认证写保护场景状态：零训练审计报告

## 1. Identity

- 项目：Measurement-Complete Scene State
- 阶段：geometry-certificate label-blind audit
- 状态：`{status}`
- 开始：`{started_at}`
- 结束：`{finished_at}`
- 配置：`{config_path}`
- 路由说明：本阶段承接中档超级科研准备，但不是正式 Research Institute 运行。

## 2. Objective And Scope

这次只回答一个问题：不读取 target RGB/depth/normal，也不训练网络时，context RGB 与
标定相机能否产生覆盖足够、几何正确、可追溯的三维证书。V5、V6a、V6a.1、固定
renderer 和主干均未修改。训练及 typed-state 接入由本报告的冻结门槛控制。

## 3. Materials And Provenance

{artifact_lines}

证书构造仅接收 context RGB 和 context cameras。每个窗口的 certificate JSON 写盘之后，
runner 才打开对应 context depth 做事后评分。运行根目录强制为 development `train`；
diagnostic-test 和 final-holdout 未授权、未读取。

## 4. Research Route

V6a.1 已封存，因为局部 RGB 竞争并不定位真实表面。本阶段换成“LoFTR 只提候选，固定
几何决定能否写入”的审计路径。认证顺序为 raw -> reciprocal -> cycle -> epipolar /
cheirality / triangulation -> third-view reprojection。任何一步失败都保留为拒绝计数，
不会交给 completion 掩盖。

## 5. Decision Audit

最终判定：`{status}`。理由：{decision.get('reason', 'not recorded')}。

{gate_lines}

关键汇总：

- 冻结窗口数：`{_format_metric(metrics.get('window_count'))}`
- 认证 source-grid coverage：`{_format_metric(metrics.get('certified_coverage_macro'))}`
- 非零窗口比例：`{_format_metric(metrics.get('nonzero_window_fraction'))}`
- 几何误差 median / p90：`{_format_metric(metrics.get('geometry_error_median_m'))}` m /
  `{_format_metric(metrics.get('geometry_error_p90_m'))}` m
- 同覆盖 LoFTR confidence median：
  `{_format_metric(metrics.get('matched_confidence_median_error_m'))}` m
- shuffled correspondence median：`{_format_metric(metrics.get('shuffled_median_error_m'))}` m
- 视图置换最大点偏差：`{_format_metric(metrics.get('permutation_max_point_delta_m'))}` m
- 风险-误差 Spearman：`{_format_metric(metrics.get('risk_error_spearman'))}`

## 6. Research Institute Record

正式 Institute MCP/route 本阶段不可用，因此以下正式角色均未实例化：

{institute_roles}

先前参与方向准备的 Curie、Boyle、Parfit 仅是降级独立顾问，不得冒充正式 Institute
记录或 PASS。他们共同建议 A+B 合同，并要求先做本次标签盲审计。

## 7. Evaluation Council Record

Council not convened。没有 frozen dossier、七席有效 ballot、独立 rescore 或 clerk gate，
因此本报告不能声称 Council PASS。

## 8. Tools And Collaborators

- 实现：Python 3.11、PyTorch、Kornia LoFTR、pytest、Ruff。
- proposer：冻结的 ScanNet `indoor_new` LoFTR；它只提候选，不是论文创新。
- geometry：fundamental matrix、互惠、三视图循环、DLT、cheirality、重投影。
- 文献补查：arXiv 接口返回 HTTP 429；NeuralRecon/TransformerFusion/TSDF/scene
  completion 的正式等价性复核仍未完成。
- Generation 10 completion-report 仍是设计；当前激活生态是 Generation 9。

## 9. Outputs And Verification

本次新增功能的聚焦测试、静态检查、运行日志及哈希见 artifact 列表。最终人工交付检查
另记录全仓测试与 Ruff 边界。结果状态只代表本阶段的机制审计，不代表训练有效、外部
基线胜出或 CVPR-ready。

## 10. Risks And Open Issues

- LoFTR 先验可能主导候选质量，后续必须做同 proposer 的合同对照。
- context depth 仅用于写盘后的审计；它不能成为模型输入或阈值选择依据。
- 匹配、三角化、free/unknown、TSDF/occupancy 各自均非新意。
- NeuralRecon、TransformerFusion 和 observation-preserving completion 的等价性仍待补查。
- 即使本审计 GO，也只允许实现 write-protected typed state，不允许直接声称 CVPR 水平。

## 11. Memory Disposition

本次结果未写入项目或全局记忆。项目没有 `.codex-memory`，用户也没有要求更新外部
memory。失败证据和审计产物只保存在本项目 `outputs` 与 `docs` 中。

## 12. Next Step

`GO_TO_STATE_INTEGRATION` 只授权下一阶段设计：将 conditional free prefix、surface
interval、behind-surface unknown、conflict 和 provenance 接入独立 typed state，并做
write-protection intervention。`STOP_CERTIFICATE_FRONTEND` 则封存该 frontend，不进入
训练；`INCONCLUSIVE` 只允许补齐缺失运行，不允许调阈值。
"""


def load_audit_config(path: str | Path) -> GeometryAuditConfig:
    config_path = Path(path)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("audit config root must be a mapping")
    allowed = {
        "seed",
        "dataset_root",
        "output_dir",
        "context_views",
        "image_size",
        "window_count",
        "window_stride",
        "permutation_windows",
        "local_bounds_m",
        "voxel_resolution",
        "certificate",
        "decision",
    }
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"unknown audit config keys: {sorted(unknown)}")
    missing = allowed - set(raw)
    if missing:
        raise ValueError(f"missing audit config keys: {sorted(missing)}")
    certificate_raw = dict(raw["certificate"])
    decision_raw = dict(raw["decision"])
    return GeometryAuditConfig(
        seed=int(raw["seed"]),
        dataset_root=str(raw["dataset_root"]),
        output_dir=str(raw["output_dir"]),
        context_views=int(raw["context_views"]),
        image_size=_pair(raw["image_size"], "image_size"),
        window_count=int(raw["window_count"]),
        window_stride=int(raw["window_stride"]),
        permutation_windows=int(raw["permutation_windows"]),
        local_bounds_m=_bounds(raw["local_bounds_m"]),
        voxel_resolution=_triple(raw["voxel_resolution"], "voxel_resolution"),
        certificate=CertificateConfig(**certificate_raw),
        decision=AuditDecisionThresholds(**decision_raw),
    )


def run_audit(config_path: str | Path) -> dict[str, Any]:
    config_file = Path(config_path).resolve()
    config = load_audit_config(config_file)
    output_dir = Path(config.output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite audit output: {output_dir}")
    root = Path(config.dataset_root).resolve()
    if root.name != "train" or any(
        token in str(root).lower() for token in ("diagnostic_test", "final_holdout")
    ):
        raise ValueError("geometry audit is restricted to the prepared development train root")
    manifests = {path.parent.name: path for path in sorted(root.glob("*/manifest.json"))}
    if not manifests:
        raise FileNotFoundError(f"no direct scene manifests found below {root}")
    entries: list[tuple[str, int]] = []
    frame_ids_by_scene: dict[str, list[int]] = {}
    for scene_id, manifest_path in manifests.items():
        manifest = load_manifest(manifest_path)
        frame_ids_by_scene[scene_id] = [frame.frame_id for frame in manifest.frames]
        entries.extend(
            (scene_id, start)
            for start in range(
                0,
                len(manifest.frames) - config.context_views + 1,
                config.window_stride,
            )
        )
    selected = select_frozen_windows(entries, count=config.window_count, seed=config.seed)

    started_at = _now()
    output_dir.mkdir(parents=True)
    certificate_dir = output_dir / "certificates"
    certificate_dir.mkdir()
    frozen_windows = []
    for scene_id, start in selected:
        ids = frame_ids_by_scene[scene_id][start : start + config.context_views]
        frozen_windows.append(
            {
                "scene_id": scene_id,
                "start": start,
                "frame_ids": ids,
                "manifest": str(manifests[scene_id]),
            }
        )
    _atomic_json(
        output_dir / "frozen_windows.json",
        {
            "seed": config.seed,
            "selection_rule": "scene-balanced sha256 ordering",
            "partition": "train",
            "windows": frozen_windows,
        },
    )

    window_metrics: list[dict[str, Any]] = []
    accessed_depth_paths: list[str] = []
    permutation_deltas: list[float] = []
    all_errors: list[float] = []
    all_risks: list[float] = []
    all_confidence_errors: list[tuple[float, float]] = []
    all_shuffled_errors: list[float] = []
    all_camera_shuffled_errors: list[float] = []

    for window_index, window in enumerate(frozen_windows):
        manifest = load_manifest(window["manifest"])
        frames = manifest.frames[window["start"] : window["start"] + config.context_views]
        context_rgb, cameras = _load_context_rgb_and_cameras(manifest, frames, config.image_size)
        build = build_certificates(context_rgb, cameras, config.certificate)
        certificate_path = certificate_dir / f"window_{window_index:03d}.json"
        certificate_written_at = _now()
        _atomic_json(
            certificate_path,
            {
                "schema_version": 1,
                "scene_id": window["scene_id"],
                "start": window["start"],
                "frame_ids": window["frame_ids"],
                "certificate_config": asdict(config.certificate),
                "label_inputs_to_builder": [],
                "written_at": certificate_written_at,
                "build": build.to_dict(),
            },
        )

        depth_loaded_at = _now()
        depths, depth_paths = _load_context_depth(manifest, frames, config.image_size)
        accessed_depth_paths.extend(str(path) for path in depth_paths)
        evaluation = _evaluate_window(build, depths, cameras, config)
        if depth_loaded_at < certificate_written_at:
            raise RuntimeError("depth access timestamp preceded certificate serialization")
        evaluation.update(
            {
                "window_index": window_index,
                "scene_id": window["scene_id"],
                "start": window["start"],
                "certificate_path": str(certificate_path),
                "certificate_sha256": _sha256(certificate_path),
                "certificate_written_at": certificate_written_at,
                "depth_loaded_at": depth_loaded_at,
                "stage_counts": build.stage_counts,
                "certified_source_coverage": build.certified_source_coverage,
            }
        )
        window_metrics.append(evaluation)
        all_errors.extend(evaluation.pop("geometry_errors_m"))
        all_risks.extend(evaluation.pop("risks"))
        all_confidence_errors.extend(
            (float(item[0]), float(item[1]))
            for item in evaluation.pop("raw_confidence_and_errors_m")
        )
        all_shuffled_errors.extend(evaluation.pop("shuffled_errors_m"))
        all_camera_shuffled_errors.extend(evaluation.pop("camera_shuffled_errors_m"))

        if window_index < config.permutation_windows:
            order = torch.tensor([2, 0, 3, 1], dtype=torch.long)[: config.context_views]
            permuted = build_certificates(
                context_rgb[order], cameras.select_views(order), config.certificate
            )
            permutation_deltas.append(_point_set_delta(build, permuted))

        _append_jsonl(output_dir / "window_metrics.jsonl", evaluation)

    confidence_errors = _matched_confidence_errors(
        all_confidence_errors, target_count=len(all_errors)
    )
    metrics = _aggregate_metrics(
        window_metrics,
        geometry_errors=all_errors,
        risks=all_risks,
        confidence_errors=confidence_errors,
        shuffled_errors=all_shuffled_errors,
        camera_shuffled_errors=all_camera_shuffled_errors,
        permutation_deltas=permutation_deltas,
    )
    decision = decide_audit(metrics, config.decision)
    finished_at = _now()
    result = {
        "schema_version": 1,
        "status": decision["status"],
        "started_at": started_at,
        "finished_at": finished_at,
        "config_path": str(config_file),
        "config_sha256": _sha256(config_file),
        "dataset_root": str(root),
        "partition": "train",
        "final_or_diagnostic_access": False,
        "certificate_before_depth": True,
        "metrics": metrics,
        "decision": decision,
        "certificate_config": asdict(config.certificate),
        "decision_thresholds": asdict(config.decision),
        "weight_artifacts": _weight_artifacts(Path(config.certificate.model_cache_dir)),
    }
    _atomic_json(output_dir / "audit_result.json", result)
    _atomic_json(
        output_dir / "access_log.json",
        {
            "dataset_root": str(root),
            "partition": "train",
            "depth_files_opened_after_certificate_write": sorted(set(accessed_depth_paths)),
            "diagnostic_or_final_access": False,
        },
    )
    artifacts = [
        output_dir / "audit_result.json",
        output_dir / "frozen_windows.json",
        output_dir / "window_metrics.jsonl",
        output_dir / "access_log.json",
        certificate_dir,
    ]
    report = render_markdown_report(
        decision,
        metrics,
        config_path=config_file,
        artifact_paths=artifacts,
        started_at=started_at,
        finished_at=finished_at,
    )
    (output_dir / "REPORT.md").write_text(report, encoding="utf-8")
    _atomic_json(
        output_dir / "artifact_hashes.json",
        {
            str(path.relative_to(output_dir)): _sha256(path)
            for path in output_dir.rglob("*")
            if path.is_file() and path.name != "artifact_hashes.json"
        },
    )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="run_geometry_certificate_audit")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--window-count", type=int)
    args = parser.parse_args(argv)
    config_path = args.config.resolve()
    if args.output_dir is None and args.window_count is None:
        result = run_audit(config_path)
    else:
        config = load_audit_config(config_path)
        config = replace(
            config,
            output_dir=str(args.output_dir.resolve()) if args.output_dir else config.output_dir,
            window_count=args.window_count or config.window_count,
            permutation_windows=min(
                config.permutation_windows, args.window_count or config.window_count
            ),
        )
        temporary_config = Path(config.output_dir).resolve().with_suffix(".resolved.yaml")
        if temporary_config.exists():
            raise FileExistsError(f"refusing to overwrite resolved config: {temporary_config}")
        temporary_config.parent.mkdir(parents=True, exist_ok=True)
        temporary_config.write_text(
            yaml.safe_dump(_config_to_dict(config), sort_keys=False), encoding="utf-8"
        )
        result = run_audit(temporary_config)
    print(json.dumps({"status": result["status"], "metrics": result["metrics"]}, sort_keys=True))
    return 0


def _load_context_rgb_and_cameras(
    manifest: SceneManifest, frames: Sequence[Any], image_size: tuple[int, int]
) -> tuple[Tensor, Cameras]:
    if manifest.root is None:
        raise ValueError("manifest root is required")
    original_height, original_width = manifest.image_size
    target_height, target_width = image_size
    rgb = []
    intrinsics = []
    poses = []
    for frame in frames:
        image = Image.open(manifest.root / frame.rgb).convert("RGB")
        if (image.height, image.width) != image_size:
            image = image.resize((target_width, target_height), Image.Resampling.BICUBIC)
        rgb.append(
            torch.from_numpy(np.asarray(image, dtype=np.float32).copy())
            .permute(2, 0, 1)
            .div(255.0)
        )
        matrix = torch.tensor(frame.intrinsics, dtype=torch.float32)
        matrix[0, 0] *= target_width / original_width
        matrix[0, 2] *= target_width / original_width
        matrix[1, 1] *= target_height / original_height
        matrix[1, 2] *= target_height / original_height
        intrinsics.append(matrix)
        poses.append(torch.tensor(frame.c2w, dtype=torch.float32))
    cameras = Cameras(torch.stack(intrinsics), torch.stack(poses), image_size)
    world_to_anchor = torch.linalg.inv(cameras.c2w[0])
    return torch.stack(rgb), transform_cameras(cameras, world_to_anchor)


def _load_context_depth(
    manifest: SceneManifest, frames: Sequence[Any], image_size: tuple[int, int]
) -> tuple[Tensor, list[Path]]:
    if manifest.root is None:
        raise ValueError("manifest root is required")
    values = []
    paths = []
    for frame in frames:
        if frame.depth is None:
            raise ValueError("post-hoc certificate audit requires context depth labels")
        path = (manifest.root / frame.depth).resolve()
        values.append(torch.from_numpy(np.load(path).astype(np.float32)))
        paths.append(path)
    depth = torch.stack(values)[:, None]
    if depth.shape[-2:] != image_size:
        depth = functional.interpolate(depth, size=image_size, mode="nearest")
    return depth[:, 0], paths


def _evaluate_window(
    build: CertificateBuildResult,
    depths: Tensor,
    cameras: Cameras,
    config: GeometryAuditConfig,
) -> dict[str, Any]:
    errors = []
    risks = []
    for certificate in build.certificates:
        error = _certificate_error(certificate, depths, cameras)
        if error is not None:
            errors.append(error)
            risks.append(_certificate_risk(certificate, config.certificate))
    raw_confidence_errors = []
    for observation in build.raw_observations:
        error = _raw_observation_error(observation, depths, cameras)
        if error is not None:
            raw_confidence_errors.append((observation.proposal_confidence, error))
    shuffled_errors = _shuffled_errors(build.certificates, depths, cameras, config)
    shuffled_cameras = Cameras(
        cameras.intrinsics,
        cameras.c2w.roll(shifts=1, dims=0),
        cameras.image_size,
    )
    camera_shuffled_errors = []
    penalty = _bounds_diagonal(config.local_bounds_m)
    for certificate in build.certificates:
        result = triangulate_dlt(
            torch.tensor(certificate.source_xy),
            torch.tensor(certificate.matched_xy),
            shuffled_cameras,
            certificate.source_view,
            certificate.matched_view,
        )
        if not result.cheirality:
            camera_shuffled_errors.append(penalty)
            continue
        error = _point_error_against_views(
            result.point,
            (
                (certificate.source_view, certificate.source_xy),
                (certificate.matched_view, certificate.matched_xy),
            ),
            depths,
            cameras,
        )
        camera_shuffled_errors.append(penalty if error is None else error)
    voxel_coverage = _voxel_coverage(
        build.certificates, config.local_bounds_m, config.voxel_resolution
    )
    return {
        "certificate_count": len(build.certificates),
        "evaluated_certificate_count": len(errors),
        "geometry_errors_m": errors,
        "risks": risks,
        "raw_confidence_and_errors_m": raw_confidence_errors,
        "shuffled_errors_m": shuffled_errors,
        "camera_shuffled_errors_m": camera_shuffled_errors,
        "geometry_error_median_m": _quantile(errors, 0.5),
        "geometry_error_p90_m": _quantile(errors, 0.9),
        "native_voxel_coverage": voxel_coverage,
    }


def _certificate_error(
    certificate: GeometryCertificate, depths: Tensor, cameras: Cameras
) -> float | None:
    point = torch.tensor(certificate.point, dtype=torch.float32)
    return _point_error_against_views(
        point,
        (
            (certificate.source_view, certificate.source_xy),
            (certificate.matched_view, certificate.matched_xy),
            (certificate.verification_view, certificate.verification_xy),
        ),
        depths,
        cameras,
    )


def _raw_observation_error(
    observation: RawObservation, depths: Tensor, cameras: Cameras
) -> float | None:
    if observation.point is None or not observation.cheirality:
        return None
    return _point_error_against_views(
        torch.tensor(observation.point),
        (
            (observation.source_view, observation.source_xy),
            (observation.matched_view, observation.matched_xy),
        ),
        depths,
        cameras,
    )


def _point_error_against_views(
    point: Tensor,
    observations: Sequence[tuple[int, tuple[float, float]]],
    depths: Tensor,
    cameras: Cameras,
) -> float | None:
    errors = []
    for view, xy in observations:
        truth = _depth_point(depths[view], xy, cameras, view)
        if truth is None:
            return None
        errors.append(float(torch.linalg.vector_norm(point - truth).item()))
    return max(errors)


def _depth_point(
    depth: Tensor, xy: tuple[float, float], cameras: Cameras, view: int
) -> Tensor | None:
    x = min(depth.shape[1] - 1, max(0, int(round(xy[0]))))
    y = min(depth.shape[0] - 1, max(0, int(round(xy[1]))))
    distance = depth[y, x]
    if not torch.isfinite(distance) or distance <= 0:
        return None
    pixel = torch.tensor([xy[0], xy[1], 1.0], dtype=cameras.dtype)
    direction = torch.linalg.inv(cameras.intrinsics[view]) @ pixel
    direction = direction / torch.linalg.vector_norm(direction).clamp_min(1e-8)
    world_direction = cameras.c2w[view, :3, :3] @ direction
    return cameras.c2w[view, :3, 3] + world_direction * distance


def _shuffled_errors(
    certificates: Sequence[GeometryCertificate],
    depths: Tensor,
    cameras: Cameras,
    config: GeometryAuditConfig,
) -> list[float]:
    by_pair: dict[tuple[int, int], list[GeometryCertificate]] = {}
    for certificate in certificates:
        by_pair.setdefault((certificate.source_view, certificate.matched_view), []).append(
            certificate
        )
    penalty = _bounds_diagonal(config.local_bounds_m)
    errors = []
    for (source_view, matched_view), values in by_pair.items():
        if len(values) < 2:
            continue
        ordered = sorted(values, key=lambda item: (item.source_xy[1], item.source_xy[0]))
        shifted = ordered[1:] + ordered[:1]
        for source, wrong_match in zip(ordered, shifted, strict=True):
            result = triangulate_dlt(
                torch.tensor(source.source_xy),
                torch.tensor(wrong_match.matched_xy),
                cameras,
                source_view,
                matched_view,
            )
            if not result.cheirality:
                errors.append(penalty)
                continue
            error = _point_error_against_views(
                result.point,
                (
                    (source_view, source.source_xy),
                    (matched_view, wrong_match.matched_xy),
                ),
                depths,
                cameras,
            )
            errors.append(penalty if error is None else error)
    return errors


def _certificate_risk(certificate: GeometryCertificate, config: CertificateConfig) -> float:
    components = (
        1.0 - certificate.proposal_confidence,
        certificate.reciprocal_error_px / config.reciprocal_threshold_px,
        certificate.cycle_error_px / config.cycle_threshold_px,
        certificate.epipolar_error_px / config.epipolar_threshold_px,
        certificate.reprojection_error_px / config.reprojection_threshold_px,
        certificate.verification_error_px / config.verification_threshold_px,
        config.min_triangulation_angle_deg
        / max(certificate.triangulation_angle_deg, config.min_triangulation_angle_deg),
    )
    return float(sum(min(1.0, max(0.0, value)) for value in components) / len(components))


def _matched_confidence_errors(
    confidence_errors: Sequence[tuple[float, float]], *, target_count: int
) -> list[float]:
    if target_count <= 0:
        return []
    ranked = sorted(confidence_errors, key=lambda item: (-item[0], item[1]))
    return [error for _, error in ranked[: min(target_count, len(ranked))]]


def _aggregate_metrics(
    windows: Sequence[dict[str, Any]],
    *,
    geometry_errors: Sequence[float],
    risks: Sequence[float],
    confidence_errors: Sequence[float],
    shuffled_errors: Sequence[float],
    camera_shuffled_errors: Sequence[float],
    permutation_deltas: Sequence[float],
) -> dict[str, Any]:
    stage_names = ("raw", "reciprocal", "cycle", "geometric", "certified")
    stage_totals = {
        name: sum(int(window["stage_counts"][name]) for window in windows) for name in stage_names
    }
    return {
        "window_count": len(windows),
        "scene_count": len({window["scene_id"] for window in windows}),
        "stage_totals": stage_totals,
        "stage_retention": {
            name: stage_totals[name] / stage_totals["raw"] if stage_totals["raw"] else 0.0
            for name in stage_names
        },
        "certified_coverage_macro": _mean(
            [float(window["certified_source_coverage"]) for window in windows]
        ),
        "nonzero_window_fraction": _mean(
            [float(window["certificate_count"] > 0) for window in windows]
        ),
        "geometry_evaluated_count": len(geometry_errors),
        "geometry_error_median_m": _quantile(geometry_errors, 0.5),
        "geometry_error_p90_m": _quantile(geometry_errors, 0.9),
        "matched_confidence_count": len(confidence_errors),
        "matched_confidence_median_error_m": _quantile(confidence_errors, 0.5),
        "shuffled_count": len(shuffled_errors),
        "shuffled_median_error_m": _quantile(shuffled_errors, 0.5),
        "camera_shuffled_median_error_m": _quantile(camera_shuffled_errors, 0.5),
        "permutation_window_count": len(permutation_deltas),
        "permutation_max_point_delta_m": max(permutation_deltas) if permutation_deltas else None,
        "native_voxel_coverage_macro": _mean(
            [float(window["native_voxel_coverage"]) for window in windows]
        ),
        "risk_error_spearman": _spearman(risks, geometry_errors),
        "risk_bin_median_errors_m": _risk_bins(risks, geometry_errors),
    }


def _point_set_delta(first: CertificateBuildResult, second: CertificateBuildResult) -> float:
    if not first.certificates and not second.certificates:
        return 0.0
    if not first.certificates or not second.certificates:
        return math.inf
    points_first = torch.tensor([item.point for item in first.certificates])
    points_second = torch.tensor([item.point for item in second.certificates])
    maximum = 0.0
    for source, target in ((points_first, points_second), (points_second, points_first)):
        for start in range(0, len(source), 1024):
            nearest = torch.cdist(source[start : start + 1024], target).amin(dim=1)
            maximum = max(maximum, float(nearest.max().item()))
    return maximum


def _voxel_coverage(
    certificates: Sequence[GeometryCertificate],
    bounds: tuple[tuple[float, float, float], tuple[float, float, float]],
    resolution: tuple[int, int, int],
) -> float:
    lower = np.asarray(bounds[0], dtype=np.float64)
    upper = np.asarray(bounds[1], dtype=np.float64)
    depth, height, width = resolution
    occupied = set()
    for certificate in certificates:
        point = np.asarray(certificate.point)
        fraction = (point - lower) / (upper - lower)
        if np.all((fraction >= 0) & (fraction < 1)):
            x = min(width - 1, int(fraction[0] * width))
            y = min(height - 1, int(fraction[1] * height))
            z = min(depth - 1, int(fraction[2] * depth))
            occupied.add((z, y, x))
    return len(occupied) / (depth * height * width)


def _risk_bins(risks: Sequence[float], errors: Sequence[float], bins: int = 4) -> list[float]:
    if len(risks) != len(errors) or not risks:
        return []
    order = np.argsort(np.asarray(risks), kind="stable")
    groups = np.array_split(order, bins)
    return [float(np.median(np.asarray(errors)[group])) for group in groups if len(group)]


def _spearman(first: Sequence[float], second: Sequence[float]) -> float | None:
    if len(first) != len(second) or len(first) < 2:
        return None
    rank_first = np.argsort(np.argsort(np.asarray(first), kind="stable"), kind="stable")
    rank_second = np.argsort(np.argsort(np.asarray(second), kind="stable"), kind="stable")
    correlation = np.corrcoef(rank_first, rank_second)[0, 1]
    return float(correlation) if np.isfinite(correlation) else None


def _weight_artifacts(root: Path) -> list[dict[str, str]]:
    if not root.exists():
        return []
    return [
        {"path": str(path.resolve()), "sha256": _sha256(path)}
        for path in sorted(root.rglob("*.ckpt"))
    ]


def _config_to_dict(config: GeometryAuditConfig) -> dict[str, Any]:
    return {
        "seed": config.seed,
        "dataset_root": config.dataset_root,
        "output_dir": config.output_dir,
        "context_views": config.context_views,
        "image_size": list(config.image_size),
        "window_count": config.window_count,
        "window_stride": config.window_stride,
        "permutation_windows": config.permutation_windows,
        "local_bounds_m": [list(row) for row in config.local_bounds_m],
        "voxel_resolution": list(config.voxel_resolution),
        "certificate": asdict(config.certificate),
        "decision": asdict(config.decision),
    }


def _stable_hash_int(*values: object) -> int:
    payload = "|".join(str(value) for value in values).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest(), "big")


def _pair(value: Any, name: str) -> tuple[int, int]:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"{name} must contain two integers")
    return int(value[0]), int(value[1])


def _triple(value: Any, name: str) -> tuple[int, int, int]:
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f"{name} must contain three integers")
    return int(value[0]), int(value[1]), int(value[2])


def _bounds(value: Any) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError("local_bounds_m must contain lower and upper vectors")
    return tuple(tuple(float(item) for item in row) for row in value)  # type: ignore[return-value]


def _bounds_diagonal(
    bounds: tuple[tuple[float, float, float], tuple[float, float, float]]
) -> float:
    return float(np.linalg.norm(np.asarray(bounds[1]) - np.asarray(bounds[0])))


def _quantile(values: Sequence[float], q: float) -> float | None:
    return float(np.quantile(values, q)) if values else None


def _mean(values: Sequence[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _format_metric(value: Any) -> str:
    if value is None:
        return "not available"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="microseconds")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _append_jsonl(path: Path, payload: Any) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, allow_nan=False) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
