"""Predeclared metrics and scene-level diagnostics for the 3D mechanism pilot.

History rows contain scene_id, continuation_id, history (FC/CF), action,
depth_absrel and optional query_id. Loss is averaged over queries, continuations,
histories, then scenes, in that order. Bootstrap samples scenes with replacement;
all histories, actions and queries of a sampled scene remain paired.
"""
from __future__ import annotations

from collections import defaultdict
from itertools import combinations
from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter

ACTIONS = ("OFF", "FUSE", "COMPLETE", "ALL")
HISTORIES = ("FC", "CF")


def _array(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float64)


def _rgb(value: Any) -> np.ndarray:
    value = _array(value).squeeze()
    if value.ndim != 3:
        raise ValueError("RGB must have three dimensions after singleton removal")
    if value.shape[-1] != 3 and value.shape[0] == 3:
        value = np.moveaxis(value, 0, -1)
    if value.shape[-1] != 3 or not np.isfinite(value).all():
        raise ValueError("RGB must be finite HWC or CHW")
    if np.any((value < 0) | (value > 1)):
        raise ValueError("RGB must be in [0, 1]")
    return value


def measurement_metrics(
    pred_rgb: Any, target_rgb: Any, pred_depth: Any, target_depth: Any, opacity: Any,
    *, region_mask: Any | None = None,
) -> dict[str, Any]:
    """GT-valid depth evaluation; no selection by predicted opacity or depth.

    SSIM uses Gaussian sigma 1.5, population covariance, reflect boundaries,
    dynamic range 1, constants .01/.03, and channel/spatial mean. This explicit
    full-image definition also works on tiny smoke images; it is not LPIPS.
    Nonfinite predictions on GT-valid pixels are rejected, rather than dropped.
    region_mask must be independently specified (e.g. true common visibility).
    """
    pred, truth = _rgb(pred_rgb), _rgb(target_rgb)
    depth, gt, alpha = (_array(x).squeeze() for x in (pred_depth, target_depth, opacity))
    if pred.shape != truth.shape or depth.shape != pred.shape[:2]:
        raise ValueError("RGB and depth shapes disagree")
    if gt.shape != depth.shape or alpha.shape != depth.shape:
        raise ValueError("Depth and opacity shapes disagree")
    region = np.ones(depth.shape, dtype=bool) if region_mask is None else np.asarray(
        region_mask, dtype=bool
    ).squeeze()
    if region.shape != depth.shape or not region.any():
        raise ValueError("Region must match image and contain pixels")
    if not np.isfinite(alpha).all() or np.any((alpha < 0) | (alpha > 1 + 1e-6)):
        raise ValueError("Opacity must be finite and in [0,1]")
    valid = region & np.isfinite(gt) & (gt > 0)
    if np.any(valid & ~np.isfinite(depth)):
        raise ValueError("Nonfinite prediction on GT-valid pixels")
    mse = float(np.mean((pred[region] - truth[region]) ** 2))
    means = [gaussian_filter(x, sigma=(1.5, 1.5, 0), mode="reflect") for x in (pred, truth)]
    mu_x, mu_y = means
    vx = gaussian_filter(pred**2, sigma=(1.5, 1.5, 0), mode="reflect") - mu_x**2
    vy = gaussian_filter(truth**2, sigma=(1.5, 1.5, 0), mode="reflect") - mu_y**2
    covariance = gaussian_filter(pred * truth, sigma=(1.5, 1.5, 0), mode="reflect")
    covariance -= mu_x * mu_y
    ssim = ((2 * mu_x * mu_y + .01**2) * (2 * covariance + .03**2)) / (
        (mu_x**2 + mu_y**2 + .01**2) * (vx + vy + .03**2)
    )
    result = {
        "rgb_mse": mse,
        "rgb_psnr": float(-10 * np.log10(mse)) if mse else None,
        "rgb_psnr_perfect": mse == 0,
        "rgb_ssim": float(ssim[region].mean()),
        "rgb_ssim_definition": "gaussian_sigma1.5_population_reflect_full_image_range1",
        "lpips": None,
        "lpips_status": "NOT_IMPLEMENTED",
        "opacity": float(alpha[region].mean()),
        "valid_depth_fraction": float(valid.sum() / region.sum()),
        "predicted_positive_depth_fraction": float(np.mean(
            np.isfinite(depth[region]) & (depth[region] > 0)
        )),
        "coverage": float(np.mean(alpha[region] > 1e-6)),
        "coverage_definition": "predicted_opacity_gt_1e-6_not_true_visibility",
        "depth_valid_count": int(valid.sum()),
        "region_pixel_count": int(region.sum()),
        "depth_absrel": None,
        "depth_rmse": None,
        "depth_delta1": None,
    }
    if valid.any():
        p, t = depth[valid], gt[valid]
        result["depth_absrel"] = float(np.mean(np.abs(p - t) / t))
        result["depth_rmse"] = float(np.sqrt(np.mean((p - t)**2)))
        ratio = np.full_like(p, np.inf)
        positive = p > 0
        ratio[positive] = np.maximum(p[positive] / t[positive], t[positive] / p[positive])
        result["depth_delta1"] = float(np.mean(ratio < 1.25))
    return result


def scene_macro(rows: list[dict], metric: str) -> dict:
    """One row per query; average queries within scene before averaging scenes."""
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        value = float(row[metric])
        if not np.isfinite(value):
            raise ValueError("Metric must be finite")
        grouped[str(row["scene_id"])].append(value)
    if not grouped:
        raise ValueError("No scenes")
    per_scene = {key: float(np.mean(values)) for key, values in sorted(grouped.items())}
    return {"mean": float(np.mean(list(per_scene.values()))), "per_scene": per_scene}


def paired_scene_bootstrap(
    values_by_scene: dict[str, float], *, draws: int = 10000, seed: int = 20260927,
) -> dict:
    if not values_by_scene or draws < 1:
        raise ValueError("Bootstrap requires scenes and positive draws")
    keys = sorted(values_by_scene)
    values = np.array([values_by_scene[key] for key in keys], dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite scene statistics")
    indices = np.random.default_rng(seed).integers(0, len(keys), size=(draws, len(keys)))
    samples = values[indices].mean(axis=1)
    return {
        "mean": float(values.mean()), "ci95": np.quantile(samples, [.025, .975]).tolist(),
        "n_scenes": len(keys), "draws": draws, "seed": seed, "unit": "scene",
        "per_scene": dict(zip(keys, values.tolist(), strict=True)),
        "loso": {key: float(np.delete(values, i).mean()) if len(keys) > 1 else None
                 for i, key in enumerate(keys)},
    }


def _validated_units(rows: list[dict]) -> list[tuple[str, str, np.ndarray]]:
    groups: dict[tuple[str, str], dict] = defaultdict(dict)
    for row in rows:
        key = (str(row["scene_id"]), str(row["continuation_id"]))
        history, action = row["history"], row["action"]
        if history not in HISTORIES or action not in ACTIONS:
            raise ValueError("Unknown history/action")
        entry = (history, action, str(row.get("query_id", "__query_mean__")))
        if entry in groups[key]:
            raise ValueError("Duplicate history/action/query")
        value = float(row["depth_absrel"])
        if not np.isfinite(value) or value < 0:
            raise ValueError("depth_absrel must be finite and nonnegative")
        groups[key][entry] = value
    if not groups:
        raise ValueError("No history rows")
    units = []
    for (scene, continuation), group in sorted(groups.items()):
        queries = {key[2] for key in group}
        expected = {(h, a, q) for h in HISTORIES for a in ACTIONS for q in queries}
        if set(group) != expected:
            raise ValueError("Incomplete or unmatched histories/actions/query sets")
        if "__query_mean__" in queries and len(queries) > 1:
            raise ValueError("Cannot mix pooled and per-query rows")
        losses = np.array([[np.mean([group[h, a, q] for q in sorted(queries)])
                            for a in ACTIONS] for h in HISTORIES])
        units.append((scene, continuation, losses[:, :1] - losses))
    return units


def analyze_history(
    rows: list[dict], *, draws: int = 10000, seed: int = 20260927,
    tie_tolerance: float = 1e-8,
) -> dict:
    """Offline oracle diagnostics. Winners are reoptimized in every resample.

    A beneficial flip requires disjoint tolerance-optimal action sets and at
    least one reward greater than tolerance. OFF participates in every argmax.
    Global and observation-only comparators are hindsight diagnostics, not
    deployable policies. This function never assigns scientific support status.
    """
    if tie_tolerance < 0 or not np.isfinite(tie_tolerance) or draws < 1:
        raise ValueError("Invalid tolerance/draw count")
    units = _validated_units(rows)
    grouped: dict[str, list] = defaultdict(list)
    details = []
    pairs = list(combinations(range(len(ACTIONS)), 2))
    for scene, continuation, gains in units:
        maxima = gains.max(axis=1)
        winners = [set(np.flatnonzero(maxima[i] - gains[i] <= tie_tolerance))
                   for i in range(2)]
        flip = winners[0].isdisjoint(winners[1]) and bool(maxima.max() > tie_tolerance)
        gamma = np.array([(gains[0, a] - gains[0, b]) - (gains[1, a] - gains[1, b])
                          for a, b in pairs])
        record = {
            "scene_id": scene, "continuation_id": continuation,
            "rewards": {h: dict(zip(ACTIONS, gains[i].tolist(), strict=True))
                        for i, h in enumerate(HISTORIES)},
            "optimal_action_sets": {h: [ACTIONS[j] for j in sorted(winners[i])]
                                    for i, h in enumerate(HISTORIES)},
            "beneficial_action_flip": flip,
            "gamma": {f"{ACTIONS[a]}__{ACTIONS[b]}": float(gamma[i])
                      for i, (a, b) in enumerate(pairs)},
        }
        details.append(record)
        grouped[scene].append({
            "actions": gains.mean(axis=0), "oracle": maxima.mean(),
            "observation_only": gains.mean(axis=0).max(),
            "harmful": (gains[:, 1:] < -tie_tolerance).mean(axis=0),
            "flip": float(flip), "gamma": gamma,
        })
    scenes = sorted(grouped)
    arrays = {name: np.stack([np.mean([x[name] for x in grouped[s]], axis=0)
                             for s in scenes])
              for name in ("actions", "oracle", "observation_only", "harmful", "flip", "gamma")}

    def summarize(indices: np.ndarray) -> dict:
        action_means = arrays["actions"][indices].mean(axis=0)
        oracle = float(arrays["oracle"][indices].mean())
        observation = float(arrays["observation_only"][indices].mean())
        best = float(action_means.max())
        return {
            "write_oracle_gain": oracle, "hindsight_global_action_gain": best,
            "global_action_gap": oracle - best,
            "observation_only_oracle_gain": observation,
            "state_dependent_action_gap": oracle - observation,
            "best_fixed_action": ACTIONS[int(np.argmax(action_means))],
            "action_means": dict(zip(ACTIONS, action_means.tolist(), strict=True)),
        }

    overall = summarize(np.arange(len(scenes)))
    indices = np.random.default_rng(seed).integers(0, len(scenes), (draws, len(scenes)))
    # Recompute the global maximization from each sampled set, including repeated scenes.
    boot_oracle = arrays["oracle"][indices].mean(axis=1)
    boot_global = arrays["actions"][indices].mean(axis=1).max(axis=1)
    boot_observation = arrays["observation_only"][indices].mean(axis=1)
    bootstrap = {}
    for name, values in {
        "write_oracle_gain": boot_oracle,
        "hindsight_global_action_gain": boot_global,
        "global_action_gap": boot_oracle - boot_global,
        "observation_only_oracle_gain": boot_observation,
        "state_dependent_action_gap": boot_oracle - boot_observation,
    }.items():
        bootstrap[name] = {"mean": overall[name],
                           "ci95": np.quantile(values, [.025, .975]).tolist()}
    return {
        "schema": "mcss.mechanism.history_statistics.v1", "n_scenes": len(scenes),
        "n_history_pairs": len(units), "actions": list(ACTIONS),
        "aggregation": "queries -> histories/continuations -> equal-weight scenes",
        "tie_tolerance": tie_tolerance,
        "beneficial_flip_definition": (
            "disjoint tolerance-optimal sets; positive winner > tolerance"
        ),
        "oracle_comparators_are_offline_only": True,
        **overall,
        "bootstrap": {"unit": "scene", "draws": draws, "seed": seed,
                      "global_action_reoptimized": True, "metrics": bootstrap},
        "loso": {s: summarize(np.delete(np.arange(len(scenes)), i))
                 if len(scenes) > 1 else None for i, s in enumerate(scenes)},
        "per_scene": {s: summarize(np.array([i])) for i, s in enumerate(scenes)},
        "gamma": {f"{ACTIONS[a]}__{ACTIONS[b]}": paired_scene_bootstrap(
            dict(zip(scenes, arrays["gamma"][:, j].tolist(), strict=True)),
            draws=draws, seed=seed) for j, (a, b) in enumerate(pairs)},
        "harmful_write_prevalence": {
            a: paired_scene_bootstrap(
                dict(zip(scenes, arrays["harmful"][:, j].tolist(), strict=True)),
                draws=draws, seed=seed) for j, a in enumerate(ACTIONS[1:])},
        "beneficial_action_flip_count": sum(x["beneficial_action_flip"] for x in details),
        "scenes_with_beneficial_flip": len({x["scene_id"] for x in details
                                           if x["beneficial_action_flip"]}),
        "beneficial_flip_rate": paired_scene_bootstrap(
            dict(zip(scenes, arrays["flip"].tolist(), strict=True)), draws=draws, seed=seed),
        "per_continuation": details,
    }
