"""Mask-free visible signals and discovery-only, locked opportunity selectors.

Matching confidence is computed over every token, including background. The
selection RGB partition is observed data, not independent evaluation data.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

from mcss.vision_probe.adaptation import ACTIONS, FastState, Readout, materialize

SCHEMA = "mcss.opportunity_selectors.v1"
ALPHAS = (0.1, 1.0, 10.0, 100.0)
THRESHOLDS = (0.0, 0.001, 0.005, 0.01, 0.02)
RGB_THRESHOLDS = (0.0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0)


def _coords(h, w, device):
    y, x = torch.meshgrid(
        torch.arange(h, device=device), torch.arange(w, device=device), indexing="ij"
    )
    return torch.stack((x, y), -1).reshape(-1, 2).float()


@torch.no_grad()
def visible_features(
    source: Tensor,
    target: Tensor,
    adapted: Tensor,
    rgb: Tensor,
    readout: Readout,
    state: FastState,
    z: Tensor,
) -> dict[str, float]:
    """Extract observed-only signals; no geometry, masks, or task labels enter."""
    for name, value in (("source", source), ("target", target), ("adapted", adapted)):
        if value.ndim != 3 or min(value.shape) <= 0 or not torch.isfinite(value).all():
            raise ValueError(f"{name} must be a finite HWC tensor")
    if target.shape != adapted.shape or source.shape[-1] != target.shape[-1]:
        raise ValueError("Descriptor shapes do not align")
    h, w, _ = target.shape
    if rgb.shape != (h, w, 3) or z.shape[:2] != (h, w):
        raise ValueError("RGB and projected features must align with target")
    before = readout.predict(z) - rgb
    after = readout.predict(materialize(z, state)) - rgb
    yy, xx = torch.meshgrid(
        torch.arange(h, device=target.device), torch.arange(w, device=target.device), indexing="ij"
    )
    probe = (yy + 2 * xx) % 4 == 3
    if not probe.any():
        raise ValueError("RGB selection partition 3 is empty")
    result = {}
    for prefix, a, b in (("rgb", before, after), ("probe", before[probe], after[probe])):
        old, new = float(a.square().mean()), float(b.square().mean())
        result.update(
            {
                f"{prefix}_mse_before": old,
                f"{prefix}_mse_after": new,
                f"{prefix}_decrease": old - new,
                f"{prefix}_relative_decrease": (old - new) / max(old, 1e-12),
            }
        )
        for channel in range(3):
            result[f"{prefix}_residual_abs_before_{channel}"] = float(a[..., channel].abs().mean())
            result[f"{prefix}_residual_abs_after_{channel}"] = float(b[..., channel].abs().mean())
            result[f"{prefix}_residual_variance_before_{channel}"] = float(
                a[..., channel].var(unbiased=False)
            )
            result[f"{prefix}_residual_variance_after_{channel}"] = float(
                b[..., channel].var(unbiased=False)
            )
    src = F.normalize(source.reshape(-1, source.shape[-1]), dim=-1)
    dst = F.normalize(adapted.reshape(-1, adapted.shape[-1]), dim=-1)
    # Target -> frozen source follows the propagation direction without labels.
    scores = dst @ src.T
    best, index = scores.max(-1)
    top = scores.topk(min(2, scores.shape[-1]), dim=-1).values
    gap = top[:, 0] - top[:, -1]
    probabilities = scores.softmax(-1)
    entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(-1)
    reverse = scores.argmax(0)
    cycle = reverse[index]
    dst_xy = _coords(h, w, target.device)
    src_xy = _coords(*source.shape[:2], target.device)
    src_xy = src_xy / src_xy.new_tensor((max(source.shape[1] - 1, 1), max(source.shape[0] - 1, 1)))
    dst_normalized = dst_xy / dst_xy.new_tensor((max(w - 1, 1), max(h - 1, 1)))
    displacement = src_xy[index] - dst_normalized
    field = displacement.reshape(h, w, 2)
    differences = []
    if h > 1:
        differences.append((field[1:] - field[:-1]).norm(dim=-1).reshape(-1))
    if w > 1:
        differences.append((field[:, 1:] - field[:, :-1]).norm(dim=-1).reshape(-1))
    drift = (adapted - target).norm(dim=-1) / target.norm(dim=-1).clamp_min(1e-12)
    result.update(
        {
            "match_cosine_mean": float(best.mean()),
            "match_cosine_median": float(best.median()),
            "match_margin_mean": float(gap.mean()),
            "match_entropy_mean": float(entropy.mean()),
            "mutual_nn_fraction": float(
                (cycle == torch.arange(len(dst), device=target.device)).float().mean()
            ),
            "cycle_error": float((dst_normalized[cycle] - dst_normalized).norm(dim=-1).mean()),
            "displacement_mean": float(displacement.norm(dim=-1).mean()),
            "displacement_std": float(displacement.norm(dim=-1).std(unbiased=False)),
            "displacement_smoothness": float(torch.cat(differences).mean()) if differences else 0.0,
            "state_a_norm": float(state.a.norm()),
            "state_b_norm": float(state.b.norm()),
            "state_total_norm": float(torch.sqrt(state.a.square().sum() + state.b.square().sum())),
            "feature_drift": float(drift.mean()),
            "feature_cosine": float(F.cosine_similarity(target, adapted, dim=-1).mean()),
            "feature_large_change_fraction": float((drift > 0.1).float().mean()),
        }
    )
    if not all(np.isfinite(v) for v in result.values()):
        raise ValueError("Visible features contain nonfinite values")
    return result


def enrich_candidates(features: list[dict], descriptors: list[Tensor] | None = None) -> list[dict]:
    """Add candidate-set statistics; optional descriptors provide exact disagreement."""
    if not features:
        raise ValueError("Candidates cannot be empty")
    values = np.array([f["probe_mse_after"] for f in features], dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Candidate RGB losses must be finite")
    if descriptors is not None:
        if len(descriptors) != len(features) or any(
            d.shape != descriptors[0].shape for d in descriptors
        ):
            raise ValueError("Descriptors must align with candidates")
        arrays = [F.normalize(d.detach().float(), dim=-1) for d in descriptors]
        disagreement = [
            float(torch.stack([(a - b).square().sum(-1).mean() for b in arrays]).mean())
            for a in arrays
        ]
    else:
        drift = np.array([f.get("feature_drift", 0.0) for f in features])
        disagreement = np.abs(drift[:, None] - drift[None, :]).mean(1).tolist()
    result = []
    for i, item in enumerate(features):
        competitors = np.delete(values, i)
        result.append(
            {
                **item,
                "rgb_rank": float((values < values[i]).sum()),
                "rgb_confidence_margin": float(competitors.min() - values[i])
                if len(competitors)
                else 0.0,
                "candidate_disagreement": disagreement[i],
            }
        )
    return result


def _trajectory_key(trajectory):
    if len(trajectory) != 2 or any(a not in ACTIONS for a in trajectory):
        raise ValueError("Trajectories must contain two known actions")
    return tuple(ACTIONS.index(a) for a in trajectory)


def _off(trajectories):
    matches = [i for i, t in enumerate(trajectories) if list(t) == ["OFF", "OFF"]]
    if len(matches) != 1 or len({tuple(t) for t in trajectories}) != len(trajectories):
        raise ValueError("Candidates require unique trajectories and exactly one OFF/OFF")
    for trajectory in trajectories:
        _trajectory_key(trajectory)
    return matches[0]


def _best(values, trajectories, maximize=True):
    return min(
        range(len(values)),
        key=lambda i: ((-1 if maximize else 1) * values[i], _trajectory_key(trajectories[i])),
    )


def _groups(rows):
    result = {}
    for row in rows:
        key = (str(row["sequence"]), str(row["target_slot"]))
        result.setdefault(key, []).append(row)
    return list(result.values())


def _matrix(rows, names):
    x = np.asarray([[r["visible"][n] for n in names] for r in rows], dtype=float)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError("Features must be finite and match the locked schema")
    return x


def _fit(rows, names, alpha):
    x = _matrix(rows, names)
    baselines = {}
    for group in _groups(rows):
        off = _off([r["trajectory"] for r in group])
        key = (str(group[0]["sequence"]), str(group[0]["target_slot"]))
        baselines[key] = float(group[off]["J"])
    y = np.array(
        [float(r["J"]) - baselines[(str(r["sequence"]), str(r["target_slot"]))] for r in rows],
        dtype=float,
    )
    mean, scale = x.mean(0), x.std(0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    normalized = (x - mean) / scale
    intercept = float(y.mean())
    coefficients = np.linalg.solve(
        normalized.T @ normalized + alpha * np.eye(len(names)), normalized.T @ (y - intercept)
    )
    return {
        "feature_names": names,
        "mean": mean.tolist(),
        "scale": scale.tolist(),
        "coefficients": coefficients.tolist(),
        "intercept": intercept,
        "alpha": float(alpha),
    }


def _pred(rows, model):
    return (_matrix(rows, model["feature_names"]) - np.array(model["mean"])) / np.array(
        model["scale"]
    ) @ np.array(model["coefficients"]) + model["intercept"]


def _choose_visible(predictions, trajectories, threshold):
    off = _off(trajectories)
    predictions = np.asarray(predictions, dtype=float).copy()
    predictions[off] = 0.0
    best = _best(predictions, trajectories)
    return best if predictions[best] > threshold else off


def _choose_rgb(features, trajectories, threshold):
    off = _off(trajectories)
    losses = [f["probe_mse_after"] for f in features]
    best = _best(losses, trajectories, maximize=False)
    relative = (losses[off] - losses[best]) / max(losses[off], 1e-12)
    return best if relative > threshold else off


def _reward(group, choice):
    off = _off([r["trajectory"] for r in group])
    return float(group[choice]["J"] - group[off]["J"])


def _sequence_mean(records):
    sequences = {}
    for sequence, reward in records:
        sequences.setdefault(sequence, []).append(reward)
    return float(np.mean([np.mean(v) for v in sequences.values()]))


def fit_selectors(rows: list[dict], config: dict) -> dict:
    """Fit only discovery rows; reject rather than silently consume other splits.

    Ridge hyperparameters use leave-one-sequence-out model predictions. RGB
    thresholds are searched directly on discovery rewards; their tuning score
    is not an out-of-fold generalization estimate.
    """
    if not rows or any(r.get("split") != "discovery" for r in rows):
        raise ValueError("Selector fitting accepts discovery rows only")
    if any(not np.isfinite(float(r["J"])) for r in rows):
        raise ValueError("Discovery rewards must be finite")
    output = {
        "schema_version": SCHEMA,
        "fit_split": "discovery",
        "training_sequences": sorted({str(r["sequence"]) for r in rows}),
        "by_resolution": {},
        "metadata": dict(config),
    }
    for resolution in sorted({str(r["resolution"]) for r in rows}):
        selected = [r for r in rows if str(r["resolution"]) == resolution]
        sequences = sorted({str(r["sequence"]) for r in selected})
        if len(sequences) < 2:
            raise ValueError("LOSO requires at least two discovery sequences per resolution")
        names = sorted(selected[0]["visible"])
        if not names or any(set(r["visible"]) != set(names) for r in selected):
            raise ValueError("Visible feature schema must be consistent")
        groups = _groups(selected)
        for group in groups:
            _off([r["trajectory"] for r in group])
        choices = []
        for alpha in ALPHAS:
            heldout = []
            for sequence in sequences:
                model = _fit([r for r in selected if str(r["sequence"]) != sequence], names, alpha)
                heldout.extend(
                    (g, _pred(g, model)) for g in groups if str(g[0]["sequence"]) == sequence
                )
            for threshold in THRESHOLDS:
                rewards = [
                    (
                        str(g[0]["sequence"]),
                        _reward(g, _choose_visible(p, [r["trajectory"] for r in g], threshold)),
                    )
                    for g, p in heldout
                ]
                choices.append((_sequence_mean(rewards), alpha, threshold))
        # Prefer more abstention and stronger regularization when rewards tie.
        cv_reward, alpha, threshold = max(choices, key=lambda x: (x[0], x[2], x[1]))
        rgb_choices = []
        for t in RGB_THRESHOLDS:
            records = [
                (
                    str(g[0]["sequence"]),
                    _reward(
                        g, _choose_rgb([r["visible"] for r in g], [r["trajectory"] for r in g], t)
                    ),
                )
                for g in groups
            ]
            rgb_choices.append((_sequence_mean(records), t))
        rgb_reward, rgb_threshold = max(rgb_choices)
        trajectories = sorted({tuple(r["trajectory"]) for r in selected}, key=_trajectory_key)
        if any({tuple(r["trajectory"]) for r in g} != set(trajectories) for g in groups):
            raise ValueError("Every discovery target must contain the same candidate set")
        fixed_rewards = []
        for t in trajectories:
            records = [
                (
                    str(g[0]["sequence"]),
                    _reward(g, next(i for i, r in enumerate(g) if tuple(r["trajectory"]) == t)),
                )
                for g in groups
            ]
            fixed_rewards.append(_sequence_mean(records))
        best_fixed = trajectories[_best(fixed_rewards, trajectories)]
        artifact = {
            **_fit(selected, names, alpha),
            "schema_version": SCHEMA,
            "fit_split": "discovery",
            "resolution": resolution,
            "training_sequences": sequences,
            "threshold": threshold,
            "rgb_threshold": rgb_threshold,
            "discovery_best_fixed": list(best_fixed),
            "metadata": {
                "cv": "leave_one_sequence_out",
                "reward": "sequence_mean_J_candidate_minus_J_OFF",
                "regression_target": "J_candidate_minus_same_pair_J_OFF",
                "off_prediction": "forced_zero_before_ranking",
                "cv_reward": cv_reward,
                "rgb_discovery_tuning_reward": rgb_reward,
                "rgb_threshold_selection": "discovery_reward_search_not_oof",
                "alpha_grid": list(ALPHAS),
                "threshold_grid": list(THRESHOLDS),
                "rgb_threshold_grid": list(RGB_THRESHOLDS),
                "config": dict(config),
            },
        }
        output["by_resolution"][resolution] = artifact
    return output


def validate_artifact(artifact: dict) -> dict:
    """Gate deployment on a complete discovery-fitted artifact."""
    if artifact.get("schema_version") != SCHEMA or artifact.get("fit_split") != "discovery":
        raise ValueError("Expected a locked discovery selector artifact")
    if "by_resolution" in artifact:
        if len(artifact["by_resolution"]) != 1:
            raise ValueError("Select the resolution-specific artifact before inference")
        artifact = next(iter(artifact["by_resolution"].values()))
        return validate_artifact(artifact)
    names = artifact.get("feature_names", [])
    if not names or len(set(names)) != len(names) or not artifact.get("training_sequences"):
        raise ValueError("Missing feature schema or training provenance")
    for name in ("coefficients", "mean", "scale"):
        array = np.asarray(artifact.get(name), dtype=float)
        if array.shape != (len(names),) or not np.isfinite(array).all():
            raise ValueError(f"Invalid artifact {name}")
    if np.any(np.asarray(artifact["scale"]) <= 0):
        raise ValueError("Artifact scales must be positive")
    for name in ("intercept", "alpha", "threshold", "rgb_threshold"):
        if name not in artifact or not np.isfinite(artifact[name]):
            raise ValueError(f"Invalid artifact {name}")
    if artifact["threshold"] < 0 or artifact["rgb_threshold"] < 0 or artifact["alpha"] <= 0:
        raise ValueError("Invalid selector hyperparameters")
    _trajectory_key(artifact["discovery_best_fixed"])
    return artifact


def predict_candidates(features: list[dict], artifact: dict) -> list[float]:
    """Return model-predicted J-minus-OFF reward without labels.

    These are raw regression outputs; selection pins the known OFF reward to
    zero before ranking and applying the abstention threshold.
    """
    model = validate_artifact(artifact)
    if not features or any(set(f) != set(model["feature_names"]) for f in features):
        raise ValueError("Inference features must match the locked schema exactly")
    return _pred([{"visible": f} for f in features], model).tolist()


def select_candidates(features: list[dict], trajectories: list[list[str]], artifact: dict) -> dict:
    model = validate_artifact(artifact)
    if len(features) != len(trajectories):
        raise ValueError("Features and trajectories must align")
    predictions = predict_candidates(features, model)
    _off(trajectories)
    fixed = [i for i, t in enumerate(trajectories) if list(t) == model["discovery_best_fixed"]]
    if len(fixed) != 1:
        raise ValueError("Locked best-fixed trajectory is absent")
    return {
        "rgb_selector": _choose_rgb(features, trajectories, model["rgb_threshold"]),
        "visible_selector": _choose_visible(predictions, trajectories, model["threshold"]),
        "best_fixed": fixed[0],
    }
