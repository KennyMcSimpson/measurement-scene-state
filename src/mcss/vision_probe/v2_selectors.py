"""Discovery-supervised nested sequence LOSO and conservative V2 policies.

Source-annotated deployment permits source masks; target labels enter only
training rows. Neither inference API accepts labels, masks, or oracle answers.
"""

from __future__ import annotations

import itertools

import numpy as np

SCHEMA = "mcss.v2_selectors.v1"
ACTIONS = ("OFF", "A", "B", "ALL")
TRAJECTORIES = tuple(itertools.product(ACTIONS, repeat=2))
ALPHAS = (0.1, 1.0, 10.0, 100.0)
THRESHOLDS = (0.0, 0.0005, 0.001, 0.002, 0.005)
TIE_TOLERANCE = 1e-12


def _groups(rows):
    groups = {}
    for row in rows:
        key = (str(row["sequence"]), str(row["target_slot"]))
        groups.setdefault(key, []).append(row)
    return [groups[k] for k in sorted(groups)]


def _paths(trajectories):
    paths = [tuple(t) for t in trajectories]
    if len(paths) != 16 or set(paths) != set(TRAJECTORIES):
        raise ValueError("Exactly 16 unique two-step trajectories required")
    return paths, paths.index(("OFF", "OFF"))


def _arrays(rows, names):
    x = np.asarray([[r["visible"][n] for n in names] for r in rows], dtype=float)
    if not np.isfinite(x).all():
        raise ValueError("Visible features must be finite")
    baselines, pair_counts = {}, {}
    for group in _groups(rows):
        paths, off = _paths([r["trajectory"] for r in group])
        key = (str(group[0]["sequence"]), str(group[0]["target_slot"]))
        baselines[key] = float(group[off]["J"])
        pair_counts[key[0]] = pair_counts.get(key[0], 0) + 1
    n_sequences = len(pair_counts)
    # J already averages foreground objects. Equal sequence -> pair -> candidate.
    weights = np.array([1 / (n_sequences * pair_counts[str(r["sequence"])] * 16) for r in rows])
    rewards = np.array(
        [float(r["J"]) - baselines[(str(r["sequence"]), str(r["target_slot"]))] for r in rows]
    )
    return x, rewards, weights


def _fit(rows, names, alpha):
    x, y, weights = _arrays(rows, names)
    mean = weights @ x
    scale = np.sqrt(weights @ (x - mean) ** 2)
    scale = np.where(scale > 1e-12, scale, 1.0)
    z = (x - mean) / scale
    intercept = float(weights @ y)
    # Match V1 ridge scale: fitting weights sum to n_rows, scaler uses sum-one.
    weights = weights * len(rows)
    coefficients = np.linalg.solve(
        z.T @ (weights[:, None] * z) + alpha * np.eye(len(names)), z.T @ (weights * (y - intercept))
    )
    identities = sorted({str(r["sequence"]) for r in rows})
    return {
        "feature_names": list(names),
        "mean": mean.tolist(),
        "scale": scale.tolist(),
        "coefficients": coefficients.tolist(),
        "intercept": intercept,
        "alpha": alpha,
        "training_sequences": identities,
        "scaler_provenance": {
            "fit_sequences": identities,
            "weighting": "equal_sequence_then_pair_then_candidate",
            "weight_sum": float(weights.sum()),
            "row_count": len(rows),
            "sequence_total_weights": {
                s: float(
                    sum(w for r, w in zip(rows, weights, strict=True) if str(r["sequence"]) == s)
                )
                for s in identities
            },
        },
        "ridge_objective": "weighted_sum_squared_error_plus_alpha_l2",
        "ridge_weight_normalization": "sum_n_rows",
        "scaler_weight_normalization": "sum_one",
    }


def _predict(features, model):
    x = np.asarray([[f[n] for n in model["feature_names"]] for f in features], dtype=float)
    if not np.isfinite(x).all():
        raise ValueError("Inference features must be finite")
    return ((x - np.asarray(model["mean"])) / np.asarray(model["scale"])) @ np.asarray(
        model["coefficients"]
    ) + model["intercept"]


def _choice(predictions, trajectories, threshold):
    paths, off = _paths(trajectories)
    candidates = [i for i in range(len(paths)) if i != off]
    maximum = max(float(predictions[i]) for i in candidates)
    ties = [i for i in candidates if maximum - float(predictions[i]) <= TIE_TOLERANCE]
    best = min(ties, key=lambda i: TRAJECTORIES.index(paths[i]))
    return best if float(predictions[best]) > threshold else off


def _decisions(groups, model, threshold, always_off=False):
    result = []
    for group in groups:
        paths, off = _paths([r["trajectory"] for r in group])
        predictions = (
            [0.0] * len(group)
            if always_off
            else _predict([r["visible"] for r in group], model).tolist()
        )
        predictions[off] = 0.0
        selected = off if always_off else _choice(predictions, paths, threshold)
        result.append(
            {
                "sequence": str(group[0]["sequence"]),
                "target_slot": group[0]["target_slot"],
                "trajectory": list(paths[selected]),
                "selected_index": selected,
                "predicted_rewards": predictions,
                "candidate_trajectories": [list(p) for p in paths],
                "reward": float(group[selected]["J"] - group[off]["J"]),
                "write": selected != off,
                "fitted_sequences": model["training_sequences"] if model else [],
            }
        )
    return result


def _score(decisions):
    groups = {}
    for row in decisions:
        groups.setdefault(row["sequence"], []).append(row)
    return {
        "net_delta_J": float(np.mean([np.mean([r["reward"] for r in g]) for g in groups.values()])),
        "write_rate": float(np.mean([np.mean([r["write"] for r in g]) for g in groups.values()])),
        "per_sequence": {
            s: {
                "net_delta_J": float(np.mean([r["reward"] for r in g])),
                "write_rate": float(np.mean([r["write"] for r in g])),
                "n_pairs": len(g),
            }
            for s, g in groups.items()
        },
    }


def _winner(candidates):
    highest = max(c["score"]["net_delta_J"] for c in candidates)
    tied = [c for c in candidates if highest - c["score"]["net_delta_J"] <= TIE_TOLERANCE]
    return min(
        tied,
        key=lambda c: (
            c["score"]["write_rate"],
            -(c["threshold"] if c["threshold"] is not None else float("inf")),
            -(c["alpha"] if c["alpha"] is not None else float("inf")),
        ),
    )


def _tune(rows, names):
    sequences = sorted({str(r["sequence"]) for r in rows})
    if len(sequences) < 2:
        raise ValueError("Grouped tuning needs at least two sequences")
    candidates, folds = [], []
    for alpha in ALPHAS:
        prediction_groups = []
        for held in sequences:
            train = [r for r in rows if str(r["sequence"]) != held]
            groups = _groups([r for r in rows if str(r["sequence"]) == held])
            model = _fit(train, names, alpha)
            folds.append({"held_out_sequences": [held], "model": model})
            prediction_groups.extend((g, model) for g in groups)
        for threshold in THRESHOLDS:
            decisions = [d for g, m in prediction_groups for d in _decisions([g], m, threshold)]
            candidates.append(
                {
                    "policy": "RIDGE_GATE",
                    "alpha": alpha,
                    "threshold": threshold,
                    "score": _score(decisions),
                    "oof_decisions": decisions,
                }
            )
    off = _decisions(_groups(rows), None, None, always_off=True)
    candidates.append(
        {
            "policy": "ALWAYS_OFF",
            "alpha": None,
            "threshold": None,
            "score": _score(off),
            "oof_decisions": off,
        }
    )
    winner = _winner(candidates)
    return winner, {
        "training_sequences": sequences,
        "folds": folds,
        "candidates": candidates,
        "selected": {k: winner[k] for k in ("policy", "alpha", "threshold", "score")},
    }


def fit_v2(rows: list[dict], base_feature_names: list[str], config: dict) -> dict:
    """Nested LOSO discovery-only fitting; no exposed validation rows are accepted."""
    if not rows or any(r.get("split") != "discovery" for r in rows):
        raise ValueError("Only discovery rows may train V2 selectors")
    if (
        not base_feature_names
        or len(set(base_feature_names)) != len(base_feature_names)
        or "f_cycle" in base_feature_names
    ):
        raise ValueError("Base features must be unique and exclude f_cycle")
    allowed = list(base_feature_names) + ["f_cycle"]
    if any(set(r["visible"]) != set(allowed) for r in rows):
        raise ValueError("Visible feature schema must match frozen base plus f_cycle")
    if any(not np.isfinite(float(r["J"])) for r in rows):
        raise ValueError("Rewards must be finite")
    for group in _groups(rows):
        _paths([r["trajectory"] for r in group])
    sequences = sorted({str(r["sequence"]) for r in rows})
    if len(sequences) < 3:
        raise ValueError("Nested LOSO requires at least three discovery sequences")
    result = {"schema_version": SCHEMA, "selectors": {}, "nested_cv_results": {}}
    for name, names in (("GateOnly", list(base_feature_names)), ("CycleGate", allowed)):
        outer, decisions = [], []
        for held in sequences:
            train = [r for r in rows if str(r["sequence"]) != held]
            winner, inner = _tune(train, names)
            model = (
                _fit(train, names, winner["alpha"]) if winner["policy"] != "ALWAYS_OFF" else None
            )
            held_decisions = _decisions(
                _groups([r for r in rows if str(r["sequence"]) == held]),
                model,
                winner["threshold"],
                winner["policy"] == "ALWAYS_OFF",
            )
            outer.append(
                {
                    "held_out_sequences": [held],
                    "training_sequences": sorted({str(r["sequence"]) for r in train}),
                    "inner_cv": inner,
                    "model": model,
                    "oof_decisions": held_decisions,
                }
            )
            decisions.extend(held_decisions)
        score = _score(decisions)
        fallback = score["net_delta_J"] <= 0
        final_tuning = None
        winner = {"policy": "ALWAYS_OFF", "alpha": None, "threshold": None}
        model = None
        if not fallback:
            winner, final_tuning = _tune(rows, names)
            if winner["policy"] != "ALWAYS_OFF":
                model = _fit(rows, names, winner["alpha"])
        artifact = {
            "schema_version": SCHEMA,
            "name": name,
            "fit_split": "discovery",
            "training_sequences": sequences,
            "feature_names": names,
            "allowed_input_features": allowed,
            "policy": winner["policy"],
            "alpha": winner["alpha"],
            "threshold": winner["threshold"],
            "model": model,
            "outer_oof_score_before_fallback": score,
            "outer_nonpositive_forced_off": fallback,
            "source_annotated": True,
            "new_signal": "f_cycle" if name == "CycleGate" else None,
            "config": dict(config),
            "grid": {"alpha": list(ALPHAS), "threshold": list(THRESHOLDS), "always_off": True},
            "tie_tolerance": TIE_TOLERANCE,
            "selection_scope": (
                "discovery_supervised; OFF fixed zero; 15 nonOFF compete; "
                "strict predicted reward > threshold"
            ),
            "performance_scope": (
                "Outer OOF precedes final fallback; fallback-selected score "
                "is not an independent estimate"
            ),
            "ridge_weight_normalization": "sum_n_rows",
            "scaler_weight_normalization": "sum_one",
            "label_aggregation": "input J is object mean; equal pair then sequence",
        }
        result["selectors"][name] = artifact
        result["nested_cv_results"][name] = {
            "outer_folds": outer,
            "outer_oof_decisions": decisions,
            "outer_oof_score_before_fallback": score,
            "final_tuning": final_tuning,
            "final_policy": winner["policy"],
            "outer_nonpositive_forced_off": fallback,
        }
    return result


def _validate(features, artifact):
    if artifact.get("schema_version") != SCHEMA or artifact.get("fit_split") != "discovery":
        raise ValueError("Expected discovery-fitted V2 artifact")
    if not artifact.get("training_sequences") or artifact.get("policy") not in (
        "ALWAYS_OFF",
        "RIDGE_GATE",
    ):
        raise ValueError("Missing training provenance or policy")
    allowed, required = set(artifact["allowed_input_features"]), set(artifact["feature_names"])
    if not features or any(not required <= set(f) or not set(f) <= allowed for f in features):
        raise ValueError("Only locked deployment-visible features are accepted")
    if any(not all(np.isfinite(float(v)) for v in f.values()) for f in features):
        raise ValueError("Inference features must be finite")
    if artifact["policy"] == "RIDGE_GATE":
        model = artifact.get("model")
        if (
            not model
            or model["feature_names"] != artifact["feature_names"]
            or model["training_sequences"] != artifact["training_sequences"]
        ):
            raise ValueError("Model provenance does not match artifact")
        if artifact["alpha"] not in ALPHAS or artifact["threshold"] not in THRESHOLDS:
            raise ValueError("Hyperparameters not in preregistered grid")
        for field in ("coefficients", "mean", "scale"):
            array = np.asarray(model[field], dtype=float)
            if array.shape != (len(required),) or not np.isfinite(array).all():
                raise ValueError("Invalid model numeric arrays")
        if np.any(np.asarray(model["scale"]) <= 0) or not np.isfinite(model["intercept"]):
            raise ValueError("Invalid model normalization")


def predict_v2(features: list[dict], artifact: dict) -> list[float]:
    """Raw reward predictions; OFF is fixed to zero by the decision rule."""
    _validate(features, artifact)
    if artifact["policy"] == "ALWAYS_OFF":
        return [0.0] * len(features)
    return _predict(features, artifact["model"]).tolist()


def select_v2(features: list[dict], trajectories: list[list[str]], artifact: dict) -> int:
    predictions = predict_v2(features, artifact)
    paths, off = _paths(trajectories)
    if len(features) != len(paths):
        raise ValueError("Features and trajectories must align")
    return (
        off
        if artifact["policy"] == "ALWAYS_OFF"
        else _choice(predictions, paths, artifact["threshold"])
    )
