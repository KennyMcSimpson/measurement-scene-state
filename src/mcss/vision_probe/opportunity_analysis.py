"""Descriptive, paired analysis of the locked two-step opportunity experiment.

Answers are used only for retrospective oracle/analysis results. This module never
fits a selector. Sequence bootstrap is primary; pair bootstrap is supplemental.
"""

from __future__ import annotations

import csv
import itertools
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ACTIONS = ("OFF", "A", "B", "ALL")
TRAJECTORIES = set(itertools.product(ACTIONS, repeat=2))
SEED = 20260926
N_BOOTSTRAP = 10000


def _score(row):
    return float(row.get("_primary", row["J"]))


def _groups(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[
            (int(row["resolution"]), row["split"], row["sequence"], int(row["target_slot"]))
        ].append(row)
    slots = defaultdict(set)
    for key, group in groups.items():
        paths = [tuple(r["trajectory"]) for r in group]
        if len(paths) != 16 or set(paths) != TRAJECTORIES:
            raise ValueError(f"Incomplete or duplicate trajectories: {key}")
        slots[key[:3]].add(key[3])
        for row in group:
            if not np.isfinite(float(_score(row))):
                raise ValueError(f"Nonfinite primary endpoint: {key}")
    if not groups:
        raise ValueError("No experiment rows")
    for key, values in slots.items():
        if values != {1, 2}:
            raise ValueError(f"Both target slots 1 and 2 required: {key}: {values}")
    return dict(sorted(groups.items()))


def _best(rows):
    # OFF first for ties; tie counts and fractional frequencies are reported separately.
    return sorted(
        rows,
        key=lambda r: (
            -float(_score(r)),
            tuple(ACTIONS.index(a) for a in r["trajectory"]),
        ),
    )[0]


def _boot(values):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return {"mean": None, "ci95": [None, None], "n": 0}
    rng = np.random.default_rng(SEED)
    samples = values[rng.integers(0, len(values), (N_BOOTSTRAP, len(values)))].mean(axis=1)
    return {
        "mean": float(values.mean()),
        "ci95": np.quantile(samples, [0.025, 0.975]).tolist(),
        "n": len(values),
    }


def _seq_means(records, value):
    groups = defaultdict(list)
    for row in records:
        groups[row["sequence"]].append(float(row[value]))
    return {key: float(np.mean(vals)) for key, vals in sorted(groups.items())}


def _distribution(records, value):
    seq = _seq_means(records, value)
    vals = np.array(list(seq.values()))
    positive = np.maximum(vals, 0)
    total = positive.sum()
    ordered = np.sort(positive)[::-1]
    loso = {s: float(np.mean([v for k, v in seq.items() if k != s])) for s in seq if len(seq) > 1}
    loso_details = {}
    for dropped in loso:
        remaining = np.array([v for k, v in seq.items() if k != dropped])
        interval = _boot(remaining)
        positive_remaining = np.maximum(remaining, 0)
        positive_total = float(positive_remaining.sum())
        loso_details[dropped] = {
            **interval,
            "positive_sequence_count": int((remaining > 1e-12).sum()),
            "top1_positive_gain_share": (
                float(positive_remaining.max() / positive_total) if positive_total else 0.0
            ),
            "directional_ci_positive": interval["ci95"][0] > 0,
        }
    return {
        "sequence_values": seq,
        "sequence_bootstrap": _boot(vals),
        "pair_bootstrap": _boot([r[value] for r in records]),
        "positive_sequences": int((vals > 1e-12).sum()),
        "sequence_std": float(vals.std()),
        "pair_mean": float(np.mean([r[value] for r in records])),
        "pair_median": float(np.median([r[value] for r in records])),
        "pair_std": float(np.std([r[value] for r in records])),
        "pair_min": min(r[value] for r in records),
        "pair_max": max(r[value] for r in records),
        "improved_pairs": sum(r[value] > 1e-12 for r in records),
        "tied_pairs": sum(abs(r[value]) <= 1e-12 for r in records),
        "worse_pairs": sum(r[value] < -1e-12 for r in records),
        "threshold_sequence_counts": {str(t): int((vals >= t).sum()) for t in (0.005, 0.01, 0.02)},
        "threshold_pair_counts": {
            str(t): sum(r[value] >= t for r in records) for t in (0.005, 0.01, 0.02)
        },
        "sequence_positive_gain_share": {
            name: float(max(value, 0) / total) if total else 0.0 for name, value in seq.items()
        },
        "top1_positive_gain_share": float(ordered[:1].sum() / total) if total else 0.0,
        "top3_positive_gain_share": float(ordered[:3].sum() / total) if total else 0.0,
        "leave_one_sequence_out": loso,
        "leave_one_sequence_out_ci": loso_details,
        "all_loso_ci_lower_positive": (
            all(r["directional_ci_positive"] for r in loso_details.values())
            if loso_details
            else None
        ),
        "loso_min": min(loso.values()) if loso else None,
        "quantiles": np.quantile(vals, [0, 0.25, 0.5, 0.75, 1]).tolist(),
    }


def _pair_records(rows):
    records = []
    for (resolution, split, sequence, slot), group in _groups(rows).items():
        off = next(
            r
            for r in group
            if r["trajectory"] == ["OFF", "OFF"] or tuple(r["trajectory"]) == ("OFF", "OFF")
        )
        best = _best(group)
        constant = _best([r for r in group if r["trajectory"][0] == r["trajectory"][1]])
        methods = {"OFF": off, "trajectory_oracle": best, "best_constant_oracle": constant}
        for action in ("A", "B", "ALL"):
            methods[f"{action}/{action}"] = next(
                r for r in group if tuple(r["trajectory"]) == (action, action)
            )
        for row in group:
            for name in row.get("selected_methods", []):
                if name in methods and methods[name] is not row:
                    raise ValueError(f"Duplicate selected method {name}: {sequence}/{slot}")
                methods[name] = row
        base = {
            "resolution": resolution,
            "split": split,
            "sequence": sequence,
            "target_slot": slot,
            "source_frame": off.get("source_frame"),
            "target_frame": off.get("target_frame"),
        }
        for name, row in methods.items():
            record = {
                **base,
                "method": name,
                "trajectory": "/".join(row["trajectory"]),
                **{m: float(row.get(m, _score(row))) for m in ("J", "F", "JF", "token_iou")},
                "foreground_pixel_count": off.get("foreground_pixel_count"),
                "foreground_pixel_fraction": off.get("foreground_pixel_fraction"),
                "foreground_token_count_224": off.get("foreground_token_counts", {}).get("224"),
                "foreground_token_count_448": off.get("foreground_token_counts", {}).get("448"),
                "gain_off": float(_score(row) - _score(off)),
                "gain_fixed": (
                    float(_score(row) - _score(methods["best_fixed"]))
                    if "best_fixed" in methods
                    else None
                ),
                "gain_constant": float(_score(row) - _score(constant)),
                "oracle_gap": float(_score(best) - _score(off)),
            }
            records.append(record)
    return records


def opportunity_summary(rows, primary_metric="J"):
    """Answer-only diagnostics; safe to call on discovery before selector fitting."""
    rows = [{**r, "_primary": r[primary_metric]} for r in rows]
    records = _pair_records(rows)
    result = {}
    for resolution, split in sorted({(r["resolution"], r["split"]) for r in records}):
        selected = [
            r
            for r in records
            if r["resolution"] == resolution
            and r["split"] == split
            and r["method"] == "trajectory_oracle"
        ]
        result[f"{resolution}/{split}"] = {
            "oracle_minus_off": _distribution(selected, "gain_off"),
            "trajectory_minus_constant": _distribution(selected, "gain_constant"),
        }
    return result


def _ranks(x):
    x = np.asarray(x)
    return np.array([np.sum(x < v) + (np.sum(x == v) - 1) / 2 for v in x], dtype=float)


def _corr(x, y):
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _prediction(x, y):
    x, y = np.asarray(x), np.asarray(y)
    positive = y > 0
    n_pos, n_neg = int(positive.sum()), int((~positive).sum())
    auc = (
        ((float(_ranks(x)[positive].sum()) - n_pos * (n_pos - 1) / 2) / (n_pos * n_neg))
        if n_pos and n_neg
        else None
    )
    # Group score ties, matching threshold-based average precision.
    ap, recall = 0.0, 0.0
    for threshold in sorted(set(x), reverse=True):
        mask = x >= threshold
        new_recall = float(positive[mask].sum() / n_pos) if n_pos else 0.0
        ap += (new_recall - recall) * float(positive[mask].mean())
        recall = new_recall
    return {
        "pearson": _corr(x, y),
        "spearman": _corr(_ranks(x), _ranks(y)),
        "auroc_positive_gain": auc,
        "average_precision_positive_gain": ap if n_pos else None,
        "n": len(x),
    }


def _signals(groups):
    signals = defaultdict(lambda: defaultdict(list))
    ranking = defaultdict(lambda: defaultdict(list))
    for key, rows in groups.items():
        prefix = f"{key[0]}/{key[1]}"
        off = next(r for r in rows if tuple(r["trajectory"]) == ("OFF", "OFF"))
        names = set.intersection(*(set(r.get("visible", {})) for r in rows))
        for name in sorted(names):
            x = [float(r["visible"][name]) for r in rows]
            y = [float(_score(r) - _score(off)) for r in rows]
            if not np.isfinite(x).all():
                continue
            signals[prefix][name].extend(zip(x, y, strict=True))
            # Both orientations are descriptive, never chosen using validation to deploy.
            for sign in (1, -1):
                order = sorted(
                    range(16),
                    key=lambda i: (
                        -sign * x[i],
                        tuple(ACTIONS.index(a) for a in rows[i]["trajectory"]),
                    ),
                )
                maximum = max(y)
                ranking[prefix][f"{name}:sign={sign}"].append(
                    {
                        "top1_hit": float(y[order[0]] >= maximum - 1e-12),
                        "top3_hit": float(max(y[i] for i in order[:3]) >= maximum - 1e-12),
                        "regret": maximum - y[order[0]],
                        "normalized_regret": (
                            (maximum - y[order[0]]) / (maximum - min(y))
                            if maximum > min(y)
                            else 0.0
                        ),
                    }
                )
    summary = {
        k: {n: _prediction(*zip(*v, strict=True)) for n, v in names.items()}
        for k, names in signals.items()
    }
    ranks = {
        k: {
            n: {m: float(np.mean([r[m] for r in vals])) for m in vals[0]}
            for n, vals in names.items()
        }
        for k, names in ranking.items()
    }
    return {
        "features": summary,
        "candidate_ranking": ranks,
        "scope": "Descriptive only; candidates share images and sequences; no iid p-values. "
        "Feature orientations are not validation-selected deployment rules.",
    }


def _objects(groups):
    output = {}
    for key, rows in groups.items():
        off = next(r for r in rows if tuple(r["trajectory"]) == ("OFF", "OFF"))
        best = _best(rows)
        baseline = {str(o["object_id"]): o for o in off.get("objects", [])}
        for obj in best.get("objects", []):
            old = baseline[str(obj["object_id"])]
            bucket = obj.get("size_bucket", "unknown")
            prefix = f"{key[0]}/{key[1]}/{bucket}"
            output.setdefault(prefix, []).append(
                {
                    "sequence": key[2],
                    "target_slot": key[3],
                    "object_id": obj["object_id"],
                    "gain": float(_score(obj) - _score(old)),
                    "pixel_count": obj.get("pixel_count"),
                    "pixel_fraction": obj.get("pixel_fraction"),
                    "token_count_224": obj.get("token_count_224"),
                    "token_count_448": obj.get("token_count_448"),
                }
            )
    return {
        key: {
            "objects": vals,
            "sequence_macro_gain": float(np.mean(list(_seq_means(vals, "gain").values()))),
        }
        for key, vals in output.items()
    }


def _write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["empty"])
        writer.writeheader()
        writer.writerows(rows)


def analyze(raw_path, output_dir, config, selectors_by_resolution=None):
    """Write locked experiment reports. Selectors are provenance, never fitted here."""
    rows = [json.loads(line) for line in Path(raw_path).read_text().splitlines() if line.strip()]
    primary = config.get("primary_metric", "J")
    rows = [{**r, "_primary": r[primary]} for r in rows]
    groups = _groups(rows)
    records = _pair_records(rows)
    opportunity = opportunity_summary(rows, primary)
    object_groups = {
        k: [
            {**r, "objects": [{**o, "_primary": o[primary]} for o in r.get("objects", [])]}
            for r in group
        ]
        for k, group in groups.items()
    }
    objects = _objects(object_groups)
    signals = _signals(groups)
    bootstrap, selectors, sequences, robustness = {}, {}, [], {}
    for prefix, summary in opportunity.items():
        resolution, split = prefix.split("/")
        subset = [r for r in records if r["resolution"] == int(resolution) and r["split"] == split]
        for method in sorted({r["method"] for r in subset}):
            selected = [r for r in subset if r["method"] == method]
            if len(selected) != len(subset) / len({r["method"] for r in subset}):
                raise ValueError(f"Incomplete selected method coverage: {prefix}/{method}")
            for seq in sorted({r["sequence"] for r in selected}):
                sr = [r for r in selected if r["sequence"] == seq]
                sequences.append(
                    {
                        "resolution": int(resolution),
                        "split": split,
                        "sequence": seq,
                        "method": method,
                        **{
                            m: float(np.mean([r[m] for r in sr]))
                            for m in ("J", "F", "JF", "token_iou", "gain_off")
                        },
                    }
                )
            diffs = {m: _distribution(selected, m) for m in ("gain_off", "gain_constant")}
            if all(r["gain_fixed"] is not None for r in selected):
                diffs["gain_fixed"] = _distribution(selected, "gain_fixed")
            bootstrap[f"{prefix}/{method}"] = diffs
            if "selector" in method:
                writes = [r for r in selected if r["trajectory"] != "OFF/OFF"]
                denom = summary["oracle_minus_off"]["sequence_bootstrap"]["mean"]
                net = diffs["gain_off"]["sequence_bootstrap"]["mean"]
                positive_denominator = sum(max(r["oracle_gap"], 0) for r in selected)
                capture = (
                    sum(max(r["gain_off"], 0) for r in selected) / positive_denominator
                    if positive_denominator > 0
                    else None
                )
                harmful = sum(r["gain_off"] < 0 for r in writes) / len(writes) if writes else None
                strong = (
                    net > 0
                    and capture is not None
                    and capture > 0
                    and harmful is not None
                    and harmful <= 0.25
                    and "gain_fixed" in diffs
                    and diffs["gain_off"]["sequence_bootstrap"]["ci95"][0] > 0
                    and diffs["gain_fixed"]["sequence_bootstrap"]["ci95"][0] > 0
                )
                selectors[f"{prefix}/{method}"] = {
                    "opportunity_capture": capture,
                    "harmful_among_writes": harmful,
                    "write_pairs": len(writes),
                    "write_rate": len(writes) / len(selected),
                    "net_gain": net,
                    "net_gain_ratio": net / denom if denom > 0 else None,
                    "improved_pairs": sum(r["gain_off"] > 1e-12 for r in selected),
                    "tied_pairs": sum(abs(r["gain_off"]) <= 1e-12 for r in selected),
                    "worse_pairs": sum(r["gain_off"] < -1e-12 for r in selected),
                    "beneficial_write_precision": (
                        sum(r["gain_off"] > 1e-12 for r in writes) / len(writes) if writes else None
                    ),
                    "beneficial_write_recall": (
                        sum(r["gain_off"] > 1e-12 for r in writes)
                        / sum(r["oracle_gap"] > 1e-12 for r in selected)
                        if any(r["oracle_gap"] > 1e-12 for r in selected)
                        else None
                    ),
                    "candidate_selection_accuracy": float(
                        np.mean([abs(r["gain_off"] - r["oracle_gap"]) <= 1e-12 for r in selected])
                    ),
                    "regret_to_oracle": float(
                        np.mean([r["oracle_gap"] - r["gain_off"] for r in selected])
                    ),
                    "off_rate": 1 - len(writes) / len(selected),
                    "H2": "STRONG" if strong else "NOT_ESTABLISHED",
                    "partial_requires": (
                        "positive net/capture and locked feature sign stable discovery/validation"
                    ),
                }
        stat = summary["oracle_minus_off"]
        non_tiny = [
            v["objects"]
            for k, v in objects.items()
            if k.startswith(prefix + "/") and k.rsplit("/", 1)[1] not in ("tiny", "unknown")
        ]
        non_tiny_rows = [r for vals in non_tiny for r in vals]
        non_tiny_gain = (
            float(np.mean(list(_seq_means(non_tiny_rows, "gain").values())))
            if non_tiny_rows
            else None
        )
        strong = (
            stat["sequence_bootstrap"]["ci95"][0] > 0
            and stat["positive_sequences"] >= 4
            and stat["top1_positive_gain_share"] <= 0.5
            and stat["loso_min"] is not None
            and stat["loso_min"] > 0
            and non_tiny_gain is not None
            and non_tiny_gain > 0
        )
        moderate = (
            stat["sequence_bootstrap"]["mean"] > 0
            and stat["positive_sequences"] >= 2
            and stat["top1_positive_gain_share"] < 0.8
        )
        summary["H1"] = "STRONG" if strong else "MODERATE" if moderate else "NOT_ESTABLISHED"
        summary["non_tiny_gain"] = non_tiny_gain
        frequency = Counter()
        constant_frequency = Counter()
        sequence_preferences = defaultdict(Counter)
        size_preferences = defaultdict(Counter)
        tie_counts = []
        for key, group in groups.items():
            if f"{key[0]}/{key[1]}" != prefix:
                continue
            maximum = max(_score(r) for r in group)
            tied = [r for r in group if abs(_score(r) - maximum) <= 1e-12]
            tie_counts.append(len(tied))
            constant_frequency[
                "/".join(
                    _best([r for r in group if r["trajectory"][0] == r["trajectory"][1]])[
                        "trajectory"
                    ]
                )
            ] += 1
            for r in tied:
                path = "/".join(r["trajectory"])
                frequency[path] += 1 / len(tied)
                sequence_preferences[key[2]][path] += 1 / len(tied)
                for obj in r.get("objects", []):
                    size_preferences[obj.get("size_bucket", "unknown")][path] += 1 / len(tied)
        p = np.array(list(frequency.values())) / sum(frequency.values())
        summary["tie_aware_oracle"] = {
            "fractional_counts": dict(frequency),
            "constant_oracle_counts_off_first_ties": dict(constant_frequency),
            "sequence_preferences": {k: dict(v) for k, v in sequence_preferences.items()},
            "object_size_preferences": {k: dict(v) for k, v in size_preferences.items()},
            "entropy_nats": float(-(p * np.log(p)).sum()),
            "tie_counts": tie_counts,
        }
        robustness[prefix] = {
            "oracle": stat,
            "object_size": {k: v for k, v in objects.items() if k.startswith(prefix + "/")},
        }
    # Partial support uses a sign fixed from discovery; never flip signs using validation.
    features = signals["features"]
    for key, result in selectors.items():
        resolution, split, method = key.split("/")
        if split != "validation" or result["H2"] == "STRONG":
            continue
        stable = []
        predefined = (
            "rgb_decrease",
            "match_cosine_mean",
            "cycle_error",
            "feature_drift",
            "state_total_norm",
        )
        discovery = features.get(f"{resolution}/discovery", {})
        candidates = [n for n in predefined if discovery.get(n, {}).get("spearman") is not None]
        chosen = (
            max(candidates, key=lambda n: abs(discovery[n]["spearman"])) if candidates else None
        )
        for name in [chosen] if chosen else []:
            dev = discovery[name]
            val = features.get(f"{resolution}/validation", {}).get(name, {})
            a, b = dev.get("spearman"), val.get("spearman")
            if a is not None and b is not None and a * b > 0:
                stable.append(name)
        result["descriptive_stable_sign_features"] = stable
        diff = bootstrap[key]["gain_off"]["sequence_bootstrap"]
        if (
            stable
            and diff["mean"] > 0
            and result["opportunity_capture"] is not None
            and result["opportunity_capture"] > 0
            and diff["ci95"][0] <= 0 <= diff["ci95"][1]
        ):
            result["H2"] = "PARTIAL"
    audits = [
        {
            "resolution": r["resolution"],
            "split": r["split"],
            "sequence": r["sequence"],
            "target_slot": r["target_slot"],
            "trajectory": r["trajectory"],
            **audit,
        }
        for r in rows
        for audit in r.get("rank_audit", [])
    ]
    rank_relationships = {}
    for resolution, split in sorted({(r["resolution"], r["split"]) for r in rows}):
        candidate_stats = []
        for key, group in groups.items():
            if key[:2] != (resolution, split):
                continue
            off = next(r for r in group if tuple(r["trajectory"]) == ("OFF", "OFF"))
            for row in group:
                audit = row.get("rank_audit", [])
                candidate_stats.append(
                    {
                        "task_gain": _score(row) - _score(off),
                        "max_rank": max((a.get("numerical_rank", 0) for a in audit), default=0),
                        "update_norm": row.get("visible", {}).get("state_total_norm", 0.0),
                    }
                )
        rank_relationships[f"{resolution}/{split}"] = {
            name: _prediction(
                [r[name] for r in candidate_stats], [r["task_gain"] for r in candidate_stats]
            )
            for name in ("max_rank", "update_norm")
        }
    reports = {
        "oracle_analysis": opportunity,
        "selector_analysis": selectors,
        "robustness_analysis": robustness,
        "bootstrap_results": {
            "seed": SEED,
            "replicates": N_BOOTSTRAP,
            "primary_unit": "sequence",
            "supplemental_unit": "pair",
            "comparisons": bootstrap,
            "multiplicity": "Descriptive percentile intervals; no multiplicity correction",
        },
        "signal_analysis": signals,
        "rank_analysis": {
            "records": audits,
            "candidate_gain_relationships": rank_relationships,
            "scope": "Numerical rank and correlation are descriptive; no causal rank claim.",
            "max_numerical_rank": max((a.get("numerical_rank", 0) for a in audits), default=None),
            "max_span_residual": max((a.get("span_residual", 0) for a in audits), default=None),
        },
    }
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    scatter = []
    visible_table = []
    feature_names = sorted({name for r in rows for name in r.get("visible", {})})
    for key, group in groups.items():
        off = next(r for r in group if tuple(r["trajectory"]) == ("OFF", "OFF"))
        for row in group:
            visible_table.append(
                {
                    "resolution": key[0],
                    "split": key[1],
                    "sequence": key[2],
                    "target_slot": key[3],
                    "trajectory": "/".join(row["trajectory"]),
                    **{name: row.get("visible", {}).get(name) for name in feature_names},
                }
            )
            scatter.append(
                {
                    "resolution": key[0],
                    "split": key[1],
                    "sequence": key[2],
                    "target_slot": key[3],
                    "trajectory": "/".join(row["trajectory"]),
                    "task_gain": _score(row) - _score(off),
                    "rgb_decrease": row.get("visible", {}).get("rgb_decrease"),
                    "probe_decrease": row.get("visible", {}).get("probe_decrease"),
                    "visible_json": json.dumps(row.get("visible", {}), sort_keys=True),
                }
            )
    _write_csv(out / "visible_features.csv", visible_table)
    _write_csv(out / "signal_scatter.csv", scatter)
    _write_csv(out / "per_pair_results.csv", records)
    _write_csv(out / "per_sequence_results.csv", sequences)
    for name, data in reports.items():
        (out / f"{name}.json").write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    return reports
