"""Raw-only scientific plots and interpretation of privileged support diagnostics."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mcss.mechanism_pilot.statistics import paired_scene_bootstrap
from mcss.mechanism_pilot.support_redesign_statistics import analyze

DESIGNS = ["R0", "ORACLE_VOLUME", "ORACLE_SUPPORT", "ORACLE_VOLUME_SUPPORT"]
LABELS = [
    "R0: GT-free baseline",
    "Oracle volume",
    "Oracle allocation",
    "Oracle volume + allocation",
]
COLORS = ["#565b66", "#3274a1", "#cc7a2f", "#8865aa"]


def write(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def run(root):
    root = Path(root)
    raw_path = root / "raw/oracle_run/results.json"
    raw = json.loads(raw_path.read_text())
    stats = analyze(raw)
    assert stats == json.loads((root / "bootstrap_results.json").read_text())
    figures, audit = root / "figures", root / "audit"
    figures.mkdir(exist_ok=True)
    audit.mkdir(exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 10,
            "svg.fonttype": "none",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 130,
        }
    )
    strata = [
        ("exposed", stats["cohorts"]["exposed"]),
        ("new_dev", stats["cohorts"]["new_dev"]),
        ("pooled", stats["pooled"]),
    ]

    def save(fig, name):
        fig.savefig(figures / f"{name}.png", bbox_inches="tight", dpi=180)
        fig.savefig(figures / f"{name}.svg", bbox_inches="tight")
        plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True)
    for ax, (name, group) in zip(axes, strata, strict=True):
        for i, design in enumerate(DESIGNS):
            value = group["designs"][design]["full_context_gain"]
            mean = value["mean"]
            low, high = value["ci95"]
            ax.errorbar(
                mean,
                i,
                xerr=[[mean - low], [high - mean]],
                fmt="s" if i == 0 else "o",
                color=COLORS[i],
                capsize=4,
                markersize=7,
                markerfacecolor="white" if i else COLORS[i],
            )
        ax.axvline(0, color="#333333", linewidth=0.8)
        ax.axhline(0.5, color="#999999", linewidth=0.7, linestyle="--")
        ax.set_title(f"{name.replace('_', ' ')} (n={group['n_scenes']})")
        ax.set_xlabel("Anchor - mean(A,B) depth AbsRel\nPositive favors full context")
        ax.grid(axis="x", alpha=0.18)
    axes[0].set_yticks(range(4), LABELS)
    axes[0].invert_yaxis()
    fig.suptitle("Scene bootstrap 95% intervals: baseline separated from GT-privileged diagnostics")
    fig.tight_layout()
    save(fig, "full_context_gain_intervals")

    pooled = stats["pooled"]["designs"]
    scene_ids = stats["pooled"]["scene_ids"]
    matrix = np.array(
        [[pooled[d]["full_context_gain"]["per_scene"][s] for d in DESIGNS] for s in scene_ids]
    )
    fig, ax = plt.subplots(figsize=(9, 9))
    limit = np.abs(matrix).max()
    im = ax.imshow(matrix, cmap="RdBu", vmin=-limit, vmax=limit, aspect="auto")
    ax.set_xticks(
        range(4), ["R0 baseline", "Oracle volume", "Oracle allocation", "Oracle combined"]
    )
    ax.set_yticks(range(len(scene_ids)), scene_ids)
    ax.axvline(0.5, color="black", linewidth=2)
    ax.axhline(6.5, color="black", linewidth=1)
    for i in range(len(scene_ids)):
        for j in range(4):
            ax.text(
                j,
                i,
                f"{matrix[i, j]:+.3f}",
                ha="center",
                va="center",
                fontsize=9,
                color="white" if abs(matrix[i, j]) > 0.65 * limit else "black",
            )
    fig.colorbar(im, ax=ax, label="Anchor - mean(A,B) AbsRel")
    ax.set_title(
        "Per-scene full-context gain\nRight three columns require privileged "
        "GT; not deployment results"
    )
    fig.tight_layout()
    save(fig, "per_scene_full_context_gain")

    exposed = set(stats["cohorts"]["exposed"]["scene_ids"])
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True)
    for ax, design, color in zip(axes, DESIGNS[1:], COLORS[1:], strict=True):
        d = pooled[design]
        corr = d["support_gain_correlation"]["by_support_metric"]["supported_surface_fraction"]
        x = corr["support_change_per_scene"]
        y = d["change_full_context_gain_vs_R0"]["per_scene"]
        for label, ids, marker in [
            ("Previously exposed", exposed, "o"),
            ("New redesign dev", set(scene_ids) - exposed, "^"),
        ]:
            ordered = sorted(ids)
            ax.scatter(
                [x[s] for s in ordered],
                [y[s] for s in ordered],
                marker=marker,
                color=color,
                s=45,
                alpha=0.8,
                label=label,
            )
        ax.axhline(0, color="#555555", linewidth=0.7)
        ax.axvline(0, color="#555555", linewidth=0.7)
        rho = corr["pearson"]
        ax.set_title(
            f"{LABELS[DESIGNS.index(design)]}\nPearson r={rho:.3f}"
            if rho is not None
            else LABELS[DESIGNS.index(design)]
        )
        ax.set_xlabel("Change in supported GT-surface fraction vs R0")
        ax.grid(alpha=0.15)
    axes[0].set_ylabel("Change in full-context gain vs R0")
    axes[-1].legend(fontsize=8)
    fig.suptitle("Descriptive association only; no causal or deployable-support inference")
    fig.tight_layout()
    save(fig, "support_vs_task_scatter")

    baseline = pooled["R0"]["methods"]
    detail = {}
    for design in DESIGNS:
        d = pooled[design]
        methods = d["methods"]

        def scene_metric(method, sid, source=methods):
            return source[method]["depth_absrel"]["per_scene"][sid]

        full_improvement = {
            s: (scene_metric("A", s, baseline) + scene_metric("B", s, baseline)) / 2
            - (scene_metric("A", s) + scene_metric("B", s)) / 2
            for s in scene_ids
        }
        anchor_worsening = {
            s: scene_metric("anchor", s) - scene_metric("anchor", s, baseline) for s in scene_ids
        }
        full_ci = paired_scene_bootstrap(full_improvement)
        anchor_ci = paired_scene_bootstrap(anchor_worsening)
        gain_change = d["change_full_context_gain_vs_R0"]["mean"]
        detail[design] = {
            "full_context_gain": d["full_context_gain"],
            "change_full_context_gain_vs_R0": d["change_full_context_gain_vs_R0"],
            "wrong_scene_damage": d["wrong_scene_damage"],
            "absolute_full_context_AbsRel": (
                methods["A"]["depth_absrel"]["mean"] + methods["B"]["depth_absrel"]["mean"]
            )
            / 2,
            "absolute_anchor_AbsRel": methods["anchor"]["depth_absrel"]["mean"],
            "absolute_full_context_improvement_vs_R0": full_ci,
            "anchor_worsening_vs_R0": anchor_ci,
            "anchor_worsening_share_of_gain_change": anchor_ci["mean"] / gain_change
            if gain_change
            else None,
            "AB_geometry": d["AB_geometry"],
            "oracle_adequacy": d["oracle_adequacy"],
            "support_gain_correlation": d["support_gain_correlation"],
            "loso_gain_mean_range": [
                min(d["full_context_gain"]["loso"].values()),
                max(d["full_context_gain"]["loso"].values()),
            ],
        }
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    x = np.arange(4)
    full = [detail[d]["absolute_full_context_AbsRel"] for d in DESIGNS]
    anchors = [detail[d]["absolute_anchor_AbsRel"] for d in DESIGNS]
    axes[0].plot(x, full, "o-", label="mean(A,B)", color="#3274a1")
    axes[0].plot(x, anchors, "s--", label="Anchor", color="#b27233")
    axes[0].set_ylabel("Absolute depth AbsRel (lower is better)")
    axes[0].set_xticks(x, ["R0", "Oracle\nvolume", "Oracle\nallocation", "Oracle\ncombined"])
    axes[0].axvline(0.5, color="#777777", linestyle=":")
    axes[0].legend()
    for i, d in enumerate(DESIGNS):
        q = detail[d]["absolute_full_context_improvement_vs_R0"]
        lo, hi = q["ci95"]
        mean = q["mean"]
        axes[1].errorbar(
            mean,
            i,
            xerr=[[mean - lo], [hi - mean]],
            fmt="s" if i == 0 else "o",
            color=COLORS[i],
            capsize=4,
        )
    axes[1].set_yticks(x, LABELS)
    axes[1].invert_yaxis()
    axes[1].axvline(0, color="#555555", linewidth=0.8)
    axes[1].axhline(0.5, color="#777777", linestyle=":")
    axes[1].set_xlabel("Absolute full-context improvement vs R0\nPositive = lower mean(A,B) AbsRel")
    fig.suptitle("Absolute task performance: positive context gain can include anchor worsening")
    fig.tight_layout()
    save(fig, "absolute_task_and_anchor_comparison")

    report = {
        "status": "PASS_RAW_RECOMPUTATION",
        "SUPPORT_BOTTLENECK_STATUS": "INCONCLUSIVE",
        "raw_sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "strata": {
            name: {
                "n_scenes": group["n_scenes"],
                "designs": {
                    d: {
                        "full_context_gain": group["designs"][d]["full_context_gain"],
                        "wrong_scene_damage": group["designs"][d]["wrong_scene_damage"],
                    }
                    for d in DESIGNS
                },
            }
            for name, group in strata
        },
        "pooled_diagnostics": detail,
        "positive_evidence": (
            "Oracle volume alone restores positive full-context gain with pooled "
            "CI above zero; new-dev stratum also has positive CI. This supports a "
            "volume contribution and must not be hidden by the combined stop."
        ),
        "stopping_reason": (
            "Predeclared combined intervention has CI crossing zero and "
            "supported-surface coverage below adequacy. Do not switch to the "
            "better volume-only gate after seeing scores. GT-free real-scene runs "
            "and matched retraining remain NOT_RUN_ORACLE_STOP."
        ),
        "not_an_ideal_support_failure": (
            "Combined mean two-view support is high but actual GT-surface "
            "neighborhood coverage remains below the frozen minimum; no "
            "perfect-support or universal representation impossibility conclusion."
        ),
        "single_factor_causal_attribution": (
            "NOT_IDENTIFIED: bounds also change voxel resolution, normalized "
            "feature/spatial distributions, opacity/path lengths, and anchor prior "
            "rendering. Allocation changes learned spatial distribution and varies "
            "by context. Frozen weights cannot separate these from learned "
            "capacity."
        ),
        "holdout_interpretation": (
            "Previously exposed and new redesign-dev strata are development "
            "diagnostics. No final-holdout inference or claim is supported by "
            "these plots."
        ),
        "DYNAMIC_TTT_RUN": False,
    }
    write(audit / "scientific_interpretation.json", report)
    lines = [
        "# Scientific interpretation: support redesign diagnostics",
        "",
        (
            "**Volume-only supplies positive evidence; the combined predeclared "
            "experiment remains inconclusive.**"
        ),
        (
            "These are GT-privileged frozen-carrier interventions, not deployable "
            "methods or mathematical upper bounds."
        ),
        "",
        "| Intervention | Full-context gain | 95% scene CI | Positive / tie / negative scenes |",
        "|---|---:|---|---|",
    ]
    for d in DESIGNS:
        g = detail[d]["full_context_gain"]
        c = g["improved_tied_worse"]
        lines.append(
            f"| {d} | {g['mean']:+.6f} | [{g['ci95'][0]:+.6f}, {g['ci95'][1]:+.6f}] "
            f"| {c['improved']}/{c['tied']}/{c['worse']} |"
        )
    lines += [
        "",
        "## Positive volume evidence and its limits",
        "",
        report["positive_evidence"],
        "",
        (
            "The previously exposed stratum is less stable than new redesign dev; "
            "both remain development evidence."
        ),
        "",
        (
            "| Intervention | Absolute full-context AbsRel | Absolute improvement "
            "vs R0 [CI] | Anchor worsening vs R0 [CI] |"
        ),
        "|---|---:|---|---:|",
    ]
    for d in DESIGNS:
        value = detail[d]
        q = value["absolute_full_context_improvement_vs_R0"]
        lines.append(
            f"| {d} | {value['absolute_full_context_AbsRel']:.6f} | "
            f"{q['mean']:+.6f} [{q['ci95'][0]:+.6f}, {q['ci95'][1]:+.6f}] | "
            f"{value['anchor_worsening_vs_R0']['mean']:+.6f} "
            f"[{value['anchor_worsening_vs_R0']['ci95'][0]:+.6f}, "
            f"{value['anchor_worsening_vs_R0']['ci95'][1]:+.6f}] |"
        )
    volume = detail["ORACLE_VOLUME"]
    combined = detail["ORACLE_VOLUME_SUPPORT"]
    lines += [
        "",
        (
            f"For volume-only, "
            f"{volume['anchor_worsening_share_of_gain_change']:.1%} of the change "
            f"in full-context gain comes from a worse anchor after changing bounds. "
            f"The remainder is an absolute full-context improvement. Therefore "
            f"positive context gain is not the same as equally large absolute task "
            f"recovery."
        ),
        "",
        "## Concentration, sensitivity and association",
        "",
    ]
    for d in DESIGNS:
        value = detail[d]
        c = value["full_context_gain"]["concentration"]["positive"]
        top = (
            "no positive contribution"
            if c["top1_fraction"] is None
            else (
                "top-1/top-3 shares of positive gains "
                f"{c['top1_fraction']:.1%}/{c['top3_fraction']:.1%}"
            )
        )
        lines.append(
            f"- {d}: {top}; leave-one-scene-out mean range {value['loso_gain_mean_range']}."
        )
    lines += [
        "",
        (
            "Positive-share concentration uses the sum of positive scene gains, "
            "not the signed net denominator. Full per-scene and stratum "
            "statistics, both concentration definitions, and LOSO values are "
            "retained in JSON. Scatter correlations are descriptive and do not "
            "identify causality."
        ),
        "",
        "## Why the stop is inconclusive",
        "",
        (
            f"Combined two-view candidate fraction is "
            f"{combined['AB_geometry']['two_view_candidate_fraction']['mean']:.2%}, "
            f"but supported GT-surface fraction is "
            f"{combined['AB_geometry']['supported_surface_fraction']['mean']:.2%}, "
            f"below 75%. Frustum support is not occlusion visibility. Finite 8^3 "
            f"geometry and 128 candidates do not guarantee ideal surface support."
        ),
        "",
        report["stopping_reason"],
        "",
        report["single_factor_causal_attribution"],
        "",
        (
            "It is unsupported to conclude either “support is useless” or “the "
            "network is the sole cause.” Volume contributes, allocation alone "
            "worsens the frozen model, and "
            "coverage/training-distribution/representation remain coupled. Final "
            "holdout stays closed; no Dynamic TTT is run."
        ),
    ]
    (audit / "scientific_interpretation.md").write_text("\n".join(lines) + "\n")
    print(
        json.dumps(
            {
                "status": report["status"],
                "pooled_full_context_absolute_improvement": {
                    d: detail[d]["absolute_full_context_improvement_vs_R0"] for d in DESIGNS
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    run(parser.parse_args().root)
