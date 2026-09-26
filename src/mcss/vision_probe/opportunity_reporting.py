"""Regenerate Chinese scientific reports and figures from saved analysis artifacts."""

# Long report prose is kept as complete translation units.
# ruff: noqa: E501
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

METHODS = {
    "OFF": "OFF/OFF",
    "best_fixed": "Discovery-fixed",
    "rgb_selector": "RGB selector",
    "visible_selector": "Visible selector",
    "best_constant_oracle": "Constant oracle",
    "trajectory_oracle": "Dynamic oracle",
}


def _json(path, default=None):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def _csv(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _n(value, digits=6):
    return "NA" if value is None else f"{float(value):.{digits}f}"


def _ci(distribution):
    result = distribution["sequence_bootstrap"]
    return f"{_n(result['mean'])} [{_n(result['ci95'][0])}, {_n(result['ci95'][1])}]"


def _mean(rows, field):
    return float(np.mean([float(r[field]) for r in rows])) if rows else None


def _plots(report, sequences, scatter):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = report / "figures"
    output.mkdir(exist_ok=True)
    with plt.rc_context({"font.size": 9, "figure.dpi": 140, "savefig.bbox": "tight"}):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for ax, resolution in zip(axes, (224, 448), strict=True):
            subset = [
                r
                for r in scatter
                if r["split"] == "validation" and int(r["resolution"]) == resolution
            ]
            ax.scatter(
                [float(r["rgb_decrease"]) for r in subset],
                [100 * float(r["task_gain"]) for r in subset],
                s=9,
                alpha=0.3,
            )
            ax.axhline(0, color="black", lw=0.7)
            ax.axvline(0, color="black", lw=0.7)
            ax.set(
                title=f"{resolution}: all validation candidates",
                xlabel="Observed RGB MSE decrease",
                ylabel="Task J change (percentage points)",
            )
        fig.savefig(output / "rgb_task_scatter.png")
        plt.close(fig)
        selected = [
            r for r in sequences if r["split"] == "validation" and int(r["resolution"]) == 448
        ]
        fig, ax = plt.subplots(figsize=(9, 4))
        means = [_mean([r for r in selected if r["method"] == m], "J") for m in METHODS]
        ax.bar(
            list(METHODS.values()),
            means,
            color=["#666666", "#8b8b8b", "#ca8a04", "#2563eb", "#7c3aed", "#059669"],
        )
        ax.set(ylabel="Sequence-macro pixel J", title="448 validation: oracles are analysis-only")
        ax.tick_params(axis="x", rotation=20)
        fig.savefig(output / "method_comparison.png")
        plt.close(fig)
        seqs = sorted({r["sequence"] for r in selected})
        fig, ax = plt.subplots(figsize=(11, 5))
        positions = np.arange(len(seqs))
        for offset, method, label in (
            (-0.18, "trajectory_oracle", "Dynamic oracle"),
            (0.18, "visible_selector", "Visible selector"),
        ):
            vals = [
                100
                * float(
                    next(
                        r["gain_off"]
                        for r in selected
                        if r["sequence"] == s and r["method"] == method
                    )
                )
                for s in seqs
            ]
            ax.bar(positions + offset, vals, 0.35, label=label)
        ax.axhline(0, color="black", lw=0.7)
        ax.set_xticks(positions, seqs, rotation=65, ha="right")
        ax.set(
            ylabel="J change vs OFF (percentage points)",
            title="448 validation: all sequences, both target slots",
        )
        ax.legend()
        fig.savefig(output / "per_sequence_gain.png")
        plt.close(fig)


def build_report(report_dir, work_dir):
    """Write README, STATUS, terminal summary, and plots; never evaluate or fit."""
    report, work = Path(report_dir), Path(work_dir)
    oracle = _json(report / "oracle_analysis.json")
    selector = _json(report / "selector_analysis.json")
    robustness = _json(report / "robustness_analysis.json")
    bootstrap = _json(report / "bootstrap_results.json")
    signals = _json(report / "signal_analysis.json")
    ranks = _json(report / "rank_analysis.json")
    config = _json(report / "config.json")
    sequences, pairs = (
        _csv(report / "per_sequence_results.csv"),
        _csv(report / "per_pair_results.csv"),
    )
    scatter = _csv(report / "signal_scatter.csv")
    primary = "448/validation"
    if primary not in oracle:
        raise ValueError("Completed 448 validation analysis is required for final reporting")
    op = oracle[primary]
    gain, extra = op["oracle_minus_off"], op["trajectory_minus_constant"]
    vis, rgb = selector[f"{primary}/visible_selector"], selector[f"{primary}/rgb_selector"]
    comparisons = bootstrap["comparisons"]
    subset = [r for r in sequences if r["split"] == "validation" and int(r["resolution"]) == 448]
    means = {
        m: {f: _mean([r for r in subset if r["method"] == m], f) for f in ("J", "F", "JF")}
        for m in METHODS
    }
    vg = comparisons[f"{primary}/visible_selector"]
    rg = comparisons[f"{primary}/rgb_selector"]
    integrity = _json(report / "integrity.json", {})
    tests = _json(report / "tests.json", {})
    final_integrity = integrity.get(
        "final_integrity", integrity.get("FINAL_INTEGRITY", integrity.get("status", "PENDING"))
    )
    if final_integrity not in ("PASS", "FAIL"):
        final_integrity = "PENDING"
    leakage = integrity.get(
        "validation_gt_used_for_selection", integrity.get("VALIDATION_GT_USED_FOR_SELECTION")
    )
    official = integrity.get(
        "davis_official_val_touched", integrity.get("DAVIS_OFFICIAL_VAL_TOUCHED")
    )

    # No absence-of-evidence inference: the independent integrity audit owns these claims.
    def flag(value):
        if value is False or value == "false":
            return "false"
        if value is True or value == "true":
            return "true"
        return "PENDING"

    tests_status = (
        ("PASS" if tests.get("exit_code") == 0 else "FAIL") if "exit_code" in tests else "PENDING"
    )
    status = {
        "OPPORTUNITY_STATUS": op.get("H1", "NOT_ESTABLISHED"),
        "VISIBLE_SIGNAL_STATUS": vis.get("H2", "NOT_ESTABLISHED"),
        "PRIMARY_RESOLUTION": 448,
        "N_VALIDATION_SEQUENCES": len({r["sequence"] for r in subset}),
        "N_VALIDATION_PAIRS": sum(
            r["method"] == "OFF" and r["split"] == "validation" and int(r["resolution"]) == 448
            for r in pairs
        ),
        "OFF_J": means["OFF"]["J"],
        "BEST_FIXED_J": means["best_fixed"]["J"],
        "CONSTANT_ORACLE_J": means["best_constant_oracle"]["J"],
        "DYNAMIC_ORACLE_J": means["trajectory_oracle"]["J"],
        "DYNAMIC_ORACLE_MINUS_OFF": gain["sequence_bootstrap"]["mean"],
        "DYNAMIC_ORACLE_MINUS_CONSTANT": extra["sequence_bootstrap"]["mean"],
        "TOP1_SEQUENCE_POSITIVE_GAIN_SHARE": gain["top1_positive_gain_share"],
        "LEAVE_ONE_SEQUENCE_OUT_MIN_GAIN": gain["loso_min"],
        "RGB_SELECTOR_J": means["rgb_selector"]["J"],
        "VISIBLE_SELECTOR_J": means["visible_selector"]["J"],
        "VISIBLE_SELECTOR_MINUS_OFF": vg["gain_off"]["sequence_bootstrap"]["mean"],
        "VISIBLE_SELECTOR_MINUS_FIXED": vg["gain_fixed"]["sequence_bootstrap"]["mean"],
        "VISIBLE_SELECTOR_ORACLE_CAPTURE": vis["opportunity_capture"],
        "HARMFUL_WRITE_RATE": vis["harmful_among_writes"],
        "VALIDATION_GT_USED_FOR_SELECTION": flag(leakage),
        "DAVIS_OFFICIAL_VAL_TOUCHED": flag(official),
        "TESTS": tests_status,
        "FINAL_INTEGRITY": final_integrity,
    }

    def render(value):
        return "NA" if value is None else str(value)

    (report / "STATUS.md").write_text(
        "\n\n".join(f"{k}=\n{render(v)}" for k, v in status.items()) + "\n", encoding="utf-8"
    )
    legacy = oracle.get("224/validation", {}).get("oracle_minus_off", {})
    legacy_gain = legacy.get("sequence_bootstrap", {}).get("mean")
    available_signals = signals["features"].get(primary, {})
    best_signals = sorted(
        ((n, v) for n, v in available_signals.items() if v.get("spearman") is not None),
        key=lambda item: -abs(item[1]["spearman"]),
    )[:5]
    signal_text = "; ".join(
        f"{n}: ρ={_n(v['spearman'])}, r={_n(v['pearson'])}, AUROC={_n(v['auroc_positive_gain'])}, AP={_n(v['average_precision_positive_gain'])}"
        for n, v in best_signals
    )
    rgb_signal = available_signals.get("rgb_decrease", {})
    shares = gain["sequence_positive_gain_share"]
    largest = max(shares, key=shares.get) if shares else "NA"
    oracle_pairs = [
        r
        for r in pairs
        if r["method"] == "trajectory_oracle"
        and r["split"] == "validation"
        and int(r["resolution"]) == 448
    ]
    selected_nonconstant = sum(len(set(r["trajectory"].split("/"))) > 1 for r in oracle_pairs)
    size_text = "; ".join(
        f"{k.rsplit('/', 1)[-1]}: n_objects={len(v['objects'])}, ΔJ={_n(v['sequence_macro_gain'])}"
        for k, v in robustness[primary]["object_size"].items()
    )
    if gain["positive_sequences"] >= 2 and gain["sequence_bootstrap"]["mean"] > 0:
        bottleneck = "当前候选家族至少存在局部机会；若选择器未获正净收益，证据更指向机会识别不足。极小收益仍可能同时反映更新家族偏弱，不能仅凭相关性定因。"
    else:
        bottleneck = "当前更新家族的任务机会尚未建立，应先核验机会量级和评价量化效应，不能直接把问题归因于选择器。"
    raw_path = report / "raw_results.jsonl"
    raw_count = (
        sum(bool(line.strip()) for line in raw_path.open(encoding="utf-8"))
        if raw_path.exists()
        else 0
    )
    answers = [
        f"高分辨率评价是否改变旧结论？224 的 oracle−OFF 为 {_n(legacy_gain)}，448 为 {_ci(gain)}。两种输入都重新提取 DINO patch；448 是真实 32×32 特征网格，并非插值 16×16 结果。这里只能直接比较本轮重评，不能把旧 token IoU 与当前 pixel J 当同一指标。",
        f"有益写入是否跨样本存在？448 共 {gain['improved_pairs']} 个正收益 pair、{gain['positive_sequences']} 个正收益 sequence；{gain['tied_pairs']} 个 pair 持平、{gain['worse_pairs']} 个更差。H1={status['OPPORTUNITY_STATUS']}。oracle 含 OFF，故非负本身不构成有效性证明。",
        f"是否被单个序列支配？最大贡献来自 {largest}，top-1={_n(gain['top1_positive_gain_share'])}，top-3={_n(gain['top3_positive_gain_share'])}；scooter-board 占 {_n(shares.get('scooter-board'))}。去掉任意一个 sequence 的最小 oracle gain={_n(gain['loso_min'])}。所有 sequence 均保留。",
        f"动态顺序是否优于 constant oracle？dynamic−constant={_ci(extra)}；正收益 pair={extra['improved_pairs']}，独立 sequence={extra['positive_sequences']}。零增益不能称为需要动态顺序。",
        f"多少样本需要非恒定轨迹？严格超过全部四种 constant 轨迹的 pair={extra['improved_pairs']}；按固定 tie-break 选出非恒定 oracle 的 pair={selected_nonconstant}。后者可能包含并列最优，不能替代前者。",
        f"RGB 与任务收益是否对齐？全部448 validation候选的 RGB decrease 与 ΔJ：Pearson={_n(rgb_signal.get('pearson'))}、Spearman={_n(rgb_signal.get('spearman'))}。散点保留全部候选；相关性是描述性结果，同一图像候选不独立，不代表因果证据。",
        f"哪些信号最相关？按 validation 的 |Spearman| 回顾性排序：{signal_text}。这是离线诊断排序，不用于重选特征、方向或阈值；完整 discovery/validation 对比见 signal_analysis.json。",
        f"RGB-only 是否有效？J={_n(means['rgb_selector']['J'])}，相对 OFF={_ci(rg['gain_off'])}；write rate={_n(rgb['write_rate'])}，有害写入占已写入 pair 的比例={_n(rgb['harmful_among_writes'])}。阈值仅在 discovery 调整，其 tuning reward 不是独立泛化估计。",
        f"visible selector 是否有实际收益？J={_n(means['visible_selector']['J'])}；相对 OFF={_ci(vg['gain_off'])}，相对 discovery-fixed={_ci(vg['gain_fixed'])}；H2={status['VISIBLE_SIGNAL_STATUS']}。只按锁定分析标准判断，不因为捕获了个别正例就声称整体有效。",
        f"捕获多少机会？positive capture={_n(vis['opportunity_capture'])}，净 ΔJ={_n(vis['net_gain'])}；写入率={_n(vis['write_rate'])}，有害写入率（分母为写入 pair）={_n(vis['harmful_among_writes'])}，precision={_n(vis['beneficial_write_precision'])}，recall={_n(vis['beneficial_write_recall'])}。正收益 capture 不扣除损害，必须结合净收益看。",
        f"当前瓶颈？{bottleneck} 尺度分桶审计：{size_text}；非 tiny 对象宏平均 gain={_n(op.get('non_tiny_gain'))}。这不足以排除其他表示或读出限制。",
        "下一阶段最小实验？保持当前原始结果与阈值不变。在新的预注册实验中，用未曾观察的新独立序列先复核机会量级；若机会稳定但当前选择器失败，只在 discovery 比较一个预先限定的信号/写入子空间对照，再锁定一次评估。不得把本轮 validation 继续用于调参，也不在本轮开启 reserve 或 DAVIS 官方 validation。",
    ]
    lines = [
        "# Opportunity 与部署可见信号实验",
        "",
        f"H1={status['OPPORTUNITY_STATUS']}；H2={status['VISIBLE_SIGNAL_STATUS']}。完整性={final_integrity}；测试={tests_status}。",
        "",
        "主指标为原始像素空间的对象平均 J，再对两个 target slots 和 sequence 分层等权平均。F 是二值边界近似次指标，JF=(J+F)/2 仅为该近似的算术汇总；这里没有 DAVIS 官方 J&F 成绩。",
        "",
        "本轮 validation 重用了历史内部 validation 的16个序列，属于新协议下锁定重评，不是全新独立 holdout；本轮正式读取发生在科学代码、配置与selector锁定之后。reserve 与 DAVIS 官方 validation 应保持封存，审计结果见 STATUS.md / integrity.json。",
        "",
        f"原始 candidate 行数={raw_count}；448 validation 序列={status['N_VALIDATION_SEQUENCES']}、pairs={status['N_VALIDATION_PAIRS']}；每pair完整16条two-step轨迹。所有target slots参与聚合。",
        "",
        "## 十二个问题",
        "",
    ]
    lines.extend(f"{i}. {answer}\n" for i, answer in enumerate(answers, 1))
    lines.extend(
        [
            "## 统一方法比较",
            "",
            "| 方法 | J | F（近似） | JF（近似） | ΔJ vs OFF：sequence bootstrap 95% CI |",
            "|---|---:|---:|---:|---|",
        ]
    )
    for method, label in METHODS.items():
        lines.append(
            f"| {label} | {_n(means[method]['J'])} | {_n(means[method]['F'])} | {_n(means[method]['JF'])} | {_ci(comparisons[f'{primary}/{method}']['gain_off'])} |"
        )
    lines.extend(
        [
            "",
            "区间用 sequence 为单位配对重采样，保留每个序列全部 pairs；10,000次、固定seed。pair bootstrap仅作补充，候选/特征检验没有多重比较校正。J/F/JF与STATUS数值采用0..1比例；绝对J差值乘100才是百分点。",
            "",
            "部署候选选择也有成本：当前实现每pair试算全部16条two-step轨迹，共32次proposal调用、64次A/B增量计算，并执行16个候选可见RGB读出与特征对应计算。这里未做匹配计算预算的效率对照，不能宣称动态选择更省计算。",
            "",
            "## 分布、量化与机制审计",
            "",
            f"Oracle ΔJ：pair mean={_n(gain['pair_mean'])}、median={_n(gain['pair_median'])}、std={_n(gain['pair_std'])}、min/max={_n(gain['pair_min'])}/{_n(gain['pair_max'])}。≥0.5/1/2百分点pair计数：{gain['threshold_pair_counts']}。轨迹熵={_n(op['tie_aware_oracle']['entropy_nats'])} nats；并列oracle按分数计数，完整频数/sequence偏好/size偏好保存在 oracle_analysis.json。",
            "",
            "令冻结RGB decoder为 D∈R^(64×3)，support RGB residual为 R∈R^(N×3)，归一化激活为 X̃∈R^(N×64)。单步增量满足 ΔW = η/(N‖D‖²_F) · D Rᵀ X̃（实现含数值clamp与标量范数裁剪）；因此 rank(ΔW)≤3，列空间属于col(D)。固定D的多步相加仍留在同一列空间。A/B组合后整体特征变换需区别于单步矩阵；该限制不证明它就是任务收益瓶颈。",
            "",
            f"数值审计记录={len(ranks.get('records', []))}；最大numerical rank={ranks.get('max_numerical_rank')}；最大decoder-span residual={ranks.get('max_span_residual', 0):.3e}。各步A/B奇异值、秩与span记录及rank/norm对gain相关性见 rank_analysis.json；阈值abs={config.get('rank_absolute_tolerance')}、relative={config.get('rank_relative_tolerance')}。",
            "",
            "![RGB-task scatter](figures/rgb_task_scatter.png)",
            "",
            "![Method comparison](figures/method_comparison.png)",
            "",
            "![Sequence contributions](figures/per_sequence_gain.png)",
            "",
            "## 可复现性与审计边界",
            "",
            "config.json、split_manifest.json、preregistration.json与锁定artifact记录配置和来源。fit只拟合PCA/whitening/RGB decoder；ridge只在discovery拟合，标准化位于LOSO训练折内。validation选择先于target mask读取；是否满足此约束以独立integrity.json为准，缺失时明确PENDING。",
            "",
            f"工作产物目录：`{work}`。完整测试结果：`tests.json`（{tests.get('summary', 'PENDING')}）。完整从头运行的 `commands.sh` 仅用于新的干净工作副本，会拒绝覆盖已有实验。当前目录请运行 `regenerate.sh` 从封存raw重新生成分析，再运行独立 `scripts/report_opportunity_probe.py --report-dir ... --work-dir ...` 生成本报告与图片。报告阶段不读取数据集、不重新训练选择器、不使用validation调整任何配置。",
            "",
            "后处理与独立审计各发生一次原生进程崩溃（exit139，原因未确诊）；启用faulthandler及单线程数值库后重试成功，全部分析逐字节一致。科学代码、raw、阈值未变，validation未重跑。详情见postprocessing_incident.json。",
            "",
        ]
    )
    (report / "README.md").write_text("\n".join(lines), encoding="utf-8")
    terminal = {
        "PRIMARY_METRIC": "J (pixel-space; F secondary approximation)",
        "PRIMARY_RESOLUTION": 448,
        "OPPORTUNITY_STATUS": status["OPPORTUNITY_STATUS"],
        "VISIBLE_SIGNAL_STATUS": status["VISIBLE_SIGNAL_STATUS"],
        **{
            name: means[method]["J"]
            for name, method in (
                ("OFF", "OFF"),
                ("BEST_FIXED", "best_fixed"),
                ("CONSTANT_ORACLE", "best_constant_oracle"),
                ("DYNAMIC_ORACLE", "trajectory_oracle"),
                ("RGB_SELECTOR", "rgb_selector"),
                ("VISIBLE_SELECTOR", "visible_selector"),
            )
        },
        "DYN_ORACLE_MINUS_OFF": status["DYNAMIC_ORACLE_MINUS_OFF"],
        "DYN_ORACLE_MINUS_CONSTANT": status["DYNAMIC_ORACLE_MINUS_CONSTANT"],
        "VISIBLE_SELECTOR_MINUS_OFF": status["VISIBLE_SELECTOR_MINUS_OFF"],
        "VISIBLE_SELECTOR_MINUS_FIXED": status["VISIBLE_SELECTOR_MINUS_FIXED"],
        "IMPROVED_SEQUENCES_ORACLE": gain["positive_sequences"],
        "IMPROVED_SEQUENCES_SELECTOR": vg["gain_off"]["positive_sequences"],
        "TOP1_GAIN_SHARE": gain["top1_positive_gain_share"],
        "LOSO_MIN_ORACLE_GAIN": gain["loso_min"],
        "ORACLE_CAPTURE": vis["opportunity_capture"],
        "HARMFUL_WRITE_RATE": vis["harmful_among_writes"],
        "VALIDATION_LEAKAGE": flag(leakage),
        "DAVIS_OFFICIAL_VAL_TOUCHED": flag(official),
        "TESTS": tests_status,
        "FINAL_INTEGRITY": final_integrity,
        "REPORT_DIR": str(report.resolve()),
    }
    banner = "=" * 60
    summary = (
        f"{banner}\nEXP-2D OPPORTUNITY + VISIBLE SELECTOR FINAL\n{banner}\n\n"
        + "\n".join(f"{k}={render(v)}" for k, v in terminal.items())
        + f"\n\n{banner}\n"
    )
    (report / "terminal_summary.txt").write_text(summary, encoding="utf-8")
    _plots(report, sequences, scatter)
    return {"status": status, "terminal_summary": summary}
