"""Render the blocked V2 report from archived analysis; no dataset access."""


# Long literal paragraphs preserve the rendered report text.
# ruff: noqa: E501
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-dir", type=Path, required=True)
    args = parser.parse_args()
    report = args.report_dir.resolve()

    def read(name):
        return json.loads((report / name).read_text())

    numbers = read("report_numbers.json")
    if numbers["confirmation_status"] != "BLOCKED_NO_INDEPENDENT_DATA":
        raise ValueError("This renderer covers the audited blocked run; do not relabel another run")
    old = read("global_trajectory_diagnostics.json")["diagnostics"]["historical_validation448"]
    methods = read("selector_analysis.json")["448/discovery"]
    cycle = methods["CycleGate_outer_OOF_before_fallback"]
    gate = methods["GateOnly_outer_OOF_before_fallback"]
    policies = read("selectors.json")
    costs = read("cost_analysis.json")
    tests = read("tests.json")

    def pp(value):
        return f"{100 * value:.6f}"

    def interval(values):
        return "[" + ", ".join(pp(x) for x in values) + "]"

    lines = [
        "# EXP-2D V2: 独立确认受阻，已完成 discovery 与旧 raw 诊断",
        "",
        "**CONFIRMATION_STATUS=BLOCKED_NO_INDEPENDENT_DATA**。新独立序列/target pairs 均为 0。",
        "H1a 与独立 H2 尚未完成；下列性能数字来自已经曝光的 V1 raw 或 discovery，不能当成新测试结果。",
        "V1 产物保留，V2 在独立目录开发、运行、测试和归档；没有打开 reserve 或 DAVIS 官方 validation。",
        "",
        "## V1 原始结果核验与真正的样本依赖空间",
        "",
        "原始 1792 行候选覆盖检查、README/STATUS 数字与旧锁定哈希核验通过。以下只后处理 448、16 个旧 validation 序列、32 pairs 的封存 raw，没有重新评价媒体。",
        "",
        "| 方法 | Pixel J（0–1） | 资格 |",
        "|---|---:|---|",
    ]
    labels = {
        "OFF": "OFF/OFF",
        "discovery_fixed16": "Discovery-fixed16（ALL/B）",
        "constant_oracle": "Per-pair constant oracle（4条）",
        "dynamic_oracle": "Per-pair dynamic oracle（16条）",
        "hindsight_global16": "Hindsight-global16（A/ALL）",
    }
    for name, label in labels.items():
        scope = "冻结部署基线" if name in ("OFF", "discovery_fixed16") else "使用答案的离线诊断"
        lines.append(f"| {label} | {old['means'][name]:.12f} | {scope} |")
    lines += [
        "",
        "| Gap | J百分点 | 95% sequence bootstrap CI（百分点） | LOSO最小值（百分点） |",
        "|---|---:|---:|---:|",
    ]
    for name in ("write_opportunity", "nonconstant_extra", "sample_dependence_gap"):
        x = old["gaps"][name]
        lines.append(f"| {name} | {pp(x['mean'])} | {interval(x['ci95'])} | {pp(x['loso_min'])} |")
    gap = old["gaps"]["write_opportunity"]
    dep = old["gaps"]["sample_dependence_gap"]
    lines += [
        "",
        f"相对 OFF 的机会出现在 {gap['sequence_counts']['improved']}/16 个序列；正贡献 top-1/top-3 占比为 {100 * gap['top1_positive_share']:.3f}% / {100 * gap['top3_positive_share']:.3f}%。",
        f"逐样本相对全局16轨迹的额外空间为 {pp(dep['mean'])} 个百分点。A/ALL 是完整旧集合上的唯一全局最优者；所有 pair 的完整最优轨迹集合没有共同交集。",
        "sample_dependence_gap 的每次 bootstrap 和 LOSO 均重新优化该子集的全局16轨迹，未固定完整数据的 winner。全轨迹 regret、并列集合、贡献和删一结果见 global_trajectory_diagnostic.json。",
        "Oracle 含 OFF，dynamic 含 constant；非负具有构造性。非恒定轨迹优势未匹配更新次数/动作多重集合，不能视为顺序因果机制。",
        "",
        "## V2 discovery：过程估计与最终策略分开",
        "",
        "12 个已授权 discovery 序列 × 2 target slots × 16 candidates = 384 行。保持 DINOv2-S/14、448/32×32、PCA64、RGB decoder、eta=5、clip=0.15、两步 A/B 写入与 pixel J 不变。复用已核验 V1 cache/readout 与封存 reward；本轮仅重算 source cycle。",
        "Source mask 在该任务部署时可用，这是 source-annotated、discovery-supervised 设置，不是无标注方法。Cycle 使用两次独立 hard cosine 匹配；前景对象等权 soft IoU，以 product intersection/probabilistic union 定义。只新增 f_cycle。",
        "Nested sequence LOSO 的 scaler、ridge 和调参只见各训练折；固定 alpha=[0.1,1,10,100]、threshold=[0,.0005,.001,.002,.005] 与 ALWAYS_OFF，sequence 总权重相等。",
        "",
        "| 外层 OOF，回退前 | mean J | ΔJ vs OFF（百分点） | 95% CI（百分点） | 写入率 | 有害写入/写入pairs | positive capture | net capture |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, value in (("GateOnly", gate), ("CycleGate（预指定主方法）", cycle)):
        x = value["comparisons"]["vs_OFF"]
        lines.append(
            f"| {name} | {value['mean_J']:.12f} | {pp(x['mean'])} | {interval(x['ci95'])} | {value['write_rate']:.2%} | {value['harmful_write_rate']:.2%} | {value['positive_capture']:.2%} | {value['net_capture']:.2%} |"
        )
    lines += [
        "",
        f"CycleGate−GateOnly 的外层 OOF 差为 {pp(cycle['secondary_vs_GateOnly']['mean'])} 个百分点，95% CI {interval(cycle['secondary_vs_GateOnly']['ci95'])}，跨过 0。不能据此认定新信号提高净收益。两者 OOF 写入率相同，因此不能仅凭这里的结果认定 GateOnly 靠更少写入取得优势。",
        f"CycleGate OOF positive_gain={cycle['positive_gain']:.12f}，harmful_loss={cycle['harmful_loss']:.12f}，净收益={cycle['net_gain']:.12f}。positive capture 不等于 net capture。",
        f"CycleGate OOF 相对 discovery-fixed16 为 {pp(cycle['comparisons']['vs_discovery_fixed16']['mean'])} 个百分点，95% CI {interval(cycle['comparisons']['vs_discovery_fixed16']['ci95'])}。fixed16 曾在完整 discovery 上选定，因此这一 discovery 内比较只是描述性诊断。",
        "上述 OOF 区间只描述已得到的 discovery 折外决策；各折共享训练序列，不是新独立 holdout 置信证明。97.5% 区间已保存，不能把 discovery 的区间包装成预注册独立 H2 成功。",
        "两方法外层净收益均 >0，按固定规则最终没有退化为 ALWAYS_OFF；均冻结为 RIDGE_GATE，alpha=100，threshold=0。最终 full-discovery tuning/in-sample 表与外层 OOF 单独保存，不能把最终调参得分当泛化。",
        "所有 selector 的对象/target/sequence 汇总、写入损害、beneficial precision/recall（写入pairs / 存在oracle机会pairs 为各自分母）、零分母 NA、≥0.5/1/2百分点计数、F/JF、tiny-object敏感性均见 selector_analysis、CSV 和 robustness_analysis。没有事后删除 tiny 对象改善主指标。",
        "",
        "## 独立数据审计与阻塞边界",
        "",
        "- [VOST 官方来源与许可](https://www.vostdataset.org/data.html)：公开为 CC BY-NC-SA 4.0，来自 Ego4D/EPIC-KITCHENS；ZIP 的 clip ID 缺少到原始视频的可核实映射，未通过视频独立性审计。该域若未来使用，属于跨数据集物体变形确认。",
        "- [FBMS-59 官方来源与条款](https://lmb.informatik.uni-freiburg.de/resources/datasets/)：仅研究用途、禁止商业使用。官方成员与评估代码表明 PGM 可能是 label 或 confidence，PPM 需要 region 映射；未找到通用且兼容 V1 的背景/void规则。按名称家族合并也不能证明原视频独立，未予放行。",
        "- 仅下载约 44.1 MB 元数据与官方评估源码；新 RGB/source mask/target mask 均为 0。未读取 mask 内容判断样本是否有利。精确/近重复影像审计因没有合格待选 cohort 而未运行，明确记为 UNVERIFIED。",
        "- split_manifest 为空确认集；predictions_manifest.complete=false，无独立 raw_results.jsonl。远程 target fetch 保持关闭，无合格 adapter；待运行命令现在会拒绝此 manifest，不能直接声称确认流程已经跑通真实外部数据。",
        "恢复条件是可审计的原视频身份、兼容许可与 source 注释、已验证的固定 mask 编码、预先锁定的帧清单和历史精确/近重复审计；之后需先全部预测冻结，再由单独 evaluator 读取 target GT。现有 reserve/官方 DAVIS val 仍不可自动启用。",
        "",
        "## 计算成本、完整性与测试",
        "",
        f"实际 discovery 计算：{sum(x['proposal_calls'] for x in costs['pairs'])} 次 proposal、{sum(x['increment_calculations'] for x in costs['pairs'])} 次 A/B 增量计算、{sum(x['cycle_calls'] for x in costs['pairs'])} 次 cycle、{sum(x['correspondence_calls'] for x in costs['pairs'])} 次双向对应调用。",
        f"候选状态计算共 {sum(x['candidate_seconds'] for x in costs['pairs']):.6f}s；cycle 共 {sum(x['cycle_seconds'] for x in costs['pairs']):.6f}s，其中对应计算 {sum(x['correspondence_seconds'] for x in costs['pairs']):.6f}s（包含关系，不能相加）。",
        "Backbone 本轮调用0，旧可见特征也复用；CPU 计算对应的本轮 CUDA 峰值为0。另记录了保存特征上的单次 selector 推理计时，nested training 未单独计时，标记 NA。这里没有独立测试全流程成本，也没有 matched-budget 效率比较；即使最后选 OFF，全候选试算成本仍已发生。",
        "Source mask 读取24次，target GT读取0；日志分开归档。674项 V1 产物与170项旧锁定源码复核；169项 V2相关实现/骨干源码锁定。新增代码不会反向改变旧锁的文件集合。",
        f"完整测试：{tests['full_suite']['passed']} passed，{tests['full_suite']['skipped']} skipped。首轮在旧 dynamic/types 路径发生原生 segmentation fault（exit139），同命令重跑通过；原因未确定，失败与重试日志均保留。唯一 skip 是公开仓库未分发的历史 V5 checkpoint，不是本轮为了通过测试而关闭检查。",
        "10份统计 JSON/CSV 两次纯封存产物重生成逐字节一致；重生成不读取媒体、不训练、不调用 selector 推理。integrity.json 的 PASS 表示已完成工作的一致性，不表示独立科学确认成功。",
        "",
        "## 对三个目标的判断",
        "",
        "H1b：旧 validation 中仍有约0.205456百分点的逐样本空间，这是被观察数据上的诊断。H1a：新独立序列尚无合格数据，收益量级、跨视频稳定性和 tiny-object依赖无法独立确认。H2：尚无独立确认，不能宣称 CycleGate 战胜 OFF 或 fixed16。",
        "旧 oracle headroom 较小；discovery selector 呈正的流程估计，但不足以得出实用部署结论。现有证据既不能证明 cycle 有效，也不能从当前未确认推断所有可见信号均不可能有效。限制包括历史曝光、样本量、外部域差异、全候选试算与 source 标注假设。",
        "",
        "## 复现与产物",
        "",
        "执行 `bash regenerate.sh` 只重生成分析与本文；`bash commands.sh discovery` 在新的 replay 目录重算 discovery 和 nested fit，拒绝覆盖本实验。`bash commands.sh tests` 运行完整测试。待确认的 predict/evaluate 命令及 adapter 阻塞在 commands.sh 中说明。",
        "配置/协议：config.json、preregistration.json、feature_contract.json；锁：selector_lock.json、implementation_lock.json；完整折信息：nested_cv_results.json；raw：discovery_rows.jsonl；决策：discovery_decisions.jsonl；数据与网络：independent_data_audit.json、resource_request_log.json、各数据源审计。",
        "统计：global_trajectory_diagnostic.json、selector_analysis.json、bootstrap_results.json、robustness_analysis.json、per_pair_results.csv、per_sequence_results.csv；成本/访问：cost_analysis.json、source_mask_access_log.json、target_gt_access_log.json；完整性/环境：integrity.json、tests.json、git_commit.txt、dirty.patch、environment.json、source snapshots。",
    ]
    (report / "README.md").write_text("\n".join(lines) + "\n")
    terminal = {
        "PRIMARY_METRIC": "PIXEL_J",
        "PRIMARY_RESOLUTION": 448,
        "DATA_INDEPENDENCE_STATUS": "UNVERIFIED",
        "EVALUATION_DOMAIN": "NONE_CONFIRMED",
        "N_NEW_SEQUENCES": 0,
        "N_NEW_PAIRS": 0,
        "OLD_RAW_GLOBAL16_DIAGNOSTIC": "COMPLETE_EXPLORATORY",
    }
    for key in (
        "OFF_J DISCOVERY_FIXED16_J HINDSIGHT_GLOBAL16_J CONSTANT_ORACLE_J DYNAMIC_ORACLE_J ORACLE_MINUS_OFF DYNAMIC_MINUS_CONSTANT ORACLE_MINUS_HINDSIGHT_GLOBAL16 GATE_ONLY_J CYCLE_GATE_J CYCLE_GATE_MINUS_OFF CYCLE_GATE_MINUS_FIXED CYCLE_GATE_MINUS_GATE_ONLY CYCLE_GATE_WRITE_RATE CYCLE_GATE_HARMFUL_WRITE_RATE CYCLE_GATE_POSITIVE_CAPTURE CYCLE_GATE_NET_CAPTURE"
    ).split():
        terminal[key] = "NA_NO_INDEPENDENT_TEST"
    terminal.update(
        PRIMARY_POLICY_ALWAYS_OFF=str(policies["CycleGate"]["policy"] == "ALWAYS_OFF").lower(),
        TARGET_GT_READ_BEFORE_PREDICTION_LOCK=0,
        RESERVE_TOUCHED="false",
        DAVIS_OFFICIAL_VAL_TOUCHED="false",
        CONFIRMATION_STATUS="BLOCKED_NO_INDEPENDENT_DATA",
        TESTS="500_PASS_1_SKIP_ON_RETRY_NATIVE_CRASH_LOGGED",
        FINAL_INTEGRITY="SEE_integrity.json",
        REPORT_DIR=str(report),
    )
    integrity_path = report / "integrity.json"
    if integrity_path.exists():
        terminal["FINAL_INTEGRITY"] = read("integrity.json")["status"]
    text = (
        "=" * 60
        + "\nEXP-2D V2: INDEPENDENT OPPORTUNITY + TASK-AWARE SELECTION\n"
        + "=" * 60
        + "\n"
        + "\n".join(f"{k}={v}" for k, v in terminal.items())
        + "\n"
        + "=" * 60
        + "\n"
    )
    (report / "terminal_summary.txt").write_text(text)
    (report / "STATUS.md").write_text("# V2 status\n\n```text\n" + text + "```\n")
    print(text)


if __name__ == "__main__":
    main()
