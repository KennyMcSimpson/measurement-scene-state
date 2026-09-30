# EXP-3D-RGBD-EVIDENCE-CARRIER-V5 修订记录

## Amendment 1：分支 A 的无几何参考条件只在可渲染场景上判定

**时间**：2026-09-29 06:07（+08；2026-09-28T22:07:25Z），在 V5 冻结并启动之后、查看任何 V5 DEV 指标之前记录。训练中的 DEV checkpoint 曲线未被查看，进度脚本只统计 checkpoint 数量。

**问题**：
- DEV 场景 ai_009_001 是 NO_HIT 场景：query 射线全部落在由上下文相机导出的 bounds 之外，renderer 不产生任何采样。任何 carrier 在该场景的 AbsRel 都恒为 1。
- 在已完成的 EXP-3D-READOUT-REFERENCE-REANALYSIS-V1 中，AbsRel 最优常数在该场景的 AbsRel 为 0.605，所以每个 carrier 在这里都比常数差约 0.395。这与 carrier 质量无关，是 bounds 规则的失败，对所有变体相同。
- 只有 8 个场景时，scene bootstrap 中包含 3 份该场景的抽样概率约 7%（>2.5%）。要让 CI 下界 > 0，C1 需要在其余 7 个场景上平均比常数好约 0.24 AbsRel，即 C1 平均 AbsRel 约 0.26。这远优于 RGB+D 逐场景直接优化的参考值 0.40。
- 因此，按冻结规则，分支 A 的参考条件几乎与 carrier 质量无关地不可达。

**修订**：
- 分支 A（以及分支 B）中“C1 的 RAW AbsRel 相对 REF_TRAIN_ABSREL_OPTIMAL 为 ABOVE”，改为用预注册敏感性分析（`reference_results.json` 的 `sensitivity_excluding_no_hit_scenes`）中的标签判定。该分析排除所有被评价变体的直接预测都为全零的场景。
- 全场景标签照常报告。
- 其余一切不变：DEPTH_STATUS、SCENE_SPECIFICITY_STATUS 与 STATIC_DEV_STATUS 的 gate，以及 C0−C1 主对比。在 NO_HIT 场景上，C0 与 C1 同为全零，差值为 0，不受影响。

**执行**：冻结的报告脚本仍按全场景标签计算 `INTERPRETATION_BRANCH`。为了不改动锁定的源文件（改动会使运行中的训练在结束时的锁检查失败），最终记录会同时给出 FROZEN_RULE_BRANCH 与 AMENDED_RULE_BRANCH，并在顶层文档中说明两者。
