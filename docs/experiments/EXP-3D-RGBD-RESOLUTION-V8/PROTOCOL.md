# EXP-3D-RGBD-RESOLUTION-V8 协议

起草：Claude（经用户授权持续推进），2026-09-29。本协议在任何 V8 训练之前冻结。按用户要求，GPU 暂停使用，全程只用 CPU（8–23 号核）。

## 0. 背景

- **V7（bounds 规则）：** 预注册分支 A。
  - 改用测得上下文深度确定 bounds 后，DEV AbsRel 从 0.459 降到 0.280，BOUNDS_GAIN +0.179，CI [+0.084, +0.279]。
  - C1 首次在全部 DEV 场景上显著优于常数。
  - 按计划，V7 C1 先在 FRESH-V2 上做一次资格验证，V8 在其后运行。
  - 本协议冻结于资格验证出结果之前，V8 的设计不参考资格验证结果。
- **V6 的区域诊断：** C1 即使在被 ≥2 个上下文视角看到、且有测得深度的区域（OBS2PLUS），AbsRel 仍约 0.48。误差主要来自表示与渲染，而不是区域看不见。
- **体素偏粗：** 16³ 网格在测得深度 bounds 下体素边长约 0.59 m（TRAIN 诊断）。
- **非学习融合诊断（只用 TRAIN，探索性）：** 把上下文深度直接体素化，只有 4 个标量在 TRAIN24 上拟合；在 48 个 TRAIN-X 场景上评价，同一批视角上常数为 0.528。

  | bounds 规则 | 16³ | 32³ |
  |---|---|---|
  | 冻结 | 0.546 | 0.527 |
  | 测得深度 | 0.421 | 0.367（77% 的视角优于常数） |

  在测得深度 bounds 下，分辨率让非学习融合的误差下限再降 0.054；在冻结 bounds 下几乎没有作用。结果存于 `audit/exploratory_nlf_resolution_train_only/`。
- **本轮问题：** 把体素网格从 16³ 提高到 32³，DEV 几何能否改善？

## 1. 数据

- **训练：** V6/V7 冻结的 TRAIN72，所有变体相同。
- **DEV：** 与 V2–V7 相同的 8 个场景，本轮是 DEV 机制诊断。
- **不使用：** FRESH-V1、FRESH-V2；受保护的 final holdout 不打开。
- **参考常数：** 无几何常数沿用由 TRAIN24 拟合的值。
- **NO_HIT 口径：** 同 V7（再分析 Amendment 1）。

## 2. 变体

三个变体都是冻结的 V5 C1 模型与 loss，在 TRAIN72 上训练 6000 步，用同一条 bounds 规则；只有体素网格不同。每个体素都是稠密证据候选。

| ID | 名称 | 网格 | 候选数 | 角色 |
|---|---|---|---|---|
| C0 | GRID16 | 16³ | 4096 | 基线，复现 V7 C1 |
| C1 | GRID32 | 32³ | 32768 | **PRIMARY_METHOD** |
| C2 | GRID24 | 24³ | 13824 | 次要，只作剂量-反应描述，不进入 gate |

- **bounds 规则：** CONTEXT_DEPTH，即 V7 C1 的规则。在锚点坐标系下取包含全部有效上下文深度反投影点与上下文相机中心的外包盒，再按冻结规则外扩（边长 ×1.1，每轴至少 1 m）。
  - 这条规则按 `PLAN_BEFORE_V7_RESULTS.md`（sha256 `a228ed81…`）选定。该计划写于 V7 出任何结果之前：V7 的 BOUNDS_GAIN 点估计 >0 时用测得深度外包盒，否则用冻结规则。prepare 脚本会核对计划的哈希以及这条选择规则。
- **相同的部分：**
  - 5150 个共享参数及其按 seed 的抽取在三种网格下完全相同，参数哈希一致；
  - checkpoint 步、训练数据流和其余训练超参与 V6、V7 相同。
- **需要推广的冻结 16³ 限制**（写在新模块里，冻结模块不改）：
  1. 深度似然尺度 s = exp(log_sigma) × 平均体素边长，按变体自己的网格计算，与 V5 的定义一致；
  2. 状态构建接受 g³ 个稠密候选；
  3. 冻结的 V2 C1 loss 接受 g³ 状态，但表面监督带宽仍保持冻结的 16 格间距，使 loss 本身不变；
  4. spatial-shuffle 对照对全部 g³ 个体素做同一随机种子的置换。
- **16³ 的一致性：** 在 16³ 下，这几处与 V5、V7 逐位一致，有单元测试覆盖（状态哈希和 loss 完全相同）。
- **CPU 耗时**（工程基准，只用 TRAIN）：每步 16³ 为 0.109 秒，24³ 为 0.159 秒，32³ 为 0.273 秒。

## 3. 判定

统计沿用冻结的 V2 估计器（10,000 次 scene paired bootstrap）。

- **RESOLUTION_STATUS：** 对 C0−C1 应用 V2 的主对比 gate，RESOLUTION_GAIN = AbsRel(C0) − AbsRel(C1)。
- **SCENE_SPECIFICITY_STATUS、STATIC_DEV_STATUS：** 均针对 C1，定义与 V5–V7 相同。
- **描述性结果：**
  - GRID24_GAP = AbsRel(C2) − AbsRel(C1)；
  - 最后一步不做选择的 DEV AbsRel；
  - 训练集拟合；
  - 学到的 σ；
  - OBS 区域；
  - 与无几何常数的比较，报告全部场景和排除 NO_HIT 场景两种口径。
- **分支（预先写定）：**
  - A：RESOLUTION_STATUS=SUPPORTED，且 C1 在全部 DEV 场景上的 RAW AbsRel 显著优于 AbsRel 最优常数。下一步在 FRESH-V2 上做一次资格验证。
  - B：RESOLUTION_GAIN 的 CI 下界 >0，或 C1 在全部场景或排除 NO_HIT 后的场景上显著优于常数，但不满足 A。
  - C：其余情况。
- 任何分支下都不运行 Dynamic TTT，`FINAL_STATIC_STATUS=NOT_ESTABLISHED`。

## 4. 禁止项

- 使用 GPU；使用 FRESH-V1 或 FRESH-V2；打开 final holdout。
- 根据 DEV 结果修改网格、bounds 规则、loss、步数、checkpoint 规则、seed、训练集或主方法。
