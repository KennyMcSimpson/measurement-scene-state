# EXP-3D-RGBD-DEPTH-BOUNDS-V7 协议

起草：Claude（经用户授权持续推进），2026-09-29。本协议在任何 V7 训练之前冻结。按用户要求，GPU 暂停使用，全程只用 CPU（8–23 号核）。

## 0. 背景

- **V6（训练规模）：** 把训练场景从 24 个扩大到 72 个，DEV AbsRel 从 0.492 降到 0.459。
  - SCALE_GAIN +0.033，CI [−0.008, +0.084]，SCALE_STATUS=NOT_ESTABLISHED，解读分支 C。
  - C1 在全部场景和排除 NO_HIT 场景两种口径下，RAW AbsRel 都没有显著优于 AbsRel 最优常数。
- **非学习融合诊断（EXP-3D-RGBD-NONLEARNED-FUSION-V1）：** 学习有增益，剩下的瓶颈在表示和渲染器一侧，其中最明显的一项是 bounds。
  - 冻结的 bounds 规则把上下文射线投到一个固定先验（near 0.214 m，far 5.672 m），这个先验只由 3 个场景得到。
  - DEV 场景 ai_009_001 的全部查询射线都落在 bounds 之外（NO_HIT）。
- **TRAIN 上的 bounds 诊断**（只用 TRAIN，不看 DEV）：

  | 规则 | 被截断的查询 GT 深度 | 体素边长 |
  |---|---|---|
  | 冻结规则 | 26% | 0.43 m |
  | 测得深度外包盒 | 3.4% | 0.59 m |
  | 外包盒，裁 1% 尾部 | 14.7% | 0.41 m |

- **本轮问题：** 在 RGB-D 轨道上，上下文深度本来就是测试时输入。用测得的上下文深度来确定每个状态的 bounds，DEV 几何能否改善？

## 1. 数据

- **训练：** V6 冻结的 TRAIN72（TRAIN24 加 48 个 TRAIN-X 场景），所有变体相同。
- **DEV：** 与 V2–V6 相同的 8 个场景，本轮是 DEV 机制诊断。FRESH-V1 不使用；受保护的 final holdout 不打开。
- **参考常数：** 无几何常数沿用由 TRAIN24 拟合的值，与 V5、V6 及再分析保持可比。
- **NO_HIT 口径：** 沿用再分析 Amendment 1 的规则。一个场景只有在所有变体的全部密封 RAW 预测都为零时才算 NO_HIT；如果测得深度 bounds 让该场景有了命中，它就不再被排除。

## 2. 变体

三个变体都是 V6 的 C1 配方：冻结的 V5 C1 模型与 loss（稠密 16³ carrier，零初始化的测得深度旁路，表面监督），72 个训练场景，6000 步。

| ID | 名称 | bounds 规则 | 角色 |
|---|---|---|---|
| C0 | FROZEN_BOUNDS | 冻结的相机射线先验规则（与 V6 C1 相同） | 基线 |
| C1 | DEPTH_BOUNDS | 锚点坐标系下的轴对齐盒，包含全部有效上下文深度反投影点和全部上下文相机中心 | **PRIMARY_METHOD** |
| C2 | DEPTH_BOUNDS_TRIM1 | 同 C1，但每个轴只取反投影点的 1%–99% 分位（相机中心仍强制在盒内） | 次要，不进入 gate |

- **bounds 的来源：**
  - C1、C2 的盒再按冻结规则外扩：边长 ×1.1，每轴至少 1 m。
  - 只用上下文帧的深度，在状态密封之前按 CONTEXT_ONLY_RGBD 读取。
  - 每个上下文角色（A/B）用它的完整上下文算一个盒；anchor 状态沿用对应完整角色的盒。
  - 训练和评价用同一条规则。
- **相同的部分：**
  - 同 seed 的初始化完全相同（参数哈希一致）；
  - checkpoint 位于 [100, 200, 300, 500, 750, 1000, 1500, 2000, 3000, 4000, 5000, 6000]；
  - 训练数据流一致；
  - 其余训练超参与 V5、V6 相同。
- **唯一差别：** bounds 规则。
- **C0 的复现：** C0 预期复现 V6 C1。结果只作描述；如果不一致，记录差异，不改变判定。

## 3. 判定

统计沿用冻结的 V2 估计器（10,000 次 scene paired bootstrap）。

- **BOUNDS_STATUS：** 对 C0−C1 应用 V2 的主对比 gate，BOUNDS_GAIN = AbsRel(C0) − AbsRel(C1)。
- **SCENE_SPECIFICITY_STATUS、STATIC_DEV_STATUS：** 均针对 C1，定义与 V5、V6 相同。
- **描述性结果：**
  - TRIM_GAIN = AbsRel(C2) − AbsRel(C1)。正值表示覆盖率比体素大小更重要。
  - 最后一步（6000 步）不做选择的 DEV AbsRel，用来衡量选择偏差。
  - 训练集拟合；
  - 学到的 σ；
  - OBS 区域；
  - 与无几何常数的比较，同时报告全部场景和排除 NO_HIT 场景两种口径。
- **分支（预先写定）：**
  - A：BOUNDS_STATUS=SUPPORTED，且在全部 DEV 场景上 C1 的 RAW AbsRel 显著优于 AbsRel 最优常数。下一轮需要新的独立队列做资格验证（FRESH-V1 已用过一次）。
  - B：BOUNDS_GAIN 的 CI 下界 >0，或 C1 在全部场景或排除 NO_HIT 后的场景上显著优于常数，但不满足 A。下一轮检验分辨率（32³）或更多训练数据。
  - C：其余情况。剩余瓶颈不在 bounds 覆盖，下一轮另写协议（分辨率、深度读出或先验）。
- 任何分支下都不运行 Dynamic TTT，`FINAL_STATIC_STATUS=NOT_ESTABLISHED`。

## 4. 禁止项

- 使用 GPU；使用 FRESH-V1；打开 final holdout。
- 根据 DEV 结果修改 bounds 规则、裁剪比例、loss、步数、checkpoint 规则、seed、训练集或主方法。
- 用任何查询 GT 来确定 bounds。
