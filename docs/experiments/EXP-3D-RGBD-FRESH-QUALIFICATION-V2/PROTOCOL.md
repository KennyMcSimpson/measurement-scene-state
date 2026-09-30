# EXP-3D-RGBD-FRESH-QUALIFICATION-V2 协议

起草：Claude（经用户授权持续推进），2026-09-29。本协议在对任何 FRESH-V2 场景做模型前向之前冻结。计算全部在 CPU（8–23 号核）上完成，不使用 GPU。

## 0. 背景

- **V7 结果：** 预注册分支 A。
  - 改用测得上下文深度确定 bounds 后，DEV AbsRel 从 0.459 降到 0.280，BOUNDS_GAIN +0.179，CI [+0.084, +0.279]。
  - C1 首次在全部 8 个 DEV 场景上显著优于 TRAIN 拟合的常数深度：平均好 +0.227，CI [+0.113, +0.342]。
  - 场景专属性与多视角融合同时成立。
- **为什么还要资格验证：**
  - 这 8 个 DEV 场景在 V2–V7 中反复使用；
  - C1 的 DEV checkpoint 选择有约 0.05 的乐观偏差（不做选择的最后一步为 0.330）；
  - V5 在 FRESH-V1 上就没有复现 DEV 上的收益。
  - 按项目规则，DEV 结果本身不构成资格证明。
- **写定的路线：** V7 出结果之前写定的计划（`EXP-3D-RGBD-RESOLUTION-V8/PLAN_BEFORE_V7_RESULTS.md`，sha256 `a228ed81…`）规定：V7 为分支 A 时，下一步是在 FRESH-V2 上对 V7 C1 做一次资格验证。

## 1. 独立队列 FRESH-V2

- **数据：** `EXP-3D-RGBD-FRESH-V2-DATA` 共 17 个场景，分布在 10 个 volume。
- **来源：** 官方 train 划分，分区表标记为从未被观察过，也从未被任何先前实验使用或考虑过。
- **独立性：** volume 与 V7 的全部 TRAIN72 和 DEV 场景都不重叠；源资产唯一；每个 volume 至多 2 个场景。
- **选择过程：** 名单在读取任何媒体之前冻结，按哈希顺序排列；有效性检查与模型无关。
- **至今未做任何模型前向。**
- **独立性审计：** prepare 阶段逐场景复核上述条件，任一不满足则拒绝冻结。
- **局限：**
  - 这些场景属于历史上的 365 场景池（执行记录为从未被观察）；
  - 有 5 个 volume 也含有 FRESH-V1 评价过的场景；
  - Hypersim 内部供应商层面的资产复用无法排除；
  - 同一 volume 内的场景可能相关，所以另报按 volume 聚类的 bootstrap（描述性）。
- **不打开的数据：** 受保护的官方 test final holdout 不打开。

## 2. 被验证的对象

- **checkpoint：** V7 已封存的全部选择，即 C0、C1、C2 各 3 个 seed 的 DEV 选择 checkpoint。逐个核对哈希，不重新训练，也不重新选择。
- **bounds：** 每个变体用自己的 V7 规则。
  - C0：冻结的相机射线先验规则。
  - C1：FRESH 上下文深度与相机的外包盒。
  - C2：同 C1，裁掉 1% 分位尾部。
  - 测得深度 bounds 只读取 FRESH 上下文帧的深度，在任何状态封存之前按 CONTEXT_ONLY_RGBD 读取。
- **其余与 V7 的 DEV 评价完全相同：** 状态构建、renderer，以及各项对照（anchor、wrong-scene、spatial shuffle、zero）。
- **数据边界：** 所有 FRESH 状态封存之后才读取 query GT。
- **估计器：** 使用 V7 修订 1 的估计器，即冻结的 V2 估计器去掉一条"各变体命中比例相同"的断言。V7 的因素正是 bounds 规则，这条断言对 V7 的变体不适用；其余一字不改。

## 3. 判定

统计用 10,000 次 scene paired bootstrap，与 V7 相同。

- **BOUNDS_STATUS_FRESH：** 对 C0−C1 应用冻结的 V2 主对比 gate，定义与 V7 的 BOUNDS_STATUS 相同。
- **SCENE_SPECIFICITY_STATUS_FRESH：** C1 的 wrong-scene 损伤与 spatial-shuffle 损伤，CI 下界都 >0。
- **STATIC_STATUS_FRESH：** C1 的冻结 V2 静态 gate。
- **REFERENCE_STATUS_FRESH（新增）：** 在全部 FRESH-V2 场景上，C1 的 RAW AbsRel 相对 TRAIN 拟合的 AbsRel 最优常数（2.3736 m）为 ABOVE。ABOVE 用冻结的 V2 证据规则判定：均值 >0、CI 下界 >0、≥75% 的场景不差。
  - 这一项在 FRESH-V1 中只是描述性结果，这里升为 gate，因为绝对精度正是此前卡住的一步。
- **FRESH_QUALIFICATION_STATUS：**
  - 四项都为 SUPPORTED 时为 QUALIFIED，表示 Core A 的静态 carrier 在 RGB-D 轨道上通过资格验证，按项目规则 Core B 的下一阶段可以开启；
  - 否则为 NOT_QUALIFIED。
- **描述性结果（不作 gate）：**
  - 按 volume 聚类的 bootstrap（BOUNDS_GAIN 与 C1 相对常数）；
  - 逐场景表；
  - OBS 区域；
  - C2；
  - 每个变体的射线命中比例；
  - 排除 NO_HIT 场景的敏感性分析。

## 4. 只看一次

- FRESH-V2 只评价一次。
- 任何 FRESH-V2 前向之后，都不修改规则、场景、阈值或 checkpoint。
- 流水线中途因代码问题失败时，只能按修订程序处理，并如实记录。

## 5. 禁止项

- 重新训练或重新选择 checkpoint；
- 根据 FRESH-V2 结果修改任何规则、场景或阈值；
- 打开受保护的 final holdout；
- 使用 GPU。
