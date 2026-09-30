# EXP-3D-RGBD-VIEWCOUNT-V9 协议

起草：Claude（经用户授权持续推进），2026-09-29。本协议在任何 V9 训练之前冻结。按用户要求，GPU 暂停使用，全程只用 CPU（8–23 号核）。

## 0. 背景

- **Core A：** V7 C1 在 FRESH-V2 上 `QUALIFIED`。
- **V8（分辨率）：** 预注册分支 B。
  - DEV AbsRel：16³ 0.280、32³ 0.259、24³ 0.244。
  - RESOLUTION_GAIN +0.021，CI [−0.017, +0.052]，CI 跨零。
  - 不做 checkpoint 选择时，32³ 比 16³ 好 0.061（0.330 对 0.269），因为 16³ 在第 2000 步之后过拟合。
- **Core B V1（冻结 carrier 上的学习写入）：** 预注册分支 B。
  - 在冻结的 V7 C1 上，缓存 4 个新帧反而让 DEV AbsRel 从 0.280 变差到 0.322。
  - 学习写入把它拉回到 0.260，但与非学习的视角计数截断（0.254）相当，也不是场景专属的。
  - 说明静态 carrier 对视角数不稳健。这正是本轮要检验的因素。
- **视角数试点（只用 TRAIN，探索性，1 个 seed）：** 两个 carrier 都只在 TRAIN24 上从头训练，在留出的 48 个 TRAIN-X 场景上评价。

  | 训练方式 | 3 视角 | 7 视角 |
  |---|---|---|
  | 固定 3 视角 | 0.366 | 0.305 |
  | 可变 3–7 视角 | 0.339 | 0.269 |

  可变视角训练在标准 3 视角上也更好。
- **本轮问题：** 训练时让上下文视角数在 3–7 之间变化，标准 3 视角 DEV 几何能否改善？这一轮同时为 Core B 第二轮准备一个对视角数稳健的 carrier。

## 1. 数据

- **训练：** TRAIN72。
- **额外视角帧：** 每个 TRAIN 场景的 4 个均匀空闲帧，即不属于任何上下文或 query 角色的帧，沿用 Core B V1 的 stream 规则，只看帧号。
- **DEV：** 与 V2–V8 相同的 8 个场景，本轮是 DEV 机制诊断。
- **参考常数：** 无几何常数沿用由 TRAIN24 拟合的值。
- **不使用：** FRESH-V1、FRESH-V2；受保护的 final holdout 不打开。

## 2. 变体

| ID | 名称 | 每步训练上下文 | 角色 |
|---|---|---|---|
| C0 | FIXED3 | 角色的 3 个上下文帧 | 基线（V8 C1 配方，32³） |
| C1 | VARIABLE3TO7 | 3 个上下文帧 + 前 m 个额外视角帧，m ~ U{0..4}，每步抽一次，A、B 共用 | **PRIMARY_METHOD** |
| C2 | FIXED7 | 3 个上下文帧 + 全部 4 个额外视角帧 | 次要，不进入 gate |

- **网格：** 32³。
  - 按 `PLAN_BEFORE_V8_RESULTS.md`（sha256 `f844ca08…`）选定。该计划写于 V8 出任何结果之前：V8 的 RESOLUTION_GAIN 点估计 >0 时用 32³，否则用 16³。V8 的点估计为 +0.021。
  - prepare 会核对计划的哈希与这条规则。
  - C0 因此复现 V8 C1。
- **bounds：** CONTEXT_DEPTH，只由角色的 3 个上下文帧计算，额外视角不改变 bounds。
- **额外视角的输入方式：** 按到达顺序接在上下文之后，缓存里的帧号改用到达序号。
- **相同的部分：**
  - 模型与 5150 个共享参数及其按 seed 的抽取；
  - loss、6000 步、checkpoint 步；
  - 场景顺序、query slot、射线索引。额外视角数用单独的随机数生成器（seed + 7919），保证各变体的数据流一致，收尾时会核对。
- **DEV 评价与 checkpoint 选择：** 用标准 3 视角上下文，与 V7、V8 相同。

## 3. 判定

统计沿用冻结的 V2 估计器（10,000 次 scene paired bootstrap）。各变体 bounds 相同，所以命中比例相同，不需要 V7 修订 1。

- **VIEWCOUNT_STATUS：** 对 C0−C1 应用 V2 的主对比 gate，VIEWCOUNT_GAIN = AbsRel(C0) − AbsRel(C1)。
- **SCENE_SPECIFICITY_STATUS、STATIC_DEV_STATUS：** 均针对 C1，定义与 V7、V8 相同。
- **描述性结果：**
  - FIXED7_GAP = AbsRel(C2) − AbsRel(C1)；
  - 最后一步不做选择的 DEV AbsRel；
  - 训练集拟合；
  - 学到的 σ；
  - OBS 区域；
  - 与无几何常数的比较。
- **分支（预先写定）：**
  - A：VIEWCOUNT_STATUS=SUPPORTED，且 C1 在全部 DEV 场景上显著优于 AbsRel 最优常数。下一步在可变视角 carrier 上做 Core B 第二轮，并需要新的独立队列做资格验证。
  - B：VIEWCOUNT_GAIN 的 CI 下界 >0，或 C1 优于常数，但不满足 A。
  - C：其余情况。Core A 配方保持固定 3 视角训练。

## 4. 禁止项

- 使用 GPU；使用 FRESH-V1 或 FRESH-V2；打开 final holdout。
- 根据 DEV 结果修改视角规则、网格、loss、步数、checkpoint 规则、seed、训练集或主方法。
