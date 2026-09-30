# EXP-3D-RGBD-TRAIN-SCALE-V6 协议

起草：Claude（经用户授权补充数据并持续推进），2026-09-29。本协议在任何 V6 训练之前冻结。按用户要求，GPU 暂停使用，全程只用 CPU（8–23 号核）。

## 0. 背景

- **V5（RGB-D 轨道）在 DEV 上：** 三个冻结 gate 同时通过，DEPTH_GAIN +0.114。
- **独立资格验证 EXP-3D-RGBD-FRESH-QUALIFICATION-V1：** 在 5 个独立场景上结果为 `NOT_QUALIFIED`。
  - 深度收益没有复现：+0.006，CI [−0.112, +0.116]。
  - 场景专属性与多视角融合复现。
  - 解读：在只有 24 个训练场景时学到的东西泛化有限，按 DEV 挑选 checkpoint 又使 DEV 收益偏乐观。
- **V5 的 RGB-D DEV 曲线：** 到 1500–2000 步仍在下降，训练未饱和。
- **本轮问题：** 按 V5 分支 B 检验训练规模。扩大训练场景，并给足训练步数，能否改善在新场景上的几何？

## 1. 数据

- **TRAIN24：** V2–V5 的 24 个 TRAIN 场景。
- **TRAIN72：** TRAIN24 加上 48 个 TRAIN-X 场景，来自 EXP-3D-RGBD-FRESH-SCALE-DATA-V1。
  - 来源：官方 train split，不在保护或排除名单中，源资产唯一。
  - volume：与 DEV 和 FRESH-V1 都不重叠，每个 volume 最多新增 2 个场景。
  - 选择：按哈希顺序锁定 68 个候选，取前 48 个通过与模型无关的有效性检查的场景，不做替换。
- **DEV：** 与 V2–V5 相同的 8 个场景，本轮是 DEV 机制诊断。FRESH-V1 不使用；受保护的 final holdout 不打开。
- **bounds 先验与参考常数：** bounds 沿用 V2 冻结的 TRAIN 先验；无几何常数沿用由 TRAIN24 拟合的值，与 V5 及再分析保持可比。

## 2. 变体

| ID | 名称 | 配方 | 训练场景 | 角色 |
|---|---|---|---|---|
| C0 | DEPTH_TRAIN24 | 冻结的 V5 C1（测得深度旁路 + 表面监督） | 24 | 基线 |
| C1 | DEPTH_TRAIN72 | 同 C0 | 72 | **PRIMARY_METHOD** |
| C2 | RGB_TRAIN72 | 冻结的 V5 C0（只用 RGB + 表面监督） | 72 | 次要，不进入 gate |

- **相同的部分：** 同 seed 的初始化完全相同，预算都是 6000 步；checkpoint 位于 [100, 200, 300, 500, 750, 1000, 1500, 2000, 3000, 4000, 5000, 6000]；其余训练超参与 V5 相同。
- **唯一差别：** C0 与 C1 只差训练场景集合。不同训练集之间的数据流按设计不同，同一训练集内的变体数据流一致。

## 3. 判定

统计沿用冻结的 V2 估计器（10,000 次 scene paired bootstrap）。

- **SCALE_STATUS：** 对 C0−C1 应用 V2 的主对比 gate。
- **SCENE_SPECIFICITY_STATUS、STATIC_DEV_STATUS：** 均针对 C1，定义与 V5 相同。
- **描述性结果：**
  - DEPTH_AT_SCALE_GAIN = AbsRel(C2) − AbsRel(C1)；
  - 最后一步（6000 步）不做选择的 DEV AbsRel，用来衡量选择偏差；
  - 训练集拟合；
  - 学到的 σ；
  - OBS 区域；
  - 与无几何常数的比较，同时报告全部场景与排除 NO_HIT 场景两种口径。
- **分支（预先写定）：**
  - A：SCALE_STATUS=SUPPORTED，且在排除 NO_HIT 场景后 C1 的 RAW AbsRel 显著优于 AbsRel 最优常数。下一轮需要新的独立队列做资格验证。
  - B：SCALE_GAIN 的 CI 下界 >0，或者 C1 显著优于常数，但不满足 A。
  - C：其余情况，说明泛化瓶颈不只是数据量，下一轮另写协议。
- 任何分支下都不运行 Dynamic TTT，`FINAL_STATIC_STATUS=NOT_ESTABLISHED`。

## 4. 禁止项

- 使用 GPU；使用 FRESH-V1；打开 final holdout。
- 根据 DEV 结果修改 loss、步数、checkpoint 规则、seed、训练集或主方法。
- 用 DEV 结果替换 TRAIN-X 场景。
