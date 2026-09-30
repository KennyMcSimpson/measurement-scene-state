# EXP-3D-RGBD-COMPLETION-V11 协议

起草：Claude（经用户授权持续推进），2026-09-29。本协议在任何 V11 训练之前冻结，且只在 Core B V2 收尾之后冻结。按用户要求，GPU 暂停使用，全程只用 CPU（8–23 号核）。

## 0. 背景

- **几何基线：** 学到的状态整体上与直接重投影持平。它在被看到的区域差得多，在没被看到的区域更好。
- **上限分析（DEV，已封存预测）：** 即使观测区域完全保留测量，整体也只比重投影好约 0.03。瓶颈在补全质量。
- **容量试验（探索性，TRAIN24 训练，48 个 TRAIN-X 场景评价）：**
  - hidden 32 相对 hidden 8，在离测量较远的像素上好 0.059 [+0.021, +0.098]；
  - 保留测量、用宽 carrier 补全的组合（FILL8）比重投影好 0.028 [−0.003, +0.063]。
- **V9：** 可变视角训练对标准 3 视角没有收益（+0.002，CI 跨零）。点估计 > 0，所以按 V9 出结果前写定的计划（`PLAN_BEFORE_V9_RESULTS.md`），V11 的基础配方为 V9 C1（可变 3–7 视角）。
- **本轮问题：**
  1. 在 TRAIN72 上训练的 V9 配方中，加宽 carrier 能否改善未观测区域的补全？
  2. 保留测量、用宽 carrier 补全的组合，能否超过直接重投影？

## 1. 数据

- **训练：** TRAIN72，额外视角帧与 V9 相同。
- **checkpoint 选择：** 8 个 DEV 场景，标准 3 视角，规则与 V7–V9 相同。DEV 结果只作次要描述。
- **主评价：** EVAL-V3，57 个场景，来自 32 个 volume（`outputs/EXP-3D-RGBD-EVAL-V3-DATA`）。
  - 按场景和源资产与全部已用场景不相交，与 DEV、FRESH 的 volume 不相交，但可能与 TRAIN72 共享 volume。
  - 它是机制队列，不作资格验证。
- **不使用：** FRESH-V1、FRESH-V2；TRAIN-X 不作评价（已被容量试验使用）；受保护的 final holdout 不打开。

## 2. 变体

| ID | 名称 | 宽度 | 参数 | 角色 |
|---|---|---|---|---|
| C0 | WIDTH8 | hidden 8，expansion 16 | 5,150 | 基线；重新训练，应与 V9 C1 逐位相同 |
| C1 | WIDTH32 | hidden 32，expansion 64 | 64,238 | **PRIMARY_METHOD** |

- **相同的部分：** V9 C1 的全部配方，包括 TRAIN72、32³、CONTEXT_DEPTH bounds、冻结的 V2 C1 loss、6000 步、checkpoint 步、学习率、可变 3–7 视角规则及其独立的随机数生成器（seed + 7919）、seed、场景顺序与射线。
- **执行：** 6 个训练（2 个宽度 × 3 个 seed）同时运行，各占 1 个 CPU 核。

## 3. EVAL-V3 评价（主）

- **状态构建：** 每个场景与角色的上下文只经上下文加载器读取（RGB-D，只含上下文帧），bounds 为 CONTEXT_DEPTH。C0、C1 的选中 checkpoint 分别构建状态，并渲染每个 primary query 相机。
- **重投影基线：** 用同一上下文的深度做重投影（1 像素 z-buffer，空洞用最近邻填补，全空时用 TRAIN 拟合的常数 2.3736 m）。
- **封存：** 全部预测写盘并记录哈希之后，才读取 query 深度。
- **区域：**
  - FAR：离任一重投影命中点超过 8 像素的有效像素；
  - NEAR：其余有效像素；
  - FILL8：NEAR 用 REPROJ_NN，FAR 用 carrier。
- **主 gate：** 使用冻结的 V2 gate 检查（均值 > 0、CI 下界 > 0、至少 75% 的场景不差、每个留一场景 > 0、正贡献中单个场景占比 ≤ 0.5），按 EVAL-V3 场景做 10,000 次 bootstrap，seed 20260928。
  - COMPLETION_STATUS：COMPLETION_GAIN = FAR AbsRel(C0) − FAR AbsRel(C1)；
  - HYBRID_STATUS：HYBRID_GAIN = AbsRel(REPROJ_NN) − AbsRel(FILL8 of C1)。
- **描述性：** C0 与 C1 在 ALL、NEAR 上的差；REPROJ_NN 与 C1 在 FAR、ALL 上的差；REPROJ_NN 与 C0 的 FILL8 的差。
- **分支（预先写定）：**
  - A：两个主 gate 都 SUPPORTED。加宽改善了补全，并且保留测量的组合超过直接重投影；下一步需要按 volume 独立的确认。
  - B：只有一个 SUPPORTED。
  - C：两个都不成立。

## 4. DEV（次要）

沿用 V9 的 DEV 流程与冻结的 V2 估计器，报告 WIDTH_GAIN_DEV = AbsRel(C0) − AbsRel(C1)、场景专属性、多视角融合与无几何常数参照。这些结果不进入分支判定。

## 5. 禁止项

- 使用 GPU；使用 FRESH-V1 或 FRESH-V2；打开 final holdout。
- 根据任何结果修改宽度、配方、区域定义、FILL8 阈值、gate、seed 或训练集。
