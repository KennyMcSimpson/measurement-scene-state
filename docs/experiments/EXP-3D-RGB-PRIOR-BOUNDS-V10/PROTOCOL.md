# EXP-3D-RGB-PRIOR-BOUNDS-V10 协议

起草：Claude（经用户授权持续推进），2026-09-29。本协议在任何 V10 训练之前冻结，且只在 V9 收尾之后冻结。按用户要求，GPU 暂停使用，全程只用 CPU（8–23 号核）。

## 0. 背景

- **RGB-only 轨道（V2–V6）：** 9 个 RGB-only carrier 与 V6 C2（TRAIN72）都没有优于 TRAIN 拟合的常数深度。DEV AbsRel 为 0.60–0.65，常数为 0.507。项目状态文件据此判断 RGB-only 已到信息上限。
- **这些实验的 bounds：** 都来自冻结的 GT-free 规则。每个上下文像素的射线延伸到先验的近、远距离，取包围盒并放大 10%，每轴至少 1 m。先验只由 3 个训练场景（ai_001_001、ai_002_001、ai_003_001）的 12 帧拟合：near 0.214 m，far 5.672 m。
- **RGB-D 轨道的对照：** 仅把 bounds 换成由测得上下文深度确定的盒子，DEV AbsRel 就从 0.459 降到 0.280（V7，BOUNDS_GAIN +0.179），并在 FRESH-V2 上通过资格验证。
- **探索性诊断（只读 TRAIN，`outputs/EXP-3D-RGB-PRIOR-BOUNDS-DIAG/`）：**

  | bounds 规则 | 测试时需要深度 | TRAIN72 query GT 点在盒外 | 平均体素边长 |
  |---|---|---|---|
  | 冻结先验（0.214 / 5.672 m） | 否 | 29.7% | 0.431 m（16³） |
  | TRAIN72 q01/q99 先验（0.396 / 15.85 m） | 否 | 2.2% | 1.176 m（16³），0.588 m（32³） |
  | 上下文深度 bounds（RGB-D 轨道，参考） | 是 | 3.9% | 0.640 m（16³） |

  冻结先验让约 30% 的 query 表面落在盒外；这部分表面无论 carrier 学到什么都渲染不出来。
- **本轮问题：** RGB-only 的失败有多少来自 bounds 截断？改用 TRAIN72 拟合的先验后，只用 RGB 构建的场景状态能否优于常数深度？

## 1. 数据

- **训练：** TRAIN72，与 V6–V9 相同。
- **DEV：** 与 V2–V9 相同的 8 个场景。本轮是 DEV 机制诊断，不是资格验证。
- **TRAIN72 先验：** prepare 时由 TRAIN72 全部已准备帧的有限正射线距离计算 1% 与 99% 分位，写入 `train72_depth_prior.json`，其哈希进入预注册。只读 TRAIN 数据。
- **参考常数：** 无几何常数沿用由 TRAIN24 拟合的值。
- **不使用：** FRESH-V1、FRESH-V2；受保护的 final holdout 不打开。

## 2. 变体

| ID | 名称 | bounds 先验 | 网格 | 角色 |
|---|---|---|---|---|
| C0 | FROZEN_PRIOR_GRID32 | 冻结的 V2 先验 | 32³ | 基线 |
| C1 | TRAIN72_PRIOR_GRID32 | TRAIN72 先验 | 32³ | **PRIMARY_METHOD** |
| C2 | TRAIN72_PRIOR_GRID16 | TRAIN72 先验 | 16³ | 次要，不进入 gate |

- **模型与训练：** V5 C0 的 RGB-only 配方。
  - 不含深度旁路的稠密 carrier，5125 个参数，逐 seed 初始化与 V5–V9 的共享参数完全相同（prepare 核对 V6 记录的哈希）。
  - 冻结的 V2 C1 表面 loss，经 V8 的网格推广接受 32³ 状态，表面带保持 16³ 的间距。
  - TRAIN72、6000 步、checkpoint 步、学习率、射线采样与 V6–V9 相同。训练时上下文固定为角色的 3 个视角，与 V9 的结果无关。
- **测试时输入：** 只有上下文 RGB 与相机。状态经 RGB-only 上下文加载器构建；封存全部 DEV 状态之前，不读取任何帧的深度。评价模块与收尾检查都会核对访问日志。
- **bounds：** 训练与评价都用各自先验的冻结 GT-free 规则，只由上下文相机计算。
- **网格：** 32³ 是 V8、V9 使用的网格。TRAIN72 先验在 16³ 下体素边长约 1.18 m，C2 用来描述这一点。
- **完整性（冻结前已检查）：** V10 的代码在 16³、冻结先验下，逐位复现 V6 C2 训练前 3 步的 loss（`audit/v6_c2_replay_check.json`）。

## 3. 判定

统计沿用冻结的 V2 估计器与 V7 修订 1：删去 C0/C1 射线命中比例必须相等的断言，其余不变。本轮的 bounds 先验按设计改变命中比例，所以需要这项修订。bootstrap 为 10,000 次 scene paired。

- **PRIOR_BOUNDS_STATUS：** 对 C0−C1 应用 V2 的主对比 gate，PRIOR_BOUNDS_GAIN = AbsRel(C0) − AbsRel(C1)。
- **SCENE_SPECIFICITY_STATUS、STATIC_DEV_STATUS：** 均针对 C1，定义与 V7–V9 相同。
- **与无几何常数的比较：** 预注册参照，报告全部场景与排除 NO_HIT 场景两种。
- **描述性结果：**
  - GRID16_GAP = AbsRel(C2) − AbsRel(C1)；
  - 各变体各场景的射线命中比例；
  - 最后一步不做选择的 DEV AbsRel；
  - 训练集拟合；
  - OBS 区域。
- **分支（预先写定）：**
  - A：PRIOR_BOUNDS_STATUS=SUPPORTED，且 C1 的 RAW AbsRel 在全部 DEV 场景上显著优于 AbsRel 最优常数（NO_HIT 场景由先验自己承担）。撤回 RGB-only 已到信息上限的判断；下一步需要新的独立队列做资格验证。
  - B：PRIOR_BOUNDS_GAIN 的 CI 下界 >0，或 C1 优于常数（全部场景或排除 NO_HIT 场景），但不满足 A。bounds 先验有部分作用；下一步检验光度证据。
  - C：其余情况。RGB-only 的失败不能由 bounds 截断解释，维持 RGB-D 轨道。

## 4. 禁止项

- 使用 GPU；使用 FRESH-V1 或 FRESH-V2；打开 final holdout。
- 根据 DEV 结果修改先验、网格、loss、步数、checkpoint 规则、seed、训练集或主方法。
