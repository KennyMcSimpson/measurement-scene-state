# EXP-3D-RGBD-DYNAMIC-WRITE-V2 协议

起草：Claude（经用户授权持续推进），2026-09-29。本协议在任何 Core B V2 训练之前冻结，且只在 V9 收尾之后冻结。按用户要求，GPU 暂停使用，全程只用 CPU（8–23 号核）。

## 0. 背景

- **Core B V1（冻结的 V7 C1 carrier，16³）：** 预注册分支 B。
  - 缓存 4 个 stream 帧后，DEV AbsRel 从 0.280（NO_STREAM）变差到 0.322（OFF），因为 carrier 训练时只见过 3 个视角。
  - 学习写入（ALL，0.260）与非学习的计数截断（OFF_CLAMP3，0.254）相当。
  - 写入也不是场景专属的：换成别的场景写出的快速矩阵，结果为 0.256。
  - 结论：写入只在修补视角数偏移。
- **V9：** 检验训练时的视角数（固定 3 / 可变 3–7 / 固定 7，32³）。
- **carrier 的选择规则：** 写于 V9 出任何结果之前，见 `PLAN_BEFORE_V9_RESULTS.md`（sha256 `42c83cec…`，prepare 会核对）。
  - V9 VIEWCOUNT_GAIN 的点估计 > 0 时，用 V9 C1（VARIABLE3TO7）；
  - 否则用 V9 C0（FIXED3，等于 V8 C1）。
- **本轮问题：** 在不再受视角数偏移困扰的 carrier 上，学习写入能否超过预注册的非学习基线，并且携带场景专属的信息？

## 1. 数据

- **写入规则训练：** TRAIN72。
- **评价：** 与 V2–V9 相同的 8 个 DEV 场景。本轮是 DEV 机制诊断，不是资格验证。
- **stream：** 沿用 Core B V1 的规则：每个场景的空闲帧（不属于任何上下文或 query 角色）中均匀取 4 帧，两个角色共用，只看帧号。
- **不使用：** FRESH-V1、FRESH-V2；受保护的 final holdout 不打开。

## 2. 设计

- **慢 carrier：** V9 选定变体的 3 个 seed 在 DEV 上选中的 checkpoint，冻结，从不更新。bounds 为 CONTEXT_DEPTH，只由热身上下文计算。
- **写入规则：** 项目原设计的 DirectWriteRule（528 个参数）。
  - 初始化、训练方式与 V1 相同：TRAIN72，3000 步，每个 stream 帧到达时都执行 ALL，不用 DEV 选择 checkpoint。
  - loss 为冻结的 V2 C1 loss，经 V8 的 32³ 推广。
- **策略与对照：** 与 V1 相同。策略为 NO_STREAM、OFF、OFF_CLAMP3、FUSE、COMPLETE、ALL、ALL_UNTRAINED；对照为 ALL_WRONG_SCENE。
- **非学习基线 BASELINE_POLICY：** 预先固定，不按 DEV 结果挑选。
  - carrier 为 V9 C1 时取 OFF（7 个视角在其训练范围内）；
  - carrier 为 V9 C0 时取 OFF_CLAMP3。

## 3. 判定

统计与 V1 相同：10,000 次 scene paired bootstrap，冻结的 V2 gate。

- **主 gate：**
  - BEYOND_BASELINE_STATUS：对 BEYOND_BASELINE = AbsRel(BASELINE_POLICY) − AbsRel(ALL) 应用 V2 gate；
  - SPECIFICITY_STATUS：对 WRITE_SPECIFICITY = AbsRel(ALL_WRONG_SCENE) − AbsRel(ALL) 应用 V2 gate。
- **次要：** WRITE_STATUS（OFF − ALL）、STREAM_STATUS（NO_STREAM − ALL）、BEYOND_CLAMP_STATUS（OFF_CLAMP3 − ALL）。
- **描述性：** FUSE 与 COMPLETE 相对 OFF 的收益、TRAINING_GAIN、每个 seed 的收益、delta1。
- **分支（预先写定）：**
  - A：两个主 gate 都 SUPPORTED。学习写入以场景专属的信息超过非学习基线；下一步需要新的独立队列，对 carrier 与写入一起做资格验证。
  - B：BEYOND_BASELINE 的点估计 > 0，但不满足 A。
  - C：其余情况。冻结的写入设计在不受视角数偏移困扰的 carrier 上也没有超过非学习基线；下一步重新设计写入。

## 4. 说明与禁止项

- V9 的 carrier 是合格的 V7 C1 配方在 DEV 上的后继，本身未经独立资格验证。
- 禁止使用 GPU、使用 FRESH-V1 或 FRESH-V2、打开 final holdout。
- 禁止根据 DEV 结果修改 carrier 选择、非学习基线、策略、步数、seed 或训练集。
