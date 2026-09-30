# EXP-3D-RGBD-VIEWCOUNT-V9：在 V8 出结果之前写定的计划

起草：Claude，2026-09-29。写于 V8 的任何 DEV 指标生成或被查看之前（V8 当时正在训练第 2 个 seed）。本文件只记录后续选择规则，不是正式协议。

## 1. 动机（只用 TRAIN 的探索性证据）

视角数试点：两个 carrier 都只在 TRAIN24 上从头训练，在留出的 48 个 TRAIN-X 场景上评价，1 个 seed。

| 训练方式 | 3 视角 | 7 视角 |
|---|---|---|
| 固定 3 视角 | 0.366 | 0.305 |
| 可变 3–7 视角 | 0.339 | 0.269 |

可变视角训练在标准的 3 视角上也更好，而且让更多视角带来的收益更大。结果存于 `EXP-3D-RGBD-DYNAMIC-WRITE-V1/audit/exploratory_viewcount_pilot_train_only/`。

## 2. V9 设计（单因素：训练时的视角数）

- **C0：** 固定 3 视角训练，即当前 Core A 配方。
- **C1：** 可变视角训练，PRIMARY_METHOD。每步的上下文为该角色的 3 个上下文帧，加上场景 4 个均匀空闲帧中的前 m 个，m 服从 U{0..4}（有 seed）。
- **C2：** 固定 7 视角训练，次要。
- **主对比：** 标准 3 视角 DEV 评价下，VIEWCOUNT_GAIN = AbsRel(C0) − AbsRel(C1)。
- **次要：** 7 视角 DEV 评价（追加 4 个 stream 帧，不写入）。

## 3. V9 的基础网格（在 V8 出结果前写定）

- V8 的 RESOLUTION_GAIN 点估计 >0 时，用 32³；否则用 16³。
- 不参考 V8 的其他任何结果。
- bounds 规则固定为 CONTEXT_DEPTH（V7 C1 规则）。

## 4. 顺序

- V8 收尾之后，先运行已准备好的 Core B V1（冻结的 V7 C1 carrier + 学习写入）。
- 然后冻结并运行 V9。
