# EXP-3D-RGBD-WIDTH-V12

单因素机制检验：在 V11 C1 的配方上把 carrier 宽度从 32 加到 64（250,542 个参数）。按 V11 出结果前写定的计划，只有 V11 的 COMPLETION_GAIN 点估计 > 0 时才运行；主对比的对照是 V11 C1 已选中的 checkpoint（不重新训练）。C0 按同一配方重新训练宽度 32，只用来检查逐位复现。主评价在从未读取过的 EVAL-V4（26 个场景）上进行；所有预测在读取 query 深度之前封存。全程 CPU。

FAR 为离上下文深度重投影命中点超过 8 像素的像素（占 21.7%）；HFILL8 在其余像素用调和插值补洞（INPAINT-V1 的主基线），在 FAR 用 carrier。

| EVAL-V4 AbsRel | ALL | NEAR | FAR |
|---|---:|---:|---:|
| REPROJ_NN | 0.3986 | 0.3424 | 0.5295 |
| REPROJ_HARMONIC | 0.3877 | 0.3327 | 0.4955 |
| V11C1 | 0.3824 | 0.3747 | 0.4060 |
| C0 | 0.3824 | 0.3747 | 0.4060 |
| C1 | 0.3807 | 0.3760 | 0.4022 |
| HFILL8_V11C1 | 0.3777 | 0.3327 | 0.4060 |
| HFILL8_C1 | 0.3775 | 0.3327 | 0.4022 |

## 判定

- WIDTH64_COMPLETION_GAIN = FAR AbsRel(V11 C1) − FAR AbsRel(V12 C1) = +0.0037，95% CI [-0.0151, +0.0222]；NOT_ESTABLISHED
- HARMONIC_HYBRID_GAIN = AbsRel(REPROJ_HARMONIC) − AbsRel(HFILL8 V12 C1) = +0.0102，95% CI [-0.0263, +0.0509]；NOT_DISTINGUISHABLE
- HFILL8_WIDTH_GAIN = AbsRel(HFILL8 V11 C1) − AbsRel(HFILL8 V12 C1) = +0.0002，95% CI [-0.0048, +0.0045]；NOT_ESTABLISHED
- DEV（次要）：C0 0.2652，C1 0.2658，WIDTH_GAIN_DEV=-0.0006，95% CI [-0.0099, +0.0054]，NOT_ESTABLISHED
- 完整性：V12 C0 与 V11 C1 的 DEV 曲线最大差 = 0.0；EVAL-V4 预测最大差 = 0.0；对照与复现实验封存预测的最大差 = 0.0

分支C：宽度 64 没有稳定改善补全，也没有让组合胜过经典几何补洞：在 TRAIN72 的数据量下，宽度 32 之后的容量收益已经看不到。

说明：DEV 的 8 个场景在 V2–V11 中反复使用，本轮只用它选择 checkpoint 并给出描述性结果；EVAL-V4 是机制队列，不是资格验证；FRESH-V1/V2 与 EVAL-V3 未使用；受保护的 final holdout 未打开。
