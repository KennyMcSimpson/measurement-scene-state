# EXP-3D-RGBD-COMPLETION-V11

单因素机制检验：carrier 宽度对补全的作用。两个变体都是 V9 C1 的配方（V9 出结果前写定的规则选出），只有宽度不同：C0 为 hidden 8（5150 个参数，重新训练并应与 V9 逐位相同），C1 为 hidden 32（64238 个参数）。主评价在 EVAL-V3（57 个场景，按场景与源资产与全部已用场景不相交，与 DEV、FRESH 的 volume 不相交，可能与 TRAIN72 共享 volume）上进行；所有预测在读取 query 深度之前封存。全程 CPU。

FAR 为离上下文深度重投影命中点超过 8 像素的像素（占 20.3%）；FILL8 在其余像素用 REPROJ_NN，在 FAR 用 carrier。

| EVAL-V3 AbsRel | ALL | NEAR | FAR |
|---|---:|---:|---:|
| REPROJ_NN | 0.2580 | 0.1852 | 0.5367 |
| C0 | 0.2874 | 0.2465 | 0.4336 |
| C1 | 0.2895 | 0.2478 | 0.4284 |
| FILL8_C0 | 0.2405 | 0.1852 | 0.4336 |
| FILL8_C1 | 0.2408 | 0.1852 | 0.4284 |

## 判定

- COMPLETION_GAIN = FAR AbsRel(C0) − FAR AbsRel(C1) = +0.0052，95% CI [-0.0056, +0.0171]；COMPLETION_STATUS=NOT_ESTABLISHED
- HYBRID_GAIN = AbsRel(REPROJ_NN) − AbsRel(FILL8 C1) = +0.0172，95% CI [-0.0117, +0.0571]；HYBRID_STATUS=NOT_ESTABLISHED
- DEV（次要）：C0 0.2575，C1 0.2652，WIDTH_GAIN_DEV=-0.0077，95% CI [-0.0209, +0.0100]，NOT_ESTABLISHED
- 完整性：V11 C0 与 V9 C1 的 DEV 曲线最大差 = 0.0

分支C：加宽没有稳定改善补全，测量保留组合也没有超过直接重投影：在这个规模下，学到的补全还不足以让状态超过几何。

说明：DEV 的 8 个场景在 V2–V10 中反复使用，本轮只用它选择 checkpoint 并给出描述性结果；EVAL-V3 是机制队列，不是资格验证；FRESH-V1/V2 未使用；受保护的 final holdout 未打开。
