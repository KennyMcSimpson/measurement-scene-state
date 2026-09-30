# EXP-3D-RGBD-DATA-SCALE-V14

单因素机制检验：V11 C1 的配方不变，只把训练场景从 TRAIN72 增加到 TRAIN72 + TRAIN-EXT（共 127 个）。计划在读取任何新训练数据之前写定；C0 按同一配方在 TRAIN72 上重新训练，必须与 V11 C1 逐位相同。主评价在 EVAL-V3 与 EVAL-V4 的合并队列（83 个场景）上进行；EVAL-FAR 的远处 query 只作描述。所有预测在读取 query 深度之前封存。全程 CPU。

FAR 为离上下文深度重投影命中点超过 8 像素的像素（占 20.7%）；HFILL8 在其余像素用调和插值补洞，在 FAR 用 carrier；AHFILL8 另把不透明度 < 0.5 的 FAR 像素也交给调和插值（V13）。

| EVAL-V3 + EVAL-V4 AbsRel | ALL | NEAR | FAR |
|---|---:|---:|---:|
| CONSTANT | 0.5138 | 0.5068 | 0.5090 |
| REPROJ_NN | 0.3021 | 0.2345 | 0.5344 |
| REPROJ_HARMONIC | 0.2890 | 0.2255 | 0.4925 |
| C0 | 0.3186 | 0.2876 | 0.4211 |
| C1 | 0.2953 | 0.2459 | 0.4430 |
| HFILL8_C0 | 0.2780 | 0.2255 | 0.4211 |
| HFILL8_C1 | 0.2886 | 0.2255 | 0.4430 |
| AHFILL8_C0 | 0.2714 | 0.2255 | 0.3933 |
| AHFILL8_C1 | 0.2737 | 0.2255 | 0.3995 |

## 判定

- DATA_GAIN = FAR AbsRel(C0 TRAIN72) − FAR AbsRel(C1 TRAIN72+EXT) = -0.0218，95% CI [-0.0482, +0.0012]，场景 34/0/46；NOT_ESTABLISHED
- HARMONIC_HYBRID_GAIN = AbsRel(REPROJ_HARMONIC) − AbsRel(HFILL8 C1) = +0.0004，95% CI [-0.0207, +0.0261]，场景 34/3/46；NOT_DISTINGUISHABLE
- HFILL8_DATA_GAIN = AbsRel(HFILL8 C0) − AbsRel(HFILL8 C1) = -0.0106，95% CI [-0.0176, -0.0045]，场景 30/3/50；HARMFUL
- 描述性：AHFILL8_DATA_GAIN = -0.0023，95% CI [-0.0072, +0.0024]，场景 42/3/38；调和插值 − AHFILL8 C1 = +0.0153，95% CI [-0.0020, +0.0383]，场景 39/3/41
- DEV（次要）：C0 0.2652，C1 0.2696，DATA_GAIN_DEV=-0.0044，95% CI [-0.0380, +0.0253]，场景 4/0/4，NOT_ESTABLISHED
- 完整性：C0 与 V11 C1 的 DEV 曲线最大差 = 0.0；C0 与 V11 C1 封存预测的最大差 = 0.0（648 个 query）

## 各队列

| 队列 | 场景 | DATA_GAIN | 调和插值 − HFILL8 C1 | 调和插值 − AHFILL8 C1 |
|---|---:|---|---|---|
| EVAL_FAR | 79 | -0.0112，95% CI [-0.0282, +0.0055]，场景 38/0/40，NOT_DISTINGUISHABLE | -0.0039，95% CI [-0.0242, +0.0206]，场景 31/1/47，NOT_DISTINGUISHABLE | +0.0162，95% CI [-0.0006, +0.0382]，场景 37/1/41，NOT_DISTINGUISHABLE |
| EVAL_V3 | 57 | -0.0253，95% CI [-0.0583, +0.0029]，场景 22/0/32，NOT_DISTINGUISHABLE | +0.0006，95% CI [-0.0255, +0.0345]，场景 24/3/30，NOT_DISTINGUISHABLE | +0.0175，95% CI [-0.0040, +0.0482]，场景 28/3/26，NOT_DISTINGUISHABLE |
| EVAL_V4 | 26 | -0.0146，95% CI [-0.0564, +0.0255]，场景 12/0/14，NOT_DISTINGUISHABLE | +0.0002，95% CI [-0.0339, +0.0368]，场景 10/0/16，NOT_DISTINGUISHABLE | +0.0105，95% CI [-0.0148, +0.0420]，场景 11/0/15，NOT_DISTINGUISHABLE |

分支 C：在 CPU 可及的规模内（训练场景 72→127），增加数据没有稳定改善补全，也没有让组合胜过经典几何补洞。

说明：DEV 的 8 个场景在 V2–V12 中反复使用，本轮只用它选择 checkpoint 并给出描述性结果；EVAL-V3/V4/FAR 是机制队列，不是资格验证；FRESH-V1/V2 未使用；受保护的 final holdout 未打开。
