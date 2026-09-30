# EXP-3D-RGBD-DYNAMIC-WRITE-V1

Core B 第一轮：在已通过资格验证的静态 carrier（V7 C1，冻结、从不更新）上，检验项目原设计的学习写入规则（DirectWriteRule）。每个 episode 为一个上下文角色（3 个 RGB-D 视角）加上 4 个按帧号规则选出的 stream RGB-D 帧；写入规则在 TRAIN72 上训练 3000 步（每个 stream 到达都执行 ALL），不用 DEV 选择 checkpoint；全程 CPU。WRITE_GAIN=AbsRel(OFF)−AbsRel(ALL)=+0.0611，95% CI [+0.0351, +0.0854]；WRITE_STATUS=SUPPORTED；解读分支=B。

| 策略 | 说明 | DEV query AbsRel (95% CI) |
|---|---|---|
| NO_STREAM | 只看上下文（静态 Core A 状态） | 0.2804 [0.1734, 0.3963] |
| OFF | 缓存 stream，不写入 | 0.3215 [0.2017, 0.4692] |
| OFF_CLAMP3 | 缓存 stream，视角计数截断为 3 | 0.2542 [0.1398, 0.3836] |
| FUSE | 写入 fuse 矩阵 | 0.2583 [0.1449, 0.3897] |
| COMPLETE | 写入 complete 矩阵 | 0.3181 [0.2012, 0.4611] |
| ALL | 写入两个矩阵（主方法） | 0.2604 [0.1445, 0.3970] |
| ALL_UNTRAINED | 写入两个矩阵，写入规则未训练 | 0.3218 [0.2018, 0.4696] |
| ALL_WRONG_SCENE | 别的场景的快速权重（对照） | 0.2564 [0.1484, 0.3806] |

| 对比 | 定义 | 结果 | 场景 improved/tied/worse |
|---|---|---|---|
| BEYOND_CLAMP_GAIN | AbsRel(OFF_CLAMP3) - AbsRel(ALL) | -0.0062，95% CI [-0.0215, +0.0081] | 2/0/6 |
| CLAMP_EFFECT | AbsRel(OFF) - AbsRel(OFF_CLAMP3) | +0.0674，95% CI [+0.0365, +0.1020] | 7/0/1 |
| CLAMP_VS_STATIC | AbsRel(NO_STREAM) - AbsRel(OFF_CLAMP3) | +0.0262，95% CI [-0.0103, +0.0610] | 5/0/3 |
| COMPLETE_GAIN | AbsRel(OFF) - AbsRel(COMPLETE) | +0.0034，95% CI [-0.0030, +0.0103] | 7/0/1 |
| FUSE_GAIN | AbsRel(OFF) - AbsRel(FUSE) | +0.0633，95% CI [+0.0346, +0.0924] | 7/0/1 |
| STREAM_CACHE_EFFECT | AbsRel(NO_STREAM) - AbsRel(OFF) | -0.0411，95% CI [-0.0950, +0.0158] | 2/0/6 |
| STREAM_WRITE_GAIN | AbsRel(NO_STREAM) - AbsRel(ALL) | +0.0200，95% CI [-0.0214, +0.0629] | 4/0/4 |
| TRAINING_GAIN | AbsRel(ALL_UNTRAINED) - AbsRel(ALL) | +0.0614，95% CI [+0.0356, +0.0851] | 7/0/1 |
| WRITE_GAIN | AbsRel(OFF) - AbsRel(ALL) | +0.0611，95% CI [+0.0351, +0.0854] | 7/0/1 |
| WRITE_SPECIFICITY | AbsRel(ALL_WRONG_SCENE) - AbsRel(ALL) | -0.0040，95% CI [-0.0209, +0.0078] | 4/0/4 |

## 判定

- WRITE_STATUS（主对比，写入对只缓存）=SUPPORTED
- STREAM_STATUS（写入对静态状态）=NOT_ESTABLISHED
- BEYOND_CLAMP_STATUS（写入对非学习的计数修正）=NOT_ESTABLISHED
- WRITE_SPECIFICITY_STATUS（换成别的场景的快速权重是否变差）=NOT_ESTABLISHED
- SEED_ROBUSTNESS=DIRECTION_REPLICATED；各 seed 的 WRITE_GAIN={"20260928": 0.0026, "20260929": 0.1132, "20260930": 0.0676}

分支B：写入主要在修补 carrier 的视角数量偏移。下一步用可变视角数重新训练静态 carrier（需要重新做静态资格验证），再检验写入。

说明：DEV 的 8 个场景在 V2–V8 中反复使用，本轮是 DEV 机制诊断，不是资格验证；FRESH-V1/V2 未使用；受保护的 final holdout 未打开。
