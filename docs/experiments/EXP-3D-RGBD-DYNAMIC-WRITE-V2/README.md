# EXP-3D-RGBD-DYNAMIC-WRITE-V2

Core B 第二轮：在 V9 的 32³ carrier（C1，按 V9 出结果前写定的计划选择；冻结、从不更新）上，重新检验项目原设计的学习写入规则（DirectWriteRule）。每个 episode 为一个上下文角色（3 个 RGB-D 视角）加上 4 个按帧号规则选出的 stream RGB-D 帧；写入规则在 TRAIN72 上训练 3000 步（每个 stream 到达都执行 ALL），不用 DEV 选择 checkpoint；全程 CPU。BEYOND_BASELINE=AbsRel(OFF)−AbsRel(ALL)=-0.0013，95% CI [-0.0046, +0.0020]；BEYOND_BASELINE_STATUS=NOT_ESTABLISHED；SPECIFICITY_STATUS=HARMFUL；解读分支=C。

| 策略 | 说明 | DEV query AbsRel (95% CI) |
|---|---|---|
| NO_STREAM | 只看上下文（静态 Core A 状态） | 0.2575 [0.1399, 0.4014] |
| OFF | 缓存 stream，不写入 | 0.2047 [0.0878, 0.3548] |
| OFF_CLAMP3 | 缓存 stream，视角计数截断为 3 | 0.2079 [0.0874, 0.3636] |
| FUSE | 写入 fuse 矩阵 | 0.2054 [0.0882, 0.3565] |
| COMPLETE | 写入 complete 矩阵 | 0.2047 [0.0874, 0.3560] |
| ALL | 写入两个矩阵（主方法） | 0.2060 [0.0887, 0.3579] |
| ALL_UNTRAINED | 写入两个矩阵，写入规则未训练 | 0.2045 [0.0877, 0.3550] |
| ALL_WRONG_SCENE | 别的场景的快速权重（对照） | 0.2052 [0.0880, 0.3568] |

| 对比 | 定义 | 结果 | 场景 improved/tied/worse |
|---|---|---|---|
| BEYOND_BASELINE | AbsRel(OFF) - AbsRel(ALL) | -0.0013，95% CI [-0.0046, +0.0020] | 4/0/4 |
| BEYOND_CLAMP_GAIN | AbsRel(OFF_CLAMP3) - AbsRel(ALL) | +0.0019，95% CI [-0.0028, +0.0071] | 5/0/3 |
| CLAMP_EFFECT | AbsRel(OFF) - AbsRel(OFF_CLAMP3) | -0.0032，95% CI [-0.0089, +0.0003] | 2/0/6 |
| CLAMP_VS_STATIC | AbsRel(NO_STREAM) - AbsRel(OFF_CLAMP3) | +0.0496，95% CI [+0.0101, +0.0983] | 7/0/1 |
| COMPLETE_GAIN | AbsRel(OFF) - AbsRel(COMPLETE) | -0.0000，95% CI [-0.0022, +0.0022] | 4/0/4 |
| FUSE_GAIN | AbsRel(OFF) - AbsRel(FUSE) | -0.0008，95% CI [-0.0020, +0.0005] | 3/0/5 |
| STREAM_CACHE_EFFECT | AbsRel(NO_STREAM) - AbsRel(OFF) | +0.0528，95% CI [+0.0139, +0.1006] | 7/0/1 |
| STREAM_WRITE_GAIN | AbsRel(NO_STREAM) - AbsRel(ALL) | +0.0515，95% CI [+0.0142, +0.0995] | 7/0/1 |
| TRAINING_GAIN | AbsRel(ALL_UNTRAINED) - AbsRel(ALL) | -0.0014，95% CI [-0.0043, +0.0014] | 3/0/5 |
| WRITE_GAIN | AbsRel(OFF) - AbsRel(ALL) | -0.0013，95% CI [-0.0046, +0.0020] | 4/0/4 |
| WRITE_SPECIFICITY | AbsRel(ALL_WRONG_SCENE) - AbsRel(ALL) | -0.0007，95% CI [-0.0014, -0.0000] | 2/0/6 |

## 判定

- BEYOND_BASELINE_STATUS（主 gate，写入对预注册的非学习基线 OFF）=NOT_ESTABLISHED
- SPECIFICITY_STATUS（主 gate，换成别的场景的快速权重是否变差，V2 gate）=HARMFUL
- WRITE_STATUS（写入对只缓存）=NOT_ESTABLISHED
- STREAM_STATUS（写入对静态状态）=SUPPORTED
- BEYOND_CLAMP_STATUS（写入对非学习的计数修正）=NOT_ESTABLISHED
- WRITE_SPECIFICITY_STATUS（换成别的场景的快速权重是否变差）=NOT_ESTABLISHED
- SEED_ROBUSTNESS=NOT_REPLICATED；各 seed 的 BEYOND_BASELINE={"20260928": -0.0008, "20260929": -0.0024, "20260930": -0.0007}

分支C：在对视角数稳健的 carrier 上，冻结的写入设计没有超过非学习基线：Core B 的核心主张未得到支持。下一步重新设计写入（例如利用 stream 帧测得深度的写入统计，或基于梯度的测试时训练）。

说明：DEV 的 8 个场景在 V2–V10 中反复使用，本轮是 DEV 机制诊断，不是资格验证；V9 的 carrier 是合格 V7 C1 配方在 DEV 上的后继，本身未经独立资格验证；FRESH-V1/V2 未使用；受保护的 final holdout 未打开。
