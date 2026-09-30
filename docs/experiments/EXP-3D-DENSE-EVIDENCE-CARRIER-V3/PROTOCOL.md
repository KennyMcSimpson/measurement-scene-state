# EXP-3D-DENSE-EVIDENCE-CARRIER-V3 协议

起草：Claude（接手 cv 对话后，经用户授权持续推进），2026-09-29。本协议在任何 V3 训练之前冻结。

## 0. 背景与动机

EXP-3D-GEOMETRY-AWARE-CARRIER-V2 首次训练 16³ learned carrier。结果：

- 表面监督收益 C0−C1 = +0.0195，95% CI [−0.0253, +0.0768]，`SURFACE_TRAINING_STATUS=NOT_ESTABLISHED`；
- full context 相对 anchor 仅 +0.0143（CI 跨零）；
- wrong-scene 与 spatial shuffle 均不造成损伤（C1：−0.0094 / +0.0016），只有清空状态造成 +0.37；
- 训练集深度 AbsRel 约 0.48，DEV 约 0.63；同类场景 direct-state 为 0.31–0.36。

V2 保存状态上的事后探索性诊断（不改变 V2 任何结论）：同场景 A/B 状态的密度相关约 0.62–0.75，跨场景约 0.36–0.63，3 帧上下文状态与仅锚点状态的相关约 0。

实现核查：carrier 的多视角证据只在 `token_count=128` 个固定候选点上投影提取（`lift_cached_observations`），然后写入 16³ 体积，再经两层 3×3×3 卷积扩散。候选点按展平序号等距选取，在 16³ 上只覆盖 3.1% 的体素，空间分布也不均匀；这个设计原本对应 8³（25% 覆盖）。V2 按协议要求保持候选语义不变，因此 16³ 网格上的证据通路实际上极其稀疏。

## 1. 研究问题

- **Q1**：V2 carrier 的场景无关行为，是否主要由稀疏的证据通路造成？
- **Q2**：只把证据候选从 128 加密到全部 4096 个体素、其他完全不变时，learned carrier 能否 (a) 更好地拟合训练集；(b) 形成场景专属的状态（wrong-scene 与 spatial shuffle 损伤的 CI 下界 > 0）；(c) 使 full context 稳定优于 anchor；(d) 降低 DEV query AbsRel？
- **Q3（次要）**：在稠密证据下，表面监督是否带来收益？

## 2. 变体

全部使用 16³ 网格、64 samples 固定 renderer、冻结的 GT-free bounds，参数量 5125，同一 seed 下初始参数完全相同（`token_count` 只改变 buffer，不消耗随机数）。

| ID | 名称 | token_count | loss | 角色 |
|---|---|---:|---|---|
| C0 | SPARSE128_SURFACE | 128 | V2 C1 loss（RGB Charbonnier + masked depth AbsRel + 0.1×surface） | 基线，即 V2 C1 配方 |
| C1 | DENSE4096_SURFACE | 4096 | 同 C0 | **PRIMARY_METHOD** |
| C2 | DENSE4096_NO_SURFACE | 4096 | V2 C0 loss（无 surface） | 次要，只回答 Q3 |

C0 与 C1 的唯一差别是候选点集合（全部体素）。不得增加 hidden channels、改变 lifting 规则（≥2 视角支持）、fusion、refinement、heads 或 renderer。C2 不参与任何 gate，不能改为主方法；C2 失败不阻塞主比较。

## 3. 数据

沿用 V2 锁定的 24 TRAIN / 8 DEV、帧角色与 GT-free bounds 规则（数据字节在 V3 preregistration 中重新哈希）。所有场景都是历史曝光场景，且 DEV 已在 V2 中被评价：**V3 是 DEV 上的机制诊断轮，不是资格轮**。`FRESH_QUALIFICATION_INCLUDED=false`，不打开任何保护集或 final holdout。

## 4. 训练与 checkpoint 选择

与 V2 完全相同：seeds [20260928, 20260929, 20260930]，3000 步，Adam lr 1e-3、常数学习率、无 weight decay，每步 1 个 TRAIN 场景、1024 条射线，梯度裁剪 1.0，FP32，无数据增强；A/B 两个上下文独立构建状态并等权；每 500 步用 V2 的 DEV 资格规则评价，每个 variant×seed 独立选择 checkpoint，不选 best seed。

执行：seed 顺序执行，同一 seed 的 C0/C1/C2 并行；CPU 绑定 8–23 号核，数值库单线程；每次（重新）启动前等待共享 GPU 至少 3072 MiB 空闲；基础设施故障（原生崩溃或 CUDA 资源分配错误）归档后从头重跑，每个运行最多 2 次；其他任何错误不重试，C0/C1 失败即停止实验。

## 5. 评价与统计

与 V2 相同的 DEV 评价：A、B、anchor、wrong-scene、spatial shuffle、zero；C0/C1 共同 opacity mask；各 variant 的 anchor/direct 静态共 mask；OBS0/OBS1/OBS2PLUS 区域诊断；10,000 次 scene paired bootstrap（seed 20260928，percentile 95%）。统计实现直接复用 V2 已冻结的估计器 `geometry_carrier_statistics`，其字段 `SURFACE_*` 在本轮表示主对比 C0−C1，即密度对比。

- 主对比：`DENSITY_GAIN = AbsRel(C0) − AbsRel(C1)`，正值表示稠密证据更好。
- 次要：`SURFACE_AT_DENSE_GAIN = AbsRel(C2) − AbsRel(C1)`，正值表示稠密证据下表面监督有益；仅描述，带 CI。

## 6. 预注册判定

- **DENSITY_STATUS**：对 C0−C1 应用 V2 的主对比 gate，包括 mean > 0、CI 下界 > 0、至少 75% 场景不差、所有 leave-one-scene-out > 0、正向 top1 贡献 ≤ 0.5、C1 的 wrong-scene 与 spatial shuffle 损伤 CI 下界 > 0、C0/C1 opacity 检查、状态完整性。取值 SUPPORTED / PARTIAL / NOT_ESTABLISHED / HARMFUL，含义同 V2。
- **SCENE_SPECIFICITY_STATUS**（C1）：wrong-scene 与 spatial shuffle 损伤的 CI 下界均 > 0 为 SUPPORTED，否则 NOT_ESTABLISHED。
- **STATIC_DEV_STATUS**（C1）：V2 的静态 gate，包括 full context > anchor（CI、75%）、wrong-scene CI > 0、同一封存状态服务多个 query、C1 的静态 opacity 检查。
- **TRAIN_FIT**：每个 variant 最后 100 步的训练深度 AbsRel，仅描述。
- **状态相关性诊断**：同场景 / 跨场景、direct / anchor，仅描述。

## 7. 预先写定的解读

- **分支 A**：DENSITY_STATUS=SUPPORTED 且 SCENE_SPECIFICITY_STATUS=SUPPORTED。稀疏证据通路是 V2 失败的主要原因之一；下一轮在稠密 carrier 上重做表面监督的 matched 对照，并建立经过独立性审计的 fresh 队列。
- **分支 B**：DENSITY_GAIN 的 CI 下界 > 0，或 SCENE_SPECIFICITY_STATUS=SUPPORTED，但未满足分支 A。证据密度有部分作用；下一轮先定位剩余瓶颈（融合、可见性、训练规模），再决定是否引入新模块。
- **分支 C**：其余情况。排除“候选稀疏”作为主因；下一轮按 V2 CASE A 引入可见性感知融合或 cost-volume 几何推理，另写协议。
- 无论哪个分支：本轮不运行 Dynamic TTT，`DYNAMIC_TTT_NEXT_STAGE_ALLOWED=false`，`FINAL_STATIC_STATUS=NOT_ESTABLISHED`，不宣称独立泛化。

## 8. 禁止项

Dynamic TTT、writer 更新、FC/CF、动作策略与 controller、轨迹搜索、新 backbone、transformer 或 cost-volume 融合、32³ carrier、renderer > 64 samples、bounds 或帧选择搜索；增加 hidden channels 或改变参数量；根据 DEV 结果修改 loss、checkpoint 规则、seed、预算或主方法。

## 9. 交付

`outputs/EXP-3D-DENSE-EVIDENCE-CARRIER-V3/` 与 `docs/experiments/EXP-3D-DENSE-EVIDENCE-CARRIER-V3/`：README、STATUS、终端摘要、preregistration 与全部合同、raw 结果、8 张图、训练曲线、成本、审计与完整性清单、测试记录、复现命令、源码快照；旧实验文件保持不变并在定稿时复核。
