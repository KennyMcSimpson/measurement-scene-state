# EXP-3D-RGBD-DEPTHSPLAT-JEPA-OBJECTIVE-TTT-ARKIT-V1：只奖励精度的逆向 JEPA 能否改进 TTT

本文件即预注册。锁定时间与哈希见 `preregistration.json`。

## 1. 问题

此前两次尝试，逆向 JEPA 预训练都让 TTT 的起点变差：

- JBT：尺度上限 3.0，以 JA 为起点，15/65。
- JC：尺度上限 0.1，以 JC 为起点，24/56。

JC 的分析指向训练目标：它对未覆盖的像素按全深度惩罚，奖励的是覆盖而不是精度。结果前馈覆盖率从 0.73 升到 0.88，前馈误差从 0.121 升到 0.173。

本实验改用只奖励精度的目标：

- 深度项：只在状态已覆盖的像素（α ≥ 0.5，掩码不传梯度）上，计算归一化深度的相对 L1；
- 颜色项：同样只在已覆盖的像素上，计算归一化颜色的 L1；
- 未覆盖的像素不产生任何损失。

本实验检验两个变体：

- **JN（3 帧）：** 只换目标，其余与 JC 完全相同（上下文 3 帧，目标为有重叠的非上下文帧）。它回答"损害是否来自目标"。
- **JK（7 帧）：** 在 JN 的基础上，把训练时的 episode 改成 TTT 评估的布局：状态由上下文 3 帧加流式帧 4 帧建成，目标是第 14、15 帧。它回答"按 TTT 的用法训练，能否给出更好的起点"。

本设计受 JBT 与 JC 结果的启发，但锁定早于本实验的任何训练与评估。

## 2. 数据

- **训练：** ARKit TRAIN，182 个扫描（EXP-3D-RGBD-ARKIT-DATA-V1）。
    - JN 的目标帧沿用 OVERLAP-V1 的命中表（命中比例 ≥ 0.05）。
    - JK 的流式帧用冻结的 `stream_frame_ids` 规则预先算好（`code/make_streams_train.py` 生成 `training/streams_TRAIN.json`，只用帧编号）。182 个扫描都有，其中 180 个是第 5、8、10、13 帧。
- **评估：** ARKIT_VAL，80 个扫描、160 个 episode、320 个查询。
    - episode 与 ENSEMBLES-ARKIT、JBT、JC 完全相同（`frame_role_lock.json` 逐字节复制）。
    - 复用以下封存（哈希核对）：
        - ENSEMBLES-ARKIT 的几何封存；
        - 发布权重（上限 0.1）的 CACHE / TTT 封存；
        - JC 的封存（上限 0.1）；
        - JBT 的 JA 封存（上限 3.0）。
- **许可：** 只用于非商业研究，媒体在 git 之外。

## 3. 模型

两个变体都只训练 1 个种子（20261008），4000 步，学习率 5e-6（ViT 5e-7），预热 200 步后余弦衰减，AdamW（weight decay 1e-4），梯度裁剪 1.0，尺度上限 0.1，使用最后一步。

- **JN1：** `train_jepa_arkit_norm.py`，由 OVERLAP-V1 的 `train_jepa_arkit_overlap.py` 经 `derive_train_norm.py` 逐字替换得到，只改训练目标；每步 2 个目标帧。
- **JK1：** `train_jepa_arkit_cache.py`，由 JN 的脚本经 `derive_train_cache.py` 逐字替换得到，只改 episode 布局。
- **评估：** 都用 V1 的 `ds_eval.py`，跑 CACHE 与 TTT_m1（100 步），尺度上限 0.1。
- **执行：** `code/jn_jobs.sh` 让两个模型同时训练；每个模型训练结束后，立即用 2 个分片评估。锁定脚本会重新派生两个训练脚本，要求逐字节相同。

## 4. 读出与指标

- **F7 读出：** α ≥ 0.5 处取射线深度 (depth_z / α) / cos，其余用 R7。
- **集成：** GEO_X = sqrt(R7 × X)。
- **指标：** AbsRel，区域为 ALL、NEAR7、FAR7。场景值 = 各角色查询平均后再对角色平均。

## 5. 主检验

主检验用 V2 gate；统计用场景 bootstrap，10,000 次，种子 20260928。

| 编号 | 对比（> 0 表示右边更好） | 区域 | 分支 |
| --- | --- | --- | --- |
| P1 | ZS_TTT − JN_TTT | ALL | JN_IMPROVES_TTT（SUPPORTED）/ _PARTIAL / _NOT_ESTABLISHED / JN_HARMS_TTT |
| P2 | ZS_TTT − JK_TTT | ALL | JK_IMPROVES_TTT（SUPPORTED）/ _PARTIAL / _NOT_ESTABLISHED / JK_HARMS_TTT |

## 6. 次要分析（只给标签）

| 编号 | 对比 | 区域 | 含义 |
| --- | --- | --- | --- |
| S1 | JC_TTT − JN_TTT | ALL | 同一上限下，只换训练目标的效果 |
| S2 | JN_TTT − JK_TTT | ALL | 7 帧布局的效果 |
| S3 | ZS_CACHE − JN_CACHE | ALL | 前馈状态 |
| S4 | ZS_CACHE − JK_CACHE | ALL | 前馈状态 |
| S5 | GEO_ZS_TTT − GEO_JK_TTT | ALL | 集成后 |
| S6 | GEO_ZS_TTT − GEO_JN_TTT | ALL | 集成后 |
| S7 | ZS_TTT − JK_TTT | NEAR7 | |
| S8 | ZS_TTT − JK_TTT | FAR7 | |

另外报告各构建器、各臂的覆盖率。复现检查：复用封存的均值必须与原实验相同。

## 7. 完整性与局限

- **完整性：** 读真值之前，所有封存都必须完整且哈希不变；分析跑两次，结果文件必须逐字节相同。
- **局限：** 每个变体只有一个种子；ARKIT_VAL 已多次用于评估，但 JN、JK 都是新模型，本实验不在其上选择任何设置；只有 80 个扫描。
