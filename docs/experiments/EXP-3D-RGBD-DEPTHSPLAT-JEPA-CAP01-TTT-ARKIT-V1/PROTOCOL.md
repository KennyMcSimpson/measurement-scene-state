# EXP-3D-RGBD-DEPTHSPLAT-JEPA-CAP01-TTT-ARKIT-V1：在 TTT 的尺度上限下训练逆向 JEPA，能否改进 TTT

本文件即预注册。锁定时间与哈希见 `preregistration.json`。

## 1. 问题

EXP-3D-RGBD-DEPTHSPLAT-JEPA-BASE-TTT-ARKIT-V1（JBT）发现，以真实数据训练的逆向 JEPA（JA）为起点做 TTT，比以发布权重为起点更差。分析指向两个原因：

- **尺度上限：** JA 在上限 3.0 下训练和使用。而对发布权重做 TTT 时，上限 0.1 本身就比 3.0 好（+0.022，70/0/10）。
- **起点质量：** 训练把覆盖率推到了 0.97，但 7 帧前馈状态的误差也随之变大（0.266）。TTT 只有 100 步，修不回起点的差距。

本实验是在看到 JBT 与 EXP-3D-RGBD-DEPTHSPLAT-JEPA-ARKIT-OVERLAP-V1 的结果之后设计的，有动机来源，但预注册在本实验的任何训练与评估之前。它只改一处：训练改在 TTT 最好的尺度上限 0.1 下进行，其余沿用 OVERLAP-V1 的配方。

问题是：**在同样的上限 0.1 下，逆向 JEPA 训练能否让 TTT 比发布权重更好？**

## 2. 数据

- **训练：** ARKit TRAIN 182 个扫描（EXP-3D-RGBD-ARKIT-DATA-V1），目标帧只取上下文命中比例 ≥ 0.05 的帧（OVERLAP-V1 的命中表）。
- **评估：** ARKIT_VAL，80 个扫描、160 个 episode、320 个查询。episode 与 EXP-3D-RGBD-ENSEMBLES-ARKIT-V1 完全相同（`frame_role_lock.json` 逐字节复制）：上下文 3 帧 + 流式帧 4 帧，查询为第 14、15 帧。
- **许可：** 只用于非商业研究，媒体在 git 之外。

## 3. 状态构建器

所有构建器都使用 V1 锁定的 `ds_eval.py`：CACHE 为 7 帧一次性前馈；TTT_m1 为 100 步 Adam，m = 1。

| 名称 | 权重 | 尺度上限 | 来源 |
| --- | --- | --- | --- |
| **JC** | 本实验训练的 JC1 | 0.1 | 本实验新跑 |
| ZS | 发布权重 | 0.1 | 复用 ENSEMBLES-ARKIT 的封存 |
| JA | JEPA-ARKIT-V1 的 JA1 | 3.0 | 复用 JBT 的封存 |
| ZS3 | 发布权重 | 3.0 | 复用 JBT 的封存 |

- **JC1 的训练：** 使用 OVERLAP-V1 的 `train_jepa_arkit_overlap.py`。
    - 种子 20261008，4000 步，学习率 5e-6（ViT 5e-7），预热 200 步后余弦衰减，AdamW（weight decay 1e-4），梯度裁剪 1.0；
    - 每步 2 个目标帧（命中比例 ≥ 0.05），损失 = 原始渲染深度相对 L1 + 0.5 × RGB L1；
    - 注入测得深度，使用最后一步；
    - **尺度上限为 0.1**，这是与 JO1 唯一的不同。
- **只训练一个种子：** 与 JBT 一致，TTT 只用一个事先指定的种子。
- **无模型基线：** R7 = REPROJ7_HARMONIC，复用 ENSEMBLES-ARKIT 的几何封存。

## 4. 读出与指标

- **F7 读出：** α ≥ 0.5 处取射线深度 (depth_z / α) / cos，其余用 R7。
- **集成：** GEO_X = sqrt(R7 × X)。
- **指标：** AbsRel，区域为 ALL、NEAR7、FAR7。场景值 = 各角色的查询平均后，再对角色平均。

## 5. 主检验

使用 V2 gate；统计用场景 bootstrap，10,000 次，种子 20260928。

| 编号 | 对比（> 0 表示右边更好） | 区域 | 分支 |
| --- | --- | --- | --- |
| P1 | ZS_TTT − JC_TTT | ALL | JEPA_CAP01_IMPROVES_TTT（SUPPORTED）/ _PARTIAL / _NOT_ESTABLISHED / _HARMFUL |
| P2 | ZS_CACHE − JC_CACHE | ALL | JEPA_CAP01_IMPROVES_CACHE（SUPPORTED）/ _PARTIAL / _NOT_ESTABLISHED / _HARMFUL |

## 6. 次要分析（只给标签）

| 编号 | 对比 | 区域 | 含义 |
| --- | --- | --- | --- |
| S1 | GEO_ZS_TTT − GEO_JC_TTT | ALL | 集成后的比较 |
| S2 | R7 − GEO_JC_TTT | ALL | JC 集成是否胜过无模型 |
| S3 | JC_CACHE − JC_TTT | ALL | JC 基座上 TTT 的增益 |
| S4 | JA_TTT − JC_TTT | ALL | 同为逆向 JEPA 训练，上限 0.1 对比 3.0 |
| S5 | ZS_TTT − JC_TTT | NEAR7 | |
| S6 | ZS_TTT − JC_TTT | FAR7 | |

另外报告各构建器、各臂的覆盖率。复现检查：复用封存的均值必须与原实验相同。

## 7. 完整性与局限

- **完整性：**
    - 读真值之前，复用的封存与新封存都必须完整，哈希不变。
    - 分析跑两次，结果文件必须逐字节相同。
- **局限：**
    - 只有一个种子。
    - 队列只有 80 个扫描，且已用于 ENSEMBLES-ARKIT 与 JBT 的评估。但 JC 从未在任何数据上评估过，本实验也不在这个队列上选择任何设置。
