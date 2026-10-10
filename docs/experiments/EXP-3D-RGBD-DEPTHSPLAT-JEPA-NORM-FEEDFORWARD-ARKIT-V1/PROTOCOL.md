# EXP-3D-RGBD-DEPTHSPLAT-JEPA-NORM-FEEDFORWARD-ARKIT-V1：只奖励精度的逆向 JEPA 能否改进前馈状态（预注册复现）

本文件即预注册。锁定时间与哈希见 `preregistration.json`。

## 1. 问题

OBJECTIVE-TTT 实验（JN、JK）的结果：

- **主检验：** 只奖励精度的逆向 JEPA（JN1）作为 TTT 起点，与发布权重打平。TTT 后 ALL AbsRel 为 0.1149 对 0.1141，30/0/50，NOT_ESTABLISHED。
- **次要分析 S3（成立）：** 不做 TTT 时，JN1 的 7 帧前馈状态（CACHE，F7 读出）比发布权重好。
    - ARKIT_VAL 上 ALL 为 0.1172 对 0.1208，+0.0036 [+0.0009, +0.0062]，60/0/20，SUPPORTED。
    - 增益在 NEAR7：0.0718 对 0.0764（R7 为 0.0757）。
    - 机理（推测）：只奖励精度的目标让模型在不可靠处降低不透明度（覆盖率 0.73 → 0.55），F7 读出在这些地方退回 R7。

S3 只是一个种子、一个队列上的多个次要分析之一。本实验在全新扫描上检验两件事：

1. 同一个模型 JN1 的前馈优势能否复现；
2. 同一配方的 3 个种子（JN1–JN3）平均下来是否也有这种优势。

本实验不做 TTT，只关心前馈状态，也就是不能运行 TTT 时的用法（TTT 每个查询约 50 秒）。

## 2. 数据

- **评估：** ARKIT_SYSTEM3，即 FULL-SYSTEM-V3 的队列（EXP-3D-RGBD-ARKIT-SYSTEM3-DATA-V1）：160 个扫描、320 个 episode、640 个查询。
    - `frame_role_lock.json` 从 V3 逐字节复制：上下文 = 角色的 3 帧，流式帧 = Core B 规则的 4 帧，查询 = 第 14、15 帧。
    - JN、JK 系列模型从未在这批扫描上评估过。
    - 本实验锁定时，V3 还没有读取任何真值（建目录和锁定脚本都会检查 V3 尚未打分）。
    - 复用 V3 的 7 帧几何封存（R7、NEAR7、射线余弦），哈希核对。
- **训练：** 与 JN1 完全相同，用 ARKit TRAIN 的 182 个扫描和 OVERLAP-V1 的命中表。
- **许可：** 只用于非商业研究，媒体在 git 之外。

## 3. 模型与评估

| 名称 | 来源 |
| --- | --- |
| ZS | 发布权重，尺度上限 0.1（本实验重新评估） |
| JN1 | OBJECTIVE-TTT 实验的检查点（哈希核对） |
| JN2、JN3 | 本实验训练：同一脚本 `train_jepa_arkit_norm.py`（与 OBJECTIVE-TTT 的逐字节相同）、同一设置，种子 20261009、20261010 |
| JK1 | OBJECTIVE-TTT 实验的 7 帧布局检查点（只用于次要分析） |

- **训练设置：** 4000 步，学习率 5e-6（ViT 5e-7），预热 200 步后余弦衰减，AdamW（weight decay 1e-4），梯度裁剪 1.0，每步 2 个目标帧，尺度上限 0.1，使用最后一步。
- **评估：** V1 的 `ds_eval.py`，只跑 CACHE 臂（上下文 3 帧 + 流式 4 帧建状态 → 渲染查询），尺度上限 0.1。
- **执行：** `code/nf_jobs.sh` 同时跑三条流水线：
    - ZS、JN1、JK1 的评估；
    - JN2 训练后评估；
    - JN3 训练后评估。

## 4. 读出与指标

- **F7 读出：** α ≥ 0.5 处取射线深度 (depth_z / α) / cos，其余用 R7。
- **JNS：** 每个查询上，JN1–JN3 三个种子 AbsRel 的平均，代表单个训练模型的期望表现。
- **集成：** GEO_X = sqrt(R7 × X)。
- **指标：** AbsRel，区域为 ALL、NEAR7、FAR7。场景值 = 各角色查询平均后再对角色平均。

## 5. 主检验

主检验用 V2 gate；统计用场景 bootstrap，10,000 次，种子 20260928。

| 编号 | 对比（> 0 表示右边更好） | 区域 | 分支 |
| --- | --- | --- | --- |
| P1 | ZS_CACHE − JN1_CACHE | ALL | JN1_FEEDFORWARD_REPLICATES（SUPPORTED）/ _PARTIAL / _NOT_ESTABLISHED / JN1_FEEDFORWARD_HARMFUL |
| P2 | ZS_CACHE − JNS_CACHE | ALL | RECIPE_IMPROVES_FEEDFORWARD（SUPPORTED）/ _PARTIAL / _NOT_ESTABLISHED / RECIPE_HARMS_FEEDFORWARD |

## 6. 次要分析（只给标签）

| 编号 | 对比 | 区域 | 含义 |
| --- | --- | --- | --- |
| S1 | R7 − JN1_CACHE | ALL | JN1 的前馈是否胜过无模型方法 |
| S2 | R7 − JNS_CACHE | ALL | 同上，三个种子平均 |
| S3 | R7 − ZS_CACHE | ALL | 发布权重的前馈对比无模型方法 |
| S4 | ZS_CACHE − JN1_CACHE | NEAR7 | |
| S5 | ZS_CACHE − JN1_CACHE | FAR7 | |
| S6 | GEO_ZS_CACHE − GEO_JN1_CACHE | ALL | 与 R7 集成后 |
| S7 | ZS_CACHE − JK1_CACHE | ALL | 7 帧布局 |
| S8 | ZS_CACHE − JN2_CACHE | ALL | 单个种子 |
| S9 | ZS_CACHE − JN3_CACHE | ALL | 单个种子 |

另外报告各模型的覆盖率。

## 7. 完整性与局限

- **完整性：**
    - 读真值之前，所有封存都必须完整，检查点哈希必须与锁定记录（JN1、JK1、发布权重）或训练产出（JN2、JN3）一致。
    - 分析跑两次，结果文件必须逐字节相同。
- **局限：**
    - ARKIT_SYSTEM3 同时被 V3 使用（检验另一组读出），但本实验的模型、读出与对比在 V3 读取真值之前锁定。
    - 只有 3 个种子。预期效应很小（ARKIT_VAL 上约 3%）。
    - 本设计受 OBJECTIVE-TTT 的 S3 启发；JN1 的检验是对同一模型的复现，JN2、JN3 是新训练的模型。
