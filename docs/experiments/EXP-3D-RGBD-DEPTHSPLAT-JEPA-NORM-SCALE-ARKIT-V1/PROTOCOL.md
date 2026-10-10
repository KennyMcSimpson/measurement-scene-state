# EXP-3D-RGBD-DEPTHSPLAT-JEPA-NORM-SCALE-ARKIT-V1：更多真实训练数据能否放大逆向 JEPA 的前馈改进

本文件即预注册。锁定时间与哈希见 `preregistration.json`。

## 1. 问题

NF 的结论：只奖励精度的逆向 JEPA（JN 配方）让 7 帧前馈状态稳定地小幅变好。

- ARKIT_SYSTEM3 上，三个种子平均 +0.0041，110/0/50，PARTIAL。
- ARKIT_VAL 上，JN1 为 +0.0036，60/0/20。

改进约 2.5%，逐场景只有约 2/3 变好。JN 只用了 182 个真实训练扫描，数据量可能是瓶颈。

本实验只改一个因素：训练数据从 182 个扫描增加到 502 个，其余设置与 JN 完全相同，包括 4000 步。

## 2. 数据

- **训练：** 502 个扫描，场馆互不重叠（`code/arkit_records_scale.py` 每次加载时检查）。
    - TRAIN 182 个：与 JN 相同。
    - 新增 320 个：此前作为评估队列、实验都已结束的 CONFIRM、SELECT、SYSTEM、SYSTEM2，各 80 个（训练折种子顺序第 360–999 位）。
    - DEV 与 JN 相同（8 个扫描，只作记录）。
- **命中表：** `training/hit_shares_SCALE.json`，由 `code/merge_hit_tables.py` 合并而成。
    - TRAIN 部分逐项与 JN 的相同；
    - 新增四个队列用 OVERLAP-V1 锁定的 `hit_share_table.py` 计算（只用上下文深度与目标相机）。
- **评估：** 两个队列，都不参与训练，场馆与训练集不相交。
    - **ARKIT_SYSTEM3：** 160 个扫描，episode 与 V3、NF 相同。复用 V3 的几何封存与 ZS（CACHE + TTT）封存，以及 NF 的 ZS、JN1–JN3 CACHE 封存。
    - **ARKIT_VAL：** 80 个扫描，episode 与 ENSEMBLES-ARKIT 相同。复用 ENSEMBLES-ARKIT 的几何封存与 ZS 封存、OBJECTIVE-TTT 的 JN1 封存；JN2、JN3 在 VAL 上的 CACHE 由本实验评估。
- **说明：** 新增的训练扫描此前用于其他实验的评估；那些实验都已结束，结论不受影响。
- **许可：** 只用于非商业研究，媒体在 git 之外。

## 3. 模型

- **JL1–JL3：** `train_jepa_arkit_scale.py`，由 JN 的 `train_jepa_arkit_norm.py` 经 `derive_train_scale.py` 逐字替换得到，只换训练记录的来源。
    - 种子 20261008、20261009、20261010，与 JN1–JN3 一一对应。
    - 其余设置与 JN 相同：4000 步，学习率 5e-6（ViT 5e-7），预热 200 步后余弦衰减，AdamW（weight decay 1e-4），梯度裁剪 1.0，每步 2 个目标帧，尺度上限 0.1，使用最后一步。
- **评估：** V1 的 `ds_eval.py`，只跑 CACHE 臂，尺度上限 0.1。
- **执行：** `code/ns_jobs.sh` 同时运行：
    - 三个模型各自训练后，在两个队列上评估；
    - JN2、JN3 在 VAL 上的评估。

## 4. 读出与指标

- **F7 读出：** α ≥ 0.5 处取射线深度 (depth_z / α) / cos，其余用 R7。
- **JNS、JLS：** 每个查询上三个种子 AbsRel 的平均。
- **CORE_B：** sqrt(R7 × ZS_TTT_F7)。
- **指标：** AbsRel，区域为 ALL、NEAR7、FAR7。场景值 = 各角色查询平均后再对角色平均。

## 5. 主检验（ARKIT_SYSTEM3）

主检验用 V2 gate；统计用场景 bootstrap，10,000 次，种子 20260928。

| 编号 | 对比（> 0 表示右边更好） | 区域 | 分支 |
| --- | --- | --- | --- |
| P1 | JNS_CACHE − JLS_CACHE | ALL | MORE_DATA_HELPS（SUPPORTED）/ _PARTIAL / _NOT_ESTABLISHED / MORE_DATA_HURTS |
| P2 | ZS_CACHE − JLS_CACHE | ALL | SCALED_RECIPE_IMPROVES_FEEDFORWARD（SUPPORTED）/ _PARTIAL / _NOT_ESTABLISHED / SCALED_RECIPE_HARMS_FEEDFORWARD |

## 6. 次要分析（只给标签）

| 编号 | 对比 | 区域 | 含义 |
| --- | --- | --- | --- |
| S1 | JNS_CACHE − JLS_CACHE | ALL（ARKIT_VAL） | P1 在第二个队列上 |
| S2 | ZS_CACHE − JLS_CACHE | ALL（ARKIT_VAL） | P2 在第二个队列上 |
| S3 | R7 − JLS_CACHE | ALL（SYSTEM3） | 对比无模型方法 |
| S4 | CORE_B − JLS_CACHE | ALL（SYSTEM3） | 规模化的前馈能否追上 Core B |
| S5 | JNS_CACHE − JLS_CACHE | NEAR7（SYSTEM3） | |
| S6 | JNS_CACHE − JLS_CACHE | FAR7（SYSTEM3） | |
| S7 | ZS_CACHE − JL1_CACHE | ALL（SYSTEM3） | 单个种子 |
| S8 | ZS_CACHE − JL2_CACHE | ALL（SYSTEM3） | 单个种子 |
| S9 | ZS_CACHE − JL3_CACHE | ALL（SYSTEM3） | 单个种子 |

另外报告覆盖率。复现检查：复用封存的均值必须与原实验相同。

## 7. 完整性与局限

- **完整性：**
    - 读真值之前，所有封存都必须完整、哈希不变，新封存的检查点哈希必须与训练产出（JL）或 NF 的检查点（JN2、JN3）一致。
    - 分析跑两次，结果文件必须逐字节相同。
- **局限：**
    - 只改了数据量，没有改步数；更多数据配更多步数，结果可能不同。
    - 只有 3 个种子。
    - SYSTEM3 与 VAL 都已用于评估 JN，但 JL 是新模型，本实验不在它们上面选择任何设置。
