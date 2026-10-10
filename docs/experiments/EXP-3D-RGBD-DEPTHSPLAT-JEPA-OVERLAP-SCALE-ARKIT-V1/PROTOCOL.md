# EXP-3D-RGBD-DEPTHSPLAT-JEPA-OVERLAP-SCALE-ARKIT-V1：更多真实训练数据能否改进逆向 JEPA 的补全

本文件即预注册。锁定时间与哈希见 `preregistration.json`。

## 1. 问题

- **NS 的结论：** 训练数据从 182 个扫描增加到 502 个，只奖励精度的逆向 JEPA 的前馈精度没有任何变化。
- **补全是另一回事：**
    - 补全指对近处新视角（第 6、7 帧）中上下文没有覆盖的 FARH 区域的预测。
    - OVERLAP-V1 中，重叠感知的逆向 JEPA（JO，182 个扫描）在 FARH 上把误差从 0.452 降到 0.341，但只有约 54% 的场景变好（PARTIAL）。
    - 所有学到的补全模型都停在这个水平。
- **本实验检验：** 补全是否受真实训练数据量限制。只改一个因素：训练数据从 182 个扫描增加到 502 个，其余与 JO 完全相同。

## 2. 数据

- **训练：** 与 NS 相同的 502 个扫描与合并命中表（`arkit_records_scale.py`、`hit_shares_SCALE.json`，与 NS 逐字节相同）。
    - TRAIN 182 个；
    - CONFIRM、SELECT、SYSTEM、SYSTEM2 各 80 个。
- **评估：** ARKIT160_OVERLAP 中的 VAL80 部分（Validation 折，80 个扫描，查询为第 6、7 帧）。
    - frame_role_lock 从 OVERLAP-V1 逐字节复制。
    - 复用 OVERLAP-V1 的 arms 封存（H3、NEAR、G1）与 JO1–JO3 的 CTX 封存，哈希核对。
    - CONFIRM80 部分已成为本实验的训练数据，不参与打分。
- **许可：** 只用于非商业研究，媒体在 git 之外。

## 3. 模型

- **JOL1–JOL3：** `train_jepa_arkit_overlap_scale.py`，由 OVERLAP-V1 的 `train_jepa_arkit_overlap.py` 经 `derive_train_overlap_scale.py` 逐字替换得到，只换训练记录的来源。
    - 种子 20261008、20261009、20261010，与 JO1–JO3 一一对应。
    - 设置与 JO 相同：原训练目标（未覆盖像素按全深度惩罚），4000 步，学习率 5e-6（ViT 5e-7），预热 200，余弦，AdamW wd 1e-4，裁剪 1.0，每步 2 个目标帧，命中比例 ≥ 0.05，尺度上限 3.0，使用最后一步。
- **评估：** V1 的 `ds_eval.py`，CTX 臂（3 个上下文帧），尺度上限 3.0。

## 4. 读出与指标

与 OVERLAP-V1 相同。

- **JO、JOL：** 至少两个种子覆盖（α ≥ 0.5）的像素取覆盖种子射线深度的平均，其余取 H3。
- **G1N：** 每个 G1 种子在不透明度 ≥ 0.5 处取 深度 / 不透明度，其余取 H3，再对种子平均。
- **区域：**
    - FARH：有上下文命中的查询中，8 像素 NEAR 之外的有效像素；
    - NEAR；
    - FAR：所有查询。
- **指标：** AbsRel。场景值 = 各角色查询平均后再对角色平均。只统计 VAL80 的场景。

## 5. 主检验

主检验用 V2 gate；统计用场景 bootstrap，10,000 次，种子 20260928。

| 编号 | 对比（> 0 表示右边更好） | 区域 | 分支 |
| --- | --- | --- | --- |
| P1 | JO − JOL | FARH | MORE_DATA_HELPS_COMPLETION（SUPPORTED）/ _PARTIAL / _NOT_ESTABLISHED / MORE_DATA_HURTS_COMPLETION |
| P2 | H3 − JOL | FARH | SCALED_COMPLETION_BEATS_HARMONIC（SUPPORTED）/ _PARTIAL / _NOT_ESTABLISHED / _HARMFUL |

## 6. 次要分析（只给标签）

| 编号 | 对比 | 区域 | 含义 |
| --- | --- | --- | --- |
| S1 | H3 − JO | FARH | 原配方在 VAL80 上（参照） |
| S2 | G1N − JOL | FARH | 对比 Hypersim 小载体 |
| S3 | JO − JOL | NEAR | |
| S4 | JO − JOL | FAR | 所有查询 |

另外报告 JO 与 JOL 在 FARH 上的覆盖率。

## 7. 完整性与局限

- **完整性：**
    - 读真值之前，所有封存都必须完整、哈希不变，新封存核对检查点哈希。
    - 分析跑两次，结果文件必须逐字节相同。
- **局限：**
    - 只有 80 个扫描，FARH 约 74 个场景。
    - 只改了数据量，没有改步数。
    - VAL80 上 JO 的结果已知。
