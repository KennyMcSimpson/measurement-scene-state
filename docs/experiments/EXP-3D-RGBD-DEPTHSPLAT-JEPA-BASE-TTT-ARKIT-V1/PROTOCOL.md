# EXP-3D-RGBD-DEPTHSPLAT-JEPA-BASE-TTT-ARKIT-V1：从真实数据训练的逆向 JEPA 基座出发做 TTT

本文件即预注册。锁定时间与哈希见 `preregistration.json`。

## 1. 问题

目前两条线是分开验证的：

- **TTT（Core B）：** 从发布权重的状态出发，在 Hypersim、Replica 和 ARKit 上都有效（与测得深度集成后稳定胜过无模型方法）。消融还表明，起点状态是学出来的这件事本身很重要（LEARNED_BASE_MATTERS）。
- **逆向 JEPA（Core A）：** 只评估过它的前馈补全，从没作为 TTT 的起点评估过。

本实验回答：**用 ARKit 真实数据做逆向 JEPA 训练得到的基座（JA），能否让 TTT 在真实数据上更好？**

## 2. 数据

- **队列：** ARKIT_VAL，共 80 个扫描、160 个 episode、320 个查询。episode 与 EXP-3D-RGBD-ENSEMBLES-ARKIT-V1 完全相同（`frame_role_lock.json` 逐字节复制）：
    - 上下文 = 角色的 3 帧；
    - 流式帧 = Core B 规则的 4 帧；
    - 查询 = primary_query。
- **为什么只用 ARKIT_VAL：** 那里已有发布权重（尺度上限 0.1）的 CACHE / TTT 封存和 7 帧几何封存，可直接复用（哈希核对），GPU 只需跑两个新构建器。
- **限制：** 这个队列的真值已经用于 ENSEMBLES-ARKIT 与 JEPA-ARKIT 的评估，但 JA 的 CACHE / TTT 从没在任何数据上评估过，本实验也不在它上面选择任何设置。
- **许可：** 只用于非商业研究，媒体在 git 之外。

## 3. 状态构建器

所有构建器都用 V1 锁定的 `ds_eval.py`：

- CACHE = 7 帧一次性前馈；
- TTT_m1 = 在 CACHE 状态上，用 7 个测得帧做 100 步 Adam，学习率倍数 m = 1（V1 在 DEV 上选定）。

| 名称 | 权重 | 尺度上限 | 来源 |
| --- | --- | --- | --- |
| JA | EXP-3D-RGBD-DEPTHSPLAT-JEPA-ARKIT-V1 的 **JA1**（种子 20261008，最后一步） | 3.0（与训练相同） | 本实验新跑 |
| ZS3 | 发布权重 | 3.0 | 本实验新跑 |
| ZS | 发布权重 | 0.1 | 复用 ENSEMBLES-ARKIT 的封存 |

- **种子：** JA 只用 JA1。这在 2026-10-09 00:15 事先指定，早于任何 JA 结果，原因是 TTT 很贵。
- **7 帧几何（无模型）：** R7 = REPROJ7_HARMONIC，以及 NEAR7 和射线余弦，复用 ENSEMBLES-ARKIT 的几何封存。

## 4. 读出与指标

- **F7 读出：** α ≥ 0.5 处取射线深度 (depth_z / α) / cos，其余用 R7。
- **几何平均集成：** GEO_X = sqrt(R7 × X)。
- **指标：** AbsRel，区域为 ALL（全部有效像素）、NEAR7、FAR7。场景值 = 各角色查询平均后再对角色平均。

## 5. 主检验

主检验用 V2 gate；统计用场景 bootstrap，10,000 次，种子 20260928。

| 编号 | 对比（> 0 表示右边更好） | 区域 | 分支 |
| --- | --- | --- | --- |
| Q1 | ZS_TTT − JA_TTT | ALL | JEPA_BASE_BEATS_RELEASED_TTT（SUPPORTED）/ _PARTIAL / _NOT_ESTABLISHED / _HARMFUL |
| Q2 | ZS3_TTT − JA_TTT | ALL | JEPA_TRAINING_HELPS_TTT（SUPPORTED）/ _PARTIAL / _NOT_ESTABLISHED / _HARMFUL |

- Q1 是实用问题：JA 基座是否胜过目前最好的 TTT 臂。
- Q2 是机制问题：在同一尺度上限下，训练是否让 TTT 更好。

## 6. 次要分析（只给标签）

- **集成：**
    - S1 GEO_ZS_TTT − GEO_JA_TTT（ALL）；
    - S2 R7 − GEO_JA_TTT（ALL）。
- **训练与 TTT 本身：**
    - S3 ZS3_CACHE − JA_CACHE（ALL），TTT 之前的训练效果；
    - S4 JA_CACHE − JA_TTT（ALL），JA 基座上 TTT 的增益；
    - S5 R7 − JA_TTT（ALL）。
- **尺度上限：** S6 ZS3_TTT − ZS_TTT（ALL），发布权重下尺度上限的影响。
- **分区：** S7、S8 为 Q1 在 NEAR7、FAR7 上的版本；S9 为 Q2 在 NEAR7 上的版本。
- **覆盖率：** 各构建器、各臂的覆盖率（α ≥ 0.5 的有效像素比例）。

## 7. 完整性与局限

- **完整性：**
    - 读真值之前，复用的封存（几何、ZS）与新封存（JA、ZS3）都必须完整，哈希不变。
    - 分析跑两次，结果文件必须逐字节相同。
- **局限：**
    - JA 只有一个种子。
    - 队列只有 80 个扫描。
    - 若 Q1 / Q2 有正面结果，需要在全新队列上确认。

## 修订 1（2026-10-09 01:33:33 +0800 记录；此时尚未读取任何真值，只改执行方式）

- **原因：** GPU 4 上我们的 6 个 TTT 进程每个实际约占 7.6 GB，比试跑时多。加上另一用户的常驻进程，共占 47.1 / 49.1 GB。这样既可能让我们的进程内存不足，也几乎没给共用进程留余地。
- **处理：**
    - 2026-10-09 01:31:48 停掉了 ZS3 的第 3 个分片（`--shard 2/3`）。显存降到 39.6 GB，其余 5 个分片照常运行。
    - 它们结束后，用完全相同的 ds_eval 命令单独重跑 ZS3 的 2/3 分片（`~/cv_work/depthsplat/jepa_ttt/jbt_rerun_zs3_shard2.sh`，日志 `audit/gpu_rerun.log`）。被停掉那次的日志保留为 `audit/ds_eval_ZS3_shard2.stopped.log`。
    - 被停掉的那次只写过前几个查询的预测文件，没有写封存；重跑会覆盖这些文件，封存只在重跑结束时对最终文件计算哈希。
- **打分：** `code/jbt_jobs.sh gpu` 会因被停的分片报告失败，所以打分改由替换后的任务链执行（`~/cv_work/depthsplat/jepa_ttt/chain2.sh`）。它先确认其余 5 个分片都已完成、重跑也已完成，再运行 `code/jbt_jobs.sh score`。
- **不变：** 设置、代码、读出、主检验与分析都不变。
