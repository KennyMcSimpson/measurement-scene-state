# EXP-3D-RGBD-DEPTHSPLAT-JEPA-NORM-SCALE-ARKIT-V1：更多真实训练数据能否放大逆向 JEPA 的前馈改进

**结论：MORE_DATA_NOT_ESTABLISHED，SCALED_RECIPE_FEEDFORWARD_PARTIAL。训练数据从 182 个扫描增加到 502 个，前馈结果没有任何变化。**

- **P1（更多数据，SYSTEM3）：** JLS 与 JNS 的 ALL AbsRel 都是 0.1567，差值 −0.0000 [−0.0014, +0.0012]，好/平/差 82/0/78，NOT_ESTABLISHED。
    - VAL 上也一样：0.1169 对 0.1167（S1，44/0/36）。
    - NEAR7、FAR7 都没有差别（S5、S6）。
- **P2（扩数据后的配方对比发布权重，SYSTEM3）：** +0.0041 [+0.0018, +0.0070]，111/0/49，PARTIAL。VAL 上为 +0.0039，58/0/22，PARTIAL。大小与一致性都与 JN 相同。
- **改进的大小非常稳定：** 四次测量都在 +0.0036 到 +0.0041 之间，一致性都在 68–73%。
    - VAL 上的 JN1；
    - SYSTEM3 上的 JNS；
    - SYSTEM3 与 VAL 上的 JLS。
- **结论：**
    - 在这个范围内（182 → 502 个扫描，步数不变），真实训练数据的数量不是瓶颈。
    - 只奖励精度的逆向 JEPA 对前馈状态约 2.5% 的改进来自训练目标本身（学会弃权），不随数据量增长。
    - 不需要为这条线下载更多数据。

**说明：**

- **预注册：** 见 [PROTOCOL.md](PROTOCOL.md)，2026-10-09 23:18:31 锁定，sha256 `d9ed9d50…`。设计受 NF 结果启发，锁定早于本实验的任何训练与评估。
- **训练数据：** TRAIN 182 个扫描加上 CONFIRM、SELECT、SYSTEM、SYSTEM2 各 80 个，共 502 个，场馆互不重叠，也都不在两个评估队列中（每次加载时检查）。
    - 新增四个队列的命中表由 OVERLAP-V1 的 `hit_share_table.py` 计算，零命中比例 0.30–0.33，与 TRAIN 相近。
    - 合并后共 7769 个可用目标帧，1004 个角色中 985 个可用。
- **训练：** JL1–JL3 同时训练，各用时约 104 分钟，显存 6.4–6.5 GB；种子与 JN1–JN3 一一对应。
- **评估：** 只跑 CACHE 臂。每个模型在 SYSTEM3 上约 5.3 分钟，在 VAL 上约 2.6 分钟；JN2、JN3 在 VAL 上各约 3.4 分钟。
    - 复用 V3、NF、ENSEMBLES-ARKIT、OBJECTIVE-TTT 的封存，12 项复现检查的均值差都是 0。
- **执行与统计：**
    - GPU 4 与另一用户的常驻进程共用。分析运行两次，结果逐字节相同。
    - 统计单位是场景，bootstrap 10,000 次，种子 20260928，判定用 V2 gate。

## 训练中的 DEV 读数

DEV 为 ARKit 训练折的 8 个扫描、第 14、15 帧，3 帧上下文，只作记录。表中为被覆盖像素 AbsRel / 覆盖率。

| 步数 | 0 | 500 | 1000 | 2000 | 3000 | 4000 |
| --- | --- | --- | --- | --- | --- | --- |
| JL1 | 0.214 / 0.19 | 0.217 / 0.18 | 0.218 / 0.13 | 0.229 / 0.13 | 0.222 / 0.11 | 0.222 / 0.11 |
| JL2 | 0.214 / 0.19 | 0.217 / 0.19 | 0.228 / 0.16 | 0.236 / 0.12 | 0.238 / 0.10 | 0.240 / 0.10 |
| JL3 | 0.214 / 0.19 | 0.223 / 0.18 | 0.225 / 0.14 | 0.246 / 0.10 | 0.262 / 0.11 | 0.264 / 0.11 |

## 结果

| 编号 | 对比（> 0 表示右边更好） | 队列与区域 | 均值 | 95% CI | 好/平/差 | 中位数 | 状态 |
| --- | --- | --- | ---: | --- | --- | ---: | --- |
| **P1** | JNS_CACHE − JLS_CACHE | SYSTEM3，ALL | −0.0000 | [−0.0014, +0.0012] | 82/0/78 | +0.0000 | **NOT_ESTABLISHED** |
| **P2** | ZS_CACHE − JLS_CACHE | SYSTEM3，ALL | +0.0041 | [+0.0018, +0.0070] | 111/0/49 | +0.0019 | **PARTIAL** |
| S1 | JNS_CACHE − JLS_CACHE | VAL，ALL | −0.0002 | [−0.0015, +0.0010] | 44/0/36 | +0.0001 | NOT_ESTABLISHED |
| S2 | ZS_CACHE − JLS_CACHE | VAL，ALL | +0.0039 | [+0.0017, +0.0061] | 58/0/22 | +0.0025 | PARTIAL |
| S3 | R7 − JLS_CACHE | SYSTEM3，ALL | +0.0063 | [+0.0034, +0.0096] | 98/0/62 | +0.0010 | PARTIAL |
| S4 | CORE_B − JLS_CACHE | SYSTEM3，ALL | +0.0002 | [−0.0014, +0.0019] | 51/0/109 | −0.0013 | NOT_ESTABLISHED |
| S5 | JNS_CACHE − JLS_CACHE | SYSTEM3，NEAR7 | +0.0003 | [−0.0010, +0.0016] | 86/0/74 | +0.0001 | NOT_ESTABLISHED |
| S6 | JNS_CACHE − JLS_CACHE | SYSTEM3，FAR7 | −0.0006 | [−0.0019, +0.0007] | 65/25/67 | +0.0000 | NOT_ESTABLISHED |
| S7 | ZS_CACHE − JL1_CACHE | SYSTEM3，ALL | +0.0046 | [+0.0020, +0.0081] | 113/0/47 | +0.0020 | PARTIAL |
| S8 | ZS_CACHE − JL2_CACHE | SYSTEM3，ALL | +0.0035 | [+0.0015, +0.0055] | 107/0/53 | +0.0019 | PARTIAL |
| S9 | ZS_CACHE − JL3_CACHE | SYSTEM3，ALL | +0.0042 | [+0.0015, +0.0077] | 107/0/53 | +0.0022 | PARTIAL |

**均值（ALL AbsRel）与覆盖率：**

| 读出 | SYSTEM3（160） | 覆盖率 | VAL（80） | 覆盖率 |
| --- | ---: | ---: | ---: | ---: |
| R7（无模型） | 0.1630 | — | 0.1209 | — |
| ZS（发布权重，前馈） | 0.1608 | 0.72 | 0.1208 | 0.73 |
| JN1 / JN2 / JN3 | 0.1572 / 0.1563 / 0.1566 | | 0.1172 / 0.1160 / 0.1169 | |
| **JNS（182 个扫描，三种子平均）** | **0.1567** | 0.53 | **0.1167** | 0.54 |
| JL1 / JL2 / JL3 | 0.1562 / 0.1573 / 0.1566 | | 0.1170 / 0.1180 / 0.1159 | |
| **JLS（502 个扫描，三种子平均）** | **0.1567** | 0.58 | **0.1169** | 0.59 |
| CORE_B（发布权重 + TTT + 与 R7 集成，参照） | 0.1569 | | 0.1161 | |

## 解读

- **数据量不是瓶颈。** 数据增加到 2.8 倍、步数不变，均值和逐场景分布都没有变化。JL 的覆盖率略高（0.58 对 0.53），精度相同。
- **这与"改进来自弃权"一致。**
    - 弃权的程度由训练目标决定，在 182 个扫描上就已学到。
    - JN 的 DEV 读数在约 2000 步后已经稳定。
- **局限：** 步数没有随数据增加；更多数据配更多步数，结果可能不同，但现有迹象不支持这一点。
- **下一个问题：** 既然改进来自弃权，那么不训练，只把发布权重的读出阈值调高（α ≥ τ），能否达到同样的效果？这是逆向 JEPA 这一正面结果的关键对照。同样的阈值调整也可能改进 Core B。

## 文件

- `PROTOCOL.md`、`preregistration.json`、`reused_seals.json`
- `eval_SYSTEM3/frame_role_lock.json`（复制自 V3）、`eval_VAL/frame_role_lock.json`（复制自 ENSEMBLES-ARKIT）
- `training/`：
    - `hit_shares_TRAIN.json`（与 OBJECTIVE-TTT 相同）；
    - 四个新增队列的命中表；
    - 合并后的 `hit_shares_SCALE.json`。
- `code/`：
    - `arkit_records_scale.py`、`merge_hit_tables.py`、`derive_train_scale.py`，以及派生结果 `train_jepa_arkit_scale.py`；
    - `hit_share_table.py`（与 OVERLAP-V1 相同）、`ds_predict.py`、`ds_finetune.py`（与 OBJECTIVE-TTT 相同）；
    - 建目录、锁定、任务与打分脚本。
- `checkpoints/JL1/`–`checkpoints/JL3/`（含 `SHA256SUMS`）
- `ds/ARKIT_SYSTEM3/JL*/`、`ds/ARKIT_VAL/{JN2,JN3,JL*}/`：CACHE 封存
- `scores/`、`results/`、`audit/`
