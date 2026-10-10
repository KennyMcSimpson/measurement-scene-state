# EXP-3D-RGBD-DEPTHSPLAT-JEPA-OVERLAP-SCALE-ARKIT-V1：更多真实训练数据能否改进逆向 JEPA 的补全

**结论：MORE_DATA_COMPLETION_NOT_ESTABLISHED，SCALED_COMPLETION_PARTIAL。训练数据从 182 个扫描增加到 502 个，补全没有改进。**

- **P1（更多数据，VAL80 的 FARH，75 个场景）：** JOL 为 0.3585，JO 为 0.3541，差值 −0.0044 [−0.0334, +0.0195]，42/0/33，NOT_ESTABLISHED。NEAR 与全部查询的 FAR 也都没有差别（S3、S4）。
- **P2（扩数据后的补全对比调和插值）：** +0.1125 [+0.0329, +0.1902]，42/0/33，PARTIAL。与原来的 JO 一样（S1：+0.1168，45/0/30，PARTIAL）：均值明显更好，但逐场景只有约 56% 变好。
- **对比 Hypersim 小载体（S2）：** G1N 的均值仍更低（0.3315 对 0.3585），差别不显著。
- **结论：**
    - 与 NS 合起来看，在这个范围内（182 → 502 个扫描，步数不变），真实训练数据的数量既不是逆向 JEPA 前馈精度的瓶颈，也不是补全的瓶颈。
    - 补全"平均有效、逐场景不稳"的问题不会因为多一些同类数据而消失。

**说明：**

- **预注册：** 见 [PROTOCOL.md](PROTOCOL.md)，2026-10-10 01:58:03 锁定，sha256 `35f85731…`。设计受 NS 结果启发，锁定早于本实验的任何训练与评估。
    - 锁定时由 OVERLAP-V1 的训练脚本重新派生出训练脚本，确认逐字节相同。
    - 记录加载器与合并命中表与 NS 逐字节相同。
- **训练：** JOL1–JOL3 同时训练，各用时约 98 分钟，显存 6.2 GB；种子与 JO1–JO3 一一对应，设置与 JO 相同。
    - DEV 读数与 JA、JO 相似：覆盖率从 0.23 升到 0.53–0.64，被覆盖像素误差从 0.33 升到约 0.49。
- **评估：** ARKIT160_OVERLAP，CTX 臂，每个模型约 3.4 分钟。
    - 只对 VAL80 部分打分：320 个查询，其中 47 个与上下文没有重叠；FARH 有 75 个场景。
    - CONFIRM80 部分已是训练数据，不参与打分。
    - 复用 OVERLAP-V1 的 arms 封存与 JO 封存，H3 与 JO 的 FARH 均值与 OVERLAP-V1 的 VAL 部分完全相同（差值 0）。
- **执行与统计：**
    - GPU 4 与另一用户的常驻进程共用。分析运行两次，结果逐字节相同。
    - 统计单位是场景，bootstrap 10,000 次，种子 20260928，判定用 V2 gate。

## 结果

| 编号 | 对比（> 0 表示右边更好） | 区域 | 均值 | 95% CI | 好/平/差 | 场景 | 状态 |
| --- | --- | --- | ---: | --- | --- | ---: | --- |
| **P1** | JO − JOL | FARH | −0.0044 | [−0.0334, +0.0195] | 42/0/33 | 75 | **NOT_ESTABLISHED** |
| **P2** | H3 − JOL | FARH | +0.1125 | [+0.0329, +0.1902] | 42/0/33 | 75 | **PARTIAL** |
| S1 | H3 − JO | FARH | +0.1168 | [+0.0478, +0.1875] | 45/0/30 | 75 | PARTIAL |
| S2 | G1N − JOL | FARH | −0.0269 | [−0.1127, +0.0442] | 43/0/32 | 75 | NOT_ESTABLISHED |
| S3 | JO − JOL | NEAR | −0.0017 | [−0.0081, +0.0045] | 40/0/37 | 77 | NOT_ESTABLISHED |
| S4 | JO − JOL | FAR（全部查询） | +0.0024 | [−0.0278, +0.0275] | 44/0/34 | 78 | NOT_ESTABLISHED |

**均值（AbsRel）与 FARH 覆盖率：**

| 读出 | FARH | FAR | NEAR | FARH 覆盖率 |
| --- | ---: | ---: | ---: | ---: |
| H3（调和插值） | 0.4709 | 0.5569 | 0.1493 | — |
| G1N（Hypersim 小载体） | **0.3315** | **0.4137** | 0.1910 | — |
| JO（182 个扫描） | 0.3541 | 0.4273 | **0.1157** | 0.81 |
| JOL（502 个扫描） | 0.3585 | 0.4249 | 0.1173 | 0.85 |

## 解读

- **补全的平台不是数据量造成的。**
    - 数据增加到 2.8 倍，JOL 与 JO 的均值、逐场景分布都几乎相同；JOL 的覆盖率略高，精度没有变好。
    - 与 NS 一起说明：在相同步数下，这一规模的真实数据增量对逆向 JEPA 没有作用。
- **逐场景不稳的问题仍在。** 只有约 56% 的场景变好，与此前所有学到的补全一样；只有按上下文深度范围选择，才能让它变得稳定（SELECTOR 实验）。
- **对后续方向的含义：** 要明显改进补全，需要的不是多一些同类数据，而是更强的先验或不同的方法。

## 文件

- `PROTOCOL.md`、`preregistration.json`、`reused_seals.json`
- `eval_OVERLAP/frame_role_lock.json`（复制自 OVERLAP-V1）
- `training/hit_shares_SCALE.json`（与 NS 相同）
- `code/`：
    - `derive_train_overlap_scale.py` 与派生结果 `train_jepa_arkit_overlap_scale.py`；
    - `arkit_records_scale.py`（与 NS 相同）、`ds_predict.py`、`ds_finetune.py`（与 OVERLAP-V1 相同）；
    - 建目录、锁定、任务与打分脚本。
- `checkpoints/JOL1/`–`checkpoints/JOL3/`（含 `SHA256SUMS`）
- `ds/ARKIT160_OVERLAP/JOL*/`：CTX 封存
- `scores/`、`results/`、`audit/`
