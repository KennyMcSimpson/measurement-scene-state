# EXP-3D-RGBD-FULL-SYSTEM-ARKIT-V2：TTT 集成加上 7 帧小载体的选择性集成

**结论：SYSTEM_IMPROVEMENT_NOT_ESTABLISHED，SYSTEM_VS_MODEL_FREE_NOT_ESTABLISHED。Core B 第四次在全新扫描上复现（SUPPORTED）。**

- **主检验 P1（不成立）：** 在 FULL-SYSTEM-V1 样本内挑出的接法没有在新数据上复现。这条接法是：在被选中的 episode 中，把 Core B 在 FAR7 区域的基底换成 sqrt(R7 × 7 帧 GA)。结果为 −0.0022 [−0.0091, +0.0042]，好/平/差 16/42/22。
- **原因：** 在这批扫描上，7 帧 GA 在 FAR7 中比 R7 差得多（0.358 对 0.281，S6 HARMFUL）。在 V1 上它只略差一点（0.295 对 0.270）。
- **次要分析 S5（成立）：** 把 GA 换成 Hypersim 训练的小载体 G1，同样的接法稳定改进了 Core B：+0.0042 [+0.0005, +0.0091]，21/43/16，ALL 从 0.1551 降到 0.1509。但这只是多个次要分析之一，而且在 V1 的同一组数据上它并不成立（+0.0002，17/49/14），所以只能算一条线索。
- **次要分析 S2：Core B 第四次在全新扫描上复现。** +0.0049 [+0.0032, +0.0071]，70/0/10，SUPPORTED。
    - 四次结果分别为：Hypersim EVAL34 71/12，ARKit VAL80 71/9，SYSTEM80 71/9，SYSTEM2 70/10。
    - TTT 本身的增益（S3）：+0.0034，62/0/18，SUPPORTED。

**说明：**

- **预注册：** 见 [PROTOCOL.md](PROTOCOL.md)。
    - 计划锁写于 2026-10-09 13:08:03，早于任何新数据下载；正式锁定于 13:16:03，sha256 `b36f2657…`。
    - 7 帧评估器与 V1 事后探索用的脚本逐字节相同，锁定时已核对。
- **数据：** EXP-3D-RGBD-ARKIT-SYSTEM2-DATA-V1。
    - 取训练折种子顺序第 840–999 位场馆，按顺序准备到 80 个合格（55 个不合格），此前从未被使用。
    - 160 个 episode 中有 77 个被选中（S ≥ 1.2）。FAR7 平均占有效像素的 23.9%。
- **执行：**
    - TTT 分 4 个分片运行，每个约 1.2 小时，峰值显存 5.1 GB，GPU 4 与另一用户的常驻进程共用。
    - arms、几何（回放差值 0）、7 帧三个封存都在 CPU 上完成。
    - 分析运行两次，结果逐字节相同。
- **统计：** 统计单位是场景，bootstrap 10,000 次，种子 20260928，判定用 V2 gate。

## 结果

| 编号 | 对比（> 0 表示右边更好） | 区域 | 均值 | 95% CI | 好/平/差 | 状态 |
| --- | --- | --- | ---: | --- | --- | --- |
| **P1** | CORE_B − SYS | ALL | −0.0022 | [−0.0091, +0.0042] | 16/42/22 | **NOT_ESTABLISHED** |
| **P2** | R7 − SYS | ALL | +0.0027 | [−0.0046, +0.0094] | 63/0/17 | **NOT_ESTABLISHED** |
| S1 | CORE_B − SYS | FAR7（78 场景） | −0.0162 | [−0.0398, +0.0037] | 16/40/22 | NOT_ESTABLISHED |
| S2 | R7 − CORE_B | ALL | +0.0049 | [+0.0032, +0.0071] | 70/0/10 | SUPPORTED |
| S3 | ZS_CACHE_F7 − ZS_TTT_F7 | ALL | +0.0034 | [+0.0001, +0.0061] | 62/0/18 | SUPPORTED |
| S4 | CORE_B − SYS_ALL | ALL | −0.0068 | [−0.0143, +0.0002] | 21/17/42 | NOT_ESTABLISHED |
| S5 | CORE_B − SYS_G1 | ALL | +0.0042 | [+0.0005, +0.0091] | 21/43/16 | SUPPORTED |
| S6 | R7 − GA7 | FAR7 | −0.0767 | [−0.1161, −0.0404] | 17/15/46 | HARMFUL |

**均值（AbsRel）：**

| 读出 | ALL | FAR7 |
| --- | ---: | ---: |
| R7（7 帧重投影 + 调和插值） | 0.160 | 0.281 |
| ZS_TTT_F7 | 0.153 | 0.277 |
| **CORE_B** | **0.155** | **0.277** |
| SYS（GA，选择后） | 0.157 | 0.294 |
| SYS_ALL（GA，不选择） | 0.162 | 0.311 |
| SYS_G1（G1，选择后） | 0.151 | 0.267 |
| GA7（FAR7 单独用 7 帧 GA） | 0.177 | 0.358 |

## 解读

- **Core B 是这条研究线最稳健的结果。**
    - 四个独立评估队列给出了几乎相同的一致性（70–71 个场景变好，9–12 个变差）：一个合成队列，三个真实队列，其中两个是今天新取的。
    - 但真实数据上的绝对增益只有约 3%（ALL 从 0.160 降到 0.155）。
- **小载体接入 Core B 的效果在不同扫描批次之间不稳定。**
    - V1 上 GA 接法看起来有效（样本内），这里不成立。
    - G1 接法在 V1 上无效，这里有效。
    - 结论：小载体在 FAR7 中的表现对扫描批次很敏感。在没有独立复现之前，不能说它能稳定改进完整系统。
- **ARKit 训练的 GA 在 7 帧设置下不如 Hypersim 训练的 G1 稳健。**
    - 一个可能的原因是：GA 的训练查询只取第 5–8 帧附近的近处视角，没有训练过"有近处流式帧时补远处空白"的情形。这一点未经检验。

## 文件

- `PROTOCOL.md`、`preregistration.json`、`audit/plan_locked_at.txt`
- `frame_role_lock.json`、`evaluation_contract.json`、`selector_seal.json`
- `code/`：建目录、锁定、任务与打分脚本、`eval_carrier_7view.py`
- `raw/eval/ARKIT_SYSTEM2/`：3 帧 arms 封存
- `raw/eval7/ARKIT_SYSTEM2/`：7 帧封存
- `geometry/`
- `ds/ARKIT_SYSTEM2/ZS/`
- `scores/`、`results/`、`audit/`
- 数据准备：`../EXP-3D-RGBD-ARKIT-SYSTEM2-DATA-V1/`
