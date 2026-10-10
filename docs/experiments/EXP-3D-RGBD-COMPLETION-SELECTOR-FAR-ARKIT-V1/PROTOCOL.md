# EXP-3D-RGBD-COMPLETION-SELECTOR-FAR-ARKIT-V1：深度范围选择规则能否推广到远处视角

本文件即预注册。锁定时间与哈希见 `preregistration.json`。

## 1. 问题

EXP-3D-RGBD-COMPLETION-SELECTOR-ARKIT-V1 在 80 个全新的 ARKit 扫描上确认了一条选择规则（SELECTOR_CONSISTENT）：

- **规则：** 当上下文深度范围 S = (p90 − p10) / 中位数 ≥ 1.2 时用学到的补全，否则用调和插值。
- **适用范围：** 只在近处视角（第 6、7 帧）上检验过，这些视角与上下文有较多重叠。

在远处视角（原评估查询，第 14、15 帧），与上下文重叠的部分少得多。此前在 ARKIT160 上，有命中的远处查询中，学到的补全与调和插值打平（JEPA-ARKIT-V1，X6）。

本实验检验：**同一条规则、同一个阈值，在同一批扫描的远处视角上是否仍然成立？**

- 这批扫描的第 14、15 帧此前从未被评估过。
- 规则和阈值都不改；每个 episode 的 S 只取决于上下文，所以直接复用上一个实验已封存的 S。

## 2. 数据

- **队列：** ARKIT_SELECT（EXP-3D-RGBD-ARKIT-SELECT-DATA-V1，80 个扫描），与上一个实验相同。
- **episode：** 记录 × 角色（A、B）；上下文为该角色的 3 帧；查询为 primary_query（第 14、15 帧）；无流式帧。
- **选择器封存：** 逐字节复制上一个实验的 `selector_seal.json`，并核对哈希。
- **许可：** 只用于非商业研究，媒体在 git 之外。

## 3. 模型

全部冻结，本实验不训练任何模型。

- **GA、G1：** 同上一个实验，用 arms 评估器在 CPU 上评估，各 3 个种子，第 24000 步。
- **JO1–JO3：** EXP-3D-RGBD-DEPTHSPLAT-JEPA-ARKIT-OVERLAP-V1 的检查点。
- **JA1–JA3：** EXP-3D-RGBD-DEPTHSPLAT-JEPA-ARKIT-V1 的检查点，即原配方，训练目标包括远处视角。
- JO 与 JA 都用 V1 的 `ds_eval.py` 在 GPU 上评估，只跑 CTX，尺度上限 3.0。

## 4. 读出与区域

- **基本读出：** H3、GAN、G1N、JO 的定义与上一个实验相同；JA 的定义与 JO 相同。
- **选择读出：** SEL_X 在该 episode 的 S ≥ 1.2 时取 X，否则取 H3。
- **区域：** FARH，即至少有一个上下文命中的查询中，NEAR 之外的有效像素。
- **指标：** AbsRel。场景值 = 各角色的查询平均后，再对角色平均。

## 5. 主检验

主检验用 V2 gate；统计用场景 bootstrap，10,000 次，种子 20260928。平局按 V2 gate 的规则计为不变差。

| 编号 | 对比（> 0 表示右边更好） | 区域 | 分支 |
| --- | --- | --- | --- |
| P1 | H3 − SEL_GAN | FARH | FAR_SELECTOR_CONSISTENT（SUPPORTED）/ FAR_SELECTOR_PARTIAL / FAR_SELECTOR_NOT_ESTABLISHED / FAR_SELECTOR_HARMFUL |

## 6. 次要分析（只给标签）

| 编号 | 对比 |
| --- | --- |
| S1 | H3 − SEL_G1N |
| S2 | H3 − SEL_JO |
| S3 | H3 − SEL_JA |
| S4 | H3 − GAN |
| S5 | H3 − G1N |
| S6 | H3 − JO |
| S7 | H3 − JA |
| S8 | GAN − SEL_GAN（选择的代价） |

另外报告（描述性）：被选中的 episode 比例；被选中与未被选中查询上的均值；S 与增益的 Spearman 相关。

## 7. 完整性与局限

- **完整性：**
    - 读真值之前，arms 封存（G1、GA）以及 JO、JA 的 DepthSplat 封存都必须完整，哈希不变。
    - 分析跑两次，结果文件必须逐字节相同。
- **局限：**
    - 扫描与上一个实验相同，只有查询帧不同。
    - 约一半远处查询没有上下文命中，不进入 FARH。
